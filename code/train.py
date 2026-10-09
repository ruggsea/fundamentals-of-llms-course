"""Pretraining (lecture 5): AdamW, warmup then cosine or WSD, gradient clipping, a token budget.

    python train.py                            # tiny defaults, a few minutes on a laptop CPU
    python train.py --schedule wsd --out runs/wsd
    python train.py --resume                   # continue from runs/example/ckpt.pt

Every --eval_every steps it writes one line to <out>/metrics.jsonl and saves <out>/ckpt.pt.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

import data
import tokenizer
from model import GPT

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="runs/example")
parser.add_argument("--n_layer", type=int, default=4)
parser.add_argument("--n_head", type=int, default=4)
parser.add_argument("--d_model", type=int, default=128)
parser.add_argument("--block_size", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--tokens", type=int, default=2_000_000, help="the compute budget: tokens seen in training")
parser.add_argument("--lr", type=float, default=3e-3)
parser.add_argument("--min_lr_frac", type=float, default=0.1, help="the schedule ends at lr * this")
parser.add_argument("--warmup_frac", type=float, default=0.05)
parser.add_argument("--schedule", choices=["cosine", "wsd"], default="cosine")
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--clip", type=float, default=1.0)
parser.add_argument("--eval_every", type=int, default=100)
parser.add_argument("--eval_batches", type=int, default=20)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
merges, specials = tokenizer.load("data/tokenizer.json")
train_ids, val_ids = data.load_split("train"), data.load_split("val")
meta = json.load(open("data/meta.json"))
tokens_per_byte = meta["val"]["tokens"] / meta["val"]["bytes"]

model = GPT(256 + len(merges) + len(specials), args.n_layer, args.n_head, args.d_model, args.block_size).to(device)
print(f"{sum(p.numel() for p in model.parameters()):,} parameters on {device}")
# weight decay on the matrices only, not on the norm gains
matrices = [p for p in model.parameters() if p.dim() >= 2]
gains = [p for p in model.parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": matrices, "weight_decay": args.weight_decay},
                         {"params": gains, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))

tokens_per_step = args.batch_size * args.block_size
max_steps = args.tokens // tokens_per_step
warmup = max(1, int(args.warmup_frac * max_steps))


def learning_rate(step):
    """Linear warmup, then cosine down to min_lr_frac * lr, or WSD: stay flat, cool down in the last 20%."""
    min_lr = args.min_lr_frac * args.lr
    if step < warmup:
        return args.lr * (step + 1) / warmup
    if args.schedule == "cosine":
        progress = (step - warmup) / max(1, max_steps - warmup)
        return min_lr + (args.lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
    decay_start = int(0.8 * max_steps)
    if step < decay_start:
        return args.lr
    return args.lr - (args.lr - min_lr) * (step - decay_start) / max(1, max_steps - decay_start)


@torch.no_grad()
def held_out_loss():
    model.eval()
    losses = []
    for _ in range(args.eval_batches):
        x, y = data.get_batch(val_ids, args.batch_size, args.block_size)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            losses.append(model(x.to(device), y.to(device))[1].item())
    model.train()
    return sum(losses) / len(losses)


os.makedirs(args.out, exist_ok=True)
ckpt_path, log_path = f"{args.out}/ckpt.pt", f"{args.out}/metrics.jsonl"
step = 0
if args.resume:
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    step = ckpt["step"]
    print(f"resumed at step {step}")
elif os.path.exists(log_path):
    os.remove(log_path)                                       # a fresh run starts a fresh log

print(f"{max_steps} steps of {tokens_per_step} tokens = {max_steps * tokens_per_step:,} tokens, "
      f"{max_steps * tokens_per_step / len(train_ids):.1f} epochs of the training set")


def evaluate_and_save(train_loss):
    val_loss = held_out_loss()
    row = {"step": step, "tokens": step * tokens_per_step, "lr": round(learning_rate(step), 6),
           "train_loss": train_loss and round(train_loss, 4), "val_loss": round(val_loss, 4),
           "val_bpb": round(val_loss / math.log(2) * tokens_per_byte, 4),   # bits per byte, the A1 unit
           "seconds": round(time.time() - start, 1)}
    print(json.dumps(row))
    with open(log_path, "a") as f:
        f.write(json.dumps(row) + "\n")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                "config": model.config, "specials": specials}, ckpt_path)


start = time.time()
if step == 0:
    evaluate_and_save(None)                                   # the loss at initialisation, about ln(vocab size)
while step < max_steps:
    for group in opt.param_groups:
        group["lr"] = learning_rate(step)
    x, y = data.get_batch(train_ids, args.batch_size, args.block_size)
    with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        _, loss = model(x.to(device), y.to(device))
    opt.zero_grad(set_to_none=True)
    loss.backward()                                           # the gradient of the loss for every parameter
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    opt.step()                                                # nudge every parameter against its gradient
    step += 1
    if step % args.eval_every == 0 or step == max_steps:
        evaluate_and_save(loss.item())

prompt = tokenizer.encode("Once upon a time", merges)
sample = model.generate(prompt, 60, temperature=0.8, top_p=0.95, stop=256 + len(merges))
print("sample:", tokenizer.decode(sample, merges))
