"""Supervised fine-tuning (lecture 10): teach the pretrained model a chat format.

    python sft.py                  # reads runs/example/ckpt.pt and data/sft.jsonl, writes runs/sft/ckpt.pt

The template, as tokens:  <|user|>What color is the sky?<|assistant|>The sky is blue.<|endoftext|>
The loss only counts the answer tokens: the model should learn to answer, not to ask.
"""
import json
import os
import random

import torch

import tokenizer
from eval import load_model

SPECIALS = tokenizer.CHAT_SPECIALS                            # <|endoftext|>, <|user|>, <|assistant|>


def chat_prompt(prompt, merges):
    return tokenizer.encode("<|user|>" + prompt + "<|assistant|>", merges, SPECIALS)


def chat_example(prompt, answer, merges):
    """Returns (x, y) for one example: y is x shifted by one, with -100 wherever y is still the prompt."""
    prompt_ids = chat_prompt(prompt, merges)
    answer_ids = tokenizer.encode(answer + "<|endoftext|>", merges, SPECIALS)
    ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids            # -100: F.cross_entropy skips these
    return ids[:-1], labels[1:]


def pad_batch(examples, pad_id):
    """Right-pad to the longest example. A causal model never looks right, so no attention mask is
    needed; the padded targets are -100, so they do not count in the loss."""
    width = max(len(x) for x, _ in examples)
    x = torch.tensor([x + [pad_id] * (width - len(x)) for x, _ in examples])
    y = torch.tensor([y + [-100] * (width - len(y)) for _, y in examples])
    return x, y


def add_chat_tokens(model):
    """Two new special tokens need two new rows in the (tied) embedding table."""
    model.resize_vocab(model.config["vocab_size"] + len(SPECIALS) - len(tokenizer.SPECIALS))


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    merges, _ = tokenizer.load("data/tokenizer.json")
    model, _ = load_model("runs/example/ckpt.pt")
    add_chat_tokens(model)
    model.train()
    pairs = [json.loads(line) for line in open("data/sft.jsonl")]
    examples = [chat_example(p["prompt"], p["answer"], merges) for p in pairs]
    print(f"{len(examples)} examples, {sum(sum(t != -100 for t in y) for _, y in examples)} answer tokens "
          f"out of {sum(len(y) for _, y in examples)}")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    eot = 256 + len(merges)
    for step in range(1, 101):
        x, y = pad_batch(random.sample(examples, 8), pad_id=eot)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0:
            print(f"step {step:3d}  answer loss {loss.item():.3f}")

    model.eval()
    for prompt in ["What color is the sky?", "What does BPE do?", "What color is the sun?", "What is a dog?"]:
        ids = chat_prompt(prompt, merges)
        answer = model.generate(ids, 40, temperature=0, stop=eot)[len(ids):]
        print(f"{prompt!r:28s} -> {tokenizer.decode(answer, merges, SPECIALS)!r}")
    os.makedirs("runs/sft", exist_ok=True)
    torch.save({"model": model.state_dict(), "config": model.config, "specials": SPECIALS}, "runs/sft/ckpt.pt")
