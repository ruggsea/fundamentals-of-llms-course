"""Supplied training infrastructure. Student update logic lives in train.py."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np
import torch
from helpers.tokenizer import Tokenizer

from model import Config, GPT
from helpers.data import sha256
from helpers.tracking import RunTracker

DEFAULTS = dict(steps=1000, batch_size=16, lr=3e-3, warmup=50,
                weight_decay=0.1, eval_every=100, seed=0, context=128,
                layers=4, heads=4, width=128, activation="silu")


def select_device(request):
    if request == "auto":
        request = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    if request == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; check your environment and allocation")
    if request == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable")
    return torch.device(request)


def autocast(device, precision):
    return torch.autocast(device.type, dtype=torch.bfloat16) if precision == "bfloat16" else nullcontext()


def batch(ids, batch_size, context, generator, device, make_targets):
    starts = torch.randint(len(ids) - context, (batch_size,), generator=generator).tolist()
    windows = np.stack([ids[start:start + context + 1] for start in starts]).astype(np.int64)
    windows = torch.from_numpy(windows).to(device)
    return make_targets(windows)


def learning_rate(step, total, warmup, peak):
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup - 1)
    return peak * (0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2)


def load_data(directory):
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text())
    for name, expected in meta["files"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Data artifact changed: {name}")
    arrays = {name: np.memmap(directory / f"{name}.bin", mode="r", dtype=meta["dtype"])
              for name in ("train", "val")}
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    # Count the original bytes represented by each token; the end token has zero bytes.
    lengths = np.array([0 if i == meta["eot_id"] else len(tokenizer.token_bytes(i))
                        for i in range(meta["vocab_size"])])
    return meta, arrays, lengths


@torch.no_grad()
def evaluate(model, ids, byte_lengths, device, precision, batch_size=8):
    """Score every target once, reset context at nonoverlapping token windows.

    The initial end token is context only; end-token losses count, end-token bytes
    do not. Right padding has ignored labels and cannot affect earlier tokens.
    """
    was_training = model.training
    model.eval()
    context = model.config.context
    total_nats, total_tokens, total_bytes = 0.0, 0, 0
    starts = range(0, len(ids) - 1, context)
    for offset in range(0, len(starts), batch_size):
        rows = starts[offset:offset + batch_size]
        x = np.zeros((len(rows), context), dtype=np.int64)
        y = np.full_like(x, -100)
        for row, start in enumerate(rows):
            window = np.asarray(ids[start:start + context + 1], dtype=np.int64)
            count = len(window) - 1
            x[row, :count], y[row, :count] = window[:-1], window[1:]
            total_bytes += int(byte_lengths[window[1:]].sum())
        n_tokens = int((y != -100).sum())
        with autocast(device, precision):
            logits, _ = model(torch.from_numpy(x).to(device))
            loss = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1),
                torch.from_numpy(y).to(device).flatten(), ignore_index=-100)
        total_nats += loss.item() * n_tokens
        total_tokens += n_tokens
    model.train(was_training)
    if not total_bytes:
        raise ValueError("Validation contains no text bytes")
    return {"val_loss": total_nats / total_tokens,
            "val_bpb": total_nats / total_bytes / math.log(2),
            "val_tokens": total_tokens, "val_bytes": total_bytes}


def source_fingerprints(folder=None):
    folder = Path(folder) if folder is not None else Path(__file__).resolve().parents[1]
    return {path.relative_to(folder).as_posix(): sha256(path)
            for path in sorted(folder.rglob("*.py"))
            if not any(part.startswith(".") for part in path.relative_to(folder).parts)}


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def run_training(args, make_targets, train_step):
    with RunTracker(args) as tracker:
        return _run_training(args, make_targets, train_step, tracker)


def _run_training(args, make_targets, train_step, tracker):
    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"Refusing to replace {out}; choose a new --out directory")
    config = {key: getattr(args, key) if getattr(args, key) is not None else default
              for key, default in DEFAULTS.items()}
    config["data"] = str(Path(args.data).resolve())
    if min(config[k] for k in ("steps", "batch_size", "eval_every")) < 1:
        raise ValueError("Steps, batch size and eval interval must be positive")
    if not 0 <= config["warmup"] < config["steps"] or config["lr"] <= 0 or config["weight_decay"] < 0:
        raise ValueError("Require 0 <= warmup < steps, lr > 0 and weight_decay >= 0")
    precision = args.precision or (
        "bfloat16" if device.type == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False) else "float32")
    if precision == "bfloat16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported(including_emulation=False)):
        raise ValueError("This trainer supports bfloat16 on capable CUDA devices only")
    meta, arrays, byte_lengths = load_data(config["data"])
    data_hash = sha256(Path(config["data"]) / "meta.json")
    if len(arrays["train"]) <= config["context"] or len(arrays["val"]) < 2:
        raise ValueError("Training split must exceed context length; validation needs at least two tokens")
    torch.manual_seed(config["seed"])
    model_config = Config(vocab_size=meta["vocab_size"], **{
        key: config[key] for key in ("context", "layers", "heads", "width", "activation")})
    model = GPT(model_config).to(device)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.ndim >= 2], "weight_decay": config["weight_decay"]},
        {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0},
    ], lr=config["lr"], betas=(0.9, 0.95))
    generator = torch.Generator().manual_seed(config["seed"])
    step, best = 0, float("inf")
    out.mkdir(parents=True)
    (out / "source").mkdir()
    code = Path(__file__).resolve().parents[1]
    for name in source_fingerprints(code):
        target = out / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(code / name, target)
    shutil.copy2(Path(config["data"]) / "tokenizer.json", out / "tokenizer.json")
    shutil.copy2(Path(config["data"]) / "meta.json", out / "data-meta.json")
    environment = subprocess.run(["uv", "pip", "freeze", "--python", os.sys.executable],
                                 check=True, text=True, capture_output=True).stdout
    (out / "environment.txt").write_text(environment)
    saved_source_hashes = source_fingerprints(out / "source")
    record = {"training": config, "model": asdict(model_config), "data_hash": data_hash,
              "source_hashes": saved_source_hashes, "precision": precision,
              "device": str(device), "threads": args.threads, "python": platform.python_version(),
              "torch": str(torch.__version__), "numpy": np.__version__,
              "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None}
    (out / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    tracker.start(out, record)
    start = time.perf_counter()
    tokens_per_step = config["batch_size"] * config["context"]
    print(json.dumps({"parameters": sum(p.numel() for p in model.parameters()),
                      "device": str(device), "precision": precision, "start_step": step}), flush=True)

    def report(train_loss, grad_norm=None):
        nonlocal best
        metrics = evaluate(model, arrays["val"], byte_lengths, device, precision)
        if not math.isfinite(metrics["val_loss"]):
            raise FloatingPointError("Validation loss is nonfinite")
        improved = metrics["val_loss"] < best
        best = min(best, metrics["val_loss"])
        seconds = time.perf_counter() - start
        row = dict(step=step, tokens=step * tokens_per_step, seconds=seconds,
                   lr=optimizer.param_groups[0]["lr"], train_loss=train_loss, grad_norm=grad_norm, **metrics)
        state = {"model": model.state_dict(), "model_config": asdict(model_config),
                 "step": step, "data_hash": data_hash,
                 "source_hashes": saved_source_hashes, "eot_id": meta["eot_id"],
                 "tokenizer": (out / "tokenizer.json").read_text()}
        save_checkpoint(out / "last.pt", state)
        if improved:
            save_checkpoint(out / "best.pt", state)
        with (out / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        tracker.log(row)

    report(None)
    while step < config["steps"]:
        lr = learning_rate(step, config["steps"], config["warmup"], config["lr"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = batch(arrays["train"], config["batch_size"], config["context"], generator, device, make_targets)
        loss_value, grad_norm = train_step(model, optimizer, x, y, autocast(device, precision))
        if not isinstance(loss_value, (float, int)) or not math.isfinite(loss_value):
            raise FloatingPointError("train_step must return a finite Python loss value")
        if not isinstance(grad_norm, (float, int)) or not math.isfinite(grad_norm) or grad_norm < 0:
            raise FloatingPointError("train_step must return a finite nonnegative gradient norm")
        step += 1
        if step % config["eval_every"] == 0 or step == config["steps"]:
            report(loss_value, grad_norm)
        else:
            row = dict(step=step, tokens=step * tokens_per_step,
                       seconds=time.perf_counter() - start, lr=lr,
                       train_loss=loss_value, grad_norm=grad_norm)
            with (out / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            tracker.log(row)
    return model


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, help="JSON defaults; explicit CLI options override them")
    result.add_argument("--out", required=True, type=Path)
    result.add_argument("--data", required=True, type=Path)
    result.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    result.add_argument("--precision", choices=("float32", "bfloat16"))
    result.add_argument("--threads", type=int, default=4)
    result.add_argument("--wandb-project", help="Optional Weights & Biases project; local logs are always saved")
    result.add_argument("--wandb-name", help="Optional display name for the W&B run")
    for key, default in DEFAULTS.items():
        result.add_argument("--" + key.replace("_", "-"), type=type(default), default=None)
    return result



def parse_args(argv=None):
    result = parser()
    preliminary, _ = result.parse_known_args(argv)
    if preliminary.config:
        settings = json.loads(preliminary.config.read_text())
        unknown = set(settings) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        result.set_defaults(**settings)
    args = result.parse_args(argv)
    if args.threads < 1:
        raise ValueError("threads must be positive")
    return args
