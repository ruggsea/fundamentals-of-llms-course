"""Evaluation (lecture 9): held-out bits per byte and perplexity, then a multiple-choice test
scored two ways, with a bootstrap confidence interval over the items.

    python eval.py runs/example/ckpt.pt
    python eval.py runs/seed0/ckpt.pt runs/seed1/ckpt.pt runs/seed2/ckpt.pt    # one checkpoint per seed
"""
import json
import math
import re
import sys

import numpy as np
import torch

import data
import tokenizer
from model import GPT


def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    model = GPT(**ckpt["config"])
    model.load_state_dict(ckpt["model"])
    return model.eval(), ckpt["specials"]


@torch.no_grad()
def bits_per_byte(model, ids, merges, specials):
    """Every held-out token predicted once, in windows of block_size. Special tokens count 0 bytes."""
    n_bytes = [len(b) for b in tokenizer.token_bytes(merges, [])] + [0] * len(specials)
    total_nats, total_bytes, total_tokens = 0.0, 0, 0
    T = model.block_size
    for start in range(0, len(ids) - 1, T):
        window = torch.from_numpy(ids[start:start + T + 1].astype(np.int64))[None]
        total_nats -= model.logprob(window).sum().item()
        total_bytes += sum(n_bytes[i] for i in window[0, 1:].tolist())
        total_tokens += window.size(1) - 1
    return total_nats / total_bytes / math.log(2), math.exp(total_nats / total_tokens)


@torch.no_grad()
def choice_logprob(model, prompt_ids, choice_ids):
    """Sum of log p(choice tokens | prompt): score an answer without generating a word."""
    ids = torch.tensor([prompt_ids + choice_ids])
    return model.logprob(ids)[0, len(prompt_ids) - 1:].sum().item()


def multiple_choice(model, items, merges, specials):
    """Per item: 1/0 correct by log-likelihood, and 1/0 correct by generating and parsing."""
    by_likelihood, by_generation = [], []
    for item in items:
        prompt_ids = tokenizer.encode(item["prompt"], merges, specials)
        scores = [choice_logprob(model, prompt_ids, tokenizer.encode(" " + c, merges, specials))
                  for c in item["choices"]]
        by_likelihood.append(int(np.argmax(scores) == item["answer"]))
        generated = model.generate(prompt_ids, 8, temperature=0)[len(prompt_ids):]
        text = tokenizer.decode(generated, merges, specials)
        found = re.search(r"\b(" + "|".join(map(re.escape, item["choices"])) + r")\b", text)
        by_generation.append(int(found is not None and found.group(1) == item["choices"][item["answer"]]))
    return np.array(by_likelihood), np.array(by_generation)


def bootstrap_ci(correct, n_resamples=10_000, seed=0):
    """Resample the items with replacement; the middle 95% of the resampled accuracies."""
    rng = np.random.default_rng(seed)
    means = rng.choice(correct, size=(n_resamples, len(correct))).mean(1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


if __name__ == "__main__":
    paths = sys.argv[1:] or ["runs/example/ckpt.pt"]
    merges, _ = tokenizer.load("data/tokenizer.json")
    val_ids = data.load_split("val")
    items = [json.loads(line) for line in open("data/mc.jsonl")]
    likelihood, generation = [], []
    for path in paths:
        model, specials = load_model(path)
        bpb, ppl = bits_per_byte(model, val_ids, merges, specials)
        a, b = multiple_choice(model, items, merges, specials)
        likelihood.append(a)
        generation.append(b)
        print(f"{path}: held-out bits per byte {bpb:.4f}, perplexity {ppl:.2f} per token, "
              f"multiple choice {a.mean():.3f} by likelihood, {b.mean():.3f} by generation")
    chance = np.mean([1 / len(item["choices"]) for item in items])
    print(f"\n{len(items)} items, chance {chance:.3f}, n_seeds={len(paths)}")
    for name, runs in [("log-likelihood", likelihood), ("generate + regex", generation)]:
        per_item = np.mean(runs, axis=0)                      # average over seeds first, then bootstrap items
        low, high = bootstrap_ci(per_item)
        print(f"{name:17s} accuracy {per_item.mean():.3f}  95% CI [{low:.3f}, {high:.3f}]  (n_seeds={len(paths)})")
