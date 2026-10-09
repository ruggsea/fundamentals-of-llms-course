"""Direct preference optimisation (lecture 11) on the SFT model.

    python dpo.py                  # reads runs/sft/ckpt.pt and data/dpo.jsonl, writes runs/dpo/ckpt.pt

For each (prompt, chosen, rejected) triple, the implicit reward of an answer is
beta * (log p_model(answer) - log p_reference(answer)), and the loss is
-log sigmoid(reward(chosen) - reward(rejected)). The reference is a frozen copy of the SFT model.
"""
import copy
import json
import os

import torch
import torch.nn.functional as F

import tokenizer
from eval import load_model
from sft import SPECIALS, chat_prompt


def answer_logprob(model, prompt_ids, answer_ids):
    """log p(answer | prompt): the sum of the answer tokens' log-probs."""
    ids = torch.tensor([prompt_ids + answer_ids])
    return model.logprob(ids)[0, len(prompt_ids) - 1:].sum()


def encode_triples(triples, merges):
    encode = lambda text: tokenizer.encode(text + "<|endoftext|>", merges, SPECIALS)
    return [(chat_prompt(t["prompt"], merges), encode(t["chosen"]), encode(t["rejected"])) for t in triples]


def dpo_step(policy, reference, encoded, opt, beta=0.1):
    """One step on all triples. Returns the loss, the mean reward margin and the share of pairs where chosen is ahead."""
    losses, margins = [], []
    for prompt, chosen, rejected in encoded:
        with torch.no_grad():
            ref_chosen = answer_logprob(reference, prompt, chosen)
            ref_rejected = answer_logprob(reference, prompt, rejected)
        reward_chosen = beta * (answer_logprob(policy, prompt, chosen) - ref_chosen)
        reward_rejected = beta * (answer_logprob(policy, prompt, rejected) - ref_rejected)
        margin = reward_chosen - reward_rejected
        losses.append(-F.logsigmoid(margin))
        margins.append(margin.item())
    loss = torch.stack(losses).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item(), sum(margins) / len(margins), sum(m > 0 for m in margins) / len(margins)


def total_logprobs(model, encoded):
    """Summed log-probs of all chosen and all rejected answers (watch both: DPO can lower both)."""
    with torch.no_grad():
        return (sum(answer_logprob(model, p, c).item() for p, c, _ in encoded),
                sum(answer_logprob(model, p, r).item() for p, _, r in encoded))


if __name__ == "__main__":
    torch.manual_seed(0)
    merges, _ = tokenizer.load("data/tokenizer.json")
    policy, _ = load_model("runs/sft/ckpt.pt")
    reference = copy.deepcopy(policy).requires_grad_(False)   # frozen: it never changes
    policy.train()
    encoded = encode_triples([json.loads(line) for line in open("data/dpo.jsonl")], merges)
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-4, weight_decay=0.0)
    chosen0, rejected0 = total_logprobs(policy, encoded)
    for step in range(1, 51):
        loss, margin, wins = dpo_step(policy, reference, encoded, opt)
        if step == 1 or step % 10 == 0:
            print(f"step {step:2d}  loss {loss:.4f}  reward margin {margin:+.3f}  chosen ahead in {wins:.0%} of pairs")
    chosen1, rejected1 = total_logprobs(policy, encoded)
    print(f"log p(chosen) summed: {chosen0:.1f} -> {chosen1:.1f};  log p(rejected): {rejected0:.1f} -> {rejected1:.1f}")
    os.makedirs("runs/dpo", exist_ok=True)
    torch.save({"model": policy.state_dict(), "config": policy.config, "specials": SPECIALS}, "runs/dpo/ckpt.pt")
