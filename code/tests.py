"""Fast checks. Run these first: python tests.py  (pytest tests.py works too)."""
import copy
import json
import math

import numpy as np
import torch

import data
import dpo
import sft
import tokenizer
from model import GPT, count_params

TOY = {"low": 5, "lower": 2, "widest": 3, "newest": 6}
HANDOUT_MERGES = ["s t", "e st", "o w", "l ow", "w est", "n e", "ne west", "w i", "wi d", "wid est", "low e", "lowe r"]


def small_tokenizer():
    """A 300-token tokenizer trained on the shipped chat data, so the tests need no download."""
    text = " ".join(" ".join(row.values()) for row in map(json.loads, open("data/dpo.jsonl")))
    return tokenizer.train(text, 300)


def test_toy_merges_match_lecture_2():
    """Same corpus and tie rule as deck 2 (CS336 A1 handout, section 2.4): same twelve merges."""
    merges = tokenizer.train_on_counts(TOY, 12)
    vocab = tokenizer.token_bytes(merges, [])
    assert [f"{vocab[a].decode()} {vocab[b].decode()}" for a, b in merges] == HANDOUT_MERGES


def test_round_trip():
    merges = small_tokenizer()
    text = "Grüße aus Graz 🌍! It's 42.<|endoftext|><|user|>hi<|assistant|>ok"
    ids = tokenizer.encode(text, merges, tokenizer.CHAT_SPECIALS)
    assert tokenizer.decode(ids, merges, tokenizer.CHAT_SPECIALS) == text
    assert ids.count(256 + len(merges)) == 1                 # <|endoftext|> is one token, never split
    assert len(ids) < len(text.encode("utf-8"))              # the merges compress


def test_get_batch_shapes_and_shift():
    ids = np.arange(1000, dtype=np.uint16)
    x, y = data.get_batch(ids, batch_size=4, block_size=16)
    assert x.shape == y.shape == (4, 16)
    assert torch.equal(y[:, :-1], x[:, 1:]) and torch.equal(y, x + 1)


def test_forward_shapes_and_initial_loss():
    torch.manual_seed(0)
    model = GPT(vocab_size=500, n_layer=2, n_head=2, d_model=32, block_size=16)
    x, y = torch.randint(0, 500, (3, 16)), torch.randint(0, 500, (3, 16))
    logits, loss = model(x, y)
    assert logits.shape == (3, 16, 500)
    assert abs(loss.item() - math.log(500)) < 0.1            # at init every token is about equally likely


def test_attention_is_causal():
    torch.manual_seed(0)
    model = GPT(vocab_size=50, n_layer=2, n_head=2, d_model=32, block_size=16)
    x = torch.randint(0, 50, (1, 10))
    changed = x.clone()
    changed[0, -1] = (x[0, -1] + 1) % 50                     # change only the last token
    assert torch.allclose(model(x)[0][0, :-1], model(changed)[0][0, :-1], atol=1e-5)


def test_generate_length():
    model = GPT(vocab_size=50, n_layer=1, n_head=2, d_model=16, block_size=8)
    assert len(model.generate([1, 2, 3], 20, temperature=0.8, top_p=0.9)) == 23   # longer than block_size too
    assert len(model.generate([1, 2, 3], 5, temperature=0)) == 8


def test_logprob_matches_loss():
    torch.manual_seed(0)
    model = GPT(vocab_size=100, n_layer=2, n_head=2, d_model=32, block_size=16)
    ids = torch.randint(0, 100, (2, 16))
    _, loss = model(ids[:, :-1], ids[:, 1:])
    assert torch.allclose(-model.logprob(ids).mean(), loss, atol=1e-5)


def test_param_counts():
    model = GPT(vocab_size=2048, n_layer=4, n_head=4, d_model=128, block_size=128)
    assert sum(p.numel() for p in model.parameters()) == count_params(**model.config)
    assert count_params(50257, 12, 12, 768, 1024, style="gpt2") == 124_439_808   # deck 4, tied head


def test_one_step_lowers_loss():
    torch.manual_seed(0)
    model = GPT(vocab_size=100, n_layer=2, n_head=2, d_model=32, block_size=16)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x, y = torch.randint(0, 100, (4, 16)), torch.randint(0, 100, (4, 16))
    before = model(x, y)[1]
    before.backward()
    opt.step()
    assert model(x, y)[1].item() < before.item()


def test_sft_masks_prompt_and_resizes():
    merges = small_tokenizer()
    x, y = sft.chat_example("What color is the sky?", "The sky is blue.", merges)
    n_prompt = len(sft.chat_prompt("What color is the sky?", merges))
    assert y[:n_prompt - 1] == [-100] * (n_prompt - 1)       # every prompt target is masked
    assert -100 not in y[n_prompt - 1:] and y[-1] == 256 + len(merges)   # the answer ends with <|endoftext|>
    model = GPT(vocab_size=256 + len(merges) + 1, n_layer=1, n_head=2, d_model=16, block_size=64)
    old = model.tok_emb.weight.detach().clone()
    sft.add_chat_tokens(model)
    assert model.tok_emb.weight.shape[0] == len(old) + 2 and torch.equal(model.tok_emb.weight[:len(old)], old)
    assert model(torch.tensor([x]), torch.tensor([y]))[1].isfinite()


def test_dpo_margin_rises():
    torch.manual_seed(0)
    merges = small_tokenizer()
    policy = GPT(vocab_size=256 + len(merges) + 3, n_layer=2, n_head=2, d_model=32, block_size=128)
    reference = copy.deepcopy(policy).requires_grad_(False)
    encoded = dpo.encode_triples([json.loads(line) for line in open("data/dpo.jsonl")], merges)
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    margins = [dpo.dpo_step(policy, reference, encoded, opt)[1] for _ in range(20)]
    assert margins[0] == 0.0 and margins[-1] > 0.5           # policy = reference at step 1, then it moves


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("PASS", name)
