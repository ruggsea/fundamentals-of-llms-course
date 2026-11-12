"""Supplied checks: shapes, finite loss, and a training step that runs.
These do not establish causal masking, target alignment or correct updates.
"""
from contextlib import nullcontext
from pathlib import Path
import sys
import math
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from model import Config, GPT
from train import make_targets, train_step

torch.set_num_threads(2)


def test_forward_shapes_and_finite_loss():
    model = GPT(Config(vocab_size=32, context=8, layers=1, heads=2, width=16))
    ids = torch.randint(32, (2, 8))
    logits, loss = model(ids, ids)
    assert logits.shape == (2, 8, 32)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert model(ids)[1] is None


def test_target_shapes():
    x, y = make_targets(torch.arange(18).reshape(2, 9))
    assert x.shape == y.shape == (2, 8)
    assert x.dtype == y.dtype == torch.long


def test_training_step_runs():
    model = GPT(Config(vocab_size=32, context=8, layers=1, heads=2, width=16))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    x, y = make_targets(torch.randint(32, (2, 9)))
    loss, grad_norm = train_step(model, optimizer, x, y, nullcontext())
    assert isinstance(loss, float) and math.isfinite(loss)
    assert isinstance(grad_norm, float) and math.isfinite(grad_norm) and grad_norm >= 0
