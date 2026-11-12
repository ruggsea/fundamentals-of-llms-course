"""Supplied normalization and rotary positions."""
import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        # Accumulate squared values in float32, including under mixed precision.
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return normalized.to(x.dtype) * self.weight


def rotate(x, cos, sin):
    """Rotate adjacent pairs, preserving the input dtype for mixed precision."""
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2).to(x.dtype)
