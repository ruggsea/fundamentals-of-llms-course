"""Small decoder: pre-RMSNorm, RoPE, SwiGLU, tied embeddings and causal attention.

An educational PyTorch implementation, inspired by Karpathy's micro/nanoGPT and
CS336's component-based approach; not compatible with pretrained model weights.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from components import RMSNorm, rotate


@dataclass(frozen=True)
class Config:
    vocab_size: int
    context: int = 128
    layers: int = 4
    heads: int = 4
    width: int = 128
    activation: str = "silu"

    def __post_init__(self):
        if min(self.vocab_size, self.context, self.layers, self.heads, self.width) < 1:
            raise ValueError("All model dimensions must be positive")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("width must divide into heads with even head dimensions for RoPE")
        if self.activation not in ("silu", "gelu"):
            raise ValueError("Supported activations: silu, gelu")


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.out = nn.Linear(config.width, config.width, bias=False)

    def forward(self, x, cos, sin):
        """TODO 1: causal attention. x is [B, T, C]; return [B, T, C].

        Project Q/K/V, split C into heads, apply rotate to Q and K,
        calculate scaled attention scores, mask future positions, normalize
        with softmax, combine values, merge heads and apply self.out.
        cos/sin contain positions up to the configured maximum context.
        See assignment.md for exact shapes, scaling and mask direction.
        """
        raise NotImplementedError("TODO 1: Attention.forward")


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activation = config.activation
        self.attn_norm, self.ffn_norm = RMSNorm(config.width), RMSNorm(config.width)
        self.attn = Attention(config)
        hidden = 64 * math.ceil((8 * config.width / 3) / 64)
        self.gate = nn.Linear(config.width, hidden, bias=False)
        self.up = nn.Linear(config.width, hidden, bias=False)
        self.down = nn.Linear(hidden, config.width, bias=False)

    def forward(self, x, cos, sin):
        """TODO 2: connect the supplied layers into a transformer block.

        Normalize before each sublayer. Keep both residual connections.
        Use self.activation to select silu (baseline) or gelu (easy variant).
        See assignment.md for the gated feed-forward equation.
        """
        raise NotImplementedError("TODO 2: Block.forward")


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = RMSNorm(config.width)
        head_dim = config.width // config.heads
        angles = torch.outer(torch.arange(config.context).float(),
                             10000.0 ** (-torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
        # Scale residual projections to keep initial residual updates small at depth.
        for block in self.blocks:
            for projection in (block.attn.out, block.down):
                nn.init.normal_(projection.weight, std=0.02 / math.sqrt(2 * config.layers))

    def forward(self, ids, targets=None):
        """TODO 3: token IDs -> logits and optional next-token loss.

        ids: integer [B, T], 1 <= T <= context.
        targets: optional integer [B, T]; -100 means ignore this target.
        Return (logits [B, T, vocab_size], scalar loss or None).
        Use the blocks, final RMSNorm and tied embedding/output weights.
        The caller already shifts the targets; do not shift them again.
        """
        raise NotImplementedError("TODO 3: GPT.forward")

    @torch.no_grad()
    def generate(self, ids, count, temperature=0.8, top_k=40, stop=None):
        if not ids or count < 0 or temperature < 0 or top_k < 0:
            raise ValueError("Need a nonempty prompt and nonnegative sampling arguments")
        self.eval()
        tokens = torch.tensor([ids], device=self.embedding.weight.device)
        for _ in range(count):
            logits = self(tokens[:, -self.config.context:])[0][:, -1].float()
            if temperature == 0:
                next_id = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    cutoff = logits.topk(min(top_k, logits.shape[-1])).values[:, -1:]
                    logits = logits.masked_fill(logits < cutoff, float("-inf"))
                next_id = torch.multinomial(logits.softmax(-1), 1)
            tokens = torch.cat((tokens, next_id), dim=1)
            if next_id.item() == stop:
                break
        return tokens[0].tolist()
