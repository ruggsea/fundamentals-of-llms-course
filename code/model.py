"""A small modern GPT (lectures 3 and 4): pre-norm RMSNorm, causal attention with RoPE, SwiGLU MLP,
no biases, output head tied to the token embedding. B = batch, T = tokens per window, d = d_model, V = vocab."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Divide each vector by its typical size, then scale each of its d numbers by a learned gain."""
    def __init__(self, d):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.gain


def rope_angles(block_size, head_dim, base=10000.0):
    """cos and sin of position t times frequency theta_i = base^(-2i / head_dim), shape (T, head_dim/2)."""
    theta = base ** (-torch.arange(0, head_dim, 2).float() / head_dim)
    angles = torch.outer(torch.arange(block_size).float(), theta)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotate each pair of numbers (x0, x1), (x2, x3), ... by its position's angle. x: (B, heads, T, head_dim)."""
    x0, x1 = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
    return rotated.flatten(-2)                                # back to (..., head_dim), pairs interleaved


class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(d, 3 * d, bias=False)            # W_Q, W_K, W_V side by side
        self.out = nn.Linear(d, d, bias=False)                # W_O: the heads' outputs back to width d

    def forward(self, x, cos, sin):
        B, T, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        # cut each vector into n_head slices: (B, T, d) -> (B, n_head, T, d / n_head)
        q, k, v = (t.view(B, T, self.n_head, d // self.n_head).transpose(1, 2) for t in (q, k, v))
        q, k = apply_rope(q, cos[:T], sin[:T]), apply_rope(k, cos[:T], sin[:T])
        # attention is a soft lookup table: how well each query matches each key...
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))    # ...but never a token that comes later
        weights = F.softmax(scores, dim=-1)
        y = weights @ v                                       # ...a weighted average of the values
        y = y.transpose(1, 2).reshape(B, T, d)                # glue the heads back together
        return self.out(y)                                    # F.scaled_dot_product_attention does all this, faster


class SwiGLU(nn.Module):
    """Widen, gate, narrow: down(silu(gate(x)) * up(x))."""
    def __init__(self, d):
        super().__init__()
        hidden = 64 * math.ceil(8 * d / 3 / 64)               # ~8d/3, same parameters as a 4d ReLU MLP
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    """Two sublayers, each reads a normed copy of the residual stream and adds its edit back."""
    def __init__(self, d, n_head):
        super().__init__()
        self.norm1, self.attn = RMSNorm(d), CausalSelfAttention(d, n_head)
        self.norm2, self.mlp = RMSNorm(d), SwiGLU(d)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, n_layer=4, n_head=4, d_model=128, block_size=128):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, n_layer=n_layer, n_head=n_head,
                           d_model=d_model, block_size=block_size)
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)      # one row of d numbers per token id
        self.blocks = nn.ModuleList(Block(d_model, n_head) for _ in range(n_layer))
        self.norm = RMSNorm(d_model)
        cos, sin = rope_angles(block_size, d_model // n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, x, targets=None):
        """x: (B, T) token ids -> logits (B, T, V). With targets, also the mean cross-entropy loss
        (in nats). Targets equal to -100 are left out of the average (sft.py masks prompts this way)."""
        h = self.tok_emb(x)
        for block in self.blocks:
            h = block(h, self.cos, self.sin)
        logits = self.norm(h) @ self.tok_emb.weight.T          # tied head: the embedding table, read backwards
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100)
        return logits, loss

    def logprob(self, ids):
        """log p(ids[:, t+1] | ids[:, :t+1]) for every t: shape (B, T-1). Decks 9 and 11 need this."""
        logits, _ = self(ids[:, :-1])
        return F.log_softmax(logits.float(), dim=-1).gather(-1, ids[:, 1:, None]).squeeze(-1)

    @torch.no_grad()
    def generate(self, ids, max_new, temperature=1.0, top_p=1.0, stop=None):
        """Append max_new tokens to the list `ids`, one forward pass each (no KV cache). temperature 0 is
        greedy; top_p < 1 samples from the fewest tokens whose probabilities add up to top_p."""
        ids = list(ids)
        device = self.tok_emb.weight.device
        for _ in range(max_new):
            context = torch.tensor([ids[-self.block_size:]], device=device)
            logits = self(context)[0][0, -1].float()        # the scores for the next token
            if temperature == 0:
                next_id = int(logits.argmax())
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                sorted_probs, order = probs.sort(descending=True)
                outside = sorted_probs.cumsum(0) - sorted_probs > top_p   # mass before this token already > top_p
                sorted_probs[outside] = 0
                next_id = int(order[torch.multinomial(sorted_probs, 1)])
            ids.append(next_id)
            if next_id == stop:
                break
        return ids

    def resize_vocab(self, new_size):
        """Add rows for new special tokens (sft.py). New rows start at the mean of the old ones."""
        old = self.tok_emb.weight.data
        new = nn.Embedding(new_size, old.size(1)).to(old.device)
        new.weight.data[:len(old)] = old
        new.weight.data[len(old):] = old.mean(0)
        self.tok_emb = new
        self.config["vocab_size"] = new_size


def count_params(vocab_size, n_layer, n_head, d_model, block_size, style="modern"):
    """Parameter count from the config alone. style="gpt2" is GPT-2's layout (biases, LayerNorm,
    learned positions, 4d GELU MLP, tied head): at GPT-2 small it gives lecture 4's 124,439,808."""
    V, L, d, T = vocab_size, n_layer, d_model, block_size
    if style == "gpt2":
        attn = 3 * d * d + 3 * d + d * d + d
        mlp = d * 4 * d + 4 * d + 4 * d * d + d
        return L * (attn + mlp + 2 * 2 * d) + V * d + T * d + 2 * d
    hidden = 64 * math.ceil(8 * d / 3 / 64)
    return L * (4 * d * d + 3 * d * hidden + 2 * d) + V * d + d
