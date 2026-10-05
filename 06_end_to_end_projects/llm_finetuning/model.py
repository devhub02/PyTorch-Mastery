"""Tiny GPT + LoRA, from scratch."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d, heads, block):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.register_buffer("mask", torch.tril(torch.ones(block, block, dtype=torch.bool)), persistent=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        s = q @ k.transpose(-2, -1) / math.sqrt(D // self.h)
        s = s.masked_fill(~self.mask[:T, :T], float("-inf"))   # token t only sees tokens <= t
        return self.proj((s.softmax(-1) @ v).transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, d, heads, block):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, heads, block)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class TinyGPT(nn.Module):
    def __init__(self, vocab, d=64, heads=4, layers=3, block=48):
        super().__init__()
        self.block = block
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block, d)
        self.blocks = nn.ModuleList(Block(d, heads, block) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.apply(self._init)
        self.head.weight = self.tok.weight           # weight tying: input and output embeddings are shared

    @staticmethod
    def _init(m):
        # GPT-style small init: keeps initial logits near zero so the first loss is about ln(vocab)
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.size(1), device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))                 # (B, T, vocab) logits

    @torch.no_grad()
    def generate(self, idx, max_new=20, temperature=0.0, stop_id=None, gen=None):
        """Autoregressive sampling. temperature=0 -> greedy argmax."""
        was_training = self.training
        self.eval()
        for _ in range(max_new):
            logits = self(idx[:, -self.block:])[:, -1]
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                nxt = torch.multinomial((logits / temperature).softmax(-1), 1, generator=gen)
            idx = torch.cat([idx, nxt], 1)
            if stop_id is not None and nxt.item() == stop_id:
                break
        self.train(was_training)
        return idx


class LoRALinear(nn.Module):
    """y = W x + b + (alpha/r) * B(A x). W,b frozen. B starts at zero so the model is unchanged at step 0."""

    def __init__(self, base: nn.Linear, r=4, alpha=8):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale

    def merged(self):
        """Fold the adapter into a plain Linear: W' = W + scale * B A  (no extra inference cost)."""
        lin = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        with torch.no_grad():
            lin.weight.copy_(self.base.weight + self.scale * self.B @ self.A)
            if self.base.bias is not None:
                lin.bias.copy_(self.base.bias)
        return lin


def apply_lora(model, r=4, alpha=8, targets=("qkv", "proj", "fc1", "fc2")):
    """Freeze everything, then wrap the target Linear layers with LoRA adapters (in place)."""
    for p in model.parameters():
        p.requires_grad_(False)
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(parent, name, LoRALinear(child, r, alpha))
    return model


def merge_lora(model):
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                setattr(parent, name, child.merged())
    return model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, train
