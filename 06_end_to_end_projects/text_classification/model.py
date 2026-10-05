"""Transformer encoder classifier written from scratch (no nn.Transformer)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        assert d % heads == 0, "d_model must be divisible by heads"
        self.h, self.dk = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x, pad_mask=None, return_attn=False):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)  # each (B,h,T,dk)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dk)                        # (B,h,T,T)
        if pad_mask is not None:                                                      # never attend to padding
            scores = scores.masked_fill(pad_mask[:, None, None, :], float("-inf"))
        attn = scores.softmax(-1)
        y = (attn @ v).transpose(1, 2).reshape(B, T, D)
        y = self.out(y)
        return (y, attn) if return_attn else y


class Block(nn.Module):
    """Pre-LayerNorm transformer block: x + Attn(LN(x)); x + MLP(LN(x))."""

    def __init__(self, d, heads, ff, drop=0.1):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = SelfAttention(d, heads)
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x, pad_mask=None, return_attn=False):
        a = self.attn(self.ln1(x), pad_mask, return_attn)
        attn = None
        if return_attn:
            a, attn = a
        x = x + self.drop(a)
        x = x + self.drop(self.mlp(self.ln2(x)))
        return (x, attn) if return_attn else x


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes=2, d=32, heads=4, layers=2, ff=64, max_len=16, drop=0.1, pad_id=0):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d, padding_idx=pad_id)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(Block(d, heads, ff, drop) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, num_classes)

    def forward(self, ids, pad_mask=None, return_attn=False):
        pos = torch.arange(ids.size(1), device=ids.device)
        x = self.tok(ids) + self.pos(pos)
        attns = []
        for b in self.blocks:
            if return_attn:
                x, a = b(x, pad_mask, True); attns.append(a)
            else:
                x = b(x, pad_mask)
        logits = self.head(self.ln(x)[:, 0])        # use the <cls> position (index 0) as sentence summary
        return (logits, attns) if return_attn else logits
