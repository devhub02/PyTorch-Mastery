"""CLIP-style dual encoder: image CNN + text transformer -> shared, L2-normalised embedding space."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    def __init__(self, dim=32, width=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.BatchNorm2d(width * 2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 2, 3, padding=1), nn.ReLU())
        self.proj = nn.Linear(width * 2, dim)

    def forward(self, x):
        return self.proj(self.net(x).mean(dim=(2, 3)))


class TextEncoder(nn.Module):
    def __init__(self, vocab, dim=32, max_len=6, heads=4):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim, padding_idx=0)
        self.pos = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, ids):
        pad = ids == 0
        h = self.tok(ids) + self.pos(torch.arange(ids.size(1)))
        h = self.enc(h, src_key_padding_mask=pad)
        keep = (~pad).unsqueeze(-1).float()
        return self.proj((h * keep).sum(1) / keep.sum(1))      # mean over real tokens only


class CLIP(nn.Module):
    def __init__(self, vocab, dim=32):
        super().__init__()
        self.image = ImageEncoder(dim)
        self.text = TextEncoder(vocab, dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))   # learnable temperature, as in CLIP

    def encode_image(self, x):
        return F.normalize(self.image(x), dim=-1)

    def encode_text(self, ids):
        return F.normalize(self.text(ids), dim=-1)

    def forward(self, x, ids):
        return self.encode_image(x), self.encode_text(ids), self.logit_scale.exp().clamp(max=100)
