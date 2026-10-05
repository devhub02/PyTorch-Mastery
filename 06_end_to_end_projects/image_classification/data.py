"""Synthetic shape images: circle / square / triangle / cross on noisy 16x16 canvases."""
import math
import torch

CLASSES = ["circle", "square", "triangle", "cross"]


def draw_shape(kind: int, size: int, g: torch.Generator) -> torch.Tensor:
    """Return a (size, size) float image in [0,1] with one randomly placed/sized shape."""
    r = 3 + torch.rand(1, generator=g).item() * 3            # radius / half-extent 3..6
    margin = r + 1
    cx = margin + torch.rand(1, generator=g).item() * (size - 2 * margin)
    cy = margin + torch.rand(1, generator=g).item() * (size - 2 * margin)
    ys, xs = torch.meshgrid(torch.arange(size).float(), torch.arange(size).float(), indexing="ij")
    dx, dy = xs - cx, ys - cy
    if kind == 0:      # filled circle
        mask = dx ** 2 + dy ** 2 <= r ** 2
    elif kind == 1:    # filled square
        mask = (dx.abs() <= r * 0.85) & (dy.abs() <= r * 0.85)
    elif kind == 2:    # filled upward triangle: width grows linearly with y
        t = (dy + r) / (2 * r)                                # 0 at apex, 1 at base
        mask = (t >= 0) & (t <= 1) & (dx.abs() <= t * r)
    elif kind == 3:    # plus/cross made of two thin bars
        mask = ((dx.abs() <= 1) & (dy.abs() <= r)) | ((dy.abs() <= 1) & (dx.abs() <= r))
    else:
        raise ValueError(f"unknown shape kind {kind}")
    return mask.float()


def make_dataset(n: int = 800, size: int = 16, noise: float = 0.15, seed: int = 0):
    """Return (images[N,1,size,size], labels[N]); class-balanced and fully deterministic."""
    g = torch.Generator().manual_seed(seed)
    labels = torch.arange(n) % len(CLASSES)
    imgs = torch.stack([draw_shape(int(k), size, g) for k in labels])
    imgs = imgs + noise * torch.randn(imgs.shape, generator=g)
    imgs = imgs.clamp(0, 1).unsqueeze(1)
    perm = torch.randperm(n, generator=g)
    return imgs[perm], labels[perm]


def train_val_split(x, y, val_frac: float = 0.25):
    n_val = int(len(x) * val_frac)
    return (x[n_val:], y[n_val:]), (x[:n_val], y[:n_val])
