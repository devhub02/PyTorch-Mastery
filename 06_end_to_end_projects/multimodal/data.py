"""Synthetic image+caption pairs: colored shapes on 16x16 RGB canvases with templated captions."""
import random
import torch

SHAPES = ["circle", "square", "triangle", "cross"]
COLORS = {"red": (1.0, 0.1, 0.1), "green": (0.1, 0.9, 0.1), "blue": (0.1, 0.2, 1.0), "yellow": (1.0, 0.9, 0.1)}
CONCEPTS = [(c, s) for c in COLORS for s in SHAPES]          # 16 (color, shape) combinations
TEMPLATES = ["a {c} {s}", "{s} in {c}", "the {s} is {c}", "photo of a {c} {s}"]
HELD_OUT = [("blue", "cross"), ("yellow", "triangle"), ("green", "square")]   # never seen in training


def _mask(shape, size, g):
    r = 3 + torch.rand(1, generator=g).item() * 2.5
    m = r + 1
    cx = m + torch.rand(1, generator=g).item() * (size - 2 * m)
    cy = m + torch.rand(1, generator=g).item() * (size - 2 * m)
    ys, xs = torch.meshgrid(torch.arange(size).float(), torch.arange(size).float(), indexing="ij")
    dx, dy = xs - cx, ys - cy
    if shape == "circle":
        return dx ** 2 + dy ** 2 <= r ** 2
    if shape == "square":
        return (dx.abs() <= r * 0.85) & (dy.abs() <= r * 0.85)
    if shape == "triangle":
        t = (dy + r) / (2 * r)
        return (t >= 0) & (t <= 1) & (dx.abs() <= t * r)
    return ((dx.abs() <= 1) & (dy.abs() <= r)) | ((dy.abs() <= 1) & (dx.abs() <= r))


def make_pairs(n=800, concepts=CONCEPTS, size=16, noise=0.1, seed=0):
    """Return images[N,3,S,S], captions[list of str], concept_id[N] (index into CONCEPTS)."""
    g = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    imgs, caps, cid = [], [], []
    for i in range(n):
        c, s = concepts[i % len(concepts)]
        img = torch.full((3, size, size), 0.1) + 0.05 * torch.rand(3, 1, 1, generator=g)
        m = _mask(s, size, g)
        for ch, v in enumerate(COLORS[c]):
            img[ch][m] = v
        img = (img + noise * torch.randn(img.shape, generator=g)).clamp(0, 1)
        imgs.append(img)
        caps.append(rng.choice(TEMPLATES).format(c=c, s=s))
        cid.append(CONCEPTS.index((c, s)))
    perm = torch.randperm(n, generator=g)
    return torch.stack(imgs)[perm], [caps[i] for i in perm], torch.tensor(cid)[perm]


class Tokenizer:
    """Word-level tokenizer over the closed caption vocabulary. id 0 = pad."""

    def __init__(self):
        words = set()
        for t in TEMPLATES:
            words |= set(t.replace("{c}", "").replace("{s}", "").split())
        words |= set(COLORS) | set(SHAPES)
        self.itos = ["<pad>", "<unk>"] + sorted(words)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def batch(self, texts, max_len=6):
        ids = torch.zeros(len(texts), max_len, dtype=torch.long)
        for i, t in enumerate(texts):
            toks = [self.stoi.get(w, 1) for w in t.lower().split()][:max_len]
            ids[i, :len(toks)] = torch.tensor(toks)
        return ids
