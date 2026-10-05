"""Training / evaluation helpers."""
import torch
import torch.nn.functional as F


def batches(x, y, bs, shuffle, gen=None):
    idx = torch.randperm(len(x), generator=gen) if shuffle else torch.arange(len(x))
    for i in range(0, len(x), bs):
        j = idx[i:i + bs]
        yield x[j], y[j]


def train_epoch(model, opt, x, y, bs=32, gen=None):
    model.train()
    tot, n = 0.0, 0
    for xb, yb in batches(x, y, bs, True, gen):
        opt.zero_grad()
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()
        tot += loss.item() * len(xb); n += len(xb)
    return tot / n


@torch.no_grad()
def evaluate(model, x, y, bs=256):
    model.eval()
    preds = torch.cat([model(xb).argmax(-1) for xb, _ in batches(x, y, bs, False)])
    loss = F.cross_entropy(model(x), y).item()
    return loss, (preds == y).float().mean().item(), preds


def confusion_matrix(preds, y, k):
    cm = torch.zeros(k, k, dtype=torch.long)
    for t, p in zip(y.tolist(), preds.tolist()):
        cm[t, p] += 1
    return cm


def fit(model, train, val, epochs=8, lr=3e-3, seed=0, verbose=True):
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for ep in range(epochs):
        tl = train_epoch(model, opt, *train, gen=gen)
        vl, va, _ = evaluate(model, *val)
        hist.append((tl, vl, va))
        if verbose:
            print(f"epoch {ep+1:2d} | train loss {tl:.3f} | val loss {vl:.3f} | val acc {va:.3f}")
    return hist
