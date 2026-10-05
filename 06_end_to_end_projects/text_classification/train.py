import torch
import torch.nn.functional as F


def make_batches(tok, texts, labels, bs, shuffle, gen=None, max_len=16):
    idx = torch.randperm(len(texts), generator=gen) if shuffle else torch.arange(len(texts))
    for i in range(0, len(texts), bs):
        j = idx[i:i + bs].tolist()
        ids, mask = tok.batch([texts[k] for k in j], max_len)
        yield ids, mask, labels[j]


@torch.no_grad()
def evaluate(model, tok, texts, labels, bs=128):
    model.eval()
    correct, loss = 0, 0.0
    for ids, mask, y in make_batches(tok, texts, labels, bs, False):
        out = model(ids, mask)
        loss += F.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(-1) == y).sum().item()
    return loss / len(texts), correct / len(texts)


def fit(model, tok, train, val, epochs=12, lr=2e-3, bs=32, seed=0, verbose=True):
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    hist = []
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for ids, mask, y in make_batches(tok, *train, bs, True, gen):
            opt.zero_grad()
            loss = F.cross_entropy(model(ids, mask), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(y); n += len(y)
        vl, va = evaluate(model, tok, *val)
        hist.append((tot / n, vl, va))
        if verbose:
            print(f"epoch {ep+1:2d} | train loss {tot/n:.3f} | val loss {vl:.3f} | val acc {va:.3f}")
    return hist


@torch.no_grad()
def predict(model, tok, text):
    model.eval()
    ids, mask = tok.batch([text])
    p = model(ids, mask).softmax(-1)[0]
    return {"label": "positive" if p[1] > p[0] else "negative", "p_positive": round(p[1].item(), 4)}
