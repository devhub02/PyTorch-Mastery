import torch
import torch.nn.functional as F
from data import lm_batch, sft_batch


def pretrain(model, ids, steps=400, bs=32, lr=3e-3, seed=0, log=100, verbose=True):
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    losses = []
    for s in range(steps):
        x, y = lm_batch(ids, model.block, bs, gen)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        losses.append(loss.item())
        if verbose and (s % log == 0 or s == steps - 1):
            print(f"pretrain step {s:4d} | loss {loss.item():.3f}")
    return losses


def finetune(model, tok, pairs, steps=300, bs=16, lr=3e-3, seed=0, log=100, verbose=True):
    """Train only parameters with requires_grad=True (e.g. LoRA adapters). Loss only on answer characters."""
    gen = torch.Generator().manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    losses = []
    for s in range(steps):
        idx = torch.randint(0, len(pairs), (bs,), generator=gen).tolist()
        x, y, m = sft_batch(tok, [pairs[i] for i in idx], block=model.block)
        ce = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), reduction="none")
        loss = (ce * m.flatten()).sum() / m.sum()          # masked mean over answer tokens only
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if verbose and (s % log == 0 or s == steps - 1):
            print(f"finetune step {s:4d} | answer loss {loss.item():.3f}")
    return losses


def complete(model, tok, prompt, max_new=12, temperature=0.0):
    idx = torch.tensor([tok.encode(prompt)])
    out = model.generate(idx, max_new=max_new, temperature=temperature, stop_id=tok.stoi["\n"])
    return tok.decode(out[0, idx.size(1):].tolist())


def exact_match(model, tok, pairs):
    hits = sum(complete(model, tok, p).strip() == a.strip() for p, a in pairs)
    return hits / len(pairs)
