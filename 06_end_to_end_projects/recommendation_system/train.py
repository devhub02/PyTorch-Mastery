import torch
import torch.nn.functional as F
from data import train_batches
from model import recall_at_k


def in_batch_softmax_loss(u_vec, i_vec, item_ids, temperature=0.1, log_q=None):
    """Row i's positive is column i; every other item in the batch is a negative for user i.
    Items that are the same as the positive (duplicates in the batch) are masked out (false negatives).
    log_q: optional log sampling probability of each batch item (logQ correction for popularity bias)."""
    logits = u_vec @ i_vec.T / temperature                       # (B, B)
    if log_q is not None:
        logits = logits - log_q[None, :]
    same = item_ids[:, None] == item_ids[None, :]
    same.fill_diagonal_(False)
    logits = logits.masked_fill(same, float("-inf"))
    return F.cross_entropy(logits, torch.arange(len(u_vec)))


def fit(model, train_pairs, test_pairs, n_users, epochs=15, bs=128, lr=5e-3, temperature=0.1,
        logq=False, seed=0, verbose=True, k=10):
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    freq = torch.bincount(train_pairs[:, 1], minlength=model.n_items).float()
    log_q_all = torch.log(freq.clamp(min=1) / freq.sum())
    hist = []
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for u, i in train_batches(train_pairs, bs, gen):
            loss = in_batch_softmax_loss(model.user_vec(u), model.item_vec(i), i, temperature,
                                         log_q_all[i] if logq else None)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        r = recall_at_k(model, train_pairs, test_pairs, n_users, k)
        hist.append((tot / n, r))
        if verbose:
            print(f"epoch {ep+1:2d} | loss {tot/n:.3f} | recall@{k} {r:.3f}")
    return hist
