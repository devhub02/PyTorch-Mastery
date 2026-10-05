"""Synthetic implicit-feedback interactions from a hidden latent-factor world."""
import torch


def make_interactions(n_users=300, n_items=200, n_groups=5, per_user=25, seed=0):
    """Users and items each belong to a taste group with a latent vector. A user clicks items with
    probability ~ softmax(affinity + popularity bias). Returns dict with arrays:
      pairs[N,2] (user, item) unique positives, item_group[n_items], n_users, n_items."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(n_groups, 6, generator=g) * 1.5
    ug = torch.randint(0, n_groups, (n_users,), generator=g)
    ig = torch.randint(0, n_groups, (n_items,), generator=g)
    uvec = centers[ug] + 0.5 * torch.randn(n_users, 6, generator=g)
    ivec = centers[ig] + 0.5 * torch.randn(n_items, 6, generator=g)
    pop = torch.randn(n_items, generator=g) * 0.8                       # some items are globally popular
    logits = uvec @ ivec.T / 3.0 + pop
    probs = logits.softmax(-1)
    rows = []
    for u in range(n_users):
        items = torch.multinomial(probs[u], per_user, replacement=False, generator=g)
        rows += [(u, int(i)) for i in items]
    return {"pairs": torch.tensor(rows), "item_group": ig, "user_group": ug, "n_users": n_users, "n_items": n_items}


def leave_k_out(pairs, n_users, k=3, seed=0):
    """Hold out k random positives per user for testing; the rest are training interactions."""
    g = torch.Generator().manual_seed(seed)
    train, test = [], []
    for u in range(n_users):
        its = pairs[pairs[:, 0] == u]
        perm = torch.randperm(len(its), generator=g)
        test.append(its[perm[:k]]); train.append(its[perm[k:]])
    return torch.cat(train), torch.cat(test)


def train_batches(train_pairs, bs, gen):
    perm = torch.randperm(len(train_pairs), generator=gen)
    for i in range(0, len(perm) - bs + 1, bs):          # drop last incomplete batch (keeps batch size fixed)
        b = train_pairs[perm[i:i + bs]]
        yield b[:, 0], b[:, 1]
