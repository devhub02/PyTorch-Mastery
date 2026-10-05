"""Two-tower retrieval model and ranking metrics."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Tower(nn.Module):
    """ID embedding (+ optional categorical side feature) -> MLP -> L2-normalised vector."""

    def __init__(self, n_ids, n_cat=0, emb=16, out=16, hidden=32):
        super().__init__()
        self.id_emb = nn.Embedding(n_ids, emb)
        self.cat_emb = nn.Embedding(n_cat, emb) if n_cat else None
        self.mlp = nn.Sequential(nn.Linear(emb * (2 if n_cat else 1), hidden), nn.ReLU(), nn.Linear(hidden, out))

    def forward(self, ids, cats=None):
        h = self.id_emb(ids)
        if self.cat_emb is not None:
            h = torch.cat([h, self.cat_emb(cats)], -1)
        return F.normalize(self.mlp(h), dim=-1)          # unit vectors -> dot product is cosine similarity


class TwoTower(nn.Module):
    def __init__(self, n_users, n_items, item_group, n_groups=0, emb=16, out=16):
        super().__init__()
        self.register_buffer("item_group", item_group)    # item side feature (known for every item)
        self.user = Tower(n_users, 0, emb, out)
        self.item = Tower(n_items, n_groups, emb, out)
        self.n_items = n_items

    def user_vec(self, u):
        return self.user(u)

    def item_vec(self, i):
        return self.item(i, self.item_group[i] if self.item.cat_emb is not None else None)

    @torch.no_grad()
    def all_item_vecs(self):
        return self.item_vec(torch.arange(self.n_items))


def recall_at_k(model, train_pairs, test_pairs, n_users, k=10):
    """Mean over users of |top-k recommended ∩ held-out| / min(k, |held-out|). Train positives are excluded."""
    model.eval()
    with torch.no_grad():
        scores = model.user_vec(torch.arange(n_users)) @ model.all_item_vecs().T
    return _recall_from_scores(scores, train_pairs, test_pairs, n_users, k)


def _recall_from_scores(scores, train_pairs, test_pairs, n_users, k):
    scores = scores.clone()
    scores[train_pairs[:, 0], train_pairs[:, 1]] = float("-inf")     # never recommend what the user already has
    top = scores.topk(k, dim=1).indices
    truth = torch.zeros_like(scores, dtype=torch.bool)
    truth[test_pairs[:, 0], test_pairs[:, 1]] = True
    hits = truth.gather(1, top).sum(1).float()
    denom = truth.sum(1).clamp(max=k).float()
    valid = truth.sum(1) > 0
    return (hits[valid] / denom[valid]).mean().item()


def popularity_scores(train_pairs, n_users, n_items):
    counts = torch.bincount(train_pairs[:, 1], minlength=n_items).float()
    return counts.expand(n_users, -1)
