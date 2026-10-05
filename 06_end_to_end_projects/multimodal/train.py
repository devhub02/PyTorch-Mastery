import torch
import torch.nn.functional as F


def clip_loss(img_emb, txt_emb, scale, group=None):
    """Symmetric InfoNCE. logits[i,j] = scale * <image_i, text_j>. Positive = diagonal.
    group: optional ids; entries with equal group id but i != j are true positives hiding as negatives -> masked."""
    logits = scale * img_emb @ txt_emb.T
    if group is not None:
        same = group[:, None] == group[None, :]
        same.fill_diagonal_(False)
        logits = logits.masked_fill(same, float("-inf"))
    t = torch.arange(len(logits))
    return (F.cross_entropy(logits, t) + F.cross_entropy(logits.T, t)) / 2


def fit(model, tok, images, captions, concept, epochs=15, bs=64, lr=3e-3, seed=0, mask_dups=True, verbose=True):
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    hist = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(images), generator=gen)
        tot, n = 0.0, 0
        for i in range(0, len(perm) - bs + 1, bs):
            j = perm[i:i + bs]
            ie, te, s = model(images[j], tok.batch([captions[k] for k in j.tolist()]))
            loss = clip_loss(ie, te, s, concept[j] if mask_dups else None)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        hist.append(tot / n)
        if verbose:
            print(f"epoch {ep+1:2d} | loss {tot/n:.3f} | logit scale {model.logit_scale.exp().item():.1f}")
    return hist


@torch.no_grad()
def retrieval_metrics(model, tok, images, captions, concept):
    """Concept-level retrieval metrics (many images share a concept, so exact-index matching would punish
    correct answers).
      i2t_R@1 / i2t_R@3: rank one canonical caption per concept present; correct if the true concept is in top-k.
      t2i_P@5: for every caption, the fraction of its top-5 retrieved images that show the right concept."""
    from data import CONCEPTS
    model.eval()
    ie = model.encode_image(images)
    present = concept.unique()
    canon = [f"a {CONCEPTS[c][0]} {CONCEPTS[c][1]}" for c in present.tolist()]
    sim_i2t = ie @ model.encode_text(tok.batch(canon)).T                     # (N images, n concepts)
    ranked = present[sim_i2t.topk(3, dim=1).indices]                        # concept ids, best first
    out = {"i2t_R@1": (ranked[:, 0] == concept).float().mean().item(),
           "i2t_R@3": (ranked == concept[:, None]).any(1).float().mean().item()}
    sim_t2i = model.encode_text(tok.batch(captions)) @ ie.T                  # (N captions, N images)
    top = sim_t2i.topk(5, dim=1).indices
    out["t2i_P@5"] = (concept[top] == concept[:, None]).float().mean().item()
    return out


@torch.no_grad()
def zero_shot_classify(model, tok, images, prompts):
    """Classify images by nearest text prompt. Returns predicted prompt index per image."""
    model.eval()
    te = model.encode_text(tok.batch(prompts))
    return (model.encode_image(images) @ te.T).argmax(-1)
