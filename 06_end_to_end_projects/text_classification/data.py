"""Synthetic sentiment corpus with negation, plus a tiny word-level tokenizer.

Sentence template:  <det> <noun> <verb> [not] [very] <adj> [filler...]
Label = 1 (positive) if (adj is positive) XOR (sentence is negated). So bag-of-words
alone is not enough: the model must combine "not" with the adjective.
"""
import random
import torch

DET = ["the", "a", "this", "that"]
NOUN = ["movie", "food", "phone", "book", "game", "hotel", "song", "show"]
VERB = ["was", "is", "seemed", "felt"]
POS = ["good", "great", "fun", "lovely", "superb", "nice"]
NEG = ["bad", "awful", "boring", "poor", "dull", "weak"]
FILL = ["today", "really", "overall", "honestly", "indeed", "again", "here"]

PAD, UNK, CLS = "<pad>", "<unk>", "<cls>"


def make_corpus(n=1200, seed=0, p_not=0.4):
    rng = random.Random(seed)
    texts, labels = [], []
    for i in range(n):
        pos_adj = rng.random() < 0.5
        neg = rng.random() < p_not
        w = [rng.choice(DET), rng.choice(NOUN), rng.choice(VERB)]
        if neg:
            w.append("not")
        if rng.random() < 0.3:
            w.append("very")
        w.append(rng.choice(POS if pos_adj else NEG))
        w += [rng.choice(FILL) for _ in range(rng.randint(0, 3))]
        texts.append(" ".join(w))
        labels.append(int(pos_adj != neg))
    return texts, torch.tensor(labels)


class Tokenizer:
    """Word-level tokenizer: lowercase + whitespace split, vocab built from a corpus."""

    def __init__(self, texts, min_freq=1):
        counts = {}
        for t in texts:
            for w in t.lower().split():
                counts[w] = counts.get(w, 0) + 1
        self.itos = [PAD, UNK, CLS] + sorted(w for w, c in counts.items() if c >= min_freq)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_id, self.unk_id, self.cls_id = 0, 1, 2

    def __len__(self):
        return len(self.itos)

    def encode(self, text, max_len=16):
        ids = [self.cls_id] + [self.stoi.get(w, self.unk_id) for w in text.lower().split()]
        return ids[:max_len]

    def decode(self, ids):
        return " ".join(self.itos[i] for i in ids if i != self.pad_id)

    def batch(self, texts, max_len=16):
        """Encode and right-pad to the longest sequence in the batch. Returns (ids, pad_mask[True=pad])."""
        enc = [self.encode(t, max_len) for t in texts]
        L = max(len(e) for e in enc)
        ids = torch.full((len(enc), L), self.pad_id, dtype=torch.long)
        for i, e in enumerate(enc):
            ids[i, :len(e)] = torch.tensor(e)
        return ids, ids == self.pad_id
