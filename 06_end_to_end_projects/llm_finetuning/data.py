"""Toy corpus for a character-level GPT: plain sentences (pretraining) + instruction pairs (finetuning)."""
import random
import torch

OPPOSITES = [("hot", "cold"), ("big", "small"), ("up", "down"), ("fast", "slow"), ("happy", "sad"),
             ("light", "dark"), ("old", "new"), ("high", "low"), ("hard", "soft"), ("wet", "dry")]
NOUNS = ["cat", "dog", "bird", "fish", "tree", "road", "lamp", "bell", "rock", "boat",
         "wolf", "moon", "star", "rain", "wind", "milk", "bread", "cake", "door", "wall"]
ADJ = [w for p in OPPOSITES for w in p]
SENT = ["the {n} is {a} .", "a {a} {n} sat here .", "i saw the {a} {n} .", "the {n} was very {a} ."]

ALPHABET = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .:?\n") 


class CharTokenizer:
    """Character-level tokenizer. id 0 is reserved for padding."""

    def __init__(self, alphabet=ALPHABET):
        self.itos = ["<pad>"] + list(alphabet)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids if i != 0)


def make_pretrain_text(n_sent=1500, seed=0):
    """Plain English-like sentences built from nouns and adjectives. No instruction format at all."""
    rng = random.Random(seed)
    return "\n".join(rng.choice(SENT).format(n=rng.choice(NOUNS), a=rng.choice(ADJ)) for _ in range(n_sent)) + "\n"


ANIMALS = {"cat", "dog", "bird", "fish", "wolf"}


def make_instruction_pairs(seed=0):
    """Return (train_pairs, eval_pairs) of (prompt, answer). Prompt ends with 'A: '; answer ends with newline.

    Two tiny 'skills' expressed as instructions:
      'opposite X' -> antonym        (20 facts, both directions)
      'kind X'     -> animal / thing (20 facts)
    The facts are in the SAME pretraining vocabulary, but the Q:/A: format and the mapping are new to the base
    model. eval_pairs == train_pairs: this toy measures whether the adapter can *absorb* the instruction format
    and the facts (recall), not generalisation to unseen facts (impossible for arbitrary facts).
    """
    train = []
    for a, b in OPPOSITES:
        for x, y in ((a, b), (b, a)):
            train.append((f"Q: opposite {x}\nA: ", y + "\n"))
    for n in NOUNS:
        train.append((f"Q: kind {n}\nA: ", ("animal" if n in ANIMALS else "thing") + "\n"))
    random.Random(seed).shuffle(train)
    return train, list(train)


def lm_batch(ids: torch.Tensor, block: int, bs: int, gen: torch.Generator):
    """Random windows for next-token prediction. Returns x, y shifted by one."""
    starts = torch.randint(0, len(ids) - block - 1, (bs,), generator=gen)
    x = torch.stack([ids[s:s + block] for s in starts])
    y = torch.stack([ids[s + 1:s + block + 1] for s in starts])
    return x, y


def sft_batch(tok, pairs, block=40):
    """Encode (prompt+answer) pairs; mask[i,t]=1 only where the TARGET is an answer character.
    Returns x, y, mask with y = x shifted left. Padding id 0 is excluded from the loss."""
    xs, ys, ms = [], [], []
    for p, a in pairs:
        ids = tok.encode(p + a)[:block + 1]
        n_prompt = len(p)
        pad = block + 1 - len(ids)
        ids = ids + [0] * pad
        m = [0] * (block)
        for t in range(block):                       # target at position t is ids[t+1]
            if t + 1 >= n_prompt and ids[t + 1] != 0:
                m[t] = 1
        xs.append(ids[:-1]); ys.append(ids[1:]); ms.append(m)
    return torch.tensor(xs), torch.tensor(ys), torch.tensor(ms, dtype=torch.float)
