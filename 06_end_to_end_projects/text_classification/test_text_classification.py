import unittest
import torch
from data import make_corpus, Tokenizer
from model import TransformerClassifier, SelfAttention
from train import fit, evaluate, predict


class Tests(unittest.TestCase):
    def test_corpus_labels(self):
        texts, y = make_corpus(200, seed=3)
        for t, l in zip(texts, y.tolist()):
            w = t.split()
            pos = any(a in w for a in ["good", "great", "fun", "lovely", "superb", "nice"])
            self.assertEqual(l, int(pos != ("not" in w)))

    def test_tokenizer(self):
        texts, _ = make_corpus(100)
        tok = Tokenizer(texts)
        ids, mask = tok.batch(["the movie was good", "this is not very bad today"])
        self.assertEqual(ids[0, 0].item(), tok.cls_id)
        self.assertTrue(mask[0, -1].item())            # shorter sentence is padded
        self.assertEqual(tok.encode("zzzz")[1], tok.unk_id)

    def test_padding_does_not_change_output(self):
        torch.manual_seed(0)
        texts, _ = make_corpus(50)
        tok = Tokenizer(texts)
        m = TransformerClassifier(len(tok), drop=0.0).eval()
        a = m(*tok.batch(["the movie was good"]))
        ids, mask = tok.batch(["the movie was good", "this is not very bad today again"])
        b = m(ids, mask)[:1]
        self.assertTrue(torch.allclose(a, b, atol=1e-5))

    def test_attention_rows_sum_to_one(self):
        att = SelfAttention(16, 4)
        _, a = att(torch.randn(2, 5, 16), return_attn=True)
        self.assertTrue(torch.allclose(a.sum(-1), torch.ones(2, 4, 5), atol=1e-5))

    def test_learns_negation(self):
        torch.manual_seed(0)
        texts, y = make_corpus(1200)
        tok = Tokenizer(texts)
        tr, va = (texts[300:], y[300:]), (texts[:300], y[:300])
        m = TransformerClassifier(len(tok))
        fit(m, tok, tr, va, epochs=12, verbose=False)
        self.assertGreater(evaluate(m, tok, *va)[1], 0.9)
        self.assertIn(predict(m, tok, "the movie was not good")["label"], ["positive", "negative"])


if __name__ == "__main__":
    unittest.main()
