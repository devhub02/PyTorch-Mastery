import unittest
import torch
from data import make_pairs, Tokenizer, CONCEPTS, HELD_OUT
from model import CLIP
from train import clip_loss, fit, retrieval_metrics, zero_shot_classify


class Tests(unittest.TestCase):
    def test_data_and_tokenizer(self):
        x, caps, cid = make_pairs(32)
        self.assertEqual(x.shape, (32, 3, 16, 16))
        self.assertEqual(len(caps), 32)
        tok = Tokenizer()
        ids = tok.batch(caps[:4])
        self.assertEqual(ids.shape, (4, 6))
        self.assertTrue((ids != 1).all())            # no unknown words in templated captions

    def test_loss_perfect_vs_random(self):
        e = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
        self.assertLess(clip_loss(e, e.clone(), 50.0).item(), 0.01)
        r = clip_loss(e, torch.nn.functional.normalize(torch.randn(8, 16), dim=-1), 1.0).item()
        self.assertGreater(r, 1.5)                  # close to ln 8 = 2.08 for scale 1

    def test_shapes_and_unit_norm(self):
        tok = Tokenizer(); m = CLIP(len(tok))
        ie, te, s = m(torch.rand(5, 3, 16, 16), tok.batch(["a red circle"] * 5))
        self.assertEqual(ie.shape, (5, 32)); self.assertEqual(te.shape, (5, 32))
        self.assertTrue(torch.allclose(ie.norm(dim=-1), torch.ones(5), atol=1e-5))

    def test_learns_retrieval(self):
        torch.manual_seed(0)
        train = [c for c in CONCEPTS if c not in HELD_OUT]
        x, caps, cid = make_pairs(640, concepts=train, seed=0)
        tok = Tokenizer(); m = CLIP(len(tok))
        fit(m, tok, x, caps, cid, epochs=20, verbose=False)
        xv, cv, idv = make_pairs(130, concepts=train, seed=1)
        r = retrieval_metrics(m, tok, xv, cv, idv)
        self.assertGreater(r["i2t_R@1"], 0.8)
        self.assertGreater(r["t2i_P@5"], 0.7)


if __name__ == "__main__":
    unittest.main()
