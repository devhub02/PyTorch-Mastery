import unittest
import torch
from data import make_interactions, leave_k_out
from model import TwoTower, recall_at_k, _recall_from_scores, popularity_scores
from train import in_batch_softmax_loss, fit


class Tests(unittest.TestCase):
    def setUp(self):
        self.d = make_interactions(n_users=120, n_items=80, per_user=20)
        self.train, self.test = leave_k_out(self.d["pairs"], 120, k=3)

    def test_split_disjoint(self):
        a = set(map(tuple, self.train.tolist())); b = set(map(tuple, self.test.tolist()))
        self.assertFalse(a & b)
        self.assertEqual(len(self.test), 120 * 3)

    def test_recall_perfect_and_zero(self):
        n_u, n_i = 120, 80
        perfect = torch.zeros(n_u, n_i); perfect[self.test[:, 0], self.test[:, 1]] = 1.0
        self.assertAlmostEqual(_recall_from_scores(perfect, self.train, self.test, n_u, 10), 1.0)
        zero = torch.zeros(n_u, n_i); zero[self.test[:, 0], self.test[:, 1]] = -1.0
        self.assertLess(_recall_from_scores(zero, self.train, self.test, n_u, 3), 0.01)

    def test_loss_masks_duplicates(self):
        u = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
        items = torch.tensor([1, 1, 2, 3])
        l = in_batch_softmax_loss(u, u.clone(), items)
        self.assertTrue(torch.isfinite(l))

    def test_beats_popularity(self):
        torch.manual_seed(0)
        m = TwoTower(120, 80, self.d["item_group"], n_groups=5)
        fit(m, self.train, self.test, 120, epochs=12, verbose=False)
        r = recall_at_k(m, self.train, self.test, 120)
        pop = _recall_from_scores(popularity_scores(self.train, 120, 80), self.train, self.test, 120, 10)
        self.assertGreater(r, pop)


if __name__ == "__main__":
    unittest.main()
