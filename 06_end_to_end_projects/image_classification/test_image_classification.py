import os, tempfile, unittest
import torch
from data import make_dataset, train_val_split, CLASSES
from model import SmallCNN, save_checkpoint, load_checkpoint, Predictor
from train import fit, evaluate


class Tests(unittest.TestCase):
    def test_dataset_shape_and_determinism(self):
        x, y = make_dataset(40, seed=1)
        x2, _ = make_dataset(40, seed=1)
        self.assertEqual(x.shape, (40, 1, 16, 16))
        self.assertTrue(torch.equal(x, x2))
        self.assertEqual(y.bincount().tolist(), [10] * 4)
        self.assertTrue(0 <= x.min() and x.max() <= 1)

    def test_forward_shape(self):
        self.assertEqual(SmallCNN()(torch.zeros(3, 1, 16, 16)).shape, (3, len(CLASSES)))

    def test_learns_and_roundtrips(self):
        torch.manual_seed(0)
        x, y = make_dataset(800)
        tr, va = train_val_split(x, y)
        m = SmallCNN()
        fit(m, tr, va, epochs=10, verbose=False)
        acc = evaluate(m, *va)[1]
        self.assertGreater(acc, 0.8)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "ck.pt")
            save_checkpoint(m, p, acc=acc)
            m2, _ = load_checkpoint(p)
            self.assertTrue(torch.allclose(m.eval()(va[0]), m2(va[0]), atol=1e-6))
            out = Predictor(p).handle_request({"image": va[0][0, 0].tolist()})
            self.assertIn(out["label"], CLASSES)
            self.assertIn("error", Predictor(p).handle_request({}))


if __name__ == "__main__":
    unittest.main()
