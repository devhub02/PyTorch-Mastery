import unittest
import torch
from data import CharTokenizer, make_pretrain_text, make_instruction_pairs, sft_batch
from model import TinyGPT, apply_lora, merge_lora, count_params, LoRALinear
from train import pretrain, finetune, exact_match, complete


class Tests(unittest.TestCase):
    def test_tokenizer_roundtrip(self):
        tok = CharTokenizer()
        s = "Q: upper cat\nA: CAT\n"
        self.assertEqual(tok.decode(tok.encode(s)), s)

    def test_sft_mask_only_answer(self):
        tok = CharTokenizer()
        x, y, m = sft_batch(tok, [("Q: a\nA: ", "B\n")], block=16)
        tgt = "".join(tok.itos[t] for t, k in zip(y[0].tolist(), m[0].tolist()) if k)
        self.assertEqual(tgt, "B\n")

    def test_causality(self):
        torch.manual_seed(0)
        m = TinyGPT(60).eval()
        a = torch.randint(1, 60, (1, 10)); b = a.clone(); b[0, -1] = (b[0, -1] % 58) + 1
        self.assertTrue(torch.allclose(m(a)[:, :-1], m(b)[:, :-1], atol=1e-5))

    def test_lora_zero_init_freeze_merge(self):
        torch.manual_seed(0)
        m = TinyGPT(60).eval(); x = torch.randint(1, 60, (2, 12)); ref = m(x)
        apply_lora(m, r=4)
        self.assertTrue(torch.allclose(m(x), ref, atol=1e-6))       # B=0 -> unchanged
        total, train = count_params(m)
        self.assertLess(train, total)
        for mod in m.modules():
            if isinstance(mod, LoRALinear):
                torch.nn.init.normal_(mod.B, std=0.1)
        out = m(x); merge_lora(m)
        self.assertTrue(torch.allclose(m(x), out, atol=1e-4))

    def test_pipeline_runs(self):
        torch.manual_seed(0)
        tok = CharTokenizer()
        ids = torch.tensor(tok.encode(make_pretrain_text(300)))
        m = TinyGPT(len(tok))
        l = pretrain(m, ids, steps=30, verbose=False)
        self.assertLess(l[-1], l[0])
        apply_lora(m, r=4)
        train, test = make_instruction_pairs()
        l = finetune(m, tok, train, steps=30, verbose=False)
        self.assertLess(l[-1], l[0])
        self.assertTrue(0 <= exact_match(m, tok, test) <= 1)


if __name__ == "__main__":
    unittest.main()
