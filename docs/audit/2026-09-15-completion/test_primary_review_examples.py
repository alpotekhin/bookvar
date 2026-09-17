"""Bounded CPU checks of the primary review's DPO and derivative corrections.

The sequence_logp function is extracted from the current chapter. These checks
do not train a language model or exercise any remote service.
"""
import ast
import math
import re
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
DPO = ROOT / "00 Учебник/12 Post-training и Alignment/05 DPO.md"


def page_sequence_logp():
    for block in re.findall(r"```python\n(.*?)\n```", DPO.read_text(), re.S):
        tree = ast.parse(block)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "sequence_logp":
                module = ast.Module(body=[node], type_ignores=[])
                namespace = {"log_softmax": F.log_softmax}
                exec(compile(module, str(DPO), "exec"), namespace)
                return namespace["sequence_logp"]
    raise AssertionError("No sequence_logp in current chapter")


class PrimaryReviewExamples(unittest.TestCase):
    def test_actual_sequence_logp_shift_mask_sum_and_backward(self):
        fn = page_sequence_logp()
        torch.manual_seed(17)
        logits = torch.randn(2, 5, 7, dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([[0, 1, 2, 3, 4], [0, 5, 6, 1, 0]])
        mask = torch.tensor([[0, 0, 1, 1, 1], [0, 0, 1, 1, 0]])
        actual = fn(logits, labels, mask)
        logp = F.log_softmax(logits, dim=-1)
        expected = torch.stack([
            sum(logp[b, t - 1, labels[b, t]] for t in range(1, 5) if mask[b, t])
            for b in range(2)
        ])
        torch.testing.assert_close(actual, expected)
        extension = torch.cat([logits, torch.zeros(2, 2, 7, dtype=logits.dtype)], dim=1)
        padded_labels = F.pad(labels, (0, 2))
        padded_mask = F.pad(mask, (0, 2))
        torch.testing.assert_close(fn(extension, padded_labels, padded_mask), actual)
        (-actual.mean()).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(float(logits.grad[:, 0].abs().sum()), 0.0)
        self.assertEqual(float(logits.grad[:, -1].abs().sum()), 0.0)

    def test_dpo_beta_coefficient_and_wrong_pair_saturation(self):
        for beta in (0.1, 0.5, 1.0, 5.0):
            for margin in (-3.0, 0.0, 0.8, 3.0):
                m = torch.tensor(margin, dtype=torch.float64, requires_grad=True)
                loss = -F.logsigmoid(beta * m)
                loss.backward()
                expected = -beta / (1 + math.exp(beta * margin))
                self.assertAlmostEqual(float(m.grad), expected, places=12)
                if margin == 0:
                    self.assertAlmostEqual(float(loss.detach()), math.log(2), places=12)
                    self.assertAlmostEqual(float(m.grad), -beta / 2, places=12)
        self.assertAlmostEqual(float(-F.logsigmoid(torch.tensor(0.4, dtype=torch.float64))), 0.5130152523999526)
        # A large negative margin is a badly ordered pair, not a vanishing gradient.
        bad_margin = torch.tensor(-100., requires_grad=True)
        (-F.logsigmoid(bad_margin)).backward()
        self.assertAlmostEqual(float(bad_margin.grad), -1.0)

    def test_euclidean_gradient_descent_requires_nonzero_gradient(self):
        def f(x, y):
            return x*x + 3*x*y + y*y
        x, y = 1., 2.
        gx, gy = 2*x + 3*y, 3*x + 2*y
        self.assertEqual((gx, gy), (8., 7.))
        self.assertLess(f(x - .01*gx, y - .01*gy), f(x, y))
        self.assertEqual(f(0., 0.), f(0. - .01*0., 0. - .01*0.))


if __name__ == "__main__":
    unittest.main(verbosity=2)
