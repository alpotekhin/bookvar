"""CPU arithmetic checks for the course additions; no LM training or GPU run."""
import math
import unittest

import torch


class CourseAttentionAndMTP(unittest.TestCase):
    def test_mtp_valid_targets_and_normalization(self):
        tokens = list(range(1, 6))
        pairs = [[(tokens[t], tokens[t + k]) for t in range(len(tokens) - k)]
                 for k in (1, 2, 3)]
        self.assertEqual([len(p) for p in pairs], [4, 3, 2])
        self.assertEqual(pairs[2], [(1, 4), (2, 5)])
        # A per-head mean gives heads equal weight; a joint-token mean does not.
        head_losses = [torch.full((n,), float(k)) for k, n in enumerate((4, 3, 2), 1)]
        self.assertAlmostEqual(float(torch.stack([x.mean() for x in head_losses]).mean()), 2.)
        self.assertAlmostEqual(float(torch.cat(head_losses).double().mean()), 16 / 9)

    def test_local_receptive_field_and_hybrid_cache(self):
        size, width = 20, 4
        influence = [{t} for t in range(size)]
        for layer in range(1, 4):
            influence = [set().union(*(influence[j] for j in range(max(0, t-width+1), t+1)))
                         for t in range(size)]
            self.assertEqual(len(influence[-1]), 1 + layer*(width-1))
            self.assertTrue(all(all(j <= t for j in positions) for t, positions in enumerate(influence)))
        self.assertEqual(24*min(32768, 4096) + 8*32768, 360448)

    def test_cla_cache_and_distinct_queries(self):
        def cache(layers, sharing):
            return 2 * 1 * 32768 * math.ceil(layers/sharing) * 8 * 128 * 2
        self.assertEqual(cache(32, 1), 4*1024**3)
        self.assertEqual(cache(32, 2), 2*1024**3)
        self.assertEqual(math.ceil(10/3), 4)
        k = torch.eye(2, dtype=torch.float64)
        v = torch.tensor([[1., 0.], [0., 2.]], dtype=torch.float64)
        q1 = torch.tensor([[2., 0.]], dtype=torch.float64)
        q2 = torch.tensor([[0., 2.]], dtype=torch.float64)
        def attend(q):
            return torch.softmax(q @ k.T / math.sqrt(2), dim=-1) @ v
        self.assertFalse(torch.allclose(attend(q1), attend(q2)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
