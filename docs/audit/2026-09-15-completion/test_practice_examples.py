"""Execute small CPU examples from the actual practice Markdown, not copies."""
from pathlib import Path
import re
import os
import subprocess
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[3]
PRACTICE = ROOT / "06 Практика"


def example(filename, marker):
    text = (PRACTICE / filename).read_text()
    blocks = re.findall(r"```python\n(.*?)\n```", text, re.S)
    matches = [block for block in blocks if marker in block]
    if len(matches) != 1:
        raise AssertionError(f"Expected one runnable {marker} example, got {len(matches)}")
    namespace = {}
    exec(compile(matches[0], str(PRACTICE / filename), "exec"), namespace)
    return namespace


class PracticeExamples(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("BOOKVAR_LEAN_BIN"), "set BOOKVAR_LEAN_BIN to the pinned Lean 4.19.0 bin directory")
    def test_lean_error_reset_and_kernel_acceptance(self):
        text = (PRACTICE / "26 Поиск доказательства в Lean с verifier.md").read_text()
        blocks = re.findall(r"```lean\n(.*?)\n```", text, re.S)
        cases = {}
        for name in ["bad", "good"]:
            matches = [b for b in blocks if f"-- bookvar: lean-{name}" in b]
            self.assertEqual(len(matches), 1, f"one {name} fixture must be present")
            cases[name] = matches[0]
        self.assertEqual(cases["bad"].split("theorem ")[1].split(" := by")[0],
                         cases["good"].split("theorem ")[1].split(" := by")[0])
        env = dict(os.environ, PATH=os.environ["BOOKVAR_LEAN_BIN"] + os.pathsep + os.environ["PATH"])
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            toolchain = re.findall(r"```text\n(leanprover/lean4:v4\.19\.0)\n```", text)
            config = re.findall(r'```toml\n(name = "bookvar_lean_smoke".*?)\n```', text, re.S)
            self.assertEqual(len(toolchain), 1)
            self.assertEqual(len(config), 1)
            (folder / "lean-toolchain").write_text(toolchain[0] + "\n")
            (folder / "lakefile.toml").write_text(config[0] + "\n")
            for name, body in cases.items():
                (folder / (name.title() + ".lean")).write_text(body + "\n")
            results = [subprocess.run(["lake", "env", "lean", name + ".lean"],
                       cwd=folder, env=env, text=True, capture_output=True, timeout=30)
                       for name in ["Bad", "Good", "Good"]]
            self.assertNotEqual(results[0].returncode, 0)
            self.assertIn("missing_lemma", results[0].stdout + results[0].stderr)
            for result in results[1:]:
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertNotIn("sorry", result.stdout + result.stderr)
            self.assertEqual(results[1].stdout, results[2].stdout)

    def test_symbolic_starter_selects_on_train_and_checks_ast(self):
        ns = example("27 Проверяемый научный поиск на символьной регрессии.md", "# bookvar: symbolic-smoke")
        self.assertEqual(ns["best_name"], "expanded_quadratic")
        self.assertEqual(ns["nmse"](ns["best"], ns["test"]), 0.0)
        self.assertEqual(ns["size"](ns["best"]), 9)
        self.assertEqual(ns["invalid_cases_checked"], 6)

    def test_rlvr_mask_reward_gradient_and_digest(self):
        first = example("24 Post-training и RLVR для математического reasoning.md", "# bookvar: rlvr-smoke")
        second = example("24 Post-training и RLVR для математического reasoning.md", "# bookvar: rlvr-smoke")
        self.assertEqual(first["digest"], second["digest"])
        self.assertEqual(first["rewards"].tolist(), [1., 0.])
        self.assertEqual(first["mask"].sum().item(), 4)
        self.assertEqual(torch.count_nonzero(first["logits"].grad[:, [0, 3]]).item(), 0)

    def test_scaling_fixture_fits_without_heldout_leakage(self):
        ns = example("22 Провести scaling-law campaign.md", "# bookvar: scaling-fixture")
        self.assertEqual(int(ns["train"].sum()), 52)
        self.assertEqual(int((~ns["train"]).sum()), 13)
        self.assertLess(ns["heldout_mae"], .05)
        self.assertTrue(ns["fit"].success)
        self.assertEqual(len(ns["fig"].axes), 3)

    def test_checkpoint_preserves_dropout_gradient(self):
        ns = example("10 Измерить checkpointing и offload.md", "# bookvar: checkpoint-rng")
        torch.testing.assert_close(ns["plain_loss"], ns["saved_loss"])
        torch.testing.assert_close(ns["plain_grad"], ns["saved_grad"])
        self.assertFalse(torch.equal(ns["plain_grad"], ns["changed_grad"]))

    def test_fp16_subnormal_and_gradient_scaling(self):
        ns = example("06a Проверить mixed precision и loss scaling.md", "# bookvar: fp16-underflow")
        self.assertEqual(ns["rounded"].tolist(), [2**-14, 2**-20, 2**-24, 0.0])
        self.assertEqual(ns["direct"].item(), 0.0)
        self.assertEqual(ns["recovered"].item(), 2**-26)

    def test_smoothed_bigram_objective_matches_counts(self):
        ns = example("01 Собрать micrograd и bigram LM.md", "# bookvar: bigram-pseudocounts")
        torch.testing.assert_close(ns["learned"], ns["expected"], atol=1e-5, rtol=1e-5)
        self.assertLess(ns["final_loss"], ns["initial_loss"])
        self.assertGreater(torch.max(torch.abs(ns["expected"] - ns["mle"])).item(), .05)

    def test_causal_attention_shapes_sdpa_and_future_independence(self):
        ns = example("02 Реализовать causal self-attention.md", "# bookvar: causal-attention")
        self.assertEqual(tuple(ns["output"].shape), (2, 3, 5, 4))
        self.assertEqual(tuple(ns["merged"].shape), (2, 5, 12))
        torch.testing.assert_close(ns["output"], ns["reference"], atol=1e-10, rtol=1e-10)
        self.assertEqual(torch.count_nonzero(ns["weights"].triu(1)).item(), 0)


if __name__ == "__main__":
    unittest.main()
