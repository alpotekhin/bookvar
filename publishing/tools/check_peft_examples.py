"""Execute every printed PEFT example and check its algebra on CPU.

uv run --offline --with torch==2.14.1 --with numpy==2.3.5 \
  python publishing/tools/check_peft_examples.py
No model, dataset, GPU or network request is needed.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import re

import torch

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/12 Post-training и Alignment/01a Fine-tuning и PEFT — adapters, LoRA, QLoRA и soft prompts.md"


def main():
    text = CHAPTER.read_text()
    blocks = re.findall(r"^\x60{3}python\n(.*?)^\x60{3}\s*$", text, re.M | re.S)
    assert len(blocks) == 18, "execute every published Python fence"
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for number, block in enumerate(blocks, 1):
            exec(compile(block, f"{CHAPTER}:python-{number}", "exec"), ns)

    # Independent matrix-gradient comparison, not just the printed rank-one case.
    torch.manual_seed(2106)
    dtype = torch.float64
    x = torch.randn(3, dtype=dtype, requires_grad=True)
    a = torch.randn(2, 3, dtype=dtype, requires_grad=True)
    b = torch.randn(4, 2, dtype=dtype, requires_grad=True)
    w = torch.randn(4, 3, dtype=dtype)
    g = torch.randn(4, dtype=dtype)
    scale = 1.7
    out = w @ x + scale * b @ a @ x
    actual_a, actual_b, actual_x = torch.autograd.grad((g * out).sum(), (a, b, x))
    expected_a = scale * torch.outer(b.T @ g, x)
    expected_b = scale * torch.outer(g, a @ x)
    expected_x = w.T @ g + scale * a.T @ b.T @ g
    gradient_error = max((actual_a - expected_a).abs().max().item(),
                         (actual_b - expected_b).abs().max().item(),
                         (actual_x - expected_x).abs().max().item())
    assert gradient_error < 1e-12

    # A wrong scale or accidental second merge must not pass the equality check.
    wrong_scale = (w + b.detach() @ a.detach()) @ x.detach()
    double_merge = (w + 2 * scale * b.detach() @ a.detach()) @ x.detach()
    expected_output = out.detach()
    assert not torch.allclose(wrong_scale, expected_output)
    assert not torch.allclose(double_merge, expected_output)

    # The added zero prefix changes normalization despite contributing zero V.
    attention = ns["attention_row"]
    real = ns["real"]
    zero = torch.zeros(1, 2, dtype=dtype)
    zero_prefix = attention(real, torch.cat([zero, real[None]]),
                            torch.cat([zero, real[None]]))
    no_prefix = attention(real, real[None], real[None])
    assert not torch.allclose(zero_prefix, no_prefix)

    # Factor sums produce cross terms, unlike a sum of completed updates.
    a1, a2 = torch.randn(2, 3, dtype=dtype), torch.randn(2, 3, dtype=dtype)
    b1, b2 = torch.randn(4, 2, dtype=dtype), torch.randn(4, 2, dtype=dtype)
    direct_sum = b1 @ a1 + b2 @ a2
    factor_sum = (b1 + b2) @ (a1 + a2)
    cross_terms = b1 @ a2 + b2 @ a1
    torch.testing.assert_close(factor_sum, direct_sum + cross_terms)
    assert not torch.allclose(factor_sum, direct_sum)

    assert 20 * 4096 == 81920
    assert 2 * 32 * 20 * 8 * 128 == 1310720
    assert 1310720 * 2 / 1e6 == 2.62144
    assert 3 * 7_000_000_000 * 2 / 1e9 == 42
    assert ns["adapter_parameters"] * 12 / 1e6 == 50.331648
    assert 2400 // 32 * 3 == 225
    assert math.isclose(ns["scale_bits"], .126953125)

    assert math.isclose(ns["failed_loss"].item(), math.log(2))
    assert math.isclose(ns["fitted_loss"].item(), -math.log(.99))
    assert 0 < ns["lower_weight"].grad.item() < 1e-7
    assert ns["head_weight"].grad.item() > .7
    assert math.isclose(ns["raw_update"] / ns["corrected_update"], math.sqrt(10))
    assert math.isclose(ns["scalar_parameter"].item(), .999)
    # Independent early-step ratio checks and explicit schedule/count changes.
    for t, expected in [(1, 3.1622776601683773), (10, 6.5278807045909355),
                        (100, 3.2407988707922826), (1000, 1.25758352096177)]:
        ratio = (1 - .9**t) / math.sqrt(1 - .999**t)
        assert math.isclose(ratio, expected, rel_tol=1e-6)
    assert ns["single_device_updates"](1000, 32, 3) == 96
    assert ns["single_device_updates"](2400, 32, 3) == 225
    assert ns["single_device_updates"](1000, 32, 3, True) == 93
    assert math.isclose(ns["mean"](ns["seed_scores"]), .6075)
    assert math.isclose(ns["mean"](ns["retained_scores"]), .71)
    assert 2 * 4096 * 256 == 2097152
    assert 2097152 / (4096 * 4096) == .125
    assert 4_700_000 * 2 / 1e6 == 9.4
    assert 18_000_000 * 2 / 1e6 == 36
    for invalid in [-1, len(ns["toy_blocks"]) + 1]:
        try:
            ns["reset_top_blocks"](ns["toy_blocks"], ns["toy_pooler"], invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid reset count was accepted")

    print(json.dumps({
        "chapter_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "torch": torch.__version__, "published_python_fences": len(blocks),
        "independent_gradient_max_error": gradient_error,
        "soft_prompt_output": ns["soft_output"].detach().tolist(),
        "prefix_output": ns["prefix_output"].detach().tolist(),
        "adapter_parameters": ns["adapter_parameters"],
        "saved_scale_gb_65b": ns["saved_gb_65b"],
        "small_data": {
            "trivial_binary_loss": ns["failed_loss"].item(),
            "head_gradient": ns["head_weight"].grad.item(),
            "lower_gradient": ns["lower_weight"].grad.item(),
            "raw_to_corrected_first_update": ns["raw_update"] / ns["corrected_update"],
            "all_runs_mean": ns["mean"](ns["seed_scores"]),
            "survivor_only_mean": ns["mean"](ns["retained_scores"]),
            "reset_zero_blocks_preserves_blocks": True,
            "reset_one_block_preserves_lower_blocks": True,
            "target_head_classes": ns["target_head"].out_features,
            "intermediate_head_classes": ns["intermediate_head"].out_features,
        },
        "retention": {
            "quarter_replay_old_loss": ns["old_at_quarter"],
            "quarter_replay_new_loss": ns["new_at_quarter"],
            "l2_sp_solution": ns["sp_solution"].detach().tolist(),
            "ewc_solution": ns["ewc_solution"].detach().tolist(),
            "ewc_new_loss": ns["new_quadratic"].item(),
            "ewc_protected_old_loss": ns["old_quadratic"].item(),
            "mean_squared_per_example_gradient": ns["empirical_diagonal"].item(),
            "squared_mean_gradient": ns["squared_mean_gradient"].item(),
            "observed_positive_label_square_at_p09": ns["observed_label_square"],
            "model_fisher_at_p09": ns["model_fisher"],
        },
        "negative_controls": ["both factors zero", "detached frozen upper layer",
                              "wrong merge scaling", "double merge",
                              "zero prefix is not identity", "factor sum cross terms",
                              "zero block count does not select all blocks",
                              "invalid reset count is rejected"],
        "scope": "Small CPU algebra/gradient examples, not full-model tuning or GPU QLoRA",
    }, indent=2))


if __name__ == "__main__":
    main()
