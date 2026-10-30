"""Execute every published reward-modeling Python fence on a tiny CPU Llama.

uv run --offline --with torch==2.14.1 --with numpy==2.3.5 \
  --with transformers==4.57.1 python publishing/tools/check_reward_modeling_examples.py
No Hub downloads or pretrained weights. This is not a performance benchmark.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import re

import torch
import transformers

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/12 Post-training и Alignment/03 Reward modeling.md"


def main():
    text = CHAPTER.read_text()
    blocks = re.findall(r"^\x60{3}python\n(.*?)^\x60{3}\s*$", text, re.M | re.S)
    assert len(blocks) == 6, "execute every published Python fence"
    ns, output = {}, io.StringIO()
    with contextlib.redirect_stdout(output):
        for i, code in enumerate(blocks):
            exec(compile(code, f"{CHAPTER}:python-{i+1}", "exec"), ns)
    score, last, pl = [ns[k] for k in ("score", "last_token_index", "plackett_luce_loss")]
    backbone, head, optimizer = [ns[k] for k in ("backbone", "head", "optimizer")]
    F = torch.nn.functional
    assert optimizer.state and all(s["step"].item() == 1 for s in optimizer.state.values())
    assert head.weight.grad.norm().item() > 0
    backbone_grad = sum(p.grad.norm().item() for p in backbone.parameters() if p.grad is not None)
    assert backbone_grad > 0
    with torch.no_grad():
        new_loss = -F.logsigmoid(
            score(backbone, head, ns["chosen_ids"], ns["chosen_mask"]) -
            score(backbone, head, ns["rejected_ids"], ns["rejected_mask"])
        ).mean()
    assert new_loss < ns["loss"], "one small step reduces this toy batch loss"
    assert last(torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 0, 0]])).tolist() == [4, 2]
    for invalid in [torch.empty((0, 2)), torch.empty((2, 0)), torch.ones(3),
                    torch.zeros((1, 3)), torch.tensor([[1, 2, 0]])]:
        try:
            last(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid mask accepted: {invalid}")
    try:
        score(backbone, head, torch.ones((2, 3), dtype=torch.long), torch.ones((1, 3)))
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch accepted")

    # Every actual sequence: no padding, right padding and left padding.
    max_padding_error = 0.0
    for ids, mask in [(ns["chosen_ids"], ns["chosen_mask"]),
                      (ns["rejected_ids"], ns["rejected_mask"])]:
        for row, am in zip(ids, mask):
            real = row[am.bool()][None, :]
            one = torch.ones_like(real)
            with torch.no_grad():
                expected = score(backbone, head, real, one)
                for count in (1, 3, 5):
                    pad = torch.full((1, count), 2, dtype=torch.long)
                    zero = torch.zeros_like(pad)
                    for cat_ids, cat_mask in [
                        (torch.cat((pad, real), 1), torch.cat((zero, one), 1)),
                        (torch.cat((real, pad), 1), torch.cat((one, zero), 1)),
                    ]:
                        actual = score(backbone, head, cat_ids, cat_mask)
                        max_padding_error = max(max_padding_error, (actual-expected).abs().item())
                        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    delta = torch.tensor([-1000., -2., 0., 2., 1000.], dtype=torch.float64,
                         requires_grad=True)
    stable = -F.logsigmoid(delta)
    assert torch.isfinite(stable).all()
    torch.testing.assert_close(stable, F.softplus(-delta))
    gradient = torch.autograd.grad(stable.sum(), delta)[0]
    torch.testing.assert_close(gradient, delta.sigmoid()-1)
    q = torch.tensor(2/3, dtype=torch.float64)
    optimum = torch.tensor(math.log(2), dtype=torch.float64, requires_grad=True)
    soft_loss = F.binary_cross_entropy_with_logits(optimum, q)
    assert torch.autograd.grad(soft_loss, optimum)[0].abs() < 1e-12
    r = torch.tensor([1.4, .2], dtype=torch.float64)
    assert torch.allclose((r[0]-r[1]).sigmoid(), ((r[0]+100)-(r[1]+100)).sigmoid())
    assert not torch.allclose((r[0]-r[1]).sigmoid(), (10*(r[0]-r[1])).sigmoid())

    torch.testing.assert_close(ns["pl_loss"].neg().exp(), torch.tensor(8/21, dtype=torch.float64))
    torch.testing.assert_close(ns["pair_losses"].sum().neg().exp(), torch.tensor(16/45, dtype=torch.float64))
    torch.testing.assert_close(pl(r), -F.logsigmoid(r[0]-r[1]))
    torch.testing.assert_close(pl(torch.zeros(3)).neg().exp(), torch.tensor(1/6))
    all_orders = sum(pl(r[list(order)]).neg().exp() for order in itertools.permutations(range(2)))
    torch.testing.assert_close(all_orders, torch.tensor(1., dtype=torch.float64))
    scores = ns["ranked_scores"]
    total = sum(pl(scores[list(order)]).neg().exp() for order in itertools.permutations(range(3)))
    torch.testing.assert_close(total, torch.tensor(1., dtype=torch.float64))
    torch.autograd.gradcheck(pl, (scores.detach().clone().requires_grad_(),))
    torch.testing.assert_close(pl(scores + 100), pl(scores))
    for invalid in [torch.zeros(1), torch.zeros((2, 2))]:
        try:
            pl(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid ranking accepted")
    torch.testing.assert_close(ns["prm_product"], torch.tensor([.81, .95**5], dtype=torch.float64))
    torch.testing.assert_close(ns["prm_minimum"], torch.tensor([.9, .95], dtype=torch.float64))
    torch.testing.assert_close(ns["pair_accuracy"], torch.tensor([.8, .8], dtype=torch.float64))
    torch.testing.assert_close(ns["calibration_loss"],
        torch.tensor([-.8*math.log(.8)-.2*math.log(.2),
                      -.8*math.log(.99)-.2*math.log(.01)], dtype=torch.float64))
    assert math.isclose(1-.99**64, .47440351247443794)

    print(json.dumps({
        "published_python_fences": len(blocks), "torch": torch.__version__,
        "transformers": transformers.__version__, "device": "cpu",
        "model_parameters": sum(p.numel() for p in backbone.parameters()) + head.weight.numel(),
        "loss_before": ns["loss"].item(), "loss_after": new_loss.item(),
        "padding_comparisons": 24, "max_padding_error": max_padding_error,
        "backbone_gradient_norm_sum": backbone_grad,
        "pl_probability": ns["pl_loss"].neg().exp().item(),
        "pair_probability_product": ns["pair_losses"].sum().neg().exp().item(),
        "calibration_loss": ns["calibration_loss"].tolist(),
        "example_stdout": output.getvalue().splitlines(),
        "chapter_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }, indent=2))
    print("PASS actual reward-modeling examples; CPU only, no model downloads")


if __name__ == "__main__":
    main()
