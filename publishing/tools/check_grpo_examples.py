"""Execute every published GRPO fence, then independent CPU checks.

uv run --offline --with torch==2.14.1 --with transformers==4.57.1 \
  --with numpy==2.3.5 --with trl==0.24.0 \
  python publishing/tools/check_grpo_examples.py
No pretrained weights, tokenizer, dataset, trainer, GPU or benchmark.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import io
import itertools
import json
import math
from pathlib import Path
import re

import torch

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/12 Post-training и Alignment/07 GRPO и DeepSeek-R1.md"


def rejects(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("invalid input accepted")


def main():
    text = CHAPTER.read_text()
    blocks = re.findall(r"^```python\n(.*?)^```\s*$", text, re.M | re.S)
    assert len(blocks) == 11, "execute all actual chapter fences"
    ns, stdout = {}, io.StringIO()
    with contextlib.redirect_stdout(stdout):
        for i, code in enumerate(blocks):
            exec(compile(code, f"{CHAPTER}:python-{i+1}", "exec"), ns)
    group, loss = ns["group_advantages"], ns["clipped_token_loss"]
    f64 = torch.float64

    # Exact KL expectation is not the fixed-sample-weight semigradient.
    expected_kl = .6*math.log(1.2)+.4*math.log(.8)
    assert math.isclose(ns["exact_kl"].item(), expected_kl, abs_tol=1e-14)
    assert math.isclose(ns["sampled_kl"].item(), expected_kl, abs_tol=1e-14)
    assert math.isclose(ns["exact_kl_gradient"].item(), .24*math.log(1.5), abs_tol=1e-14)
    assert math.isclose(ns["sampled_kl_gradient"].item(), .1, abs_tol=1e-14)

    # Group arithmetic, population/sample std and stop-gradient behavior.
    r = torch.tensor([1.1, 0., 1.1, .1], dtype=f64, requires_grad=True)
    a = group(r, eps=1e-12)
    expected_std = math.sqrt(sum((v-.575)**2 for v in r.tolist())/4)
    torch.testing.assert_close(a, (r.detach()-.575)/(expected_std+1e-12))
    assert not a.requires_grad
    torch.testing.assert_close(group(r+100, eps=1e-12), a)
    torch.testing.assert_close(group(10*r, eps=1e-12), a)
    torch.testing.assert_close(group(r, scale=False), r.detach()-.575)
    assert torch.equal(group(torch.ones(2, 4)), torch.zeros(2, 4))
    sample = (r.detach()-r.detach().mean())/r.detach().std(correction=1)
    torch.testing.assert_close(sample, a*math.sqrt(3/4), atol=1e-10, rtol=1e-10)
    for bad in ([], [1.], [[1.]], torch.empty(0, 4), torch.ones(1, 2, 3),
                [1., float("nan")], [float("inf"), 0.], [1+1j, 0.]):
        rejects(group, bad)
    for eps in (0, -1, float("inf"), float("nan")):
        rejects(group, [1., 0.], eps=eps)
    torch.testing.assert_close(ns["group_a"][0], group(r))

    # The two signs in all three clipping regions; expected scalar derivatives.
    clipping_cases = 0
    for advantage in (1., -1.):
        for ratio in (.5, 1., 1.5):
            old = torch.tensor([[-2.]], dtype=f64, requires_grad=True)
            new = torch.tensor([[-2.+math.log(ratio)]], dtype=f64, requires_grad=True)
            adv = torch.tensor([advantage], dtype=f64, requires_grad=True)
            result = loss(new, old, adv, torch.ones(1, 1, dtype=torch.bool))
            result.backward()
            clipped = (advantage > 0 and ratio > 1.2) or (advantage < 0 and ratio < .8)
            expected_gradient = 0. if clipped else -ratio*advantage
            assert math.isclose(new.grad.item(), expected_gradient, abs_tol=1e-14)
            assert old.grad is None and adv.grad is None
            clipping_cases += 1
    assert math.isclose(ns["clip_example_loss"].item(), .0625, abs_tol=1e-14)
    torch.testing.assert_close(ns["new_token_lp"].grad,
        torch.tensor([[0., -.175], [0., .3375]], dtype=f64))

    mask = torch.tensor([[True, False, True], [True, True, False]])
    old = torch.full((2, 3), -2., dtype=f64)
    new = (old+torch.tensor([[1.1, .9, .9], [1.05, .95, 1.1]], dtype=f64).log()).requires_grad_()
    advantages = torch.tensor([.4, -.7], dtype=f64)
    assert torch.autograd.gradcheck(lambda z: loss(z, old, advantages, mask), (new,))
    clean_loss = loss(new, old, advantages, mask)
    clean_grad, = torch.autograd.grad(clean_loss, new)
    # Padding removed BEFORE exp: NaN, +/-infinity and enormous values are inert.
    for padding in (float("nan"), float("inf"), -float("inf"), 1e30, -1e30):
        padded_new = new.detach().masked_fill(~mask, padding).requires_grad_()
        padded_old = old.masked_fill(~mask, -padding)
        actual = loss(padded_new, padded_old, advantages, mask)
        gradient, = torch.autograd.grad(actual, padded_new)
        torch.testing.assert_close(actual, clean_loss)
        torch.testing.assert_close(gradient, clean_grad)
        assert torch.equal(gradient[~mask], torch.zeros_like(gradient[~mask]))
    rejects(loss, new, old, advantages, mask.float())
    rejects(loss, new, old, advantages[:, None], mask)
    rejects(loss, new, old[:, :2], advantages, mask)
    rejects(loss, new, old, advantages, torch.zeros_like(mask))
    rejects(loss, new[:, 0], old[:, 0], advantages, mask[:, 0])
    rejects(loss, new.half(), old.half(), advantages, mask)
    rejects(loss, new, old.to(torch.complex128), advantages, mask)
    rejects(loss, new, old, [1j, 0j], mask)
    for eps in (0, 1, -1, float("nan"), float("inf")):
        rejects(loss, new, old, advantages, mask, clip_eps=eps)
    for bad_new, bad_old, bad_a in (
        (new.detach().masked_fill(mask, float("nan")), old, advantages),
        (new, old.masked_fill(mask, float("inf")), advantages),
        (new, old, [float("nan"), 1.]),
        (new.detach().masked_fill(mask, 1e6), old, advantages),
    ):
        rejects(loss, bad_new, bad_old, bad_a, mask)

    # EOS is included by the caller's response mask, PAD excluded, lengths unequal.
    token_ids = torch.tensor([[10, 2, 0], [11, 12, 2]])  # EOS=2, PAD=0
    response_mask = token_ids != 0
    lp = torch.full((2, 3), -2., dtype=f64, requires_grad=True)
    objective = loss(lp, lp.detach(), [1., -1.], response_mask)
    grad, = torch.autograd.grad(objective, lp)
    torch.testing.assert_close(grad, torch.tensor([[-.25, -.25, 0.], [1/6, 1/6, 1/6]], dtype=f64))

    # Process reward: normalize all four steps jointly, then sum FUTURE steps.
    norm = torch.tensor([[3, -1], [3, -5]], dtype=f64)/math.sqrt(11)
    torch.testing.assert_close(ns["normalized_steps"], norm, atol=2e-8, rtol=2e-8)
    expected_steps = torch.stack((norm[:, 0]+norm[:, 1], norm[:, 1]), dim=1)
    torch.testing.assert_close(ns["step_advantages"], expected_steps, atol=2e-8, rtol=2e-8)
    torch.testing.assert_close(ns["process_token_advantages"], expected_steps.repeat_interleave(2, -1), atol=2e-8, rtol=2e-8)

    # Actual microbatch fence and unequal microbatch partitions have one denominator.
    assert math.isclose(ns["sequence_mean"].item(), 1.5)
    assert math.isclose(ns["token_mean"].item(), 42/22)
    assert math.isclose(ns["constant_mean"].item(), 1.05)
    torch.testing.assert_close(ns["full_gradient"], ns["micro_gradient"])
    assert ns["wrong_gradient"][0].item() == .25
    assert ns["wrong_gradient"][2].item() == .025
    for partition in ((1, 21), (2, 7, 13), (5, 5, 5, 7)):
        values = torch.arange(22., dtype=f64, requires_grad=True)
        full, = torch.autograd.grad(values.mean(), values)
        start = 0
        for size in partition:
            (values[start:start+size].sum()/22).backward()
            start += size
        torch.testing.assert_close(values.grad, full)

    # Exhaustive finite-group expectations, independent closed-form references.
    enumeration_cases = 0
    for size in (2, 3, 4, 6, 8):
        for p in (.01, .2, .5, .9, .99):
            reinforce = baseline = loo = theoretical = practical = 0.
            for actions in itertools.product((0., 1.), repeat=size):
                k = sum(actions)
                probability = p**k*(1-p)**(size-k)
                scores = [x-p for x in actions]
                mean = k/size
                reinforce += probability*sum(x*s for x, s in zip(actions, scores))/size
                baseline += probability*sum((x-mean)*s for x, s in zip(actions, scores))/size
                loo += probability*sum((x-(k-x)/(size-1))*s for x, s in zip(actions, scores))/size
                update = (sum(x*s for x, s in zip(actions, scores))/k if k else 0.)-sum(scores)/size
                theoretical += probability*update
                practical += probability*update if k else 0.
                enumeration_cases += 1
            for got, expected in ((reinforce, p*(1-p)), (baseline, (size-1)/size*p*(1-p)),
                                  (loo, p*(1-p)), (theoretical, (1-(1-p)**size)*(1-p)),
                                  (practical, (1-p)-(1-p)**size)):
                assert math.isclose(got, expected, abs_tol=2e-14), (size, p, got, expected)
    assert math.isclose(ns["maxrl_theoretical"], .288)
    assert math.isclose(ns["maxrl_practical"], .16)
    assert ns["baseline_expectations"] == {"reinforce": .16000000000000003,
        "group_mean": .08000000000000002, "leave_one_out": .16000000000000003}

    # Full trajectory correction and token correction optimize different targets.
    for name, expected in (("unweighted", .032), ("token", .096), ("sequence", .288)):
        assert math.isclose(ns["offpolicy_gradients"][name].item(), expected, abs_tol=1e-14)
    assert math.isclose(ns["offpolicy_exact"].item(), .288, abs_tol=1e-14)
    assert math.isclose(ns["offpolicy_surrogate"].item(), .096, abs_tol=1e-14)

    # GSPO's rooted ratio and Kimi's quadratic penalty are different surrogates.
    assert math.isclose(ns["gspo_ratio"].item(), math.sqrt(.9))
    torch.testing.assert_close(ns["grpo_comparison_gradient"], torch.tensor([[0., -.3]], dtype=f64))
    torch.testing.assert_close(ns["gspo_gradient"], torch.full((1, 2), -math.sqrt(.9)/2, dtype=f64))
    torch.testing.assert_close(ns["kimi_gradient"], torch.tensor([-.2, .2], dtype=f64))
    a, tau = ns["kimi_a"], ns["kimi_tau"]
    lp = ns["kimi_new_lp"].detach().requires_grad_()
    residual_loss = ((a-tau*(lp-ns["kimi_old_lp"])).square()/(2*tau)).mean()
    equivalent_gradient, = torch.autograd.grad(residual_loss, lp)
    torch.testing.assert_close(equivalent_gradient, ns["kimi_gradient"])

    assert importlib.metadata.version("trl") == "0.24.0"
    config = ns["args"]
    assert config.num_generations == config.per_device_train_batch_size*config.gradient_accumulation_steps == 8
    assert config.loss_type == "grpo" and config.scale_rewards == "group"
    assert config.beta == .001 and config.epsilon == .2
    assert config.use_cpu and not config.bf16 and not config.fp16
    assert config.max_completion_length == 512 and config.report_to == []

    print(json.dumps({"python_fences": len(blocks), "device": "cpu", "torch": torch.__version__,
        "trl": importlib.metadata.version("trl"), "clipping_cases": clipping_cases,
        "finite_group_cases": enumeration_cases, "example_stdout": stdout.getvalue().splitlines(),
        "chapter_sha256": hashlib.sha256(text.encode()).hexdigest()}, indent=2))
    print("PASS actual GRPO fences; scalar/tensor math and config only, not model training")


if __name__ == "__main__":
    main()
