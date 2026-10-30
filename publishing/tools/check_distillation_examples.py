"""Execute the chapter's actual Python fences, then independent CPU checks.

uv run --offline --with torch==2.14.1 --with transformers==4.57.1 \
  --with numpy==2.3.5 python publishing/tools/check_distillation_examples.py
No pretrained weights, tokenizer, dataset, GPU or full Trainer run.
"""
from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import re

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/12 Post-training и Alignment/08 Reasoning distillation.md"
SOURCE = ROOT / "05 Источники/Originals/distillation-2026-10-25/gkd_trainer.py"


def rejects(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("invalid input accepted")


def main():
    torch.set_num_threads(1)
    text = CHAPTER.read_text()
    blocks = re.findall(r"^```python\n(.*?)^```\s*$", text, re.M | re.S)
    assert len(blocks) == 8, "execute all actual chapter fences"
    ns, stdout = {}, io.StringIO()
    with contextlib.redirect_stdout(stdout):
        for i, code in enumerate(blocks):
            exec(compile(code, f"{CHAPTER}:python-{i+1}", "exec"), ns)
    kd, seq = ns["kd_loss"], ns["sequence_distillation_loss"]
    f64 = torch.float64

    # The consolidated feature-KD example includes the projection's gradient.
    assert ns["feature_loss"].item() == 2.5
    torch.testing.assert_close(ns["feature_projection"].grad, torch.tensor([[-2.], [4.]], dtype=f64))
    torch.testing.assert_close(ns["feature_student"].grad, torch.tensor([1.], dtype=f64))
    assert ns["feature_teacher"].grad is None

    assert math.isclose(ns["seq_full_ce"].item(), -.6*math.log(.3)-.4*math.log(.7), abs_tol=1e-14)
    assert math.isclose(ns["seq_mode_ce"].item(), -math.log(.3), abs_tol=1e-14)
    torch.testing.assert_close(ns["seq_full_gradient"], torch.tensor([-.3, .3], dtype=f64))
    torch.testing.assert_close(ns["seq_mode_gradient"], torch.tensor([-.7, .7], dtype=f64))
    torch.testing.assert_close(ns["mixed_gradient"], torch.tensor([[[-.35, .3, .05]]], dtype=f64))

    # Causal shift, unequal target counts, EOS included and PAD excluded.
    generator = torch.Generator().manual_seed(107)
    logits = torch.randn(2, 5, 7, generator=generator, dtype=f64, requires_grad=True)
    ids = torch.tensor([[6, 5, 3, 2, 0], [6, 4, 3, 1, 2]])
    mask = torch.tensor([[False, False, True, True, False], [False, False, True, True, True]])
    selected = mask[:, 1:]
    lp = logits[:, :-1].log_softmax(-1)
    nll = -lp.gather(-1, ids[:, 1:, None]).squeeze(-1)
    expected = nll[selected].mean()
    actual = seq(logits, ids, mask)
    torch.testing.assert_close(actual, expected)
    assert torch.autograd.gradcheck(lambda z: seq(z, ids, mask), (logits,))
    grad, = torch.autograd.grad(actual, logits)
    full_mask = torch.cat((selected, torch.zeros(2, 1, dtype=torch.bool)), dim=1)
    assert torch.equal(grad[~full_mask], torch.zeros_like(grad[~full_mask]))
    for padding in (float("nan"), float("inf"), -float("inf"), 1e30):
        padded = logits.detach().masked_fill(~full_mask[..., None], padding).requires_grad_()
        value = seq(padded, ids, mask)
        derivative, = torch.autograd.grad(value, padded)
        torch.testing.assert_close(value, actual)
        torch.testing.assert_close(derivative, grad)
    for args in ((logits[:, :, 0], ids, mask), (logits, ids.float(), mask),
                 (logits, ids, mask.float()), (logits, ids, torch.zeros_like(mask)),
                 (logits.to(torch.complex128), ids, mask),
                 (logits.detach().masked_fill(full_mask[..., None], float("nan")), ids, mask),
                 (logits, ids.masked_fill(mask, -1), mask),
                 (logits, ids.masked_fill(mask, 7), mask)):
        rejects(seq, *args)

    # Independent categorical KL and analytic temperature-scaled derivatives.
    student = torch.randn(2, 4, 5, generator=generator, dtype=f64, requires_grad=True)
    teacher = torch.randn(2, 4, 5, generator=generator, dtype=f64, requires_grad=True)
    target_mask = torch.tensor([[True, False, False, False], [True, True, True, False]])
    counts = target_mask.sum(-1)
    checks = 0
    for temperature in (.5, 1., 2., 8.):
        p, q = (teacher.detach()/temperature).softmax(-1), (student/temperature).softmax(-1)
        categorical = (p*(p.log()-q.log())).sum(-1)*temperature**2
        for reduction in ("token_mean", "sequence_mean"):
            value = kd(student, teacher, target_mask, temperature=temperature, reduction=reduction)
            weights = target_mask.to(f64)
            if reduction == "token_mean":
                weights /= target_mask.sum()
            else:
                weights /= counts[:, None]*len(counts)
            torch.testing.assert_close(value, (categorical*weights).sum())
            student_grad, teacher_grad = torch.autograd.grad(value, (student, teacher), allow_unused=True, retain_graph=True)
            torch.testing.assert_close(student_grad, temperature*(q-p)*weights[..., None])
            assert teacher_grad is None
            assert torch.autograd.gradcheck(
                lambda z: kd(z, teacher, target_mask, temperature=temperature, reduction=reduction), (student,))
            for padding in (float("nan"), float("inf"), -float("inf"), 1e30):
                padded_s = student.detach().masked_fill(~target_mask[..., None], padding).requires_grad_()
                padded_t = teacher.detach().masked_fill(~target_mask[..., None], padding)
                padded_loss = kd(padded_s, padded_t, target_mask, temperature=temperature, reduction=reduction)
                padded_grad, = torch.autograd.grad(padded_loss, padded_s)
                torch.testing.assert_close(padded_loss, value)
                torch.testing.assert_close(padded_grad, student_grad)
            checks += 1
    # Softmax has no sensitivity to a common shift of every logit.
    torch.testing.assert_close(kd(student+100, teacher-27, target_mask), kd(student, teacher, target_mask))
    high = student.detach().requires_grad_()
    high_loss = kd(high, teacher, target_mask, temperature=1e6)
    high_grad, = torch.autograd.grad(high_loss, high)
    centered_difference = (student.detach()-student.detach().mean(-1, keepdim=True)
                           -teacher.detach()+teacher.detach().mean(-1, keepdim=True))/5
    reference_gradient = centered_difference*target_mask[..., None]/target_mask.sum()
    torch.testing.assert_close(high_grad, reference_gradient, rtol=2e-5, atol=2e-7)
    for temperature in (0, -1, float("nan"), float("inf")):
        rejects(kd, student, teacher, target_mask, temperature=temperature)
    for args, kwargs in (
        ((student, teacher[:, :, :4], target_mask), {}),
        ((student[:, :, :0], teacher[:, :, :0], target_mask), {}),
        ((student, teacher, target_mask.float()), {}),
        ((student, teacher, torch.zeros_like(target_mask)), {}),
        ((student.to(torch.complex128), teacher, target_mask), {}),
        ((student, teacher.to(torch.complex128), target_mask), {}),
        ((student, teacher, target_mask), {"reduction": "batchmean"}),
        ((student.detach().masked_fill(target_mask[..., None], float("nan")), teacher, target_mask), {}),
    ):
        rejects(kd, *args, **kwargs)

    # The actual, reviewed upstream static loss only; do not import/run the trainer.
    source_bytes = SOURCE.read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest() == "0f4e9f47a2c722fbd925ea006ba3172448ef4d15679883df3400569b49c254cd"
    tree = ast.parse(source_bytes)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GKDTrainer")
    method = copy.deepcopy(next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                                and node.name == "generalized_jsd_loss"))
    method.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    scope = {"torch": torch, "F": F}
    exec(compile(module, str(SOURCE)+":isolated-static-loss", "exec"), scope)
    upstream = scope["generalized_jsd_loss"]
    p, q = teacher.detach().softmax(-1), student.softmax(-1)
    labels = torch.zeros_like(target_mask, dtype=torch.long).masked_fill(~target_mask, -100)
    for beta in (0., .1, .5, .9, 1.):
        if beta == 0:
            divergence = (p*(p.log()-q.log())).sum(-1)
        elif beta == 1:
            divergence = (q*(q.log()-p.log())).sum(-1)
        else:
            mixture = beta*p+(1-beta)*q
            divergence = (beta*p*(p.log()-mixture.log())+(1-beta)*q*(q.log()-mixture.log())).sum(-1)
        expected = divergence[target_mask].mean()
        value = upstream(student, teacher.detach(), labels=labels, beta=beta)
        torch.testing.assert_close(value, expected)
        actual_grad, = torch.autograd.grad(value, student, retain_graph=True)
        expected_grad, = torch.autograd.grad(expected, student, retain_graph=True)
        torch.testing.assert_close(actual_grad, expected_grad)
        # Two unequal microbatches keep the original global denominator.
        split = sum(upstream(student[i:i+1], teacher.detach()[i:i+1], labels=labels[i:i+1],
                             beta=beta, num_items_in_batch=int(target_mask.sum())) for i in range(2))
        torch.testing.assert_close(split, value)
    assert ns["div_results"]["wide"][0] < ns["div_results"]["mode"][0]
    assert ns["div_results"]["wide"][1] > ns["div_results"]["mode"][1]

    # Real randomly initialized Llama CPU step, not a reasoning evaluation.
    assert importlib.metadata.version("transformers") == "4.57.1"
    assert ns["gkd_ids"].shape == (2, 6) and not ns["gkd_ids"].requires_grad
    assert torch.equal(ns["gkd_ids"][:, :3], ns["gkd_prompts"])
    assert torch.equal(ns["gkd_mask"].sum(-1), torch.tensor([3, 3]))
    assert not ns["gkd_mask"][:, :2].any() and ns["gkd_mask"][:, 2:].all()
    assert not ns["gkd_teacher_logits"].requires_grad
    assert ns["gkd_loss_after"].item() < ns["gkd_loss_before"].item()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in ns["gkd_student"].parameters())
    for name, parameter in ns["gkd_teacher"].named_parameters():
        assert parameter.grad is None and not parameter.requires_grad
        assert torch.equal(parameter, ns["gkd_teacher_before"][name])

    assert sum((395285, 211129, 10124, 10395, 177812)) == 804745
    assert math.isclose(1-(1-.9)**4, .9999)
    assert math.isclose(1-(1-.1)**4, .3439)
    ranks = torch.arange(3, dtype=f64)
    weights = 2.**(-ranks)
    torch.testing.assert_close(weights/weights.sum(), torch.tensor([4/7, 2/7, 1/7], dtype=f64))

    print(json.dumps({"python_fences": len(blocks), "device": "cpu", "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"), "temperature_reductions": checks,
        "upstream_jsd_beta_cases": 5, "example_stdout": stdout.getvalue().splitlines(),
        "chapter_sha256": hashlib.sha256(text.encode()).hexdigest()}, indent=2))
    print("PASS actual distillation fences, math, isolated upstream loss and tiny Llama step; not a full training run")


if __name__ == "__main__":
    main()
