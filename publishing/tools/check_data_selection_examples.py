"""Run all eight actual chapter programs and independent CPU checks.

Uses NumPy/PyTorch/scikit-learn. The added OOF example fits three small synthetic
logistic regressions. No downloaded data, weights, GPU runs or benchmarks.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import itertools
import json
from pathlib import Path
import re

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/11 Pre-training и Scaling/41b Отбор обучающих примеров — EL2N, GraNd и forgetting.md"
SOURCE = ROOT / "05 Источники/Originals/data-selection-2026-10-28/data-diet/data_diet/forgetting.py"


def rejects(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def main():
    text = CHAPTER.read_text()
    blocks = re.findall(r"^```python\n(.*?)^```\s*$", text, re.M | re.S)
    assert len(blocks) == 8
    scopes = []
    output = io.StringIO()
    for i, code in enumerate(blocks):
        scope = {}
        with contextlib.redirect_stdout(output):
            exec(compile(code, f"{CHAPTER}:python-{i+1}", "exec"), scope)
        scopes.append(scope)

    # Both copied source helpers retain the entire exact original function span.
    original = SOURCE.read_text()
    for node in ast.parse(original).body:
        if isinstance(node, ast.FunctionDef) and node.name in (
                "init_forget_stats", "update_forget_stats"):
            assert ast.get_source_segment(original, node) in blocks[2]

    rng = np.random.default_rng(19)
    probability_cases = 0
    for classes in (2, 3, 7):
        p = rng.dirichlet(np.ones(classes), size=100)
        labels = rng.integers(classes, size=100)
        y = np.eye(classes)[labels]
        e = np.linalg.norm(p-y, axis=1)
        assert np.all(e >= 0) and np.all(e <= np.sqrt(2))
        logits = torch.tensor(np.log(p), dtype=torch.float64, requires_grad=True)
        loss = torch.nn.functional.cross_entropy(
            logits, torch.tensor(labels), reduction="sum")
        gradient, = torch.autograd.grad(loss, logits)
        np.testing.assert_allclose(gradient.numpy(), p-y, atol=1e-14)
        probability_cases += len(p)

    # Averaging squared norms can reverse the ensemble ranking.
    errors_a = np.array([0., 1.])
    errors_b = np.array([.6, .6])
    assert errors_a.mean() < errors_b.mean()
    assert (errors_a**2).mean() > (errors_b**2).mean()

    # Per-example full gradient for a linear head, independent of the chapter's W=0.
    gradient_cases = 0
    for classes in (2, 4):
        for dim in (1, 3):
            for _ in range(10):
                W = torch.tensor(rng.normal(size=(classes, dim)), requires_grad=True)
                h = torch.tensor(rng.normal(size=dim))
                target = int(rng.integers(classes))
                z = W @ h
                loss = torch.nn.functional.cross_entropy(z[None], torch.tensor([target]))
                actual, = torch.autograd.grad(loss, W)
                error = z.softmax(0)-torch.nn.functional.one_hot(torch.tensor(target), classes)
                torch.testing.assert_close(actual, error[:, None]*h[None])
                torch.testing.assert_close(actual.norm(), error.norm()*h.norm())
                gradient_cases += 1

    init = scopes[2]["init_forget_stats"]
    update = scopes[2]["update_forget_stats"]
    Namespace = scopes[2]["SimpleNamespace"]
    for values in itertools.product((0, 1), repeat=5):
        state = init(Namespace(num_train_examples=1))
        for v in values:
            update(state, np.array([0]), np.array([v]))
        expected = sum(a == 1 and b == 0 for a, b in zip(values, values[1:]))
        assert state.num_forgets.tolist() == [expected]
        assert state.never_correct.tolist() == ([] if any(values) else [0])

    score = scopes[3]["score_by_id"]
    pool = scopes[3]["LabelledPool"]()
    model = torch.nn.Identity()
    Loader, Subset = scopes[3]["DataLoader"], scopes[3]["Subset"]
    expected = {101: np.sqrt(2)*.01, 7: np.sqrt(2)*.49, 55: np.sqrt(2)*.99}
    id_cases = 0
    for order in itertools.permutations(range(3)):
        for batch_size in (1, 2, 3, 4):
            result = score(model, Loader(Subset(pool, order), batch_size=batch_size), pool.ids)
            for key in expected:
                assert np.isclose(result[key], expected[key], atol=1e-14)
            assert model.training
            id_cases += 1
    for order, expected_ids in (([0, 0, 2], pool.ids), ([0, 1], pool.ids),
                                ([0, 1, 2], [101, 7, 999]),
                                ([0, 1, 2], [101, 101, 55])):
        rejects(score, model, Loader(Subset(pool, order), batch_size=2), expected_ids)
        assert model.training
    model.eval()
    score(model, Loader(pool, batch_size=2), pool.ids)
    assert not model.training
    pool.logits[0, 0] = float("nan")
    rejects(score, model, Loader(pool, batch_size=2), pool.ids)
    assert not model.training

    print(output.getvalue(), end="")
    print(json.dumps({
        "chapter_sha256": hashlib.sha256(CHAPTER.read_bytes()).hexdigest(),
        "actual_programs": len(blocks), "independent_program_namespaces": True,
        "probability_cases": probability_cases, "linear_gradient_cases": gradient_cases,
        "forgetting_histories": 32, "id_order_batch_cases": id_cases,
        "id_error_cases": 5, "source_helpers_byte_exact": True,
        "numpy": np.__version__, "torch": torch.__version__,
        "synthetic_oof_logistic_fits": 3, "benchmark_reproduced": False,
    }, indent=2))


if __name__ == "__main__":
    main()
