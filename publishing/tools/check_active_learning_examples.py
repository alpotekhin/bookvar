"""Execute all actual active-learning chapter programs and CPU cross-checks.

Fixed arrays and a two-round toy logistic model only. No external annotation,
dataset/weight downloads, modAL installation, GPU or performance benchmark.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/11 Pre-training и Scaling/41c Active learning — запрос меток и выбор примеров.md"
NOTEBOOK = ROOT / "05 Источники/Originals/active-learning-2026-10-28/modal/uncertainty_sampling.ipynb"


def rejects(fn, value):
    try:
        fn(value)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def main():
    text = CHAPTER.read_text()
    blocks = re.findall(r"^```python\n(.*?)^```\s*$", text, re.M | re.S)
    assert len(blocks) == 5
    scopes, output = [], io.StringIO()
    for i, block in enumerate(blocks):
        scope = {}
        with contextlib.redirect_stdout(output):
            exec(compile(block, f"{CHAPTER}:python-{i+1}", "exec"), scope)
        scopes.append(scope)
    source = json.loads(NOTEBOOK.read_text())
    for index in (3, 12):
        assert "".join(source["cells"][index]["source"]) in blocks[0]

    # Binary least confidence, negative margin and entropy have the same optimum.
    p = np.linspace(0., 1., 1001)
    distributions = np.column_stack([p, 1-p])
    safe = np.where(distributions > 0, distributions, 1.)
    uncertainty = 1-distributions.max(axis=1)
    margin = np.abs(2*p-1)
    entropy = -(distributions*np.log(safe)).sum(axis=1)
    assert uncertainty.argmax() == margin.argmin() == entropy.argmax() == 500
    np.testing.assert_allclose(uncertainty, np.minimum(p, 1-p), atol=1e-14)
    assert np.all(np.diff(entropy[:501]) >= 0)
    assert np.all(np.diff(entropy[500:]) <= 0)

    stats = scopes[2]["committee_statistics"]
    rng = np.random.default_rng(29)
    committee_cases = 0
    for classes in (2, 3, 7):
        for members in (1, 2, 5):
            for _ in range(10):
                probabilities = rng.dirichlet(np.ones(classes), size=members)
                h, disagreement = stats(probabilities)
                consensus = probabilities.mean(axis=0)
                expected = sum(sum(float(q)*np.log(float(q)/float(m))
                                   for q, m in zip(row, consensus))
                               for row in probabilities)/members
                assert np.isclose(disagreement, expected, atol=1e-14)
                assert -1e-14 <= disagreement <= h+1e-14
                assert h <= np.log(classes)+1e-14
                _, same = stats(np.repeat(probabilities[:1], members, axis=0))
                assert abs(same) < 1e-14
                committee_cases += 1
    h, disagreement = stats([[1., 0.], [0., 1.]])
    assert np.isclose(h, np.log(2)) and np.isclose(disagreement, np.log(2))
    invalid = ([], np.empty((0, 2)), np.empty((2, 0)), [[.8, .1]],
               [[-.1, 1.1]], [[float("nan"), 0.]], [[float("inf"), 0.]], [.5, .5])
    for value in invalid:
        rejects(stats, value)

    # Independently differentiate each possible binary label.
    gradient_cases = 0
    for dim in (1, 2, 4):
        for _ in range(20):
            w = torch.tensor(rng.normal(size=dim), requires_grad=True)
            h = torch.tensor(rng.normal(size=dim))
            probability = torch.sigmoid(w @ h).detach()
            norms = []
            for label in (0., 1.):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    w @ h, torch.tensor(label, dtype=torch.float64))
                gradient, = torch.autograd.grad(loss, w)
                torch.testing.assert_close(gradient, (probability-label)*h)
                norms.append(gradient.norm())
            actual = (1-probability)*norms[0]+probability*norms[1]
            torch.testing.assert_close(actual, 2*probability*(1-probability)*h.norm())
            gradient_cases += 1

    loop = scopes[4]
    assert set(loop["oracle_calls"]) == set(loop["labelled"])-{10, 20}
    assert set(loop["labelled"]).isdisjoint(loop["pool"])
    assert set(loop["labelled"]) | loop["pool"] == set(loop["features"])
    assert all(loop["labelled"][i] == int(loop["features"][i] > 0)
               for i in loop["oracle_calls"])
    assert [(r[3], r[4]) for r in loop["trace"]] == [(3, 3), (4, 2)]
    assert 100*2 < 60*5  # Fewer queried labels need not reduce annotation time.
    assert np.isclose(.9*1+.1*.5, .95)  # Traffic mix differs from hard-only queries.

    print(output.getvalue(), end="")
    print(json.dumps({
        "chapter_sha256": hashlib.sha256(CHAPTER.read_bytes()).hexdigest(),
        "actual_programs": 5, "independent_program_namespaces": True,
        "original_notebook_spans_byte_exact": [3, 12],
        "binary_probability_points": len(p), "committee_cases": committee_cases,
        "committee_invalid_cases": len(invalid), "autograd_egl_cases": gradient_cases,
        "actual_annotation_rounds": len(loop["trace"]),
        "actual_queried_ids": loop["oracle_calls"],
        "numpy": np.__version__, "torch": torch.__version__,
        "benchmark_reproduced": False,
    }, indent=2))


if __name__ == "__main__":
    main()
