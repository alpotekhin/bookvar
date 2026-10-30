"""Execute all reader programs and retain the original FixMatch/graph checks.

CPU tensors and finite arrays only. No CIFAR/model download, historical
TensorFlow installation, full FixMatch training or reproduced benchmark.
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
import torch.nn.functional as F
from topic_consolidation_baseline import before_topic_consolidation

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/11 Pre-training и Scaling/41e Обучение с частичной разметкой — pseudo-labeling и FixMatch.md"


def main():
    checked_text = before_topic_consolidation(CHAPTER.read_text(), CHAPTER,
        stop_before_journal="2026-10-29-consistency-training-additions.json")
    blocks = re.findall(r"^```python\n(.*?)^```\s*$", checked_text, re.M | re.S)
    assert len(blocks) == 10
    scopes, stdout = [], io.StringIO()
    for index, block in enumerate(blocks):
        scope = {}
        with contextlib.redirect_stdout(stdout):
            exec(compile(block, f"{CHAPTER}:python-{index+1}", "exec"), scope)
        scopes.append(scope)
    # The five original programs retain their order and independent checks.
    scopes = scopes[-5:]

    rng = np.random.default_rng(28)
    gradient_cases = 0
    for n in (1, 4, 7):
        for classes in (2, 3, 5):
            for threshold in (0., .6, .95, 1.):
                q = torch.tensor(rng.dirichlet(np.ones(classes), n), requires_grad=True)
                logits = torch.tensor(rng.normal(size=(n, classes)), requires_grad=True)
                targets = q.detach().argmax(dim=1)
                mask = q.detach().max(dim=1).values >= threshold
                loss = (F.cross_entropy(logits, targets, reduction="none") * mask).mean()
                loss.backward()
                probability = logits.detach().softmax(dim=1)
                expected = probability.clone()
                expected[torch.arange(n), targets] -= 1
                expected *= mask[:, None] / n
                torch.testing.assert_close(logits.grad, expected)
                assert q.grad is None
                if not mask.any():
                    assert loss.item() == 0 and torch.count_nonzero(logits.grad).item() == 0
                else:
                    selected_mean = F.cross_entropy(logits[mask], targets[mask])
                    torch.testing.assert_close(loss, mask.double().mean() * selected_mean)
                gradient_cases += 1

    graph_cases = 0
    for n in range(3, 11):
        for repeat in range(8):
            conductance = rng.uniform(.2, 3., n-1)
            W = np.zeros((n, n))
            for i, weight in enumerate(conductance):
                W[i, i+1] = W[i+1, i] = weight
            L = np.diag(W.sum(axis=1)) - W
            U, A = list(range(1, n-1)), [0, n-1]
            f = np.linalg.solve(L[np.ix_(U, U)], -L[np.ix_(U, A)] @ np.array([0., 1.]))
            # Independent electrical-chain solution: cumulative resistances.
            resistance = 1/conductance
            expected = np.cumsum(resistance)[:-1] / resistance.sum()
            np.testing.assert_allclose(f, expected, atol=1e-13)
            assert np.all((0 <= f) & (f <= 1))
            graph_cases += 1
    # An unlabelled disconnected component has no boundary value.
    disconnected = np.array([[1., -1.], [-1., 1.]])
    assert np.linalg.matrix_rank(disconnected) == 1
    try:
        np.linalg.solve(disconnected, np.zeros(2))
    except np.linalg.LinAlgError:
        pass
    else:
        raise AssertionError("unanchored component was not singular")
    weighted = scopes[3]["W"].copy()
    weighted[3,4] = weighted[4,3] = 10.
    laplacian = np.diag(weighted.sum(axis=1)) - weighted
    U, A = [1,2,3], [0,4]
    f = np.linalg.solve(laplacian[np.ix_(U,U)], -laplacian[np.ix_(U,A)] @ np.array([0.,1.]))
    np.testing.assert_allclose(f, np.array([10.,20.,30.])/31)
    # Independently differentiate the stated edge energy.
    torch_w = torch.tensor(weighted)
    point = torch.tensor([.1,.3,.8,.4,.9], dtype=torch.float64, requires_grad=True)
    pairs = torch.triu(torch_w, diagonal=1)
    energy = .5 * (pairs * (point[:,None]-point[None,:]).square()).sum()
    gradient, = torch.autograd.grad(energy, point)
    torch.testing.assert_close(gradient, torch.tensor(laplacian) @ point.detach())

    # Two different target functions agree on all available labelled corners.
    corners = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
    horizontal = (corners[:,0] > 0).astype(int)
    vertical = (corners[:,1] > 0).astype(int)
    np.testing.assert_array_equal(horizontal[[0,3]], vertical[[0,3]])
    assert np.count_nonzero(horizontal != vertical) == 2
    assert 100 - 5.07 == 94.93 and 100 - 11.39 == 88.61
    assert (1 << 20) * 64 == 67108864
    assert 64 + 2*7*64 == 960
    assert 4 * 1000000**2 == 4_000_000_000_000

    # Source inspection is not represented as executing TensorFlow.
    original = (ROOT / "05 Источники/Originals/semi-supervised-2026-10-28/fixmatch/fixmatch.py").read_text()
    for span in ("pseudo_labels = tf.stop_gradient(tf.nn.softmax(logits_weak))",
                 "tf.reduce_max(pseudo_labels, axis=1) >= confidence",
                 "loss_xeu = tf.reduce_mean(loss_xeu * pseudo_mask)"):
        assert span in original
    threshold_source = (ROOT / "05 Источники/Originals/semi-supervised-2026-10-28/sklearn/threshold.py").read_text()
    assert "total_samples = y.shape[0]" in threshold_source
    assert "self_training_clf.fit(X_train, y_train)" in threshold_source
    assert "total_samples\n            - np.unique(self_training_clf.labeled_iter_" in threshold_source
    assert 569 - (379 - 33) == 190 + 33  # Dataset-size offset, not pseudo labels.

    print(stdout.getvalue(), end="")
    print(json.dumps({"chapter_sha256": hashlib.sha256(CHAPTER.read_bytes()).hexdigest(),
                      "executed_program_revision_sha256": hashlib.sha256(checked_text.encode()).hexdigest(),
                      "programs_executed": len(blocks), "masked_gradient_cases": gradient_cases,
                      "weighted_graph_cases": graph_cases, "unanchored_component": "singular",
                      "edge_energy_autograd_cases": 1, "weighted_last_edge_case": True,
                      "wrong_pseudo_label_true_class_before": scopes[2]["before"],
                      "wrong_pseudo_label_true_class_after": scopes[2]["after"],
                      "numpy": np.__version__, "torch": torch.__version__}, indent=2))


if __name__ == "__main__":
    main()
