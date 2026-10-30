"""Execute the reader programs and independently check geometric selection.

No model/label downloads, training benchmarks, Gurobi or historical sklearn
installation. The archived Google class is executed with explicitly substituted
NumPy distances and a minimal flatten_X base, to isolate its selection logic.
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

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/11 Pre-training и Scaling/41d Геометрический отбор — k-center, coreset и herding.md"
GOOGLE = ROOT / "05 Источники/Originals/active-learning-2026-10-28/google/kcenter_greedy.py"


def main():
    text = CHAPTER.read_text()
    blocks = re.findall(r"^```python\n(.*?)^```\s*$", text, re.M | re.S)
    assert len(blocks) == 6
    scopes, stdout = [], io.StringIO()
    for index, block in enumerate(blocks):
        scope = {"farthest_first": scopes[0]["farthest_first"]} if index == 1 else {}
        with contextlib.redirect_stdout(stdout):
            exec(compile(block, f"{CHAPTER}:python-{index + 1}", "exec"), scope)
        scopes.append(scope)
    select = scopes[0]["farthest_first"]
    rng = np.random.default_rng(20261028)
    cases, permutations, zero_cases = 0, 0, 0

    # Independent full-matrix enumeration, not the reader's running-min update.
    for dim in (1, 2, 4):
        for n in range(2, 9):
            for repeat in range(8):
                z = rng.integers(-3, 4, size=(n, dim)).astype(float)
                ids = rng.permutation(np.arange(n) * 7 + 101)
                pairwise = np.sqrt(np.sum((z[:, None] - z[None, :]) ** 2, axis=-1))
                positions = {int(identifier): row for row, identifier in enumerate(ids)}
                for seed_count in range(min(2, n - 1) + 1):
                    seeds = list(map(int, ids[:seed_count]))
                    candidates = list(range(seed_count, n))
                    for budget in range(1, min(3, n - seed_count) + 1):
                        chosen, nearest = select(z, ids, seeds, budget)
                        rows = list(range(seed_count)) + [positions[i] for i in chosen]
                        expected = pairwise[:, rows].min(axis=1)
                        np.testing.assert_allclose(nearest, expected, atol=1e-14)
                        assert len(chosen) == len(set(chosen)) == budget
                        assert not set(chosen) & set(seeds)
                        optimum = min(float(pairwise[:, list(range(seed_count)) + list(subset)]
                                            .min(axis=1).max())
                                      for subset in itertools.combinations(candidates, budget))
                        assert nearest.max() <= 2 * optimum + 1e-12
                        zero_cases += int(optimum == 0)
                        # Check the actual farthest-first rule at every selection.
                        earlier = list(range(seed_count))
                        for identifier in chosen:
                            available = [i for i in range(n) if i not in earlier]
                            distances = pairwise[:, earlier].min(axis=1) if earlier else np.full(n, np.inf)
                            best = min(available, key=lambda i: (-distances[i], int(ids[i])))
                            assert identifier == ids[best]
                            earlier.append(positions[identifier])
                        order = rng.permutation(n)
                        reordered, _ = select(z[order], ids[order], seeds, budget)
                        assert reordered == chosen
                        permutations += 1
                        cases += 1

    # The metric guarantee can be tight even in one dimension.
    chosen, nearest = select([[0.], [1.], [2.]], [0, 1, 2], [], 1)
    assert chosen == [0] and nearest.max() == 2.
    assert np.abs(np.array([0., 1., 2.]) - 1.).max() == 1.
    _, nearest = select([[0.], [1.]], [0, 1], [0], 0)
    np.testing.assert_array_equal(nearest, [0., 1.])

    invalid = [([], [], [], 1), ([[1., np.nan]], [1], [], 1),
               ([[np.inf]], [1], [], 1), ([[0.], [1.]], [1, 1], [], 1),
               ([[0.]], [1.], [], 1), ([[0.]], [1, 2], [], 1),
               ([[0.]], [1], [2], 0), ([[0.]], [1], [1, 1], 0),
               ([[0.]], [1], [], -1), ([[0.]], [1], [], 1.5),
               ([[0.]], [1], [], True), ([[0.]], [1], [], 2),
               ([[0.]], [1], [], 0), ([[0.]], [1], [True], 0)]
    for args in invalid:
        try:
            select(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid input accepted: {args}")

    # Execute the archived Google class body, with imports removed explicitly.
    source = GOOGLE.read_text()
    parsed = ast.parse(source)
    parsed.body = [node for node in parsed.body if not isinstance(node, (ast.Import, ast.ImportFrom))]

    class FlattenBase:
        def flatten_X(self):
            return self.X.reshape(len(self.X), -1)

    class IdentityModel:
        def transform(self, x):
            return x

    def numpy_distances(x, y, metric):
        assert metric == "euclidean"
        return np.linalg.norm(x[:, None] - y[None, :], axis=-1)

    scope = {"np": np, "SamplingMethod": FlattenBase, "pairwise_distances": numpy_distances}
    exec(compile(parsed, str(GOOGLE), "exec"), scope)
    cls = scope["kCenterGreedy"]
    with contextlib.redirect_stdout(stdout):
        try:
            cls(np.zeros((4, 2)), None, 0).select_batch_(IdentityModel(), [0], 2)
        except AssertionError:
            seed_assertion = True
        else:
            raise AssertionError("expected all-zero selected-seed assertion")
        repeated = cls(np.zeros((4, 2)), None, 0).select_batch_(IdentityModel(), [2], 2)
    assert repeated == [0, 0]

    # Uniform additive grid bound over many different query centre sets.
    grid = scopes[4]
    points, small, radius = grid["points"], grid["small"], grid["radius"]
    grid_cases = 0
    for centres_count in (1, 2, 4):
        for _ in range(40):
            centres = rng.normal(size=(centres_count, 2))
            p, q = radius(points, centres), radius(small, centres)
            assert q <= p + 1e-14
            assert p <= q + grid["width"] * np.sqrt(2.) + 1e-14
            grid_cases += 1

    # Telescoping identity under all eight actual updates, and the moment caveat.
    history = scopes[5]
    sequence, weights = history["sequence"], history["weights"]
    for t in range(1, len(sequence) + 1):
        assert np.isclose(.25 - np.mean(sequence[:t]), (weights[t] - weights[0]) / t)
    assert np.mean([-1., 0., 1.]) == np.mean([0.])
    assert not np.isclose(np.mean(np.square([-1., 0., 1.])), np.mean(np.square([0.])))
    assert abs(.25 - sequence[0]) == .75
    assert abs(.25 - 0.) == .25
    assert abs(.25 - np.mean(sequence[:4])) == 0.
    assert np.isclose(abs(.25 - np.mean(sequence[:5])), 3/20)
    expansions = 0
    for t in range(len(sequence)):
        selected_sum = sum(sequence[:t])
        w = (t + 1) * .25 - selected_sum
        assert np.isclose(w, weights[t])
        for candidate in (0., 1.):
            direct = (.25 - (selected_sum + candidate) / (t + 1)) ** 2
            expanded = (w*w - 2*w*candidate + candidate*candidate) / (t + 1) ** 2
            assert np.isclose(direct, expanded)
            expansions += 1

    print(stdout.getvalue(), end="")
    print(json.dumps({"chapter_sha256": hashlib.sha256(CHAPTER.read_bytes()).hexdigest(),
                      "programs_executed": len(blocks), "bruteforce_metric_cases": cases,
                      "stable_id_permutations": permutations, "zero_optimum_cases": zero_cases,
                      "invalid_cases": len(invalid), "grid_query_cases": grid_cases,
                      "herding_prefixes": len(sequence), "mean_error_expansions": expansions,
                      "google_class_seed_assertion": seed_assertion,
                      "google_class_duplicate_result": list(map(int, repeated)),
                      "google_distance_backend": "explicit NumPy adapter, not historical sklearn",
                      "numpy": np.__version__}, indent=2))


if __name__ == "__main__":
    main()
