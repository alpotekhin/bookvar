#!/usr/bin/env python3
"""Check the chapter's swap arithmetic and the pinned upstream weight scheme."""
from __future__ import annotations

import ast
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/15 Embeddings, Retrieval и RAG/62 Reranking — cross-encoder и late interaction.md"
FIXTURE = ROOT / "publishing/fixtures/lambdaloss-8a880ec7"


def dcg(grades, k):
    return sum((2. ** grade - 1.) / np.log2(i + 2) for i, grade in enumerate(grades[:k]))


def main():
    chapter = CHAPTER.read_text()
    blocks = re.findall(r"^```python\n(.*?)^```", chapter, re.M | re.S)
    block, = [b for b in blocks if "def ndcg_swap_weights(" in b]
    namespace = {}
    exec(compile(block, str(CHAPTER), "exec"), namespace)
    weights = namespace["ndcg_swap_weights"]

    meta = json.loads((FIXTURE / "sources.json").read_text())
    for row in meta["files"]:
        assert hashlib.sha256((FIXTURE / row["file"]).read_bytes()).hexdigest() == row["sha256"]
    assert (FIXTURE / "LICENSE").read_bytes() == (
        ROOT / "site/public/licenses/sbert-8a880ec7-Apache-2.0.txt").read_bytes()

    source = ast.parse((FIXTURE / "lambda_loss.py").read_text())
    names = {"BaseWeightingScheme", "LambdaRankScheme"}
    classes = [n for n in source.body if isinstance(n, ast.ClassDef) and n.name in names]
    upstream = {"torch": torch, "nn": nn}
    exec(compile(ast.Module(body=classes, type_ignores=[]), "lambda_loss.py", "exec"), upstream)
    scheme = upstream["LambdaRankScheme"]()
    rng = np.random.default_rng(20261109)
    swaps = gradients = source_matrices = 0
    for n in range(2, 9):
        for _ in range(4):
            scores = rng.permutation(n).astype(float) / n
            grades = rng.integers(0, 4, size=n)
            grades[0] = 2
            order = np.argsort(-scores)
            ranked = grades[order]
            for k in range(1, n + 1):
                ideal = dcg(sorted(grades, reverse=True), k)
                before = dcg(ranked, k)
                exact = weights(scores, grades, k)
                for i, j in combinations(range(n), 2):
                    swapped = ranked.copy()
                    swapped[i], swapped[j] = swapped[j], swapped[i]
                    assert np.isclose(exact[order[i], order[j]], abs(dcg(swapped, k) - before) / ideal)
                    swaps += 1
                discount = torch.log2(torch.arange(2, n + 2, dtype=torch.float64))[None, :]
                rel = torch.tensor(ranked)[None, :]
                gain = (2. ** rel.double() - 1.) / ideal
                actual = scheme(gain, discount, rel).numpy()[0]
                full_discounts = weights(scores, grades, k, truncate_discount=False)
                np.testing.assert_allclose(actual, full_discounts[np.ix_(order, order)], atol=1e-12)
                source_matrices += 1

            preferred, rejected = np.where(grades[:, None] > grades[None, :])
            frozen = weights(scores, grades)[preferred, rejected]
            score_tensor = torch.tensor(scores, requires_grad=True)
            loss = (torch.tensor(frozen) * nn.functional.softplus(
                score_tensor[rejected] - score_tensor[preferred])).sum()
            loss.backward()
            derivative = -frozen / (1. + np.exp(scores[preferred] - scores[rejected]))
            expected = np.zeros(n)
            np.add.at(expected, preferred, derivative)
            np.add.at(expected, rejected, -derivative)
            np.testing.assert_allclose(score_tensor.grad.numpy(), expected, atol=1e-12)
            gradients += 1

    invalid = [([], [], None), ([0], [0], None), ([0], [1, 0], None),
               ([float("nan")], [1], None), ([0], [-1], None),
               ([0], [1], 0), ([0], [1], True), ([0], [1], 2)]
    for args in invalid:
        try:
            weights(*args)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid input: {args}")
    print(json.dumps({
        "chapter_example": "pass",
        "brute_force_swap_comparisons": swaps,
        "pinned_scheme_matrix_comparisons": source_matrices,
        "fixed_weight_autograd_comparisons": gradients,
        "invalid_inputs_rejected": len(invalid),
        "scope": "Exact swap arithmetic and upstream LambdaRankScheme, not a full training run.",
    }, indent=2))


if __name__ == "__main__":
    main()
