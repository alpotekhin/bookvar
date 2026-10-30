#!/usr/bin/env python3
"""CPU checks for chapter 55c, not a serving-engine or FlashInfer integration test.

Executes the actual Markdown Python example, then checks its attention result
against a scalar reference, request isolation, and NHD/HND axis semantics.
Requires NumPy; uses no PyTorch, CUDA, NCCL, model weights or HTTP service.
"""
from __future__ import annotations

import contextlib
import io
import math
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / (
    "00 Учебник/14 Inference и оптимизация/"
    "55c Serving engines — vLLM, SGLang, TensorRT-LLM и FlashInfer.md"
)
ATOL = 1e-12


def offsets_to_owner(indptr: np.ndarray, count: int) -> np.ndarray:
    """Validate the example's nonempty-sequence contract; not a library API."""
    if (
        indptr.ndim != 1
        or len(indptr) < 2
        or not np.issubdtype(indptr.dtype, np.integer)
        or indptr[0] != 0
        or indptr[-1] != count
        or np.any(np.diff(indptr) <= 0)
    ):
        raise ValueError("Offsets must partition all rows into nonempty sequences")
    return np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))


def scalar_reference(q, k, v, indptr):
    """Independent per-query implementation; no packed mask or matrix multiply."""
    output = np.zeros_like(v)
    for start, end in zip(indptr[:-1], indptr[1:]):
        for query in range(start, end):
            logits = [
                sum(float(a) * float(b) for a, b in zip(q[query], k[key]))
                / math.sqrt(q.shape[1])
                for key in range(start, query + 1)
            ]
            peak = max(logits)
            numerators = [math.exp(score - peak) for score in logits]
            denominator = sum(numerators)
            for dim in range(v.shape[1]):
                output[query, dim] = sum(
                    numerator * float(v[key, dim])
                    for numerator, key in zip(numerators, range(start, query + 1))
                ) / denominator
    return output


def assert_close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=ATOL, atol=ATOL)


def check_markdown_example():
    text = CHAPTER.read_text(encoding="utf-8")
    blocks = re.findall(
        r"<!-- checked-example: serving-packed-attention -->\n```python\n(.*?)^```\s*$",
        text, flags=re.M | re.S,
    )
    assert len(blocks) == 1, "Expect exactly one marked CPU example; upstream GPU snippets are not executed"
    namespace = {"__name__": "__chapter_example__"}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(blocks[0], str(CHAPTER) + ":python-example", "exec"), namespace)
    assert "Границы запросов сохранены" in stdout.getvalue()
    assert_close(namespace["indptr"], np.array([0, 5, 8, 12]))
    reference = scalar_reference(
        namespace["q"], namespace["k"], namespace["v"], namespace["indptr"]
    )
    assert_close(namespace["packed"], reference)
    assert_close(namespace["wrong"][:5], reference[:5])
    assert not np.allclose(namespace["wrong"][5:], reference[5:])
    error = float(np.max(np.abs(namespace["packed"] - reference)))
    print(f"PASS Markdown block and independent scalar reference; max error {error:.3e}")
    return namespace


def check_request_isolation(ns):
    attention = ns["attention"]
    q, k, v = ns["q"], ns["k"], ns["v"]
    allowed = ns["causal"] & ns["same_request"]
    changed_v = v.copy()
    changed_v[:5] += 100.0
    correct = attention(q, k, changed_v, allowed)
    assert_close(correct[:5], ns["packed"][:5] + 100.0)
    assert_close(correct[5:], ns["packed"][5:])

    leaked = attention(q, k, changed_v, ns["causal"])
    assert np.max(np.abs(leaked[5:8] - ns["wrong"][5:8])) > 1.0
    assert np.max(np.abs(leaked[8:] - ns["wrong"][8:])) > 1.0
    print("PASS request isolation; wrong global causal mask leaks across both boundaries")


def check_offsets_and_layout(ns):
    owner = offsets_to_owner(ns["indptr"], 12)
    np.testing.assert_array_equal(owner, ns["owner"])
    malformed = (
        np.array([0, 5, 8, 11]),  # Uncovered final row.
        np.array([-1, 5, 8, 12]),
        np.array([1, 5, 8, 12]),
        np.array([0, 5, 4, 12]),
        np.array([0, 5, 5, 12]),  # Empty sequence excluded by this example contract.
        np.array([0.0, 5.0, 8.0, 12.0]),
    )
    for offsets in malformed:
        try:
            offsets_to_owner(offsets, 12)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted malformed offsets: {offsets}")

    # Two heads, unlike the chapter's one-head simplification.
    rng = np.random.default_rng(19)
    q, k, v = rng.normal(size=(3, 12, 2, 4))
    allowed = ns["causal"] & ns["same_request"]
    attention = ns["attention"]
    nhd = np.stack(
        [attention(q[:, h], k[:, h], v[:, h], allowed) for h in range(2)], axis=1
    )
    k_hnd = np.ascontiguousarray(k.transpose(1, 0, 2))
    v_hnd = np.ascontiguousarray(v.transpose(1, 0, 2))
    hnd = np.stack(
        [attention(q[:, h], k_hnd[h], v_hnd[h], allowed) for h in range(2)], axis=1
    )
    assert_close(nhd, hnd)

    # Reinterpreting the contiguous HND data as NHD without a permutation is wrong.
    mislabeled_k, mislabeled_v = k_hnd.reshape(k.shape), v_hnd.reshape(v.shape)
    wrong = np.stack(
        [
            attention(q[:, h], mislabeled_k[:, h], mislabeled_v[:, h], allowed)
            for h in range(2)
        ], axis=1
    )
    assert not np.allclose(nhd, wrong)
    print("PASS offset guards and NHD/HND permutation; relabel/reshape control fails")


def check_generated_token_accounting():
    # Bookkeeping only: these are stipulated outputs, not an executed language model.
    cached, selected = [], [12, 4, "EOS"]
    new_inputs = [[7, 3, 9], [12], [4]]
    observed = []
    for incoming, output in zip(new_inputs, selected):
        cached.extend(incoming)
        observed.append((cached.copy(), output))
    assert observed == [
        ([7, 3, 9], 12),
        ([7, 3, 9, 12], 4),
        ([7, 3, 9, 12, 4], "EOS"),
    ]
    assert "EOS" not in cached
    print("PASS stipulated prefill/decode token-versus-KV accounting")


def main():
    ns = check_markdown_example()
    check_request_isolation(ns)
    check_offsets_and_layout(ns)
    check_generated_token_accounting()
    print("CPU-only mathematical checks passed; no engine/GPU/performance claim.")


if __name__ == "__main__":
    main()
