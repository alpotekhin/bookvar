#!/usr/bin/env python3
"""Execute the seven completed EN chapters' actual CPU teaching examples.

No pretrained weights, inference endpoint, GPU kernels or performance benchmark.
Run with the cached torch 2.14.1 / transformers 4.57.1 / numpy 2.3.5 environment.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import re

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "en/00 Textbook/14 Inference and optimization"
CASES = {
    "54 Decoding and next-token selection.md": (6, list(range(6))),
    "55 KV cache batching and PagedAttention.md": (5, list(range(5))),
    "55b Scheduling — continuous batching, chunked prefill, and prefix caching.md": (6, list(range(6))),
    "55c Serving engines — vLLM, SGLang, TensorRT-LLM, and FlashInfer.md": (7, [5, 6]),
    "56 FlashAttention.md": (2, [0]),
    "57 Quantizing language models.md": (6, list(range(6))),
    "58 Speculative decoding.md": (7, list(range(7))),
}


def main():
    torch.set_num_threads(1)
    results, namespaces = [], {}
    for name, (count, selected) in CASES.items():
        path = DIRECTORY / name
        text = path.read_text()
        blocks = [m[1] for m in re.findall(
            r"^(```|~~~)python\n(.*?)^\1\s*$", text, re.M | re.S)]
        assert len(blocks) == count, (name, len(blocks))
        ns, output = {}, io.StringIO()
        with contextlib.redirect_stdout(output):
            for index in selected:
                exec(compile(blocks[index], f"{path}:python-{index + 1}", "exec"), ns)
        namespaces[name.split()[0]] = ns
        results.append({
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "python_blocks": count,
            "executed_blocks_one_based": [index + 1 for index in selected],
            "not_executed_blocks_one_based": [i + 1 for i in range(count) if i not in selected],
            "stdout": output.getvalue().splitlines(),
        })

    # Test the full blocked attention against an independent dense reference,
    # including partial tiles and gradients through all three inputs.
    blocked = namespaces["56"]["blocked_attention"]
    torch.manual_seed(7)
    attention_cases = 0
    for n in (2, 7, 11):
        for d in (1, 4):
            for qb, kb in ((1, 1), (4, 3), (16, 2)):
                for causal in (False, True):
                    q, k, v = [torch.randn(n, d, dtype=torch.float64, requires_grad=True)
                               for _ in range(3)]
                    actual, lse = blocked(q, k, v, q_block=qb, k_block=kb, causal=causal)
                    scores = q @ k.T / math.sqrt(d)
                    if causal:
                        scores = scores.masked_fill(~torch.ones(n, n, dtype=torch.bool).tril(), -torch.inf)
                    expected = scores.softmax(-1) @ v
                    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
                    torch.testing.assert_close(lse, scores.logsumexp(-1), atol=1e-12, rtol=1e-12)
                    upstream = torch.randn_like(expected)
                    got = torch.autograd.grad((actual * upstream).sum(), (q, k, v), retain_graph=True)
                    want = torch.autograd.grad((expected * upstream).sum(), (q, k, v))
                    for a, b in zip(got, want):
                        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)
                    attention_cases += 1

    quant = namespaces["57"]
    for low in range(-8, 8):
        for high in range(-8, 8):
            assert quant["unpack_int4_pair"](quant["pack_int4_pair"](low, high)) == (low, high)
    # QAT example: STE update, rather than the true derivative of round.
    master, scale = .26, .1
    fake = round(master / scale) * scale
    updated = master - .1 * 2 * (fake - 1)
    assert math.isclose(updated, .40)
    assert math.isclose((fake - 1) ** 2, .49)
    assert math.isclose((round(updated / scale) * scale - 1) ** 2, .36)

    spec = namespaces["58"]
    # Verify accepted mass plus residual over many distributions, including
    # zero proposal support, equal distributions and disjoint support.
    rng = np.random.default_rng(20261029)
    distributions = [(np.array([1., 0., 0.]), np.array([0., .4, .6])),
                     (np.array([.2, .5, .3]), np.array([.2, .5, .3]))]
    distributions += [(rng.dirichlet(np.ones(3)), rng.dirichlet(np.ones(3)))
                      for _ in range(100)]
    for proposal, target in distributions:
        accepted = np.minimum(proposal, target)
        rejected = 1 - accepted.sum()
        residual = np.maximum(target - proposal, 0)
        if residual.sum() > 1e-14:
            result = accepted + rejected * residual / residual.sum()
        else:
            result = accepted
        np.testing.assert_allclose(result, target, atol=1e-14)
    assert spec["expected_progress"](0, 4) == 1
    assert spec["expected_progress"](1, 4) == 5
    print(json.dumps({
        "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "status": "pass",
        "versions": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "numpy", "tokenizers")},
        "chapters": results,
        "executed_python_blocks": sum(len(v[1]) for v in CASES.values()),
        "independent_checks": {"attention_output_lse_gradient_cases": attention_cases,
                               "int4_pairs": 256, "residual_distributions": len(distributions),
                               "qat_ste_update": True},
        "limitations": ["Serving source fragments 1–5 require pinned runtime/GPU objects and were not run.",
                        "FlashAttention block 2 is a CUDA-only diagnostic and was not run.",
                        "CPU correctness is not a latency, throughput, quality or GPU validation."],
    }, indent=2))


if __name__ == "__main__":
    main()
