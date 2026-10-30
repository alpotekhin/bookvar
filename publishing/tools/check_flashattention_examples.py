#!/usr/bin/env python3
"""Execute chapter 56's CPU code and independently check its mathematics.

This is a correctness check, not a FlashAttention GPU implementation or benchmark.
Run with the repository's Python environment containing PyTorch.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import re
import tarfile
from fractions import Fraction
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/14 Inference и оптимизация/56 FlashAttention.md"
ATOL, RTOL = 2e-11, 2e-10


def check_source_excerpt(text):
    folder = ROOT / "05 Источники/Originals/flashattention-2026-10-19"
    info = json.loads((folder / "excerpts.json").read_text(encoding="utf-8"))
    archive = ROOT / info["archive"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == info["archive_sha256"]
    snapshot = folder / info["snapshot"]
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == info["snapshot_sha256"]
    with tarfile.open(archive) as package:
        source = package.extractfile(info["member"]).read().decode("utf-8")
    assert source.rstrip() + "\n" == snapshot.read_text(encoding="utf-8")
    start = source.index(info["start"])
    selected = source[start:source.index(info["end_exclusive"], start)].strip()
    selected = selected.replace(r"\paragraph{Compiler reordering}", "**Compiler reordering**")
    selected = selected.replace(r"\paragraph{Register pressure}", "**Register pressure**")
    selected = selected.replace(r"\vS", r"\mathbf{S}")
    selected = selected.replace(
        r"Section~\ref{sec:2-stage-sass}",
        "[Section B.2](https://arxiv.org/html/2407.08608v1)",
    )
    ident = info["excerpt_id"]
    displayed = match(
        rf"<!-- source-excerpt: {ident}:start -->([\s\S]*?)<!-- source-excerpt: {ident}:end -->",
        text,
    ).group(1)
    displayed = "\n".join(re.sub(r"^> ?", "", line) for line in displayed.splitlines())
    assert " ".join(selected.split()) == " ".join(displayed.split())
    assert info["paper_url"] in text and info["license_url"] in text
    return {"scope": info["scope"], "source_sha256": info["snapshot_sha256"]}


def match(pattern, text):
    found = re.search(pattern, text)
    if found is None:
        raise AssertionError(f"Chapter evidence missing: {pattern}")
    return found


def dense_reference(q, k, v, causal):
    scores = torch.einsum("id,jd->ij", q, k) * (q.shape[-1] ** -0.5)
    if causal:
        visible = torch.ones_like(scores, dtype=torch.bool).tril()
        scores = scores.masked_fill(~visible, -torch.inf)
    p = scores.softmax(-1)
    return torch.einsum("ij,jd->id", p, v), scores.logsumexp(-1), p


@torch.no_grad()
def recomputed_backward(q, k, v, output, lse, upstream, qb, kb, causal):
    """Evaluate the chapter's block-gradient equations, independently of autograd."""
    n, d = q.shape
    scale = d ** -0.5
    dq, dk, dv = (torch.zeros_like(x) for x in (q, k, v))
    reduction = (output * upstream).sum(-1)
    for i in range(0, n, qb):
        qi, gi = q[i:i + qb], upstream[i:i + qb]
        for j in range(0, n, kb):
            kj, vj = k[j:j + kb], v[j:j + kb]
            score = qi @ kj.T * scale
            if causal:
                rows = torch.arange(i, i + len(qi))
                cols = torch.arange(j, j + len(kj))
                score = score.masked_fill(cols[None, :] > rows[:, None], -torch.inf)
            probability = (score - lse[i:i + qb, None]).exp()
            dp = gi @ vj.T
            ds = probability * (dp - reduction[i:i + qb, None])
            dq[i:i + qb] += ds @ kj * scale
            dk[j:j + kb] += ds.T @ qi * scale
            dv[j:j + kb] += probability.T @ gi
    return dq, dk, dv


def check_numbers(text, namespace):
    n, d = map(int, match(r"При \$N=(\d+)\$, \$d=(\d+)\$ и двух байтах", text).groups())
    memory = {
        "square_mib": n * n * 2 / 2**20,
        "qkvo_mib": 4 * n * d * 2 / 2**20,
        "lse_kib": n * 4 / 2**10,
        "intermediate_traffic_mib": 4 * n * n * 2 / 2**20,
    }
    patterns = {
        "square_mib": r"\| Одна полная матрица .*?\| (\d+) МиБ \|",
        "qkvo_mib": r"\| Все четыре матрицы .*?\| (\d+) МиБ \|",
        "lse_kib": r"\| Один построчный вектор .*?\| (\d+) КиБ \|",
        "intermediate_traffic_mib": r"(\d+) МиБ\. Это не измерение",
    }
    for key, pattern in patterns.items():
        assert memory[key] == float(match(pattern, text).group(1)), key

    n_parallel, query_tile = map(
        int, match(r"для \$N=(\d+)\$, \$B_q=(\d+)\$", text).groups()
    )
    heads, blocks, tasks = map(
        int, match(r"\$(\d+)\\cdot(\d+)=(\d+)\$ задач", text).groups()
    )
    assert blocks == math.ceil(n_parallel / query_tile)
    assert tasks == heads * blocks
    rows, cols = map(int, match(r"блок scores \$(\d+)\\times(\d+)\$", text).groups())
    extra_kib = rows * cols * 4 / 1024
    assert extra_kib == float(match(r"(\d+) КиБ регистрового", text).group(1))

    q, k, v = (namespace[name] for name in ("q_example", "k_example", "v_example"))
    scores = (q @ k.T)[0] / math.sqrt(q.shape[-1])
    first_max = scores[:2].max()
    first_weights = (scores[:2] - first_max).exp()
    first_ell = first_weights.sum().item()
    first_u = (first_weights @ v[:2, 0]).item()
    new_max = scores.max()
    rescale = (first_max - new_max).exp().item()
    last_weight = (scores[2] - new_max).exp().item()
    final_ell = rescale * first_ell + last_weight
    final_u = rescale * first_u + last_weight * v[2, 0].item()
    parsed_rows = []
    for label in ("Первые два", "Все три"):
        row = match(rf"\| {label} \|[^\n]+", text).group(0)
        fields = [item.strip().strip("$") for item in row.split("|")[1:-1]]
        parsed_rows.append(tuple(float(Fraction(item)) for item in fields[2:]))
    expected_rows = [
        (first_ell, first_u, first_u / first_ell),
        (final_ell, final_u, final_u / final_ell),
    ]
    for actual, expected in zip(expected_rows, parsed_rows):
        torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected))
    assert math.isclose(rescale, 0.5, abs_tol=1e-14)
    wrong = (first_u + last_weight * v[2, 0].item()) / (first_ell + last_weight)
    printed_wrong = float(match(r"\$\(25\+40\)/\(1\.5\+1\)=(\d+)\$", text).group(1))
    assert wrong == printed_wrong == 26
    # Read the new backward table from the chapter, then compare with autograd.
    score_input = scores.detach().clone().requires_grad_()
    value_input = v[:, 0].detach().clone().requires_grad_()
    probability = score_input.softmax(-1)
    scalar_output = probability @ value_input
    ds, dv = torch.autograd.grad(scalar_output, (score_input, value_input))
    printed = []
    for index in range(1, 4):
        row = match(rf"\| {index} \|[^\n]+", text).group(0)
        fields = [part.strip().strip("$") for part in row.split("|")[2:-1]]
        printed.append([float(Fraction(part)) for part in fields])
    expected = torch.stack((probability.detach(), dv, ds), dim=1)
    torch.testing.assert_close(torch.tensor(printed, dtype=torch.float64), expected,
                               atol=1e-14, rtol=1e-14)
    assert math.isclose(scalar_output.item(), 30.0, abs_tol=1e-14)
    assert abs(ds.sum().item()) < 1e-14
    reconstructed = (scores - math.log(7)).exp()
    torch.testing.assert_close(reconstructed, probability.detach(), atol=1e-14, rtol=1e-14)
    return {**memory, "query_tasks": tasks, "pipeline_extra_kib": extra_kib,
            "online_rows": expected_rows, "rescale": rescale,
            "backward_example": {"output": scalar_output.item(), "rows": printed,
                                 "score_gradient_sum": ds.sum().item()}}


def main():
    torch.set_num_threads(1)
    torch.manual_seed(56014)
    text = CHAPTER.read_text(encoding="utf-8")
    source_evidence = check_source_excerpt(text)
    # A single marked block is the public implementation under test.
    blocks = re.findall(
        r"<!-- checked-example: flashattention-cpu -->\s*```python\n(.*?)\n```",
        text, re.DOTALL,
    )
    assert len(blocks) == 1, "Expected exactly one marked CPU implementation"
    code = blocks[0]
    namespace = {"__name__": "bookvar_chapter_example"}
    with contextlib.redirect_stdout(io.StringIO()) as captured:
        exec(compile(code, str(CHAPTER), "exec"), namespace)
    assert captured.getvalue().strip() == "30.0"
    blocked = namespace["blocked_attention"]
    numbers = check_numbers(text, namespace)
    for index, snippet in enumerate(re.findall(r"```python\n(.*?)\n```", text, re.DOTALL)):
        compile(snippet, f"chapter56-block{index}", "exec")

    errors = {name: 0.0 for name in ("output", "lse", "autograd", "recomputed_backward", "D_identity")}

    def compare(name, actual, expected, atol=ATOL, rtol=RTOL):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        error = (actual.detach() - expected.detach()).abs().max().item()
        errors[name] = max(errors.get(name, 0.0), error)

    cases = 0
    shapes = [(1, 1), (5, 3), (7, 5), (13, 8), (17, 4)]
    tilings = [(1, 1), (4, 3), (8, 5), (32, 32)]
    for n, d in shapes:
        original = [torch.randn(n, d, dtype=torch.float64) for _ in range(3)]
        upstream = torch.randn(n, d, dtype=torch.float64)
        upstream_lse = torch.randn(n, dtype=torch.float64)
        for causal in (False, True):
            for qb, kb in tilings:
                q, k, v = [x.clone().requires_grad_(True) for x in original]
                qr, kr, vr = [x.clone().requires_grad_(True) for x in original]
                output, lse = blocked(q, k, v, qb, kb, causal)
                reference, ref_lse, probabilities = dense_reference(qr, kr, vr, causal)
                compare("output", output, reference)
                compare("lse", lse, ref_lse)
                dense_dp = upstream @ vr.T
                compare("D_identity", (probabilities * dense_dp).sum(-1),
                        (reference * upstream).sum(-1))
                grad_output = torch.autograd.grad(
                    (reference * upstream).sum(), (qr, kr, vr), retain_graph=True
                )
                manual = recomputed_backward(q, k, v, output, lse, upstream, qb, kb, causal)
                for actual, expected in zip(manual, grad_output):
                    compare("recomputed_backward", actual, expected)
                # Include an independent upstream gradient for LSE as well as O.
                grad = torch.autograd.grad(
                    (output * upstream).sum() + (lse * upstream_lse).sum(), (q, k, v)
                )
                ref_grad = torch.autograd.grad(
                    (reference * upstream).sum() + (ref_lse * upstream_lse).sum(), (qr, kr, vr)
                )
                for actual, expected in zip(grad, ref_grad):
                    compare("autograd", actual, expected)
                with sdpa_kernel(SDPBackend.MATH):
                    sdpa = F.scaled_dot_product_attention(
                        q[None, None], k[None, None], v[None, None],
                        dropout_p=0.0, is_causal=causal,
                    )[0, 0]
                compare("sdpa_math", output, sdpa)
                cases += 1

    # Independent central finite differences through PyTorch's gradcheck,
    # with both O and LSE returned and every q/k/v element perturbed.
    for causal in (False, True):
        inputs = tuple(torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
                       for _ in range(3))
        assert torch.autograd.gradcheck(
            lambda q, k, v: blocked(q, k, v, 2, 2, causal), inputs,
            eps=1e-6, atol=2e-6, rtol=2e-4,
        )

    # Large scores overflow a naive exp, but stable dense/blocked calculations agree.
    extreme_q = torch.ones(5, 1, dtype=torch.float64)
    extreme_k = torch.tensor([[1000.0], [999.0], [1002.0], [-1000.0], [1001.0]],
                             dtype=torch.float64)
    extreme_v = torch.arange(5, dtype=torch.float64)[:, None]
    assert not torch.isfinite(extreme_k.exp()).all()
    for causal in (False, True):
        out, lse = blocked(extreme_q, extreme_k, extreme_v, 3, 2, causal)
        ref_out, ref_lse, _ = dense_reference(extreme_q, extreme_k, extreme_v, causal)
        compare("output", out, ref_out)
        compare("lse", lse, ref_lse)
        assert torch.isfinite(out).all() and torch.isfinite(lse).all()

    # A deliberate mutation must break the chapter's worked example.
    definition = code.split("# Realize scores", 1)[0]
    old = "alpha = torch.exp(m - m_new)"
    assert definition.count(old) == 1
    bad_namespace = {}
    exec(compile(definition.replace(old, "alpha = torch.ones_like(m)"),
                 "negative-control-no-rescale", "exec"), bad_namespace)
    args = [namespace[name] for name in ("q_example", "k_example", "v_example")]
    broken_output, _ = bad_namespace["blocked_attention"](*args, 2, 2)
    expected = torch.full_like(broken_output, 30.0)
    try:
        torch.testing.assert_close(broken_output, expected, atol=ATOL, rtol=RTOL)
    except AssertionError:
        pass
    else:
        raise AssertionError("Negative control unexpectedly passed")
    assert torch.equal(broken_output, torch.full_like(broken_output, 26.0))

    # Formula in the FP8 discussion, before quantization: rotation preserves QK^T.
    q, k = torch.randn(2, 5, 4, dtype=torch.float64)
    rotation, _ = torch.linalg.qr(torch.randn(4, 4, dtype=torch.float64))
    compare("orthogonal_scores", (q @ rotation) @ (k @ rotation).T, q @ k.T)

    # A small ordinary FP32 case checks the prose's dtype distinction, not FP8.
    q, k, v = torch.randn(3, 11, 5)
    output, lse = blocked(q, k, v, 4, 3, True)
    ref_out, ref_lse, _ = dense_reference(q, k, v, True)
    compare("fp32_output", output, ref_out, atol=3e-6, rtol=2e-5)
    compare("fp32_lse", lse, ref_lse, atol=3e-6, rtol=2e-5)

    print(json.dumps({
        "status": "PASS", "torch": torch.__version__, "device": "cpu",
        "executed_markdown_blocks": 1, "matrix_cases": cases,
        "shapes": shapes, "tilings": tilings, "causal": [False, True],
        "numerical_examples": numbers, "maximum_absolute_errors": errors,
        "source_excerpt": source_evidence,
        "double_tolerance": {"atol": ATOL, "rtol": RTOL},
        "gradcheck": {"cases": 2, "eps": 1e-6, "atol": 2e-6, "rtol": 2e-4},
        "negative_control": {"observed": 26.0, "correct": 30.0, "rejected": True},
        "gpu_execution": "NOT RUN", "benchmark": "NOT RUN",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
