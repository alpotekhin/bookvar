#!/usr/bin/env python3
"""Validate S8 prose examples on CPU; does not compile or execute Triton.

Run with a Python environment containing torch; repository paths resolve from this script.
These are CPU algorithm models, not executions of the chapter's Triton kernels.
"""

import ast
import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

p = Path(
    "00 Учебник/10 ML Systems/"
    "08 GPU kernels и Triton — от программы к измерению.md"
)
text = p.read_text()
blocks = re.findall(r"```python\n(.*?)\n```", text, re.S)
for code in blocks:
    ast.parse(code)
print("Python fenced blocks parsed:", len(blocks))

# Execute the self-contained CPU examples from the current chapter,
# rather than a separately copied implementation. Do not execute GPU fences.
cpu_scopes = {}
for marker in ("correct_block =", "def chunked_linear_ce(",
               "def replay_fixed_buffer():", "def launch_bound_finish("):
    selected = [code for code in blocks if marker in code]
    assert len(selected) == 1, (marker, len(selected))
    scope = {}
    exec(compile(selected[0], str(p) + ":" + marker, "exec"), scope)
    cpu_scopes[marker] = scope
print("Actual CPU chapter fences executed: padding softmax, chunked CE, buffer binding and launch queue")

alias_scope = cpu_scopes['def replay_fixed_buffer():']
assert alias_scope['output'] == alias_scope['aliased_result'] == [8] * 5
assert alias_scope['saved_result'] == [6] * 5
assert alias_scope['buffer'] == [9] * 5
assert alias_scope['captured_buffer'] == [4] * 5
assert alias_scope['buffer'] is not alias_scope['captured_buffer']

queue = cpu_scopes['def launch_bound_finish(']['launch_bound_finish']
queue_cases = 0
for count in range(1, 11):
    for launch in (1, 3, 10):
        for kernel in (1, 2, 11):
            closed_form = launch + kernel + (count - 1) * max(launch, kernel)
            assert queue(count, launch, kernel) == closed_form
            queue_cases += 1
for capture in range(0, 501, 10):
    first = capture // (52 - 20) + 1
    assert capture + first * 20 < first * 52
    assert not (capture + (first - 1) * 20 < (first - 1) * 52)
print('Buffer rebinding/output-alias negative controls; launch queue:', queue_cases,
      'cases; strict amortization boundaries: 51 passed (CPU models only)')

base = subprocess.check_output(["git", "show", "HEAD:" + str(p)], text=True)


def units(source):
    return re.search(
        r"source_unit_id:\n(.*?)primary_sources:", source, re.S
    ).group(1)


assert units(base) == units(text)
print(
    "Original source_unit_id preserved:",
    len(re.findall(r"^  - ", units(text), re.M)),
)

prose = re.sub(r"```[^\n]*\n.*?\n```", "", text, flags=re.S)
prose = re.sub(r"`[^`\n]*`", "", prose)  # inline array examples are not Obsidian links
links = re.findall(r"\[\[([^\]|]+)", prose)
missing = []
for raw in links:
    path = Path(raw.removeprefix("02 Areas/ML & DL/").split("#")[0])
    if not path.exists() and not path.with_suffix(".md").exists():
        missing.append(raw)
assert not missing, missing
print("Local wikilinks/assets resolved:", len(links))

trace = json.loads(
    Path(
        "05 Источники/Courses/Stanford CS336 Spring 2026/"
        "Lectures/repository/var/traces/lecture_06.json"
    ).read_text()
)
print("Trace top-level keys:", list(trace.keys()))

# CPU models check algorithms, not Triton compilation or device execution.
torch.manual_seed(8)

for n in range(1026):
    valid = [
        i
        for pid in range((n + 255) // 256)
        for i in range(pid * 256, (pid + 1) * 256)
        if i < n
    ]
    assert valid == list(range(n))
print("Vector-add indexing/masks: 1026 lengths passed")

softmax_cases = online_cases = 0
for n in (1, 2, 3, 5, 255, 256, 257, 4096):
    for x in (
        torch.full((n,), -5.0, dtype=torch.float64),
        torch.randn(n, dtype=torch.float64) * 30,
        torch.linspace(-1000, 1000, n, dtype=torch.float64),
    ):
        block = 1 << (n - 1).bit_length()
        padded = torch.full((block,), -torch.inf, dtype=torch.float64)
        padded[:n] = x
        exponentials = (padded - padded.max()).exp()
        actual = (exponentials / exponentials.sum())[:n]
        expected = x.softmax(0)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            (x + 1000).softmax(0), expected, rtol=1e-11, atol=1e-12
        )
        softmax_cases += 1

        for chunk in (1, 3, 4, 32, 256, 1024):
            m = torch.tensor(-torch.inf, dtype=torch.float64)
            den = torch.tensor(0.0, dtype=torch.float64)
            for values in x.split(chunk):
                new_m = torch.maximum(m, values.max())
                den = (m - new_m).exp() * den + (values - new_m).exp().sum()
                m = new_m
            torch.testing.assert_close(
                (x - m).exp() / den, expected, rtol=1e-12, atol=1e-12
            )
            online_cases += 1
print(
    "Padded/stable softmax cases:",
    softmax_cases,
    "; online recurrence cases:",
    online_cases,
)

worked = torch.tensor([4., 5., 7., 8.], dtype=torch.float64)
old_den = 1 + math.exp(-1)
old_rescaled = math.exp(-3) * old_den
new_den = old_rescaled + old_den
probs = (worked - 8).exp() / new_den
assert f"{old_den:.6f}" == "1.367879"
assert f"{old_rescaled:.6f}" == "0.068103"
assert f"{new_den:.6f}" == "1.435982"
assert [f"{v:.6f}" for v in probs] == [
    "0.012755", "0.034671", "0.256187", "0.696387"
]
torch.testing.assert_close(probs, worked.softmax(0), rtol=1e-12, atol=1e-12)
assert not math.isclose(((worked - 8).exp() / (2 * old_den)).sum().item(), 1)
print("Online rescaling worked example and missing-rescale negative control: passed")

values = torch.tensor(
    [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0],
    dtype=torch.float64,
)
acc = torch.zeros(4, dtype=torch.float64)
for chunk in values.split(4):
    acc[: chunk.numel()] += chunk
assert acc.tolist() == [13.0, 13.0, 6.0, 7.0] and acc.sum().item() == 39
print("Original row-sum figure numeric example: 39, passed")

matmul_cases = 0
for m, n, k in ((1, 1, 1), (3, 5, 7), (17, 19, 23), (64, 65, 33)):
    a = torch.randn(m, k, dtype=torch.float64)
    b = torch.randn(k, n, dtype=torch.float64)
    out = torch.zeros(m, n, dtype=torch.float64)
    for mi in range(0, m, 16):
        for nj in range(0, n, 16):
            acc = torch.zeros(16, 16, dtype=torch.float64)
            for ki in range(0, k, 8):
                ta = torch.zeros(16, 8, dtype=torch.float64)
                tb = torch.zeros(8, 16, dtype=torch.float64)
                ta[: min(16, m - mi), : min(8, k - ki)] = a[
                    mi : mi + 16, ki : ki + 8
                ]
                tb[: min(8, k - ki), : min(16, n - nj)] = b[
                    ki : ki + 8, nj : nj + 16
                ]
                acc += ta @ tb
            out[mi : mi + 16, nj : nj + 16] = acc[
                : min(16, m - mi), : min(16, n - nj)
            ]
    torch.testing.assert_close(out, a @ b, rtol=1e-12, atol=1e-12)
    matmul_cases += 1
print("Padded tiled GEMM CPU model:", matmul_cases, "shapes passed")

# Parse the matrices actually printed in the worked derivation.
worked_gemm = text.split("Числовой пример с двумя шагами по $K$:", 1)[1].split(
    "На одном таком шаге", 1
)[0]
matrices = []
for body in re.findall(r"\\begin\{bmatrix\}(.*?)\\end\{bmatrix\}", worked_gemm, re.S):
    matrices.append(torch.tensor([
        [float(v) for v in row.strip().split("&")]
        for row in body.strip().split(r"\\")
    ], dtype=torch.float64))
assert len(matrices) == 11
a, b, a0, b0, c0, a1, b1, c1, printed0, printed1, final = matrices
torch.testing.assert_close(a0 @ b0, c0)
torch.testing.assert_close(a1 @ b1, c1)
torch.testing.assert_close(c0, printed0)
torch.testing.assert_close(c1, printed1)
torch.testing.assert_close(c0 + c1, final)
torch.testing.assert_close(a @ b, final)
print("All 11 printed GEMM matrices and both partial products: passed")

chunked_linear_ce = cpu_scopes["def chunked_linear_ce("]["chunked_linear_ce"]
ce_cases = 0
for n, d, v in ((1, 1, 2), (3, 2, 4), (7, 5, 11), (13, 7, 17)):
    for ignore in (False, True):
        h = torch.randn(n, d, dtype=torch.float64, requires_grad=True)
        w = torch.randn(d, v, dtype=torch.float64, requires_grad=True)
        target = torch.randint(v, (n,))
        if ignore and n > 1:
            # Includes entire ignored chunks when chunk_size is 1 or 2.
            target[:n // 2] = -100
        reference = torch.nn.functional.cross_entropy(h @ w, target)
        ref_dh, ref_dw = torch.autograd.grad(reference, (h, w))
        for chunk in (1, 2, 3, 4, 8, n + 5):
            loss, dh, dw = chunked_linear_ce(h, w, target, chunk)
            for actual, expected in ((loss, reference), (dh, ref_dh), (dw, ref_dw)):
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
            ce_cases += 1
for target, chunk in ((torch.full((3,), -100), 2), (torch.tensor([0, 1, 3]), 0)):
    try:
        chunked_linear_ce(torch.ones(3, 2), torch.ones(2, 4), target, chunk)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid CE contract to be rejected")
print("Chunked CE loss, input and weight gradients vs autograd:", ce_cases, "cases passed")

ce_scope = cpu_scopes["def chunked_linear_ce("]
per_token = torch.nn.functional.cross_entropy(
    ce_scope["h"] @ ce_scope["w"], ce_scope["target"], reduction="none"
)
assert [f"{v:.6f}" for v in per_token] == ["0.626523", "0.626523", "2.820075"]
assert f"{per_token.mean():.6f}" == "1.357707"
assert f"{(per_token[:2].mean() + per_token[2]) / 2:.6f}" == "1.723299"
assert 32768 * 128000 * 2 == 8388608000
assert 8388608000 / 2**30 == 7.8125
assert 512 * 128000 * 2 == 131072000
assert 131072000 / 2**20 == 125
print("CE numeric walkthrough, wrong mean-of-means and memory estimates: passed")

# Arbitrary per-token upstream weights must enter before the weight reduction.
h, w, target = ce_scope["h"], ce_scope["w"], ce_scope["target"]
upstream = torch.tensor([1., 2., 5.], dtype=h.dtype)
weighted_loss = (
    torch.nn.functional.cross_entropy(h @ w, target, reduction="none") * upstream
).sum()
weighted_dh, weighted_dw = torch.autograd.grad(weighted_loss, (h, w))
g = (h @ w).softmax(1)
g[torch.arange(3), target] -= 1
g = g * upstream[:, None]
torch.testing.assert_close(g @ w.T, weighted_dh, rtol=1e-12, atol=1e-12)
torch.testing.assert_close(h.T @ g, weighted_dw, rtol=1e-12, atol=1e-12)
# Scaling a gradient reduced with equal token weights cannot recover these rows.
assert not torch.allclose(ce_scope["dw"] * upstream.sum(), weighted_dw)
print("Per-token upstream weighting before grad-weight reduction: passed")

for n in (1, 3, 7, 257):
    x = torch.randn(n, dtype=torch.float64, requires_grad=True)
    g = torch.randn_like(x)
    y = x.softmax(0)
    (grad,) = torch.autograd.grad((y * g).sum(), x)
    formula = y * (g - (g * y).sum())
    torch.testing.assert_close(formula, grad, rtol=1e-12, atol=1e-12)
print("Softmax gradient formula vs torch autograd: 4 cases passed")

y = torch.tensor([0.25, 0.75], dtype=torch.float64)
g = torch.tensor([2., 0.], dtype=torch.float64)
torch.testing.assert_close(y * (g - (y * g).sum()), torch.tensor(
    [0.375, -0.375], dtype=torch.float64
))
assert not torch.allclose(y * g, y * (g - (y * g).sum()))
print("Printed softmax backward example and missing-reduction negative control: passed")

assert (2 * 64 * 64 * 32) / (2 * (64 * 32 + 32 * 64)) == 32
assert ((256 + 127) // 128) * ((256 + 127) // 128) == 4
assert ((256 + 127) // 128) * ((257 + 127) // 128) == 6
print("Arithmetic intensity and tile-count examples: passed")
print("torch", torch.__version__, "; CUDA available:", torch.cuda.is_available())
print("Chapter SHA-256:", hashlib.sha256(p.read_bytes()).hexdigest())
print("No Triton compilation, CUDA execution or GPU performance claim is made by this check.")
