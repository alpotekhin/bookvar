"""Verify actual CPU programs and ordering. No CUDA, MXFP8 kernel or timing experiment."""
import ast
import contextlib
from fractions import Fraction
from hashlib import sha256
import inspect
import io
import json
from pathlib import Path
import re
import runpy
import textwrap
import torch

root = Path(__file__).resolve().parents[2]
path = root / "00 Учебник/10 ML Systems/05 Численные форматы и mixed precision.md"
body = path.read_text()
legacy = runpy.run_path(str(root / "publishing/tools/check_precision_examples.py"))
blocks = re.findall(r"(?:~~~|```)python\n(.*?)\n(?:~~~|```)", body, re.S)
assert len(blocks) == 4
for block in blocks:
    ast.parse(block)
cpu = [block for block in blocks if 'with torch.autocast("cuda"' not in block]
assert len(cpu) == 3
namespaces = []
for block in cpu:
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()) as output:
        exec(compile(block, path.name, "exec"), ns)
    namespaces.append(ns)
    print(output.getvalue().strip())
acc = next(ns for ns in namespaces if "scaler" in ns)
assert acc["theta"].item() == -1 and acc["theta"].grad.item() == 8
assert acc["scaler"].get_scale() == 128

# Reproduce the early-unscale error with a real enabled CPU GradScaler.
theta = torch.nn.Parameter(torch.tensor(1.0))
opt = torch.optim.SGD([theta], lr=.25)
scaler = torch.amp.GradScaler("cpu", init_scale=128.)
scaler.scale(.5 * theta).backward()
scaler.unscale_(opt)
assert theta.grad.item() == .5
scaler.scale(7.5 * theta).backward()
assert theta.grad.item() == 960.5
try:
    scaler.unscale_(opt)
except RuntimeError as error:
    assert "unscale_() has already been called" in str(error)
else:
    raise AssertionError("A second unscale must be rejected")
scaler.step(opt)
assert theta.item() == -239.125  # No second automatic division.
scaler.update()

# The same bad sum cannot be fixed by either of the two scale factors.
mixed = Fraction(1,2)*128 + Fraction(15,2)*64
assert mixed == 544 and mixed/64 == Fraction(17,2) and mixed/128 == Fraction(17,4)

# Compare actual clip ordering with the exact norm calculation at six scales.
for scale in [1., 2., 8., 16., 128., 1024.]:
    results = []
    for before in [False, True]:
        p = torch.nn.Parameter(torch.zeros(2))
        opt = torch.optim.SGD([p], lr=1.)
        scaler = torch.amp.GradScaler("cpu", init_scale=scale)
        scaler.scale((p * torch.tensor([3.,4.])).sum()).backward()
        if before:
            torch.nn.utils.clip_grad_norm_([p], 1.)
        scaler.unscale_(opt)
        if not before:
            torch.nn.utils.clip_grad_norm_([p], 1.)
        results.append(p.grad.clone())
        scaler.step(opt)
        scaler.update()
    expected = torch.tensor([.6,.8])
    torch.testing.assert_close(results[0], expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(results[1], expected/scale, atol=1e-6, rtol=1e-6)

# Nonfinite gradients skip SGD and reduce the loss scale.
p = torch.nn.Parameter(torch.tensor(1.))
opt = torch.optim.SGD([p], lr=1.)
scaler = torch.amp.GradScaler("cpu", init_scale=128.)
scaler.scale(p * float("inf")).backward()
scaler.step(opt)
scaler.update()
assert p.item() == 1. and scaler.get_scale() == 64.

# An explicit unscale records finite-status before later user mutations.
p = torch.nn.Parameter(torch.tensor(1.))
opt = torch.optim.SGD([p], lr=1.)
scaler = torch.amp.GradScaler("cpu", init_scale=128.)
scaler.scale(2*p).backward()
scaler.unscale_(opt)
p.grad.fill_(float("inf"))
scaler.step(opt)
scaler.update()
assert torch.isneginf(p).item() and scaler.get_scale() == 128.

mx = next(ns for ns in namespaces if "rowwise_roundtrip" in ns)
row, col = mx["rowwise"], mx["columnwise"]
expected_row = torch.ones(32,32)
expected_row[0] = 0.
expected_row[0,0] = 2**20
torch.testing.assert_close(row, expected_row, rtol=0, atol=0)
torch.testing.assert_close(col, expected_row.T, rtol=0, atol=0)
assert (row != col).sum().item() == 62
# Cross-check actual CPU E4M3 casts against every finite code and midpoint
# in the independently enumerated rational legacy codebook.
book = legacy["codebook"]
quantize = legacy["e4m3"]
values = [v for v,_ in book] + [(a+b)/2 for (a,_),(b,_) in zip(book,book[1:])]
for sign in [1, -1]:
    inputs = torch.tensor([float(sign*v) for v in values], dtype=torch.float32)
    restored = inputs.to(torch.float8_e4m3fn).float()
    expected = torch.tensor([float(quantize(sign*v)) for v in values])
    torch.testing.assert_close(restored, expected, rtol=0, atol=0)

# The CPU tests above exercise the same reviewed unscale/step/update functions.
pinned_path = root / "05 Источники/Originals/precision-accumulation-2026-11-04/grad_scaler.py"
if pinned_path.exists():
    tree = ast.parse(pinned_path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GradScaler")
    for name in ["unscale_", "step", "update"]:
        pinned = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
        actual = ast.parse(textwrap.dedent(inspect.getsource(getattr(torch.amp.GradScaler,name)))).body[0]
        # inspect.getsource/dedent changes the indentation inside docstrings only.
        for function in [pinned, actual]:
            doc = function.body[0].value
            assert isinstance(doc, ast.Constant) and isinstance(doc.value, str)
            doc.value = inspect.cleandoc(doc.value)
        assert ast.dump(pinned, include_attributes=False) == ast.dump(actual, include_attributes=False), name
    print("PASS installed GradScaler unscale_/step/update AST equals pinned source")
else:
    print("Pinned private implementation unavailable; runtime ordering checks still executed")
assert len(re.findall(r"!\[\[02 Areas/ML & DL/",body)) == 7
print("PASS three published CPU programs, early-unscale counterexample, six clipping scales, "
      "overflow skip, all32x32 MX orientation cells and 506 native E4M3 cast controls")
print("CUDA AMP program parsed, not executed; no hardware-packed MXFP8 or GEMM measured")
print("torch=" + torch.__version__)
print("chapter_sha256=" + sha256(path.read_bytes()).hexdigest())
