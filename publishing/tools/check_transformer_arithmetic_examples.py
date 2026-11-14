"""CPU checks for S4 examples and its simplified communication model."""
import contextlib
from fractions import Fraction as Q
import hashlib
import io
from math import isclose, prod
from pathlib import Path
import re
import numpy as np

root = Path(__file__).resolve().parents[2]
path = root / "00 Учебник/10 ML Systems/04 Арифметика Transformer и MoE.md"
body = path.read_text()
blocks = re.findall(r"```python\n(.*?)\n```", body, re.S)
assert len(blocks) == 1
ns = {}
with contextlib.redirect_stdout(io.StringIO()) as output:
    exec(compile(blocks[0], path.name, "exec"), ns)
assert output.getvalue().strip() == "13.0 5.0"
np.testing.assert_array_equal(ns["Y"], [[5, 2], [11, 4]])
np.testing.assert_array_equal(ns["dX"], [[1, 4], [0, 1]])
np.testing.assert_array_equal(ns["dW"], [[1, 5], [2, 8]])
for name, other, grad in (("X", "W", "dX"), ("W", "X", "dW")):
    for index in np.ndindex(ns[name].shape):
        plus, minus = ns[name].copy(), ns[name].copy()
        plus[index] += 1e-5
        minus[index] -= 1e-5
        if name == "X":
            numeric = (ns["loss"](plus, ns[other]) - ns["loss"](minus, ns[other])) / 2e-5
        else:
            numeric = (ns["loss"](ns[other], plus) - ns["loss"](ns[other], minus)) / 2e-5
        assert isclose(numeric, ns[grad][index], rel_tol=1e-8, abs_tol=1e-8)

H, I, L, V, S = 4096, 11008, 32, 32000, 8192
attn, ffn = 4 * H * H, sum(prod(s) for s in ((H, I), (H, I), (I, H)))
pmat = L * (attn + ffn) + V * H
p = pmat + V * H + (2 * L + 1) * H
assert (attn, ffn, pmat, p) == (67108864, 135266304, 6607077376, 6738415616)
assert isclose(16 * p / 2**30, 100.41021728515625)
assert isclose(16 * p / 8 / 2**30, 12.551277160644531)
assert (2*S*H/2**20, 2*S*I/2**20, 2*32*S*S/2**30, 2*S*V/2**20) == (64,172,4,500)
assert 2 * 2 * L * S * 32 * 128 == 4 * 2**30
assert 2 * (attn + ffn) * Q(7,8) == 354156544
assert isclose(3 * 354156544 / 200e9 * 1000, 5.31234816)
assert 8192 * 28672 * 3 == 704643072
assert 3 * 256 * 7168 * 2048 == 11274289152
assert 2 * 11274289152 == 21 * 2**30
qwen = 3*128*4096*1536 + 2*4096*(64+4)*128 + 4096*128 + 2*4096 + 2*128
assert qwen == 2487755008
mla = 7168*1536 + 1536 + 1536*128*192 + 7168*(512+64) + 512 + 512*128*256 + 128*128*7168
deepseek = 3*256*7168*2048 + 3*7168*2048 + mla + 7168*256 + 256 + 2*7168
assert deepseek == 11507286272
assert 2*8192*7168*Q(7,8) == 102760448
assert 2*Q(8192,8)*8*7168*Q(7,8) == 102760448
assert Q(8-1,32+8-1) == Q(7,39)
assert Q(480,8*200)/Q(1,2) == Q(3,5)
assert Q(480+160,8*200)/Q(1,2) == Q(4,5)
assert Q(480,8*200)/Q(45,100) == Q(2,3)

# Compare direct byte/FLOP calculations with the derived ratios at every grid point.
N, R, W, b = 8192, 800*10**12, 200*10**9, 2
aF, aT = Q(2*N*W,b*R), Q(3*I*W,b*R)
assert (aF, aT) == (Q(2048,1000), Q(4128,1000))
for pf in range(1,17):
    for pt in range(1,17):
        tc = Q(6*N*H*I, pf*pt*R)
        tf = Q(3*b*H*I*(pf-1),pf*pt*W)
        tt = Q(2*b*N*H*(pt-1),pf*pt*W)
        assert tf/tc == Q(pf-1)/aF
        assert tt/tc == Q(pt-1)/aT
        assert (max(tf,tt) <= tc) == (pf-1 <= aF and pt-1 <= aT)
        assert (tf+tt <= tc) == (Q(pf-1)/aF + Q(pt-1)/aT <= 1)
assert max(Q(1)/aF,Q(3)/aT) < 1 < Q(1)/aF+Q(3)/aT

# The closed-form continuous optimum, including cases where it is on a boundary.
for af in (Q(1,10), Q(1), Q(2048,1000), Q(100)):
    for at in (Q(1,10), Q(1), Q(4128,1000), Q(100)):
        t = min(Q(1), max(Q(0), Q(1,2) + (af-at)/(2*af*at)))
        optimum = (1+af*t)*(1+at*(1-t))
        for j in range(1001):
            u=Q(j,1000)
            assert optimum >= (1+af*u)*(1+at*(1-u))

for formula in (r"\frac{2N_{\rm global}W}{bR}",r"\frac{3IW}{bR}",r"p-1\le6IW/(bR)",
                r"\frac12+\frac{a_{\rm F}-a_{\rm T}}{2a_{\rm F}a_{\rm T}}"):
    assert formula in body
assert body.count('id="cs336-systems-arithmetic"') == 1
assert len(re.findall(r"!\[\[02 Areas/ML & DL/",body)) == 10

# Independently sum causal pairs and derive the shapes of each matrix.
section = body.split('<a id="arith-attention-share"></a>', 1)[1].split('## 3.', 1)[0]
opt = {"350M": (1024, 4096), "1.3B": (2048, 8192)}
rows = re.findall(r"^\| OPT-(350M|1\.3B), (\d+) \| ([\d,]+) \| ([\d,]+) \| ([\d,]+)% \|$", section, re.M)
assert [(name, s) for name, s, *_ in rows] == [("350M", "2048"), ("1.3B", "2048"), ("350M", "1024")]
for name, s, displayed_linear, displayed_mix, displayed_share in rows:
    s = int(s)
    width, intermediate = opt[name]
    linear = 2 * s * sum(prod(shape) for shape in
                         [(width, width)] * 4 + [(width, intermediate), (intermediate, width)])
    mix = 4 * width * sum(range(1, s + 1))
    share = Q(mix, linear + mix)
    assert share == Q(s + 1, 12 * width + s + 1)
    for value, displayed in ((linear / 1e9, displayed_linear),
                             (mix / 1e9, displayed_mix), (share * 100, displayed_share)):
        assert f"{float(value):.2f}".replace(".", ",") == displayed
assert (60 + 40 // 5, 20 + 80 // 5) == (68, 36)
print("PASS actual NumPy block, eight finite differences, resource arithmetic, MFU/HFU, "
      "256 communication grids, 16 continuous optima and three OPT FLOP table rows; CPU only")
print("chapter_sha256=" + hashlib.sha256(path.read_bytes()).hexdigest())
