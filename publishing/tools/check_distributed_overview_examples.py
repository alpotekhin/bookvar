"""CPU arithmetic and the actual AMP fragment from chapter 44; no GPU claim."""
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
import re
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
REL = "00 Учебник/11 Pre-training и Scaling/44 Distributed training и mixed precision.md"
text = (ROOT / REL).read_text()
text = re.sub(r'<!-- BEGIN REVIEW: placement dimensions 2026-10-28 -->\n.*?<!-- END REVIEW: placement dimensions 2026-10-28 -->\n\n', '', text, flags=re.S)
before = subprocess.check_output(["git", "show", f"HEAD:{REL}"], cwd=ROOT, text=True)

def require(fragment):
    assert fragment in text, f"Chapter fixture changed: {fragment}"

# A loss mean and a gradient mean must have the same token weights.
require("$(2+18)/8=2{,}5$")
assert (F(2) + 18) / 8 == F(5, 2)
assert (F(2, 2) + F(18, 6)) / 2 == 2
targets = [[F(1), F(3)], [F(2), F(4), F(6), F(8), F(10), F(12)]]
w = F(5)
M = sum(map(len, targets))
reference = sum(w-y for batch in targets for y in batch) / M
locals_ = [F(2, M) * sum(w-y for y in batch) for batch in targets]
assert sum(locals_) / 2 == reference
naive = sum(sum(w-y for y in b) / len(b) for b in targets) / 2
assert naive != reference
# Splitting each rank's work into microbatches does not introduce /A again.
parts = [targets[0][:1], targets[0][1:], targets[1][:3], targets[1][3:]]
assert sum(F(2, M) * sum(w-y for y in b) for b in parts) / 2 == reference
print("PASS token-weighted loss/gradient and microbatch normalization")

require("$[11,22]$")
x0, x1 = [1, 2], [10, 20]
reduced = [a+b for a,b in zip(x0,x1)]
shards = [[reduced[0]], [reduced[1]]]
assert reduced == [11,22] == sum(shards, [])
assert x0+x1 == [1,2,10,20]
print("PASS all-reduce, all-gather, reduce-scatter example")

def mm(a,b):
    return [[sum(u*v for u,v in zip(row,col)) for col in zip(*b)] for row in a]
X = [[1,2,3,4], [5,6,7,8]]
W = [[1,2,3,4,5,6], [2,0,1,3,2,1], [4,1,0,1,0,2], [1,1,2,2,3,3]]
full = mm(X,W)
col_left, col_right = mm(X,[r[:3] for r in W]), mm(X,[r[3:] for r in W])
assert [a+b for a,b in zip(col_left,col_right)] == full
row_left, row_right = mm([r[:2] for r in X],W[:2]), mm([r[2:] for r in X],W[2:])
assert [[a+b for a,b in zip(r,s)] for r,s in zip(row_left,row_right)] == full
print("PASS column concatenation vs row summation")

# Execute the visible two-matrix example, not a parallel hidden implementation.
blocks = re.findall(r"```python\n(.*?)\n```", text, re.S)
assert len(blocks) == 2
tp_block = next(block for block in blocks if "import torch" in block)
ns, output = {}, io.StringIO()
with redirect_stdout(output):
    exec(compile(tp_block, REL + ":two-matrix-tp", "exec"), ns)
expected = "\n".join(re.findall(r"^# (.+)$", tp_block, re.M))
assert output.getvalue().strip() == expected
assert ns["y"].item() == ns["split_y"].item() == 42
assert ns["loss"].item() == ns["split_loss"].item() == 2
assert [p.item() for p in ns["parts"]] == [9, 33]
assert [p.grad.tolist() for p in ns["xs"]] == [[[10, 4]], [[50, 8]]]
assert ns["dx"].tolist() == [[60, 12]]
assert ns["a"].grad.tolist() == [[2, 4, 6, 8], [4, 8, 12, 16]]
assert ns["b"].grad.tolist() == [[2], [8], [6], [12]]
require("\\nabla_X\\mathcal L=[60\\ 12].")
require("$Z_0=1\\cdot1+4\\cdot2=9$")
require("$Z_1=3\\cdot3+6\\cdot4=33$")
# Negative controls: averaging output partials or dropping one input-gradient
# contribution changes this mathematical model; neither is a TP reduction.
assert sum(p.item() for p in ns["parts"])/2 == 21 != ns["y"].item()
assert not ns["torch"].equal(ns["xs"][0].grad, ns["x"].grad)
print("PASS actual CPU PyTorch TP block, full output/loss/dX/dA/dB and negative controls")

# The prose supplies all timings; this is a dependency calculation, not a timer.
require("три буфера готовы через 4, 8 и 12 ms")
require("каждая редукция занимает 3 ms")
require("с 12 до 21 ms")
require("4–7,\n8–11 и 12–15 ms")
ready, duration, end = [4, 8, 12], 3, 0
intervals = []
for time in ready:
    start = max(time, end)
    end = start + duration
    intervals.append((start, end))
assert intervals == [(4, 7), (8, 11), (12, 15)]
assert max(ready) + duration*len(ready) == 21
assert end - max(ready) == 3
require("должен включать и forward, и backward")
print("PASS overlap schedule, exposed tail and no_sync forward/backward condition")

require("$a,d$ эксперту $E_0$, а $b,c$ — эксперту $E_1$")
owners = {"a": 0, "b": 0, "c": 1, "d": 1}
experts = {"a": 0, "b": 1, "c": 1, "d": 0}
dispatch = [(token, owners[token], experts[token]) for token in owners]
assert [(t, a, b) for t, a, b in dispatch if a != b] == [("b", 0, 1), ("d", 1, 0)]
expert_batches = [[t for t in owners if experts[t] == rank] for rank in range(2)]
assert expert_batches == [["a", "d"], ["b", "c"]]
returned = [[t for batch in expert_batches for t in batch if owners[t] == rank]
            for rank in range(2)]
# Delivery order can differ; restore by the retained original token index.
assert [sorted(batch, key=list(owners).index) for batch in returned] == [["a", "b"], ["c", "d"]]
print("PASS expert dispatch/combine and restoration of original token order")

def two_dimensional_costs(b, d, h, f, w, p, q):
    compute = 6*b*d*h/(p*q*f)
    fsdp = 6*d*h*(q-1)/(p*q*w)
    tp = 4*b*d*(p-1)/(p*q*w)
    return compute, fsdp, tp

require("F_{\\mathrm{local}}=\\frac{6bdh}{pq}")
require("t_{\\mathrm{FSDP}}=\\frac{6dh(q-1)}{pqw}")
require("t_{\\mathrm{TP}}=\\frac{4bd(p-1)}{pqw}")
require("N=pq\\le(1+u)(1+v)")
require("\\frac{q-1}{u}+\\frac{p-1}{v}\\le1")
require("N\\le\\frac{(uv+u+v)^2}{4uv}")
b, h, w_power, f_coeff, f_power = re.search(
    r"пусть \$b=(\d+)\$, \$d=h=(\d+)\$, \$w=10\^\{(\d+)\}\$ байт/s,\s*"
    r"\$f=(\d+\{,\}\d+)\\cdot10\^\{(\d+)\}\$", text).groups()
b, d, h = int(b), int(h), int(h)
f, w = float(f_coeff.replace("{,}", "."))*10**int(f_power), 10**int(w_power)
u, v = b*w/f, 3*h*w/(2*f)
assert math.isclose(u, 6) and math.isclose(v, 6)
independent = two_dimensional_costs(b, d, h, f, w, 7, 7)
shared = two_dimensional_costs(b, d, h, f, w, 4, 4)
assert math.isclose(independent[0], max(independent[1:]))
assert math.isclose(shared[0], sum(shared[1:]))
assert sum(independent[1:]) > independent[0]
assert (1+u)*(1+v) == 49 and (u*v+u+v)**2/(4*u*v) == 16
for degree in [4, 7]:
    assert b % degree == d % degree == h % degree == 0
# Check the continuous maximum including boundary cases, rather than treating
# the unconstrained quadratic vertex as valid for every hardware/model shape.
for u in [.001, .1, 1, 6, 20, 1000]:
    for v in [.001, .1, 1, 6, 20, 1000]:
        p0 = (1+v+v/u)/2
        p = min(1+v, max(1, p0))
        q = 1+u-u*(p-1)/v
        best = p*q
        assert p >= 1 and q >= 1-1e-9
        assert math.isclose((q-1)/u+(p-1)/v, 1, abs_tol=1e-9)
        for i in range(101):
            candidate = 1+v*i/100
            count = candidate*(1+u-u*(candidate-1)/v)
            assert count <= best + 1e-7
        bound = (u*v+u+v)**2/(4*u*v) if 1 <= p0 <= 1+v else 1+max(u, v)
        assert math.isclose(best, bound, rel_tol=1e-8, abs_tol=1e-8)
print("PASS full 2D FSDP+TP capacity calculation, independent/shared links and boundary optima")

lines = text.splitlines()
start = next(i for i, line in enumerate(lines) if line.startswith("| GPU и длина"))
llama_rows = []
for line in lines[start+2:]:
    if not line.startswith("|"): break
    values = list(map(int, re.findall(r"\d+", line)))
    # GPU count, sequence length, TP, CP, PP, DP, MFU percent.
    assert len(values) == 7
    gpu, length, tp, cp, pp, dp, mfu = values
    assert tp*cp*pp*dp == gpu
    llama_rows.append(values)
assert llama_rows == [[8192, 8192, 8, 1, 16, 64, 43],
                      [16384, 8192, 8, 1, 16, 128, 41],
                      [16384, 131072, 8, 16, 16, 8, 38]]
assert 64*32*8192 == 128*16*8192 == 16777216
require("2048 H800, PP=16, EP=64")
print("PASS published Llama group sizes, short-context batch equality, DeepSeek training boundary")

require("112–126")
require("$14+2+2+48=66$")
require("$14+2+2+16=34$")
require("26,25 ms")
require("35 ms")
assert 7*(2+2+4+4+4) == 112
assert 7*(2+4+4+4+4) == 126
assert F(112,8) == 14
assert 14+2+2+48 == 66 and 14+2+2+16 == 34
q_bf16, q_fp32 = F(7,8)*(2*2+2), F(7,8)*(2*2+4)
assert q_bf16 == F(21,4) and q_fp32 == 7
assert q_bf16 / 200 * 1000 == F(105,4)
assert q_fp32 / 200 * 1000 == 35
require("3/11\\approx27{,}3")
require("3/35\\approx8{,}6")
assert round(100*3/11,1) == 27.3
assert round(100*3/35,1) == 8.6
assert 2*4*4*2 == 4*4*4 == 64
print("PASS memory, distinct collective dtypes, pipeline fractions, process groups")

# Run the chapter's actual fragment on mocks, including gradient scaling order.
amp_block = next(block for block in blocks if "optimizer.zero_grad" in block)
ast.parse(amp_block)
events = []
class Scalar:
    def __init__(self, value): self.value = F(value)
    def __truediv__(self, other): return Scalar(self.value / other)
    def backward(self): events.append(("backward", self.value))
class Autocast:
    def __init__(self, **kw):
        assert kw == {"device_type":"cuda", "dtype":"fp16"}
    def __enter__(self): events.append("autocast-enter")
    def __exit__(self,*args): events.append("autocast-exit")
class Model:
    def __call__(self, batch): return Scalar(batch)
    def parameters(self): return ["parameter"]
class Optimizer:
    def zero_grad(self, **kw):
        assert kw == {"set_to_none":True}
        events.append("zero")
class Scaler:
    def scale(self, loss):
        events.append("scale")
        return Scalar(loss.value*8)
    def unscale_(self, opt): events.append("unscale")
    def step(self, opt): events.append("step")
    def update(self): events.append("update")
def clip(parameters, max_norm):
    assert parameters == ["parameter"] and max_norm == 1
    events.append("clip")
ns = dict(optimizer=Optimizer(), microbatches=[2,6], model=Model(),
          scaler=Scaler(), training_dtype="fp16", max_norm=1,
          torch=SimpleNamespace(autocast=Autocast,
              nn=SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=clip))))
exec(compile(amp_block, REL, "exec"), ns)
assert events == ["zero", "autocast-enter", "autocast-exit", "scale", ("backward", F(8)),
                  "autocast-enter", "autocast-exit", "scale", ("backward", F(24)),
                  "unscale", "clip", "step", "update"]
print("PASS actual AMP fragment AST and mocked accumulation/unscale/clip/step order")

ids = re.findall(r"^  - ((?:lecture|assignment)-[^\n]+)", before, re.M)
assert all(i in text for i in ids) and len(ids) == 7
for anchor in re.findall(r'<a id="([^"]+)"', before):
    require(f'<a id="{anchor}"')
figures = re.findall(r'!\[\[02 Areas/ML & DL/(.*?)\]\]',text)
assert len(figures) == 9
assert all((ROOT/f).is_file() for f in figures)
records = json.loads((ROOT/"00 Учебник/Assets/Figures/curated/distributed-overview-2026-10-23/assets.json").read_text())
assert len(records) == 2
for record in records:
    for path, digest in [(record["asset"], record["sha256"]),
                         (record["source_asset"], record["source_asset_sha256"]),
                         (record["source_page_snapshot"], record["source_page_sha256"]),
                         (record["license_evidence"], record["license_evidence_sha256"])]:
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest
    original = (ROOT/record["source_asset"]).read_bytes()
    assert (ROOT/record["asset"]).read_bytes() == original
    svg = ET.fromstring(original)
    assert svg.tag.endswith("}svg")
    assert not any(e.tag.split("}")[-1] in {"script", "foreignObject"} for e in svg.iter())
    assert not re.search(r"\bon\w+=|(?:xlink:)?href=[\"']https?://", original.decode())
    require(record["asset"])
archive = ROOT/"05 Источники/Originals/distributed-overview-2026-10-23"
assert "license: apache-2.0" in (archive/"README.md").read_text()
assert (archive/"Apache-2.0.txt").read_bytes() == (ROOT/"site/public/licenses/ultrascale-Apache-2.0.txt").read_bytes()
print("PASS source IDs, explicit anchors, nine original figures and new SVG byte identity")
print("LIMITS: no CUDA, NCCL, distributed process, throughput, or GPU memory test")
print("Chapter SHA-256:", hashlib.sha256((ROOT/REL).read_bytes()).hexdigest())
