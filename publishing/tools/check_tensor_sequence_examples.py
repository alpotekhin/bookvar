"""Chapter 44c: CPU/NumPy reference arithmetic, layouts and finite differences.

Requires NumPy. All ranks are arrays in ONE process; this does not test a
distributed backend, NCCL, GPU kernels, throughput or peak device memory.
"""
from pathlib import Path
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
import re
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REL = "00 Учебник/11 Pre-training и Scaling/44c Tensor и sequence parallelism.md"
text = (ROOT / REL).read_text()
rng = np.random.default_rng(4403)


def close(a, b, *, atol=2e-10, rtol=2e-10):
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)


def finite_gradient(fn, value):
    """Central finite difference in float64; fn reads the perturbed array."""
    out = np.empty_like(value)
    step = 1e-6
    for ix in np.ndindex(value.shape):
        old = value[ix]
        value[ix] = old + step
        hi = fn()
        value[ix] = old - step
        lo = fn()
        value[ix] = old
        out[ix] = (hi - lo) / (2 * step)
    return out


def check_grads(loss, arrays, expected):
    for value, grad in zip(arrays, expected):
        close(finite_gradient(loss, value), grad, atol=3e-8, rtol=3e-6)


# Run the actual public example, not a separately maintained approximation.
blocks = re.findall(r"```python\n(.*?)\n```", text, re.S)
assert len(blocks) == 3
def execute_example(name):
    match = re.search(r"<!-- example:" + re.escape(name) + r" -->\s*```python\n(.*?)\n```", text, re.S)
    assert match, name
    code = match.group(1)
    ast.parse(code)
    namespace, stdout = {}, io.StringIO()
    with redirect_stdout(stdout):
        exec(compile(code, REL + "#" + name, "exec"), namespace)
    return namespace, stdout.getvalue()

ns, mlp_stdout = execute_example("tp-paired-mlp")
split_mlp = ns["split_mlp"]

x0 = np.array([[1., 2.]])
a0 = np.array([[1., 0., 2., -1.], [0., 1., 1., 2.]])
b0 = np.array([[1., 0.], [0., 1.], [1., 1.], [-1., 2.]])
close(x0 @ a0, [[1, 2, 4, 3]])
col = [x0 @ ar for ar in np.split(a0, 2, axis=1)]
close(np.concatenate(col, axis=1), x0 @ a0)
row = [xr @ ar for xr, ar in zip(np.split(x0, 2, axis=1), np.split(a0, 2))]
close(row[0], [[1, 0, 2, -1]])
close(row[1], [[0, 2, 2, 4]])
close(sum(row), x0 @ a0)
y_parts = [np.maximum(x0 @ ar, 0) @ br
           for ar, br in zip(np.split(a0, 2, axis=1), np.split(b0, 2))]
close(y_parts[0], [[1, 2]])
close(y_parts[1], [[1, 10]])
close(sum(y_parts), [[2, 12]])
g0 = np.ones((1, 2))
dx_parts = [(g0 @ br.T) @ ar.T
            for ar, br in zip(np.split(a0, 2, axis=1), np.split(b0, 2))]
close(dx_parts[0], [[1, 1]])
close(dx_parts[1], [[3, 4]])
close(sum(dx_parts), [[4, 5]])
bias = np.array([.3, -.7])
assert not np.allclose(sum(y + bias for y in y_parts), sum(y_parts) + bias)
assert not np.allclose(np.maximum(row[0], 0) + np.maximum(row[1], 0),
                       np.maximum(sum(row), 0))
print("PASS written matrix example, column/row split, backward [4,5], bias/nonlinearity traps")


def mlp_backward(x, a, b, g, p):
    aa, bb = np.split(a, p, axis=1), np.split(b, p)
    dxs, das, dbs = [], [], []
    for ar, br in zip(aa, bb):
        u = np.tanh(x @ ar)
        dz = (g @ br.T) * (1 - u * u)
        dxs.append(dz @ ar.T)
        das.append(x.T @ dz)
        dbs.append(u.T @ g)
    return sum(dxs), np.concatenate(das, axis=1), np.concatenate(dbs), dxs


x = rng.normal(size=(6, 4)) * .3
a = rng.normal(size=(4, 8)) * .3
b = rng.normal(size=(8, 4)) * .3
g = rng.normal(size=(6, 4))
dense = np.tanh(x @ a) @ b
dense_grad = mlp_backward(x, a, b, g, 1)[:3]
for p in (1, 2, 4):
    close(split_mlp(x, a, b, p), dense)
    grads = mlp_backward(x, a, b, g, p)[:3]
    for actual, expected in zip(grads, dense_grad):
        close(actual, expected)
    public_grads = ns["mlp_backward"](x, a, b, g, p)
    for actual, expected in zip(public_grads, dense_grad):
        close(actual, expected)
    check_grads(lambda: np.sum(split_mlp(x, a, b, p) * g), (x, a, b), public_grads)
assert not np.allclose(dense_grad[0] * 2, dense_grad[0])
try:
    split_mlp(x, a, b, 3)
except ValueError:
    pass
else:
    raise AssertionError("nondivisible TP must be rejected")
print("PASS actual chapter MLP forward AND backward: TP=1/2/4 and x/A/B finite differences")


def swiglu(x, wu, wg, wo, g, p):
    ys, dxs, dus, dgs, dos = [], [], [], [], []
    for ur, gr, outr in zip(np.split(wu, p, axis=1), np.split(wg, p, axis=1),
                           np.split(wo, p)):
        u, v = x @ ur, x @ gr
        sigmoid = 1 / (1 + np.exp(-v))
        silu = v * sigmoid
        h = u * silu
        dh = g @ outr.T
        du = dh * silu
        dv = dh * u * (sigmoid + v * sigmoid * (1 - sigmoid))
        ys.append(h @ outr)
        dxs.append(du @ ur.T + dv @ gr.T)
        dus.append(x.T @ du)
        dgs.append(x.T @ dv)
        dos.append(h.T @ g)
    return (sum(ys), (sum(dxs), np.concatenate(dus, axis=1),
                     np.concatenate(dgs, axis=1), np.concatenate(dos)))


wg = rng.normal(size=a.shape) * .2
base_y, base_grad = swiglu(x, a, wg, b, g, 1)
tp_y, tp_grad = swiglu(x, a, wg, b, g, 2)
close(tp_y, base_y)
for actual, expected in zip(tp_grad, base_grad):
    close(actual, expected)
check_grads(lambda: np.sum(swiglu(x, a, wg, b, g, 2)[0] * g),
            (x, a, wg, b), tp_grad)
print("PASS SwiGLU aligned columns and both input-gradient branches, finite differences")

# Explicit arrays for the transpose/adjoint of SP collectives.
# All-gather takes distinct row shards to replicas; its adjoint sums rank
# contributions, then scatters rows. Reduce-scatter is the opposite mapping.
shards = np.split(x, 2)
gathered = np.concatenate(shards)
partial_grads = [rng.normal(size=x.shape) for _ in range(2)]
rs_grads = np.split(sum(partial_grads), 2)
close(sum(np.sum(gathered * u) for u in partial_grads),
      sum(np.sum(u * v) for u, v in zip(shards, rs_grads)))
partials = [rng.normal(size=x.shape) for _ in range(2)]
output_shards = np.split(sum(partials), 2)
grad_shards = np.split(g, 2)
close(sum(np.sum(u * v) for u, v in zip(output_shards, grad_shards)),
      sum(np.sum(u * np.concatenate(grad_shards)) for u in partials))

# The MLP sees gathered X, while its output and input gradient stay row-sharded.
aa, bb = np.split(a, 2, axis=1), np.split(b, 2)
partial_y = [np.tanh(gathered @ ar) @ br for ar, br in zip(aa, bb)]
close(np.concatenate(np.split(sum(partial_y), 2)), dense)
sp_grad = mlp_backward(np.concatenate(shards), a, b,
                       np.concatenate(grad_shards), 2)
close(np.concatenate(np.split(sum(sp_grad[3]), 2)), dense_grad[0])
print("PASS SP all-gather/reduce-scatter adjoints and MLP input/output row placement")


def rms(x, gamma, eps=1e-5):
    return x * gamma / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def rms_grad(x, gamma, g, eps=1e-5):
    inv = 1 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    dx = g * gamma * inv - x * inv**3 * np.mean(g * gamma * x, axis=-1, keepdims=True)
    dg = np.sum(g * x * inv, axis=0)
    return dx, dg


gamma = rng.normal(size=x.shape[-1])
yrms = rms(x, gamma)
close(np.concatenate([rms(xx, gamma) for xx in np.split(x, 2)]), yrms)
dx, dg = rms_grad(x, gamma, g)
locals_ = [rms_grad(xx, gamma, gg) for xx, gg in zip(np.split(x, 2), np.split(g, 2))]
close(np.concatenate([l[0] for l in locals_]), dx)
close(sum(l[1] for l in locals_), dg)
assert not np.allclose(sum(l[1] for l in locals_) / 2, dg)
check_grads(lambda: np.sum(rms(x, gamma) * g), (x, gamma), (dx, dg))
wrong = np.concatenate([rms(xx, gm) for xx, gm in
                        zip(np.split(x, 2, axis=-1), np.split(gamma, 2))], axis=-1)
assert not np.allclose(wrong, yrms)
keep = .75
mask = (rng.random(x.shape) < keep) / keep
residual = rng.normal(size=x.shape)
close(np.concatenate([y * m + r for y, m, r in zip(np.split(yrms, 2),
                      np.split(mask, 2), np.split(residual, 2))]), yrms * mask + residual)
print("PASS RMSNorm full-D row shards, x/gamma finite differences, SUM not mean, dropout/residual")


def attention(q, k, v, *, causal=True, backward=None):
    """One batch, [heads, sequence, head_dim]; GQA handled before this function."""
    z = q @ k.swapaxes(-1, -2) / math.sqrt(q.shape[-1])
    if causal:
        z = np.where(np.arange(k.shape[1])[None, :] <= np.arange(q.shape[1])[:, None],
                     z, -np.inf)
    prob = np.exp(z - np.max(z, axis=-1, keepdims=True))
    prob /= np.sum(prob, axis=-1, keepdims=True)
    out = prob @ v
    if backward is None:
        return out
    dv = prob.swapaxes(-1, -2) @ backward
    dp = backward @ v.swapaxes(-1, -2)
    dz = prob * (dp - np.sum(dp * prob, axis=-1, keepdims=True))
    return out, (dz @ k / math.sqrt(q.shape[-1]),
                 dz.swapaxes(-1, -2) @ q / math.sqrt(q.shape[-1]), dv)


q = rng.normal(size=(4, 4, 2)) * .4
k = rng.normal(size=q.shape) * .4
v = rng.normal(size=q.shape)
go = rng.normal(size=q.shape)
ref, grad_ref = attention(q, k, v, backward=go)
check_grads(lambda: np.sum(attention(q, k, v) * go), (q, k, v), grad_ref)
# TP partitions heads then row-partitions the output projection.
wo = rng.normal(size=(8, 3))
head_out = [attention(qq, kk, vv) for qq, kk, vv in
            zip(np.split(q, 2), np.split(k, 2), np.split(v, 2))]
join_heads = lambda h: h.transpose(1, 0, 2).reshape(h.shape[1], -1)
close(sum(join_heads(h) @ w for h, w in zip(head_out, np.split(wo, 2))),
      join_heads(ref) @ wo)

# GQA: four query heads share two KV heads. TP=4 replicates each KV head
# on two ranks; gradients for replicas of the SAME head have to be summed.
kg, vg = k[:2].copy(), v[:2].copy()
krepeat, vrepeat = np.repeat(kg, 2, axis=0), np.repeat(vg, 2, axis=0)
gqa_ref, full_gqa_grad = attention(q, krepeat, vrepeat, backward=go)
gqa_out, dqs = [], []
dkg, dvg = np.zeros_like(kg), np.zeros_like(vg)
for h in range(4):
    kvhead = h // 2
    yy, (dq, dk, dv) = attention(q[h:h+1], kg[kvhead:kvhead+1],
                                 vg[kvhead:kvhead+1], backward=go[h:h+1])
    gqa_out.append(yy)
    dqs.append(dq)
    dkg[kvhead] += dk[0]
    dvg[kvhead] += dv[0]
close(np.concatenate(gqa_out), gqa_ref)
close(np.concatenate(dqs), full_gqa_grad[0])
close(dkg, full_gqa_grad[1].reshape(2, 2, 4, 2).sum(axis=1))
close(dvg, full_gqa_grad[2].reshape(2, 2, 4, 2).sum(axis=1))
check_grads(lambda: np.sum(attention(q, np.repeat(kg, 2, axis=0),
                                    np.repeat(vg, 2, axis=0)) * go),
            (q, kg, vg), (np.concatenate(dqs), dkg, dvg))
print("PASS MHA head/output TP, GQA replicated KV forward and correctly summed KV gradients")


def seq_to_heads(shards):
    """All-to-all simulation: send head slices, concatenate remote sequences."""
    c = len(shards)
    if shards[0].shape[0] % c:
        raise ValueError("equal Ulysses split requires divisible head count")
    sends = [np.split(local, c, axis=0) for local in shards]
    return [np.concatenate([sends[src][dst] for src in range(c)], axis=1)
            for dst in range(c)]


def heads_to_seq(shards):
    c = len(shards)
    sends = [np.split(local, c, axis=1) for local in shards]
    return [np.concatenate([sends[src][dst] for src in range(c)], axis=0)
            for dst in range(c)]


qparts, kparts, vparts = [seq_to_heads(np.split(t, 2, axis=1)) for t in (q, k, v)]
ulysses_head = [attention(qq, kk, vv) for qq, kk, vv in zip(qparts, kparts, vparts)]
close(np.concatenate(heads_to_seq(ulysses_head), axis=1), ref)
goheads = seq_to_heads(np.split(go, 2, axis=1))
head_grads = [attention(qq, kk, vv, backward=gg)[1]
              for qq, kk, vv, gg in zip(qparts, kparts, vparts, goheads)]
for i in range(3):
    close(np.concatenate(heads_to_seq([hh[i] for hh in head_grads]), axis=1), grad_ref[i])
close(np.concatenate(heads_to_seq(seq_to_heads(np.split(q, 2, axis=1))), axis=1), q)
print("PASS actual Ulysses sequence↔head permutation, forward and Q/K/V gradient placement")


def ring_attention(q, k, v, c, *, causal=True):
    """Streaming state with global query/key indices; permuted KV order per rank."""
    outputs = []
    width = q.shape[1] // c
    assert q.shape[1] % c == 0
    for rank in range(c):
        ix = np.arange(rank * width, (rank + 1) * width)
        qr = q[:, ix]
        m = np.full(qr.shape[:2] + (1,), -np.inf)
        ell = np.zeros_like(m)
        out = np.zeros_like(qr)
        for step in range(c):
            source = (rank - step) % c
            jx = np.arange(source * width, (source + 1) * width)
            z = qr @ k[:, jx].swapaxes(-1, -2) / math.sqrt(q.shape[-1])
            allowed = jx[None, :] <= ix[:, None] if causal else np.ones((width, width), bool)
            z = np.where(allowed, z, -np.inf)
            for h in range(q.shape[0]):
                for i in range(width):
                    valid = allowed[i]
                    if not np.any(valid):
                        continue
                    zz = z[h, i, valid]
                    new_m = max(float(m[h, i, 0]), float(np.max(zz)))
                    old_scale = math.exp(float(m[h, i, 0]) - new_m)
                    prob = np.exp(zz - new_m)
                    out[h, i] = old_scale * out[h, i] + prob @ v[h, jx[valid]]
                    ell[h, i, 0] = old_scale * ell[h, i, 0] + prob.sum()
                    m[h, i, 0] = new_m
        assert np.all(ell > 0)
        outputs.append(out / ell)
    return np.concatenate(outputs, axis=1)


for causal in (False, True):
    expected = attention(q, k, v, causal=causal)
    for c in (1, 2, 4):
        close(ring_attention(q, k, v, c, causal=causal), expected)
check_grads(lambda: np.sum(ring_attention(q, k, v, 2) * go), (q, k, v), grad_ref)
close(ring_attention(q * 100, k * 100, v, 2), attention(q * 100, k * 100, v))
assert np.isclose((math.exp(0)*0 + math.exp(math.log(3))*10) / 4, 7.5)
assert (0 + 10) / 2 != 7.5
print("PASS Ring online softmax: causal/noncausal c=1/2/4, empty blocks, large logits, gradients")

# Written byte totals and mesh/head constraints.
MiB = 1024**2
N = 8192 * 8192 * 2
assert N == 128 * MiB
ar, one_way = 2 * 7 * N // 8, 7 * N // 8
assert ar == 224 * MiB and one_way == 112 * MiB
assert 2 * ar == 4 * one_way == 448 * MiB
assert 2 * 32768 * 8192 * 2 == 1024**3
assert 2 * 4096 * 8192 * 2 == 128 * MiB
assert 28672 // 8 == 3584 and 32768 // 8 // 8 == 512
assert 64 // 8 == 8 and 8 // 8 == 1 and (8 // 8) % 8 != 0
zigzag = [[1, 2, 15, 16], [3, 4, 13, 14], [5, 6, 11, 12], [7, 8, 9, 10]]
assert [sum(r) for r in zigzag] == [34] * 4
assert [sum(range(r*4+1, r*4+5)) for r in range(4)] == [10, 26, 42, 58]
print("PASS communication bytes, memory, TP×CP head limits, original zigzag figure counts")

vocab_ns, vocab_stdout = execute_example("tp-vocab-ce")
vocab_ce = vocab_ns["vocab_cross_entropy"]
close(vocab_ns["loss"], 0.44018969856119533)
close(np.concatenate(vocab_ns["gradients"]),
      [0.03205860328008499, 0.08714431874203257, 0.23688281808991013, -0.35608574011202765])
for shift in (0., 1000., -1000.):
    logits = rng.normal(size=8) + shift
    for p in (1, 2, 4):
        shards = np.split(logits.copy(), p)
        for target in range(8):
            loss, gradients = vocab_ce(shards, target)
            exp = np.exp(logits - logits.max())
            expected = exp / exp.sum()
            expected[target] -= 1
            close(loss, np.log(exp.sum()) + logits.max() - logits[target])
            close(np.concatenate(gradients), expected)
            check_grads(lambda: vocab_ce(shards, target)[0], shards, gradients)
logits = rng.normal(size=8)
for target in range(8):
    ragged = [logits[:3], logits[3:]]
    close(vocab_ce(ragged, target)[0], vocab_ce([logits], target)[0])
for target in (-1, 8):
    try:
        vocab_ce([logits[:4], logits[4:]], target)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid target was accepted")
# Exact vocabulary-row embedding decomposition, including repeated tokens.
embedding = rng.normal(size=(8, 4))
ids = np.array([0, 3, 7, 0])
parts = []
for begin in (0, 4):
    local = np.zeros((len(ids), 4))
    owned = (begin <= ids) & (ids < begin + 4)
    local[owned] = embedding[begin:begin + 4][ids[owned] - begin]
    parts.append(local)
close(sum(parts), embedding[ids])
assert "0.440190" in text and "1.553002" in text
print("PASS actual vocabulary CE: every target, TP=1/2/4, large logit shifts, finite differences and embedding rows")

cost_ns, cost_stdout = execute_example("tp-communication")
T, D, F = (cost_ns[k] for k in ("tokens", "hidden", "intermediate"))
C, W, be = (cost_ns[k] for k in ("flops_per_second", "bytes_per_second", "element_bytes"))
close(1 + 3 * F * W / (be * C), 25)
close(1 + 6 * F * W / (be * C), 49)
assert cost_stdout.splitlines() == ["16 4.027 8.053 2.517", "32 2.013 4.027 2.6"]
for value in ("4.027 ms", "2.517 ms", "2.013 ms", "2.600 ms"):
    assert value in text
cp_tensor_bytes = 1 * 32768 * 32 * 128 * 2
assert cp_tensor_bytes == 256 * MiB
assert 3 * cp_tensor_bytes * 7 // 64 == 84 * MiB
assert cp_tensor_bytes * 7 // 64 == 28 * MiB
assert 4 * cp_tensor_bytes * 7 // 64 == 112 * MiB
assert 2 * cp_tensor_bytes * 7 // 8 == 448 * MiB
close(be * C / (2 * W), 1792)
for width, overlaps in ((1024, False), (4096, True)):
    calc = 4 * 4096 * width**2 / C
    send = 2 * 4096 * width * be / W
    assert (calc >= send) is overlaps
assert "1792 токена" in text and "блок на 1024 токена" in text
close(0.5 + 1.5, 2)
assert (0.5 + 1.5) / 2 != 2
close(10 / (1 / 3 + 1), 7.5)
print("PASS TP arithmetic limits 25/49, printed times, Ulysses/Ring 112/448 MiB and CP loss normalization")

s, batch, hidden, heads, tp = 2048, 2, 8192, 64, 8
unit = s * batch * hidden
quadratic = 5 * heads * s**2 * batch
saved_memory = [(34 * unit + quadratic) / MiB,
                (10 * unit + (24 * unit + quadratic) / tp) / MiB,
                (34 * unit + quadratic) / tp / MiB,
                34 * unit / tp / MiB]
table_text = text.split("| Схема для одного слоя | Сохранения, MiB |", 1)[1].split("\n\n", 1)[0]
displayed_memory = [int(v) for v in re.findall(r"\|\s*(\d+)\s*\|", table_text)]
assert displayed_memory == saved_memory == [3648, 736, 456, 136]
assert unit / MiB == 32 and quadratic / MiB == 2560
print("PASS source-model activation decomposition and all four memory-table rows")

# Reconstruct all four FSDP×TP logical coordinates: FSDP gathers only the
# TP-owned portion; this is an array model, not four distributed processes.
mesh_x = rng.normal(size=(12, 8)) * .2
mesh_weights = [rng.normal(size=s) * .2 for s in ((8, 16), (8, 16), (16, 8))]
mesh_go = rng.normal(size=(12, 8))
mesh_reference, mesh_grad = swiglu(mesh_x, *mesh_weights, mesh_go, 1)
mesh_outputs, mesh_grads = [], []
for fsdp in range(2):
    xx, gg = np.split(mesh_x, 2)[fsdp], np.split(mesh_go, 2)[fsdp]
    reconstructed = [[], [], []]
    for tp in range(2):
        for index, weight in enumerate(mesh_weights):
            tp_axis, fsdp_axis = (1, 0) if index < 2 else (0, 1)
            tp_piece = np.split(weight, 2, axis=tp_axis)[tp]
            stored = np.split(tp_piece, 2, axis=fsdp_axis)
            assert [s.shape for s in stored] == ([(4, 8)] * 2 if index < 2 else [(8, 4)] * 2)
            gathered = np.concatenate(stored, axis=fsdp_axis)
            assert gathered.shape == (8, 8)
            close(gathered, tp_piece)
            reconstructed[index].append(gathered)
    restored = [np.concatenate(parts, axis=1 if index < 2 else 0)
                for index, parts in enumerate(reconstructed)]
    yy, grads = swiglu(xx, *restored, gg, 2)
    assert yy.shape == (6, 8)
    mesh_outputs.append(yy)
    mesh_grads.append(grads)
close(np.concatenate(mesh_outputs), mesh_reference)
close(np.concatenate([g[0] for g in mesh_grads]), mesh_grad[0])
for index in (1, 2, 3):
    close(sum(g[index] for g in mesh_grads), mesh_grad[index])
print("PASS 2×2 FSDP/TP stored and gathered shapes, forward and all parameter-gradient sums")

before = subprocess.check_output(["git", "show", f"HEAD:{REL}"], cwd=ROOT, text=True)
ids = re.findall(r"^  - ((?:lecture|assignment)-[^\n]+)", before, re.M)
assert len(ids) == 5 and all(i in text for i in ids)
assert all(f'<a id="{a}"' in text for a in re.findall(r'<a id="([^"]+)"', before))
assert '<a id="shape-ledger-для-swiglu">' in text
assert '<a id="trade-offs-и-проверка">' in text
for value in ("224 MiB", "112 MiB", "448 MiB", "128 MiB", "$dX=[4,5]$", "$Y=[2,12]$"):
    assert value in text
figures = re.findall(r'!\[\[02 Areas/ML & DL/(.*?)\]\]', text)
assert len(figures) == 6
assert all((ROOT / f).is_file() for f in figures)
print("PASS retained source IDs/anchors, chapter fixtures and six original figure paths")
print(json.dumps({"numpy": np.__version__, "chapter_sha256": hashlib.sha256(text.encode()).hexdigest(),
                  "actual_python_fences": len(blocks), "figure_count": len(figures),
                  "mlp_stdout": mlp_stdout.strip(), "vocab_stdout": vocab_stdout.strip(),
                  "cost_stdout": cost_stdout.strip()}, ensure_ascii=False, indent=2))
print("LIMITS: NumPy in one CPU process; no distributed backend, CUDA, NCCL or performance measurement")
