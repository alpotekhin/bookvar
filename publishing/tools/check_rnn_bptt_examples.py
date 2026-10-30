#!/usr/bin/env python3
"""Execute every published block and independently check the recurrent mechanisms."""
from pathlib import Path
from math import isclose, log, tanh
import hashlib
import re
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/04 RNN, LSTM и Seq2Seq/01 RNN и BPTT.md"
text = CHAPTER.read_text()
blocks = re.findall(r"\x60\x60\x60python\n(.*?)\n\x60\x60\x60", text, flags=re.S)
assert len(blocks) == 6, len(blocks)
namespaces = []
for number, block in enumerate(blocks, 1):
    namespace = {}
    exec(compile(block, f"{CHAPTER}:block{number}", "exec"), namespace)
    namespaces.append(namespace)
    print("PASS standalone published block", number)

# Complete neural LM, independently count each matrix and the target multiplicities.
n = namespaces[0]
model, docs = n["model"], n["docs"]
assert [p.numel() for p in model.parameters()] == [24, 12, 16, 4, 32, 8]
assert n["N"] == 11
floor = (3*(-2/3*log(2/3)-1/3*log(1/3))+2*log(2))/11
assert isclose(floor, .29962153327312085)
assert 8*3+3*3+3*3+3+8*3+8 == 77
assert 77-8*3 == 53
E = torch.randn(8,3,dtype=torch.float64)
h = torch.randn(3,dtype=torch.float64)
assert (E@h).shape == (8,)
assert isclose(tanh(.8*.65441326871168), .48042035097063157)
final = n["objective"]().item()
assert floor <= final < floor+.001, (floor, final)
assert n["continue_text"]([2, 3]) == ["бежит", "<EOS>"]
assert n["continue_text"]([2, 2, 3]) == ["<EOS>"]
with torch.no_grad():
    for document in docs:
        ids = document[:, :-1]
        full_logits, full_h = model(ids)
        first, h = model(ids[:, :2])
        last, h = model(ids[:, 2:], h)
        torch.testing.assert_close(torch.cat([first, last], 1), full_logits)
        torch.testing.assert_close(h, full_h)
        changed = ids.clone()
        changed[:, -1] = (changed[:, -1]+1) % 8
        logits_changed, _ = model(changed)
        torch.testing.assert_close(full_logits[:, :-1], logits_changed[:, :-1])
    pair = torch.tensor([[6, 2, 2, 3], [6, 2, 3, 0]])
    both, _ = model(pair)
    for i in range(2):
        single, _ = model(pair[i:i+1])
        torch.testing.assert_close(both[i:i+1], single)
    # Loss masking does not freeze the PAD-time hidden state.
    ids = torch.tensor([[6, 2, 3]])
    _, before_pad = model(ids)
    _, after_pad = model(torch.tensor([[0]]), before_pad)
    assert not torch.allclose(before_pad, after_pad)
print("PASS 96 parameters, 11 targets, full-prefix entropy bound, generation, causality, chunks, batches, PAD state")

n = namespaces[1]
assert isclose(n["manual_w"], .2533642027283495)
assert isclose(n["manual_v"], .5389161748923231)
def scalar_loss(v, w, bias=0.):
    h = 0.
    for x in [1., 0., 1.]:
        h = tanh(v*x+w*h+bias)
    return .5*h*h
eps = 1e-6
for value, fn in [
    (n["manual_w"], lambda e: scalar_loss(.5, .8+e)),
    (n["manual_v"], lambda e: scalar_loss(.5+e, .8)),
    (sum(n["delta"]), lambda e: scalar_loss(.5, .8, e)),
]:
    finite = (fn(eps)-fn(-eps))/(2*eps)
    assert abs(finite-value) < 1e-9, (finite, value)
assert n["after"] < n["before"]
print("PASS scalar BPTT, all shared contributions, finite differences, SGD loss reduction")

# Independently derive all gradients of a batched tanh RNN with CE at EVERY step.
torch.manual_seed(123)
dtype = torch.float64
B, T, d, m, V = 2, 4, 3, 4, 5
x = torch.randn(B, T, d, dtype=dtype, requires_grad=True)
wx = (.2*torch.randn(m, d, dtype=dtype)).requires_grad_()
wh = (.2*torch.randn(m, m, dtype=dtype)).requires_grad_()
bh = torch.randn(m, dtype=dtype, requires_grad=True)
wy = (.2*torch.randn(V, m, dtype=dtype)).requires_grad_()
by = torch.randn(V, dtype=dtype, requires_grad=True)
h0 = torch.randn(B, m, dtype=dtype, requires_grad=True)
target = torch.tensor([[1, 2, 3, 4], [4, 2, 1, 0]])
hs, logits = [h0], []
for t in range(T):
    h = torch.tanh(x[:, t]@wx.T+hs[-1]@wh.T+bh)
    hs.append(h)
    logits.append(h@wy.T+by)
O = torch.stack(logits, 1)
loss = F.cross_entropy(O.reshape(-1,V), target.reshape(-1), reduction="sum")
actual = torch.autograd.grad(loss, (x, wx, wh, bh, wy, by, h0))
with torch.no_grad():
    e = O.softmax(-1)-F.one_hot(target,V)
    dx = torch.zeros_like(x)
    dwx, dwh, dbh, dwy, dby = [torch.zeros_like(z) for z in (wx,wh,bh,wy,by)]
    future = torch.zeros(B,m,dtype=dtype)
    for t in range(T-1,-1,-1):
        g = e[:,t]@wy + future@wh
        delta = g*(1-hs[t+1].square())
        dwh += delta.T@hs[t]
        dwx += delta.T@x[:,t]
        dbh += delta.sum(0)
        dwy += e[:,t].T@hs[t+1]
        dby += e[:,t].sum(0)
        dx[:,t] = delta@wx
        future = delta
    dh0 = future@wh
    for got, expected in zip(actual,(dx,dwx,dwh,dbh,dwy,dby,dh0)):
        torch.testing.assert_close(got,expected,atol=1e-12,rtol=1e-12)
    J2 = torch.diag(1-hs[2][0].square())@wh
    J3 = torch.diag(1-hs[3][0].square())@wh
def two_steps(h):
    return torch.tanh(x[0,2].detach()@wx.detach().T+
        torch.tanh(x[0,1].detach()@wx.detach().T+h@wh.detach().T+bh.detach())@wh.detach().T+bh.detach())
jac = torch.autograd.functional.jacobian(two_steps, hs[1][0].detach())
torch.testing.assert_close(jac,J3@J2,atol=1e-12,rtol=1e-12)
assert not torch.allclose(J3@J2,J2@J3)
A=torch.diag(torch.tensor([2.,0.],dtype=dtype))
C=torch.diag(torch.tensor([0.,2.],dtype=dtype))
assert torch.linalg.matrix_norm(A,2) == 2 and torch.linalg.matrix_norm(C,2) == 2
assert torch.count_nonzero(C@A) == 0
assert isclose(1-tanh(3)**2,.0098660371654402)
print("PASS all-step CE BPTT: every parameter/input/h0 gradient and ordered noncommuting Jacobians")

n = namespaces[3]
assert isclose(n["full"][0],n["cut"][0])
assert isclose(n["full"][1],n["cut"][1])
assert n["cut"][2] < n["full"][2]
assert isclose(n["reset"][0],tanh(.5))
assert n["reset"][2] == 0
# Detach is not zero_grad: the parameter still receives a gradient in the new fragment.
assert n["cut"][2] > 0
# Gradient clipping is global, preserving direction; NaN raises with the shown flag.
p=nn.Parameter(torch.tensor([1.],dtype=dtype))
q=nn.Parameter(torch.tensor([2.],dtype=dtype))
p.grad=torch.tensor([3.],dtype=dtype);q.grad=torch.tensor([4.],dtype=dtype)
norm=nn.utils.clip_grad_norm_([p,q],1.,error_if_nonfinite=True)
assert norm == 5
torch.testing.assert_close(p.grad,torch.tensor([.6],dtype=dtype),atol=2e-7,rtol=0)
torch.testing.assert_close(q.grad,torch.tensor([.8],dtype=dtype),atol=2e-7,rtol=0)
assert isclose((p.grad/q.grad).item(),3/4)
p.grad.fill_(float("nan"))
try:
    nn.utils.clip_grad_norm_([p],1.,error_if_nonfinite=True)
except RuntimeError:
    pass
else:
    raise AssertionError("Nonfinite gradient must be rejected")
n = namespaces[4]
assert n["start"] == 3 and n["target"].shape == (3,)
assert all(torch.isfinite(p).all() for p in n["params"])
assert len(n["optimizer"].param_groups[0]["params"]) == 2
print("PASS detach vs reset, complete aligned TBPTT loop, clipping direction/norm/nonfinite guard")

# Pins are checked against all five actual publication images, not metadata alone.
archive = ROOT/"05 Источники/Originals/rnn-bptt-2026-10-21"
rows = [
    ("d2l-recurrent/unfolded-rnn.svg","d2l-unfolded-rnn.svg","6f21d86aca0cba957ccd93e6e5904beacde382fbb97849ceeb6f79cc30d0437f"),
    ("d2l-recurrent/rnn-bptt.svg","d2l-rnn-bptt.svg","edfd5bc723d46d6e5d6b6c18b7faf2652f284a7e24d78e20be7e5643315f2ae4"),
    ("rnn-bptt-2026-10-21/d2l-rnn.svg","d2l-rnn.svg","d19bd5d2bf26ec93f87cbcdc72ef1373971f184345a5b856f4d6896269405545"),
    ("rnn-bptt-2026-10-21/d2l-rnn-train.svg","d2l-rnn-train.svg","cf26c7c7dbfad4f8cb2a1f0876aa216327b006b52dda249c65a09f8d636bac46"),
    ("rnn-bptt-2026-10-21/d2l-truncated-bptt.svg","d2l-truncated-bptt.svg","1fe6798ddee72f020f49d97d1b8fa65bd684010f2217156d6da9ad11487e7ba9")
]
for publication, member, expected in rows:
    source = (archive/member).read_bytes()
    local = (ROOT/"00 Учебник/Assets/Figures/curated"/publication).read_bytes()
    assert source == local
    assert hashlib.sha256(local).hexdigest() == expected
print("PASS five original SVG byte copies and fixed source hashes")
# The accepted explanation is preserved outside one bounded addition and its date.
restored = re.sub(r"<!-- rnn-execution 2026-10-26 -->\n.*?<!-- /rnn-execution -->\n\n", "", text, flags=re.S)
restored = restored.replace("last_updated: 2026-10-26", "last_updated: 2026-10-21", 1)
assert hashlib.sha256(restored.encode()).hexdigest() == "d9cb27f7d76de93f41fe3f252ce1da1bcf53897197decbabaf257dbddfabaa25"

# Compare the printed chunked recurrence with an independently written loop,
# including incomplete final chunks, singleton axes and chunks longer than T.
n = namespaces[5]
recurrent = n["recurrent"]
cases = 0
for B in (1, 2, 3):
    for T in (1, 2, 4, 7):
        for d in (1, 3):
            for m in (1, 4):
                torch.manual_seed(B*1000+T*100+d*10+m)
                X = torch.randn(B,T,d,dtype=dtype,requires_grad=True)
                wx = (.2*torch.randn(m,d,dtype=dtype)).requires_grad_()
                wh = (.2*torch.randn(m,m,dtype=dtype)).requires_grad_()
                bias = torch.randn(m,dtype=dtype,requires_grad=True)
                h0 = torch.randn(B,m,dtype=dtype,requires_grad=True)
                variables = (X,wx,wh,bias,h0)
                h = h0
                states = []
                for t in range(T):
                    h = torch.tanh(X[:,t]@wx.T+h@wh.T+bias)
                    states.append(h)
                H = torch.stack(states,1)
                expected_grads = torch.autograd.grad(H.square().sum(),variables)
                for F in range(1,T+2):
                    result = recurrent(*variables,chunk_size=F)
                    got_grads = torch.autograd.grad(result.square().sum(),variables)
                    torch.testing.assert_close(result,H,atol=1e-12,rtol=1e-12)
                    for got,expected in zip(got_grads,expected_grads):
                        torch.testing.assert_close(got,expected,atol=1e-12,rtol=1e-12)
                    cases += 1
                with torch.no_grad():
                    delta = torch.zeros_like(H)
                    future = torch.zeros(B,m,dtype=dtype)
                    for t in range(T-1,-1,-1):
                        delta[:,t] = (2*H[:,t]+future@wh)*(1-H[:,t].square())
                        future = delta[:,t]
                    previous = torch.cat([h0[:,None],H[:,:-1]],1)
                    dwx = delta.reshape(B*T,m).T@X.reshape(B*T,d)
                    dwh = delta.reshape(B*T,m).T@previous.reshape(B*T,m)
                    torch.testing.assert_close(dwx,expected_grads[1],atol=1e-12,rtol=1e-12)
                    torch.testing.assert_close(dwh,expected_grads[2],atol=1e-12,rtol=1e-12)
assert cases == 216

# Simulate the stated ideal dependency schedule, not concurrent CPU/GPU execution.
for L in range(1,9):
    for T in range(1,9):
        done = set()
        for slot in range(1,L+T):
            ready = [(layer,t) for layer in range(1,L+1) for t in range(1,T+1)
                     if layer+t-1 == slot]
            for layer,t in ready:
                assert layer == 1 or (layer-1,t) in done
                assert t == 1 or (layer,t-1) in done
            done.update(ready)
        assert len(done) == L*T and (L,T) in done
assert 32*1000*512*4/2**20 == 62.5
assert 32*20*512*4/2**20 == 1.25
source = (ROOT/"05 Источники/Originals/rnn-execution-2026-10-26/d2l-deep-rnn.md").read_text()
excerpt = text.split("<!-- source-excerpt: d2l-stacked-inputs -->\n",1)[1].split("\n<!-- /source-excerpt -->",1)[0]
excerpt = "\n".join(line.removeprefix("> ") for line in excerpt.splitlines())
assert " ".join(excerpt.split()) in " ".join(source.split())
assert hashlib.sha256((ROOT/"00 Учебник/Assets/Figures/curated/rnn-execution-2026-10-26/d2l-deep-rnn.svg").read_bytes()).hexdigest() == "518484464a1fd5fce1bde4fefdbbf68402fde18f62c460ee1ab680fe3965d8e0"
print("PASS accepted chapter preservation; 216 projection/gradient cases; 48 batched weight reductions; 64 dependency grids; source excerpt and original SVG")
print("ALL RNN/BPTT CHECKS PASSED")
