#!/usr/bin/env python3
"""Run the published self-attention examples and independent mechanism checks."""
from pathlib import Path
import hashlib
import json
import itertools
import math
import re
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V.md"
text = CHAPTER.read_text()
blocks = re.findall(r"\x60\x60\x60python\n(.*?)\n\x60\x60\x60", text, re.S)
assert len(blocks) == 4
namespaces = []
for i, code in enumerate(blocks, 1):
    ns = {}
    exec(compile(code, f"{CHAPTER}:block{i}", "exec"), ns)
    namespaces.append(ns)
    print("PASS standalone published block", i)

dtype = torch.float64
u = namespaces[0]
torch.testing.assert_close(u["out"], torch.tensor([
    [1.,0.], [.5,1.], [4/3,1.], [.75,1.5]
], dtype=dtype))
assert torch.equal(u["A_mean"].triu(1), torch.zeros(4,4,dtype=dtype))

# An exact finite probability space, not a noisy variance simulation.
d = 2
products = []
for values in itertools.product([-1.,1.], repeat=2*d):
    q, k = values[:d], values[d:]
    products.append(sum(a*b for a,b in zip(q,k)))
samples = torch.tensor(products, dtype=dtype)
assert samples.mean().item() == 0
assert samples.var(unbiased=False).item() == d
torch.testing.assert_close((samples/math.sqrt(d)).var(unbiased=False), torch.tensor(1.,dtype=dtype))
p = torch.tensor([0.,8.], dtype=dtype).softmax(0)[1].item()
assert math.isclose(p, .9996646498695336)
assert math.isclose(p*(1-p), .00033523767075636815)
print("PASS prefix averaging, exact variance and softmax saturation")

n = namespaces[2]
attention = n["attention"]
Q,K,V,A,Z = [n[key].detach() for key in ["Q","K","V","A","Z"]]
torch.testing.assert_close(Q, K)
torch.testing.assert_close(A[-1], torch.tensor([
    .24825507825772306,.24825507825772306,.5034898434845538
], dtype=dtype))
torch.testing.assert_close(Z[-1], torch.tensor([1.7587246087113844,1.],dtype=dtype))
assert not torch.allclose(A,A.T)
assert n["WK"] is not n["WQ"]
assert n["WK"].data_ptr() != n["WQ"].data_ptr()
for key in ["X","WQ","WK","WV"]:
    assert n[key].grad is not None and n[key].grad.abs().sum() > 0
assert n["after"] < n["before"]
assert math.isclose(n["before"],2.0465561246435064)
assert math.isclose(n["after"],1.8381402347944618)
torch.testing.assert_close(n["dS"][-1],torch.tensor([
    -.08301342987951799,-1.0161359017643923,1.0991493316439103
],dtype=dtype))
assert torch.count_nonzero(n["G"][:2]) == 0
torch.testing.assert_close(n["dA"][-1], torch.tensor([
    1.7587246087113846, -2., 4.276173826134154
], dtype=dtype))
weighted_signal = (A[-1] * n["dA"][-1]).sum()
assert math.isclose(weighted_signal.item(), 2.093112249287013)
torch.testing.assert_close(n["dS"].sum(-1), torch.zeros(3, dtype=dtype), atol=1e-14, rtol=0)
torch.testing.assert_close(n["dV"][0], torch.tensor([
    .4366123153694282, -.24825507825772308
], dtype=dtype))
torch.testing.assert_close(n["dQ"][-1], torch.tensor([
    .7185165867447092, .05869935919736113
], dtype=dtype))
assert torch.count_nonzero(n["dQ"][:2]) == 0
torch.testing.assert_close(n["dK"][0], torch.tensor([
    -.05869935919736113, -.05869935919736113
], dtype=dtype))
assert torch.all(n["dK"].norm(dim=-1) > 0)
assert torch.all(n["dV"].norm(dim=-1) > 0)
print("PASS backward worked stages: values, weights, softmax centering and query/key row support")

# Verify rows against the scalar definition; this also checks the softmax axis.
expected = []
for query in Q:
    weights = torch.stack([(query*key).sum() for key in K])/math.sqrt(Q.size(-1))
    expected.append(sum((w*value for w,value in zip(weights.softmax(0),V)),torch.zeros(V.size(-1),dtype=dtype)))
torch.testing.assert_close(Z, torch.stack(expected))
torch.testing.assert_close(Z, F.scaled_dot_product_attention(Q[None],K[None],V[None],dropout_p=0)[0])

torch.manual_seed(13)
q = torch.randn(2,3,4,dtype=dtype)
k = torch.randn(2,5,4,dtype=dtype)
v = torch.randn(2,5,6,dtype=dtype)
output, weights = attention(q,k,v)
assert output.shape == (2,3,6) and weights.shape == (2,3,5)
torch.testing.assert_close(output,F.scaled_dot_product_attention(q,k,v,dropout_p=0))
torch.testing.assert_close(output,torch.stack([attention(a,b,c)[0] for a,b,c in zip(q,k,v)]))
qg = torch.randn(1,2,3,dtype=dtype,requires_grad=True)
kg = torch.randn(1,4,3,dtype=dtype,requires_grad=True)
vg = torch.randn(1,4,2,dtype=dtype,requires_grad=True)
assert torch.autograd.gradcheck(lambda q,k,v: attention(q,k,v)[0],(qg,kg,vg))
print("PASS complete projected example, all backward paths, independent loops, SDPA and finite differences")

# Exercise the published projection/backward code on a non-identity input too.
dense_backward = blocks[2].split('print("weights:"')[0].replace(
    'X = torch.eye(3, dtype=dtype, requires_grad=True)',
    'X = torch.tensor([[2.,1.,0.],[0.,1.,3.],[1.,0.,2.]], dtype=dtype, requires_grad=True)')
assert dense_backward != blocks[2].split('print("weights:"')[0]
exec(compile(dense_backward,"dense-projection-backward","exec"),{})

# Q/K roles are directed; a query need not prefer its own position.
_, directed = attention(torch.ones(2,1,dtype=dtype),
    torch.tensor([[0.],[2.]],dtype=dtype),torch.eye(2,dtype=dtype))
assert directed[0,1] > directed[0,0]
# Changing values changes the output, but not the weights.
changed, same_weights = attention(Q,K,V+torch.tensor([1.,-2.],dtype=dtype))
torch.testing.assert_close(same_weights,A)
torch.testing.assert_close(changed,Z+torch.tensor([1.,-2.],dtype=dtype))
torch.testing.assert_close(attention(Q,K,2*V)[0],2*Z)

perm = torch.tensor([2,0,1])
X,WQ,WK,WV = [n[key].detach() for key in ["X","WQ","WK","WV"]]
base, wa = attention(X@WQ,X@WK,X@WV)
permuted, wap = attention(X[perm]@WQ,X[perm]@WK,X[perm]@WV)
torch.testing.assert_close(permuted,base[perm])
torch.testing.assert_close(wap,wa[perm][:,perm])
masked = u["A_mean"]@u["X"]
perm4 = torch.tensor([2,0,1,3])
assert not torch.allclose(u["A_mean"]@u["X"][perm4],masked[perm4])
assert X.size(1)*(2*WQ.size(1)+WV.size(1)) == 18
print("PASS value/content separation, non-self maximum and qualified permutation equivariance")

addition = json.loads((ROOT / "docs/audit/2026-10-29-attention-consolidation-additions.json").read_text())
record = next(row for row in addition["files"] if row["source"] == str(CHAPTER.relative_to(ROOT)))
assert hashlib.sha256(text.encode()).hexdigest() == record["after_sha256"]
restored = text
for change in reversed(record["changes"]):
    assert restored.count(change["new"]) == 1
    restored = restored.replace(change["new"], change["old"], 1)
assert hashlib.sha256(restored.encode()).hexdigest() == record["before_sha256"]
assert "transformer_self-attention_visualization.png" in text
restored = re.sub(r"<!-- attention-identifiability 2026-10-26 -->\n.*?<!-- /attention-identifiability -->\n\n", "", restored, flags=re.S)
restored = restored.replace("last_updated: 2026-10-26", "last_updated: 2026-10-22", 1)
assert hashlib.sha256(restored.encode()).hexdigest() == "d402a9a6810e8076173faf2e0b0e9043dbffa3673d7edb794e16e9c29f7ffd60"
n = namespaces[3]
assert torch.all(torch.stack(n["rows"]) > 0)
assert not torch.allclose(*n["rows"])
torch.testing.assert_close(n["delta"].sum(),torch.tensor(0.,dtype=dtype),atol=1e-15,rtol=0)
# All rows come from explicitly constructed rank-one QK scores, not arbitrary
# probability matrices assumed to be realizable by a fixed-width head.
for q_value in torch.linspace(-20,20,161,dtype=dtype):
    query = q_value.clone().requires_grad_()
    scores = torch.stack([query*0,query*0,query])
    row = scores.softmax(0)
    result = row@n["V"]
    torch.testing.assert_close(result,torch.full((2,),.5,dtype=dtype),atol=1e-15,rtol=0)
    slope, = torch.autograd.grad(result[0],query,retain_graph=True)
    assert abs(slope.item()) < 1e-15
    weight_slope, = torch.autograd.grad(row[-1],query)
    assert weight_slope > 0
print("PASS accepted chapter preservation; finite Q/K counterexample; 161 queries with equal outputs, changing weights and zero output derivative")
print("chapter_sha256",hashlib.sha256(CHAPTER.read_bytes()).hexdigest())
