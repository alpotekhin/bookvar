#!/usr/bin/env python3
"""Execute chapter examples and check them against scalar math and pinned code."""
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
from types import ModuleType, SimpleNamespace
from contextlib import redirect_stdout
import ast
import hashlib
import io
import math
import re
import sys
import subprocess
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
chapter = ROOT/"00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура.md"
blocks = re.findall(r"~~~python\n(.*?)\n~~~", chapter.read_text(), re.S)
assert len(blocks) == 3
env = {}
for block in blocks:
    exec(compile(block, str(chapter), "exec"), env)
assert sum(v.numel() for v in env["p"].values()) == 46
assert abs(float(env["loss"].detach())-2.4054769191515373) < 1e-12
assert abs(float(env["new_loss"])-2.3664810981834323) < 1e-12
assert env['processed'] == env['cache']['k'].shape[1] == 4
assert env['sequence'].shape[1] == 5
assert env['sequence'].tolist() == [[0, 1, 1, 1, 1]]
print('PASS published generation loop: four processed/cached positions, five token IDs')

# Recreate the explicitly given initial state, before the optimizer mutates it.
e = {}
with redirect_stdout(io.StringIO()):
    exec(blocks[1].split("optimizer = ")[0], e)
weights = {k:v.detach().numpy().copy() for k,v in e["p"].items()}
ids = e["ids"].numpy()
targets = e["targets"].numpy()

def scalar_norm(x, gain):
    return np.array([[[
        float(v)*float(g)/math.sqrt(math.fsum(float(a)**2 for a in row)/len(row))
        for v,g in zip(row,gain)] for row in batch] for batch in x])

def scalar_forward(w, token_ids):
    x = w["E"][token_ids]
    r = scalar_norm(x, w["norm1"])
    q0, k0, v = r@w["q"], r@w["k"], r@w["v"]
    def rotated(a):
        out = np.zeros_like(a)
        for b in range(a.shape[0]):
            for t in range(a.shape[1]):
                c = complex(float(a[b,t,0]), float(a[b,t,1])) * complex(math.cos(t),math.sin(t))
                out[b,t] = (c.real,c.imag)
        return out
    q, k = rotated(q0), rotated(k0)
    a = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for b in range(x.shape[0]):
        for t in range(x.shape[1]):
            scores = [math.fsum(float(u)*float(z) for u,z in zip(q[b,t],k[b,j]))/math.sqrt(2)
                      for j in range(t+1)]
            maximum = max(scores)
            exps = [math.exp(s-maximum) for s in scores]
            a[b,t,:t+1] = np.array(exps)/math.fsum(exps)
    h = x+(a@v)@w["o"]
    z = scalar_norm(h, w["norm2"])
    gate, up = z@w["gate"], z@w["up"]
    ff = (gate/(1+np.exp(-gate))*up)@w["down"]
    y = h+ff
    logits = scalar_norm(y,w["final"])@w["U"]
    return logits,dict(a=a,h=h,z=z,ff=ff,y=y)

np_logits,np_trace = scalar_forward(weights,ids)
np.testing.assert_allclose(e["logits"].detach().numpy(),np_logits,atol=1e-12,rtol=1e-12)
for key,value in np_trace.items():
    np.testing.assert_allclose(e["trace"][key].detach().numpy(),value,atol=1e-12,rtol=1e-12)

def scalar_loss(w):
    logits,_ = scalar_forward(w,ids)
    loss = []
    for b in range(logits.shape[0]):
        for t in range(logits.shape[1]):
            row = logits[b,t]
            m = max(row)
            loss.append(m+math.log(math.fsum(math.exp(float(v)-m) for v in row))
                        -float(row[targets[b,t]]))
    return math.fsum(loss)/len(loss)

assert abs(scalar_loss(weights)-float(e["loss"].detach())) < 1e-12
e["loss"].backward()
max_error = 0.
step = 1e-6
for name,value in weights.items():
    for index in np.ndindex(value.shape):
        original = value[index]
        value[index] = original+step
        plus = scalar_loss(weights)
        value[index] = original-step
        minus = scalar_loss(weights)
        value[index] = original
        observed = float(e["p"][name].grad[index])
        max_error = max(max_error,abs((plus-minus)/(2*step)-observed))
assert max_error < 2e-8, max_error
assert e["p"]["E"].data_ptr() != e["p"]["U"].data_ptr()
print("PASS independent scalar states, NLL and all 46 parameter gradients;",max_error)

# Preserve function bodies from the official release; do not instantiate CUDA caches
# or replace its inference-only model with an alleged training implementation.
source = ROOT/"05 Источники/Originals/full-transformer-2026-10-21/llama1-model.py"
tree = ast.parse(source.read_text())
selected = [n for n in tree.body if getattr(n,"name",None) in
            {"ModelArgs","RMSNorm","precompute_freqs_cis","reshape_for_broadcast","apply_rotary_emb"}]
module = ModuleType("bookvar_llama1_reference")
module.__dict__.update(torch=torch,nn=nn,dataclass=dataclass,Optional=Optional,Tuple=Tuple)
sys.modules[module.__name__] = module
exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),"exec"),module.__dict__)
assert module.ModelArgs().norm_eps == 1e-5
assert module.ModelArgs().max_seq_len == 1024
torch.manual_seed(13)
for dtype in [torch.float32,torch.float16,torch.bfloat16,torch.float64]:
    x = torch.randn(2,3,8).to(dtype)
    g = torch.randn(8).to(dtype)
    ref = module.RMSNorm(8,eps=1e-6).to(dtype)
    ref.weight.data.copy_(g)
    actual = env["rmsnorm"](x,g)
    torch.testing.assert_close(actual,ref(x))
assert env["rmsnorm"](torch.ones(1,2,dtype=torch.float16),
                      torch.ones(2,dtype=torch.float32)).dtype == torch.float32
big = torch.tensor([[60000.,30000.]],dtype=torch.float16)
assert torch.isfinite(env["rmsnorm"](big,torch.ones(2,dtype=torch.float16))).all()
assert not torch.isfinite(big.square()).all()
assert torch.equal(env["rmsnorm"](torch.zeros(1,2),torch.ones(2)),torch.zeros(1,2))
small = torch.full((1,2),1e-6)
assert not torch.allclose(env["rmsnorm"](small,torch.ones(2)),
                          env["rmsnorm"](small*10,torch.ones(2)))

q = torch.randn(2,3,4,8)
k = torch.randn_like(q)
freqs = module.precompute_freqs_cis(8,16)[4:7]
qr,kr = module.apply_rotary_emb(q,k,freqs)
def pair_rotation(x, positions, theta=10000.):
    y = x.double().clone()
    for t,pos in enumerate(positions):
        for j in range(x.shape[-1]//2):
            angle = pos*theta**(-2*j/x.shape[-1])
            a,b = x[:,t,:,2*j].double(),x[:,t,:,2*j+1].double()
            y[:,t,:,2*j] = a*math.cos(angle)-b*math.sin(angle)
            y[:,t,:,2*j+1] = a*math.sin(angle)+b*math.cos(angle)
    return y
torch.testing.assert_close(qr.double(),pair_rotation(q,[4,5,6]),atol=2e-6,rtol=2e-6)
torch.testing.assert_close(kr.double(),pair_rotation(k,[4,5,6]),atol=2e-6,rtol=2e-6)
qa = pair_rotation(q,[1,2,3])
ka = pair_rotation(k,[8,9,10])
qb = pair_rotation(q,[8,9,10])
kb = pair_rotation(k,[15,16,17])
torch.testing.assert_close((qa*ka).sum(-1),(qb*kb).sum(-1),atol=1e-12,rtol=1e-12)
assert abs(math.cos(2)-(-0.4161468365471424)) < 1e-14
print("PASS original RMSNorm/RoPE bodies, dtypes, finite-epsilon limits and position shifts")

# The released FFN forward body is exercised with ordinary dense matrices.
ffn_cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="FeedForward")
ffn_forward = next(n for n in ffn_cls.body if isinstance(n,ast.FunctionDef) and n.name=="forward")
ffn_module = ModuleType("bookvar_llama1_ffn")
ffn_module.F = F
exec(compile(ast.Module(body=[ffn_forward],type_ignores=[]),str(source),"exec"),ffn_module.__dict__)
wg = torch.tensor([[0.,1.,1.],[1.,1.,-1.]],dtype=torch.float64)
wu = torch.tensor([[1.,-1.,0.],[-1.,2.,-.5]],dtype=torch.float64)
wd = torch.tensor([[1.,2.],[-1.,1.],[.5,-.5]],dtype=torch.float64)
layers = SimpleNamespace(w1=lambda x:x@wg,w3=lambda x:x@wu,w2=lambda x:x@wd)
x = torch.tensor([[1.,-1.]],dtype=torch.float64)
g,v = x@wg,x@wu
torch.testing.assert_close(g,torch.tensor([[-1.,0.,2.]],dtype=torch.float64))
torch.testing.assert_close(v,torch.tensor([[2.,-3.,.5]],dtype=torch.float64))
out = ffn_module.forward(layers,x)
products = [-2/(1+math.exp(1)),0.,1/(1+math.exp(-2))]
expected_ffn = torch.tensor([[products[0]+.5*products[2],
                              2*products[0]-.5*products[2]]],dtype=torch.float64)
torch.testing.assert_close(out,expected_ffn,atol=1e-12,rtol=1e-12)
assert not torch.allclose(out,(F.silu(v)*g)@wd)

# Causality and a one-token cache with absolute RoPE positions.
tokens = torch.tensor([[0,1,2,1]])
full,_ = e["forward"](tokens)
changed,_ = e["forward"](torch.tensor([[0,2,1,2]]))
torch.testing.assert_close(full[:,0],changed[:,0],atol=0,rtol=0)
with torch.no_grad():
    p = e["p"]
    keys,values,outputs = [],[],[]
    for pos in range(tokens.shape[1]):
        x = p["E"][tokens[:,pos:pos+1]]
        r = e["rms"](x,p["norm1"])
        def at_position(z):
            a,b = z[...,0],z[...,1]
            return torch.stack((a*math.cos(pos)-b*math.sin(pos),
                                a*math.sin(pos)+b*math.cos(pos)),dim=-1)
        q,k = at_position(r@p["q"]),at_position(r@p["k"])
        keys.append(k); values.append(r@p["v"])
        a = (q@torch.cat(keys,1).transpose(-2,-1)/math.sqrt(2)).softmax(-1)
        h = x+(a@torch.cat(values,1))@p["o"]
        z = e["rms"](h,p["norm2"])
        y = h+(F.silu(z@p["gate"])*(z@p["up"]))@p["down"]
        outputs.append(e["rms"](y,p["final"])@p["U"])
    torch.testing.assert_close(torch.cat(outputs,1),full,atol=1e-12,rtol=1e-12)
# The early source's square mask does not cover a multi-token cached suffix.
old,new = 3,2
old_mask = torch.triu(torch.full((new,new),-torch.inf),diagonal=old+1)
assert old_mask.shape != (new,old+new)
correct = torch.cat([torch.zeros(new,old),torch.triu(torch.full((new,new),-torch.inf),1)],1)
assert torch.isfinite(correct).tolist() == [[True,True,True,True,False],[True,True,True,True,True]]
print("PASS three-projection SwiGLU, causal prefix and sequential cache")

# Exercise the newly published cached forward, including a multi-token suffix.
# A reset position still has legal tensor shapes: compare values, not just shapes.
tokens = torch.tensor([[0, 1, 2, 1, 0]])
with torch.no_grad():
    expected, full_trace = e['forward'](tokens)
    for sizes in ([1, 1, 1, 1, 1], [2, 2, 1], [3, 2], [5]):
        cache, parts, offset = None, [], 0
        for size in sizes:
            logits, trace = e['forward'](tokens[:, offset:offset+size], cache)
            cache = trace['cache']
            parts.append(logits)
            offset += size
            assert cache['k'].shape == cache['v'].shape == (1, offset, 2)
        torch.testing.assert_close(torch.cat(parts, 1), expected, atol=1e-12, rtol=1e-12)
    _, prefix_trace = e['forward'](tokens[:, :3])
    suffix, _ = e['forward'](tokens[:, 3:], prefix_trace['cache'])
    torch.testing.assert_close(suffix, expected[:, 3:], atol=1e-12, rtol=1e-12)
    original_rope = e['rope']
    e['rope'] = lambda x, start_pos=0: original_rope(x, 0)
    wrong, _ = e['forward'](tokens[:, 3:], prefix_trace['cache'])
    e['rope'] = original_rope
    wrong_delta = (wrong - expected[:, 3:]).abs().max().item()
    assert wrong_delta > .01
    # A common causal prefix can be reused for a different suffix.
    alternate = torch.tensor([[0, 1, 2, 0, 2]])
    alt_full, _ = e['forward'](alternate)
    alt_cached, _ = e['forward'](alternate[:, 3:], prefix_trace['cache'])
    torch.testing.assert_close(alt_cached, alt_full[:, 3:], atol=1e-12, rtol=1e-12)
print('PASS all cached chunk partitions, shared causal prefix and reset-RoPE negative control;', wrong_delta)

# Run the unmodified Attention.forward body on CPU, replacing only its constructor
# and model-parallel projections with the same tiny dense matrices. This is not
# a claim that the CUDA/fairscale release was launched as a complete model.
attention_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Attention')
attention_forward = next(n for n in attention_cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
cpu_source = ModuleType('bookvar_llama1_attention')
cpu_source.__dict__.update(torch=torch, F=F, math=math, Optional=Optional,
                           apply_rotary_emb=module.apply_rotary_emb)
exec(compile(ast.Module(body=[attention_forward], type_ignores=[]), str(source), 'exec'), cpu_source.__dict__)
def holder():
    return SimpleNamespace(n_local_heads=1, head_dim=2,
        wq=lambda x:x@e['p']['q'], wk=lambda x:x@e['p']['k'],
        wv=lambda x:x@e['p']['v'], wo=lambda x:x@e['p']['o'],
        cache_k=torch.zeros(1, 8, 1, 2, dtype=torch.float64),
        cache_v=torch.zeros(1, 8, 1, 2, dtype=torch.float64))
with torch.no_grad():
    raw = e['p']['E'][tokens]
    normalized = e['rms'](raw, e['p']['norm1'])
    freqs = module.precompute_freqs_cis(2, 8)
    mask = torch.triu(torch.full((5, 5), -torch.inf), diagonal=1)
    whole_attention = cpu_source.forward(holder(), normalized, 0, freqs[:5], mask)
    torch.testing.assert_close(whole_attention, full_trace['h']-raw, atol=2e-6, rtol=2e-6)
    cached_source = holder()
    prefix_mask = torch.triu(torch.full((3, 3), -torch.inf), diagonal=1)
    cpu_source.forward(cached_source, normalized[:, :3], 0, freqs[:3], prefix_mask)
    suffix_mask = torch.cat((torch.zeros(2, 3), torch.triu(torch.full((2, 2), -torch.inf), 1)), 1)
    suffix_attention = cpu_source.forward(cached_source, normalized[:, 3:], 3, freqs[3:5], suffix_mask)
    torch.testing.assert_close(suffix_attention, whole_attention[:, 3:], atol=2e-6, rtol=2e-6)
    early_mask = torch.triu(torch.full((2, 2), -torch.inf), diagonal=3+1)
    try:
        cpu_source.forward(cached_source, normalized[:, 3:], 3, freqs[3:5], early_mask)
    except RuntimeError as err:
        assert 'size of tensor' in str(err)
    else:
        raise AssertionError('The historical square mask unexpectedly accepted the cached two-token suffix')
print('PASS actual pinned Attention.forward on CPU; rectangular-mask equivalence and historical-mask failure')

# Verify the paper columns themselves before testing their interpretation in prose.
pdf = ROOT/'05 Источники/Originals/full-transformer-2026-10-21/llama1-v1.pdf'
assert hashlib.sha256(pdf.read_bytes()).hexdigest() == '2e663675ae36ad12adb2f5a05281bac2747ecf8d23d92bedd9f937a89fee7136'
common = subprocess.check_output(['pdftotext', '-f', '4', '-l', '4', '-layout', str(pdf), '-'], text=True)
assert re.search(r'13B\s+78\.1\s+80\.1\s+50\.4\s+79\.2\s+73\.0\s+74\.8\s+52\.7\s+56\.4', common)
quant = subprocess.check_output(['pdftotext', '-f', '6', '-l', '6', '-layout', str(pdf), '-'], text=True)
assert re.search(r'62B\s+27\.6\s+43\.4\s+52\.4\s+68\.5', quant)
assert re.search(r'65B\s+10\.6\s+20\.5\s+50\.9\s+69\.7', quant)
bias = subprocess.check_output(['pdftotext', '-f', '9', '-l', '9', '-layout', str(pdf), '-'], text=True)
assert re.search(r'Gender\s+70\.6\s+62\.6\s+65\.7', bias)
assert 'Table 12: CrowS-Pairs' in bias and 'Higher score indicates higher bias' in bias
print('PASS paper columns: Table 3 ARC, Table 7 voting reversal, Table 12 gender bias direction')

def rounded_ffn(d):
    return 256*((int(8*d/3)+255)//256)
assert [rounded_ffn(d) for d in [4096,5120,6656,8192]] == [11008,13824,17920,22016]
assert 3*4096*11008 == 135266304
assert 8*4096**2 == 134217728
assert abs(135266304/134217728-1-.0078125) < 1e-14

# Count real module parameter shapes without allocating 6.7B weights.
with torch.device("meta"):
    parts = [nn.Embedding(32000,4096),nn.Linear(4096,32000,bias=False)]
    for _ in range(32):
        parts.extend([nn.Linear(4096,4096,bias=False) for _ in range(4)])
        parts.extend([nn.Linear(4096,11008,bias=False) for _ in range(2)])
        parts.append(nn.Linear(11008,4096,bias=False))
        parts.extend([module.RMSNorm(4096) for _ in range(2)])
    parts.append(module.RMSNorm(4096))
count = sum(p.numel() for part in parts for p in part.parameters())
assert count == 2*32000*4096+32*(4*4096**2+3*4096*11008+2*4096)+4096 == 6738415616
assert abs(count*2/2**30-12.551277160644531) < 1e-10
assert 2*32*2048*32*128*2 == 2**30
assert (2*32*2048*8*128*2)/(2*32*2048*32*128*2) == .25
assert abs(sum([.67,.15,.045,.045,.045,.025,.02])-1) < 1e-14
assert 1.4e12*.045 == 63e9 and 1e12*.045 == 45e9
assert abs(1.4e12/(380*2048)/86400-21) < .2
assert math.isclose(3e-4*.1,3e-5,rel_tol=1e-15)
print("PASS all sizes, parameter count, memory, mixture and training-budget arithmetic")
print("No released-weight quality, GPU throughput or full pretraining reproduction claim.")
print('CHAPTER_SHA256', hashlib.sha256(chapter.read_bytes()).hexdigest())
