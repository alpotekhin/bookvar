#!/usr/bin/env python3
"""Independent arithmetic, gradients and cache checks for the decoder chapter."""
from pathlib import Path
import ast
import contextlib
import copy
import hashlib
import io
import itertools
import json
import math
import re
from types import ModuleType
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT/"00 Учебник/07 Анатомия современной LLM/02 Современный decoder block.md"
blocks = re.findall(r"~~~python\n(.*?)\n~~~", CHAPTER.read_text(), re.S)
assert len(blocks) == 2
env = {}
for index, block in enumerate(blocks):
    exec(compile(block, f"{CHAPTER}:python-{index+1}", "exec"), env)
model, x = env["model"], env["x"]
assert sum(p.numel() for p in model.parameters()) == 128

def parameters(m):
    return {n: p.detach().numpy().copy() for n, p in m.named_parameters()}

def independent(inputs, p, nq=2, nkv=1, dh=2, theta=10000., window=None):
    """NumPy plus explicit per-head/per-position loops; no Torch forward calls."""
    bsz, t, d = inputs.shape
    norm = lambda a, gain: a/np.sqrt((a*a).mean(-1, keepdims=True)+1e-5)*gain
    r = norm(inputs, p["norm1.gain"])
    q = (r @ p["q.weight"].T).reshape(bsz,t,nq,dh)
    k = (r @ p["k.weight"].T).reshape(bsz,t,nkv,dh)
    v = (r @ p["v.weight"].T).reshape(bsz,t,nkv,dh)
    for arr, heads in [(q,nq),(k,nkv)]:
        for b in range(bsz):
            for i in range(t):
                for a in range(heads):
                    for j in range(dh//2):
                        angle = i*theta**(-2*j/dh)
                        u,w = arr[b,i,a,2*j:2*j+2].copy()
                        arr[b,i,a,2*j] = u*math.cos(angle)-w*math.sin(angle)
                        arr[b,i,a,2*j+1] = u*math.sin(angle)+w*math.cos(angle)
    scores = np.empty((bsz,nq,t,t))
    weights = np.zeros_like(scores)
    mixed = np.zeros((bsz,t,nq,dh))
    for b in range(bsz):
        for a in range(nq):
            group = a//(nq//nkv)
            for i in range(t):
                for j in range(t):
                    scores[b,a,i,j] = np.dot(q[b,i,a],k[b,j,group])/math.sqrt(dh)
                valid = [j for j in range(i+1) if window is None or j > i-window]
                row = scores[b,a,i,valid]
                probs = np.exp(row-row.max())
                probs /= probs.sum()
                weights[b,a,i,valid] = probs
                for j, prob in zip(valid,probs):
                    mixed[b,i,a] += prob*v[b,j,group]
    h = inputs+mixed.reshape(bsz,t,nq*dh)@p["o.weight"].T
    z = norm(h,p["norm2.gain"])
    gate,up = z@p["gate.weight"].T,z@p["up.weight"].T
    activated = gate/(1+np.exp(-gate))
    product = activated*up
    ff = product@p["down.weight"].T
    return h+ff, dict(q=q.transpose(0,2,1,3),k=k.transpose(0,2,1,3),
                     v=v.transpose(0,2,1,3),scores=scores,weights=weights,
                     mixed=mixed.transpose(0,2,1,3),h=h,z=z,gate=gate,up=up,
                     activated=activated,product=product,ff=ff,y=h+ff)

np_x=x.numpy().copy()
p=parameters(model)
ref, trace=independent(np_x,p)
actual,cache,observed=model(x)
for name, expected in trace.items():
    np.testing.assert_allclose(observed[name].detach().numpy(), expected,atol=2e-12,rtol=2e-12)
np.testing.assert_allclose(ref[0,-1],[2.130613,1.753410,2.743589,.454736],atol=5e-7,rtol=0)
local,local_trace=independent(np_x,p,window=2)
np.testing.assert_allclose(local[0,-1],[2.098376,1.807588,2.688386,.529815],atol=5e-7,rtol=0)
assert cache[0].numel()+cache[1].numel()==12
assert sum(a.numel()*a.element_size() for a in cache)==96
print("PASS independent per-position/per-head states and 128-parameter/96-byte accounting")

# Every published gate/up/product table cell must come from the actual example.
rows=re.findall(r"^\| ([1-6]) \| ([−\d.]+) \| ([−\d.]+) \| ([−\d.]+) \|$",
                CHAPTER.read_text(),re.M)
assert len(rows)==6
for row in rows:
    j=int(row[0])-1
    values=[float(v.replace("−","-")) for v in row[1:]]
    expected=[trace[n][0,-1,j] for n in ("gate","up","product")]
    np.testing.assert_allclose(values,expected,atol=5e-5,rtol=0)
assert trace["activated"][0,-1,0]<0 and trace["activated"][0,-1,5]>1
print("PASS all six published SwiGLU rows and non-probabilistic gate values")

# The derivative check uses a scalar probe on all output positions, not LM quality.
probe=torch.arange(12,dtype=torch.float64).reshape(1,3,4)/10-.35
xx=x.detach().clone().requires_grad_(True)
yy,_,_=model(xx)
objective=(yy*probe).sum()
model.zero_grad()
objective.backward()
np_probe=probe.numpy()
def objective_np(inputs, weights):
    return float((independent(inputs,weights)[0]*np_probe).sum())
worst=0.
step=1e-5
for name,value in p.items():
    expected=np.empty_like(value)
    for index in np.ndindex(value.shape):
        original=value[index]
        value[index]=original+step
        plus=objective_np(np_x,p)
        value[index]=original-step
        minus=objective_np(np_x,p)
        value[index]=original
        expected[index]=(plus-minus)/(2*step)
    measured=dict(model.named_parameters())[name].grad.numpy()
    worst=max(worst,float(np.max(np.abs(expected-measured))))
    np.testing.assert_allclose(measured,expected,atol=2e-8,rtol=2e-7)
input_grad=np.empty_like(np_x)
for index in np.ndindex(np_x.shape):
    original=np_x[index]
    np_x[index]=original+step
    plus=objective_np(np_x,p)
    np_x[index]=original-step
    minus=objective_np(np_x,p)
    np_x[index]=original
    input_grad[index]=(plus-minus)/(2*step)
np.testing.assert_allclose(xx.grad.numpy(),input_grad,atol=2e-8,rtol=2e-7)
worst=max(worst,float(np.max(np.abs(input_grad-xx.grad.numpy()))))
print("PASS all 128 parameter and 12 input finite-difference gradients; max abs",worst)

# Execute the second published block as a two-layer model, not two independent calls.
layers,through_stack=env["layers"],env["through_stack"]
assert len(layers)==2 and sum(p.numel() for p in layers.parameters())==256
assert all(a.data_ptr()!=b.data_ptr() for a,b in zip(layers[0].parameters(),layers[1].parameters()))
weights=[parameters(layer) for layer in layers]
def independent_stack(inputs, ps):
    h=inputs
    for item in ps:
        h,_=independent(h,item)
    return h
full,full_caches=through_stack(x)
np.testing.assert_allclose(full.detach().numpy(),independent_stack(x.numpy(),weights),atol=2e-12,rtol=2e-12)
np.testing.assert_allclose(full[0,-1].detach().numpy(),[3.149362,2.561249,3.351288,.944138],atol=5e-7,rtol=0)
_,prefix_caches=through_stack(x[:,:1])
np.testing.assert_allclose(prefix_caches[0][0].detach().flatten().numpy(),[1.4141994204496,0],atol=2e-12,rtol=0)
np.testing.assert_allclose(prefix_caches[1][0].detach().flatten().numpy(),[1.4760829005312175,.009013218699099736],atol=2e-12,rtol=0)
np.testing.assert_allclose(prefix_caches[1][1].detach().flatten().numpy(),[1.3438371583379616,-.12323252349415599],atol=2e-12,rtol=0)
assert sum(v.numel()*v.element_size() for c in full_caches for v in c)==192

def stack_gradients(bounds,detach=False):
    layers.zero_grad(set_to_none=True)
    inp=x.detach().clone().requires_grad_(True)
    cache=None
    outputs=[]
    for lo,hi in zip(bounds[:-1],bounds[1:]):
        if detach and cache is not None:
            cache=[tuple(v.detach() for v in c) for c in cache]
        out,cache=through_stack(inp[:,lo:hi],cache,start_pos=lo)
        outputs.append(out)
    result=torch.cat(outputs,dim=1)
    loss=result[:,-1].square().mean()
    loss.backward()
    return result.detach(),cache,inp.grad.clone(),[p.grad.clone() for p in layers.parameters()]

full_result,_,full_input_grad,full_param_grad=stack_gradients([0,3])
for cuts in itertools.product((False,True),repeat=2):
    bounds=[0]+[i+1 for i,cut in enumerate(cuts) if cut]+[3]
    result,caches,input_grad,param_grad=stack_gradients(bounds)
    torch.testing.assert_close(result,full_result,atol=2e-12,rtol=2e-12)
    torch.testing.assert_close(input_grad,full_input_grad,atol=2e-12,rtol=2e-12)
    for left,right in zip(param_grad,full_param_grad):
        torch.testing.assert_close(left,right,atol=2e-12,rtol=2e-12)
    for left,right in zip(caches,full_caches):
        for a,b in zip(left,right):
            torch.testing.assert_close(a,b,atol=2e-12,rtol=2e-12)

# Independently differentiate the two-layer example, including the stated prefix gradient.
def stack_objective_np(inputs,ps):
    return float(np.square(independent_stack(inputs,ps)[0,-1]).mean())
stack_input=x.numpy().copy()
stack_worst=0.
reference_input=np.empty_like(stack_input)
for idx in np.ndindex(stack_input.shape):
    original=stack_input[idx]
    stack_input[idx]=original+step
    plus=stack_objective_np(stack_input,weights)
    stack_input[idx]=original-step
    minus=stack_objective_np(stack_input,weights)
    stack_input[idx]=original
    reference_input[idx]=(plus-minus)/(2*step)
np.testing.assert_allclose(full_input_grad.numpy(),reference_input,atol=2e-8,rtol=2e-7)
stack_worst=float(np.max(np.abs(reference_input-full_input_grad.numpy())))
for layer,ps in enumerate(weights):
    for name,value in ps.items():
        # Compare with full-pass gradients, not the latest partition left on the modules.
        grad=full_param_grad[layer*len(ps)+list(ps).index(name)].numpy()
        expected=np.empty_like(value)
        for idx in np.ndindex(value.shape):
            original=value[idx]
            value[idx]=original+step
            plus=stack_objective_np(stack_input,weights)
            value[idx]=original-step
            minus=stack_objective_np(stack_input,weights)
            value[idx]=original
            expected[idx]=(plus-minus)/(2*step)
        np.testing.assert_allclose(grad,expected,atol=2e-8,rtol=2e-7)
        stack_worst=max(stack_worst,float(np.max(np.abs(grad-expected))))
np.testing.assert_allclose(full_input_grad[0,0].numpy(),[-.3164,.1146,.3253,.2356],atol=5e-5,rtol=0)

detached,_,detached_input_grad,detached_param_grad=stack_gradients([0,1,3],detach=True)
torch.testing.assert_close(detached,full_result,atol=2e-12,rtol=2e-12)
assert torch.count_nonzero(detached_input_grad[0,0])==0
assert torch.count_nonzero(full_input_grad[0,0])==4
detached_delta=max((a-b).abs().max().item() for a,b in zip(full_param_grad,detached_param_grad))
assert detached_delta>.4

# Same cache shape is insufficient: deliberately feed first-layer cache to the second layer.
with torch.no_grad():
    intermediate,first_layer_cache,_=layers[0](x[:,:1])
    suffix,_,_=layers[0](x[:,1:],start_pos=1,cache=first_layer_cache)
    wrong,_,_=layers[1](suffix,start_pos=1,cache=first_layer_cache)
    wrong_cache_delta=(wrong-full_result[:,1:]).abs().max().item()
assert wrong_cache_delta>.01
print("PASS two-layer independent states, all chunk partitions, 256 parameter/12 input finite-difference gradients; max abs",stack_worst)
print("PASS detached-cache same-forward/different-backward control; parameter delta",detached_delta)
print("PASS wrong-layer same-shape cache control; output delta",wrong_cache_delta)

torch.manual_seed(31)
cases=[(4,2,1,2),(4,4,2,2),(7,4,2,4),(6,3,3,2)]
partitions=0
for d,nq,nkv,dh in cases:
    m=env["DecoderBlock"](d=d,nq=nq,nkv=nkv,dh=dh,f=9).double()
    inp=torch.randn(2,5,d,dtype=torch.float64)
    for window in (None,1,2,4):
        y,c,tr=m(inp,window=window)
        reference,nt=independent(inp.numpy(),parameters(m),nq,nkv,dh,window=window)
        np.testing.assert_allclose(y.detach().numpy(),reference,atol=2e-12,rtol=2e-12)
        # Every composition of length 5, including all-token and cached multi-token chunks.
        for cuts in itertools.product((False,True),repeat=4):
            bounds=[0]+[i+1 for i,c in enumerate(cuts) if c]+[5]
            cache=None
            outs=[]
            for lo,hi in zip(bounds[:-1],bounds[1:]):
                out,cache,_=m(inp[:,lo:hi],start_pos=lo,cache=cache,window=window)
                outs.append(out)
            torch.testing.assert_close(torch.cat(outs,dim=1),y,atol=2e-12,rtol=2e-12)
            assert cache[0].shape==(2,nkv,5,dh)
            partitions+=1
        q,k,v=tr["q"],tr["k"],tr["v"]
        indices=torch.arange(5)
        allowed=indices[None,:]<=indices[:,None]
        if window is not None:
            allowed&=indices[None,:]>indices[:,None]-window
        sdpa=F.scaled_dot_product_attention(q,
                k.repeat_interleave(nq//nkv,1),v.repeat_interleave(nq//nkv,1),
                attn_mask=allowed,dropout_p=0.)
        torch.testing.assert_close(tr["mixed"],sdpa,atol=2e-12,rtol=2e-12)
        changed=inp.clone()
        changed[:,3:]+=4
        y2,_,_=m(changed,window=window)
        torch.testing.assert_close(y[:,:3],y2[:,:3],atol=0,rtol=0)
        assert not torch.allclose(y[:,3:],y2[:,3:])
print("PASS NumPy and SDPA over MHA/GQA, unequal inner width, 4 windows,",partitions,"chunk partitions")

# Source implementation is read, not imported with CUDA/fairscale side effects.
source=ROOT/"05 Источники/Originals/llama-2026-10-22/llama3-model.py"
tree=ast.parse(source.read_text())
names={"repeat_kv","precompute_freqs_cis","reshape_for_broadcast","apply_rotary_emb"}
module=ModuleType("bookvar_llama3_functions")
module.__dict__.update(torch=torch,Tuple=tuple)
selected=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
assert len(selected)==4
exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),"exec"),module.__dict__)
kv=torch.arange(2*3*2*4).reshape(2,3,2,4)
repeated=module.repeat_kv(kv,2)
assert repeated[0,0,:,0].tolist()==[0,0,4,4]
torch.testing.assert_close(repeated,kv.repeat_interleave(2,2))
assert not torch.equal(repeated,kv.repeat(1,1,2,1))
m=env["DecoderBlock"](d=8,nq=4,nkv=2,dh=4).double()
q=torch.randn(2,3,4,4)
k=torch.randn(2,3,2,4)
freq=module.precompute_freqs_cis(4,20,theta=10000.)[6:9]
qr,kr=module.apply_rotary_emb(q,k,freq)
oursq=m.rope(q.transpose(1,2).double(),torch.arange(6,9)).transpose(1,2)
oursk=m.rope(k.transpose(1,2).double(),torch.arange(6,9)).transpose(1,2)
torch.testing.assert_close(qr.double(),oursq,atol=2e-6,rtol=2e-6)
torch.testing.assert_close(kr.double(),oursk,atol=2e-6,rtol=2e-6)
print("PASS original pinned repeat_kv and RoPE bodies; no whole-CUDA-model execution claim")

# Verify actual failure modes, rather than only testing successful shapes.
m=env["DecoderBlock"](d=4,nq=4,nkv=2,dh=2).double()
inp=torch.randn(1,5,4,dtype=torch.float64)
out,_,tr=m(inp)
wrong=F.scaled_dot_product_attention(tr["q"],tr["k"].repeat(1,2,1,1),
           tr["v"].repeat(1,2,1,1),is_causal=True)
assert wrong.shape==tr["mixed"].shape
assert not torch.allclose(wrong,tr["mixed"])
joined_wrong=tr["mixed"].reshape(1,5,-1)
joined_right=tr["mixed"].transpose(1,2).contiguous().reshape(1,5,-1)
assert joined_wrong.shape==joined_right.shape
assert not torch.allclose(joined_wrong,joined_right)
_,cache,_=m(inp[:,:3])
badq=m.rope(
    m.q(m.norm1(inp[:,3:])).reshape(1,2,4,2).transpose(1,2),torch.arange(2))
correctq=m.rope(
    m.q(m.norm1(inp[:,3:])).reshape(1,2,4,2).transpose(1,2),torch.arange(3,5))
assert not torch.allclose(badq,correctq)
allowed=torch.arange(3)[None,:]<=torch.arange(1,3)[:,None]
assert allowed.tolist()==[[True,True,False],[True,True,True]]
assert not torch.equal(allowed,torch.ones(2,3,dtype=torch.bool).tril())
# window two can relay 0 -> 1 -> 2 through two layers.
a=np.tril(np.ones((5,5),dtype=int))-np.tril(np.ones((5,5),dtype=int),-2)
assert a[2,0]==0 and (a@a)[2,0]>0
print("PASS wrong grouping/reshape/RoPE and rectangular-mask counterexamples; two-layer local path")

# Check every configuration field used in the prose against archived pinned files.
archive=ROOT/"05 Источники/Originals/decoder-block-2026-10-22"
configs={n:json.loads((archive/(n+"-config.json")).read_text())
         for n in ("t5-v1_1-base","qwen15-14b","mistral7b-v01")}
assert (configs["t5-v1_1-base"]["d_model"],configs["t5-v1_1-base"]["d_ff"],
        configs["t5-v1_1-base"]["vocab_size"])==(768,2048,32128)
assert configs["t5-v1_1-base"]["feed_forward_proj"]=="gated-gelu"
assert configs["t5-v1_1-base"]["dropout_rate"]==.1
assert (configs["qwen15-14b"]["hidden_size"],configs["qwen15-14b"]["intermediate_size"],
        configs["qwen15-14b"]["vocab_size"])==(5120,13696,152064)
assert (configs["mistral7b-v01"]["hidden_size"],configs["mistral7b-v01"]["intermediate_size"],
        configs["mistral7b-v01"]["vocab_size"],configs["mistral7b-v01"]["sliding_window"])==(4096,14336,32000,4096)
qwen_tree=ast.parse((archive/"modeling_qwen.py").read_text())
cls=next(n for n in qwen_tree.body if isinstance(n,ast.ClassDef) and n.name=="QWenMLP")
source_text=ast.unparse(cls)
assert source_text.count("config.intermediate_size // 2")==3
original_qwen=json.loads((archive/"qwen14b-original-config.json").read_text())
assert original_qwen["intermediate_size"]==27392
assert original_qwen["intermediate_size"]//2==13696
t5=(ROOT/"05 Источники/Originals/t5-2026-10-22/bi_v1_11B.gin").read_text()
for line in ("d_model = 1024","num_layers = 24","d_ff = 65536","num_heads = 128","d_kv = 128"):
    assert line in t5
assert round(1024/24,1)==42.7
assert 128*128/1024==16 and 65536/1024==64
assert 2*32000*4096==262144000 and (100/80)**2==1.5625
assert math.isclose((1-.001*.1)*2,1.9998)
assert (1+1)+(1+1)**2==6 and 1+1+1**2==3
print("PASS config width/vocab/recipe boundaries, original Qwen half-width and T5-11B 24-layer correction")
print("No released-weight quality, GPU memory peak, kernel speed or complete-course transfer claim.")
print("CHAPTER_SHA256",hashlib.sha256(CHAPTER.read_bytes()).hexdigest())
