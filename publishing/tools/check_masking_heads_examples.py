#!/usr/bin/env python3
"""Execute the chapter and independently verify masks, layout, MHA and gradients."""
from pathlib import Path
import hashlib
import math
import re
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, einsum, reduce, EinopsError

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/05 Attention и Transformer/Masking, multi-head и формы тензоров.md"
text = CHAPTER.read_text()
blocks = re.findall(r"\x60\x60\x60python\n(.*?)\n\x60\x60\x60",text,re.S)
assert len(blocks) == 6
spaces = []
for i,code in enumerate(blocks,1):
    ns={}
    exec(compile(code,f"{CHAPTER}:block{i}","exec"),ns)
    spaces.append(ns)
    print("PASS standalone published block",i)

dtype=torch.float64
a=spaces[0]
assert math.isclose(a["late"].sum().item(),.9099694268296196)
assert math.isclose((a["late"]@a["values"]).item(),11.54697897884417)
assert math.isclose((a["early"]@a["values"]).item(),12.689414213699951)
assert a["late_extreme"].sum().item()==0
torch.testing.assert_close(a["late"]/a["late"].sum(),a["early"])
b=spaces[1]
assert b["split"][0,0].tolist()==[[0,1],[4,5],[8,9]]
assert b["split"][0,1].tolist()==[[2,3],[6,7],[10,11]]
assert b["wrong"][0,0].tolist()==[[0,1],[2,3],[4,5]]
assert b["split"].shape==b["wrong"].shape and not torch.equal(b["split"],b["wrong"])
named=spaces[2]
assert named["named_scores"][0,0,0].tolist()==[9.,5.,1.]
assert named["norm_sq"][0,0].tolist()==[1.,41.,145.]
torch.testing.assert_close(named["restored"],named["projected"])
interleaved=rearrange(named["projected"],
    "batch token (coord head) -> batch head token coord",head=2)
assert interleaved.shape==named["q_heads"].shape
assert not torch.equal(interleaved,named["q_heads"])
mislabelled=einsum(named["q_heads"].transpose(0,1),named["k_heads"],
    "batch head query coord, batch head key coord -> batch head query key")
assert mislabelled.shape==named["named_scores"].shape
assert not torch.equal(mislabelled,named["named_scores"])
try:
    rearrange(torch.zeros(2,3,5),
        "batch token (head coord) -> batch head token coord",head=2)
except EinopsError:
    pass
else:
    raise AssertionError("indivisible named head dimension accepted")

torch.manual_seed(26)
named_cases=0
for batch in (1,2):
    for heads in (1,2,3):
        for queries in (1,3):
            for keys in (1,4):
                for width in (1,2):
                    rawq=torch.randn(batch,queries,heads*width,dtype=dtype,requires_grad=True)
                    rawk=torch.randn(batch,keys,heads*width,dtype=dtype,requires_grad=True)
                    qh=rearrange(rawq,"b t (h c) -> b h t c",h=heads)
                    kh=rearrange(rawk,"b t (h c) -> b h t c",h=heads)
                    scores=einsum(qh,kh,"b h q c, b h k c -> b h q k")
                    loop=torch.stack([torch.stack([torch.stack([torch.stack([
                        (qh[bi,hi,qi]*kh[bi,hi,ki]).sum()
                        for ki in range(keys)]) for qi in range(queries)])
                        for hi in range(heads)]) for bi in range(batch)])
                    torch.testing.assert_close(scores,loop)
                    ellipsis=einsum(qh,kh,"... q c, ... k c -> ... q k")
                    torch.testing.assert_close(ellipsis,loop)
                    torch.testing.assert_close(
                        rearrange(qh,"b h t c -> b t (h c)"),rawq)
                    torch.testing.assert_close(
                        reduce(qh.square(),"b h t c -> b h t","sum"),
                        qh.square().sum(-1))
                    upstream=torch.randn_like(scores)
                    actual_grads=torch.autograd.grad((scores*upstream).sum(),(rawq,rawk))
                    loop_grads=torch.autograd.grad((loop*upstream).sum(),(rawq,rawk))
                    for actual,expected in zip(actual_grads,loop_grads):
                        torch.testing.assert_close(actual,expected)
                    named_cases+=1
assert named_cases==48
print("PASS named axes: 48 rectangular cases, scalar-loop outputs/all gradients, ellipsis and inverse merge")
print("PASS named axes: (head coord) vs (coord head), swapped equal axes, reduction and invalid factorization")
c=spaces[3]
assert c["cached"].item()==2.5 and c["upper_left"].item()==1
d=spaces[4]
assert d["block_allowed"].tolist()==[[True,True,True,True,False],
                                    [True,True,True,True,True]]
assert d["block_cached"].flatten().tolist()==[2.5,3.]
assert d["block_upper_left"].flatten().tolist()==[1.,1.5]
assert d["block_unmasked"].flatten().tolist()==[3.,3.]
from torch.nn.attention.bias import causal_lower_right
right=F.scaled_dot_product_attention(
    d["q_block"],d["k_block"],d["v_block"],
    attn_mask=causal_lower_right(2,5),dropout_p=0.)
torch.testing.assert_close(right,d["block_cached"])
vblock=d["v_block"].clone().requires_grad_()
first=F.scaled_dot_product_attention(
    d["q_block"],d["k_block"],vblock,
    attn_mask=d["block_allowed"],dropout_p=0.)[0,0,0,0]
dv=torch.autograd.grad(first,vblock)[0].flatten()
torch.testing.assert_close(dv,torch.tensor([.25,.25,.25,.25,0.],dtype=dtype))
print("PASS masking normalization/underflow, head layout and absolute cache positions")
print("PASS two-query cache block, lower-right equivalence and no future-value gradient")

n=spaces[5]
attention=n["masked_attention"]
torch.manual_seed(21)
q=torch.randn(2,3,2,4,dtype=dtype,requires_grad=True)
k=torch.randn(2,3,5,4,dtype=dtype,requires_grad=True)
v=torch.randn(2,3,5,6,dtype=dtype,requires_grad=True)
allowed=torch.rand(2,1,2,5)>.35
allowed[0,:,1]=False
out,w=attention(q,k,v,allowed)
assert out.shape==(2,3,2,6) and w.shape==(2,3,2,5)
expanded=allowed.expand_as(w)
rows=[]
for bi in range(2):
    heads=[]
    for hi in range(3):
        queries=[]
        for qi in range(2):
            selected=allowed[bi,0,qi].nonzero().flatten()
            if len(selected):
                scores=(k[bi,hi,selected]*q[bi,hi,qi]).sum(-1)/2
                queries.append(scores.softmax(-1)@v[bi,hi,selected])
            else:
                queries.append(torch.zeros(6,dtype=dtype))
        heads.append(torch.stack(queries))
    rows.append(torch.stack(heads))
loop=torch.stack(rows)
torch.testing.assert_close(out,loop)
sdpa=F.scaled_dot_product_attention(q,k,v,attn_mask=allowed,dropout_p=0)
assert torch.isfinite(sdpa).all()
torch.testing.assert_close(out,sdpa)
assert torch.count_nonzero(w[~expanded])==0
torch.testing.assert_close(w.sum(-1),expanded.any(-1).to(dtype))
g=torch.randn_like(out)
ours=torch.autograd.grad((out*g).sum(),(q,k,v))
reference=torch.autograd.grad((loop*g).sum(),(q,k,v))
for x,y in zip(ours,reference):
    torch.testing.assert_close(x,y)
    assert torch.isfinite(x).all()
qg=torch.randn(1,1,2,2,dtype=dtype,requires_grad=True)
kg=torch.randn(1,1,3,2,dtype=dtype,requires_grad=True)
vg=torch.randn(1,1,3,1,dtype=dtype,requires_grad=True)
mg=torch.tensor([[True,False,True],[False,False,False]])
assert torch.autograd.gradcheck(lambda q,k,v:attention(q,k,v,mg)[0],(qg,kg,vg))
print("PASS rectangular batched/head attention, empty-row policy, scalar loops, SDPA and finite differences")

Cls=n["MultiHeadSelfAttention"]
model=Cls(4,2).double()
native=nn.MultiheadAttention(4,2,bias=False,batch_first=True,dtype=dtype)
with torch.no_grad():
    native.in_proj_weight.copy_(torch.cat([
        model.q_proj.weight,model.k_proj.weight,model.v_proj.weight]))
    native.out_proj.weight.copy_(model.out_proj.weight)
x=torch.randn(2,3,4,dtype=dtype,requires_grad=True)
xn=x.detach().clone().requires_grad_()
actual,_=model(x)
future=torch.ones(3,3,dtype=torch.bool).triu(1)
expected,_=native(xn,xn,xn,attn_mask=future,need_weights=False)
torch.testing.assert_close(actual,expected)
g=torch.randn_like(actual)
grad=torch.autograd.grad((actual*g).sum(),(x,*model.parameters()))
ngrad=torch.autograd.grad((expected*g).sum(),(xn,native.in_proj_weight,native.out_proj.weight))
torch.testing.assert_close(grad[0],ngrad[0])
torch.testing.assert_close(torch.cat(grad[1:4]),ngrad[1])
torch.testing.assert_close(grad[4],ngrad[2])
assert sum(p.numel() for p in model.parameters())==4*4**2
counts=[]
for heads in (2,4):
    width=4//heads
    candidate=Cls(4,heads).double()
    y,weights=candidate(torch.zeros(2,3,4,dtype=dtype))
    params=sum(p.numel() for p in candidate.parameters())
    assert params==64 and y.shape==(2,3,4)
    assert weights.numel()==2*heads*3*3
    qk_macs=2*heads*3*3*width
    av_macs=2*heads*3*width*3
    assert qk_macs==av_macs==72
    counts.append((heads,width,params,weights.numel(),qk_macs,av_macs))
assert [c[3] for c in counts]==[36,72]
print("PASS (heads, head_dim, parameters, score elements, QK MACs, AV MACs)",counts)

model,valid=n["model"],n["valid"]
x=n["x"].detach()
base,w=model(x,valid)
changed=x.clone()
changed[~valid]=1000
torch.testing.assert_close(model(changed,valid)[0][valid],base[valid])
changed=x.clone()
changed[:,3]+=500
torch.testing.assert_close(model(changed,valid)[0][:,:3],base[:,:3])
assert torch.count_nonzero(base[~valid])==0
assert torch.count_nonzero(w[2])==0
assert torch.count_nonzero(w.triu(1))==0
assert torch.count_nonzero(n["x"].grad[~valid])==0
torch.testing.assert_close(n["toy_out"][0,0],torch.tensor([32.,-6.],dtype=dtype))
torch.testing.assert_close(torch.tensor([6.,6.],dtype=dtype)@n["WO"],torch.tensor([24.,6.],dtype=dtype))
for bad in [valid.long(),valid[:,:3]]:
    try: model(x,bad)
    except ValueError: pass
    else: raise AssertionError("invalid validity mask accepted")
try: Cls(5,2)
except ValueError: pass
else: raise AssertionError("indivisible head width accepted")
print("PASS full MHA against native outputs/all gradients, causal independence, padding and two-head projection")

# B=T can make a semantically wrong [B,T] mask broadcast successfully.
torch.manual_seed(22)
q,k,v=[torch.randn(3,2,3,2,dtype=dtype) for _ in range(3)]
mask=torch.tensor([[True,False,True],[False,True,True],[True,True,False]])
correct=attention(q,k,v,mask[:,None,None,:])[0]
wrong=attention(q,k,v,mask)[0]
assert correct.shape==wrong.shape and not torch.allclose(correct,wrong)
print("PASS same-shape broadcasting counterexample")
print("torch_version",torch.__version__)
print("chapter_sha256",hashlib.sha256(CHAPTER.read_bytes()).hexdigest())
