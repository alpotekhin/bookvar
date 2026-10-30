#!/usr/bin/env python3
"""Execute published positional examples and independently check the mechanisms."""
from pathlib import Path
import ast
import copy
import hashlib
import math
import re
import torch

ROOT=Path(__file__).resolve().parents[2]
CHAPTER=ROOT/"00 Учебник/05 Attention и Transformer/04 Позиционная информация.md"
SRC=ROOT/"05 Источники/Originals/positional-information-2026-10-21"
text=CHAPTER.read_text()
blocks=re.findall(r"\x60\x60\x60python\n(.*?)\n\x60\x60\x60",text,re.S)
assert len(blocks)==6
spaces=[]
for i,code in enumerate(blocks,1):
 ns={}
 exec(compile(code,f"{CHAPTER}:block{i}","exec"),ns)
 spaces.append(ns)
 print("PASS standalone published block",i)
dtype=torch.float64

# Permutation property, including the transformed mask; causal means are a counterexample.
torch.manual_seed(44)
x=torch.randn(3,4,dtype=dtype)
wq,wk=torch.randn(4,2,dtype=dtype),torch.randn(4,2,dtype=dtype)
wv=torch.randn(4,3,dtype=dtype)
p=torch.tensor([2,0,1])
def attention(x,mask=None):
 scores=(x@wq)@(x@wk).T/math.sqrt(2)
 if mask is not None: scores=scores.masked_fill(~mask,-torch.inf)
 return scores.softmax(-1)@(x@wv)
torch.testing.assert_close(attention(x[p]),attention(x)[p])
mask=torch.tensor([[True,False,True],[False,True,True],[True,True,False]])
torch.testing.assert_close(attention(x[p],mask[p][:,p]),attention(x,mask)[p])
mean=torch.ones(3,3,dtype=dtype).tril()
mean=mean/mean.sum(-1,keepdim=True)
values=torch.tensor([1.,3.,7.],dtype=dtype)
torch.testing.assert_close(mean@values,torch.tensor([1.,2.,11/3],dtype=dtype))
torch.testing.assert_close(mean@values.flip(0),torch.tensor([7.,5.,11/3],dtype=dtype))
assert not torch.allclose(mean@values.flip(0),(mean@values).flip(0))

# Learned-table example: independent full backward, since Q=K=V=h has three paths.
n=spaces[0]
h,a,z=n["h"].detach(),n["weights"].detach(),n["z"].detach()
torch.testing.assert_close(h,torch.tensor([[2.,1.],[1.,1.]],dtype=dtype))
assert math.isclose(a[0,0].item(),.8044296825069569)
assert math.isclose(z[0,0].item(),1.8044296825069568)
g=torch.zeros_like(z); g[0]=z[0] # d mean(z[0]^2)/dz[0]
dv=a.T@g
da=g@h.T
ds=a*(da-(da*a).sum(-1,keepdim=True))
dh=(ds@h+ds.T@h)/math.sqrt(2)+dv
torch.testing.assert_close(n["e"].grad,dh)
torch.testing.assert_close(n["P"].grad[:2],dh)
assert torch.count_nonzero(n["P"].grad[2:])==0
print("PASS permutation/masks, causal counterexample and all learned-position gradient paths")

# Sinusoids: scalar implementation, shift matrices, norm and relative-only PE dot products.
sinusoidal=spaces[1]["sinusoidal"]
for d in [2,4,8,32]:
 positions=torch.tensor([0.,1.,2.,31.,128.],dtype=dtype)
 actual=sinusoidal(positions,d)
 expected=torch.tensor([[f(float(i)/10000**(2*k/d)) for k in range(d//2)
                        for f in (math.sin,math.cos)] for i in positions],dtype=dtype)
 torch.testing.assert_close(actual,expected)
 torch.testing.assert_close(actual.square().sum(-1),torch.full((5,),d/2,dtype=dtype))
 for k in range(d//2):
  angle=7/10000**(2*k/d)
  c,s=math.cos(angle),math.sin(angle)
  U=torch.tensor([[c,s],[-s,c]],dtype=dtype)
  torch.testing.assert_close(actual[:,2*k:2*k+2]@U.T,
                             sinusoidal(positions+7,d)[:,2*k:2*k+2])
 expected_dot=torch.tensor([[sum(math.cos((float(i)-float(j))/10000**(2*k/d))
                                for k in range(d//2)) for j in positions] for i in positions],dtype=dtype)
 torch.testing.assert_close(actual@actual.T,expected_dot)
# Different scaling transforms are not interchangeable.
torch.testing.assert_close(torch.tensor([1.,40000**(-.5)]),torch.tensor([1.,.005]))
assert not torch.allclose(torch.tensor([1.,.005]),torch.tensor([.25,.0025]))
print("PASS sinusoidal scalar values, shift signs, norm, PE dot products and scaling distinction")

n=spaces[2]
assert math.isclose(n["z"].item(),18.517688345869864)
torch.testing.assert_close(n["messages"],torch.tensor([11.,20.,28.],dtype=dtype))
# More than one edge may use a table row: gradients accumulate.
q=torch.tensor(1.3,dtype=dtype)
kt=torch.tensor([.2,.3,-.7],dtype=dtype,requires_grad=True)
vt=torch.tensor([1.,0.,-2.],dtype=dtype,requires_grad=True)
ids=torch.tensor([0,0,2,2])
v=torch.tensor([3.,4.,5.,7.],dtype=dtype)
a=(q*(torch.full((4,),2.,dtype=dtype)+kt[ids])).softmax(-1)
messages=v+vt[ids]
z=a@messages
z.backward()
for bucket in [0,1,2]:
 chosen=ids==bucket
 torch.testing.assert_close(vt.grad[bucket],a.detach()[chosen].sum())
 torch.testing.assert_close(kt.grad[bucket],
                            (q*a.detach()*(messages.detach()-z.detach()))[chosen].sum())
print("PASS relative key/value messages, clipped classes and reused Shaw-table gradients")

def load_source_function(filename,name):
 expected_hash={
  "modeling_t5-v4.57.1.py":"ee1a043f5c03c684a2bbb0213ea3b3f8be272a03ab884bdca57c31e7315e7b7e",
  "alibi-transformer.py":"faaf4b53c07fa69dec09a6b3faff279b72c3bc5fb5b018629a95471337dc0e87"
 }[filename]
 assert hashlib.sha256((SRC/filename).read_bytes()).hexdigest()==expected_hash
 tree=ast.parse((SRC/filename).read_text())
 node=copy.deepcopy(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name))
 node.decorator_list=[]
 module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
 scope={"torch":torch,"math":math}
 exec(compile(module,str(SRC/filename),"exec"),scope)
 return scope[name]

n=spaces[3]
bucket=n["t5_bucket"]
official=load_source_function("modeling_t5-v4.57.1.py","_relative_position_bucket")
r=torch.arange(-2048,2049)
for bidir in [False,True]:
 for count in [8,16,32]:
  for maximum in [32,128]:
   torch.testing.assert_close(bucket(r,bidir,count,maximum),official(r,bidir,count,maximum))
distance=n["distance"]
torch.testing.assert_close(bucket(-distance,False),torch.tensor([0,1,7,8,15,16,21,26,31,31]))
torch.testing.assert_close(bucket(-distance,True),torch.tensor([0,1,7,8,9,10,12,14,15,15]))
torch.testing.assert_close(bucket(distance,True),torch.tensor([0,17,23,24,25,26,28,30,31,31]))
assert torch.count_nonzero(bucket(torch.tensor([1,100]),False))==0
table=n["table"]
weights=n["weights"].detach()
values=n["values"]
out=n["z"].detach()
for b in range(32):
 chosen=n["bucket_ids"]==b
 expected=(weights[:,chosen]*(values[chosen]-out[:,None])).sum(-1)
 torch.testing.assert_close(table.weight.grad[b],expected)
w=table.weight.detach().clone().requires_grad_()
assert torch.autograd.gradcheck(lambda w:(w[n["bucket_ids"]].T).softmax(-1)@values,(w,))
print("PASS exact T5 buckets against pinned source, direction/range, shared gradients and finite differences")

n=spaces[4]
rope=n["apply_rope"]
def dense_rotation(position,d,base=10000.):
 matrices=[]
 for k in range(d//2):
  a=position*base**(-2*k/d)
  c,s=math.cos(a),math.sin(a)
  matrices.append(torch.tensor([[c,-s],[s,c]],dtype=dtype))
 return torch.block_diag(*matrices)
q,k=n["q"].detach(),n["k"].detach()
for b in range(2):
 for h in range(3):
  for i in range(4):
   R=dense_rotation(i,4)
   torch.testing.assert_close(n["qr"][b,h,i],q[b,h,i]@R.T)
   torch.testing.assert_close(n["kr"][b,h,i],k[b,h,i]@R.T)
# A nonzero example catches the sign; scalar zero examples would not.
q0,k0=torch.tensor([1.,2.],dtype=dtype),torch.tensor([3.,4.],dtype=dtype)
R2=dense_rotation(math.pi,2)
R1=dense_rotation(math.pi/2,2)
rotq,rotk=q0@R2.T,k0@R1.T
torch.testing.assert_close(rotq,torch.tensor([-1.,-2.],dtype=dtype))
torch.testing.assert_close(rotk,torch.tensor([-4.,3.],dtype=dtype))
assert math.isclose((rotq@rotk).item(),-2.)
assert math.isclose((q0@R2@R1.T@k0).item(),2.)
assert math.isclose((q0@dense_rotation(-math.pi/2,2)@k0).item(),-2.)
probe=torch.randn(2,3,4,dtype=dtype,requires_grad=True)
positions=torch.tensor([1,4,7])
assert torch.autograd.gradcheck(lambda x:rope(x,positions),(probe,))
g=torch.randn_like(probe)
actual=torch.autograd.grad((rope(probe,positions)*g).sum(),probe)[0]
expected=torch.stack([torch.stack([g[b,i]@dense_rotation(int(p),4)
                                  for i,p in enumerate(positions)]) for b in range(2)])
torch.testing.assert_close(actual,expected)
# Heterogeneous positions require an explicit head axis, not a [B,T] broadcast.
hetero=torch.tensor([[0,1,2,3],[7,8,9,10]])[:,None,:].expand(2,3,4)
hetero_out=rope(q,hetero)
for b in range(2):
 for h in range(3):
  for i in range(4):
   torch.testing.assert_close(hetero_out[b,h,i],q[b,h,i]@dense_rotation(int(hetero[b,h,i]),4).T)
for bad_x,bad_pos in [(torch.randn(2,3),torch.arange(2)),
                     (q,torch.zeros(2,4))]:
 try: rope(bad_x,bad_pos)
 except ValueError: pass
 else: raise AssertionError("invalid rotary shape accepted")
# [a,b,c,d] -> [a,c,b,d] converts adjacent pairing into split-half pairing.
perm=torch.tensor([0,2,1,3])
frequency=torch.tensor([1.,.01],dtype=dtype)
angles=n["positions"][:,None]*frequency
c,s=torch.cat([angles.cos()]*2,-1),torch.cat([angles.sin()]*2,-1)
def half_rotate(x): return torch.cat((-x[...,2:],x[...,:2]),-1)
xp=q[...,perm]
half=xp*c+half_rotate(xp)*s
torch.testing.assert_close(half[...,perm],n["qr"])
wrong=q*c+half_rotate(q)*s
assert not torch.allclose(wrong,n["qr"])
def partial(x,p): return torch.cat((rope(x[...,:2],p),x[...,2:]),-1)
torch.testing.assert_close(partial(q,n["positions"])[...,2:],q[...,2:])
torch.testing.assert_close(partial(q,n["positions"])@partial(k,n["positions"]).transpose(-1,-2),
                           partial(q,n["positions"]+9)@partial(k,n["positions"]+9).transpose(-1,-2))
assert not torch.allclose(n["wrong"],n["cached"])
# Periodicity contradicts a pointwise monotone-distance interpretation.
scores=[float(q0.new_tensor([1.,0.])@dense_rotation(d*math.pi/2,2)@q0.new_tensor([1.,0.]))
        for d in range(5)]
torch.testing.assert_close(torch.tensor(scores),torch.tensor([1.,0.,-1.,0.,1.]),atol=1e-7,rtol=1e-7)
default_scores=[float(q0.new_tensor([1.,0.])@dense_rotation(d,2)@q0.new_tensor([1.,0.]))
                for d in [0,3,6]]
torch.testing.assert_close(torch.tensor(default_scores,dtype=dtype),
                           torch.tensor([1.,math.cos(3),math.cos(6)],dtype=dtype))
assert default_scores[2]>default_scores[1]
print("PASS RoPE matrices/sign, all-pair gradients, layouts, partial width, cache and non-monotone scores")

# The published incremental example has an independently computable cosine score.
p=torch.arange(4,dtype=dtype)
scores=(p[:,None]-p[None,:]).cos()/math.sqrt(2)
weights=scores.masked_fill(p[None,:]>p[:,None],-torch.inf).softmax(-1)
expected=weights@torch.arange(1.,5.,dtype=dtype)
torch.testing.assert_close(n["incremental"].flatten(),expected)
torch.testing.assert_close(expected,torch.tensor(
 [1.,1.5805557848615206,2.3027101501815075,3.06122252101854],dtype=dtype))
torch.testing.assert_close(weights[-1],torch.tensor(
 [.10487104907403684,.15735456721780225,.309455197323745,.428319186384416],dtype=dtype))
assert n["block_allowed"].tolist()==[[True,True,True,False],[True,True,True,True]]
torch.testing.assert_close(n["block_output"],n["full_demo"][...,2:,:])
torch.testing.assert_close(n["key_cache"],n["rotated"])
torch.testing.assert_close(n["value_cache"],n["values"])
assert math.isclose(n["double_rotated"].item(),2.5)
assert not torch.allclose(n["double_rotated"],n["full_demo"][...,-1:,:])

# Distinct random Q/K and dv!=dk: any chunking preserves all outputs/gradients.
import torch.nn.functional as F
for boundaries in [[0,1,2,3,4],[0,2,4],[0,1,4],[0,4]]:
 q0=n["q"].detach().clone().requires_grad_()
 k0=n["k"].detach().clone().requires_grad_()
 v0=torch.randn(2,3,4,3,dtype=dtype,requires_grad=True)
 positions=torch.arange(4)
 allowed=positions[None,:]<=positions[:,None]
 full=F.scaled_dot_product_attention(rope(q0,positions),rope(k0,positions),v0,
                                    attn_mask=allowed,dropout_p=0.)
 kc=vc=None
 outputs=[]
 for start,end in zip(boundaries,boundaries[1:]):
  new=rope(k0[...,start:end,:],positions[start:end])
  kc=new if kc is None else torch.cat((kc,new),dim=-2)
  newv=v0[...,start:end,:]
  vc=newv if vc is None else torch.cat((vc,newv),dim=-2)
  permitted=positions[None,:end]<=positions[start:end,None]
  outputs.append(F.scaled_dot_product_attention(
   rope(q0[...,start:end,:],positions[start:end]),kc,vc,
   attn_mask=permitted,dropout_p=0.))
 chunks=torch.cat(outputs,dim=-2)
 torch.testing.assert_close(chunks,full)
 probe=torch.randn_like(full)
 gf=torch.autograd.grad((full*probe).sum(),(q0,k0,v0))
 gc=torch.autograd.grad((chunks*probe).sum(),(q0,k0,v0))
 for a,b in zip(gf,gc): torch.testing.assert_close(a,b)
print("PASS incremental cosine example, two-query mask, double-rotation negative control and four chunkings/all gradients")

n=spaces[5]
slopes=n["alibi_slopes"]
official=load_source_function("alibi-transformer.py","get_slopes")
for heads in list(range(1,18))+[32,64]:
 assert slopes(heads)==official(heads)
torch.testing.assert_close(n["slopes"],2.**(-torch.arange(1,9,dtype=dtype)))
assert math.isclose(float(n["weights"]@n["values"]),32.344406289660526)
# For different Q/K lengths, absolute positions still define the relative bias.
qp=torch.tensor([6,7])
kp=torch.arange(8)
m=torch.tensor([.5,.25],dtype=dtype)[:,None,None]
allowed=kp[None,:]<=qp[:,None]
content=torch.randn(2,2,8,dtype=dtype)
full=(content-m*(qp[:,None]-kp[None,:])).masked_fill(~allowed,-torch.inf)
column=(content+m*kp).masked_fill(~allowed,-torch.inf)
torch.testing.assert_close(full.softmax(-1),column.softmax(-1))
print("PASS ALiBi source slopes, exact weights/ratio and row-constant equivalence for rectangular cache")
print("torch_version",torch.__version__)
print("chapter_sha256",hashlib.sha256(CHAPTER.read_bytes()).hexdigest())
