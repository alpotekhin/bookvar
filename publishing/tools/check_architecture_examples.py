#!/usr/bin/env python3
"""Execute the chapter and check objectives, masks, leakage and pinned code."""
from pathlib import Path
import ast
import hashlib
import math
import re
from types import SimpleNamespace
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT/"00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна.md"
body = CHAPTER.read_text()
blocks = re.findall(r"\x60\x60\x60python\n(.*?)\n\x60\x60\x60",body,re.S)
assert len(blocks)==4
spaces=[]
torch.set_num_threads(1)
for i,code in enumerate(blocks,1):
 ns={}
 exec(compile(code,f"{CHAPTER}:block{i}","exec"),ns)
 spaces.append(ns)
 print("PASS standalone published block",i)

n=spaces[0]
assert n["labels"].tolist()==[-100,-100,-100,6,7]
assert n["corrupted"][4]==n["original"][4]
assert math.isclose(n["loss"].item(),1.83258146374831)
for i in range(5):
 p=n["probs"][i]
 expected=p-torch.nn.functional.one_hot(n["original"][i],10).to(p.dtype)
 if not n["selected"][i]:
  expected=torch.zeros_like(p)
 torch.testing.assert_close(n["logits"].grad[i],expected)
assert n["next_labels"].tolist()==[3,4,5,6,7,2]
assert n["decoder_input"].tolist()==[1,3,4,5,6,7]
assert math.isclose(n["causal_loss"].item(),6*math.log(10))
torch.testing.assert_close(torch.tensor([1.,-1.]).softmax(-1),
 torch.tensor([.8807970779778823,.11920292202211755]))
print("PASS MLM selection/replacement distinction, unchanged target, six NTP targets and head example")

n=spaces[1]
visibility,attention=n["visibility"],n["masked_attention"]
for size in (1,2,3,6):
 for mode in ("full","causal","right-to-left"):
  mask=visibility(size,mode)
  for i in range(size):
   for j in range(size):
    assert mask[i,j].item()==(mode=="full" or (j<=i if mode=="causal" else j>=i))
 for p in range(1,size+1):
  mask=visibility(size,"prefix",p)
  for i in range(size):
   for j in range(size):
    expected=(j<p) if i<p else ((j<p) or j<=i)
    assert mask[i,j].item()==expected
for p in (0,5):
 try:
  visibility(4,"prefix",p)
 except ValueError:
  pass
 else:
  raise AssertionError("invalid prefix accepted")

torch.manual_seed(29)
q=torch.randn(3,2,dtype=torch.float64,requires_grad=True)
k=torch.randn(4,2,dtype=torch.float64,requires_grad=True)
v=torch.randn(4,3,dtype=torch.float64,requires_grad=True)
mask=torch.tensor([[True,False,True,False],[False,True,True,True],[True,True,False,False]])
actual,w=attention(q,k,v,mask)
loop=[]
for i in range(3):
 js=mask[i].nonzero().flatten()
 scores=torch.stack([(q[i]*k[j]).sum()/math.sqrt(2) for j in js])
 loop.append(sum(a*v[j] for a,j in zip(scores.softmax(-1),js)))
loop=torch.stack(loop)
native=F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
torch.testing.assert_close(actual,loop)
torch.testing.assert_close(actual,native)
g=torch.randn_like(actual)
ag=torch.autograd.grad((actual*g).sum(),(q,k,v),retain_graph=True)
lg=torch.autograd.grad((loop*g).sum(),(q,k,v))
for a,b in zip(ag,lg):
 torch.testing.assert_close(a,b)
assert torch.count_nonzero(w[~mask])==0
torch.testing.assert_close(w.sum(-1),torch.ones(3,dtype=w.dtype))
try:
 attention(q,k,v,torch.zeros_like(mask))
except ValueError:
 pass
else:
 raise AssertionError("empty rows accepted")
assert torch.autograd.gradcheck(lambda q,k,v:attention(q,k,v,mask)[0],(q,k,v))
# Zero output-logit gradients on unselected MLM rows do not block value paths.
values=torch.randn(4,2,dtype=torch.float64,requires_grad=True)
h,_=attention(torch.zeros_like(values),torch.zeros_like(values),values,visibility(4,"full"))
mlm_head=torch.tensor([[1.,-.5],[.2,.8]],dtype=values.dtype)
F.cross_entropy((h@mlm_head)[[1]],torch.tensor([0])).backward()
assert values.grad[0].abs().sum()>0 and values.grad[3].abs().sum()>0
print("PASS four visibility modes, rectangular attention, gradients/gradcheck and indirect MLM paths")

# The appended position may change all full-attention states, but not old
# causal states or states of a fixed bidirectional prefix plus causal suffix.
torch.testing.assert_close(n["full5"], torch.full((5,2),1.4,dtype=torch.float64))
torch.testing.assert_close(n["causal5"][:4],n["causal"])
torch.testing.assert_close(n["prefix5"][:4],n["prefix4"])
torch.testing.assert_close(n["new_query"][0],n["causal5"][-1])
extended=n["extended"]
zero=torch.zeros_like(extended)
prefix3,_=attention(zero,zero,extended,visibility(5,"prefix",3))
torch.testing.assert_close(n["prefix5"][:2],torch.full((2,2),.5,dtype=extended.dtype))
torch.testing.assert_close(prefix3[:2],torch.ones(2,2,dtype=extended.dtype))

# Exercise the same invariant with nonzero Q/K, three residual attention/FFN
# layers, fixed parameters and positionwise normalization. Reuse actual K/V
# at each layer; do not substitute recomputing the old hidden states.
generator=torch.Generator().manual_seed(41)
layers=[tuple(torch.randn(4,4,generator=generator,dtype=torch.float64)*.2
              for _ in range(5)) for _ in range(3)]
sequence=torch.randn(5,4,generator=generator,dtype=torch.float64)
def layer_step(h,parameters,mask,past=None):
 wq,wk,wv,w1,w2=parameters
 normalized=F.layer_norm(h,(4,))
 q,k,v=normalized@wq,normalized@wk,normalized@wv
 if past is not None:
  k,v=torch.cat([past[0],k]),torch.cat([past[1],v])
 context,_=attention(q,k,v,mask)
 h=h+context
 h=h+F.silu(F.layer_norm(h,(4,))@w1)@w2
 return h,(k,v)
def stack(h,mode,prefix=0):
 states=[]
 for layer in layers:
  h,cache=layer_step(h,layer,visibility(len(h),mode,prefix))
  states.append((h,cache))
 return h,states
for mode,prefix in [("causal",0),("prefix",2)]:
 dense,all_states=stack(sequence,mode,prefix)
 old,old_states=stack(sequence[:4],mode,prefix)
 for (new_h,new_kv),(old_h,old_kv) in zip(all_states,old_states):
  torch.testing.assert_close(new_h[:4],old_h)
  for new,old_value in zip(new_kv,old_kv):
   torch.testing.assert_close(new[:4],old_value)
 for cuts in [(2,5),(2,3,5),(2,3,4,5)]:
  caches=[None]*len(layers)
  outputs=[]
  start=0
  for end in cuts:
   h=sequence[start:end]
   mask=visibility(end,mode,prefix)[start:end]
   for layer_index,parameters in enumerate(layers):
    h,caches[layer_index]=layer_step(h,parameters,mask,caches[layer_index])
   outputs.append(h)
   start=end
  torch.testing.assert_close(torch.cat(outputs),dense)
  for cached,(_,expected) in zip(caches,all_states):
   for actual,wanted in zip(cached,expected):
    torch.testing.assert_close(actual,wanted)
full_old,_=stack(sequence[:4],"full")
full_new,_=stack(sequence,"full")
assert not torch.allclose(full_old,full_new[:4])
prefix_old,_=stack(sequence,"prefix",2)
prefix_new,_=stack(sequence,"prefix",3)
assert not torch.allclose(prefix_old[:2],prefix_new[:2])
print("PASS append arithmetic, fixed-prefix boundary and three-layer K/V reuse for three chunkings")

n=spaces[2]
corrupt=n["corrupt_spans"]
assert len(n["inputs"])==9 and len(n["targets"])==6
assert n["valid"].sum().item()==7 and n["labels"].tolist()[-2:]==[-100,-100]
for words,mask,expected_in,expected_out in [
 (["a","b","c"],[True,True,True],["<extra_id_0>"],["<extra_id_0>","a","b","c","<extra_id_1>"]),
 (["a","b","c"],[True,False,True],["<extra_id_0>","b","<extra_id_1>"],["<extra_id_0>","a","<extra_id_1>","c","<extra_id_2>"]),
 (["a","b","c"],[False,True,False],["a","<extra_id_0>","c"],["<extra_id_0>","b","<extra_id_1>"])
]:
 assert corrupt(words,mask)==(expected_in,expected_out)
for words,mask in [(["a"],[]),(["a"],[False])]:
 try:
  corrupt(words,mask)
 except ValueError:
  pass
 else:
  raise AssertionError("invalid corruption accepted")
src=ROOT/"05 Источники/Originals/positional-information-2026-10-21/modeling_t5-v4.57.1.py"
assert hashlib.sha256(src.read_bytes()).hexdigest()=="ee1a043f5c03c684a2bbb0213ea3b3f8be272a03ab884bdca57c31e7315e7b7e"
fn=next(x for x in ast.walk(ast.parse(src.read_text())) if isinstance(x,ast.FunctionDef) and x.name=="_shift_right")
ns={"torch":torch,"is_torch_fx_proxy":lambda _:False}
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(src),"exec"),ns)
obj=SimpleNamespace(config=SimpleNamespace(decoder_start_token_id=0,pad_token_id=0))
torch.testing.assert_close(ns["_shift_right"](obj,n["labels"]),n["decoder_ids"])
assert n["allowed"][0,0] and not n["wrong"][0].any()
print("PASS T5 spans/sentinels, EOS/ignored PAD, valid PAD-ID start and pinned HF shift")

n=spaces[3]
assert n["correct"][3].item()==11.125 and n["leaking"][3].item()==12.25
# Jacobian confirms whether the future value influences the first target row.
values=n["x"].clone().requires_grad_()
good_grad=torch.autograd.grad(n["two_layers"](values,n["good"])[3].sum(),values)[0]
bad_grad=torch.autograd.grad(n["two_layers"](values,n["bad"])[3].sum(),values)[0]
assert good_grad[-1].item()==0 and bad_grad[-1].item()==.125
src=ROOT/"05 Источники/Originals/architecture-patterns-2026-10-21/unilm-seq2seq-loader.py"
assert hashlib.sha256(src.read_bytes()).hexdigest()=="c50ca5a774d1bfd3554e8f59c5e7b9796d12c3f52d2c5d116589f37bf89d91aa"
tree=ast.parse(src.read_text())
branch=next(x for x in ast.walk(tree) if isinstance(x,ast.If) and x.lineno==270)
assert len(branch.body)==3
code=compile(ast.fix_missing_locations(ast.Module(body=branch.body,type_ignores=[])),str(src),"exec")
for la in range(4):
 for lb in range(4):
  for padding in range(3):
   length=la+lb+3
   max_len=length+padding
   native=torch.zeros(max_len,max_len,dtype=torch.long)
   obj=SimpleNamespace(_tril_matrix=torch.ones(max_len,max_len,dtype=torch.long).tril())
   ns={"self":obj,"torch":torch,"tokens_a":["a"]*la,"tokens_b":["b"]*lb,"input_mask":native,"len":len}
   exec(code,ns)
   torch.testing.assert_close(native[:length,:length].bool(),
                               visibility(length,"prefix",la+2))
   assert not native[:,length:].any()
print("PASS two-layer leakage/Jacobians and pinned UniLM source/target/padding mask block")
print("PASS chapter SHA256",hashlib.sha256(CHAPTER.read_bytes()).hexdigest())
