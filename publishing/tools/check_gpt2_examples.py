#!/usr/bin/env python3
"""Run published GPT-2 examples and verify independent/source contracts."""
from pathlib import Path
from types import SimpleNamespace
import ast
import hashlib
import importlib.util
import json
import math
import re
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
CHAPTER=ROOT/"00 Учебник/06 Encoder, Decoder и Encoder-Decoder/05 GPT-2 — zero-shot через язык.md"
SOURCE=ROOT/"05 Источники/Originals/gpt2-2026-10-22"
body=CHAPTER.read_text()
blocks=re.findall(r"\x60\x60\x60python\n(.*?)\n\x60\x60\x60",body,re.S)
assert len(blocks)==2
spaces=[]
for i,block in enumerate(blocks,1):
    ns={}
    exec(compile(block,f"{CHAPTER}:block{i}","exec"),ns)
    spaces.append(ns)
    print("PASS standalone block",i)

dist=spaces[0]["distribution"]
z=torch.tensor([2.,1.,0.,-1.],dtype=torch.float64)
torch.testing.assert_close(dist(z),z.exp()/z.exp().sum())
assert torch.count_nonzero(dist(torch.tensor([2.,1.,1.,0.]),k=2))==3
try:
    dist(z,temperature=0)
    raise RuntimeError("zero temperature accepted")
except AssertionError:
    pass
print("PASS temperature, top-k and threshold ties")

n=spaces[1]
model,ids=n["model"],n["ids"]
with torch.no_grad():
    full,_=model(ids)
    for cuts in range(1<<(ids.shape[1]-1)):
        ends=[i+1 for i in range(ids.shape[1]-1) if cuts&(1<<i)]+[ids.shape[1]]
        start,cache,outputs=0,None,[]
        for end in ends:
            logits,cache=model(ids[:,start:end],cache)
            outputs.append(logits)
            assert all(k.shape[-2]==end and v.shape[-2]==end for k,v in cache)
            start=end
        torch.testing.assert_close(torch.cat(outputs,1),full,atol=1e-10,rtol=1e-10)
    changed=ids.clone(); changed[0,-1]=6
    changed_logits,_=model(changed)
    torch.testing.assert_close(full[:,:-1],changed_logits[:,:-1])
    assert (full[:,-1]-changed_logits[:,-1]).abs().max()>1e-6
    _,past=model(ids[:,:3])
    correct,_=model(ids[:,3:4],past)
    wrong_position,_=model(ids[:,3:4])
    assert (correct-wrong_position).abs().max()>1e-6
    _,near_limit=model(torch.ones(1,15,dtype=torch.long))
    try:
        model(torch.ones(1,2,dtype=torch.long),near_limit)
        raise RuntimeError("position overflow accepted")
    except AssertionError:
        pass
assert model.word.weight.grad.norm()>0 and model.blocks[0].qkv.weight.grad.norm()>0
print("PASS all 16 chunkings/full logits, causality, cache lengths, position offset/context bound")

b=model.blocks[0]
ref=nn.TransformerEncoderLayer(16,4,64,dropout=0.,batch_first=True,norm_first=True,
    activation=lambda x:torch.nn.functional.gelu(x,approximate="tanh")).double().eval()
pairs=[(b.qkv.weight,ref.self_attn.in_proj_weight),(b.qkv.bias,ref.self_attn.in_proj_bias),
       (b.proj.weight,ref.self_attn.out_proj.weight),(b.proj.bias,ref.self_attn.out_proj.bias),
       (b.fc.weight,ref.linear1.weight),(b.fc.bias,ref.linear1.bias),
       (b.out.weight,ref.linear2.weight),(b.out.bias,ref.linear2.bias),
       (b.ln1.weight,ref.norm1.weight),(b.ln1.bias,ref.norm1.bias),
       (b.ln2.weight,ref.norm2.weight),(b.ln2.bias,ref.norm2.bias)]
with torch.no_grad():
    for a,c in pairs:c.copy_(a)
x=torch.randn(2,5,16,dtype=torch.float64,requires_grad=True)
y=x.detach().clone().requires_grad_()
actual,_=b(x)
expected=ref(y,src_mask=torch.ones(5,5,dtype=torch.bool).triu(1))
torch.testing.assert_close(actual,expected,atol=1e-10,rtol=1e-10)
probe=torch.randn_like(actual)
ag=torch.autograd.grad((actual*probe).sum(),(x,)+tuple(a for a,_ in pairs))
eg=torch.autograd.grad((expected*probe).sum(),(y,)+tuple(c for _,c in pairs))
for a,e in zip(ag,eg):torch.testing.assert_close(a,e,atol=1e-9,rtol=1e-9)
print("PASS pre-norm output and input/all parameter gradients against independent PyTorch block")

for name in ["hparams.json","hparams-355M.json","hparams-774M.json","hparams-1558M.json"]:
    h=json.loads((SOURCE/name).read_text())
    d,L,V,T=(h[k] for k in ("n_embd","n_layer","n_vocab","n_ctx"))
    count=V*d+T*d+L*(12*d*d+13*d)+2*d
    assert count=={12:124439808,24:354823168,36:774030080,48:1557611200}[L]
    assert h["n_head"]==d//64
assert 17*16+16*16+2*(12*16**2+13*16)+2*16==sum(p.numel() for p in model.parameters())==7120
assert math.isclose(math.exp(6/3),7.38905609893,abs_tol=1e-10)
assert math.isclose(math.exp(6/5),3.32011692274,abs_tol=1e-10)
assert .6>.4 and .6*.2<.4*.8
assert .8*.4>.2*.5 and .4<.5  # full versus partial scoring reverses ranking
assert 2*12*12*1024*64==18874368
assert 18874368*4/2**20==72 and 18874368*2/2**20==36
print("PASS four released model counts, toy count, PPL units and candidate/suffix ranking")

hashes={"model.py":"2ff5065f5cac3dc93065f8a3f36eb013e407972a80abcd4ee6b89a7138a0d5a6",
"encoder.py":"84caddd892377b74de8b61f1fdf2437de9dc025c2f3bc73a981edcb0605e4f9a",
"sample.py":"7a726916bdc416b296665b52ea7b87581152e16794c8eaff90bf4f89b1235771",
"encoder.json":"196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
"vocab.bpe":"1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"}
for name,digest in hashes.items():
    assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==digest
spec=importlib.util.spec_from_file_location("gpt2_original_encoder",SOURCE/"encoder.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
encoder=module.get_encoder(SOURCE.name,str(SOURCE.parent))
for text,tokens in [("Hello world",[15496,995]),("Hello",[15496]),(" hello",[23748]),
    (" hello!",[23748,0]),("Привет",[140,253,21169,18849,38857,16843,20375])]:
    assert encoder.encode(text)==tokens and encoder.decode(tokens)==text
assert encoder.encoder["<|endoftext|>"]==50256 and encoder.decode([140])=="\ufffd"
assert len(module.bytes_to_unicode())==len(set(module.bytes_to_unicode().values()))==256
print("PASS actual original tokenizer IDs/round trips, byte bijection and incomplete UTF-8")

source=(SOURCE/"model.py").read_text()
excerpt=re.search(r'\x60\x60\x60python title="Original implementation — Modified MIT"\n(.*?)\n\x60\x60\x60',body,re.S)[1]
assert excerpt in source
assert (SOURCE/"LICENSE").read_bytes()==(ROOT/"site/public/licenses/openai-gpt2-Modified-MIT.txt").read_bytes()
mask_fn=next(f for f in ast.parse(source).body if isinstance(f,ast.FunctionDef) and f.name=="attention_mask")
ns={"tf":SimpleNamespace(range=torch.arange,cast=lambda x,dtype:x.to(dtype))}
exec(compile(ast.fix_missing_locations(ast.Module(body=[mask_fn],type_ignores=[])),"original:attention_mask","exec"),ns)
for q,k in [(1,5),(2,5),(5,5),(3,8)]:
    torch.testing.assert_close(ns["attention_mask"](q,k,dtype=torch.bool),
        torch.tensor([[j<=k-q+i for j in range(k)] for i in range(q)]))
top_fn=next(f for f in ast.parse((SOURCE/"sample.py").read_text()).body
            if isinstance(f,ast.FunctionDef) and f.name=="top_k_logits")
tf=SimpleNamespace(equal=lambda a,b:a==b,cond=lambda p,a,b:a() if p else b(),
    nn=SimpleNamespace(top_k=lambda x,k:torch.topk(x,k)),newaxis=None,
    where=torch.where,ones_like=torch.ones_like)
ns={"tf":tf}
exec(compile(ast.fix_missing_locations(ast.Module(body=[top_fn],type_ignores=[])),"original:top_k_logits","exec"),ns)
for z in [torch.tensor([2.,1.,0.,-1.]),torch.tensor([2.,1.,1.,0.])]:
    for k in [0,1,2,4]:
        torch.testing.assert_close(ns["top_k_logits"](z[None],k).softmax(-1)[0],dist(z,k=k))
print("PASS unchanged licensed excerpt; original mask/top-k via tensor-primitive shim, not a TensorFlow run")

# Run the actual source body with a toy model adapter returning delta K/V.
# Only accumulation/order are tested: the draw is deterministic and filters are off.
sample_tree=ast.parse((SOURCE/"sample.py").read_text())
body_fn=next(f for f in ast.walk(sample_tree)
             if isinstance(f,ast.FunctionDef) and f.name=="body")
def pack_cache(cache):
    return torch.stack([torch.stack((k,v),dim=1) for k,v in cache],dim=1)
def unpack_cache(past):
    return None if past is None else [(past[:,i,0],past[:,i,1])
                                      for i in range(len(model.blocks))]
calls=[]
def source_step(_hparams,tokens,past=None):
    length=0 if past is None else past.shape[-2]
    logits,whole=model(tokens,unpack_cache(past))
    delta=pack_cache(whole)[...,length:,:]
    calls.append((length,tokens.shape[1],delta.shape[-2],logits.detach().clone()))
    return {"logits":logits,"presents":delta}
def no_top_p(logits,p):
    assert p==1
    return logits
body_tf=SimpleNamespace(to_float=float,int32=torch.int32,
    multinomial=lambda logits,num_samples,output_dtype:logits.argmax(-1,keepdim=True).to(output_dtype),
    concat=lambda values,axis:torch.cat(values,dim=axis))
body_ns={"step":source_step,"hparams":None,"temperature":1.,"top_k":0,"top_p":1.,
         "top_k_logits":ns["top_k_logits"],"top_p_logits":no_top_p,"tf":body_tf}
exec(compile(ast.fix_missing_locations(ast.Module(body=[body_fn],type_ignores=[])),
             "original:sample_sequence.body","exec"),body_ns)
with torch.no_grad():
    output=ids[:,:3].clone()
    past,prev=None,output
    for generated in range(1,5):
        past,prev,output=body_ns["body"](past,prev,output)
        assert output.shape[1]==3+generated
        assert past.shape[-2]==3+generated-1
        torch.testing.assert_close(prev.long(),output[:,-1:])
        reference_logits,reference_cache=model(output[:,:-1])
        torch.testing.assert_close(calls[-1][-1][:,-1],reference_logits[:,-1],atol=1e-10,rtol=1e-10)
        torch.testing.assert_close(past,pack_cache(reference_cache),atol=1e-10,rtol=1e-10)
    assert [c[:3] for c in calls]==[(0,3,3),(3,1,1),(4,1,1),(5,1,1)]
    _,old=model(ids[:,:3])
    _,whole=model(ids[:,3:4],old)
    wrong=unpack_cache(torch.cat((pack_cache(old),pack_cache(whole)),dim=-2))
    assert wrong[0][0].shape[-2]==7 and whole[0][0].shape[-2]==4
    bad_logits,_=model(ids[:,4:5],wrong)
    right_logits,_=model(ids[:,4:5],whole)
    assert (bad_logits-right_logits).abs().max()>1e-6
print("PASS executed original sampling body: delta contract, source call lengths, P+G-1 and duplicated-prefix negative control")

note=(ROOT/"Papers/GPT 2.0.md").read_text()
for fact in ["124M", "355M", "774M", "52.66%", "63.24%", "18.34", "42.16",
             "26.58", "21.40", "15.03", "93.45", "93.30", "127 тыс.",
             "демонстрации в контексте"]:
    assert fact in note
for stale in ["Date: 2019-01-02","ROUGE-L ~15","Все метрики монотонно", "DeepNorm в DeepSeek"]:
    assert stale not in note
print("PASS legacy note model sizes, actual evaluation rows, summary metrics, demonstrations and non-monotonic CBT-CN")
print("PASS chapter SHA256",hashlib.sha256(CHAPTER.read_bytes()).hexdigest())
