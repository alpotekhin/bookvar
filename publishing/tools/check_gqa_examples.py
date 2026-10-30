"""Execute the published GQA cases and independently check grouping/cache semantics."""
from pathlib import Path
import ast,hashlib,itertools,math,re
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
CHAPTER=ROOT/'00 Учебник/08 Эффективный Attention и длинный контекст/01 MHA, MQA и GQA.md'
body=CHAPTER.read_text()
blocks=re.findall(r'~~~python\n(.*?)\n~~~',body,re.S)
assert len(blocks)==3
torch.set_num_threads(1)
ns={}
for i,code in enumerate(blocks,1):
    exec(compile(code,f'{CHAPTER.name}:block{i}','exec'),ns)
    print('PASS actual published block',i)
grouped=ns['grouped_attention']
prob=math.exp(1)/(math.exp(1)+1)
q=math.sqrt(2)*torch.eye(2,dtype=torch.float64).reshape(1,2,1,2)
k=torch.tensor([[[[1.,0.],[0.,1.]]]],dtype=torch.float64)
v=torch.tensor([[[[2.,0.],[0.,4.]]]],dtype=torch.float64)
out,cache=grouped(q,k[:,:,1:],v[:,:,1:],(k[:,:,:1],v[:,:,:1]))
expected=torch.tensor([[2*prob,4*(1-prob)],[2*(1-prob),4*prob]],dtype=torch.float64)
torch.testing.assert_close(out[0,:,0],expected)
assert cache[0].shape==(1,1,2,2)
torch.testing.assert_close(ns['before'],torch.tensor([4-2*prob,2+2*prob],dtype=torch.float64))
assert ns['after'].item()==3
print('PASS shared-KV outputs and numerical mean-pooling counterexample')

def by_head(q,k,v,old=0):
    batch,heads,tokens,dh=q.shape
    r=heads//k.shape[1]
    result=[]
    for head in range(heads):
        rows=[]
        for t in range(tokens):
            count=old+t+1
            weights=(q[:,head,t:t+1]@k[:,head//r,:count].transpose(-2,-1)/math.sqrt(dh)).softmax(-1)
            rows.append(weights@v[:,head//r,:count])
        result.append(torch.cat(rows,dim=1))
    return torch.stack(result,dim=1)

def chunks(length):
    for bits in itertools.product((False,True),repeat=length-1):
        edges=[0]+[i+1 for i,b in enumerate(bits) if b]+[length]
        yield list(zip(edges,edges[1:]))

for batch,hq,hkv in [(1,4,1),(1,4,2),(1,4,4),(2,6,2)]:
    torch.manual_seed(batch+hq+hkv)
    shape=(batch,hq,5,2)
    original=[torch.randn(shape,dtype=torch.float64),
              torch.randn(batch,hkv,5,2,dtype=torch.float64),
              torch.randn(batch,hkv,5,2,dtype=torch.float64)]
    ref_inputs=[x.clone().requires_grad_() for x in original]
    reference=by_head(*ref_inputs)
    reference.square().sum().backward()
    for parts in chunks(5):
        inputs=[x.clone().requires_grad_() for x in original]
        cache=None
        outs=[]
        for start,end in parts:
            out,cache=grouped(*(x[:,:,start:end] for x in inputs),cache)
            outs.append(out)
            assert cache[0].shape==(batch,hkv,end,2)
        result=torch.cat(outs,dim=2)
        torch.testing.assert_close(result,reference,rtol=1e-12,atol=1e-12)
        result.square().sum().backward()
        for x,ref in zip(inputs,ref_inputs):
            torch.testing.assert_close(x.grad,ref.grad,rtol=1e-11,atol=1e-11)
    repeat_k=original[1].repeat_interleave(hq//hkv,dim=1)
    repeat_v=original[2].repeat_interleave(hq//hkv,dim=1)
    allowed=torch.ones(5,5,dtype=torch.bool).tril()
    sdpa=F.scaled_dot_product_attention(original[0],repeat_k,repeat_v,attn_mask=allowed)
    torch.testing.assert_close(sdpa,reference,rtol=1e-12,atol=1e-12)
print('PASS all 16 five-token partitions, outputs and Q/K/V gradients: MQA/GQA/MHA and batch=2')

# Wrong grouping and wrong rectangular mask must not pass the positive test.
torch.manual_seed(31)
q=torch.randn(1,4,3,2,dtype=torch.float64)
k=torch.randn(1,2,5,2,dtype=torch.float64)
v=torch.randn_like(k)
expected=by_head(q,k,v,old=2)
out,_=grouped(q,k[:,:,2:],v[:,:,2:],(k[:,:,:2],v[:,:,:2]))
torch.testing.assert_close(out,expected)
correct_mask=torch.arange(5)[None,:]<=torch.arange(2,5)[:,None]
assert correct_mask.sum(-1).tolist()==[3,4,5]
wrong=F.scaled_dot_product_attention(q,k.repeat(1,2,1,1),v.repeat(1,2,1,1),attn_mask=correct_mask)
assert not torch.allclose(wrong,expected)
wrong_mask=F.scaled_dot_product_attention(q,k.repeat_interleave(2,1),v.repeat_interleave(2,1),
    attn_mask=torch.ones(3,5,dtype=torch.bool).tril())
assert not torch.allclose(wrong_mask,expected)
print('PASS negative controls: repeat ordering and unshifted causal mask')

def split_gradient(detach):
    q=torch.arange(40,dtype=torch.float64).reshape(1,4,5,2).div(20).requires_grad_()
    k=torch.arange(20,dtype=torch.float64).reshape(1,2,5,2).div(20).requires_grad_()
    v=torch.sin(k.detach()).requires_grad_()
    first,c=grouped(q[:,:,:2],k[:,:,:2],v[:,:,:2])
    if detach:
        c=tuple(x.detach() for x in c)
    rest,_=grouped(q[:,:,2:],k[:,:,2:],v[:,:,2:],c)
    y=torch.cat((first,rest),dim=2)
    y.square().sum().backward()
    return y,k.grad,v.grad
normal=split_gradient(False)
detached=split_gradient(True)
torch.testing.assert_close(normal[0],detached[0])
assert not torch.allclose(normal[1],detached[1])
assert not torch.allclose(normal[2],detached[2])
print('PASS detach preserves values but changes past-K/V gradients')

source=ROOT/'05 Источники/Originals/gqa-2026-10-22/llama-model.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='b409ea3db2a4029afd4129246d3b5e76ea95b6503364842545403dbcd7476281'
tree=ast.parse(source.read_text())
node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='repeat_kv')
scope={'torch':torch}
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),scope)
for heads,r in [(1,4),(2,2),(4,1)]:
    x=torch.arange(2*3*heads*2).reshape(2,3,heads,2)
    y=scope['repeat_kv'](x,r)
    assert torch.equal(y,x.repeat_interleave(r,dim=2))
    shared=y.untyped_storage().data_ptr()==x.untyped_storage().data_ptr()
    assert shared==(heads==1 or r==1)
    print('PASS pinned Llama repeat_kv',heads,r,'shared storage',shared)

for hkv,kv_bytes,per_device,total in [(32,2**32,512*2**20,2**32),
                                    (8,2**30,128*2**20,2**30),
                                    (1,128*2**20,128*2**20,2**30)]:
    assert 2*32*8192*hkv*128*2==kv_bytes
    per_head=2*32*8192*128*2
    local_heads=hkv//8 if hkv>=8 else 1  # declared TP scheme only
    assert local_heads*per_head==per_device
    assert 8*per_device==total
    d=4096
    mats=[torch.empty(d,d,device='meta'),torch.empty(d,d,device='meta'),
          torch.empty(d,hkv*128,device='meta'),torch.empty(d,hkv*128,device='meta')]
    p=sum(m.numel() for m in mats)
    assert p=={32:67108864,8:41943040,1:34603008}[hkv]
    pair_work=sum(4*128 for _ in range(8192) for _ in range(32))
    assert pair_work==134217728
    moved=2*(2*4096+2*8192*hkv*128)
    intensity=pair_work/moved
    assert f'{intensity:.2f}' in body
assert 1-41943040/67108864==.375
print('PASS logical/physical TP cache, parameter inventories and declared one-read intensity')
print('No GPU timing, HBM traffic, distributed execution or model quality measured')
print('chapter SHA256',hashlib.sha256(CHAPTER.read_bytes()).hexdigest())
