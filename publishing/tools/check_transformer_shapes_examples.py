"""Verify published arithmetic and actual CPU matmul calls, not GPU speed."""
from pathlib import Path
import ast,hashlib,math,re
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

ROOT=Path(__file__).resolve().parents[2]
CHAPTER=ROOT/'00 Учебник/07 Анатомия современной LLM/05 Transformer с нуля — формы, параметры и стоимость.md'
body=CHAPTER.read_text()
blocks=re.findall(r'~~~python\n(.*?)\n~~~',body,re.S)
assert len(blocks)==7
torch.set_num_threads(1)
ns={}
for i,code in enumerate(blocks,1):
    exec(compile(code,f'{CHAPTER.name}:block{i}','exec'),ns)
    print('PASS actual published block',i)
assert ns['heads'][0,0].tolist()==[[0,1],[4,5]]
assert ns['wrong'][0,0].tolist()==[[0,1],[2,3]]
assert ns['heads'].contiguous().untyped_storage().data_ptr()!=ns['x'].untyped_storage().data_ptr()
layout,cost=ns['parameter_layout'],ns['matrix_cost']
assert sum(p.numel() for p in ns['untied'].parameters())==22696448
assert sum(p.numel() for p in ns['tied'].parameters())==17576448
assert sum(p.numel() for p in ns['tied'].state_dict().values())==22696448
assert ns['tied'].head.weight is ns['tied'].embedding.weight
assert ns['inputs'].tolist()==[[0,1]] and ns['targets'].tolist()==[[1,2]]
expected_losses=[math.log(sum(math.exp(k) for k in (2,1,-2)))-1,
                 math.log(sum(math.exp(k) for k in (1,2,-1)))+1]
assert np.allclose(ns['token_nll'].tolist(),[expected_losses],rtol=0,atol=1e-12)
assert math.isclose(ns['loss'].item(),sum(expected_losses)/2,abs_tol=1e-12)
assert ns['next_logits'].tolist()==[[1,2,-1]]
assert np.allclose(ns['shared_grad'][2].tolist(),[-.4692275995665407,-.9582745295637655],
                   rtol=0,atol=1e-12)
assert np.linalg.norm(ns['input_grad'][0].tolist())>0
print('PASS shifted targets, manual cross-entropy, last row and weight-tying gradient')
assert sum(p.numel() for p in layout(50257,1600,4288,48).parameters())==1640452800
assert 50257*1600+1024*1600+48*(12*1600**2+13*1600)+2*1600==1557611200
for t,total,attention in [(1024,3516769894400,322122547200),
                          (16384,133577729638400,82463372083200)]:
    xl=cost(1,t,50257,1600,4288,48)
    assert sum(xl.values())==total and xl['attention']==attention
    assert f'{total/1e12:.3f}' in body and f'{attention/total:.1%}' in body
print('PASS A1/GPT-2 XL inventories and full-square resource table')
assert ns['forward_flops']==9265217536
assert ns['training_flops']==27795652608
for key in ('projections','ffn','attention'):
    value=ns['costs'][key]
    assert f'{value / 4 / 1e9:.6f}' in body
    assert f'{value / 1e9:.6f}' in body
assert f"{ns['costs']['head'] / 1e9:.6f}" in body
assert f"{ns['forward_flops'] / 1e9:.6f}" in body
assert sum(cost(2,256,10000,512,1344,4).values())==19067305984
assert ns['accumulated_gradient']()==7.5
assert abs(ns['accumulated_gradient'](True)-16/3)<1e-14
assert ns['global_tokens']/ns['step_seconds']==20480
assert abs(ns['mfu']-.27795652608)<1e-14
assert abs(ns['wrong_mfu']-.03474456576)<1e-14
print('PASS printed values, rounded GFLOP table, shared storage and unequal-token weighting')

# Compare parameter enumeration with the exact small course function.
source=ROOT/'05 Источники/Courses/Stanford CS336 Spring 2026/Lectures/repository/lecture_02.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='f731b3849afc970a062f1fe692d7d80f5952cf82c2a4f6f2803a8fb9a9dabfd6'
tree=ast.parse(source.read_text())
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_num_parameters')
env={'nn':torch.nn}
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(source),'exec'),env)
for vocab,dim,hidden,layers in [(7,4,10,2),(19,8,24,3),(10000,512,1344,4)]:
    for tied in (False,True):
        m=layout(vocab,dim,hidden,layers,tied)
        independent=vocab*dim+layers*(4*dim*dim+3*dim*hidden+2*dim)+dim
        if not tied:
            independent+=vocab*dim
        assert env['get_num_parameters'](m)==independent
print('PASS pinned CS336 parameter enumeration and three independent inventories')

for batch in (1,2,3):
    for tokens in (1,2,3,8,128,256):
        for dim,hidden,layers,vocab in [(4,10,2,7),(512,1344,4,10000)]:
            parts=cost(batch,tokens,vocab,dim,hidden,layers)
            # Accumulate per-layer per-token projections and each query/key pair.
            linear=layers*batch*tokens*(4*2*dim*dim+3*2*dim*hidden)
            pairs=sum(tokens for _ in range(tokens))
            attention=layers*batch*pairs*4*dim
            head=batch*tokens*2*dim*vocab
            assert sum(parts.values())==linear+attention+head
            triangular=cost(batch,tokens,vocab,dim,hidden,layers,True)
            valid_pairs=sum(range(1,tokens+1))
            assert triangular['attention']==layers*batch*valid_pairs*4*dim
print('PASS independent full-square and triangular pair counting')

class MatrixWork(TorchDispatchMode):
    """Record only dense mm/bmm metadata under the declared 2mkn convention."""
    def __init__(self):
        self.events=[]
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        output=func(*args,**(kwargs or {}))
        if func in (torch.ops.aten.mm.default,torch.ops.aten.bmm.default):
            a,b=args[:2]
            assert a.shape[-1]==b.shape[-2]
            flops=2*math.prod(a.shape[:-2])*a.shape[-2]*a.shape[-1]*b.shape[-1]
            self.events.append((str(func),tuple(a.shape),tuple(b.shape),flops))
        return output
    @property
    def total(self):
        return sum(e[-1] for e in self.events)

# A complete tiny pre-norm MHA/SwiGLU graph; RoPE reuses the actual
# previously reviewed chapter function and contributes no dense mm/bmm.
rope_chapter=ROOT/'00 Учебник/07 Анатомия современной LLM/04 RoPE.md'
rope_block=re.findall(r'~~~python\n(.*?)\n~~~',rope_chapter.read_text(),re.S)[0]
nodes=[n for n in ast.parse(rope_block).body
       if isinstance(n,ast.FunctionDef) and n.name in ('rope','frequencies')]
assert len(nodes)==2
rope_ns={'torch':torch}
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(rope_chapter),'exec'),rope_ns)
def forward(m,ids,heads,eps=1e-5,trace=None):
    x=m.embedding(ids)
    B,T,D=x.shape
    d=D//heads
    positions=torch.arange(T)
    omega=rope_ns['frequencies'](d)
    mask=torch.ones(T,T,dtype=torch.bool).tril()
    def norm(z,g):
        return z*torch.rsqrt(z.square().mean(-1,keepdim=True)+eps)*g
    for w in m.blocks:
        n=norm(x,w['norm1'])
        q,k,v=[F.linear(n,w[key]).reshape(B,T,heads,d) for key in ('q','k','v')]
        q=rope_ns['rope'](q,positions,omega).transpose(1,2)
        k=rope_ns['rope'](k,positions,omega).transpose(1,2)
        v=v.transpose(1,2)
        scores=(q@k.transpose(-2,-1))/math.sqrt(d)
        weights=scores.masked_fill(~mask,-torch.inf).softmax(-1)
        merged=(weights@v).transpose(1,2).reshape(B,T,D)
        x=x+F.linear(merged,w['o'])
        n=norm(x,w['norm2'])
        if trace is not None:
            trace.update(scores=scores,weights=weights,attention_residual=x,ffn_input=n)
        x=x+F.linear(F.silu(F.linear(n,w['gate']))*F.linear(n,w['up']),w['down'])
        if trace is not None:
            trace['ffn_residual']=x
    return F.linear(norm(x,m.final_norm),m.head.weight)

# The prose example uses one head and two positions, with all projections
# fixed to the identity and epsilon set to zero for nonzero inputs.
tiny=layout(3,2,2,1)
tiny.to_empty(device='cpu').double()
with torch.no_grad():
    tiny.embedding.weight.copy_(ns['output_rows'])
    tiny.head.weight.copy_(ns['output_rows'])
    tiny.final_norm.fill_(1.)
    for name, value in tiny.blocks[0].items():
        value.copy_(torch.eye(2, dtype=torch.float64) if value.ndim==2 else torch.ones_like(value))
trace={}
toy_logits=forward(tiny,ns['inputs'],1,eps=0.,trace=trace)
s2=math.sqrt(2.)
rows={
    'scores': (trace['scores'][0,0,1], [-s2*math.sin(1.),s2]),
    'weights': (trace['weights'][0,0,1], [.06886647254766655,.93113352745233345]),
    'attention_residual': (trace['attention_residual'][0,1], [.09739189946970447,2.316821662903391]),
    'ffn_input': (trace['ffn_input'][0,1], [.059396635739346684,1.4129656894853633]),
    'ffn_residual': (trace['ffn_residual'][0,1], [.09920825148579816,3.922450938470705]),
    'first_logits': (toy_logits[0,0], [s2,0.,-s2]),
    'last_logits': (toy_logits[0,1], [.035757439048064665,1.413761438699162,-.035757439048064665]),
}
for actual, expected in rows.values():
    torch.testing.assert_close(actual,torch.tensor(expected,dtype=actual.dtype),rtol=0,atol=2e-14)
torch.testing.assert_close(trace['weights'][0,0,0],torch.tensor([1.,0.],dtype=torch.float64))
torch.testing.assert_close(forward(tiny,ns['inputs'][:,:1],1,eps=0.),toy_logits[:,:1])
assert math.isclose(F.cross_entropy(toy_logits.flatten(0,1),ns['targets'].flatten()).item(),
                    1.7622035228662254,abs_tol=2e-14)
print('PASS complete two-position example and independent prefix-only computation')

for tied in (False,True):
    torch.manual_seed(17)
    m=layout(7,4,10,2,tied)
    m.to_empty(device='cpu').double()
    # Materialization can recreate Parameters; restore the intended alias.
    if tied:
        m.head.weight=m.embedding.weight
    for p in m.parameters():
        torch.nn.init.normal_(p,mean=0,std=.1)
    ids=torch.tensor([[1,2,3],[3,4,1]])
    expected=sum(cost(2,3,7,4,10,2).values())
    with MatrixWork() as record:
        logits=forward(m,ids,2)
    assert logits.shape==(2,3,7)
    assert record.total==expected,(record.events,expected)
    assert len(record.events)==2*9+1
    with MatrixWork() as record:
        forward(m,ids,2).square().sum().backward()
    assert record.total==3*expected,(record.total,expected)
    assert all(p.grad is not None for p in m.parameters())
print('PASS observed CPU mm/bmm calls: complete tiny LM forward/backward, tied and untied')

for input_grad,weight_grad,multiplier in [(True,True,3),(True,False,2),(False,True,2),(False,False,1)]:
    x=torch.randn(2,2,dtype=torch.float64,requires_grad=input_grad)
    w=torch.randn(2,2,dtype=torch.float64,requires_grad=weight_grad)
    with MatrixWork() as record:
        y=x@w.T
        if input_grad or weight_grad:
            y.sum().backward()
    assert record.total==16*multiplier
print('PASS frozen-input/frozen-weight backward work and no-backward control')

# Verify the reused graph against the actual author's source function.
source=ROOT/'05 Источники/Originals/transformer-shapes-2026-10-22/jax-roofline.md'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='aeb33bc64b70552a631f8102a59cca6dbae2dd08675bdfeaa9b7c8e8502e4492'
match=re.search(r'def roofline\(B, D, F\):\n(?:  [^\n]*\n)+',source.read_text())
assert match
upstream={'np':np}
exec(compile(match.group(),str(source),'exec'),upstream)
for dim in (1024,4096):
    for n in (1,16,64,128,256,512):
        expected=upstream['roofline'](n,dim,dim)
        assert math.isclose(ns['mixed_weight_roofline'](n,dim,dim),expected,rel_tol=1e-14)
    r=1.97e14/8.2e11
    critical=r*dim/(2*dim-4*r)
    compute=2*critical*dim*dim/1.97e14
    memory=(4*critical*dim+dim*dim)/8.2e11
    assert math.isclose(compute,memory,rel_tol=1e-14)
print('PASS pinned JAX graph formula and critical batch sizes')
assert 16*22696448==363143168
assert 2*2*16*4096**2==1073741824
assert math.isclose(363143168/2**20,346.3203125)
print('PASS declared memory units; not an allocated-memory or performance measurement')
print('chapter SHA256',hashlib.sha256(CHAPTER.read_bytes()).hexdigest())
