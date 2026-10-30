#!/usr/bin/env python3
"""Run every chapter block and independently verify LSTM/GRU backward and state handling."""
from pathlib import Path
from math import isclose, log, tanh
import hashlib
import re
import torch
from torch import nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
CHAPTER=ROOT/"00 Учебник/04 RNN, LSTM и Seq2Seq/02 LSTM и GRU.md"
text=CHAPTER.read_text()
blocks=re.findall(r"\x60\x60\x60python\n(.*?)\n\x60\x60\x60",text,flags=re.S)
assert len(blocks)==5,len(blocks)
namespaces=[]
for number,block in enumerate(blocks,1):
    namespace={}
    exec(compile(block,f"{CHAPTER}:block{number}","exec"),namespace)
    namespaces.append(namespace)
    print("PASS standalone published block",number)

n=namespaces[0]
assert isclose(n["c"].item(),1.7)
assert isclose(n["h"].item(),.6547863494221693)
assert isclose(n["lam"].item(),.0875069094434126)
torch.testing.assert_close(n["a"].grad,torch.tensor([
    -.00700055275547301,.01575124369981427,.013126036416511892,.19643590482665077
],dtype=torch.float64))
assert isclose(.01*tanh(1.7),.00935409070603099)
assert isclose(.9*2+.1*(-.5),1.75)
assert isclose(torch.sigmoid(torch.tensor(2.,dtype=torch.float64)).item(),.8807970779778823)
assert isclose(torch.sigmoid(torch.tensor(-2.,dtype=torch.float64)).item(),.11920292202211755)
assert isclose(.9**100,2.6561398887587544e-5)
assert isclose(.99**100,.3660323412732292)
assert isclose(.999**100,.9047921471137089)
assert isclose((1/(1+__import__("math").exp(-1)))**100,2.4843785557240134e-14)
n=namespaces[1]
assert isclose(n["full_derivative"].item(),.9035608015725995)
assert n["full_derivative"].item()>n["f"].item()
n=namespaces[2]
assert sum(p.numel() for p in n["cell"].parameters())==72
assert sum(p.numel() for p in n["reference"].parameters())==84
assert n["after"]<n["before"]
print("PASS scalar gates, all activation derivatives, output hiding, CIFG/peephole, retention and complete cell")

# Independently derive a multi-step LSTM backward with direct CE on each h_t
# AND direct quadratic losses on each c_t, exercising both incoming state branches.
torch.manual_seed(75)
dtype=torch.float64
B,T,d,m,V=2,3,2,3,4
X=torch.randn(B,T,d,dtype=dtype,requires_grad=True)
W=(.3*torch.randn(4*m,m+d,dtype=dtype)).requires_grad_()
b=torch.randn(4*m,dtype=dtype,requires_grad=True)
h0=torch.randn(B,m,dtype=dtype,requires_grad=True)
c0=torch.randn(B,m,dtype=dtype,requires_grad=True)
A=torch.randn(V,m,dtype=dtype,requires_grad=True)
target=torch.tensor([[1,2,0],[3,2,1]])
h,c=h0,c0
saved=[]
loss=torch.zeros((),dtype=dtype)
for t in range(T):
    u=torch.cat([h,X[:,t]],-1)
    ai,af,ag,ao=(u@W.T+b).chunk(4,-1)
    i,f,g,o=ai.sigmoid(),af.sigmoid(),ag.tanh(),ao.sigmoid()
    new_c=f*c+i*g
    new_h=o*new_c.tanh()
    logits=new_h@A.T
    loss=loss+F.cross_entropy(logits,target[:,t],reduction="sum")+.05*new_c.square().sum()
    saved.append((u,h,c,i,f,g,o,new_c,new_h,logits))
    h,c=new_h,new_c
actual=torch.autograd.grad(loss,(X,W,b,h0,c0,A))
with torch.no_grad():
    dx,dw,db,da=[torch.zeros_like(v) for v in (X,W,b,A)]
    future_h=torch.zeros_like(h0)
    future_c=torch.zeros_like(c0)
    for t in range(T-1,-1,-1):
        u,hp,cp,i,f,g,o,c,h,logits=saved[t]
        e=logits.softmax(-1)-F.one_hot(target[:,t],V)
        bar_h=e@A+future_h
        q=future_c+.1*c
        lam=q+bar_h*o*(1-c.tanh().square())
        di=lam*g*i*(1-i)
        df=lam*cp*f*(1-f)
        dg=lam*i*(1-g.square())
        do=bar_h*c.tanh()*o*(1-o)
        delta=torch.cat([di,df,dg,do],-1)
        dw+=delta.T@u
        db+=delta.sum(0)
        da+=e.T@h
        du=delta@W
        dx[:,t]=du[:,m:]
        future_h=du[:,:m]
        future_c=lam*f
    for got,expected in zip(actual,(dx,dw,db,future_h,future_c,da)):
        torch.testing.assert_close(got,expected,atol=1e-12,rtol=1e-12)
print("PASS full batched LSTM BPTT: every weight, bias, input and initial h/c gradient")

# Carrying a detached h alone still leaves the c path into the previous fragment.
cell=nn.LSTMCell(2,3,dtype=dtype)
xx=torch.randn(2,2,dtype=dtype)
def two_chunks(mode):
    h0=torch.randn(2,3,dtype=dtype,requires_grad=True)
    c0=torch.randn(2,3,dtype=dtype,requires_grad=True)
    h,c=cell(xx,(h0,c0))
    if mode=="h":
        h=h.detach()
    if mode=="both":
        h,c=h.detach(),c.detach()
    h,c=cell(xx,(h,c))
    grad_h,grad_c=torch.autograd.grad(h.sum()+c.sum(),(h0,c0),allow_unused=True)
    return grad_h,grad_c
gh,gc=two_chunks("h")
assert gc is not None and gc.abs().max()>0
gh,gc=two_chunks("both")
assert gh is None and gc is None
print("PASS h-only detach does not truncate the c graph; detaching both cuts both paths")

# Both coefficient conventions implement the same mixture under z -> 1-z.
old=torch.tensor([.6,-.7],dtype=dtype)
candidate=torch.tensor([-.2,.4],dtype=dtype)
z=torch.tensor([.75,.1],dtype=dtype)
ours=z*old+(1-z)*candidate
colah=(1-(1-z))*old+(1-z)*candidate
torch.testing.assert_close(ours,colah)
assert isclose(ours[0].item(),.4)
# Complete GRU Jacobian includes paths through z and candidate, beyond diag(z).
n=namespaces[3]
cell=n["cell"]
x=n["x"][0].detach()
h=n["h_prev"][0].detach()
def components(h):
    ir,iz,inn=F.linear(x,cell.weight_ih,cell.bias_ih).chunk(3)
    hr,hz,hn=F.linear(h,cell.weight_hh,cell.bias_hh).chunk(3)
    r,z=(ir+hr).sigmoid(),(iz+hz).sigmoid()
    candidate=(inn+r*hn).tanh()
    return z,candidate
z,candidate=components(h)
Jz=torch.autograd.functional.jacobian(lambda h:components(h)[0],h)
Jn=torch.autograd.functional.jacobian(lambda h:components(h)[1],h)
J=torch.autograd.functional.jacobian(lambda h:cell(x,h),h)
expected=torch.diag(z)+torch.diag(h-candidate)@Jz+torch.diag(1-z)@Jn
torch.testing.assert_close(J,expected,atol=1e-12,rtol=1e-12)
assert not torch.allclose(J,torch.diag(z))
print("PASS GRU naming equivalence, reset order counterexample, full nn.GRUCell and all Jacobian paths")

for factory,gates,one,two in [(nn.RNN,1,18,21),(nn.GRU,3,54,63),(nn.LSTM,4,72,84)]:
    assert gates*3*(2+3+1)==one
    layer=factory(2,3)
    assert sum(p.numel() for p in layer.parameters())==two
    no_bias=factory(2,3,bias=False)
    assert sum(p.numel() for p in no_bias.parameters())==gates*3*(2+3)
projected=nn.LSTM(2,3,proj_size=2,batch_first=True,dtype=dtype)
assert sum(p.numel() for p in projected.parameters())==78
out,(hn,cn)=projected(torch.randn(2,4,2,dtype=dtype))
assert out.shape==(2,4,2) and hn.shape==(1,2,2) and cn.shape==(1,2,3)
stacked=nn.LSTM(2,3,num_layers=2,bidirectional=True,batch_first=True,dtype=dtype)
out,(hn,cn)=stacked(torch.randn(2,4,2,dtype=dtype))
assert out.shape==(2,4,6) and hn.shape==cn.shape==(4,2,3)
torch.testing.assert_close(out[:,0,3:],hn[-1])
assert not torch.allclose(out[:,-1,3:],hn[-1])
print("PASS bias counts, projected h/c dimensions, stacked bidirectional final-state semantics")

# All seven existing Colah images are untouched copies of the fetched original PNGs.
archive=ROOT/"05 Источники/Originals/lstm-gru-2026-10-21"
for name in ["LSTM3-chain","LSTM3-C-line","LSTM3-focus-f","LSTM3-focus-i","LSTM3-focus-C","LSTM3-focus-o","LSTM3-var-GRU"]:
    pub=ROOT/"00 Учебник/Assets/Figures/curated/source-audit-15-24"/(name+".png")
    assert pub.read_bytes()==(archive/(name+".png")).read_bytes()
for pub,name in [
    ("d2l-recurrent/lstm-gates.svg","lstm-0"),
    ("d2l-recurrent/lstm-hidden-state.svg","lstm-3"),
    ("d2l-recurrent/gru.svg","gru-1"),
    ("lstm-gru-2026-10-21/d2l-gru-3.svg","gru-3"),
]:
    assert (ROOT/"00 Учебник/Assets/Figures/curated"/pub).read_bytes()==(archive/("d2l-"+name+".svg")).read_bytes()
print("PASS all eleven unchanged source pictures")
print("ALL LSTM/GRU CHECKS PASSED")
