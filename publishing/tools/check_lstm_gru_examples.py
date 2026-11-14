#!/usr/bin/env python3
"""Run every chapter block and independently verify LSTM/GRU backward and state handling."""
from pathlib import Path
from math import isclose, log, tanh
import hashlib
import re
import torch
from torch import nn
import torch.nn.functional as F
from topic_consolidation_baseline import before_topic_consolidation

ROOT=Path(__file__).resolve().parents[2]
CHAPTER=ROOT/"00 Учебник/04 RNN, LSTM и Seq2Seq/02 LSTM и GRU.md"
text=CHAPTER.read_text()
blocks=re.findall(r"\x60\x60\x60python\n(.*?)\n\x60\x60\x60",text,flags=re.S)
assert len(blocks)==8,len(blocks)
namespaces=[]
new_namespaces={}
for number,block in enumerate(blocks,1):
    namespace={}
    exec(compile(block,f"{CHAPTER}:block{number}","exec"),namespace)
    if "biased_gru" in namespace:
        new_namespaces["biases"]=namespace
    elif "bidirectional_lstm" in namespace:
        new_namespaces["directions"]=namespace
    elif "projected_lstm" in namespace:
        new_namespaces["projection"]=namespace
    else:
        namespaces.append(namespace)
    print("PASS standalone published block",number)

assert len(namespaces)==5 and len(new_namespaces)==3
baseline=before_topic_consolidation(text,str(CHAPTER.relative_to(ROOT)))
assert hashlib.sha256(baseline.encode()).hexdigest()=="c0afc59f36af36d3616792c6dc8873cdee8ce419fa8cf0db000698f2393283f4"
old_blocks=re.findall(r"\x60\x60\x60python\n(.*?)\n\x60\x60\x60",baseline,flags=re.S)
assert [b for b in blocks if not any(k in b for k in
    ("biased_gru","bidirectional_lstm","projected_lstm"))]==old_blocks

n=namespaces[0]
assert isclose(n["c"].item(),1.7)
assert isclose(n["h"].item(),.6547863494221693)
assert isclose(n["lam"].item(),.0875069094434126)
torch.testing.assert_close(n["a"].grad,torch.tensor([
    -.00700055275547301,.01575124369981427,.013126036416511892,.19643590482665077
],dtype=torch.float64))
assert isclose(.01*tanh(1.7),.00935409070603099)
# The same weights realize both steps using x=0 and x=1.
dtype=torch.float64
trace_cell=nn.LSTMCell(1,1,dtype=dtype)
ai=torch.logit(torch.tensor(.2,dtype=dtype))
before=torch.stack([ai,torch.logit(torch.tensor(.9,dtype=dtype)),
                    torch.atanh(torch.tensor(-.5,dtype=dtype)),torch.logit(torch.tensor(.01,dtype=dtype))])
after=torch.stack([ai,torch.logit(torch.tensor(.99,dtype=dtype)),
                   torch.tensor(0.,dtype=dtype),torch.logit(torch.tensor(.7,dtype=dtype))])
with torch.no_grad():
    trace_cell.bias_ih.copy_(before)
    trace_cell.bias_hh.zero_()
    trace_cell.weight_ih[:,0].copy_(after-before)
    trace_cell.weight_hh.zero_()
h0,c0=torch.zeros(1,1,dtype=dtype),torch.full((1,1),2.,dtype=dtype)
h1,c1=trace_cell(h0,(h0,c0))
h2,c2=trace_cell(torch.ones_like(h0),(h1,c1))
assert isclose(c1.item(),1.7)
assert isclose(h1.item(),.00935409070603099)
assert isclose(c2.item(),1.683)
assert isclose(h2.item(),.6532748416398817)
_,without_memory=trace_cell(torch.ones_like(h0),(h1,torch.zeros_like(c1)))
assert without_memory.abs().max().item()<1e-12
print("PASS two-step LSTM with shared weights: small h, carried c, then restored readout")
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

# The seven historical Colah images remain byte-identical, even when not embedded.
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
print("PASS all eleven historical source assets preserved")
# Independent gradient checks for projected recurrent states, not just output shapes.
torch.manual_seed(1103)
projection_cases=0
for B in (1,2):
  for T in (1,3):
    for d in (1,2):
      for m in (2,3,5):
        for p in range(1,m):
          layer=nn.LSTM(d,m,proj_size=p,batch_first=True,dtype=dtype)
          x=torch.randn(B,T,d,dtype=dtype,requires_grad=True)
          h0=torch.randn(1,B,p,dtype=dtype,requires_grad=True)
          c0=torch.randn(1,B,m,dtype=dtype,requires_grad=True)
          native,(hn,cn)=layer(x,(h0,c0))
          h,c=h0[0],c0[0]
          history=[]
          for t in range(T):
            branches=[]
            for gate in range(4):
              sl=slice(gate*m,(gate+1)*m)
              branches.append(x[:,t]@layer.weight_ih_l0[sl].T
                + h@layer.weight_hh_l0[sl].T
                + layer.bias_ih_l0[sl]+layer.bias_hh_l0[sl])
            i,f,g,o=branches
            c=f.sigmoid()*c+i.sigmoid()*g.tanh()
            h=(o.sigmoid()*c.tanh())@layer.weight_hr_l0.T
            history.append(h)
          manual=torch.stack(history,1)
          torch.testing.assert_close(manual,native)
          torch.testing.assert_close(h[None],hn)
          torch.testing.assert_close(c[None],cn)
          leaves=(x,h0,c0,*layer.parameters())
          loss=manual.square().sum()+.2*h.square().sum()+.3*c.square().sum()
          reference=native.square().sum()+.2*hn.square().sum()+.3*cn.square().sum()
          for a,b in zip(torch.autograd.grad(loss,leaves),
                         torch.autograd.grad(reference,leaves)):
            torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-11)
          assert sum(v.numel() for v in layer.parameters())==4*m*(d+p+2)+p*m
          unprojected=4*m*(d+m+2)
          assert 4*m*(d+p+2)+p*m-unprojected==m*(5*p-4*m)
          projection_cases+=1
assert projection_cases==56
assert sum(v.numel() for v in nn.LSTM(2,6,proj_size=5).parameters())==246
assert sum(v.numel() for v in nn.LSTM(2,6).parameters())==240
print("PASS",projection_cases,"projected LSTM state/all-gradient and parameter-count cases")

# Explicit independent direction cells, with copied parameters and nonzero boundaries.
direction_cases=0
for factory,cell_factory in ((nn.LSTM,nn.LSTMCell),(nn.GRU,nn.GRUCell)):
  for B in (1,2):
    for T in (1,2,4):
      for d in (1,3):
        for m in (1,2,3):
          layer=factory(d,m,batch_first=True,bidirectional=True,dtype=dtype)
          x=torch.randn(B,T,d,dtype=dtype,requires_grad=True)
          h0=torch.randn(2,B,m,dtype=dtype,requires_grad=True)
          c0=torch.randn(2,B,m,dtype=dtype,requires_grad=True) if factory==nn.LSTM else None
          out,state=layer(x,(h0,c0) if c0 is not None else h0)
          native_h=state[0] if c0 is not None else state
          cells=[];outputs=[];finals_h=[];finals_c=[]
          for direction in (0,1):
            cell=cell_factory(d,m,dtype=dtype)
            suffix="_reverse" if direction else ""
            with torch.no_grad():
              for name in ("weight_ih","weight_hh","bias_ih","bias_hh"):
                getattr(cell,name).copy_(getattr(layer,name+"_l0"+suffix))
            cells.append(cell)
            h=h0[direction];c=c0[direction] if c0 is not None else None
            aligned=[None]*T
            for t in (range(T-1,-1,-1) if direction else range(T)):
              if c is None:
                h=cell(x[:,t],h)
              else:
                h,c=cell(x[:,t],(h,c))
              aligned[t]=h
            outputs.append(torch.stack(aligned,1))
            finals_h.append(h)
            if c is not None:finals_c.append(c)
          manual=torch.cat(outputs,-1)
          manual_h=torch.stack(finals_h)
          torch.testing.assert_close(manual,out)
          torch.testing.assert_close(manual_h,native_h)
          torch.testing.assert_close(out[:,-1,:m],native_h[0])
          torch.testing.assert_close(out[:,0,m:],native_h[1])
          original_leaves=(x,h0,c0) if c0 is not None else (x,h0)
          lhs=manual.square().sum()+.2*manual_h.square().sum()
          rhs=out.square().sum()+.2*native_h.square().sum()
          if c0 is not None:
            manual_c=torch.stack(finals_c)
            torch.testing.assert_close(manual_c,state[1])
            lhs=lhs+.3*manual_c.square().sum()
            rhs=rhs+.3*state[1].square().sum()
          a=torch.autograd.grad(lhs,(*original_leaves,*(p for cell in cells for p in cell.parameters())))
          b=torch.autograd.grad(rhs,(*original_leaves,*layer.parameters()))
          for got,expected in zip(a,b):torch.testing.assert_close(got,expected,atol=1e-11,rtol=1e-11)
          direction_cases+=1
assert direction_cases==72
print("PASS",direction_cases,"LSTM/GRU direction-alignment, nonzero state and all-gradient cases")

# Distinct candidate-bias gradients: b_hn receives the reset multiplier.
for case in range(80):
  B=(1,2,3,5)[case%4]
  cell=nn.GRUCell(1,1,dtype=dtype)
  with torch.no_grad():
    for parameter in cell.parameters():parameter.zero_()
    cell.weight_ih[0,0]=1.
    cell.bias_ih[2]=.1
    cell.bias_hh[2]=.8
  rates=.01+.98*torch.rand(B,1,dtype=dtype)
  actual=cell(torch.logit(rates),torch.zeros_like(rates))
  torch.testing.assert_close(actual,.5*(.1+.8*rates).tanh())
  bih,bhh=torch.autograd.grad(actual.sum(),(cell.bias_ih,cell.bias_hh))
  g=.5*(1-(.1+.8*rates).tanh().square())
  torch.testing.assert_close(bih[2],g.sum())
  torch.testing.assert_close(bhh[2],(rates*g).sum())
  assert not torch.allclose(bih[2],bhh[2])
print("PASS 80 reset-weighted candidate-bias forward/gradient cases")

new_asset=ROOT/"00 Учебник/Assets/Figures/curated/lstm-gru-contracts-2026-11-03/d2l-birnn.svg"
assert hashlib.sha256(new_asset.read_bytes()).hexdigest()=="aa4e678afa0e31c160374974709512d4d7e0121eeab7a1af1ad7725403afc018"
assert new_asset.read_bytes()==(ROOT/"05 Источники/Originals/lstm-gru-contracts-2026-11-03/d2l-birnn.svg").read_bytes()
for filename in ["d2l-lstm-1.svg","d2l-lstm-2.svg","d2l-gru-2.svg"]:
    source=ROOT/"00 Учебник/Assets/Figures/curated/lstm-gru-author-2026-11-10"/filename
    assert source.read_bytes()==(archive/filename).read_bytes()
print("PASS complete pinned bidirectional SVG and three additional D2L originals")
print("ALL LSTM/GRU CHECKS PASSED")
