"""Execute the published blocks; independently verify every numerical mechanism."""
from pathlib import Path
import math
import re
import torch
import torch.nn.functional as F
import numpy as np
from scipy.integrate import quad

ROOT=Path(__file__).resolve().parents[2]
CHAPTER=ROOT/"00 Учебник/01 Основы нейронных сетей/05 Инициализация, нормализация и residual connections.md"
text=CHAPTER.read_text()
assert not re.search(r'^\$$',text,re.M)
assert not re.search(r'`{1,2}(?=\d)',text), 'section references must not open inline code'
blocks=re.findall(r'```python\n(.*?)```',text,re.S)
assert len(blocks)==3
for i,code in enumerate(blocks,1):
    exec(compile(code,f"{CHAPTER}:block{i}","exec"),{})
    print("PASS published block",i)
np.testing.assert_allclose(.9**100,.0000265614,atol=1e-10)
np.testing.assert_allclose(1.1**100,13780.61234,atol=1e-5)
dtype=torch.float64

# Incoming AND outgoing symmetry, and a counterexample with different outgoing weights.
for out,prediction,grad in [([.5,.5],2.,[2.,2.]),([1.,2.],6.,[12.,24.])]:
    w=torch.tensor([1.,1.],dtype=dtype,requires_grad=True)
    b=torch.zeros(2,dtype=dtype,requires_grad=True)
    v=torch.tensor(out,dtype=dtype,requires_grad=True)
    h=w*2+b;pred=v@h;(.5*pred.square()).backward()
    assert pred.item()==prediction
    torch.testing.assert_close(w.grad,torch.tensor(grad,dtype=dtype))
    if out==[.5,.5]:
        torch.testing.assert_close(v.grad,torch.tensor([4.,4.],dtype=dtype))
        for _ in range(5):
            with torch.no_grad():
                for p in (w,b,v):p-=.1*p.grad
            for p in (w,b,v):p.grad=None
            (.5*((v*(w*2+b)).sum()).square()).backward()
            for p in (w,b,v):assert p[0].item()==p[1].item()

# Exact ensemble with constant input, and actual initializer scaling.
s=1/math.sqrt(2)
np.testing.assert_allclose(np.var([3*a+3*b for a in (-s,s) for b in (-s,s)]),9)
np.testing.assert_allclose([1000*.01**2,1000,4*.1,16*.1,math.sqrt(6/20)],[.1,1000,.4,1.6,.5477225575],atol=1e-10)
for init,std in [(torch.nn.init.xavier_normal_,math.sqrt(.1)),
                 (lambda w,generator:torch.nn.init.kaiming_normal_(w,nonlinearity="relu",generator=generator),math.sqrt(.5))]:
    a=torch.Generator().manual_seed(41);b=torch.Generator().manual_seed(41)
    w=torch.empty(16,4,dtype=dtype);init(w,generator=a)
    torch.testing.assert_close(w,torch.randn(16,4,dtype=dtype,generator=b)*std)
pdf=lambda z:math.exp(-z*z/4)/math.sqrt(4*math.pi)
mean=quad(lambda z:z*pdf(z),0,np.inf)[0]
second=quad(lambda z:z*z*pdf(z),0,np.inf)[0]
np.testing.assert_allclose([mean,second,second-mean**2],[1/math.sqrt(math.pi),1,1-1/math.pi],atol=1e-12)
for a in (0.,.1,.5,1.):
    f=lambda z:z if z>=0 else a*z
    q=quad(lambda z:f(z)**2*pdf(z),-np.inf,np.inf)[0]
    np.testing.assert_allclose(q,1+a*a,atol=1e-12)

# Batch dependence and eval exceptions; no_grad does not stop running-state updates.
eps=1e-5
bn=torch.nn.BatchNorm1d(1,affine=False,track_running_stats=False,eps=eps).double()
column=torch.tensor([[1.],[5.]],dtype=dtype)
mixed=torch.tensor([[1.],[1.],[5.]],dtype=dtype)
np.testing.assert_allclose(bn(column)[0].item(),-2/math.sqrt(4+eps),atol=1e-14)
np.testing.assert_allclose(bn(mixed)[0].item(),-4/math.sqrt(32+9*eps),atol=1e-14)
bn.eval()
assert bn.running_mean is None and bn.running_var is None
torch.testing.assert_close(bn(column),torch.tensor([[-1.],[1.]],dtype=dtype)*2/math.sqrt(4+eps))
stateful=torch.nn.BatchNorm1d(1,affine=False,momentum=1.).double()
with torch.no_grad():stateful(column)
assert stateful.training and stateful.num_batches_tracked.item()==1
torch.testing.assert_close(stateful.running_mean,torch.tensor([3.],dtype=dtype))
before=stateful.running_mean.clone();stateful.eval()
with torch.no_grad():stateful(column+20)
torch.testing.assert_close(stateful.running_mean,before)

# Temporal normalization leaks future information; per-token LN does not.
causal=torch.nn.BatchNorm1d(1,affine=False,track_running_stats=False,eps=eps).double()
seqs=[torch.tensor([[[1.,v,3.]]],dtype=dtype) for v in (5.,9.)]
np.testing.assert_allclose([causal(z)[0,0,0].item() for z in seqs],[-2/math.sqrt(8/3+eps),(-10/3)/math.sqrt(104/9+eps)],atol=1e-14)
np.testing.assert_allclose([causal(z)[0,0,0].item() for z in seqs],[-1.2247448714,-.9805806757],atol=3e-6)
x=torch.tensor([[[1.,3.],[5.,7.]]],dtype=dtype)
y=x.clone();y[:,1,:]=torch.tensor([50.,70.])
torch.testing.assert_close(F.layer_norm(x,(2,))[:,0,:],F.layer_norm(y,(2,))[:,0,:])
assert not torch.allclose(F.layer_norm(x,(2,2))[:,0,:],F.layer_norm(y,(2,2))[:,0,:])

# Forward/affine/invariances with exact epsilon qualifications.
def ln(x,eps=0.):return F.layer_norm(x,x.shape,eps=eps)
def rms(x,eps=0.):return x*torch.rsqrt(x.square().mean()+eps)
x=torch.tensor([1.,3.],dtype=dtype)
torch.testing.assert_close(ln(x),torch.tensor([-1.,1.],dtype=dtype))
torch.testing.assert_close(ln(x+100),ln(x))
torch.testing.assert_close(rms(x),x/math.sqrt(5))
np.testing.assert_allclose(rms(x).mean().item(),2/math.sqrt(5))
gain=torch.tensor([2.,1.],dtype=dtype)
torch.testing.assert_close(ln(x)*gain+torch.tensor([.5,-.5],dtype=dtype),torch.tensor([-1.5,.5],dtype=dtype))
np.testing.assert_allclose((rms(x)*gain).square().mean().sqrt().item(),math.sqrt(1.3))
for fun in (ln,rms):
    torch.testing.assert_close(fun(x*3),fun(x))
    torch.testing.assert_close(fun(-x),-fun(x))
    assert not torch.allclose(fun(x*.1,.1),fun(x,.1))
assert not torch.allclose(rms(x+100),rms(x))

# Explicit VJPs vs autograd, gamma/beta derivatives, and finite differences.
for eps in (0.,1e-5,.1):
    for name in ("ln","rms"):
        x=torch.tensor([1.,2.,3.],dtype=dtype,requires_grad=True)
        gain=torch.tensor([2.,1.,.5],dtype=dtype,requires_grad=True)
        bias=torch.tensor([.1,.2,.3],dtype=dtype,requires_grad=True)
        g=torch.tensor([1.,0.,0.],dtype=dtype)
        raw=x-x.mean() if name=="ln" else x
        denom=(raw.square().mean()+eps).sqrt();z=raw/denom
        ((z*gain+(bias if name=="ln" else 0))*g).sum().backward()
        u=gain.detach()*g;zd=z.detach()
        expected=(u-zd*(u*zd).mean())/denom.detach()
        if name=="ln":expected-=u.mean()/denom.detach()
        torch.testing.assert_close(x.grad,expected)
        torch.testing.assert_close(gain.grad,g*zd)
        if name=="ln":torch.testing.assert_close(bias.grad,g)
        fun=lambda q:(ln(q,eps) if name=="ln" else rms(q,eps))
        assert torch.autograd.gradcheck(fun,(x.detach().requires_grad_(),))
x=torch.tensor([1.,1.],dtype=dtype,requires_grad=True)
ln(x,.01)[0].backward()
torch.testing.assert_close(x.grad,torch.tensor([5.,-5.],dtype=dtype))

# Shortcut qualification, shapes, cancellation, zero last projection learns first.
x=torch.tensor(-1.,requires_grad=True);torch.relu(x+0*x).backward();assert x.grad.item()==0
x=torch.tensor(2.,requires_grad=True);(x-x).backward();assert x.grad.item()==0
assert torch.nn.Conv2d(64,128,1,stride=2)(torch.zeros(1,64,32,32)).shape==(1,128,16,16)
for initial in (0.,2.):
    w1=torch.tensor(initial,requires_grad=True);w2=torch.tensor(0.,requires_grad=True)
    (.5*(1+w2*F.relu(w1)-2)**2).backward()
    assert w1.grad.item()==0 and w2.grad.item()==(-2 if initial==2 else 0)
    if initial==2:
        with torch.no_grad():w2-=.1*w2.grad
        (.5*(1+w2*F.relu(w1)-2)**2).backward()
        assert w1.grad.item()!=0

# Pre/post full Jacobians, not only output values.
x=torch.tensor([1.,2.,3.],dtype=dtype)
A=torch.tensor([[.1,.2,0.],[0.,-.1,.1],[.3,0.,.2]],dtype=dtype)
N=lambda z:ln(z,1e-5)
pre=lambda z:z+A@N(z);post=lambda z:N(z+A@z)
JN=torch.autograd.functional.jacobian(N,x)
torch.testing.assert_close(torch.autograd.functional.jacobian(pre,x),torch.eye(3,dtype=dtype)+A@JN)
torch.testing.assert_close(torch.autograd.functional.jacobian(post,x),torch.autograd.functional.jacobian(N,x+A@x)@(torch.eye(3,dtype=dtype)+A))
for mode in ("pre","post"):
    v=x.clone().requires_grad_();z=v if mode=="pre" else ln(v);z[0].backward()
    target=torch.tensor([1.,0.,0.],dtype=dtype) if mode=="pre" else torch.tensor([1.,-2.,1.],dtype=dtype)/(6*math.sqrt(2/3))
    torch.testing.assert_close(v.grad,target)
assert math.sqrt(1+99)==10
np.testing.assert_allclose(99*(1/math.sqrt(99))**2,1)
print("PASS symmetry, Xavier/He moments/API, BN modes/state/axes, causal leakage, "
      "LN/RMS forward/affine/invariance/VJPs, FP16 overflow, residual shapes/gradients, "
      "zero-projection learning, pre/post Jacobians and residual scaling. torch",torch.__version__)
