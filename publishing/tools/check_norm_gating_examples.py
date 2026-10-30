"""Execute chapter snippets and independently check normalization/stability examples."""
from pathlib import Path
import hashlib
import math
import re
import subprocess
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
CHAPTER=ROOT/"00 Учебник/07 Анатомия современной LLM/03 Pre-norm, RMSNorm, SwiGLU и residual.md"
text=CHAPTER.read_text()
blocks=re.findall(r"~~~python\n(.*?)\n~~~",text,re.S)
assert len(blocks)==4
env={}
for i,block in enumerate(blocks):
    exec(compile(block,f"{CHAPTER}:block-{i+1}","exec"),env)
print("PASS all four published blocks in chapter order")

x=np.array([1.,2.,3.])
np.testing.assert_allclose(x+np.array([.1*x[0],-.1*x[1],0]),[1.1,1.8,3])
residual_input=torch.tensor(x,requires_grad=True)
residual_output=residual_input+residual_input*torch.tensor([.1,-.1,0.],dtype=torch.float64)
residual_output.sum().backward()
np.testing.assert_allclose(residual_input.grad.numpy(),[1.1,.9,1],atol=1e-14)
center=x-x.mean()
s=np.sqrt(np.mean(center**2))
z=center/s
jn=(np.eye(3)-np.ones((3,3))/3-np.outer(z,z)/3)/s
expected_pre=x+.1*z
expected_post=z
np.testing.assert_allclose(env["pre"](env["x"]).numpy(),expected_pre,atol=1e-14)
np.testing.assert_allclose(env["post"](env["x"]).numpy(),expected_post,atol=1e-14)
for fun,expected in [(env["pre"],np.eye(3)+.1*jn),(env["post"],jn)]:
    observed=torch.autograd.functional.jacobian(fun,env["x"]).numpy()
    np.testing.assert_allclose(observed,expected,atol=1e-14)
np.testing.assert_allclose((np.eye(3)+.1*jn)[0],[1.0204,-.0408,.0204],atol=5e-5)
np.testing.assert_allclose(jn[0],[.2041,-.4082,.2041],atol=5e-5)
assert torch.autograd.functional.jacobian(lambda z:z-z,env["x"]).count_nonzero()==0
print("PASS pre/post outputs, full Jacobians and residual cancellation")

v=np.array([3.,4.,0.,0.])
eps=0.
r=np.sqrt(np.mean(v*v))
np.testing.assert_allclose(v/r,[1.2,1.6,0,0])
np.testing.assert_allclose((v-v.mean())/np.std(v),[.7001,1.2603,-.9802,-.9802],atol=5e-5)
assert math.isclose(np.mean(v/r),.7)
assert math.isclose(np.mean(v*v),np.var(v)+np.mean(v)**2)
rms=lambda a,e=0.:a/np.sqrt(np.mean(a*a)+e)
ln=lambda a,e=0.:(a-a.mean())/np.sqrt(np.var(a)+e)
for f in (rms,ln):
    np.testing.assert_allclose(f(v*3),f(v),atol=1e-14)
    np.testing.assert_allclose(f(-v),-f(v),atol=1e-14)
    assert not np.allclose(f(v*.001,.001),f(v,.001))
np.testing.assert_allclose(ln(v+10),ln(v),atol=1e-14)
assert not np.allclose(rms(v+10),rms(v))
assert not math.isclose(np.mean((rms(v)*np.array([2.,1.,1.,1.]))**2),1)
assert math.isclose((v/abs(v[0]))[1],4/3)
assert np.sqrt(np.mean(np.array([0.,0.,3.,4.])[:1]**2))==0

inp=torch.tensor(v,dtype=torch.float32,requires_grad=True)
gain=torch.ones(4,requires_grad=True)
result=env["rms_norm"](inp,gain,eps=0.)
result[0].backward()
u=np.array([1.,0.,0.,0.])
analytic=u/r-v*np.dot(u,v)/(4*r**3)
np.testing.assert_allclose(inp.grad.numpy(),analytic,atol=5e-8,rtol=0)
np.testing.assert_allclose(gain.grad.numpy(),u*v/r,atol=5e-8,rtol=0)
np.testing.assert_allclose(analytic,[.256,-.192,0,0],atol=1e-14)
np.testing.assert_allclose(np.dot(v,analytic),0,atol=1e-14)
wrong_input=torch.tensor(v,dtype=torch.float64,requires_grad=True)
wrong=(wrong_input/(wrong_input.square().mean().sqrt().detach()))
wrong[0].backward()
np.testing.assert_allclose(wrong_input.grad.numpy(),[.4,0,0,0])
step=1e-5
finite=np.empty(4)
for j in range(4):
    a=v.copy();a[j]+=step
    b=v.copy();b[j]-=step
    finite[j]=(rms(a)[0]-rms(b)[0])/(2*step)
np.testing.assert_allclose(finite,analytic,atol=1e-10,rtol=0)
print("PASS LN/RMS axes, affine/invariance qualifications, pRMS subset and independent VJP/finite differences")

small=np.array([1e-4,0,0,0])
inside=rms(small,1e-5)[0]
outside=(small/(np.sqrt(np.mean(small*small))+1e-5))[0]
np.testing.assert_allclose([inside,outside],[.03162,1.66667],atol=5e-6,rtol=0)
assert np.all(env["naive"].numpy()==0)
np.testing.assert_allclose(env["stable"].float().numpy(),[1.2,1.6,0,0],atol=5e-4,rtol=0)
zero=torch.zeros(4,requires_grad=True)
env["rms_norm"](zero,torch.ones(4))[0].backward()
np.testing.assert_allclose(zero.grad.numpy(),[1/math.sqrt(1e-5),0,0,0],atol=2e-5,rtol=0)
batch=torch.tensor([[[1.,2.,3.,4.],[5.,6.,7.,8.]]])
changed=batch.clone();changed[:,1,:]*=10
torch.testing.assert_close(env["rms_norm"](batch,torch.ones(4))[:,0],env["rms_norm"](changed,torch.ones(4))[:,0])
bad=lambda a:a*torch.rsqrt(a.square().mean((-1,-2),keepdim=True)+1e-5)
assert not torch.allclose(bad(batch)[:,0],bad(changed)[:,0])
print("PASS epsilon-placement large difference, zero-input derivative, FP16 finite-wrong result and causal axis control")

sigmoid=lambda t:1/(1+math.exp(-t))
expected=np.array([sigmoid(1)*2,-sigmoid(-1)*3])
np.testing.assert_allclose(env["ffn_output"].numpy()[0],expected,atol=1e-14)
np.testing.assert_allclose(expected,[1.4621,-.8068],atol=5e-5)
assert 3*4096*11008==135266304
assert 2*4096*16384==134217728
assert math.isclose(8/3*4096,10922.6666666667)

q=np.array([100.,0.]);keys=np.array([[100.,0.],[0.,100.]])
scores=keys@q/math.sqrt(2)
normalized=np.stack([rms(k) for k in keys])@rms(q)/math.sqrt(2)
probs=np.exp(normalized-normalized.max());probs/=probs.sum()
np.testing.assert_allclose(scores,[7071.0678118655,0],atol=1e-10)
np.testing.assert_allclose(normalized,[math.sqrt(2),0],atol=1e-14)
np.testing.assert_allclose(probs,[.8044,.1956],atol=5e-5)
assert np.max(np.abs(normalized))<=math.sqrt(2)+1e-14
assert (normalized*100).max()>math.sqrt(2)
print("PASS actual SwiGLU matrices, matched weight counts and separate QK-RMS example")

p=np.array([sigmoid(1),sigmoid(-1)])
base=math.log1p(math.exp(-1))
log_z=1000+base
np.testing.assert_allclose(env["log_z"].item(),log_z,atol=1e-13)
np.testing.assert_allclose(env["penalty"].item(),1e-5*log_z**2,atol=1e-13)
np.testing.assert_allclose(env["z"].grad.numpy(),2e-5*log_z*p,atol=1e-14)
np.testing.assert_allclose([1e-5*log_z**2,1e-5*(1+base)**2],[10.0063,.0000172],atol=5e-5)
logits=torch.tensor([1000.,999.],dtype=torch.float32)
assert torch.isnan(logits.exp()/logits.exp().sum()).all()
np.testing.assert_allclose(logits.softmax(-1).numpy(),p,atol=3e-8)
shifted=env["z"].detach()-env["log_z"].detach()
torch.testing.assert_close(shifted.softmax(-1),env["z"].detach().softmax(-1))
assert abs(torch.logsumexp(shifted,-1).item())<1e-12
capped=10*np.tanh(np.array([20.,10.])/10)
cap_probs=np.exp(capped-capped.max());cap_probs/=cap_probs.sum()
np.testing.assert_allclose(capped,[9.6403,7.6159],atol=5e-5)
np.testing.assert_allclose(cap_probs,[.8833,.1167],atol=5e-5)
assert env["correct"][0,1].item()==0
np.testing.assert_allclose(env["wrong"][0,1].item(),.1230,atol=5e-5)
print("PASS stable-softmax distinction, independent z-loss derivative/shift, soft-capping and forbidden-key leak")

branch=lambda a:a+ln(.1*a)
np.testing.assert_allclose(branch(x),x+z,atol=1e-14)
assert not np.allclose(branch(x),expected_pre)
print("PASS branch-output normalization differs from both pre- and post-residual normalization")

# Pin actual paper columns instead of relying only on literal chapter numbers.
paper=ROOT/"05 Источники/Originals/initialization-normalization-2026-10-21/rmsnorm.pdf"
assert hashlib.sha256(paper.read_bytes()).hexdigest()=="a38ab771022fc10525d2b6b23de40307b2ca7b2bf463f5219da6a0ba1c41225f"
page=subprocess.check_output(["pdftotext","-layout","-f","7","-l","7",str(paper),"-"],text=True)
for pattern in [r"LayerNorm\s+26\.6\s+27\.7\s+248±1\.31s",
                r"RMSNorm\s+26\.8\s+27\.7\s+231±0\.04s",
                r"pRMSNorm\s+26\.5\s+27\.8\s+225±1\.63s"]:
    assert re.search(pattern,page),pattern
assert math.isclose((248-231)/248*100,6.854838709677419)
print("PASS RMSNorm Table 4 actual rows and 6.9% time reduction, not whole-LLM speedup")
print("No released-weight training, GPU latency or universal stability claim.")
print("CHAPTER_SHA256",hashlib.sha256(CHAPTER.read_bytes()).hexdigest())
