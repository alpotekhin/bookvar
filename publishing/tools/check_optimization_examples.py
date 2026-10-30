"""Execute the current chapter; independently check each numerical mechanism."""
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO
import copy
import math
import re

import numpy as np
import torch
from topic_consolidation_baseline import before_topic_consolidation

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/01 Основы нейронных сетей/02 Оптимизация и стабильность обучения.md'
text = CHAPTER.read_text()
text = before_topic_consolidation(
    text, CHAPTER, stop_before_journal="2026-10-29-retention-additions.json")
assert not re.search(r'^\$$', text, re.M), 'display math needs $$ delimiters'
# The independent memory-optimizer addition has its own executable checker.
text = re.sub(r'<!-- BEGIN REVIEW: optimizer memory 2026-10-28 -->\n.*?'
              r'<!-- END REVIEW: optimizer memory 2026-10-28 -->\n\n', '', text, flags=re.S)
blocks = re.findall(r'```python\n(.*?)```', text, re.S)
assert len(blocks) == 4
scopes=[]
for i,code in enumerate(blocks,1):
    scope={}
    with redirect_stdout(StringIO()) as captured:
        exec(compile(code,f'{CHAPTER}:block{i}','exec'),scope)
    scopes.append(scope)
    print(f'block {i}:',captured.getvalue().strip() or 'PASS')
optim,schedule,precision,checkpoint=scopes
probe=torch.nn.Parameter(torch.zeros(10,dtype=torch.float32))
probe_opt=torch.optim.AdamW([probe]);assert not probe_opt.state
probe.grad=torch.ones_like(probe);probe_opt.step()
state=probe_opt.state[probe]
assert state['exp_avg'].numel()+state['exp_avg_sq'].numel()==2*probe.numel()
assert sum(state[k].numel()*state[k].element_size() for k in ('exp_avg','exp_avg_sq'))==8*probe.numel()
np.testing.assert_allclose(8_000_000/1024**2,7.62939453125)
np.testing.assert_allclose(optim['paths']['SGD'],[1,.9,.81,.729],atol=1e-14)
np.testing.assert_allclose(optim['paths']['Momentum'],[1,.9,.72,.486],atol=1e-14)
np.testing.assert_allclose(optim['paths']['Adam'],[1,.9,.8004122277,.7015862723],atol=1e-8)
np.testing.assert_allclose([optim['regularized']['Adam+L2'],optim['regularized']['AdamW']],[1.9,1.88],atol=1e-8)

# Independent recurrence, including every intermediate in the three-step table.
q=1.;m=v=0.
for t in range(1,4):
    g=q;m=.9*m+.1*g;v=.999*v+.001*g*g
    mh=m/(1-.9**t);vh=v/(1-.999**t)
    if t==1: np.testing.assert_allclose([m,v,mh,vh],[.1,.001,1,1],atol=1e-15)
    if t==2: np.testing.assert_allclose([m,v,mh,vh],[.1800000001,.0018090000018,.9473684216,.9049524771],atol=1e-8)
    q-=.1*mh/(math.sqrt(vh)+1e-8)
    np.testing.assert_allclose(q,optim['paths']['Adam'][t],atol=1e-14)

# Mini-batch mean/variance and the exact quadratic critical-batch calculation.
assert np.mean([4,-2])==1 and np.var([4,-2])==9
assert np.mean([4,1,1,-2])==1 and np.var([4,1,1,-2])==4.5
np.testing.assert_allclose(np.linalg.norm([-.03,-.04])/np.linalg.norm([3,4]),.01)
# A per-example function has no derivative through other batch elements.
batch=torch.arange(6,dtype=torch.float64).reshape(3,2).requires_grad_()
batch.square()[1].sum().backward()
torch.testing.assert_close(batch.grad[0],torch.zeros(2,dtype=batch.dtype))
torch.testing.assert_close(batch.grad[2],torch.zeros(2,dtype=batch.dtype))
assert torch.count_nonzero(batch.grad[1])==2
full_loss=lambda x:.5*x*x+x+5
assert full_loss(0)-full_loss(-1)==.5
for B,expected in [(1,.05),(9,.25),(90,5/11)]:
    eta=1/(1+9/B)
    expected_decrease=eta-.5*eta*eta*(1+9/B)
    np.testing.assert_allclose(expected_decrease,expected,atol=1e-15)

# Original D2L geometric examples: their two original figures do not share eta.
def trajectory(lr,beta,steps=20):
    point=np.array([-5.,-2.]);velocity=np.zeros(2);path=[point.copy()]
    for _ in range(steps):
        g=np.array([.2,4.])*point
        velocity=beta*velocity+g
        point=point-lr*velocity;path.append(point.copy())
    return np.array(path)
np.testing.assert_allclose(trajectory(.4,0)[1],[-4.6,1.2],atol=1e-14)
np.testing.assert_allclose(trajectory(.4,0)[-1],[-.943467,-.000073],atol=.5e-6)
np.testing.assert_allclose(trajectory(.6,0)[-1],[-.387814,-1673.365109],atol=.5e-6)
np.testing.assert_allclose(trajectory(.6,.5)[2],[-3.572,-1.52],atol=1e-14)
np.testing.assert_allclose(trajectory(.6,.5)[-1],[.007188,.002553],atol=.5e-6)
assert abs(.9**10-.3486784401)<1e-14
velocity=np.zeros(2)
for g,expected in zip(([1,4],[1,-4],[1,4]),([1,4],[1.5,-2],[1.75,3])):
    velocity=.5*velocity+g;np.testing.assert_array_equal(velocity,expected)

# Lookahead-parameter convention, not a claim about stored torch SGD variables.
theta=1.;shift=0.
for _ in range(2):
    ahead=theta+.9*shift;shift=.9*shift-.1*ahead;theta+=shift
np.testing.assert_allclose([ahead,shift,theta],[.81,-.171,.729],atol=1e-15)
np.testing.assert_allclose(.1/math.sqrt(17),.0242535625,atol=1e-10)
np.testing.assert_allclose(.1/math.sqrt(1.54),.0805822964,atol=1e-10)

# Adam scale invariance is an eps=0 statement, and large gradients do not imply smaller steps.
for factor in (1,10,100):
    g=np.array([10.,1.])*factor; m=.1*g;v=.001*g*g
    np.testing.assert_allclose(.1*(m/.1)/np.sqrt(v/.001),[.1,.1],atol=1e-14)
g=1e-12;eps=1e-8
pytorch_step=.1*g/(abs(g)+eps)
course_step=.1*math.sqrt(1-.999)/(1-.9)*(.1*g)/(math.sqrt(.001*g*g)+eps)
assert not math.isclose(pytorch_step,course_step,rel_tol=.1)
# Reparameterizing eps makes the rearrangement genuinely equivalent.
same=.1*math.sqrt(1-.999)/(1-.9)*(.1*g)/(math.sqrt(.001*g*g)+eps*math.sqrt(1-.999))
np.testing.assert_allclose(same,pytorch_step,atol=1e-18)

cosine=schedule['cosine_lr']
for t,value in [(0,0),(2,.1),(6,.055),(8,.0231801948),(10,.01),(11,.01)]:
    np.testing.assert_allclose(cosine(t),value,atol=1e-10)
np.testing.assert_allclose(cosine(6,total=18),.0868198052,atol=1e-10)
assert all(schedule['schedule'][i]>=schedule['schedule'][i+1] for i in range(1,9))

# Global norm clips a concatenated vector, not each tensor independently.
p=torch.nn.Parameter(torch.zeros(1,dtype=torch.float64));q=torch.nn.Parameter(torch.zeros(1,dtype=torch.float64))
p.grad=torch.tensor([6.],dtype=p.dtype);q.grad=torch.tensor([8.],dtype=q.dtype)
old=torch.nn.utils.clip_grad_norm_([p,q],5,error_if_nonfinite=True)
assert old.item()==10
np.testing.assert_allclose([p.grad.item(),q.grad.item()],[3,4],atol=5e-7)
adam=torch.optim.Adam([p,q],lr=.1);adam.step()
np.testing.assert_allclose([p.item(),q.item()],[-.1,-.1],atol=1e-9)
np.testing.assert_allclose(precision['restored'].item(),1.0011717677116394e-8,atol=1e-15)
assert torch.finfo(torch.float8_e4m3fn).max==448
assert torch.finfo(torch.float8_e5m2).max==57344

# Weights-only resumption does not match; demonstrate without changing the published state.
clone=torch.nn.Linear(1,1,dtype=torch.float64);clone.load_state_dict(checkpoint['checkpoint']['model'])
fresh=torch.optim.AdamW(clone.parameters(),lr=.1,weight_decay=.01)
torch.set_rng_state(checkpoint['checkpoint']['rng'])
for t in range(6,11): checkpoint['train_step'](clone,fresh,t)
assert any(not torch.equal(clone.state_dict()[k],v) for k,v in checkpoint['reference'].items())
for key,value in checkpoint['reference'].items():
    torch.testing.assert_close(checkpoint['resumed'].state_dict()[key],value,rtol=0,atol=0)
assert checkpoint['resumed_opt'].state_dict()['state'].keys()==checkpoint['opt'].state_dict()['state'].keys()

# Fixed-basis example illustrates coordinate dependence; it is not a full SOAP run.
G=np.array([[3.,1.],[1.,3.]])
Q=np.array([[1.,1.],[1.,-1.]])/math.sqrt(2)
np.testing.assert_allclose(Q.T@G@Q,np.diag([4,2]),atol=1e-14)
np.testing.assert_array_equal(np.sign(G),np.ones((2,2)))
np.testing.assert_allclose(Q@np.eye(2)@Q.T,np.eye(2),atol=1e-14)
U,s,Vh=np.linalg.svd(G)
np.testing.assert_allclose(s,[4,2],atol=1e-14)
np.testing.assert_allclose(U@Vh,np.eye(2),atol=1e-14)
np.testing.assert_allclose(np.linalg.svd(G/4,compute_uv=False),[1,.5],atol=1e-14)
a,b,c=3.4445,-4.7750,2.0315
X=G/np.linalg.norm(G); A=X@X.T
polynomial=a*X+b*A@X+c*A@A@X
np.testing.assert_allclose(np.linalg.svd(polynomial,compute_uv=False),
    np.sort(a*(s/np.linalg.norm(G))+b*(s/np.linalg.norm(G))**3+c*(s/np.linalg.norm(G))**5)[::-1],atol=1e-14)
assert a*0+b*0**3+c*0**5==0
assert not np.allclose(polynomial,np.eye(2))
print('PASS: four published blocks; original D2L trajectories; gradient noise/quadratic steps; '
      'momentum/Nesterov; Adagrad/RMSProp/Adam/AdamW; epsilon convention; schedules; '
      'global clipping; FP16/BF16/scaling/FP8 bounds; exact checkpoint continuation and weights-only '
      'counterexample; fixed-basis adaptation, polar vs spectral, finite Newton-Schulz. torch',torch.__version__)
