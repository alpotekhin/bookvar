"""Execute the published sequence-model examples; CPU algebra, not kernel benchmarks."""
from pathlib import Path
import hashlib
import itertools
import math
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FILE = ROOT / '00 Учебник/09 Dense FFN и Mixture of Experts/03 Mamba, RWKV, RetNet и гибридные архитектуры.md'
body = FILE.read_text()
blocks = re.findall(r'~~~python\n(.*?)\n~~~', body, re.S)
assert len(blocks) == 6
ns = {}
for block in blocks:
    exec(compile(block, str(FILE), 'exec'), ns)
print('PASS six published blocks', hashlib.sha256(FILE.read_bytes()).hexdigest())

# Independent scalar LTI unrolling, including nonzero initial state.
rng = np.random.default_rng(73)
T, D, N = 7, 3, 2
x = rng.normal(size=(T,D)); dt=rng.uniform(.01,.9,(T,D))
A=-rng.uniform(.2,2,(D,N)); B=rng.normal(size=(T,N)); C=rng.normal(size=(T,N))
h0=rng.normal(size=(D,N)); direct=rng.normal(size=D)
ys, states=ns['selective_ssm'](x,dt,A,B,C,direct,h0)
scalar=np.zeros_like(states)
prev=h0.copy()
for t in range(T):
    for d in range(D):
        for n in range(N):
            scalar[t,d,n]=math.exp(dt[t,d]*A[d,n])*prev[d,n]+dt[t,d]*B[t,n]*x[t,d]
    prev=scalar[t]
np.testing.assert_allclose(states,scalar,atol=1e-14)
expected=np.array([[sum(scalar[t,d,n]*C[t,n] for n in range(N))+direct[d]*x[t,d]
                    for d in range(D)] for t in range(T)])
np.testing.assert_allclose(ys,expected,atol=1e-14)

def partitions(n):
    for mask in range(1<<(n-1)):
        ends=[i+1 for i in range(n-1) if mask&(1<<i)]+[n]
        yield list(zip([0]+ends[:-1],ends))

for cuts in partitions(T):
    start_state=h0.copy(); out=[]
    for lo,hi in cuts:
        y,h=ns['selective_ssm'](x[lo:hi],dt[lo:hi],A,B[lo:hi],C[lo:hi],direct,start_state)
        out.extend(y); start_state=h[-1]
    np.testing.assert_allclose(out,ys,atol=1e-14)
    np.testing.assert_allclose(start_state,states[-1],atol=1e-14)

# Every binary parenthesization composes ordered affine transitions.
def all_compositions(pairs):
    if len(pairs)==1: return [pairs[0]]
    result=[]
    for cut in range(1,len(pairs)):
        for left in all_compositions(pairs[:cut]):
            for right in all_compositions(pairs[cut:]):
                a,b=left; c,d=right
                result.append((c*a,c*b+d))
    return result
a=np.exp(dt[:,:,None]*A)
b=dt[:,:,None]*B[:,None,:]*x[:,:,None]
for aa,bb in all_compositions(list(zip(a,b))):
    np.testing.assert_allclose(aa*h0+bb,states[-1],atol=2e-14)
assert len(all_compositions(list(zip(a,b))))==132
print('PASS independent selective states, 64 stream partitions, 132 ordered affine trees')

# SSD: scalar reference, dense form, all chunk lengths and stream partitions.
q=rng.normal(size=(T,2)); k=rng.normal(size=(T,2)); v=rng.normal(size=(T,3))
g=rng.uniform(.1,.99,T); initial=rng.normal(size=(2,3))
reference=np.zeros((T,3))
for t in range(T):
    for p in range(3):
        reference[t,p]=math.prod(g[:t+1])*sum(q[t,n]*initial[n,p] for n in range(2))
        for j in range(t+1):
            reference[t,p]+=math.prod(g[j+1:t+1])*sum(q[t,n]*k[j,n] for n in range(2))*v[j,p]
yr,sr=ns['ssd_recurrent'](q,k,v,g,initial)
np.testing.assert_allclose(yr,reference,atol=2e-14)
for chunk in range(1,T+2):
    yc,sc=ns['ssd_chunked'](q,k,v,g,chunk,initial)
    np.testing.assert_allclose(yc,reference,atol=2e-14)
    np.testing.assert_allclose(sc,sr,atol=2e-14)
for cuts in partitions(T):
    state=initial.copy(); out=[]
    for lo,hi in cuts:
        y,state=ns['ssd_chunked'](q[lo:hi],k[lo:hi],v[lo:hi],g[lo:hi],hi-lo,state)
        out.extend(y)
    np.testing.assert_allclose(out,reference,atol=2e-14)
    np.testing.assert_allclose(state,sr,atol=2e-14)
g_zero=g.copy(); g_zero[3]=0
np.testing.assert_allclose(ns['ssd_chunked'](q,k,v,g_zero,3,initial)[0],
                           ns['ssd_recurrent'](q,k,v,g_zero,initial)[0],atol=2e-14)
assert not np.allclose(((q@k.T)*np.tril(np.ones((T,T))))@v,reference)
print('PASS SSD scalar/dense/blocked, 8 chunk lengths, 64 partitions, zero decay and wrong-mask control')

positive_q=rng.uniform(.1,2,(T,2)); positive_k=rng.uniform(.1,2,(T,2))
yn,_,_=ns['normalized_linear'](positive_q,positive_k,v)
for t in range(T):
    weights=np.array([sum(positive_q[t,n]*positive_k[j,n] for n in range(2)) for j in range(t+1)])
    np.testing.assert_allclose(yn[t],(weights[:,None]*v[:t+1]).sum(0)/weights.sum())
try:
    ns['normalized_linear'](np.zeros((1,2)),np.ones((1,2)),np.ones((1,1)))
    raise AssertionError('zero denominator was accepted')
except ValueError:
    pass

unit_k=k/np.linalg.norm(k,axis=1,keepdims=True)
beta=rng.uniform(0,1,T)
yd,sd=ns['gated_delta'](q,unit_k,v,g,beta,initial)
state=initial.copy(); manual=[]
for t in range(T):
    transition=g[t]*(np.eye(2)-beta[t]*np.outer(unit_k[t],unit_k[t]))
    state=transition@state+beta[t]*np.outer(unit_k[t],v[t])
    manual.append(q[t]@state)
np.testing.assert_allclose(manual,yd,atol=1e-14)
np.testing.assert_allclose(state,sd,atol=1e-14)
# Nonorthogonal replacement changes prior keys: not an unbounded dictionary.
kk=np.array([1.,1.])/math.sqrt(2)
_, changed=ns['gated_delta'](kk[None],kk[None],np.array([[40.]]),np.ones(1),np.ones(1),
                            np.array([[30.],[20.]]))
np.testing.assert_allclose(kk@changed,[40.])
assert not np.isclose(changed[0,0],30)
print('PASS normalized attention direct sums, zero denominator, gated transition and key interference')

# TTT gradients are evaluated at the block-start state; outputs remain causal.
for batch in (1,2,3,8):
    W0=rng.normal(size=(3,2))
    yt,W=ns['ttt_linear'](k,v,q,eta=.2,batch_size=batch,W0=W0)
    current=W0.copy(); out=[]
    for start in range(0,T,batch):
        base=current.copy()
        for t in range(start,min(start+batch,T)):
            for p,n in itertools.product(range(3),range(2)):
                prediction=sum(base[p,j]*k[t,j] for j in range(2))
                current[p,n]-=.2*(prediction-v[t,p])*k[t,n]
            out.append([sum(current[p,n]*q[t,n] for n in range(2)) for p in range(3)])
    np.testing.assert_allclose(yt,out,atol=1e-13)
    np.testing.assert_allclose(W,current,atol=1e-13)
    for cut in range(1,T):
        perturbed=v.copy(); perturbed[cut:]+=100
        yc,_=ns['ttt_linear'](k,perturbed,q,eta=.2,batch_size=batch,W0=W0)
        np.testing.assert_allclose(yc[:cut],yt[:cut],atol=1e-13)
online=ns['ttt_linear'](k,v,q,eta=.2,batch_size=1,W0=initial.T)[0]
delta=ns['gated_delta'](q,k,v,np.ones(T),np.full(T,.2),initial)[0]
np.testing.assert_allclose(online,delta,atol=1e-13)
print('PASS TTT independent block-start gradients, future perturbations and online delta equivalence')

# Older worked examples and memory/rollback arithmetic.
vals=[2.,4.,8.]; aa=bb=0.; outputs=[]
for val in vals:
    outputs.append((aa+val)/(bb+1))
    aa=.5*aa+val; bb=.5*bb+1
np.testing.assert_allclose(outputs,[2.,3.,5.2])
retention=[]; state=0
for val in vals:
    state=.5*state+val; retention.append(state)
np.testing.assert_allclose(retention,[2.,5.,10.5])
previous=np.array([[2.,4.],[6.,8.]])
direction=np.array([1.,0.])
transition=np.diag([.5,.8])-np.outer(direction,np.array([.2,.3])*direction)
rwkv7=previous@transition+np.outer([10.,20.],[1.,0.])
np.testing.assert_allclose(rwkv7,[[10.6,3.2],[21.8,6.4]])
angle=math.pi/2
rotation=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
state=np.array([1.,0.]); rotated=[]
for _ in range(4):
    state=rotation@state; rotated.append(state.copy())
np.testing.assert_allclose(rotated,[[0,1],[-1,0],[0,-1],[1,0]],atol=1e-14)
for lam in (0,.5,1):
    alpha=math.exp(-.2)
    new=alpha*2+(1-lam)*.2*alpha*3+lam*.2*4
    if lam==1: np.testing.assert_allclose(new,alpha*2+.2*4)
assert 2*131072*32*8*128*2==16*2**30
assert 24*(4096*16*4+4096*4*2)==7077888==6.75*2**20
state=2.5; history=[]
for val in [1,2,3,4]:
    state=.5*state+val; history.append(state)
np.testing.assert_allclose(history,[2.25,3.125,4.5625,6.28125])
assert history[1]!=history[-1]
print('PASS RWKV-4/RetNet, rotation, endpoint weighting, memory and accepted-prefix rollback')

# RetNet: independently verify rotated keys, all block sizes, initial state and tail.
q0_ret = rng.normal(size=(7, 2))
k0_ret = rng.normal(size=(7, 2))
v0_ret = rng.normal(size=(7, 3))
phase_ret = np.exp(1j * np.arange(1, 8) * .7)
qc_ret = (q0_ret[:, 0] + 1j*q0_ret[:, 1]) * phase_ret
kc_ret = (k0_ret[:, 0] + 1j*k0_ret[:, 1]) * phase_ret
Q_ret = np.column_stack((qc_ret.real, qc_ret.imag))
K_ret = np.column_stack((kc_ret.real, kc_ret.imag))
S0_ret = rng.normal(size=(2, 3))
gamma_ret = .57
manual_ret, state_ret = [], S0_ret.copy()
for t in range(7):
    state_ret = gamma_ret*state_ret + np.outer(K_ret[t], v0_ret[t])
    manual_ret.append([
        sum(Q_ret[t, j]*state_ret[j, d] for j in range(2))
        for d in range(3)
    ])
for block_size in range(1, 9):
    forms_ret = ns['retention_three_forms'](
        Q_ret, K_ret, v0_ret, gamma_ret, block_size, S0_ret
    )
    for output in forms_ret[:3]:
        np.testing.assert_allclose(output, manual_ret, atol=2e-14)
    np.testing.assert_allclose(forms_ret[3], state_ret, atol=2e-14)
    np.testing.assert_allclose(forms_ret[4], state_ret, atol=2e-14)

complex_state_ret = S0_ret[0] - 1j*S0_ret[1]
complex_out_ret, wrong_out_ret = [], []
wrong_state_ret = complex_state_ret.copy()
for t in range(7):
    complex_state_ret = gamma_ret*complex_state_ret + kc_ret[t].conjugate()*v0_ret[t]
    complex_out_ret.append((qc_ret[t]*complex_state_ret).real)
    wrong_state_ret = gamma_ret*wrong_state_ret + kc_ret[t]*v0_ret[t]
    wrong_out_ret.append((qc_ret[t]*wrong_state_ret).real)
np.testing.assert_allclose(complex_out_ret, manual_ret, atol=2e-14)
assert not np.allclose(wrong_out_ret, manual_ret)
# Completing a block offline must not expose its future values to earlier rows.
for cut in range(1, 7):
    changed_v_ret = v0_ret.copy()
    changed_v_ret[cut:] += 100
    changed_ret = ns['retention_three_forms'](
        Q_ret, K_ret, changed_v_ret, gamma_ret, 3, S0_ret
    )
    for output in changed_ret[:3]:
        np.testing.assert_allclose(output[:cut], np.array(manual_ret)[:cut], atol=2e-14)
bad_tail_ret = gamma_ret**7*S0_ret + K_ret.T @ v0_ret
assert not np.allclose(bad_tail_ret, state_ret)
print('PASS RetNet complex/real equivalence, 8 block sizes, nonzero state, tail and future controls')
