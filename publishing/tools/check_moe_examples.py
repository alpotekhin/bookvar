"""Execute the chapter's actual code and independently check routing arithmetic.

CPU algebra only. No GPU, all-to-all kernel, checkpoint training or benchmark.
"""
from pathlib import Path
import hashlib
import itertools
import math
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FILE = ROOT / '00 Учебник/09 Dense FFN и Mixture of Experts/02 Mixture of Experts — routing, capacity и serving.md'
body = FILE.read_text()
blocks = re.findall(r'~~~python\n(.*?)\n~~~', body, re.S)
assert len(blocks) == 4
ns = {}
for block in blocks:
    exec(compile(block, str(FILE), 'exec'), ns)
print('PASS four published blocks', hashlib.sha256(FILE.read_bytes()).hexdigest())

def scalar_softmax(row):
    vals = [math.exp(float(x) - max(row)) for x in row]
    return [x / sum(vals) for x in vals]

def scalar_moe(x, ids, weights, up, down, keep=None):
    out = np.zeros_like(x)
    for t in range(len(x)):
        for slot, e in enumerate(ids[t]):
            if keep is not None and not keep[t, slot]:
                continue
            hidden = [max(0., sum(x[t, a] * up[e, a, j] for a in range(x.shape[1])))
                      for j in range(up.shape[2])]
            for b in range(x.shape[1]):
                out[t, b] += weights[t, slot] * sum(hidden[j] * down[e, j, b] for j in range(len(hidden)))
    return out

x, ids, weights, up, down = [ns[k] for k in ('x','ids','weights','up','down')]
np.testing.assert_array_equal(ids, [[0,1],[0,2],[0,3],[0,1]])
expected = [[1.25,1.5],[.25,1.],[2.,8/15],[0.,4/7]]
np.testing.assert_allclose(ns['y'], expected, atol=1e-14)
np.testing.assert_allclose(ns['y'], scalar_moe(x,ids,weights,up,down), atol=1e-14)
np.testing.assert_array_equal(np.bincount(ids.ravel(),minlength=4), [4,2,1,1])
for p in itertools.permutations(range(4)):
    p = list(p)
    got = ns['dispatch_combine'](x[p],ids[p],weights[p],up,down)
    np.testing.assert_allclose(got,ns['y'][p],atol=1e-14)
np.testing.assert_allclose(ns['y_limited'], scalar_moe(x,ids,weights,up,down,ns['keep']), atol=1e-14)
np.testing.assert_allclose(ns['y_limited'][2], [14/15,0], atol=1e-14)
np.testing.assert_allclose(ns['y_limited'][3] + x[3], [-1,1], atol=1e-14)
for capacity in range(5):
    keep=ns['capacity_mask'](ids,4,capacity)
    np.testing.assert_array_equal(np.bincount(ids[keep],minlength=4),np.minimum([4,2,1,1],capacity))
    np.testing.assert_allclose(ns['dispatch_combine'](x,ids,weights,up,down,keep),
                               scalar_moe(x,ids,weights,up,down,keep),atol=1e-14)
assert not np.allclose(ns['y_limited'],ns['y'])
assert not np.allclose(ns['dispatch_combine'](x,ids,weights[:,::-1],up,down),ns['y'])
print('PASS scalar dispatch/combine, 24 permutations, capacity 0..4, wrong-weight control')

def finite_gradient(fn, a, step=1e-6):
    out=np.zeros_like(a)
    for coord in np.ndindex(a.shape):
        plus=a.copy(); minus=a.copy()
        plus[coord]+=step; minus[coord]-=step
        out[coord]=(fn(plus)-fn(minus))/(2*step)
    return out

logits=np.log([.5,.3,.15,.05])
values=np.array([[2.,0.],[0.,4.],[-1.,1.],[3.,2.]])
incoming=np.array([1.,2.])
def task(logits,k,renorm):
    p=np.array(scalar_softmax(logits))
    selected=sorted(range(4),key=lambda i:-p[i])[:k]
    w=p[selected]
    if renorm: w=w/w.sum()
    return sum(float(incoming @ values[e])*a for e,a in zip(selected,w))
g1=finite_gradient(lambda z:task(z,1,False),logits)
g2=finite_gradient(lambda z:task(z,2,True),logits)
np.testing.assert_allclose(g1,[.5,-.3,-.15,-.05],atol=2e-9)
np.testing.assert_allclose(g2,[-1.40625,1.40625,0,0],atol=2e-9)
np.testing.assert_allclose(finite_gradient(lambda z:task(z,1,True),logits),0,atol=1e-12)
print('PASS top-1/top-2 fixed-route task gradients and normalized-top-1 zero gradient')

pair=ns['pair_p']
lb,lz,grad=ns['router_losses'](np.log(pair)+10)
np.testing.assert_allclose([lb,lz],[.011,.1],atol=1e-14)
def independent_aux(z,alpha=.01,beta=.001):
    p=np.array([scalar_softmax(row) for row in z])
    count=[sum(int(max(range(len(row)),key=lambda i:row[i])==e) for row in p) for e in range(z.shape[1])]
    f=[c/len(z) for c in count]
    balance=alpha*z.shape[1]*sum(f[e]*sum(row[e] for row in p)/len(z) for e in range(z.shape[1]))
    sums=[max(row)+math.log(sum(math.exp(v-max(row)) for v in row)) for row in z]
    return balance+beta*sum(v*v for v in sums)/len(z)
for shift in (-1000.,-10.,0.,10.,1000.):
    z=np.log(pair)+shift
    b,q,g=ns['router_losses'](z)
    np.testing.assert_allclose(b,.011,atol=1e-13)
    np.testing.assert_allclose(q,.001*shift*shift,atol=1e-10)
    np.testing.assert_allclose(b+q,independent_aux(z),atol=1e-9)
    if abs(shift)<=10:
        numeric=finite_gradient(independent_aux,z)
        error=float(np.max(np.abs(numeric-g)))
        assert error<2e-9,error
        print('PASS auxiliary gradient shift',shift,'max error',error)
counter=np.array([[.999,.001]]*49+[[.499,.501]]*51)
np.testing.assert_allclose(ns['router_losses'](np.log(counter),alpha=1.,beta=0.)[0],.99024,atol=1e-14)
for sharp in (np.array([[.999999,.000001]]),np.array([[.1,.2,.3,.4]])):
    assert abs(ns['router_losses'](np.log(sharp))[1])<1e-28
print('PASS consistent sub-1 balance example, stable LSE +/-1000 and sharp zero-z case')

scores=np.array([[.60,.55,.54,.10]])
bias=np.array([0.,0.,.10,0.])
bi,bw=ns['select_experts'](scores,2,scores+bias)
np.testing.assert_array_equal(bi,[[2,0]])
np.testing.assert_allclose(bw,[[.54/1.14,.60/1.14]],atol=1e-14)
wrong=(scores+bias)[0,bi[0]]; wrong/=wrong.sum()
assert not np.allclose(wrong,bw[0])
np.testing.assert_array_equal(ns['node_ids'],[[0,1]])
np.testing.assert_allclose(ns['node_weights'],[[.8/1.55,.75/1.55]],atol=1e-14)

def scalar_nodes(scores,k,groups,allowed,bias):
    all_ids=[]; all_w=[]; width=scores.shape[1]//groups
    for row in scores:
        ranking=row+bias
        sums=[]
        for group in range(groups):
            e=list(range(group*width,(group+1)*width))
            sums.append(sum(sorted((ranking[i] for i in e),reverse=True)[:k//allowed]))
        chosen=sorted(range(groups),key=lambda g:(-sums[g],g))[:allowed]
        candidates=[e for e in range(len(row)) if e//width in chosen]
        selected=sorted(candidates,key=lambda e:(-ranking[e],e))[:k]
        all_ids.append(selected)
        w=row[selected]; all_w.append(w/w.sum())
    return np.array(all_ids),np.array(all_w)
rng=np.random.default_rng(13)
for groups,allowed,k in ((2,1,2),(4,2,4),(8,4,8),(4,4,4)):
    s=rng.uniform(.01,.99,(7,groups*4)); b=rng.uniform(-.1,.1,groups*4)
    actual=ns['node_limited_route'](s,k,groups,allowed,b)
    reference=scalar_nodes(s,k,groups,allowed,b)
    np.testing.assert_array_equal(actual[0],reference[0])
    np.testing.assert_allclose(actual[1],reference[1],atol=1e-14)
print('PASS biased original-score weights and independently enumerated node-limited routes')

owners=np.array([0,0,1,1]); origins=np.array([0,0,1,1])
remote=int((owners[ids]!=origins[:,None]).sum())
assert remote==4 and 2*remote*2*2==32
assert 3*1024*2048==6291456 and 8*6291456==50331648
assert 8*6291456*2/2**20==96 and 2*6291456*2/2**20==24
assert math.comb(8,2)==28 and math.comb(16,4)==1820
assert math.ceil(1.25*4096*2/8)==1280
assert 2*4096*2*4096*2/2**20==128
assert (64*8/256,4096*8/256)==(2,128)
np.testing.assert_allclose(math.log(math.exp(10)+math.exp(9)),10.313261687518223)
print('PASS parameter/memory/capacity/EP/serving arithmetic; no physical network claim')
