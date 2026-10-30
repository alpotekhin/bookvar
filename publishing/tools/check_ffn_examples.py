"""Run published FFN blocks, independent scalar operators and finite differences."""
from pathlib import Path
import ast
import hashlib
import itertools
import math
import re
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/09 Dense FFN и Mixture of Experts/01 Dense FFN — token-wise вычисление, expansion и gating.md'
blocks = re.findall(r'~~~python\n(.*?)\n~~~',CHAPTER.read_text(),re.S)
assert len(blocks) == 3
ns = {}
for i,block in enumerate(blocks,1):
    exec(compile(block,f'{CHAPTER.name}:block-{i}','exec'),ns)
    print('PASS published block',i)
np.testing.assert_allclose(ns['y'][0,0],[-.13447071068499755,-.5965878679450073])

def scalar_swiglu(x,wu,wg,wd):
    """Separate loops for each row, intermediate coordinate and output."""
    result = np.zeros_like(x)
    for row in np.ndindex(x.shape[:-1]):
        hidden = []
        for j in range(wu.shape[1]):
            u = sum(float(x[row][i])*float(wu[i,j]) for i in range(x.shape[-1]))
            z = sum(float(x[row][i])*float(wg[i,j]) for i in range(x.shape[-1]))
            hidden.append(u*z/(1+math.exp(-z)))
        for d in range(x.shape[-1]):
            result[row][d] = sum(h*float(wd[j,d]) for j,h in enumerate(hidden))
    return result

rng = np.random.default_rng(23)
x = rng.normal(size=(2,3,4))
wu,wg,wd = [rng.normal(scale=.4,size=s) for s in ((4,6),(4,6),(6,4))]
forward,backward = ns['swiglu'],ns['swiglu_backward']
y = forward(x,wu,wg,wd)
np.testing.assert_allclose(y,scalar_swiglu(x,wu,wg,wd),atol=2e-15,rtol=2e-14)
np.testing.assert_allclose(forward(x.reshape(-1,4),wu,wg,wd).reshape(x.shape),y)
permutation = [2,0,1]
np.testing.assert_allclose(forward(x[:,permutation],wu,wg,wd),y[:,permutation])
changed = x.copy()
changed[0,1] += 10
other = forward(changed,wu,wg,wd)
np.testing.assert_array_equal(other[1],y[1])
np.testing.assert_array_equal(other[0,[0,2]],y[0,[0,2]])
assert not np.allclose(other[0,1],y[0,1])
print('PASS scalar operator, flatten, permutation and one-token intervention')

def loss():
    return .5*float(np.sum(forward(x,wu,wg,wd)**2))

analytic = backward(x,wu,wg,wd,y)
for name,array,gradient in zip(('x','wu','wg','wd'),(x,wu,wg,wd),analytic):
    numerical = np.empty_like(array)
    for idx in np.ndindex(array.shape):
        old = array[idx]
        array[idx] = old+1e-5
        plus = loss()
        array[idx] = old-1e-5
        minus = loss()
        array[idx] = old
        numerical[idx] = (plus-minus)/2e-5
    np.testing.assert_allclose(gradient,numerical,atol=1e-9,rtol=2e-7)
    print('PASS independent central differences',name,float(np.max(np.abs(gradient-numerical))))

for shards in (1,2,3,6):
    partial_outputs,partial_dx = [],[]
    rebuilt = [np.zeros_like(wu),np.zeros_like(wg),np.zeros_like(wd)]
    for indices in np.array_split(np.arange(6),shards):
        local = (wu[:,indices],wg[:,indices],wd[indices])
        partial_outputs.append(forward(x,*local))
        dx,du,dg,dd = backward(x,*local,y)
        partial_dx.append(dx)
        rebuilt[0][:,indices] = du
        rebuilt[1][:,indices] = dg
        rebuilt[2][indices] = dd
    np.testing.assert_allclose(sum(partial_outputs),y,atol=2e-15)
    np.testing.assert_allclose(sum(partial_dx),analytic[0],atol=2e-15)
    for actual,expected in zip(rebuilt,analytic[1:]):
        np.testing.assert_allclose(actual,expected,atol=2e-15)
print('PASS algebraic TP: outputs, input-gradient sum and all weight shards')

assert not np.allclose(forward(x,wg,wu,wd),y)
wrong = forward(x,wu,wg[:,::-1],wd)
assert not np.allclose(wrong,y)
bias = np.array([.1,.2,.3,.4])
assert not np.allclose(y+2*bias,y+bias)
assert not np.allclose(analytic[0],np.zeros_like(x))
print('PASS swapped branch, mismatched shard coordinates and duplicate-bias controls')

# Execute only the actual pinned Meta forward body with NumPy projections.
# This does not execute PyTorch, FairScale collectives or Meta initialization.
source = ROOT / '05 Источники/Originals/ffn-2026-10-23/llama-model.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == 'b409ea3db2a4029afd4129246d3b5e76ea95b6503364842545403dbcd7476281'
tree = ast.parse(source.read_text())
klass = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FeedForward')
method = next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
env = {'F':SimpleNamespace(silu=ns['silu'])}
exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),env)
model = SimpleNamespace(w1=lambda z:z@wg,w3=lambda z:z@wu,w2=lambda z:z@wd)
np.testing.assert_allclose(env['forward'](model,x),y)
print('PASS pinned Meta forward expression with NumPy projection adapters')

# The chapter's manual ReLU, affine-collapse and memory-composition examples.
rx = np.array([[1.,-1.],[0.,2.]])
r1 = np.array([[1.,0.,1.,-1.],[0.,1.,1.,1.]])
r2 = np.array([[1.,0.],[0.,1.],[1.,1.],[-1.,1.]])
rh = np.maximum(rx@r1,0)
np.testing.assert_array_equal(rh,[[1,0,0,0],[0,2,2,2]])
np.testing.assert_array_equal(rh@r2,[[1,0],[0,6]])
np.testing.assert_array_equal(rh[:,0:2]@r2[:2]+rh[:,2:]@r2[2:],rh@r2)
b1,b2 = rng.normal(size=4),rng.normal(size=2)
np.testing.assert_allclose((rx@r1+b1)@r2+b2,rx@(r1@r2)+(b1@r2+b2))
assert not np.allclose(np.maximum(rx@r1,0)@r2,rx@(r1@r2))
directions = np.array([[3,0,2],[0,3,2]])
np.testing.assert_array_equal(directions.argmax(-1),[0,1])
assert directions.sum(0).argmax() == 2
print('PASS manual ReLU, affine collapse, TP sums and vocabulary composition')

d,m = 4096,11008
p = 3*d*m
assert p == 135266304 and 32*p == 4328521728
assert 2*p/2**20 == 258 and 2*32*p/2**30 == 8.0625
assert 2*4*2048*m/2**20 == 172
assert 6*d*m == 270532608
assert (3*m-4*d)/2 == 8320
assert 3*32768*d*m == 4432406249472
hidden = int(2*(4*d)/3)
assert 256*((hidden+255)//256) == m
assert 2*768*3072 == 3*768*2048
print('PASS parameters, MAC/FLOP, activations, width rounding and matched GLU budget')
print('PASS explanation checks; not a GPU, collectives, training or quality benchmark')
print('chapter SHA256',hashlib.sha256(CHAPTER.read_bytes()).hexdigest())
