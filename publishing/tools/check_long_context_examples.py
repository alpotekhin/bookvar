"""Execute five published NumPy blocks; independently check masks, cache and pooling."""
from pathlib import Path
import hashlib
import itertools
import math
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/08 Эффективный Attention и длинный контекст/03 Длинный контекст — расширение, разреженность и оценивание.md'
blocks = re.findall(r'~~~python\n(.*?)\n~~~', CHAPTER.read_text(), re.S)
assert len(blocks) == 5
ns = {}
for i, block in enumerate(blocks, 1):
    exec(compile(block, f'{CHAPTER.name}:block-{i}', 'exec'), ns)
    if i == 1:
        assert math.isclose(ns['two_layers'](ns['global_pattern'])[3, 0], 1/24)
        assert ns['two_layers'](ns['causal_pattern'])[3, 0] == 0
    elif i == 2:
        np.testing.assert_allclose(ns['whole'][:, 0], [1,1.5,2,2.5,3.5,4.5,5.5,6.5,7.5,8.5])
        np.testing.assert_array_equal(ns['cache'][0], [6,7,8,9])
    elif i == 3:
        a = 1 / (1 + math.exp(1/math.sqrt(2)))
        b = 1 / (1 + math.exp((2-2*a)/math.sqrt(2)))
        np.testing.assert_allclose(ns['h2_shared'][-1], [a+b, 3-a-b])
        np.testing.assert_allclose(ns['h2_separate'][-1], [.94529248,3.05470752], atol=5e-9)
    elif i == 4:
        np.testing.assert_array_equal(ns['index_scores'], [1,1,.5,3,2.5,0])
        np.testing.assert_array_equal(ns['selected'], [3,4])
        assert math.isclose(ns['sparse'].item(), 8)
        assert math.isclose(ns['full'].item(), 212/16)
    else:
        np.testing.assert_allclose(ns['csa_pool'], [[2.,20.],[3.75,45.]])
        np.testing.assert_allclose(ns['hca_pool'], [[4.,40.]])
        np.testing.assert_allclose(ns['csa_out'], [[19.75/3,205/3]])
        np.testing.assert_allclose(ns['hca_out'], [[20/3,200/3]])
    print('PASS published block', i)

def reference(q, k, v, window):
    """Independent scalar dot products, causal key ranges and scalar softmax."""
    result = []
    for t, query in enumerate(q):
        keys = range(max(0,t-window+1), t+1)
        scores = [sum(float(a)*float(b) for a,b in zip(query,k[s]))/math.sqrt(len(query)) for s in keys]
        weights = [math.exp(score-max(scores)) for score in scores]
        result.append([sum(weight*float(v[s,d]) for weight,s in zip(weights,keys))/sum(weights)
                       for d in range(v.shape[-1])])
    return np.array(result)

rng = np.random.default_rng(23)
q, k, v = [rng.normal(size=(8, d)) for d in (3,3,2)]
for window in (1,2,4,12):
    expected = reference(q,k,v,window)
    for cuts in itertools.product((False,True), repeat=7):
        boundaries = [0]+[i+1 for i,cut in enumerate(cuts) if cut]+[8]
        cache, outputs = None, []
        for start,end in zip(boundaries,boundaries[1:]):
            out,cache = ns['window_attention'](q[start:end],k[start:end],v[start:end],window,cache)
            outputs.append(out)
            np.testing.assert_array_equal(cache[0], np.arange(max(0,end-window),end))
            np.testing.assert_array_equal(cache[1], k[max(0,end-window):end])
            np.testing.assert_array_equal(cache[2], v[max(0,end-window):end])
        np.testing.assert_allclose(np.concatenate(outputs),expected,rtol=2e-14,atol=2e-14)
print('PASS all 128 partitions × 4 windows against independent scalar reference')

# Two ways of mishandling a chunk must fail, not accidentally agree.
q0 = k0 = np.zeros((10,2))
v0 = np.arange(1.,11.)[:,None]
correct,_ = ns['window_attention'](q0,k0,v0,4)
premature = ns['attention'](q0[4:5],k0[3:7],v0[3:7],np.array([[1,1,0,0]],bool))
assert premature.item() == 4.5 and correct[4,0] == 3.5
unshifted = ns['attention'](q0[4:7],k0[:7],v0[:7],np.tril(np.ones((3,7),bool)))
assert not np.allclose(unshifted,correct[4:7])
print('PASS premature-overwrite and top-left-mask negative controls')

# Boolean paths give the receptive field, not the degree of factual retention.
rows,cols = np.arange(16)[:,None],np.arange(16)[None,:]
local = (cols<=rows) & (cols>=rows-3)
paths = np.eye(16,dtype=bool)
for count,wanted in enumerate((4,7,10),1):
    paths = (local.astype(int)@paths.astype(int))>0
    assert paths[-1].sum() == wanted
print('PASS three-layer receptive fields: 4, 7, 10')

# A causal stack cannot transmit a changed future value to an earlier query.
mask = ns['causal_pattern']
before = ns['two_layers'](mask).copy()
ns['values'][7,0] = 999
after = ns['two_layers'](mask)
np.testing.assert_array_equal(before[:7],after[:7])
print('PASS future-value intervention')

# A separate two-layer RoPE calculation: even perfectly rephased old keys
# do not reconstruct hidden states originally computed under a different grid.
def rotate(x, base):
    out = x.copy()
    for t,j in itertools.product(range(len(x)), range(2)):
        angle = t*base**(-j/2)
        c,s = math.cos(angle),math.sin(angle)
        a,b = x[t,2*j:2*j+2]
        out[t,2*j:2*j+2] = [a*c-b*s,a*s+b*c]
    return out

x = np.array([[-1,.5,.3,-.2],[.2,-.7,.8,.1],[.4,.6,-.9,.3],
              [-.5,.2,.1,.7],[.9,-.4,.2,-.3]])
causal = np.tril(np.ones((5,5),bool))
def block(x,base):
    rotated = rotate(x,base)
    return x+ns['attention'](rotated,rotated,x,causal)
old,new = block(x,100.),block(x,10000.)
fresh = block(new,10000.)[-1:]
stale_values = np.concatenate((old[:-1],new[-1:]))
stale = new[-1:]+ns['attention'](rotate(new,10000.)[-1:],rotate(stale_values,10000.),
                                stale_values,np.ones((1,5),bool))
delta = float(np.max(np.abs(fresh-stale)))
assert math.isclose(delta,.0023262835464963327,abs_tol=2e-15)
print('PASS changed-grid upper-layer cache counterexample',delta)

gib = 2**30
kv = lambda layers,t: 2*layers*t*8*128*2
assert kv(32,8192) == gib and kv(32,131072) == 16*gib
assert (kv(24,4096)+kv(8,32768))/gib == 1.375
assert kv(math.ceil(32/2),32768)/gib == 2
ratio = (131072*131073)/(8192*8193)
assert abs(ratio-255.9707) < .0001
assert 12000/4 == 3000 and 16383/4 == 4095.75
theta = 10000.**(-2*np.arange(64)/128)
ntk_ratio = 4.**(-2*np.arange(64)/126)
np.testing.assert_allclose(ntk_ratio[[0,32,63]],[1,.494528,.25],atol=.0001)
assert math.isclose((1+.1*math.log(4))**2,1.296476993,rel_tol=1e-8)
assert 1000*16*131072 == 2097152000
assert 15000*480*131072 == 943718400000
assert math.isclose(2/2,1) and math.isclose(2/3,.6666666666666666)
print('PASS memory, pair counts, scaling, training-token arithmetic and metric example')
print('PASS CPU explanation checks; no sparse GPU kernel or full-model quality claim')
print('chapter SHA256',hashlib.sha256(CHAPTER.read_bytes()).hexdigest())

# CSA/HCA: an independent scalar reference normalizes across positions per feature.
def scalar_pool(values_a, logits_a, stride, values_b=None, logits_b=None):
    rows = []
    for i in range(len(values_a)//stride):
        result = []
        for d in range(values_a.shape[1]):
            pairs = [(float(logits_a[j,d]), float(values_a[j,d]))
                     for j in range(i*stride, (i+1)*stride)]
            if values_b is not None and i > 0:
                pairs += [(float(logits_b[j,d]), float(values_b[j,d]))
                          for j in range((i-1)*stride, i*stride)]
            largest = max(score for score, _ in pairs)
            weights = [math.exp(score-largest) for score, _ in pairs]
            result.append(sum(w*value for w, (_, value) in zip(weights, pairs))/sum(weights))
        rows.append(result)
    return np.array(rows).reshape(-1, values_a.shape[1])

a_pool, b_pool = rng.normal(size=(7,3)), rng.normal(size=(7,3))
za_pool, zb_pool = rng.normal(size=(7,3)), rng.normal(size=(7,3))
for stride in (1,2,3,8):
    for overlap in (False, True):
        b_arg, z_arg = (b_pool, zb_pool) if overlap else (None, None)
        actual = ns['pool_position_groups'](a_pool, za_pool, stride, b_arg, z_arg)
        expected = scalar_pool(a_pool, za_pool, stride, b_arg, z_arg)
        np.testing.assert_allclose(actual, expected, atol=2e-14)
        for prefix in range(1,8):
            short = ns['pool_position_groups'](
                a_pool[:prefix], za_pool[:prefix], stride,
                None if b_arg is None else b_arg[:prefix],
                None if z_arg is None else z_arg[:prefix]
            )
            np.testing.assert_allclose(short, actual[:prefix//stride], atol=2e-14)
        for query in range(7):
            changed_a = a_pool.copy()
            changed_b = b_pool.copy()
            changed_a[query+1:] += 100
            changed_b[query+1:] -= 200
            altered = ns['pool_position_groups'](
                changed_a, za_pool, stride,
                changed_b if overlap else None, zb_pool if overlap else None
            )
            # A query reads only complete preceding groups, not its own group.
            np.testing.assert_allclose(
                altered[:query//stride], actual[:query//stride], atol=2e-14
            )
# The first overlapped entry must not silently add zero-valued padding to its denominator.
first_pool = ns['pool_position_groups'](
    np.array([[1.,10.],[3.,30.]]), np.zeros((2,2)), 2,
    np.zeros((2,2)), np.zeros((2,2))
)
np.testing.assert_allclose(first_pool, [[2.,20.]])
assert not np.allclose(first_pool, [[1.,10.]])
# Two separately normalized branches do not equal concatenation before one softmax.
local = np.array([[7.,70.],[9.,90.]])
long = np.array([[3.75,45.]])
q = np.zeros((1,2))
combined = ns['attention'](q, np.concatenate((long,local)),
                            np.concatenate((long,local)), np.ones((1,3),bool))
separate = (ns['attention'](q,long,long,np.ones((1,1),bool))
            + ns['attention'](q,local,local,np.ones((1,2),bool)))
assert not np.allclose(combined, separate)
assert math.isclose(2/(2+math.exp(0)), 2/3)
# Inverse output rotation is linear; this does not assert anything about trained logits.
def rotation_pool(angle):
    return np.array([[math.cos(angle), -math.sin(angle)],
                     [math.sin(angle), math.cos(angle)]])
values_pool = rng.normal(size=(3,2))
weights_pool = np.array([.2,.3,.5])
positions_pool = np.array([0.,2.,5.])
query_position_pool = 6.
left_pool = rotation_pool(-query_position_pool) @ sum(
    w*(rotation_pool(pos) @ value)
    for w,pos,value in zip(weights_pool, positions_pool, values_pool)
)
right_pool = sum(
    w*(rotation_pool(pos-query_position_pool) @ value)
    for w,pos,value in zip(weights_pool, positions_pool, values_pool)
)
np.testing.assert_allclose(left_pool, right_pool, atol=1e-14)
assert 1048576//4 == 262144 and 1048576//128 == 8192
assert 1024+128 == 1152 and 8192+128 == 8320
print('PASS CSA/HCA scalar pooling, padding, prefix/tail, causality, shared softmax and rotation')
