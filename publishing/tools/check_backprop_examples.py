"""Execute published blocks, original micrograd and manual backprop cases."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import ast
import hashlib
import math
import re
import runpy
import textwrap

import numpy as np
import torch
from torch.autograd.gradcheck import GradcheckError

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/01 Основы нейронных сетей/01 Нейрон, градиент и backpropagation.md'
SOURCE = ROOT / '05 Источники/Originals/backpropagation-2026-10-21/micrograd-engine.py'
current_text = CHAPTER.read_text()
additions = re.findall(r'<!-- BEGIN REVIEW: (forward mode|custom backward) 2026-10-27 -->\n(.*?)<!-- END REVIEW: \1 2026-10-27 -->\n\n', current_text, re.S)
assert [name for name, _ in additions] == ['forward mode', 'custom backward']
baseline = re.sub(r'<!-- BEGIN REVIEW: JAX transforms 2026-10-27 -->\n.*?<!-- END REVIEW: JAX transforms 2026-10-27 -->\n\n', '', current_text, flags=re.S)
baseline = re.sub(r'<!-- BEGIN REVIEW: (forward mode|custom backward) 2026-10-27 -->\n.*?<!-- END REVIEW: \1 2026-10-27 -->\n\n', '', baseline, flags=re.S)
baseline = baseline.replace('last_updated: 2026-10-27', 'last_updated: 2026-10-21')
baseline = baseline.replace(r'''\begin{aligned}
\frac{\partial L}{\partial W_{kj}}
&=\sum_i X_{ik}G_{ij},\\
\frac{\partial L}{\partial b_j}&=\sum_i G_{ij}.
\end{aligned}''', r'''\frac{\partial L}{\partial W_{kj}}
=\sum_i X_{ik}G_{ij},
\qquad
\frac{\partial L}{\partial b_j}=\sum_i G_{ij}.''')
baseline = baseline.replace(r'''\begin{aligned}
b&=(0.5,-0.5),\\[4pt]
Z&=\begin{pmatrix}4.5&4.5\\5.5&-6.5\end{pmatrix}.
\end{aligned}''', r'''b=(0.5,-0.5),\qquad
Z=\begin{pmatrix}4.5&4.5\\5.5&-6.5\end{pmatrix}.''')
assert hashlib.sha256(baseline.encode()).hexdigest() == 'd329325d295f0fc50ddbf013412c96e0f24f8a31bd6733754e08ae23a5baa198'
blocks = re.findall(r'```python\n(.*?)```', baseline, re.S)
assert len(blocks) == 6
# The published class excerpt is explicitly not a standalone program.
engine = SOURCE.read_text()
methods = ast.parse(engine).body[0].body
def original_method(name):
    n = next(n for n in methods if isinstance(n, ast.FunctionDef) and n.name == name)
    return ''.join(engine.splitlines(keepends=True)[n.lineno-1:n.end_lineno]).rstrip()
mul = original_method('__mul__')
backward = original_method('backward')
assert textwrap.dedent(blocks[1]).strip() == textwrap.dedent(mul) + '\n\n' + textwrap.dedent(backward)
scopes = []
for i, code in enumerate(blocks):
    if i == 1:
        scopes.append(None)
        continue
    scope = {}
    with redirect_stdout(StringIO()) as output:
        exec(compile(code, f'{CHAPTER}:block{i+1}', 'exec'), scope)
    scopes.append(scope)
    print(f'block {i+1}:', output.getvalue().strip() or 'PASS')

matrix, _, neuron, accumulation, vjp, difference = scopes
for key, expected in {'Z': [[4.5,4.5],[5.5,-6.5]], 'W': [[-2,11],[3,1]],
                      'X': [[0,7],[-5,8]], 'b': [0,5]}.items():
    actual = matrix[key].detach() if key == 'Z' else matrix[key].grad
    np.testing.assert_allclose(actual, expected, atol=1e-14)
assert matrix['loss'].item() == -11.5
np.testing.assert_allclose([neuron['before'],neuron['after']], [1.068893290777046,.3299351372391118], atol=1e-14)
np.testing.assert_allclose(neuron['w'].detach(), [2.1149878723657722,-.7700242552684557], atol=1e-14)
assert abs(neuron['b'].item()+.3850121276342279) < 1e-14
np.testing.assert_allclose(neuron['w'].grad, [-1.149878723657721,-2.299757447315442], atol=1e-14)

# Use the exact original engine, not a reconstructed Value implementation.
Value = runpy.run_path(str(SOURCE))['Value']
x,y,z = Value(-2),Value(5),Value(-4)
out = (x+y)*z; out.backward()
assert out.data == -12 and (x.grad,y.grad,z.grad) == (-4,-4,3)
a,b = Value(2),Value(1)
out = (a+b)*(b+1); out.backward()
assert out.data == 6 and (a.grad,b.grad) == (2,5)
x = Value(3); out = x*x; out.backward()
assert out.data == 9 and x.grad == 6
x,y,z,w = [Value(t) for t in (3,-4,2,-1)]
# max(z,w), with z>w, locally equals z; w is not a graph ancestor here.
out = 2*(x*y+z); out.backward()
assert out.data == -20 and (x.grad,y.grad,z.grad,w.grad) == (-8,6,2,0)
w = Value(2); out = (w*1)**2+(w*3)**2; out.backward()
assert out.data == 40 and w.grad == 40
x = Value(3); yy = x+2*x; out = yy**2/2; out.backward()
assert yy.data == 9 and x.grad == 27
x = Value(3); out = x+(-x); out.backward(); assert x.grad == 0

# Both matrix broadcasting and an activation backward, checked independently.
xx = torch.arange(12, dtype=torch.float64).reshape(2,3,2).requires_grad_()
bb = torch.tensor([.5,-.5],dtype=torch.float64,requires_grad=True)
(xx+bb).square().sum().backward()
torch.testing.assert_close(bb.grad, (2*(xx.detach()+bb.detach())).sum((0,1)))
xx = matrix['X'].detach().clone().requires_grad_()
ww = matrix['W'].detach().clone().requires_grad_()
zz = xx@ww+matrix['b'].detach()
gg = matrix['G']
(zz.tanh()*gg).sum().backward()
local = gg*(1-zz.detach().tanh().square())
torch.testing.assert_close(ww.grad, xx.detach().T@local)
torch.testing.assert_close(xx.grad, local@ww.detach().T)

# Unequal microbatches: weighted means match a five-example mean at fixed weights.
q = torch.tensor(2., dtype=torch.float64, requires_grad=True)
data = torch.tensor([1.,2.,3.,4.,5.], dtype=torch.float64)
full = torch.autograd.grad((q*data).square().mean(),q)[0]
(.4*(q*data[:2]).square().mean()).backward()
(.6*(q*data[2:]).square().mean()).backward()
torch.testing.assert_close(q.grad, full)
wrong = .5*(2*q.detach()*data[:2].square().mean()+2*q.detach()*data[2:].square().mean())
assert not torch.isclose(wrong,full)

# VJP and individual output derivatives are different requests.
u = torch.tensor(2.,requires_grad=True)
torch.testing.assert_close(torch.autograd.functional.jacobian(lambda u: torch.stack((u*u,3*u)),u),torch.tensor([4.,3.]))
assert vjp['u'].grad.item() == 10

# Detach, freeze and no_grad alter the actual recorded function, not its values.
for detached, expected in [(False,4),(True,2)]:
    x=torch.tensor(2.,requires_grad=True); out=x*(x.detach() if detached else x)
    out.backward(); assert out.item()==4 and x.grad.item()==expected
x=torch.tensor(2.,requires_grad=True); w=torch.tensor(3.)
(w*x).backward(); assert x.grad.item()==3 and w.grad is None
with torch.no_grad():
    assert not (x*x).requires_grad
module=torch.nn.Linear(1,1).eval()
assert module(torch.ones(1,1)).requires_grad

# Masking after an invalid operation does not remove it from the backward graph.
for safe in (False,True):
    x=torch.tensor([1.,1.],requires_grad=True); div=torch.tensor([0.,1.]); mask=div!=0
    out=(x[mask]/div[mask]).sum() if safe else (x/div)[mask].sum()
    out.backward(); assert out.item()==1
    if safe: torch.testing.assert_close(x.grad,torch.tensor([0.,1.]))
    else: assert torch.isnan(x.grad[0]) and x.grad[1]==1
x=torch.tensor(2.,requires_grad=True); out=x*x
with torch.no_grad(): x.add_(1)
try: out.backward()
except RuntimeError as err: assert 'modified by an inplace operation' in str(err)
else: raise AssertionError('expected saved-value version check')

for eps in (1e-3,1e-5,1e-6):
    f=difference['scalar_loss']; a,b=2.,1.
    np.testing.assert_allclose([(f(a+eps,b)-f(a-eps,b))/(2*eps),
                               (f(a,b+eps)-f(a,b-eps))/(2*eps)],[2,5],atol=1e-8)
assert torch.autograd.gradcheck(lambda a,b:(a+b)*(b+1),
    (torch.tensor(2.,dtype=torch.float64,requires_grad=True),torch.tensor(1.,dtype=torch.float64,requires_grad=True)))
zero=torch.tensor(0.,requires_grad=True); zero.relu().backward(); assert zero.grad.item()==0
assert (max(0,1e-6)-max(0,-1e-6))/(2e-6)==.5
np.testing.assert_allclose(np.array([6.,8.])*min(1,5/10),[3,4])
assert 2**20==1048576 and abs(.5**20-9.5367431640625e-7)<1e-20
print('PASS: five runnable blocks; unchanged original methods; scalar/branched/tied/residual '
      'graphs; tensor backward/broadcast; SGD; accumulation; VJP; detach/freeze/modes; '
      'invalid-mask and in-place failures; central differences/gradcheck; clipping. torch',torch.__version__)

# Execute all new fences exactly as printed, in their explanatory order.
new_blocks = [code for _, addition in additions
              for code in re.findall(r'```python\n(.*?)```', addition, re.S)]
assert len(new_blocks) == 9, len(new_blocks)
new = {'__name__': '__chapter_example__'}
for i, code in enumerate(new_blocks, 1):
    with redirect_stdout(StringIO()) as output:
        exec(compile(code, f'{CHAPTER}:AD-block{i}', 'exec'), new)
    print(f'AD block {i}:', output.getvalue().strip() or 'PASS')

qmd = (ROOT/'05 Источники/Originals/frameworks-2026-10-27/chapter-2bd97c5.qmd').read_text()
def source_example(label):
    region = qmd.split('{#'+label, 1)[1]
    return region.split('```{.python}\n', 1)[1].split('```', 1)[0].strip()
assert new_blocks[0].removeprefix('from math import sin, cos\n\n').strip() == source_example('lst-forward_mode_ad')
assert new_blocks[4].removeprefix('import torch\n\n').strip() == source_example('lst-custom-autograd-function')

# Independent arithmetic and directional seeds, including zero and negative.
for value in (-3., -1., 0., .5, 2.):
    expected = value*value*math.sin(value)
    derivative = 2*value*math.sin(value)+value*value*math.cos(value)
    np.testing.assert_allclose(new['f'](value), (expected, derivative), atol=1e-14)
    for seed in (-2., 0., 1., 3.):
        p, j = torch.func.jvp(new['square_sine'],
            (torch.tensor(value, dtype=torch.double),),
            (torch.tensor(seed, dtype=torch.double),))
        assert abs(p.item()-expected) < 1e-13
        assert abs(j.item()-seed*derivative) < 1e-13

# Follow every primal/tangent/adjoint in the paper's two original tables.
v = {'x1':2., 'x2':5., 'v1':math.log(2.), 'v2':10., 'v3':math.sin(5.)}
v['v4'] = v['v1']+v['v2']; v['v5'] = v['v4']-v['v3']
assert abs(v['v5']-11.652071455223086) < 1e-14
tangents = (1., 0., .5, 5., 0., 5.5, 5.5)
assert tangents[-1] == new['d1'].item() == 5.5
bar_v5=1.; bar_v4=bar_v5; bar_v3=-bar_v5
bar_v1=bar_v4; bar_v2=bar_v4
bar_x1=bar_v2*v['x2'] + bar_v1/v['x1']
bar_x2=bar_v3*math.cos(v['x2']) + bar_v2*v['x1']
np.testing.assert_allclose(new['full_gradient'], [bar_x1,bar_x2], atol=1e-14)

adjoint_cases = 0
for a in (.5, 2., 4.):
    for b in (-1., 0., 5.):
        point=torch.tensor([a,b],dtype=torch.double)
        gradient=torch.tensor([1/a+b,a-math.cos(b)],dtype=torch.double)
        primal, pullback=torch.func.vjp(new['baydin'],point)
        torch.testing.assert_close(pullback(torch.ones_like(primal))[0],gradient)
        for direction in ((1.,0.),(0.,1.),(2.,-1.)):
            h=point.new_tensor(direction)
            _, jh=torch.func.jvp(new['baydin'],(point,),(h,))
            torch.testing.assert_close(jh,gradient@h)
            for seed in (-3.,0.,2.):
                g=point.new_tensor(seed)
                torch.testing.assert_close(g*jh,pullback(g)[0]@h)
                adjoint_cases += 1
        h=point.new_tensor([1.,-2.])
        _, hvp=torch.func.jvp(torch.func.grad(new['curvature_loss']),(point,),(h,))
        hessian=point.new_tensor([[2*b,2*a],[2*a,-math.sin(b)]])
        torch.testing.assert_close(hvp,hessian@h)

# Compare both custom interfaces with ordinary built-in operations, including
# engine reduction of expanded gradients for the original combined interface.
generator=torch.Generator().manual_seed(27)
shapes = [((),(),()), ((2,),(2,),(2,)), ((2,1),(1,3),()),
          ((),(2,3),(3,)), ((1,3),(2,1),(2,3))]
for shapes3 in shapes:
    args=tuple(torch.randn(s,dtype=torch.double,generator=generator).requires_grad_()
               for s in shapes3)
    reference=args[0]*args[1]+args[2]
    incoming=torch.randn(reference.shape,dtype=torch.double,generator=generator)
    expected=torch.autograd.grad(reference,args,incoming,retain_graph=True)
    for cls in (new['MultiplyAdd'],new['MultiplyAddAD']):
        actual=cls.apply(*args)
        torch.testing.assert_close(actual,reference)
        for got,want in zip(torch.autograd.grad(actual,args,incoming),expected):
            torch.testing.assert_close(got,want)
        assert torch.autograd.gradcheck(cls.apply,args,
            check_forward_ad=cls is new['MultiplyAddAD'])
        assert torch.autograd.gradgradcheck(cls.apply,args)

# Negative control: correct first-derivative values can hide a broken graph
# for differentiating backward. This is not an accepted implementation.
class DetachedBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,y,z):
        ctx.save_for_backward(x,y)
        return x*y+z
    @staticmethod
    def backward(ctx,g):
        x,y=ctx.saved_tensors
        return (g*y).detach(),(g*x).detach(),g.detach()
args=tuple(torch.tensor([v],dtype=torch.double,requires_grad=True) for v in (2.,3.,1.))
assert torch.autograd.gradcheck(DetachedBackward.apply,args)
try:
    torch.autograd.gradgradcheck(DetachedBackward.apply,args)
except GradcheckError:
    pass
else:
    raise AssertionError('Detached backward must fail higher-order checking')

print(f'PASS: {len(new_blocks)} additional actual fences, exact Harvard examples, '
      f'20 square/sine directions, all paper trace values, {adjoint_cases} adjoint '
      'identities, 9 explicit HVP checks, both custom interfaces over 5 shape '
      'patterns, reverse/forward gradcheck, gradgradcheck and a failing negative control.')
print('SHA-256:',hashlib.sha256(current_text.encode()).hexdigest())
