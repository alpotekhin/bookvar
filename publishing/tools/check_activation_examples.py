"""Execute the current chapter and verify activation mechanisms in float64."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import hashlib
import math
import re

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, ndtr
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/01 Основы нейронных сетей/02 Функции активации.md"
chapter_text = CHAPTER.read_text()
blocks = re.findall(r"```python\n(.*?)```", chapter_text, re.S)
assert len(blocks) == 5
scopes = []
for i, code in enumerate(blocks, 1):
    scope = {}
    with redirect_stdout(StringIO()) as captured:
        exec(compile(code, str(CHAPTER)+f":block{i}", "exec"), scope)
    print(f"Block {i}:\n{captured.getvalue()}", end="")
    scopes.append(scope)

# Chain rule: a supplied upstream derivative differs from a complete output loss.
assert F.relu(torch.tensor(-2.)).item() == 0.
np.testing.assert_allclose(3*expit(-2)*(1-expit(-2)), .314981, atol=.5e-6)
np.testing.assert_allclose(expit([-5, 0, 5]), [.006693, .5, .993307], atol=.5e-6)
np.testing.assert_allclose(expit([-5, 0, 5])*(1-expit([-5, 0, 5])),
                           [.006648, .25, .006648], atol=.5e-6)
start = torch.tensor(0., dtype=torch.float64, requires_grad=True)
state = start
for depth in range(10):
    state = torch.sigmoid(state-(0 if depth == 0 else .5))
state.backward()
np.testing.assert_allclose(start.grad.item(), .25**10, atol=1e-20)
assert .25**10 < 1e-6
h = np.array([[.2, .8], [.8, .2]])
np.testing.assert_allclose((h*np.array([[1.], [-1.]])).mean(axis=0), [-.3, .3])

np.testing.assert_allclose(scopes[0]["loss"].item(), 5.006715, atol=.5e-6)
np.testing.assert_allclose(scopes[0]["z"].grad.item(), -.993307, atol=.5e-6)
for value, target in [(-1000., 1.), (1000., 0.), (-5., 1.), (5., 0.)]:
    logit = torch.tensor(value, dtype=torch.float64, requires_grad=True)
    loss = F.binary_cross_entropy_with_logits(logit, torch.tensor(target))
    loss.backward()
    assert torch.isfinite(loss)
    np.testing.assert_allclose(logit.grad.item(), expit(value)-target, atol=1e-15)
p = expit(-5)
np.testing.assert_allclose(((p-1)/(p*(1-p)))*p*(1-p), p-1, atol=1e-15)

np.testing.assert_allclose(np.tanh([-1, 1]).mean(), 0, atol=1e-15)
np.testing.assert_allclose(np.tanh([1, 1]).mean(), .761594, atol=.5e-6)
zz = np.linspace(-8, 8, 101)
np.testing.assert_allclose(np.tanh(zz), 2*expit(2*zz)-1, atol=1e-15)
np.testing.assert_allclose(1-np.tanh([2, 3])**2, [.07065, .00987], atol=.5e-5)

# Isolated dead ReLU versus the explicitly specified LeakyReLU.
for slope, expected_dw, expected_db in [(0., 0., 0.), (.01, -.0154, -.01025)]:
    w = torch.tensor(-1., dtype=torch.float64, requires_grad=True)
    b = torch.tensor(-1., dtype=torch.float64, requires_grad=True)
    xx = torch.tensor([1., 2.], dtype=torch.float64)
    out = F.leaky_relu(xx*w+b, negative_slope=slope)
    ((out-1).square()/2).mean().backward()
    np.testing.assert_allclose([w.grad.item(), b.grad.item()], [expected_dw, expected_db], atol=1e-15)
zero = torch.tensor(0., requires_grad=True)
zero.relu().backward()
assert zero.grad.item() == 0.
negative = torch.tensor([-2.], dtype=torch.float64, requires_grad=True)
alpha = torch.tensor([.01], dtype=torch.float64, requires_grad=True)
(3*F.prelu(negative, alpha)).sum().backward()
assert alpha.grad.item() == -6.

# ELU's one-sided slopes match at zero only for alpha == 1.
for alpha in [1., 2.]:
    eps = 1e-7
    left = -alpha*np.expm1(-eps)/eps
    right = 1.
    np.testing.assert_allclose(left, alpha, rtol=1e-6)
    assert np.isclose(left, right) == (alpha == 1)
np.testing.assert_allclose([np.expm1(-2), np.exp(-2), np.exp(-5)],
                          [-.8647, .1353, .006738], atol=5e-5)

rho = lambda z: np.exp(-z*z/2)/math.sqrt(2*math.pi)
gelu = lambda z: z*ndtr(z)
gelu_grad = lambda z: ndtr(z)+z*rho(z)
silu = lambda z: z*expit(z)
silu_grad = lambda z: expit(z)+z*expit(z)*(1-expit(z))
for fn, derivative, minimum_x, minimum_value in [
    (gelu, gelu_grad, -.7518, -.1700),
    (silu, silu_grad, -1.2785, -.2785),
]:
    root = brentq(derivative, -2, 0)
    np.testing.assert_allclose(root, minimum_x, atol=5e-5)
    np.testing.assert_allclose(fn(root), minimum_value, atol=5e-5)
    assert derivative(root-.01) < 0 < derivative(root+.01)
    assert fn(root) < fn(-100) and fn(root) < fn(100)

# Published z=-1 table and automatic/analytical/finite-difference slopes away from kinks.
cases = [
    (torch.sigmoid, lambda z: expit(z), lambda z: expit(z)*(1-expit(z)), .268941, .196612),
    (torch.tanh, np.tanh, lambda z: 1-np.tanh(z)**2, -.761594, .419974),
    (F.relu, lambda z: np.maximum(0, z), lambda z: (z>0).astype(float), 0., 0.),
    (lambda z: F.leaky_relu(z, .01), lambda z: np.where(z<0, .01*z, z),
     lambda z: np.where(z<0, .01, 1), -.01, .01),
    (F.elu, lambda z: np.where(z<0, np.expm1(z), z),
     lambda z: np.where(z<0, np.exp(z), 1), -.632121, .367879),
    (lambda z: F.gelu(z, approximate="none"), gelu, gelu_grad, -.158655, -.083315),
    (F.silu, silu, silu_grad, -.268941, .072329),
]
points = np.array([-5., -2., -1., -.1, .1, 1., 2., 5.])
for torch_fn, formula, derivative, at_minus1, slope_minus1 in cases:
    z = torch.tensor(points, dtype=torch.float64, requires_grad=True)
    out = torch_fn(z); out.sum().backward()
    np.testing.assert_allclose(out.detach().numpy(), formula(points), atol=2e-15)
    np.testing.assert_allclose(z.grad.numpy(), derivative(points), atol=2e-15)
    eps = 1e-5
    np.testing.assert_allclose((formula(points+eps)-formula(points-eps))/(2*eps),
                              derivative(points), atol=1e-9)
    np.testing.assert_allclose(formula(np.array([-1.])), [at_minus1], atol=.5e-6)
    np.testing.assert_allclose(derivative(np.array([-1.])), [slope_minus1], atol=.5e-6)
np.testing.assert_allclose(gelu_grad(np.array([-1., 1.])), [-.083315, 1.083315], atol=.5e-6)
np.testing.assert_allclose(silu_grad(np.array([-2., -1., 2.])), [-.090784, .072329, 1.090784], atol=.5e-6)
np.testing.assert_allclose(F.gelu(torch.tensor(-1., dtype=torch.float64), approximate="tanh").item(),
                           -.158808, atol=.5e-6)
np.testing.assert_allclose(points*expit(0*points), points/2)
np.testing.assert_allclose(points*expit(1000*points), np.maximum(points, 0), atol=1e-15)

# Replace only the third hidden activation in the preceding chapter's MLP.
first_object = torch.tensor([1., 2.], dtype=torch.float64)
W1 = torch.tensor([[1., -1., 2.], [2., 1., -1.]], dtype=torch.float64)
W2 = torch.tensor([[.2, -.2], [.1, .1], [-.25, .25]], dtype=torch.float64)
pre = first_object @ W1 + first_object.new_tensor([0., 1., -1.])
torch.testing.assert_close(pre, first_object.new_tensor([5., 2., -1.]))
comparison = chapter_text.split("### Замена ReLU в одном нейроне MLP", 1)[1].split("</", 1)[0]
ran = []
for name, fn, expected in [
    ("ReLU", F.relu, (0., .880797, 0.)),
    ("GELU", F.gelu, (-.158655, .888878, -.004629)),
    ("SiLU", F.silu, (-.268941, .894209, .003826)),
]:
    z3 = pre[2].clone().requires_grad_()
    h3 = fn(z3)
    h = torch.cat((pre[:2].relu(), h3[None]))
    logits = h @ W2
    p0 = logits.softmax(dim=0)[0]
    loss = F.cross_entropy(logits[None, :], torch.tensor([0]))
    loss.backward()
    np.testing.assert_allclose([h3.item(), p0.item(), z3.grad.item()], expected,
                               atol=.5e-6, rtol=0)
    published = next(line for line in comparison.splitlines()
                     if line.startswith(f"| {name} |")).split("|")[2:5]
    published = [float(v.strip().replace("−", "-").replace(",", ".")) for v in published]
    np.testing.assert_array_equal(np.round([h3.item(), p0.item(), z3.grad.item()], 5),
                                  published)
    # Check the derivative of the complete loss, not only the local slope.
    def first_loss(value):
        with torch.no_grad():
            h3 = fn(z3.new_tensor(value))
            return F.cross_entropy(
                (torch.cat((pre[:2].relu(), h3[None])) @ W2)[None, :], torch.tensor([0])
            ).item()
    eps = 1e-5
    finite = (first_loss(-1+eps) - first_loss(-1-eps)) / (2*eps)
    np.testing.assert_allclose(finite, z3.grad.item(), atol=1e-10, rtol=0)
    ran.append(z3.grad.item())
assert ran[1] < ran[0] == 0 < ran[2]

# SwiGLU's actual matrix example, both branch derivatives, and equal matrix counts.
np.testing.assert_allclose(scopes[1]["v"].numpy(), [[1.462117, .806824]], atol=.5e-6)
np.testing.assert_allclose(scopes[1]["out"].numpy(), [[.655293, .806824]], atol=.5e-6)
gate = torch.tensor(0., dtype=torch.float64, requires_grad=True)
up = torch.tensor(4., dtype=torch.float64, requires_grad=True)
product = F.silu(gate)*up
product.backward()
np.testing.assert_allclose([product.item(), up.grad.item(), gate.grad.item()], [0, 0, 2])
assert 2*3*12 == 3*3*8 == 72

# The published Maxout uses all affine branches before reducing the piece axis.
Maxout = scopes[2]["Maxout"]
torch.testing.assert_close(scopes[2]["out"].detach(),
    torch.tensor([[2.], [.5], [2.]], dtype=torch.float64))
torch.testing.assert_close(scopes[2]["x"].grad,
    torch.tensor([[-1.], [0.], [1.]], dtype=torch.float64))
torch.testing.assert_close(scopes[2]["layer"].projections.weight.grad,
    torch.tensor([[2.], [-2.], [-.25]], dtype=torch.float64))
torch.testing.assert_close(scopes[2]["layer"].projections.bias.grad,
    torch.ones(3, dtype=torch.float64))
assert sum(p.numel() for p in scopes[2]["batched"].parameters()) == 18
# The chapter's 2D example is independent of the authors' Figure 6 weights.
plane = Maxout(2, 1, 4).double()
with torch.no_grad():
    plane.projections.weight.copy_(torch.tensor(
        [[1., 0.], [-1., 0.], [0., 1.], [0., -1.]], dtype=torch.float64))
    plane.projections.bias.zero_()
xy = torch.tensor([[2., 1.], [1., 2.], [1., 1.]], dtype=torch.float64,
                  requires_grad=True)
xy_scores = plane.projections(xy)
torch.testing.assert_close(xy_scores, torch.tensor(
    [[2., -2., 1., -1.], [1., -1., 2., -2.], [1., -1., 1., -1.]],
    dtype=torch.float64))
heights = plane(xy)
torch.testing.assert_close(heights.flatten(), torch.tensor([2., 2., 1.],
                                                           dtype=torch.float64))
heights[:2].sum().backward()
torch.testing.assert_close(xy.grad[:2], torch.eye(2, dtype=torch.float64))
step = torch.tensor([1e-5, -1e-5], dtype=torch.float64)
left = (heights[-1] - plane(xy[-1] - step)) / step[0]
right = (plane(xy[-1] + step) - heights[-1]) / step[0]
torch.testing.assert_close(torch.cat([left, right]),
                           torch.tensor([-1., 1.], dtype=torch.float64))
for bad in [0, -1]:
    try:
        Maxout(2, 3, bad)
    except ValueError:
        pass
    else:
        raise AssertionError("nonpositive piece count accepted")

torch.manual_seed(336)
probe = Maxout(2, 3, 4).double()
xx = torch.randn(2, 5, 2, dtype=torch.float64, requires_grad=True)
weights = probe.projections.weight.detach().reshape(3, 4, 2)
biases = probe.projections.bias.detach().reshape(3, 4)
# Independent shape convention [output, piece, input], unlike the layer's reshape.
scores = torch.einsum("...d,ikd->...ik", xx.detach(), weights) + biases
manual = torch.tensor([
    [max(float(scores[b, t, i, j]) for j in range(4))
     for i in range(3)] for b in range(2) for t in range(5)
], dtype=torch.float64).reshape(2, 5, 3)
values = probe(xx)
assert values.shape == (2, 5, 3)
torch.testing.assert_close(values.detach(), manual)
assert probe(torch.zeros(2, dtype=torch.float64)).shape == (3,)
upstream = torch.randn(2, 5, 3, dtype=torch.float64)
(values * upstream).sum().backward()
winners = scores.argmax(-1)
mask = F.one_hot(winners, num_classes=4).double() * upstream[..., None]
torch.testing.assert_close(xx.grad, torch.einsum("...ik,ikd->...d", mask, weights))
torch.testing.assert_close(probe.projections.weight.grad.reshape(3, 4, 2),
    torch.einsum("btik,btd->ikd", mask, xx.detach()))
torch.testing.assert_close(probe.projections.bias.grad.reshape(3, 4), mask.sum((0, 1)))
# No equal maxima in this probe; central differences therefore test ordinary derivatives.
ordered = scores.sort(-1).values
assert (ordered[..., -1] - ordered[..., -2]).min() > 1e-5
with torch.no_grad():
    eps = 1e-6
    for coord in [(0, 0, 0), (1, 4, 1)]:
        plus, minus = xx.detach().clone(), xx.detach().clone()
        plus[coord] += eps
        minus[coord] -= eps
        finite = ((probe(plus) - probe(minus)) * upstream).sum() / (2*eps)
        torch.testing.assert_close(finite, xx.grad[coord], atol=1e-8, rtol=1e-8)

torch.testing.assert_close(scopes[3]["first"].grad,
    torch.tensor([[1., 0., 0.]], dtype=torch.float64))
torch.testing.assert_close(scopes[3]["shared"].grad,
    torch.tensor([[.5, .5, 0.]], dtype=torch.float64))
losing = torch.tensor([[1., 0.], [-2., -3.]], requires_grad=True)
losing.max(dim=-1).values.sum().backward()
torch.testing.assert_close(losing.grad, torch.tensor([[1., 0.], [1., 0.]]))
grid = np.linspace(-3, 3, 601)
np.testing.assert_allclose(np.maximum(grid, -grid), np.abs(grid))
for alpha in [0., .01, .2, 1.]:
    np.testing.assert_allclose(np.maximum(alpha*grid, grid),
        np.where(grid < 0, alpha*grid, grid))
# Tangent construction: exact error identity and uniform compact-domain bound.
domain = np.linspace(-1, 1, 1001)
for count in [3, 5, 11, 33]:
    points = np.linspace(-1, 1, count)
    envelope = (2*domain[:, None]*points - points**2).max(1)
    error = domain**2 - envelope
    np.testing.assert_allclose(error, ((domain[:, None]-points)**2).min(1), atol=1e-15)
    assert error.min() >= -1e-15 and error.max() <= (2/(count-1))**2/4 + 1e-15
# A single maxout is convex; -abs(x) violates Jensen's inequality at -1, 1.
u, v, theta = -.7, .8, .35
envelope = lambda x: max(-2*x-1, 0, 2*x-1)
assert envelope(theta*u+(1-theta)*v) <= theta*envelope(u)+(1-theta)*envelope(v)
assert -abs(0.) > (-abs(-1.)-abs(1.))/2
np.testing.assert_allclose([envelope(x) for x in [-1, -.5, 0, .5, 1]], [1, 0, 0, 0, 1])

sym = np.array([-2., -1., 1., 2.]); relu = np.maximum(sym, 0)
np.testing.assert_allclose([np.mean(sym**2), np.mean(relu**2), relu.mean(), relu.var()],
                           [2.5, 1.25, .75, .6875])
bad_logits = torch.tensor([-2., -1.], dtype=torch.float64)
np.testing.assert_allclose(bad_logits.softmax(-1).numpy(), [.268941, .731059], atol=.5e-6)
torch.testing.assert_close(bad_logits.relu().softmax(-1), torch.tensor([.5, .5], dtype=torch.float64))
assert F.relu(torch.tensor(-2.)).item() == 0. != -2.

# Retained D2L files must be byte-identical to their freshly downloaded sources.
source_dir = ROOT / "05 Источники/Originals/activation-functions-2026-10-21"
for name in ["sigmoid", "sigmoid-gradient", "tanh", "tanh-gradient", "relu", "relu-gradient"]:
    figure = ROOT / f"00 Учебник/Assets/Figures/curated/source-first-7-14/d2l-{name}.svg"
    assert figure.read_bytes() == (source_dir / f"d2l-{name}.svg").read_bytes()
print("PASS: 5 published blocks; local chain rule vs output BCE; 10-layer chain; "
      "7 activation values/derivatives; minima; tanh centering; dead/leaky/PReLU; ELU slopes; "
      "GELU approximation; SwiGLU branches/matrices; Maxout values/shapes/VJP/finite differences; "
      "2D regions and directional slopes; tie rules; never-winning branch; convexity and tangent error; "
      "moments; logits and residual; 6 source SVG copies.")
print("Chapter SHA256", hashlib.sha256(CHAPTER.read_bytes()).hexdigest())
print("PyTorch", torch.__version__)

