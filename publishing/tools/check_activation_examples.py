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
blocks = re.findall(r"```python\n(.*?)```", CHAPTER.read_text(), re.S)
assert len(blocks) == 3
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

# SwiGLU's actual matrix example, both branch derivatives, and equal matrix counts.
np.testing.assert_allclose(scopes[1]["v"].numpy(), [[1.462117, .806824]], atol=.5e-6)
np.testing.assert_allclose(scopes[1]["out"].numpy(), [[.655293, .806824]], atol=.5e-6)
gate = torch.tensor(0., dtype=torch.float64, requires_grad=True)
up = torch.tensor(4., dtype=torch.float64, requires_grad=True)
product = F.silu(gate)*up
product.backward()
np.testing.assert_allclose([product.item(), up.grad.item(), gate.grad.item()], [0, 0, 2])
assert 2*3*12 == 3*3*8 == 72

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
print("PASS: 3 published blocks; local chain rule vs output BCE; 10-layer chain; "
      "7 activation values/derivatives; minima; tanh centering; dead/leaky/PReLU; ELU slopes; "
      "GELU approximation; SwiGLU branches/matrices; moments; logits and residual; 6 source SVG copies.")
print("PyTorch", torch.__version__)

