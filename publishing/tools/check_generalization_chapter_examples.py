"""Check published examples and independent calculations with CPU/NumPy."""
from pathlib import Path
from itertools import product
from hashlib import sha256
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/00 Математические и ML-основания/06 Переобучение, регуляризация и оценивание.md"
text = CHAPTER.read_text()
blocks = re.findall(r"```python\n(.*?)```", text, re.S)
assert len(blocks) == 3
scope = {}
for i, code in enumerate(blocks):
    exec(compile(code, f"{CHAPTER}:python-{i+1}", "exec"), scope)
np.testing.assert_allclose(scope["errors"], [
    [.16204346344099427, .09657706635830812],
    [.023216666666666663, .001044716957079676],
    [0, .634623909520895],
], atol=1e-11)
assert scope["errors"][2][0] < 1e-20
assert scope["errors"][2][1] > scope["errors"][0][1] > scope["errors"][1][1]

# Six pairs of fitted models and independent future answers.
answers = np.array([1., 3.])
for predictions, bias_sq, variance, risk in [
    ([1., 2., 3.], 0, 2/3, 5/3),
    ([1.5]*3, .25, 0, 1.25),
    ([0.]*3, 4, 0, 5),
]:
    predictions = np.asarray(predictions)
    actual = ((answers[:, None] - predictions[None, :])**2).mean()
    np.testing.assert_allclose([
        (predictions.mean()-2)**2, predictions.var(), actual
    ], [bias_sq, variance, risk])
    assert np.isclose(actual, 1 + bias_sq + variance)

# Scalar L2, finite differences, scale and the data-loss normalization.
x, y = np.array([1., 2.]), np.array([1., 3.])
def data_loss(w):
    return .5 * np.mean((w*x-y)**2)
def objective(w, lam):
    return data_loss(w) + lam*w*w/2
for lam, expected in [(0., 1.4), (2.5, .7), (22.5, .14)]:
    w = 3.5/(2.5+lam)
    assert np.isclose(w, expected)
    derivative = (objective(w+1e-5, lam)-objective(w-1e-5, lam))/2e-5
    assert abs(derivative) < 1e-8
    assert objective(w, lam) < objective(w+.1, lam)
np.testing.assert_allclose([(3*1.4-4.5)**2, (3*.7-4.5)**2], [.09, 5.76])
assert np.isclose((1.4/10)**2, 1.4**2/100)

theta, gradient, eta, lam = 2., .3, .1, .2
assert np.isclose(theta-eta*(gradient+lam*theta), 1.93)
assert np.isclose((1-eta*lam)*theta-eta*gradient, 1.93)
w = theta
for _ in range(10):
    w *= 1-eta*lam
assert np.isclose(w, theta*(1-eta*lam)**10)

# Fixed-preconditioner example, not a complete Adam step.
theta = np.ones(2)
P = np.diag([1., .01])
np.testing.assert_allclose(theta-eta*P@(lam*theta), [.98, .9998])
np.testing.assert_allclose((1-eta*lam)*theta, [.98, .98])

# L1/elastic net, including zero and negative coefficients.
grid = np.linspace(-3, 3, 6001)
for a in [-2., -.2, 0., .2, 2.]:
    w = np.sign(a)*max(abs(a)-.5, 0)
    cost = lambda v: .5*(v-a)**2+.5*np.abs(v)
    assert cost(w) <= np.min(cost(grid))+1e-12
    w_elastic = w/1.5
    grad = w_elastic-a+.5*np.sign(w_elastic)+.5*w_elastic
    if w_elastic:
        assert abs(grad) < 1e-12
np.testing.assert_allclose([.2/1.5, 2/1.5], [.1333333333333, 1.3333333333333])
weights = np.array([3., 4.])
weights *= 2/np.linalg.norm(weights)
np.testing.assert_allclose(weights, [1.2, 1.6])
assert np.isclose(np.linalg.norm(weights), 2)

# Stopping time and the best checkpoint are different.
best_loss, bad, best_epoch, stop_epoch = float("inf"), 0, None, None
for epoch, value in [(0,1.30), (5,.70), (10,.65), (15,.72), (20,.90)]:
    if value < best_loss:
        best_epoch, best_loss, bad = epoch, value, 0
    else:
        bad += 1
    if bad == 2:
        stop_epoch = epoch
        break
assert (best_epoch, stop_epoch) == (10, 20)

# Published gradient-descent trajectory: no penalty was added to the loss.
path = np.asarray(scope["early_path"])
np.testing.assert_allclose(path, [
    [0, 0, 4.5, 2],
    [1, 1.5, 1.125, .125],
    [2, 2.25, .28125, .03125],
    [3, 2.625, .0703125, .1953125],
])
np.testing.assert_allclose(path[:, 1], 3*(1-.5**np.arange(4)))
assert np.all(np.diff(path[:, 2]) < 0)
assert path[3, 3] > path[2, 3]
assert scope["best_step"] == 2
assert .5*(3-2)**2 == .5  # validation loss at the converged training solution

# Exact inverted dropout; equality of means does not commute with squaring.
activation = np.array([2., 4.])
masks = np.array(list(product([0., 1.], repeat=2)))
outputs = masks*activation/.5
np.testing.assert_allclose(outputs.mean(0), activation)
np.testing.assert_allclose((outputs**2).mean(0), activation**2/.5)
assert not np.allclose((outputs**2).mean(0), outputs.mean(0)**2)

# Exact paired bootstrap, without a coverage guarantee for four observations.
d = scope["d"]
assert len(scope["bootstrap_means"]) == 4**4
assert d.mean() == .25
assert d[[0,1,1,3]].mean() == .75
assert d[[2,2,0,1]].mean() == -.25
assert (scope["low"], scope["high"]) == (-.5, 1.)
assert scope["low"] <= 0 <= scope["high"]

print("PASS: 3 published blocks, fits, risk decomposition, penalties, decay, checkpoint/gradient trajectory, dropout, paired bootstrap")
print("chapter sha256:", sha256(CHAPTER.read_bytes()).hexdigest())
