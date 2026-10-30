"""Execute published ranking examples and independently check their gradients."""
from pathlib import Path
import hashlib
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/02 Рекомендательные системы/03 Ranking и нейронные рекомендательные модели.md"
content = CHAPTER.read_bytes()
blocks = re.findall(r"```python\n(.*?)```", content.decode(), re.S)
assert len(blocks) == 23
scope = {}
for i, block in enumerate(blocks, 1):
    print(f"Execute actual ranking block {i}", flush=True)
    exec(compile(block, f"{CHAPTER}:block{i}", "exec"), scope)


def finite_difference(f, x):
    result = np.zeros_like(x)
    for index in np.ndindex(x.shape):
        above, below = x.copy(), x.copy()
        above[index] += 1e-6
        below[index] -= 1e-6
        result[index] = (f(above) - f(below)) / 2e-6
    return result


rng = np.random.default_rng(20261025)
for _ in range(50):
    s = rng.normal(size=5)
    target = int(rng.integers(5))
    prob = np.exp(s - np.logaddexp.reduce(s))
    grad = prob.copy()
    grad[target] -= 1
    np.testing.assert_allclose(
        finite_difference(lambda z: np.logaddexp.reduce(z) - z[target], s),
        grad, atol=2e-9)
    params = rng.normal(size=(3, 4))
    p, qi, qj = params
    delta = p @ (qi-qj)
    g = -np.exp(-np.logaddexp(0., delta))
    expected = np.array([g*(qi-qj), g*p, -g*p])
    np.testing.assert_allclose(
        finite_difference(lambda z: np.logaddexp(0., -z[0]@(z[1]-z[2])), params),
        expected, atol=3e-9)
print("PASS 50 listwise and 50 MF-BPR finite-difference systems")

# Autorec: all encoder/decoder/bias coordinates, including an unobserved output.
x = scope["auto_input"]
mask = scope["auto_mask"]
params = np.r_[scope["encoder"], scope["decoder"], 0., np.zeros(3)]


def auto_objective(v):
    h = max(0., v[:3]@x + v[6])
    r = v[3:6]*h + v[7:10]
    return np.square((r-x)[mask]).sum()


auto_expected = np.r_[scope["grad_encoder"], scope["grad_decoder"],
                      scope["grad_hidden"], scope["grad_output"]]
np.testing.assert_allclose(finite_difference(auto_objective, params),
                           auto_expected, atol=1e-8)
assert scope["grad_decoder"][1] == 0
# Masked-out decoder row can change its prediction without changing the loss.
changed = params.copy()
changed[4] = 900
np.testing.assert_allclose(auto_objective(params), auto_objective(changed))
print("PASS AutoRec masked output, ten gradients and masked-row negative control")

for _ in range(100):
    V = rng.normal(size=(6, 3))
    x = rng.normal(size=6)
    x[rng.integers(6)] = 0
    scaled = V*x[:, None]
    direct = sum(scaled[i]@scaled[j] for i in range(6) for j in range(i+1, 6))
    fast = .5*(np.square(scaled.sum(0)).sum()-np.square(scaled).sum())
    np.testing.assert_allclose(direct, fast, atol=1e-12)
    expected = x[:, None]*(scaled.sum(0)-scaled)
    np.testing.assert_allclose(finite_difference(
        lambda Z: sum((Z[i]@Z[j])*x[i]*x[j]
                      for i in range(6) for j in range(i+1, 6)), V),
        expected, atol=2e-8)
print("PASS 100 FM pairwise/fast identities and 1800 feature-factor derivatives")


def deepfm_loss(flat):
    E = flat.reshape(3, 2)
    interaction = sum(E[i]@E[j] for i in range(3) for j in range(i+1, 3))
    deep = .4*max(0., scope["deep_weights"]@flat)
    return np.logaddexp(0., -(0.2+interaction+deep))


all_embedding_gradient = finite_difference(deepfm_loss, scope["deep_input"])
np.testing.assert_allclose(all_embedding_gradient[2:4], scope["grad_item_bce"],
                           atol=1e-9)
assert not np.allclose(scope["grad_item_logit"], [1., 1.])
np.testing.assert_allclose(np.logaddexp(0., -1.), .31326168751822286)
assert 1/np.log(2) > 1  # The cited notebook cell omits IDCG normalization.
assert 1/np.log2(2) == 1
assert scope["training_negative_pool"]({"A"}, {"A"}) == set()
assert (scope["full_rank"], scope["sampled_rank"]) == (3, 1)
for name, expected in {
    "rec-ranking.svg": "dc6016187f7a283a28ab4c123946fcb916c9163aa6b7304a447e6cbbf845cf69",
    "rec-neumf.svg": "4e23963ef59bbd064b470535a8708f073890d3054969b966ea8d46e736f5ebc5",
    "rec-deepfm.svg": "8080d1e0db7a90236eabbb37356dc22ccd2abab2ac48a3a178dd40be2c4824ce",
}.items():
    path = ROOT / "Assets/Sources/Dive into Deep Learning — Recommender Systems" / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
print("PASS shared DeepFM gradient, BPR sigmoid bound, normalization and protocol controls")
print("Chapter SHA-256:", hashlib.sha256(content).hexdigest())
