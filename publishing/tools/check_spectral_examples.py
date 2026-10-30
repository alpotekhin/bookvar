"""Execute the four chapter fences, then check the graph-cut derivations independently."""

from pathlib import Path
from itertools import product
import hashlib
import re
import numpy as np
from scipy.linalg import eigh, expm
from scipy.sparse.csgraph import laplacian, connected_components
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/01 Классическое машинное обучение/06a Спектральная кластеризация и графовый Laplacian.md"
chapter_bytes = CHAPTER.read_bytes()
blocks = re.findall(r"```python\n(.*?)```", chapter_bytes.decode(), re.S)
assert len(blocks) == 4
scope = {}
for index, code in enumerate(blocks, 1):
    exec(compile(code, f"{CHAPTER}:block{index}", "exec"), scope)

W, D, L = (scope[name] for name in ["W", "D", "L"])
d = np.diag(D)
np.testing.assert_allclose(scope["trial"] @ D @ scope["trial"], 4.2)
np.testing.assert_allclose((scope["trial"] @ L @ scope["trial"])/4.2, 2/21)
np.testing.assert_allclose(scope["eigenvalues"], [0, 1.1-np.sqrt(1.01), 2, 1.1+np.sqrt(1.01)], atol=1e-14)
expected_v = np.array([1, np.sqrt(1.01)-.1, .1-np.sqrt(1.01), -1])
expected_v /= np.linalg.norm(expected_v)
np.testing.assert_allclose(scope["v2"], expected_v, atol=1e-14)
np.testing.assert_allclose(scope["normalized_values"], [0, 1/11, 21/11, 2], atol=1e-14)
np.testing.assert_allclose(laplacian(W), L)
np.testing.assert_allclose(laplacian(W, normed=True), scope["Lsym"])

# Direct edge sums, not matrix identities used as their own expected values.
rng = np.random.default_rng(20261025)
cases = 0
for n in range(2, 9):
    for _ in range(12):
        upper = np.triu(rng.uniform(.01, 2, size=(n, n)), 1)
        graph = upper + upper.T
        degrees = graph.sum(axis=1)
        degree_matrix = np.diag(degrees)
        lap = degree_matrix - graph
        values = np.linalg.eigvalsh(lap)
        normalized_values = eigh(lap, degree_matrix, eigvals_only=True)
        for bits in product([False, True], repeat=n-1):
            chosen = np.array([True, *bits])
            if chosen.all():
                continue
            other = ~chosen
            cut = sum(graph[i, j] for i in range(n) for j in range(n)
                      if chosen[i] and other[j])
            a, b = chosen.sum(), other.sum()
            va, vb = degrees[chosen].sum(), degrees[other].sum()
            ratio = cut * (1/a + 1/b)
            ncut = cut * (1/va + 1/vb)
            f = np.where(chosen, np.sqrt(b/a), -np.sqrt(a/b))
            g = np.where(chosen, np.sqrt(vb/va), -np.sqrt(va/vb))
            np.testing.assert_allclose(f.sum(), 0, atol=1e-12)
            np.testing.assert_allclose(f @ f, n)
            np.testing.assert_allclose(f @ lap @ f, n * ratio)
            np.testing.assert_allclose(g @ degrees, 0, atol=1e-12)
            np.testing.assert_allclose(g @ degree_matrix @ g, degrees.sum())
            np.testing.assert_allclose(g @ lap @ g, degrees.sum() * ncut)
            assert values[1] <= ratio + 1e-12
            assert normalized_values[1] <= ncut + 1e-12
            H = np.column_stack([chosen/np.sqrt(a), other/np.sqrt(b)])
            Hn = np.column_stack([chosen/np.sqrt(va), other/np.sqrt(vb)])
            np.testing.assert_allclose(H.T @ H, np.eye(2), atol=1e-12)
            np.testing.assert_allclose(np.trace(H.T @ lap @ H), ratio)
            np.testing.assert_allclose(Hn.T @ degree_matrix @ Hn, np.eye(2), atol=1e-12)
            np.testing.assert_allclose(np.trace(Hn.T @ lap @ Hn), ncut)
            cases += 1
        f = rng.normal(size=n)
        energy = sum(graph[i,j]*(f[i]-f[j])**2 for i in range(n) for j in range(i+1,n))
        np.testing.assert_allclose(f @ lap @ f, energy)
print(f"PASS {cases} discrete partitions: RatioCut/Ncut identities, constraints and relaxed lower bounds")

# Same four-node state: stationary-conditioned exit versus uniform initialization.
P = W / d[:, None]
pi = d / d.sum()
np.testing.assert_allclose(P.sum(axis=1), 1)
np.testing.assert_allclose(pi @ P, pi)
np.testing.assert_allclose(pi[:, None] * P, pi[None, :] * P.T)
A = np.array([True, True, False, False])
exit_probability = (pi[A] / pi[A].sum()) @ P[np.ix_(A, ~A)].sum(axis=1)
uniform_exit = P[np.ix_(A, ~A)].sum(axis=1).mean()
np.testing.assert_allclose([exit_probability, uniform_exit], [1/21, 1/22])
assert not np.isclose(exit_probability, uniform_exit)
normalized_v = scope["F"][:, 1]
np.testing.assert_allclose(normalized_v @ d, 0, atol=1e-14)
assert not np.allclose(scope["Lsym"] @ scope["v2"], scope["normalized_values"][1] * scope["v2"])
np.testing.assert_allclose(L, np.diag((W+np.eye(4)*7).sum(axis=1))-(W+np.eye(4)*7))
# SciPy ignores self-loops, whereas including them in a manual D changes Lsym.
np.testing.assert_allclose(laplacian(W+7*np.eye(4), normed=True), scope["Lsym"])
loop_d = d+7
assert not np.allclose(L / np.sqrt(loop_d[:,None]*loop_d[None,:]), scope["Lsym"])
actual = SpectralClustering(2, affinity="precomputed", n_init=20, random_state=0).fit_predict(W)
assert adjusted_rand_score(actual, A.astype(int)) == 1
print("PASS generalized eigenvectors, stationary exits, diagonal convention and actual four-node API")

# Any orthogonal basis of the complete nullspace has the same normalized geometry.
U0, T0 = scope["U0"], scope["T0"]
np.testing.assert_allclose(np.linalg.norm(U0,axis=1), [.5, 1/np.sqrt(2), .5, 1/np.sqrt(2), 1/np.sqrt(2)])
for _ in range(100):
    Q, _ = np.linalg.qr(rng.normal(size=(2,2)))
    rotated = U0 @ Q
    rotated /= np.linalg.norm(rotated, axis=1, keepdims=True)
    np.testing.assert_allclose(rotated @ rotated.T, T0 @ T0.T, atol=1e-13)
np.testing.assert_allclose(scope["values0"][:2], 0, atol=1e-14)
assert connected_components(scope["W0"],directed=False)[0] == 2
assert np.count_nonzero(np.isclose(np.linalg.eigvalsh(scope["L0"]),0,atol=1e-12)) == 2
isolated = np.pad(scope["W0"], ((0,1),(0,1)))
assert isolated.sum(axis=1)[-1] == 0
assert connected_components(isolated,directed=False)[0] == 3
# A permitted basis with one direction omitted has zero rows; do not assume
# that the numerical solver always chooses this particular basis.
indicator_basis = np.array([[.5,0], [1/np.sqrt(2),0], [.5,0], [0,1/np.sqrt(2)], [0,1/np.sqrt(2)]])
assert np.any(np.linalg.norm(indicator_basis[:,:1],axis=1) == 0)

# Heat flow on the five-node graph converges to separate component means.
h0 = np.array([0., 2, 7, -3, 5])
heat = expm(-30*scope["L0"]) @ h0
np.testing.assert_allclose(heat, [3,3,3,1,1], atol=1e-11)
np.testing.assert_allclose(heat.sum(), h0.sum())
print("PASS 100 nullspace rotations, disconnected/isolated vertices and heat-flow limit")

# One fixed dataset only; no quality or timing benchmark is claimed.
true_labels = scope["true_labels"]
assert round(adjusted_rand_score(true_labels, scope["ordinary_labels"]),6) == .256364
assert round(adjusted_rand_score(true_labels, scope["labels"]),6) == .516047
assert scope["components"] == 1
assert scope["results"][15] == (1, 1.0)
assert scope["results"][30][0] == 1
directed = scope["directed"]
np.testing.assert_allclose(directed.maximum(directed.T).data, 1)
assert np.any(((directed+directed.T)*.5).data == .5)
native = SpectralClustering(2, affinity="nearest_neighbors", n_neighbors=30,
    n_init=20, random_state=0).fit(scope["X"])
self_graph = scope["kneighbors_graph"](scope["X"], n_neighbors=30,
    mode="connectivity", include_self=True)
np.testing.assert_allclose(native.affinity_matrix_.toarray(), ((self_graph+self_graph.T)*.5).toarray())
np.testing.assert_allclose(native.affinity_matrix_.diagonal(), 1)
np.testing.assert_allclose(directed.diagonal(), 0)
print("PASS all four chapter fences; moons graph parameters affect the partition")
print("chapter_sha256", hashlib.sha256(chapter_bytes).hexdigest())
