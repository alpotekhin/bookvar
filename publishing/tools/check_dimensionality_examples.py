"""Execute every chapter fence and independently check its mathematical examples.

Small CPU correctness cases only. This is not a quality or performance benchmark.
"""

from pathlib import Path
import hashlib
import re
import numpy as np
from scipy.optimize import brentq
from scipy.spatial.distance import cdist
from sklearn.manifold import trustworthiness
from umap.umap_ import smooth_knn_dist, find_ab_params

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/01 Классическое машинное обучение/07 PCA, t-SNE и UMAP.md"
chapter_bytes = CHAPTER.read_bytes()
blocks = re.findall(r"```python\n(.*?)```", chapter_bytes.decode(), re.S)
assert len(blocks) == 7
scope = {}
for index, code in enumerate(blocks, 1):
    print(f"Execute actual chapter block {index}", flush=True)
    exec(compile(code, f"{CHAPTER}:block{index}", "exec"), scope)

np.testing.assert_allclose(scope["Z"].ravel(), [-2, 0, 0, 2])
np.testing.assert_allclose(scope["error"], 2)
np.testing.assert_allclose(scope["pca"].explained_variance_ratio_, [.8])
np.testing.assert_allclose(scope["pca"].explained_variance_, [8/3])
np.testing.assert_allclose(np.cov(scope["white_codes"], rowvar=False), np.eye(2))
rng = np.random.default_rng(20261025)
for n in [7, 13, 31]:
    for d in [2, 3, 5]:
        X = rng.normal(size=(n, d)) @ np.diag(np.arange(1, d+1)) + rng.normal(size=d)
        Xc = X - X.mean(axis=0)
        _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        eig, eigenvectors = np.linalg.eigh(np.cov(X, rowvar=False))
        np.testing.assert_allclose(eig[::-1], s*s/(n-1), atol=1e-12)
        for m in range(1, d+1):
            W = Vt[:m].T
            error = np.sum((Xc-Xc@W@W.T)**2)
            np.testing.assert_allclose(error, np.sum(s[m:]**2), atol=1e-12)
            np.testing.assert_allclose(W@W.T, eigenvectors[:,-m:]@eigenvectors[:,-m:].T, atol=1e-12)
print("PASS SVD/reconstruction/variance and whitening, 9 independent covariance cases")

P, conditional = scope["P"], scope["conditional"]
np.testing.assert_allclose(P, P.T)
np.testing.assert_allclose(np.diag(P), 0)
np.testing.assert_allclose(P.sum(), 1)
for row in conditional:
    p = row[row > 0]
    np.testing.assert_allclose(np.exp(-np.sum(p*np.log(p))), 2, atol=1e-10)
gaussian_row = scope["gaussian_row"]
for distances, target in [([0, 0, 1], 1.5), ([1, 2, 3], 3), ([1, 2, 3], 1)]:
    try:
        gaussian_row(distances, target)
    except ValueError:
        pass
    else:
        raise AssertionError("Unattainable/boundary perplexity not rejected")
for scale in [.01, 1, 100]:
    reference, sigma = gaussian_row([1, 4, 13], 2)
    changed, scaled_sigma = gaussian_row(np.array([1,4,13])*scale**2, 2)
    np.testing.assert_allclose(changed, reference, atol=1e-10)
    np.testing.assert_allclose(scaled_sigma, sigma*scale, rtol=1e-10)

objective, Y = scope["student_objective"], scope["Y"]
loss, gradient, Q = objective(Y, P)
numerical = np.zeros_like(Y)
for index in np.ndindex(Y.shape):
    plus, minus = Y.copy(), Y.copy()
    plus[index] += 1e-6
    minus[index] -= 1e-6
    numerical[index] = (objective(plus, P)[0]-objective(minus, P)[0])/2e-6
np.testing.assert_allclose(gradient, numerical, atol=1e-9)
rotation = np.array([[0., -1], [1, 0]])
np.testing.assert_allclose(objective(Y@rotation+7, P)[0], loss)
assert not np.isclose(objective(Y*10, P)[0], loss)
np.testing.assert_allclose(Q.sum(), 1)
assert scope["loss_after"] < scope["loss_before"]
delta = Y[:,None]-Y[None,:]
inverse = 1/(1+np.sum(delta**2,axis=2))
np.fill_diagonal(inverse, 0)
exaggerated = 4*np.sum(((4*P-Q)*inverse)[:,:,None]*delta,axis=1)
# Differentiate the literal KL(4P||Q) independently.
def unnormalized_kl(y):
    q = objective(y, P)[2]
    present = P>0
    return np.sum(4*P[present]*np.log(4*P[present]/q[present]))
literal = np.zeros_like(Y)
for index in np.ndindex(Y.shape):
    plus, minus = Y.copy(), Y.copy()
    plus[index] += 1e-6
    minus[index] -= 1e-6
    literal[index] = (unnormalized_kl(plus)-unnormalized_kl(minus))/2e-6
np.testing.assert_allclose(literal, 4*gradient, atol=1e-9)
assert not np.allclose(exaggerated, literal)
print("PASS t-SNE entropy, ties, scale, pair normalization, finite differences and early exaggeration")

np.testing.assert_allclose(scope["sigma"], 2.616122579581014)
np.testing.assert_allclose(scope["directed_weights"], [1, .6823278038280193, .3176721961719807])
for count in [4, 6, 10]:
    for _ in range(10):
        distances = np.r_[0, np.cumsum(rng.uniform(.2, 3, count-1))]
        rho = distances[1]
        root = brentq(lambda s: np.exp(-(distances[1:]-rho)/s).sum()-np.log2(count), 1e-8, 1000)
        actual_sigma, actual_rho = smooth_knn_dist(distances[None].astype(np.float32), float(count))
        np.testing.assert_allclose(actual_rho, [rho], rtol=1e-6)
        np.testing.assert_allclose(actual_sigma, [root], rtol=3e-5, atol=3e-5)
for weight in [.2, .5, .8]:
    optimum = np.sqrt((1-weight)/weight)
    np.testing.assert_allclose(scope["pair_cross_entropy"](optimum, weight)[2], 0, atol=1e-14)
    for distance in [.25, .5, 2, 5]:
        q, value, derivative = scope["pair_cross_entropy"](distance, weight)
        numeric = (scope["pair_cross_entropy"](distance+1e-6, weight)[1]
                   -scope["pair_cross_entropy"](distance-1e-6, weight)[1])/2e-6
        np.testing.assert_allclose(numeric, derivative, atol=1e-9)
        assert 0 < q < 1 and value > 0
np.testing.assert_allclose(.8+.3-.8*.3, .86)
np.testing.assert_allclose(1.2*2/(2*5), .24)
for min_dist in [0, .1, .5]:
    a, b = find_ab_params(1, min_dist)
    assert a > 0 and b > 0
    assert 0 < 1/(1+a*.01**(2*b)) < 1  # No hard minimum-distance constraint.
print("PASS 30 reference UMAP scale calibrations, union and 12 pairwise finite differences")

# Independent rank calculation, excluding the point itself in both spaces.
original = rng.normal(size=(12,5))
mapped = rng.normal(size=(12,2))
order = np.argsort(cdist(original, original), axis=1)
new_order = np.argsort(cdist(mapped, mapped), axis=1)
for k in [1, 2, 4]:
    penalty = 0
    for i in range(12):
        ranks = {j:rank for rank,j in enumerate(order[i])}
        for j in new_order[i,1:k+1]:
            penalty += max(0, ranks[j]-k)
    manual = 1-2*penalty/(12*k*(24-3*k-1))
    np.testing.assert_allclose(manual, trustworthiness(original,mapped,n_neighbors=k))
np.testing.assert_allclose(scope["preprocessor"].mean_, scope["train_digits"].mean(axis=0))
assert not np.allclose(scope["preprocessor"].mean_, scope["digits"].mean(axis=0))
assert scope["tsne_coordinates"].shape == scope["umap_coordinates"].shape == (140,2)
assert not hasattr(scope["tsne"], "transform")
np.testing.assert_array_equal(scope["mapper"].embedding_, scope["training_snapshot"])
assert np.isfinite(scope["new_coordinates"]).all()
print("PASS trustworthiness ranks, train-only PCA, real TSNE/UMAP and fixed training embedding")
print("chapter_sha256", hashlib.sha256(chapter_bytes).hexdigest())
