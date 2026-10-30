"""Execute the kernel chapter and independently verify its mathematical examples."""

from pathlib import Path
import hashlib
import re
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.svm import SVC
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/01 Классическое машинное обучение/09 Kernels, SVM и random Fourier features.md"
content = CHAPTER.read_bytes()
blocks = re.findall(r"```python\n(.*?)```", content.decode(), re.S)
assert len(blocks) == 9
scope = {}
for index, code in enumerate(blocks, 1):
    print(f"Execute actual chapter block {index}", flush=True)
    exec(compile(code, f"{CHAPTER}:block{index}", "exec"), scope)

rng = np.random.default_rng(20261025)
for _ in range(25):
    X = rng.normal(size=(7, 2))
    Z = scope["quadratic_features"](X)
    np.testing.assert_allclose(Z @ Z.T, (X @ X.T)**2, atol=1e-12)
    assert np.linalg.eigvalsh(Z @ Z.T).min() > -1e-10
    w = rng.normal(size=2)
    b = rng.normal()
    f = X @ w + b
    projected = X - f[:, None] * w / (w @ w)
    np.testing.assert_allclose(projected @ w + b, 0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(X-projected,axis=1), np.abs(f)/np.linalg.norm(w))
    scale = rng.uniform(.1, 9)
    np.testing.assert_allclose((X@(scale*w)+scale*b)/np.linalg.norm(scale*w), f/np.linalg.norm(w))
print("PASS 25 feature-map, PSD, projection and parameter-scale cases")

X = scope["X_line"]
y = scope["y_line"]
for C, model in zip([.02,.1,10.],scope["soft_models"]):
    # Symmetry and convexity ensure an optimum exists with b=0.
    objective = lambda w: .5*w*w + C*np.maximum(0,1-y*X.ravel()*w).sum()
    direct = minimize_scalar(objective,bounds=(0,2),method="bounded",options={"xatol":1e-12})
    learned_w = model.coef_[0,0]
    np.testing.assert_allclose(objective(learned_w), direct.fun,atol=2e-8)
    alpha = np.zeros(4)
    alpha[model.support_] = np.abs(model.dual_coef_[0])
    margin = y*model.decision_function(X)
    assert np.all(alpha >= -1e-9) and np.all(alpha <= C+1e-9)
    assert np.all(margin[alpha < C-1e-7] >= 1-1e-7)
    assert np.all(margin[alpha > 1e-7] <= 1+1e-7)
    for j in range(4):
        if margin[j] < 1-1e-6:
            np.testing.assert_allclose(alpha[j],C,atol=1e-8)
    np.testing.assert_allclose(np.diag(y)@(X@X.T)@np.diag(y),
                               (y[:,None]*X)@(y[:,None]*X).T)
for b in [-.76,-.2,0,.5,.76]:
    margin = y*(X.ravel()*.12+b)
    slack = np.maximum(0,1-margin)
    np.testing.assert_allclose(.5*.12**2+.02*slack.sum(),.0728,atol=1e-12)
    assert np.all(margin <= 1+1e-12)
assert (.5*.12**2+.02*np.maximum(0,1-y*(X.ravel()*.12+.8)).sum()) > .0728
print("PASS independent soft-SVM objectives, KKT cases, primal/dual values and bias interval")

model = scope["kernel_model"]
queries = rng.normal(size=(31,2))
cross_kernel = rbf_kernel(queries,scope["X_ring"],gamma=scope["kernel_gamma"])
manual = cross_kernel[:,model.support_] @ model.dual_coef_[0] + model.intercept_[0]
np.testing.assert_allclose(manual,model.decision_function(queries),atol=1e-10)
np.testing.assert_allclose(manual,scope["precomputed"].decision_function(cross_kernel),atol=1e-7)
try:
    scope["precomputed"].predict(rbf_kernel(queries,gamma=scope["kernel_gamma"]))
except ValueError:
    pass
else:
    raise AssertionError("A test-by-test Gram matrix is not a test-by-train matrix")
print("PASS manual binary-kernel predictions and wrong precomputed-shape negative control")

paired = scope["paired_fourier"]
for d in [1,2,5]:
    data = rng.normal(size=(11,d))
    omega = rng.normal(size=(d,24))
    features = paired(data,omega)
    differences = data[:,None,:]-data[None,:,:]
    manual = np.cos(differences@omega).mean(axis=2)
    np.testing.assert_allclose(features@features.T,manual,atol=1e-14)
    np.testing.assert_allclose(np.diag(features@features.T),1,atol=1e-14)
    assert np.linalg.eigvalsh(features@features.T).min() >= -1e-12
negative_pair = paired(np.array([[0.],[1.]]),np.array([[np.pi]]))
np.testing.assert_allclose(negative_pair@negative_pair.T,[[1,-1],[-1,1]])
for D, mean, empirical, predicted in scope["rff_variance_results"]:
    assert abs(mean-scope["target_kernel"]) < 5*np.sqrt(predicted/4000)
    assert abs(empirical/predicted-1) < .1
# A nested draw need not improve every prefix: [0] is exact for a=1 at delta=0,
# so use a nonzero distance and a deliberately unrepresentative extra frequency.
target=np.exp(-.5)
first=np.arccos(target)
assert abs(np.cos(first)-target) < 1e-15
assert abs((np.cos(first)+np.cos(np.pi))/2-target) > .5
print("PASS paired/phase identities, PSD with a negative entry, variance and nonmonotonic-error control")

for d in [2,4]:
    data = rng.normal(size=(15,d))
    numeric = RBFSampler(gamma=.7,n_components=31,random_state=8).fit(data)
    changed = RBFSampler(gamma=.7,n_components=31,random_state=8).fit(data*7+3)
    np.testing.assert_array_equal(numeric.random_weights_,changed.random_weights_)
    np.testing.assert_array_equal(numeric.random_offset_,changed.random_offset_)
    np.testing.assert_allclose(numeric.transform(data),np.sqrt(2/31)*np.cos(
        data@numeric.random_weights_+numeric.random_offset_),atol=1e-14)
    scaled=RBFSampler(gamma="scale",n_components=31,random_state=8).fit(data)
    np.testing.assert_allclose(scaled._gamma,1/(d*data.var()))
print("PASS real RBFSampler source contract, gamma scale and data-independent numeric gamma")

nystrom=scope["nystrom_from_anchors"]
for d in [1,3]:
    for _ in range(10):
        data=rng.normal(size=(8,d))
        kernel=lambda A,B: rbf_kernel(A,B,gamma=.7)
        exact=kernel(data,data)
        z=nystrom(data,data[[0,3,6]],kernel)
        approximation=z@z.T
        assert np.linalg.eigvalsh(exact-approximation).min() > -1e-9
        np.testing.assert_allclose(approximation[[0,3,6]],exact[[0,3,6]],atol=1e-8)
        full=nystrom(data,data,kernel)
        np.testing.assert_allclose(full@full.T,exact,atol=1e-8)
duplicate=nystrom(scope["kernel_points"],np.array([[-1.],[-1.]]),scope["example_kernel"])
assert np.isfinite(duplicate).all()
assert np.linalg.matrix_rank(duplicate) == 1
raw=scope["K_poly"]
assert not np.allclose(raw@raw.T,raw)
try:
    nystrom(np.ones((2,1)),np.ones((2,1)),lambda A,B: -np.ones((len(A),len(B))))
except ValueError:
    pass
else:
    raise AssertionError("An indefinite anchor Gram matrix must be rejected")
print("PASS 20 Nystrom projection cases, rank/full-anchor recovery, duplicates and invalid kernel")
for _ in range(15):
    features=rng.normal(size=(9,5))
    anchors=features[[0,4,6]]
    W=anchors@anchors.T
    B=features@anchors.T
    P=anchors.T@np.linalg.pinv(W)@anchors
    np.testing.assert_allclose(P,P.T,atol=1e-12)
    np.testing.assert_allclose(P@P,P,atol=1e-12)
    np.testing.assert_allclose(features@P@features.T,B@np.linalg.pinv(W)@B.T,atol=1e-11)
    assert np.linalg.eigvalsh(features@(np.eye(5)-P)@features.T).min() > -1e-11
print("PASS 15 explicit feature-space projectors, idempotence and discarded Gram directions")

assert len(scope["candidate_rows"]) == 33
assert len(scope["X_train"]) == 360
assert len(scope["X_validation"]) == len(scope["X_test"]) == 120
for row in scope["candidate_rows"]:
    pipeline=row[-1]
    np.testing.assert_allclose(pipeline.named_steps["scale"].mean_,scope["X_train"].mean(axis=0))
    assert pipeline.named_steps["svc"].fit_status_ == 0
    if row[1] == "nystrom":
        source=pipeline.named_steps["scale"].transform(scope["X_train"])
        feature_map=pipeline.named_steps["map"]
        np.testing.assert_array_equal(feature_map.components_,source[feature_map.component_indices_])
assert scope["selected_test_accuracy"] > .8
print("PASS 33 actual pipelines, train-only fitted statistics and anchors, selected test contract")
print("chapter_sha256",hashlib.sha256(content).hexdigest())
