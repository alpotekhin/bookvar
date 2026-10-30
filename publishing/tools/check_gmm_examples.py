"""Execute all GMM chapter fences and check the derivations independently."""

from pathlib import Path
import hashlib
import re
import numpy as np
from scipy.special import logsumexp, xlogy
from scipy.stats import multivariate_normal, norm
from sklearn.mixture._gaussian_mixture import _estimate_gaussian_parameters

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/01 Классическое машинное обучение/08 Gaussian mixture и EM.md"
contents = CHAPTER.read_bytes()
blocks = re.findall(r"```python\n(.*?)```", contents.decode(), re.S)
assert len(blocks) == 8
scope = {}
for index, code in enumerate(blocks, 1):
    print(f"Execute actual chapter block {index}", flush=True)
    exec(compile(code, f"{CHAPTER}:block{index}", "exec"), scope)

rng = np.random.default_rng(20261025)
logpdf = scope["gaussian_logpdf"]
for d in [1, 2, 5]:
    for _ in range(8):
        B = rng.normal(size=(d,d))
        covariance = B@B.T + np.eye(d)*.1
        mean = rng.normal(size=d)
        X = rng.normal(size=(9,d))
        actual = logpdf(X, mean, covariance)
        np.testing.assert_allclose(actual,
            multivariate_normal.logpdf(X,mean=mean,cov=covariance), rtol=1e-10)
        A = rng.normal(size=(d,d)) + np.eye(d)*3
        b = rng.normal(size=d)
        transformed = logpdf(X@A.T+b, A@mean+b, A@covariance@A.T)
        np.testing.assert_allclose(transformed, actual-np.linalg.slogdet(A)[1], atol=1e-9)
try:
    logpdf([[0,0]], np.zeros(2), np.array([[1,2],[2,1]]))
except np.linalg.LinAlgError:
    pass
else:
    raise AssertionError("Indefinite covariance must not be accepted")
print("PASS 24 Cholesky/SciPy and affine-density cases, indefinite-covariance control")

expectation = scope["expectation_1d"]
np.testing.assert_allclose(scope["posterior"][0], [1/(1+np.exp(-4)),1/(1+np.exp(4))])
np.testing.assert_allclose(scope["point_scores"][1], -.5*np.log(2*np.pi)-2)
assert scope["posterior"][-1,1] == 1
assert scope["point_scores"][-1] < -4000
assert scope["entropy"][-1] < 1e-160
boundary = 2+np.log(9)/4
unequal = expectation([2.,boundary], np.array([.9,.1]), np.array([0.,4.]), np.ones(2))[0]
np.testing.assert_allclose(unequal, [[.9,.1],[.5,.5]])
x = scope["x_train"]
weights, means, variances = (scope[name] for name in ["weights_new","means_new","variances_new"])
gamma = scope["gamma_old"]
reference = .5*norm.pdf(x,0,1) + .5*norm.pdf(x,4,1)
np.testing.assert_allclose(scope["logp_old"], np.log(reference))
np.testing.assert_allclose(scope["counts"], [1.982019934212, 2.017980065788], atol=1e-11)
np.testing.assert_allclose(scope["logp_fixed"].sum(), -6.9442527585648985)
np.testing.assert_allclose(scope["logp_full"].sum(), -5.737298069233694)
assert not np.allclose(variances,
    (gamma*(x[:,None]-scope["means_old"])**2).sum(axis=0)/scope["counts"])
assert not np.allclose(variances, variances*scope["counts"]/(scope["counts"]-1))
for covariance_type in ["full","diag","tied","spherical"]:
    nk, mu, cov = _estimate_gaussian_parameters(x[:,None],gamma,0.,covariance_type)
    np.testing.assert_allclose(nk, scope["counts"], atol=1e-13)
    np.testing.assert_allclose(mu.ravel(),means,atol=1e-13)
    if covariance_type == "tied":
        np.testing.assert_allclose(cov, [[np.dot(scope["counts"],variances)/len(x)]])
    else:
        np.testing.assert_allclose(cov.ravel(), variances)
print("PASS posterior/prior shift, density versus entropy, weighted M-step and four covariance APIs")

# Q as a function of unconstrained mean, log-variance and logits.
def q_value(parameters):
    mu = parameters[:2]
    var = np.exp(parameters[2:4])
    logits = parameters[4:]
    w = np.exp(logits-logsumexp(logits))
    return np.sum(gamma*expectation(x,w,mu,var)[2])
optimum = np.r_[means,np.log(variances),np.log(weights)]
for j in range(len(optimum)):
    plus, minus = optimum.copy(), optimum.copy()
    plus[j] += 1e-5
    minus[j] -= 1e-5
    np.testing.assert_allclose((q_value(plus)-q_value(minus))/2e-5,0,atol=2e-9)
for _ in range(100):
    assert q_value(optimum+rng.normal(scale=.2,size=6)) <= q_value(optimum)+1e-12

bound = scope["bound_value"]
for _ in range(60):
    points = rng.normal(size=12)
    w = rng.dirichlet(np.ones(3))
    mu = rng.normal(size=3)
    var = rng.uniform(.3,3,size=3)
    posterior, logp, joint = expectation(points,w,mu,var)
    q = rng.dirichlet(np.ones(3),size=len(points))
    kl = np.sum(xlogy(q,q)-xlogy(q,posterior))
    np.testing.assert_allclose(logp.sum()-bound(q,joint),kl,atol=1e-12)
    np.testing.assert_allclose(logp.sum(),bound(posterior,joint),atol=1e-12)
    assert kl >= -1e-12
np.testing.assert_allclose(scope["gap"], .21614168670497874)
assert np.all(np.diff(scope["likelihood_trace"]) >= -1e-10)
assert scope["converged"]
np.testing.assert_allclose(scope["fit_means"], [.5,4.5],atol=1e-10)
np.testing.assert_allclose(scope["fit_variances"], [.25,.25],atol=1e-10)
for points, initial_means, message in [
    (np.zeros(3), np.zeros(1), "Singular Gaussian component"),
    (x, np.array([0., 1e6]), "Numerically empty component"),
]:
    K = len(initial_means)
    try:
        scope["fit_em_1d"](points, np.full(K, 1/K), initial_means, np.ones(K))
    except ValueError as error:
        assert str(error) == message
    else:
        raise AssertionError(f"Expected {message}")
print("PASS six stationary derivatives, 100 Q perturbations, 60 ELBO/KL identities and EM trace")
print("PASS singular and numerically empty component failure paths")

# Library reg_covar really performs the diagonal addition in the counterexample.
_, _, regularized_cov = _estimate_gaussian_parameters(
    scope["regularization_data"][:,None], np.ones((2,1)), 1., "full")
np.testing.assert_allclose(regularized_cov, [[[2.]]])
assert scope["regularized_score"] < scope["exact_score"]
assert np.all(np.diff(scope["collapse_scores"]) > 0)
permuted, permuted_logp, _ = expectation(x,weights[::-1],means[::-1],variances[::-1])
np.testing.assert_allclose(permuted_logp,scope["logp_full"])
np.testing.assert_allclose(permuted[:,::-1],scope["gamma_new"])

expected_counts = {"spherical":7,"diag":9,"tied":8,"full":11}
for kind, count in expected_counts.items():
    assert scope["parameter_count"](2,2,kind) == count
assert scope["parameter_count"](3,20,"full")-scope["parameter_count"](2,20,"full") == 231
assert len(scope["candidates"]) == 16
assert all(m.converged_ for m in scope["candidates"])
model = scope["best_validation"]
assert (model.n_components,model.covariance_type) == (2,"full")
np.testing.assert_array_equal(model.predict(scope["X_test"]),scope["test_posterior"].argmax(axis=1))
assert len(scope["X_fit"]) == 480 and len(scope["X_validation"]) == len(scope["X_test"]) == 160
for variance in [1.,.25,.01]:
    post = expectation(np.array([1.,2.,3.]),np.array([.5,.5]),np.array([0.,4.]),np.full(2,variance))[0]
    np.testing.assert_allclose(post[1],[.5,.5])
    assert post[0,0] > .98 and post[-1,0] < .02
print("PASS actual covariance regularization, label permutation, 16 model candidates, AIC/BIC and hard-limit ties")
print("chapter_sha256", hashlib.sha256(contents).hexdigest())
