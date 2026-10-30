"""Check the published linear-model examples and their mathematical claims.

Requires NumPy, SciPy and scikit-learn. No downloads or external experiments.
"""
from pathlib import Path
import hashlib
import re
import numpy as np
from scipy.special import expit, softmax
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, LogisticRegression, Lasso
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/01 Классическое машинное обучение/02 Линейная регрессия и классификация.md"
chapter_text = CHAPTER.read_text()
blocks = re.findall(r"```python\n(.*?)```", chapter_text, re.S)
assert len(blocks) == 3
scope = {}
for i, code in enumerate(blocks):
    exec(compile(code, f"{CHAPTER}:python-{i+1}", "exec"), scope)
X, y, beta = scope["X"], scope["y"], scope["beta"]
np.testing.assert_allclose(X.T @ X, [[3, 9], [9, 29]])
np.testing.assert_allclose(X.T @ y, [10, 33])
np.testing.assert_allclose(scope["prediction"], [11/6, 10/3, 29/6])
np.testing.assert_allclose([sum(scope["residual"]**2), np.mean(scope["residual"]**2)],
                           [1/6, 1/18])
reg = LinearRegression().fit(scope["x"][:, None], y)
np.testing.assert_allclose([reg.intercept_, reg.coef_[0]], beta)
assert np.linalg.matrix_rank(X) == 2
np.testing.assert_allclose(np.linalg.cond(X.T @ X), np.linalg.cond(X)**2, rtol=1e-12)
assert np.linalg.eigvalsh(2*X.T@X).min() > 0
# Finite-difference check away from the solution.
theta = np.array([.2, -.3])
f = lambda v: np.sum((X@v-y)**2)
numeric = []
for j in range(2):
    delta = np.eye(2)[j]*1e-5
    numeric.append((f(theta+delta)-f(theta-delta))/(2e-5))
np.testing.assert_allclose(numeric, 2*X.T@(X@theta-y), atol=1e-8)
# The direction, not the gradient, increases a coefficient for positive x*r.
assert (X[:, 1] @ y) > 0
assert (-.01 * (2*X.T@(X@np.zeros(2)-y)))[1] > 0

# Rank deficiency changes identifiability, not the fitted projection.
u = np.array([1., 2., 3.])
A = np.column_stack([u, 2*u])
np.testing.assert_allclose(A @ [3., 0.], A @ [1., 1.])
assert np.linalg.matrix_rank(A) == 1
np.testing.assert_allclose(np.linalg.lstsq(A, 3*u, rcond=None)[0], [.6, 1.2])
np.testing.assert_allclose(scope["minimum_norm"], [.6, 1.2])
assert scope["matrix_rank"] == 1 and scope["residual_sums"].size == 0
np.testing.assert_allclose(np.sum(scope["minimum_norm"]**2), 1.8)
on_span = np.array([2., 4.])
off_span = np.array([1., 0.])
alternatives = np.array([[3., 0.], [1., 1.], [.6, 1.2]])
np.testing.assert_allclose(alternatives @ on_span, [6., 6., 6.])
np.testing.assert_allclose(alternatives @ off_span, [3., 1., .6])
# The empty residuals result is a rank/shape convention, not a zero-error flag.
noisy_target = 3*u + np.array([1., -1., 0.])
noisy_fit, empty_residuals, rank_deficient, _ = np.linalg.lstsq(
    A, noisy_target, rcond=None
)
assert empty_residuals.size == 0 and rank_deficient == 1
assert np.sum((noisy_target-A@noisy_fit)**2) > 1
# In the preceding pipeline, both complete one-hot columns sum to the intercept.
one_hot = np.array([[1.,0.], [0.,1.], [0.,1.], [1.,0.]])
with_intercept = np.column_stack([np.ones(4), one_hot])
assert np.linalg.matrix_rank(with_intercept) == 2
np.testing.assert_allclose(with_intercept @ [1., -1., -1.], np.zeros(4))

# Odds, thresholds and the published one-observation logistic update.
np.testing.assert_allclose(
    [scope["logit"], scope["probability"], scope["probability"]/(1-scope["probability"])],
    [np.log(2), 2/3, 2]
)
np.testing.assert_allclose(scope["gradient"], [-2/3, -1/3])
np.testing.assert_allclose(scope["logit_next"]-scope["logit"], .6)
np.testing.assert_allclose(expit(scope["logit_next"]), .7846794057566)
np.testing.assert_allclose([expit(2*np.log(2)), np.exp(2*np.log(2))], [.8, 4])
assert np.log(2) < np.log(.7/.3) < np.log(4)
np.testing.assert_allclose([-np.log(.8), -np.log(.2)], [.22314355131421, 1.6094379124341])
# Finite-difference check of binary loss wrt weights and intercept.
x = np.array([2., 1.])
t = np.array([np.log(2), -np.log(2), 0.])
def logistic_loss(v):
    s = x@v[:2]+v[2]
    return np.logaddexp(0, s)-s
for j in range(3):
    delta = np.eye(3)[j]*1e-5
    finite = (logistic_loss(t+delta)-logistic_loss(t-delta))/(2e-5)
    np.testing.assert_allclose(finite, [-2/3, -1/3, -1/3][j], atol=1e-9)
separable = [2*np.logaddexp(0, -w) for w in [0., 1., 5., 10.]]
assert all(a>b>0 for a,b in zip(separable, separable[1:]))
# Linear boundary in original coordinates, including the sign.
w, b = np.array([2., -3.]), .7
x0 = 1.2
x1 = -w[0]/w[1]*x0-b/w[1]
assert abs(w@[x0, x1]+b) < 1e-12

# Multiclass probability, gradient, shift invariance, stable large logits.
s = scope["scores"]
np.testing.assert_allclose(softmax(s), [4/7, 2/7, 1/7])
np.testing.assert_allclose(scope["class_loss"], np.log(7))
np.testing.assert_allclose(softmax(s+10000), softmax(s), atol=1e-12)
loss = lambda logits: np.logaddexp.reduce(logits)-logits[2]
for j in range(3):
    d = np.eye(3)[j]*1e-5
    np.testing.assert_allclose((loss(s+d)-loss(s-d))/(2e-5),
                               scope["gradient_scores"][j], atol=1e-10)

# XOR requires an interaction: additive scores have equal diagonal sums.
corners = np.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
labels = np.array([0,1,1,0])
np.testing.assert_array_equal((-np.prod(corners,axis=1)>0).astype(int), labels)
for seed in range(10):
    rng = np.random.default_rng(seed)
    h, g = rng.normal(size=2), rng.normal(size=2)
    scores = np.array([h[0]+g[0], h[0]+g[1], h[1]+g[0], h[1]+g[1]])
    np.testing.assert_allclose(scores[0]+scores[3], scores[1]+scores[2])
poly_model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LogisticRegression(C=10)
).fit(corners, labels)
np.testing.assert_array_equal(poly_model.predict(corners), labels)

# Centered ridge, intercept exclusion and scalar L1 threshold.
z = (scope["area"]-60)/20
for alpha, expected in [(0.,1.5), (2.,.75), (18.,.15)]:
    fit = Ridge(alpha=alpha).fit(z[:,None], y)
    np.testing.assert_allclose([fit.intercept_, fit.coef_[0]], [10/3, expected])
    D = np.diag([0.,1.])
    augmented = np.column_stack([np.ones(3), z])
    beta_r = np.linalg.solve(augmented.T@augmented+alpha*D, augmented.T@y)
    np.testing.assert_allclose(beta_r, [10/3, expected])
G = np.array([[1., .9], [.9, 1.]])
v = np.array([.9, 1.])
np.testing.assert_allclose(np.linalg.solve(G,v), [0,1], atol=1e-12)
shrunk = np.linalg.solve(G+np.eye(2), v)
np.testing.assert_allclose(shrunk, [.282131661442, .373040752351], atol=1e-12)
assert shrunk[0] > 0 and np.linalg.norm(shrunk) < 1
for gamma in [0.,2.,6.,10.]:
    optimum = max(1.5-gamma/4,0)
    grid = np.linspace(-3,3,12001)
    objective = lambda w: np.sum((y-(10/3+w*z))**2,axis=0)+gamma*abs(w)
    costs = np.array([objective(w) for w in grid])
    assert objective(optimum) <= costs.min()+1e-12
np.testing.assert_allclose(Lasso(alpha=1/3).fit(z[:,None], y).coef_, [1.])
assert (1.5/100)**2 == 1.5**2/10000

# An inner RidgeCV after scaling cannot refit that earlier scaler per fold.
class RecordingScaler(StandardScaler):
    means = []
    def fit(self, X, y=None, **kwargs):
        super().fit(X, y, **kwargs)
        type(self).means.append(float(self.mean_[0]))
        return self
data = np.array([0.,1.,2.,100.,101.,102.])[:,None]
targets = 2*data.ravel()+1
cv = KFold(2)
RecordingScaler.means.clear()
make_pipeline(
    RecordingScaler(), RidgeCV(alphas=[.1,1.],cv=cv)
).fit(data, targets)
assert RecordingScaler.means == [51.]
RecordingScaler.means.clear()
GridSearchCV(
    make_pipeline(RecordingScaler(),Ridge()),
    {"ridge__alpha":[.1,1.]}, cv=cv, scoring="neg_mean_squared_error"
).fit(data,targets)
np.testing.assert_allclose(RecordingScaler.means, [101.,1.,101.,1.,51.])
assert len(re.findall(r"!\[\[", chapter_text)) == 13
print("PASS: 3 published blocks, rank-deficient predictions and residuals, OLS, logistic/softmax gradients, XOR, penalties, inner-fold isolation, 13 figures")
print("chapter sha256:", hashlib.sha256(CHAPTER.read_bytes()).hexdigest())
