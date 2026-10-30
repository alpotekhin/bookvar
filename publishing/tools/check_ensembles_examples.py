"""Check the ensemble chapter's published code and mechanism examples."""
from pathlib import Path
from itertools import product
import hashlib
import re
import numpy as np
from scipy.special import expit, softmax, logsumexp
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    BaggingRegressor, RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/01 Классическое машинное обучение/04 Bagging, random forest и gradient boosting.md"
blocks = re.findall(r"```python\n(.*?)```", CHAPTER.read_text(), re.S)
assert len(blocks) == 3
scope = {}
for i, code in enumerate(blocks):
    exec(compile(code, f"{CHAPTER}:python-{i+1}", "exec"), scope)
assert scope["oob_counts"].min() > 0
np.testing.assert_allclose(scope["manual_oob"], scope["bag"].oob_prediction_)
# Each tree must contribute only to points absent from that tree's draw.
for i in range(len(scope["y"])):
    eligible = [j for j, ix in enumerate(scope["bag"].estimators_samples_) if i not in ix]
    assert len(eligible) == scope["oob_counts"][i]
    answer = np.mean([scope["bag"].estimators_[j].predict(scope["X"][i:i+1])[0]
                      for j in eligible])
    np.testing.assert_allclose(answer, scope["manual_oob"][i])
# Exact expectation over all 5^5 bootstrap draws, rather than a Monte Carlo claim.
draws = list(product(range(5), repeat=5))
assert [0,0,2,4,4].count(0) == 2
np.testing.assert_allclose(np.mean([0 not in draw for draw in draws]), (4/5)**5)
np.testing.assert_allclose(np.mean([len(set(draw)) for draw in draws]), 3.3616)
np.testing.assert_allclose(100*(.99**100), 36.60323412732292)
assert np.mean([2,5,8]) == 5 and np.mean([8,9,10]) == 9
probs = np.array([.51,.51,.01])
assert (probs > .5).sum() == 2 and probs.mean() < .5
np.testing.assert_allclose(probs.mean(), 1.03/3)

# Variance of the mean from an explicitly constructed covariance matrix.
for B in [10,100]:
    for rho in [.2,.8]:
        cov = np.full((B,B),rho)
        np.fill_diagonal(cov,1)
        numeric = np.ones(B)@cov@np.ones(B)/B**2
        np.testing.assert_allclose(numeric,rho+(1-rho)/B)
np.testing.assert_allclose([.2+.8/10,.2+.8/100,.8+.2/10,.8+.2/100],
                         [.28,.208,.82,.802])

rng = np.random.default_rng(4)
xx = rng.normal(size=(80,4))
yy = xx[:,0]+xx[:,1]**2-xx[:,2]*xx[:,3]
bag_columns = BaggingRegressor(
    estimator=DecisionTreeRegressor(), max_features=.5, n_estimators=3, random_state=0
).fit(xx, yy)
assert all(len(columns) == 2 for columns in bag_columns.estimators_features_)
assert all(tree.n_features_in_ == 2 for tree in bag_columns.estimators_)
forest = RandomForestRegressor(max_features=2, n_estimators=3, random_state=0).fit(xx, yy)
assert all(tree.n_features_in_ == 4 and tree.max_features_ == 2
           for tree in forest.estimators_)
assert RandomForestRegressor().max_features == 1.0
assert RandomForestClassifier().max_features == "sqrt"
clf = RandomForestClassifier(
    n_estimators=5, min_samples_leaf=4, random_state=0
).fit(xx, yy > np.median(yy))
np.testing.assert_allclose(clf.predict_proba(xx[:5]),
                           np.mean([t.predict_proba(xx[:5]) for t in clf.estimators_],axis=0))

# Binary AdaBoost in the ±1 convention; the hypothetical learner makes one mistake.
sign_y = np.array([1,1,-1,-1])
sign_pred = np.array([1,1,-1,1])
weights = np.ones(4)/4
eps = np.sum(weights*(sign_y != sign_pred))
alpha = .5*np.log((1-eps)/eps)
new_weights = weights*np.exp(-alpha*sign_y*sign_pred)
new_weights /= new_weights.sum()
np.testing.assert_allclose(new_weights,[1/6,1/6,1/6,.5])
np.testing.assert_allclose(-(1-eps)*np.exp(-alpha)+eps*np.exp(alpha),0,atol=1e-15)
assert (1-eps)*np.exp(-alpha)+eps*np.exp(alpha) < 1
# SAMME's binary convention doubles the coefficient and updates mistakes only.
samme_alpha = np.log((1-eps)/eps)
samme_weights = weights*np.exp(samme_alpha*(sign_y != sign_pred))
samme_weights /= samme_weights.sum()
np.testing.assert_allclose(new_weights,samme_weights)

# The two actual fitted stumps match the full worked trajectory.
expected = [[4,4,4,4],[3,3,5,5],[2,10/3,16/3,16/3]]
np.testing.assert_allclose(scope["stages"],expected)
np.testing.assert_allclose([t.tree_.threshold[0] for t in scope["stumps"]],[2.5,1.5])
np.testing.assert_allclose([np.mean((scope["y_boost"]-p)**2) for p in scope["stages"]],
                           [5,2,1])
library_gb = GradientBoostingRegressor(
    n_estimators=2,learning_rate=.5,max_depth=1,random_state=0
).fit(scope["X_boost"],scope["y_boost"])
np.testing.assert_allclose(library_gb.predict(scope["X_boost"]),expected[-1])
full_step = expected[0]+scope["stumps"][0].predict(scope["X_boost"])
np.testing.assert_allclose(np.mean((scope["y_boost"]-full_step)**2),1)

# Gradient and Hessian finite differences for binary CE in logit coordinates.
for target in [0.,1.]:
    z=.7; step=1e-4
    f=lambda a: np.logaddexp(0,a)-target*a
    np.testing.assert_allclose((f(z+step)-f(z-step))/(2*step),expit(z)-target,atol=1e-9)
    np.testing.assert_allclose((f(z+step)-2*f(z)+f(z-step))/step**2,
                              expit(z)*(1-expit(z)),atol=1e-7)
np.testing.assert_allclose([expit(2),expit(1),expit(-.5),expit(.5)],
                           [.8807970779778823,.7310585786300049,.3775406687981454,.6224593312018546])
np.testing.assert_allclose(scope["newton_model"].predict_proba(scope["X_boost"])[:,1],
                           expit([-2,-2,2,2]))
# The intercept minimizes training loss before any tree is added.
# Class proportions are imbalanced here; a zero logit would not be optimal.
init_x = np.arange(10).reshape(-1, 1)
init_y = np.array([0] * 8 + [1] * 2)
prior_model = GradientBoostingClassifier(
    n_estimators=1, max_depth=1, random_state=0
).fit(init_x, init_y)
prior_probability = prior_model.init_.predict_proba(init_x)[:, 1]
np.testing.assert_allclose(prior_probability, .2)
initial_logit = np.log(.2 / .8)
np.testing.assert_allclose(initial_logit, -1.3862943611198906)
np.testing.assert_allclose(expit(initial_logit), prior_probability)
after_one_tree = initial_logit + prior_model.learning_rate * prior_model.estimators_[0, 0].predict(init_x)
np.testing.assert_allclose(prior_model.decision_function(init_x), after_one_tree)
np.testing.assert_allclose(prior_model.predict_proba(init_x)[:, 1], expit(after_one_tree))
pseudo_residuals = init_y - prior_probability
np.testing.assert_allclose(pseudo_residuals, [-.2] * 8 + [.8] * 2)
np.testing.assert_allclose(pseudo_residuals.sum(), 0, atol=1e-14)
constant_loss = lambda c: np.sum(np.logaddexp(0, c) - init_y * c)
assert constant_loss(initial_logit) < constant_loss(0)
for c in [initial_logit - .1, initial_logit + .1]:
    assert constant_loss(initial_logit) < constant_loss(c)
# Reordering rows changes neither the priors nor the optimal constant.
reordered = GradientBoostingClassifier(
    n_estimators=1, max_depth=1, random_state=0
).fit(init_x[::-1], init_y[::-1])
np.testing.assert_allclose(reordered.init_.predict_proba(init_x),
                           prior_model.init_.predict_proba(init_x))
# A public zero-initialization route uses a different first residual target.
zero_model = GradientBoostingClassifier(
    n_estimators=1, max_depth=1, init="zero", random_state=0
).fit(init_x, init_y)
assert zero_model.init_ == "zero"
np.testing.assert_allclose(init_y - expit(0), [-.5] * 8 + [.5] * 2)
# Squared-error initialization is the target mean from the published toy.
np.testing.assert_allclose(library_gb.init_.predict(scope["X_boost"]), [4] * 4)
for G,H in [(1,.5),(-1,.5)]:
    for penalty in [0,1]:
        w=-G/(H+penalty)
        np.testing.assert_allclose(G+(H+penalty)*w,0,atol=1e-15)
np.testing.assert_allclose(.5*(1/.5+1/.5),2)
np.testing.assert_allclose(softmax([0,0,0])-np.array([0,1,0]),[1/3,-2/3,1/3])
for j in range(3):
    delta=np.eye(3)[j]*1e-5
    f=lambda v: logsumexp(v)-v[1]
    np.testing.assert_allclose((f(delta)-f(-delta))/(2e-5),
                              (softmax([0,0,0])-np.array([0,1,0]))[j],atol=1e-9)
multiclass = GradientBoostingClassifier(
    n_estimators=2,max_depth=1,random_state=0
).fit(xx, np.arange(80)%3)
assert multiclass.estimators_.shape == (2,3)

# Binning makes the first three points indistinguishable, regardless of tree count.
raw_x = np.array([1,2,3,100])
bin_x = np.array([0,0,0,1]).reshape(-1,1)
target = np.array([0.,1.,0.,1.])
binned = DecisionTreeRegressor(max_depth=1).fit(bin_x,target)
np.testing.assert_allclose(binned.predict(bin_x),[1/3,1/3,1/3,1])
np.testing.assert_allclose(np.mean((target-binned.predict(bin_x))**2),1/6)
many = GradientBoostingRegressor(
    max_depth=2,n_estimators=100,learning_rate=.2,random_state=0
).fit(bin_x,target)
np.testing.assert_allclose(many.predict(bin_x)[:3],many.predict(bin_x)[0])
assert np.mean((target-many.predict(bin_x))**2) >= 1/6-1e-12
# Construct the stated depth-two tree; greedy fitting need not choose this structure.
constructed = np.where(raw_x <= 2.5,
                       np.where(raw_x <= 1.5,0,1),
                       np.where(raw_x <= 51.5,0,1))
np.testing.assert_array_equal(constructed,target)
assert scope["newton_model"].max_bins == 255
missing_model = HistGradientBoostingClassifier(
    min_samples_leaf=1, max_iter=1, learning_rate=1,
    early_stopping=False, random_state=0
).fit(np.array([0,1,6,np.nan]).reshape(-1,1), [0,0,1,1])
assert missing_model.predict([[np.nan]])[0] == 1
unseen_nan_model = HistGradientBoostingClassifier(
    min_samples_leaf=1, max_iter=1, learning_rate=1,
    early_stopping=False, random_state=0
).fit(np.array([0,1,2,3]).reshape(-1,1), [0,1,1,1])
assert unseen_nan_model.predict([[np.nan]])[0] == 1
category_x = np.tile(np.arange(4), 20).reshape(-1,1)
category_y = np.tile([0,1,0,1], 20)
native = HistGradientBoostingClassifier(
    categorical_features=[0], max_iter=1, max_leaf_nodes=2,
    learning_rate=1, min_samples_leaf=1, early_stopping=False, random_state=0
).fit(category_x, category_y)
np.testing.assert_array_equal(native.predict(category_x), category_y)
# Stumps are additive: diagonal score sums for XOR always agree.
a=np.array([.2,.7]);b=np.array([-.1,1.1])
scores=np.array([a[i]+b[j] for i,j in [(0,0),(0,1),(1,0),(1,1)]])
np.testing.assert_allclose(scores[0]+scores[3],scores[1]+scores[2])
# Stopping after two non-improvements notices the best third round only at round five.
losses=[1,.8,.7,.71,.72];best=float("inf");stale=0
for iteration,value in enumerate(losses,start=1):
    if value < best:
        best=value;stale=0;best_iteration=iteration
    else:
        stale+=1
    if stale == 2:
        break
assert iteration == 5 and best_iteration == 3
print("PASS: 3 published blocks; exact bootstrap/OOB; variance; forest feature scopes;"
      " AdaBoost weights; boosting trajectory; CE gradient/Hessian; Newton leaves;"
      " optimal squared-error and imbalanced BCE intercepts; zero initialization;"
      " multiclass trees; binning limit; stump interactions; stopping vs best iteration.")
print(f"Chapter SHA-256: {hashlib.sha256(CHAPTER.read_bytes()).hexdigest()}")
