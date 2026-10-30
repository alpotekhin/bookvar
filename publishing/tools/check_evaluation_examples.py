"""Execute published examples and verify evaluation arithmetic and split isolation."""
from pathlib import Path
from itertools import product
import hashlib
import re
import warnings
import numpy as np
from scipy.special import expit
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    balanced_accuracy_score, average_precision_score, roc_auc_score,
    brier_score_loss, mean_absolute_error, mean_squared_error,
    root_mean_squared_error, median_absolute_error, r2_score,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import (
    StratifiedKFold, StratifiedGroupKFold, KFold, GroupKFold, TimeSeriesSplit,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/01 Классическое машинное обучение/05 Оценивание, кросс-валидация и выбор порога.md"
blocks = re.findall(r"```python\n(.*?)```", CHAPTER.read_text(), re.S)
assert len(blocks) == 2
scope = {}
exec(compile(blocks[0], str(CHAPTER) + ":block1", "exec"), scope)
y, scores = scope["y"], scope["scores"]
for threshold, counts, cost, f1 in [(.5,(1,1,1,97),11,.5),(.3,(2,0,1,97),1,.8)]:
    tn, fp, fn, tp = confusion_matrix(y, scores >= threshold, labels=[0,1]).ravel()
    assert (tp,fn,fp,tn) == counts
    assert 10*fn+fp == cost
    np.testing.assert_allclose(f1_score(y,scores>=threshold),f1)
np.testing.assert_allclose(balanced_accuracy_score(y,scores>=.5),(.5+97/98)/2)
np.testing.assert_allclose(balanced_accuracy_score(y,np.zeros_like(y)),.5)
for t, tp_expected, fp_expected in [(1,0,0),(.9,1,0),(.6,1,1),(.4,2,1),(.1,2,98)]:
    pred=scores>=t
    assert np.sum((y==1)&pred)==tp_expected and np.sum((y==0)&pred)==fp_expected
pairs = [(a>b)+.5*(a==b) for a in scores[y==1] for b in scores[y==0]]
assert len(pairs)==196 and sum(pairs)==195
np.testing.assert_allclose(roc_auc_score(y,scores),195/196)
np.testing.assert_allclose(average_precision_score(y,scores),.5+.5*2/3)
np.testing.assert_allclose(average_precision_score(y,np.ones_like(scores)),.02)
assert roc_auc_score([0,0,1,1],[4,3,2,1]) == 0
np.testing.assert_allclose(brier_score_loss(y,scores),.017)
np.testing.assert_allclose(brier_score_loss(y,np.full(100,.02)),.0196)
np.testing.assert_allclose([-np.log(.9),-np.log(.01)],[.10536051565782628,4.605170185988091])
np.testing.assert_allclose(expit(.5*np.array([2,4])-1),[.5,.7310585786300049])
isotonic = IsotonicRegression().fit([1,2,3,4],[0,1,0,1])
np.testing.assert_allclose(isotonic.predict([1,2,3,4]),[0,.5,.5,1])
assert np.sum((isotonic.predict([1,2,3,4])-[0,1,0,1])**2)==.5
for p in [.01,.09,.091,.2,.5,.99]:
    assert (1*(1-p) <= 10*p) == (p >= 1/11)
assert 100/500 == .2 and 100/200 == .5
# An upper capacity bound does not force accepting negative-benefit decisions.
np.testing.assert_allclose([1 * (1 - .01), 10 * .01], [.99, .1])
assert 1 * (1 - .01) > 10 * .01
# Micro and macro recall for a classifier that misses the minority class.
np.testing.assert_allclose(recall_score([0]*8+[1]*2,[0]*10,average="micro"),.8)
np.testing.assert_allclose(recall_score([0]*8+[1]*2,[0]*10,average="macro"),.5)
# Five folds cannot each receive one of only two positives.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    folds=list(StratifiedKFold(5).split(np.zeros((100,1)),y))
assert sum(y[test].sum()==0 for train,test in folds)==3
assert any("least populated class" in str(w.message) for w in caught)
# Iris-like ordered classes are absent from the corresponding train split.
ordered_y=np.repeat([0,1,2],50)
for train,test in KFold(3).split(ordered_y):
    assert not set(ordered_y[train]) & set(ordered_y[test])
groups=np.repeat(np.arange(10),3)
for train,test in GroupKFold(5).split(np.zeros((30,1)),groups=groups):
    assert not set(groups[train]) & set(groups[test])
for train,test in TimeSeriesSplit(3,gap=2).split(np.arange(30)):
    assert max(train)+2 < min(test)
# The exact 12-object example published beside the original zero-gap diagram.
gap_example = list(TimeSeriesSplit(n_splits=3, test_size=2, gap=2).split(np.arange(12)))
for (train, test), (train_stop, gap, expected_test) in zip(
    gap_example,
    [(4, [4, 5], [6, 7]), (6, [6, 7], [8, 9]), (8, [8, 9], [10, 11])],
):
    np.testing.assert_array_equal(train, np.arange(train_stop))
    np.testing.assert_array_equal(test, expected_test)
    np.testing.assert_array_equal(np.arange(train[-1] + 1, test[0]), gap)
    assert not set(train) & set(gap) and train[-1] < test[0]
    # Each label uses the next two outcomes, unavailable from the test start.
    assert train[-1] + 2 < test[0]
    assert all(index + 2 >= test[0] for index in gap)
# Reproduce only the source gallery's data and split indices, not its pictures.
gallery_rng = np.random.RandomState(1338)
gallery_x = gallery_rng.randn(100, 10)
gallery_y = np.repeat([0, 1, 2], [10, 30, 60])
group_prior = gallery_rng.dirichlet([2] * 10)
gallery_groups = np.repeat(np.arange(10), gallery_rng.multinomial(100, group_prior))
group_test_memberships = np.zeros(100, dtype=int)
for train, test in GroupKFold(4).split(gallery_x, gallery_y, gallery_groups):
    assert not set(gallery_groups[train]) & set(gallery_groups[test])
    group_test_memberships[test] += 1
np.testing.assert_array_equal(group_test_memberships, np.ones(100))
for index, (train, test) in enumerate(TimeSeriesSplit(4).split(gallery_x)):
    np.testing.assert_array_equal(train, np.arange(20 * (index + 1)))
    np.testing.assert_array_equal(test, np.arange(20 * (index + 1), 20 * (index + 2)))
# All positives in one indivisible group cannot occur in every test fold.
single_positive_group = np.repeat(np.arange(6), 3)
single_group_y = (single_positive_group == 0).astype(int)
stratified_group_folds = list(StratifiedGroupKFold(3).split(
    np.zeros((18, 1)), single_group_y, single_positive_group
))
assert sum(single_group_y[test].sum() > 0 for train, test in stratified_group_folds) == 1
for train, test in stratified_group_folds:
    assert not set(single_positive_group[train]) & set(single_positive_group[test])
# Equal fold sizes with different nonlinear F1 aggregation.
a_y=np.array([1]+[0]*9); a_p=np.array([1]+[0]*9)
b_y=np.array([1]+[0]*9); b_p=np.array([1]*9+[0])
np.testing.assert_allclose([f1_score(a_y,a_p),f1_score(b_y,b_p)],[1,.2])
np.testing.assert_allclose(f1_score(np.r_[a_y,b_y],np.r_[a_p,b_p]),1/3)
# AUC scales can be inconsistent across models, although each ranks perfectly.
assert roc_auc_score([0,1],[.1,.2]) == roc_auc_score([0,1],[.8,.9]) == 1
assert roc_auc_score([0,1,0,1],[.1,.2,.8,.9]) == .75
fold_labels = np.array([0, 1, 0, 1])
fold_scores = np.array([.1, .2, .8, .9])
oof_pairs = [(a > b) + .5 * (a == b)
             for a in fold_scores[fold_labels == 1] for b in fold_scores[fold_labels == 0]]
assert len(oof_pairs) == 4 and sum(oof_pairs) == 3
shifted_scores = fold_scores.copy()
shifted_scores[2:] -= .7
assert roc_auc_score(fold_labels, shifted_scores) == 1
for subset in [slice(0, 2), slice(2, 4)]:
    assert roc_auc_score(fold_labels[subset], shifted_scores[subset]) == 1
# Observe actual preprocessing fits while executing the full published procedure.
fits=[]
original_fit=StandardScaler.fit
def observed_fit(self,X,y=None,sample_weight=None):
    fits.append(np.array(X,copy=True))
    return original_fit(self,X,y,sample_weight=sample_weight)
StandardScaler.fit=observed_fit
try:
    exec(compile(blocks[1],str(CHAPTER)+":block2","exec"),scope)
finally:
    StandardScaler.fit=original_fit
assert len(fits)==120
X,yy,result=scope["X"],scope["y"],scope["result"]
outer=list(StratifiedKFold(3,shuffle=True,random_state=3).split(X,yy))
for fold,(train,test) in enumerate(outer):
    test_rows={tuple(row) for row in X[test]}
    train_rows={tuple(row) for row in X[train]}
    for fitted_rows in fits[fold*40:(fold+1)*40]:
        rows={tuple(row) for row in fitted_rows}
        assert rows <= train_rows and not rows & test_rows
    model=result["estimator"][fold]
    pred=model.predict(X[test])
    np.testing.assert_allclose(-result["test_cost"][fold],scope["mean_cost"](yy[test],pred))
    pos_index=list(model.classes_).index(1)
    np.testing.assert_allclose(result["test_auc"][fold],roc_auc_score(yy[test],model.predict_proba(X[test])[:,pos_index]))
# Regression units, mean baseline and the special constant-target case.
truth=np.array([1,2,3]); pred=np.array([1,2,6])
np.testing.assert_allclose([mean_absolute_error(truth,pred),mean_squared_error(truth,pred),root_mean_squared_error(truth,pred),median_absolute_error(truth,pred),r2_score(truth,pred)],[1,3,np.sqrt(3),0,-3.5])
assert r2_score(truth,np.full(3,2))==0
assert r2_score(truth,np.full(3,4))<0
assert mean_squared_error(truth*1000,pred*1000)==3_000_000
np.testing.assert_allclose([mean_absolute_error(truth*1000,pred*1000),root_mean_squared_error(truth*1000,pred*1000)],[1000,1000*np.sqrt(3)])
# The preserved patient-bootstrap example: independent unit and estimand differ.
visits=[np.array([1,0]),np.array([-1]),np.array([1,0,0])]
patient_means=np.array([v.mean() for v in visits])
np.testing.assert_allclose(patient_means.mean(),-1/18)
np.testing.assert_allclose(np.concatenate(visits).mean(),1/6)
draw=[0,0,2]
np.testing.assert_allclose(patient_means[draw].mean(),4/9)
np.testing.assert_allclose(np.concatenate([visits[i] for i in draw]).mean(),3/7)
replicas=[patient_means[list(ix)].mean() for ix in product(range(3),repeat=3)]
np.testing.assert_allclose(np.quantile(replicas,[.025,.975],method="inverted_cdf"),[-1,.5])
np.testing.assert_allclose(np.quantile(replicas,[.025,.975]),[-.7111111111111111,.46388888888888874])
print("PASS: 2 published blocks; classification, ranking, calibration, costs, CV;"
      " exact temporal gap table; original gallery split indices; grouped stratification limit;"
      " pooled AUC pairs and scale shift; all 120 preprocessing fits; regression and 27 paired cluster-bootstrap replicas")
print(f"Chapter SHA-256: {hashlib.sha256(CHAPTER.read_bytes()).hexdigest()}")
