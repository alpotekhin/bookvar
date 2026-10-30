"""Verify the published decision-tree calculations; no downloads or plots."""
from pathlib import Path
import hashlib
import itertools
import re
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/01 Классическое машинное обучение/03 Деревья решений.md"
text = CHAPTER.read_text()
# The real three-class example matches the original plot's counts.
np.testing.assert_allclose(np.array([103, 52, 6]) / 161,
                           [0.639751552795031, 0.3229813664596273, 0.037267080745341616])
assert sum([103, 52, 6]) == 161 and sum([4, 1, 90]) == 95
assert 161 + 95 == 256
np.testing.assert_array_equal(np.array([103,52,6]) + [4,1,90], [107,53,96])
blocks = re.findall(r"```python\n(.*?)```", text, re.S)
assert len(blocks) == 3
scope = {}
for i, code in enumerate(blocks):
    exec(compile(code, f"{CHAPTER}:python-{i+1}", "exec"), scope)
X, y, stump = scope["X"], scope["y"], scope["stump"]
# The introductory leaf example must follow classes_, not mention classes in reverse order.
intro_leaf = DecisionTreeClassifier(min_samples_split=6).fit(
    np.arange(5).reshape(-1, 1), [1, 1, 1, 1, 0]
)
np.testing.assert_array_equal(intro_leaf.classes_, [0, 1])
np.testing.assert_allclose(intro_leaf.predict_proba([[0]]), [[1/5, 4/5]])
np.testing.assert_array_equal(intro_leaf.predict([[0]]), [1])

def gini(labels):
    p = np.bincount(labels, minlength=2) / len(labels)
    return 1 - p @ p

def children_cost(t):
    mask = X[:, 0] <= t
    return (mask.sum() * gini(y[mask]) + (~mask).sum() * gini(y[~mask])) / len(y)

expected = [4/9, 2/5, 12/35, 4/15, 2/5, 19/60, 6/35, 3/10, 2/5]
costs = [children_cost(t) for t in np.arange(1.5, 10)]
np.testing.assert_allclose(costs, expected)
assert np.argmin(costs) == 6
np.testing.assert_allclose(gini(y), .48)
np.testing.assert_allclose(.48 - children_cost(7.5), 54/175)
assert stump.tree_.threshold[0] == 7.5
np.testing.assert_allclose(stump.predict_proba([[6], [9]]), [[1/7, 6/7], [1, 0]])
assert np.mean(stump.predict(X) != y) == .1
candidate = DecisionTreeClassifier(max_depth=1, random_state=0).fit(
    (X[:, 0] > 5.5).astype(int).reshape(-1, 1), y
)
assert np.mean(candidate.predict((X[:, 0] > 5.5).astype(int).reshape(-1, 1)) != y) == .3

full = scope["full_tree"]
assert full.get_depth() == 3 and full.get_n_leaves() == 4
np.testing.assert_array_equal(full.predict(X), y)
np.testing.assert_allclose(full.feature_importances_, [1])
np.testing.assert_allclose(stump.feature_importances_, [1])
np.testing.assert_allclose(scope["path"].ccp_alphas, [0, 3/35, 54/175])
np.testing.assert_allclose(scope["path"].impurities, [0, 6/35, .48])
assert [DecisionTreeClassifier(random_state=0, ccp_alpha=a).fit(X, y).get_n_leaves()
        for a in [0, .1, .4]] == [4, 2, 1]
# Independently enumerate all pruned subtrees of the fitted tree.
def subtrees(node):
    tr = full.tree_
    as_leaf = (tr.weighted_n_node_samples[node] / len(y) * tr.impurity[node], 1)
    if tr.children_left[node] == tr.children_right[node]:
        return [as_leaf]
    return [as_leaf] + [
        (r1+r2, l1+l2)
        for (r1, l1), (r2, l2) in itertools.product(
            subtrees(tr.children_left[node]), subtrees(tr.children_right[node]))
    ]
all_trees = subtrees(0)
for alpha, leaves in [(0, 4), (.1, 2), (.4, 1)]:
    selected = min(all_trees, key=lambda item: item[0] + alpha * item[1])
    assert selected[1] == leaves
np.testing.assert_allclose([.48/3, (6/35)/2, 2/15], [.16, 3/35, 2/15])
np.testing.assert_allclose([4*.1, 6/35+2*.1, .48+.1], [.4, .3714285714285714, .58])

# Node-local and whole-sample impurity gains differ.
np.testing.assert_allclose(gini(y[:7]) - (3/7)*gini(y[4:7]), 8/147)
np.testing.assert_allclose((7/10)*(8/147), 4/105)
limited = DecisionTreeClassifier(random_state=0, min_impurity_decrease=.05).fit(X, y)
assert limited.get_n_leaves() == 2
xx = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
yy = np.array([0, 1, 1, 0])
for j in [0, 1]:
    mask = xx[:, j] == 0
    assert gini(yy) == .5 and gini(yy[mask]) == .5 and gini(yy[~mask]) == .5
xor = DecisionTreeClassifier(max_depth=2, random_state=0).fit(xx, yy)
np.testing.assert_array_equal(xor.predict(xx), yy)
assert xor.get_n_leaves() == 4
xor_stop = DecisionTreeClassifier(
    max_depth=2, min_impurity_decrease=1e-6, random_state=0
).fit(xx, yy)
assert xor_stop.get_n_leaves() == 1
np.testing.assert_allclose(xor_stop.predict_proba(xx), np.full((4, 2), .5))

# Regression means, split search, and extrapolation.
xr, yr, reg = scope["X_reg"], scope["y_reg"], scope["reg"]
assert yr.mean() == 3.5 and np.sum((yr-yr.mean())**2) == 17
sse = []
for t in [1.5, 2.5, 3.5]:
    mask = xr[:, 0] <= t
    sse.append(sum(np.sum((part-part.mean())**2) for part in [yr[mask], yr[~mask]]))
np.testing.assert_allclose(sse, [26/3, 1, 26/3])
np.testing.assert_allclose(reg.tree_.value.ravel(), [3.5, 1.5, 5.5])
np.testing.assert_allclose(reg.predict([[1], [4], [100]]), [1.5, 5.5, 5.5])
assert np.mean((reg.predict(xr)-yr)**2) == .25
median_tree = DecisionTreeRegressor(criterion="absolute_error", min_samples_split=4).fit(
    [[1], [2], [3]], [1, 2, 100]
)
assert median_tree.predict([[100]])[0] == 2
for c in [0, 1, 1.5, 2, 2.5, 10, 100]:
    assert np.sum(np.abs(np.array([1,2,100])-2)) <= np.sum(np.abs(np.array([1,2,100])-c))

# Parameter meanings, duplicate features, and affine/nonlinear transformations.
unit = DecisionTreeClassifier(min_samples_split=10, random_state=0).fit(
    np.arange(10).reshape(-1,1), [0]*9+[1]
)
assert sorted(unit.tree_.n_node_samples[unit.tree_.children_left == -1]) == [1, 9]
leaf2 = DecisionTreeClassifier(min_samples_leaf=2, random_state=0).fit(
    np.arange(10).reshape(-1,1), [0]*9+[1]
)
assert leaf2.tree_.n_node_samples[leaf2.tree_.children_left == -1].min() >= 2
duplicate = DecisionTreeClassifier(random_state=0).fit([[1], [1]], [0, 1])
assert duplicate.get_n_leaves() == 1
unit_original = DecisionTreeClassifier(random_state=0).fit([[1], [4]], [0, 1])
unit_scaled = DecisionTreeClassifier(random_state=0).fit([[100], [400]], [0, 1])
unit_squared = DecisionTreeClassifier(random_state=0).fit([[1], [16]], [0, 1])
assert unit_original.tree_.threshold[0] == 2.5
assert unit_scaled.tree_.threshold[0] == 250
assert unit_squared.tree_.threshold[0] == 8.5
assert unit_original.predict([[2.7]])[0] == 1
assert unit_scaled.predict([[270]])[0] == 1
assert unit_squared.predict([[2.7**2]])[0] == 0
weighted = DecisionTreeClassifier(min_samples_split=4).fit(
    [[1], [2], [3]], [0, 1, 1], sample_weight=[4, 1, 1]
)
np.testing.assert_allclose(weighted.predict_proba([[0]]), [[2/3, 1/3]])
missing = DecisionTreeClassifier(random_state=0).fit(
    np.array([0, 1, 6, np.nan]).reshape(-1,1), [0,0,1,1]
)
np.testing.assert_array_equal(missing.predict([[np.nan], [6]]), [1,1])
unseen_nan = DecisionTreeClassifier(random_state=0).fit([[0], [1], [2], [3]], [0,1,1,1])
assert unseen_nan.predict([[np.nan]])[0] == 1
val_x = np.array([[1], [2], [8], [9]])
val_y = np.array([1,1,0,0])
assert np.mean(stump.predict(val_x) == val_y) == 1
assert np.mean(stump.predict(val_x[::-1]) == val_y) == 0
# With duplicated inputs the fitted stump never switches to the intact copy.
dup = scope["duplicate_stump"]
used, unused = scope["used"], scope["unused"]
assert {used, unused} == {0, 1}
assert dup.get_n_leaves() == 2 and dup.get_depth() == 1
check, targets = scope["X_check"], scope["y_check"]
state_before = [a.copy() for a in [dup.tree_.feature, dup.tree_.threshold, dup.tree_.value]]
np.testing.assert_array_equal(dup.predict(check), targets)
changed = check.copy()
changed[:, used] = changed[::-1, used]
assert dup.score(changed, targets) == 0
changed = check.copy()
changed[:, unused] = changed[::-1, unused]
assert dup.score(changed, targets) == 1
for before, after in zip(state_before, [dup.tree_.feature, dup.tree_.threshold, dup.tree_.value]):
    np.testing.assert_array_equal(before, after)
assert len(re.findall(r"!\[\[", text)) == 13
print("PASS: 3 published blocks; class order; all splits; XOR; regression; all pruned subtrees;"
      " growth limits; feature transformations; weights; NaN; fixed-model permutation; 13 figures.")
print("chapter sha256:", hashlib.sha256(CHAPTER.read_bytes()).hexdigest())
