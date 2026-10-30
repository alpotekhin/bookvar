"""Execute the published split examples and check the chapter's toy calculations."""
from pathlib import Path
from math import isclose, sqrt
from fractions import Fraction
from hashlib import sha256
import ast
import doctest
import os
import re
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/00 Математические и ML-основания/05 Train, validation и test.md"
text = CHAPTER.read_text()
blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
assert len(blocks) == 6
scope = {}
leakage_scores = []
for i, code in enumerate(blocks, 1):
    exec(compile(code, f"{CHAPTER}:python-{i}", "exec"), scope)
    if i >= 4:
        leakage_scores.append(scope["accuracy_score"](scope["y_test"], scope["y_pred"]))
    if i == 4:
        leaky_model = scope["gbc"]
        leaky_selection = scope["X_selected"].copy()
    if i == 5:
        proper_predictions = scope["y_pred"].copy()
np.testing.assert_allclose(scope["fold_losses"], [18, 9, 18])
assert scope["targets"].mean() == 5

# Imported code is the original upstream sequence, not a synthetic substitute.
# Ignore only interactive prompts, output reprs, comments and blank-line layout.
source = (ROOT / "05 Источники/Originals/data-splits-generalization-2026-10-22/sklearn-common-pitfalls.rst").read_text()
assert sha256(source.encode()).hexdigest() == "6e2b42374fbb521c35f29eeab05bcbfa9744eed344b5f92e41013338ff60a2f5"
section = source[source.index(">>> import numpy as np"):source.index(">>> from sklearn.model_selection import cross_val_score")]
original = "\n".join(e.source for e in doctest.DocTestParser().get_examples(section))
assert ast.dump(ast.parse("\n".join(blocks[2:]))) == ast.dump(ast.parse(original))
assert "# Incorrect preprocessing: the entire data is transformed" in blocks[3]
np.testing.assert_allclose(leakage_scores, [.76, .5, .5])
assert leaky_selection.shape == (200, 25)
assert scope["X_train"].shape == (150, 10000)
assert scope["X_test"].shape == (50, 10000)
np.testing.assert_array_equal(proper_predictions, scope["y_pred"])
_, test_ids = scope["train_test_split"](np.arange(200), random_state=42)
np.testing.assert_array_equal(scope["y_test"], scope["y"][test_ids])
np.testing.assert_array_equal(scope["X_test"], scope["X"][test_ids])
# A genuinely new random target is independent of both the selected columns
# and the fitted predictor. This empirical check complements that argument;
# it is not a proof of an effect size or a significance threshold.
new_rng = np.random.RandomState(7)
new_X = new_rng.standard_normal((1000, 25))
new_y = new_rng.choice(2, 1000)
fresh_accuracy = scope["accuracy_score"](new_y, leaky_model.predict(new_X))
assert .45 < fresh_accuracy < .55
print("upstream leakage accuracies:", leakage_scores, "fresh random set:", fresh_accuracy)

# Selection and the independent final evaluation use different observations.
validation_correct = np.array([900, 930, 910])
assert validation_correct.argmax() == 1
assert validation_correct.max() / 1000 == .93
assert 1800 / 2000 == .9

# Exactly one of the 1024 possible independent bit strings answers all ten
# fixed binary questions. The maximum across 100 draws is not a new ability.
perfect = Fraction(1, 2**10)
probability = 1 - (1 - perfect) ** 100
assert isclose(float(probability), .09308265650895886)
assert Fraction(1, 2) == Fraction(2**10, 2**11)  # next independent bit

# Nested CV: choose mean-vs-zero inside each outer training partition.
y = scope["targets"]
indices = np.arange(6)
outer_errors, inner_errors, choices = [], [], []
fits = 0
for outer_test in np.array_split(indices, 3):
    development = np.setdiff1d(indices, outer_test)
    scores = []
    for rule in ("mean", "zero"):
        losses = []
        for inner_val in np.array_split(development, 2):
            inner_train = np.setdiff1d(development, inner_val)
            assert not np.intersect1d(inner_train, outer_test).size
            assert not np.intersect1d(inner_val, outer_test).size
            fitted = y[inner_train].mean() if rule == "mean" else 0.
            losses.append(np.mean((y[inner_val] - fitted)**2))
            fits += 1
        scores.append(np.mean(losses))
    choice = int(np.argmin(scores))
    fitted = y[development].mean() if choice == 0 else 0.
    fits += 1
    choices.append(choice)
    inner_errors.append(scores)
    outer_errors.append(np.mean((y[outer_test] - fitted)**2))
np.testing.assert_allclose(inner_errors, [[13, 46], [25, 38], [13, 26]])
assert choices == [0, 0, 0] and fits == 15
np.testing.assert_allclose(outer_errors, [18, 9, 18])

# Ordinary row split overlaps all users. Group split overlaps none.
users = np.array(["u17", "u17", "u42", "u42", "u99", "u99"])
assert set(users[[0, 2, 4]]) & set(users[[1, 3, 5]]) == set(users)
assert not set(users[:4]) & set(users[4:])
split_group = scope["split_group"]
before = {s: split_group(s) for s in users}
after = {s: split_group(s) for s in ["u120", *reversed(users), "u999"]}
assert all(before[s] == after[s] for s in users)
assert {s: before[s] for s in ["u17", "u42", "u99"]} == {
    "u17": "train", "u42": "test", "u99": "train",
}
# Check stable SHA-256 assignment across processes with different hash seeds.
for seed in ("1", "2"):
    env = dict(os.environ, PYTHONHASHSEED=seed)
    output = subprocess.check_output(
        [sys.executable, "-c", blocks[1] + "\nprint(split_group('u17'))"],
        env=env, text=True,
    )
    assert output.strip() == "train"

# A stable hash does not order time; an explicit boundary does.
assert ["train" if x < 8 else "test" for x in [2, 9, 1]] == [
    "train", "test", "train"
]
train_days, val_days, test_days = range(1, 21), range(21, 26), range(26, 31)
assert max(train_days) < min(val_days) < min(test_days)
# If labels are available only after the complete seven-day horizon, an
# event from day 14 is not available at the end-of-day-20 training boundary.
assert 13 + 7 <= 20 < 14 + 7

# Centering leaks the held-out feature even without its label.
x_train, x_test = np.array([0., 2.]), np.array([100.])
mu_train = x_train.mean()
mu_all = np.r_[x_train, x_test].mean()
assert mu_train == 1 and mu_all == 34
np.testing.assert_allclose(x_train - mu_train, [-1, 1])
np.testing.assert_allclose(x_test - mu_train, [99])
np.testing.assert_allclose(x_train - mu_all, [-34, -32])

# Sampling variability under the explicitly stated iid Bernoulli model.
assert isclose(sqrt(.9 * .1 / 100), .03)
assert isclose(sqrt(.9 * .1 / 1000), .009486832980505138)
assert isclose(sqrt(.9 * .1 / 400), .03 / 2)
assert isclose(.99**100, .3660323412732292)

print("PASS: 6 published blocks; upstream AST parity, leakage/pipeline reproduction, selection, nested CV, groups, time, hashes, SE, rare class")
print("chapter sha256:", sha256(CHAPTER.read_bytes()).hexdigest())
