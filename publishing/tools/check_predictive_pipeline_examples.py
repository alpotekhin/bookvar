"""Execute the published pipeline example and check the chapter arithmetic.

Run with NumPy, pandas and scikit-learn installed. No network or data downloads.
"""
from pathlib import Path
from datetime import datetime, timedelta
from hashlib import sha256
import re
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/01 Классическое машинное обучение/01 Задача обучения и predictive pipeline.md"
text = CHAPTER.read_text()
blocks = re.findall(r"```python\n(.*?)```", text, re.S)
assert len(blocks) == 2
scope = {}
for i, code in enumerate(blocks, 1):
    exec(compile(code, str(CHAPTER) + f":python-{i}", "exec"), scope)
preprocess = scope["fitted_preprocess"]
np.testing.assert_allclose(scope["encoded"], [[2, 1, 1, 0], [0, 0, 0, 0]])
assert scope["probabilities"].shape == (2, 2)
np.testing.assert_allclose(scope["probabilities"].sum(axis=1), [1, 1])
assert preprocess.transform(scope["X_train"]).shape == (4, 4)
assert scope["X_train"].shape == (4, 3)
numeric = preprocess.named_transformers_["num"]
imputer = numeric.named_steps["simpleimputer"]
scaler = numeric.named_steps["standardscaler"]
np.testing.assert_allclose(imputer.statistics_, [6, 80])
np.testing.assert_allclose(scaler.mean_, [6, 80])
np.testing.assert_allclose(scaler.scale_, [3, 20])
encoder = preprocess.named_transformers_["cat"].named_steps["onehotencoder"]
assert encoder.categories_[0].tolist() == ["basic", "premium"]
assert preprocess.get_feature_names_out().tolist() == [
    "num__tenure_months", "num__monthly_spend", "cat__plan_basic", "cat__plan_premium"
]
assert scope["model"].classes_.tolist() == [0, 1]

# The final prediction uses the same fitted coefficients as the published fit,
# not a new illustrative weight vector unrelated to the pipeline.
classifier = scope["classifier"]
assert classifier.coef_.shape == (1, 4)
np.testing.assert_allclose(classifier.coef_[0], [0, 0, .67483035, -.67483035], atol=1e-6)
assert abs(classifier.intercept_[0]) < 1e-12
np.testing.assert_allclose(scope["logits_from_parts"], [.67483035, 0], atol=1e-6)
np.testing.assert_allclose(scope["positive_probability"], [.66258391, .5], atol=1e-6)
np.testing.assert_allclose(
    scope["logits_from_parts"], scope["encoded"] @ classifier.coef_[0] + classifier.intercept_[0]
)
assert scope["positive_column"] == 1
assert .6 < scope["positive_probability"][0] < .7
# The unknown plan contributes zero one-hot coordinates, not a zero probability.
np.testing.assert_array_equal(scope["encoded"][1], np.zeros(4))
assert np.isclose(scope["positive_probability"][1], .5)

# Raising the two separating weights improves the data term but costs more L2.
train_encoded = preprocess.transform(scope["X_train"])
def data_loss(weights):
    z = train_encoded @ weights + classifier.intercept_[0]
    return np.mean(np.logaddexp(0, z) - scope["y_train"]*z)
larger_weights = classifier.coef_[0] * 1.1
assert data_loss(larger_weights) < data_loss(classifier.coef_[0])
assert np.sum(larger_weights**2) > np.sum(classifier.coef_[0]**2)

# Pipeline forwards y to intermediate fit calls when a transform needs it.
class TargetRecordingTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y):
        self.seen_y_ = np.array(y, copy=True)
        return self
    def transform(self, X):
        return X
probe = make_pipeline(TargetRecordingTransformer(), DummyClassifier())
probe.fit(train_encoded, scope["y_train"])
np.testing.assert_array_equal(probe.steps[0][1].seen_y_, scope["y_train"])

# score receives labels only for comparison and leaves all fitted state intact.
before = [a.copy() for a in [classifier.coef_, classifier.intercept_, scaler.mean_, scaler.scale_,
                           imputer.statistics_, encoder.categories_[0]]]
assert scope["model"].score(scope["X_train"], scope["y_train"]) == 1
assert scope["model"].score(scope["X_train"], 1-scope["y_train"]) == 0
for expected, actual in zip(before, [classifier.coef_, classifier.intercept_, scaler.mean_, scaler.scale_,
                                    imputer.statistics_, encoder.categories_[0]]):
    np.testing.assert_array_equal(actual, expected)

# Positive labels may be known before the horizon ends. A missing event before
# the complete horizon is not an observed negative. Cohort eligibility is
# separate from whether the individual label has already become known.
start = datetime(2026, 7, 1)
end = start + timedelta(days=30)
cutoff = datetime(2026, 7, 15)
assert end == datetime(2026, 7, 31)
def observed_label(event, observed_until):
    if event is not None and start < event <= min(end, observed_until):
        return 1
    return 0 if observed_until >= end else None
assert observed_label(datetime(2026, 7, 10), cutoff) == 1
assert observed_label(None, cutoff) is None
assert observed_label(None, end) == 0
assert observed_label(datetime(2026, 8, 1), end) == 0
assert cutoff < end  # neither recent row is yet eligible for the mature cohort

# Majority baseline versus the candidate risk-detection rule.
tp, fp, fn, tn = 4, 6, 1, 89
np.testing.assert_allclose(
    [(tp+tn)/100, tp/(tp+fp), tp/(tp+fn)], [.93, .4, .8]
)
assert (tn+fp)/100 == .95
answers = np.array([1., 2., 9.])
assert answers.mean() == 4 and np.median(answers) == 2
np.testing.assert_allclose(
    [np.mean((answers-c)**2) for c in [4., 2.]], [38/3, 50/3]
)
np.testing.assert_allclose(
    [np.mean(abs(answers-c)) for c in [4., 2.]], [10/3, 8/3]
)

# fit stores the observations; k controls voting.
for k, expected in [(1, 1), (3, 0)]:
    knn = KNeighborsClassifier(n_neighbors=k).fit([[2], [5], [9]], [1, 0, 0])
    assert knn.predict([[3]])[0] == expected

# Changing units changes raw distances, not standardized ones.
q, a, b = np.array([6., 100.]), np.array([5., 80.]), np.array([1., 100.])
raw = [np.linalg.norm(q-a), np.linalg.norm(q-b)]
thousands = [np.linalg.norm((q-v)/[1., 1000.]) for v in [a, b]]
scaled = [np.linalg.norm((q-v)/[3., 20.]) for v in [a, b]]
np.testing.assert_allclose(raw, [np.sqrt(401), 5])
np.testing.assert_allclose(thousands, [np.sqrt(1.0004), 5])
np.testing.assert_allclose(scaled, [np.sqrt(10/9), 5/3])
assert np.argmin(raw) == 1
assert np.argmin(thousands) == np.argmin(scaled) == 0
np.testing.assert_allclose(
    scaled, [np.linalg.norm(((q-v)/[1., 1000.])/[3., .02]) for v in [a, b]]
)
assert (2/100)**2 == 2**2/10000

# Train scaling and accidentally refitted scaling have different meanings.
np.testing.assert_allclose(
    StandardScaler().fit_transform([[3], [9], [3], [9]]).ravel(), [-1, 1, -1, 1]
)
assert StandardScaler().fit([[0], [2]]).mean_[0] == 1
assert StandardScaler().fit([[0], [2], [100]]).mean_[0] == 34
assert StandardScaler().fit_transform([[12]])[0, 0] == 0

# Cross-validation clones and refits the full pipeline, including preprocessing.
cv = [([0, 1], [2, 3]), ([0, 2], [1, 3])]
result = cross_validate(
    scope["model"], scope["X_train"], scope["y_train"],
    cv=cv, return_estimator=True, error_score="raise"
)
means = [
    fitted.named_steps["columntransformer"].named_transformers_["num"]
    .named_steps["standardscaler"].mean_
    for fitted in result["estimator"]
]
np.testing.assert_allclose(means, [[6, 80], [3, 60]])
np.testing.assert_allclose(scope["scaler"].mean_, [6, 80])

# Top-k and threshold policies operate on already computed risk scores.
risks = np.array([.9, .8, .7, .6, .4, .2])
labels = np.array([1, 0, 1, 0, 1, 0])
assert labels[:2].sum() == 1
selected = risks >= .5
assert selected.sum() == 4 and labels[selected].sum() == 2
np.testing.assert_allclose(
    [labels[:2].mean(), labels[:2].sum()/labels.sum(),
     labels[selected].mean(), labels[selected].sum()/labels.sum()],
    [.5, 1/3, .5, 2/3]
)
assert text.count("![[") == 7
print("PASS: 2 published blocks; fitted coefficients/logits/probabilities, L2 tradeoff, y forwarding, score state isolation, label maturity, fold isolation, arithmetic")
print("chapter sha256:", sha256(CHAPTER.read_bytes()).hexdigest())
