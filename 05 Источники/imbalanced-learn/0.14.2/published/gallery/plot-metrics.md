---
title: "Metrics specific to imbalanced learning"
type: external-resource
status: imported-source
language: en
source_kind: gallery-example
source_version: 0.14.2
source_commit: 8504e95f0160f61d1b617ca66f779646d2ee609e
source_sha256: 748be36e07b324adef90a85b8784c03c2c4585affaedd2f8772208d8244bd7bc
license: MIT
---

> [!note] Pinned original source
> This readable page was generated from the complete upstream file
> [`examples/evaluation/plot_metrics.py`](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/8504e95f0160f61d1b617ca66f779646d2ee609e/examples/evaluation/plot_metrics.py) at commit
> [`8504e95f0160f61d1b617ca66f779646d2ee609e`](https://github.com/scikit-learn-contrib/imbalanced-learn/commit/8504e95f0160f61d1b617ca66f779646d2ee609e). The original is preserved
> byte-for-byte in the Bookvar source corpus and verified by SHA-256
> `748be36e07b324adef90a85b8784c03c2c4585affaedd2f8772208d8244bd7bc`. The project is distributed under the
> [MIT license](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/8504e95f0160f61d1b617ca66f779646d2ee609e/LICENSE).

# Metrics specific to imbalanced learning

Specific metrics have been developed to evaluate classifier which
has been trained using imbalanced data. `imblearn` provides mainly
two additional metrics which are not implemented in `sklearn`: (i)
geometric mean and (ii) index balanced accuracy.

```python
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT
```

```python
print(__doc__)

RANDOM_STATE = 42
```

First, we will generate some imbalanced dataset.

```python
from sklearn.datasets import make_classification

X, y = make_classification(
    n_classes=3,
    class_sep=2,
    weights=[0.1, 0.9],
    n_informative=10,
    n_redundant=1,
    flip_y=0,
    n_features=20,
    n_clusters_per_class=4,
    n_samples=5000,
    random_state=RANDOM_STATE,
)
```

We will split the data into a training and testing set.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=RANDOM_STATE
)
```

We will create a pipeline made of a `SMOTE`
over-sampler followed by a `LogisticRegression`
classifier.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
```

```python
from imblearn.pipeline import make_pipeline

model = make_pipeline(
    StandardScaler(),
    SMOTE(random_state=RANDOM_STATE),
    LogisticRegression(max_iter=10_000, random_state=RANDOM_STATE),
)
```

Now, we will train the model on the training set and get the prediction
associated with the testing set. Be aware that the resampling will happen
only when calling `fit`: the number of samples in `y_pred` is the same than
in `y_test`.

```python
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

The geometric mean corresponds to the square root of the product of the
sensitivity and specificity. Combining the two metrics should account for
the balancing of the dataset.

```python
from imblearn.metrics import geometric_mean_score

print(f"The geometric mean is {geometric_mean_score(y_test, y_pred):.3f}")
```

The index balanced accuracy can transform any metric to be used in
imbalanced learning problems.

```python
from imblearn.metrics import make_index_balanced_accuracy

alpha = 0.1
geo_mean = make_index_balanced_accuracy(alpha=alpha, squared=True)(geometric_mean_score)

print(
    f"The IBA using alpha={alpha} and the geometric mean: "
    f"{geo_mean(y_test, y_pred):.3f}"
)
```

```python
alpha = 0.5
geo_mean = make_index_balanced_accuracy(alpha=alpha, squared=True)(geometric_mean_score)

print(
    f"The IBA using alpha={alpha} and the geometric mean: "
    f"{geo_mean(y_test, y_pred):.3f}"
)
```
