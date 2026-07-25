---
title: "Usage of pipeline embedding samplers"
type: external-resource
status: imported-source
language: en
source_kind: gallery-example
source_version: 0.14.2
source_commit: 8504e95f0160f61d1b617ca66f779646d2ee609e
source_sha256: adeb634da81911b85e42ee397ee8c6ad848dba7d1218f4c9597d52a80a6e9aca
license: MIT
---

> [!note] Pinned original source
> This readable page was generated from the complete upstream file
> [`examples/pipeline/plot_pipeline_classification.py`](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/8504e95f0160f61d1b617ca66f779646d2ee609e/examples/pipeline/plot_pipeline_classification.py) at commit
> [`8504e95f0160f61d1b617ca66f779646d2ee609e`](https://github.com/scikit-learn-contrib/imbalanced-learn/commit/8504e95f0160f61d1b617ca66f779646d2ee609e). The original is preserved
> byte-for-byte in the Bookvar source corpus and verified by SHA-256
> `adeb634da81911b85e42ee397ee8c6ad848dba7d1218f4c9597d52a80a6e9aca`. The project is distributed under the
> [MIT license](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/8504e95f0160f61d1b617ca66f779646d2ee609e/LICENSE).

# Usage of pipeline embedding samplers

An example of the :class:~imblearn.pipeline.Pipeline` object (or
`make_pipeline` helper function) working with
transformers and resamplers.

```python
# Authors: Christos Aridas
#          Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT
```

```python
print(__doc__)
```

Let's first create an imbalanced dataset and split in to two sets.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_classes=2,
    class_sep=1.25,
    weights=[0.3, 0.7],
    n_informative=3,
    n_redundant=1,
    flip_y=0,
    n_features=5,
    n_clusters_per_class=1,
    n_samples=5000,
    random_state=10,
)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
```

Now, we will create each individual steps that we would like later to combine

```python
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

pca = PCA(n_components=2)
enn = EditedNearestNeighbours()
smote = SMOTE(random_state=0)
knn = KNeighborsClassifier(n_neighbors=1)
```

Now, we can finally create a pipeline to specify in which order the different
transformers and samplers should be executed before to provide the data to
the final classifier.

```python
from imblearn.pipeline import make_pipeline

model = make_pipeline(pca, enn, smote, knn)
```

We can now use the pipeline created as a normal classifier where resampling
will happen when calling `fit` and disabled when calling `decision_function`,
`predict_proba`, or `predict`.

```python
from sklearn.metrics import classification_report

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```
