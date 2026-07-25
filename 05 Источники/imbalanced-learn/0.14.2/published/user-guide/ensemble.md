---
title: "Ensemble of samplers"
type: external-resource
status: imported-source
language: en
source_kind: user-guide
source_version: 0.14.2
source_commit: 8504e95f0160f61d1b617ca66f779646d2ee609e
source_sha256: 64faa14c0188484d757b8d1d0a5d386cb479fd77a20b770cb1b7e9c8b69080a5
license: MIT
---

> [!note] Pinned original source
> This readable page was generated from the complete upstream file
> [`doc/ensemble.rst`](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/8504e95f0160f61d1b617ca66f779646d2ee609e/doc/ensemble.rst) at commit
> [`8504e95f0160f61d1b617ca66f779646d2ee609e`](https://github.com/scikit-learn-contrib/imbalanced-learn/commit/8504e95f0160f61d1b617ca66f779646d2ee609e). The original is preserved
> byte-for-byte in the Bookvar source corpus and verified by SHA-256
> `64faa14c0188484d757b8d1d0a5d386cb479fd77a20b770cb1b7e9c8b69080a5`. The project is distributed under the
> [MIT license](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/8504e95f0160f61d1b617ca66f779646d2ee609e/LICENSE).

# Ensemble of samplers

## Classifier including inner balancing samplers

### Bagging classifier

In ensemble classifiers, bagging methods build several estimators on different
randomly selected subset of data. In scikit-learn, this classifier is named
`BaggingClassifier`. However, this classifier does not
allow each subset of data to be balanced. Therefore, when training on an imbalanced
data set, this classifier will favor the majority classes:
```python
>>> from sklearn.datasets import make_classification
>>> X, y = make_classification(n_samples=10000, n_features=2, n_informative=2,
...                            n_redundant=0, n_repeated=0, n_classes=3,
...                            n_clusters_per_class=1,
...                            weights=[0.01, 0.05, 0.94], class_sep=0.8,
...                            random_state=0)
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.metrics import balanced_accuracy_score
>>> from sklearn.ensemble import BaggingClassifier
>>> from sklearn.tree import DecisionTreeClassifier
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
>>> bc = BaggingClassifier(DecisionTreeClassifier(), random_state=0)
>>> bc.fit(X_train, y_train) #doctest:
BaggingClassifier(...)
>>> y_pred = bc.predict(X_test)
>>> balanced_accuracy_score(y_test, y_pred)
0.77...
```

In `BalancedBaggingClassifier`, each bootstrap sample will be further
resampled to achieve the `sampling_strategy` desired. Therefore,
`BalancedBaggingClassifier` takes the same parameters as the
scikit-learn `BaggingClassifier`. In addition, the
sampling is controlled by the parameter `sampler` or the two parameters
`sampling_strategy` and `replacement`, if one wants to use the
`RandomUnderSampler`:
```python
>>> from imblearn.ensemble import BalancedBaggingClassifier
>>> bbc = BalancedBaggingClassifier(DecisionTreeClassifier(),
...                                 sampling_strategy='auto',
...                                 replacement=False,
...                                 random_state=0)
>>> bbc.fit(X_train, y_train)
BalancedBaggingClassifier(...)
>>> y_pred = bbc.predict(X_test)
>>> balanced_accuracy_score(y_test, y_pred)
0.8...
```

Changing the `sampler` will give rise to different known implementations
[maclin1997empirical](/sources/imbalanced-learn/0-14-2/user-guide/bibliography/#maclin1997empirical), [hido2009roughly](/sources/imbalanced-learn/0-14-2/user-guide/bibliography/#hido2009roughly),
[wang2009diversity](/sources/imbalanced-learn/0-14-2/user-guide/bibliography/#wang2009diversity). You can refer to the following example which shows these
different methods in practice:
[`sphx_glr_auto_examples_ensemble_plot_bagging_classifier.py`](https://imbalanced-learn.org/stable/)

### Forest of randomized trees

`BalancedRandomForestClassifier` is another ensemble method in which
each tree of the forest will be provided a balanced bootstrap sample
[chen2004using](/sources/imbalanced-learn/0-14-2/user-guide/bibliography/#chen2004using). This class provides all functionality of the
`RandomForestClassifier`:
```python
>>> from imblearn.ensemble import BalancedRandomForestClassifier
>>> brf = BalancedRandomForestClassifier(
...     n_estimators=100, random_state=0, sampling_strategy="all", replacement=True,
...     bootstrap=False,
... )
>>> brf.fit(X_train, y_train)
BalancedRandomForestClassifier(...)
>>> y_pred = brf.predict(X_test)
>>> balanced_accuracy_score(y_test, y_pred)
0.8...
```

### Boosting

Several methods taking advantage of boosting have been designed.

`RUSBoostClassifier` randomly under-samples the dataset before performing
a boosting iteration [seiffert2009rusboost](/sources/imbalanced-learn/0-14-2/user-guide/bibliography/#seiffert2009rusboost):
```python
>>> from imblearn.ensemble import RUSBoostClassifier
>>> rusboost = RUSBoostClassifier(n_estimators=200, random_state=0)
>>> rusboost.fit(X_train, y_train)
RUSBoostClassifier(...)
>>> y_pred = rusboost.predict(X_test)
>>> balanced_accuracy_score(y_test, y_pred)
0...
```

A specific method which uses `AdaBoostClassifier` as
learners in the bagging classifier is called "EasyEnsemble". The
`EasyEnsembleClassifier` allows bagging AdaBoost learners which are
trained on balanced bootstrap samples [liu2008exploratory](/sources/imbalanced-learn/0-14-2/user-guide/bibliography/#liu2008exploratory). Similarly to
the `BalancedBaggingClassifier` API, one can construct the ensemble as:
```python
>>> from imblearn.ensemble import EasyEnsembleClassifier
>>> eec = EasyEnsembleClassifier(random_state=0)
>>> eec.fit(X_train, y_train)
EasyEnsembleClassifier(...)
>>> y_pred = eec.predict(X_test)
>>> balanced_accuracy_score(y_test, y_pred)
0.6...
```

> [!note] Examples
> - [`sphx_glr_auto_examples_ensemble_plot_comparison_ensemble_classifier.py`](https://imbalanced-learn.org/stable/)
