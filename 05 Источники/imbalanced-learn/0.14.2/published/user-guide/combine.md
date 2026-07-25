---
title: "Combination of over- and under-sampling"
type: external-resource
status: imported-source
language: en
source_kind: user-guide
source_version: 0.14.2
source_commit: 8504e95f0160f61d1b617ca66f779646d2ee609e
source_sha256: abe9d8702a0ecba1f273dee6beda52f5155c1514768a55ff46fdd439ac6d7084
license: MIT
---

> [!note] Pinned original source
> This readable page was generated from the complete upstream file
> [`doc/combine.rst`](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/8504e95f0160f61d1b617ca66f779646d2ee609e/doc/combine.rst) at commit
> [`8504e95f0160f61d1b617ca66f779646d2ee609e`](https://github.com/scikit-learn-contrib/imbalanced-learn/commit/8504e95f0160f61d1b617ca66f779646d2ee609e). The original is preserved
> byte-for-byte in the Bookvar source corpus and verified by SHA-256
> `abe9d8702a0ecba1f273dee6beda52f5155c1514768a55ff46fdd439ac6d7084`. The project is distributed under the
> [MIT license](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/8504e95f0160f61d1b617ca66f779646d2ee609e/LICENSE).

# Combination of over- and under-sampling

We previously presented `SMOTE` and showed that this method can generate
noisy samples by interpolating new points between marginal outliers and
inliers. This issue can be solved by cleaning the space resulting
from over-sampling.

In this regard, Tomek's link and edited nearest-neighbours are the two cleaning
methods that have been added to the pipeline after applying SMOTE over-sampling
to obtain a cleaner space. The two ready-to use classes imbalanced-learn
implements for combining over- and undersampling methods are: (i)
`SMOTETomek` [batista2004study](/sources/imbalanced-learn/0-14-2/user-guide/bibliography/#batista2004study) and (ii) `SMOTEENN`
[batista2003balancing](/sources/imbalanced-learn/0-14-2/user-guide/bibliography/#batista2003balancing).

Those two classes can be used like any other sampler with parameters identical
to their former samplers:
```python
>>> from collections import Counter
>>> from sklearn.datasets import make_classification
>>> X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
...                            n_redundant=0, n_repeated=0, n_classes=3,
...                            n_clusters_per_class=1,
...                            weights=[0.01, 0.05, 0.94],
...                            class_sep=0.8, random_state=0)
>>> print(sorted(Counter(y).items()))
[(0, 64), (1, 262), (2, 4674)]
>>> from imblearn.combine import SMOTEENN
>>> smote_enn = SMOTEENN(random_state=0)
>>> X_resampled, y_resampled = smote_enn.fit_resample(X, y)
>>> print(sorted(Counter(y_resampled).items()))
[(0, 4060), (1, 4381), (2, 3502)]
>>> from imblearn.combine import SMOTETomek
>>> smote_tomek = SMOTETomek(random_state=0)
>>> X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
>>> print(sorted(Counter(y_resampled).items()))
[(0, 4499), (1, 4566), (2, 4413)]
```

We can also see in the example below that `SMOTEENN` tends to clean more
noisy samples than `SMOTETomek`.

[![Official imbalanced-learn figure: sphx_glr_plot_comparison_combine_001.png](https://imbalanced-learn.org/stable/_images/sphx_glr_plot_comparison_combine_001.png)](https://imbalanced-learn.org/stable/auto_examples/combine/plot_comparison_combine.html)

> [!note] Examples
> - [`sphx_glr_auto_examples_combine_plot_comparison_combine.py`](https://imbalanced-learn.org/stable/)
