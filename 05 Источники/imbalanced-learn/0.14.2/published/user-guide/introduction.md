---
title: "Introduction"
type: external-resource
status: imported-source
language: en
source_kind: user-guide
source_version: 0.14.2
source_commit: 8504e95f0160f61d1b617ca66f779646d2ee609e
source_sha256: 07502004bf49b25b725a99f35d7c77fa90f0cf03eb98c77dbcc109bb540fc4f1
license: MIT
---

> [!note] Pinned original source
> This readable page was generated from the complete upstream file
> [`doc/introduction.rst`](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/8504e95f0160f61d1b617ca66f779646d2ee609e/doc/introduction.rst) at commit
> [`8504e95f0160f61d1b617ca66f779646d2ee609e`](https://github.com/scikit-learn-contrib/imbalanced-learn/commit/8504e95f0160f61d1b617ca66f779646d2ee609e). The original is preserved
> byte-for-byte in the Bookvar source corpus and verified by SHA-256
> `07502004bf49b25b725a99f35d7c77fa90f0cf03eb98c77dbcc109bb540fc4f1`. The project is distributed under the
> [MIT license](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/8504e95f0160f61d1b617ca66f779646d2ee609e/LICENSE).

# Introduction

### API's of imbalanced-learn samplers

The available samplers follow the
[scikit-learn API](https://scikit-learn.org/stable/getting_started.html#fitting-and-predicting-estimator-basics)
using the base estimator
and incorporating a sampling functionality via the `sample` method:

**Estimator**

    The base object, implements a `fit` method to learn from data:
```text
estimator = obj.fit(data, targets)
```

**Resampler**

    To resample a data sets, each sampler implements a `fit_resample` method:
```text
data_resampled, targets_resampled = obj.fit_resample(data, targets)
```

Imbalanced-learn samplers accept the same inputs as scikit-learn estimators:

- `data`, 2-dimensional array-like structures, such as:
   - Python's list of lists `list`,
   - Numpy arrays `ndarray`,
   - Panda dataframes `DataFrame`,
   - Scipy sparse matrices `csr_matrix` or `csc_matrix`;

- `targets`, 1-dimensional array-like structures, such as:
   - Numpy arrays `ndarray`,
   - Pandas series `Series`.

The output will be of the following type:

- `data_resampled`, 2-dimensional aray-like structures, such as:
   - Numpy arrays `ndarray`,
   - Pandas dataframes `DataFrame`,
   - Scipy sparse matrices `csr_matrix` or `csc_matrix`;

- `targets_resampled`, 1-dimensional array-like structures, such as:
   - Numpy arrays `ndarray`,
   - Pandas series `Series`.

> [!note] Pandas in/out
> Unlike scikit-learn, imbalanced-learn provides support for pandas in/out.
> Therefore providing a dataframe, will output as well a dataframe.

> [!note] Sparse input
> For sparse input the data is **converted to the Compressed Sparse Rows
> representation** (see `scipy.sparse.csr_matrix`) before being fed to the
> sampler. To avoid unnecessary memory copies, it is recommended to choose the
> CSR representation upstream.

### Problem statement regarding imbalanced data sets

The learning and prediction phrases of machine learning algorithms
can be impacted by the issue of **imbalanced datasets**. This imbalance
refers to the difference in the number of samples across different classes.
We demonstrate the effect of training a [Logistic Regression classifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
with varying levels of class balancing by adjusting their weights.

![[00 Учебник/Assets/Figures/imbalanced-learning/plot_comparison_over_sampling.png|Decision function of LogisticRegression, weight=(0.01, 0.01, 0.98), weight=(0.01, 0.05, 0.94), weight=(0.2, 0.1, 0.7), weight=(0.33, 0.33, 0.33)]]

*Guillaume Lemaitre, [Compare over-sampling samplers](https://imbalanced-learn.org/0.14/auto_examples/over-sampling/plot_comparison_over_sampling.html), imbalanced-learn 0.14.2. [Original PNG](https://imbalanced-learn.org/0.14/_images/sphx_glr_plot_comparison_over_sampling_001.png) · [MIT](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/8504e95f0160f61d1b617ca66f779646d2ee609e/LICENSE). Unmodified.*

As expected, the decision function of the Logistic Regression classifier varies significantly
depending on how imbalanced the data is. With a greater imbalance ratio, the decision function
tends to favour the class with the larger number of samples, usually referred to as the
**majority class**.
