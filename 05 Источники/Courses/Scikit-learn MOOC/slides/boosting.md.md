---
title: "Ensemble of tree-based models"
type: external-resource
status: imported-source
language: en
source_kind: markdown
source_commit: 0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6
---

> [!note] Original source material
> This page preserves [`slides/boosting.md`](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/slides/boosting.md) from the
> [scikit-learn MOOC](https://github.com/INRIA/scikit-learn-mooc/tree/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6) at commit `0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6`.
> Course material is licensed under [CC BY 4.0](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/LICENSE).
> Bookvar changed only publication markup, local asset paths, and characters
> required for safe rendering.

class: titlepage

.header[MOOC Machine learning with scikit-learn]

# Ensemble of tree-based models

## Part 2: boosting and gradient boosting

![[Assets/Sources/Scikit-learn MOOC/figures/scikit-learn-logo.svg|Source figure]]

---

# Boosting for classification

.pull-left[![[Assets/Sources/Scikit-learn MOOC/figures/boosting0.svg|Source figure]]]

---

# Boosting for classification

.pull-left[![[Assets/Sources/Scikit-learn MOOC/figures/boosting1.svg|Source figure]]]
.pull-right[![[Assets/Sources/Scikit-learn MOOC/figures/boosting_trees1.svg|Source figure]]]

???
A first shallow tree starts to separate circles from squares.
Mistakes done by this first tree model shall be corrected
by a second tree model.

---
# Boosting for classification

.pull-left[![[Assets/Sources/Scikit-learn MOOC/figures/boosting2.svg|Source figure]]]
.pull-right[![[Assets/Sources/Scikit-learn MOOC/figures/boosting_trees2.svg|Source figure]]]

???
So now, the second tree refines the first tree.
The final model is a weighted sum of these two trees.

---
# Boosting for classification

.pull-left[![[Assets/Sources/Scikit-learn MOOC/figures/boosting3.svg|Source figure]]]
.pull-right[![[Assets/Sources/Scikit-learn MOOC/figures/boosting_trees3.svg|Source figure]]]

???

Ensembling via boosting makes it possible to progressively refine the
predictions of the previous model.

At each step we focus on mistakes of the previous model to correct them.

Even if the first models are underfitting (shallow trees), adding more trees
makes it possible to perfectly classify all the training set data points.

---

# Boosting for regression

![[Assets/Sources/Scikit-learn MOOC/figures/boosting/boosting_iter1.svg|Source figure]]

---

# Boosting for regression

![[Assets/Sources/Scikit-learn MOOC/figures/boosting/boosting_iter_sized1.svg|Source figure]]

---

# Boosting for regression

![[Assets/Sources/Scikit-learn MOOC/figures/boosting/boosting_iter_orange1.svg|Source figure]]

---

# Boosting for regression

![[Assets/Sources/Scikit-learn MOOC/figures/boosting/boosting_iter2.svg|Source figure]]

---

# Boosting for regression

![[Assets/Sources/Scikit-learn MOOC/figures/boosting/boosting_iter_sized2.svg|Source figure]]

---

# Boosting for regression

![[Assets/Sources/Scikit-learn MOOC/figures/boosting/boosting_iter_orange2.svg|Source figure]]

---

# Boosting for regression

![[Assets/Sources/Scikit-learn MOOC/figures/boosting/boosting_iter3.svg|Source figure]]

---

# Boosting for regression

![[Assets/Sources/Scikit-learn MOOC/figures/boosting/boosting_iter_sized3.svg|Source figure]]

---

# Boosting for regression

![[Assets/Sources/Scikit-learn MOOC/figures/boosting/boosting_iter_orange3.svg|Source figure]]

---

# Boosting for regression

![[Assets/Sources/Scikit-learn MOOC/figures/boosting/boosting_iter4.svg|Source figure]]

---

# Boosting vs Gradient Boosting

**Traditional Boosting**
.small[`sklearn.ensemble.AdaBoostClassifier`]
- Mispredicted **samples are re-weighted** at each step
- Can use any base model that accepts `sample_weight`

--

**Gradient Boosting**
.small[`sklearn.ensemble.HistGradientBoostingClassifier`]
- Each base model predicts the **negative error** of previous models
- `sklearn` use decision trees as the base model


???

In practice, gradient boosting is more flexible thanks to the use of cost
functions and tend to exhibits better predictive performance than traditional
boosting.

---
# Gradient Boosting and binned features

- `sklearn.ensemble.GradientBoostingClassifier`
  - Implementation of the traditional (exact) method
  - Fine for small data sets
  - Too slow for `n_samples` > 10,000

--

- `sklearn.ensemble.HistGradientBoostingClassifier`
  - Discretize numerical features (256 levels)
  - Efficient multi core implementation
  - **Much, much faster** when `n_samples` is large

???
Like traditional decision trees `GradientBoostingClassifier` and
`GradientBoostingRegressor` internally rely on sorting the features values
which as an `n * log(n)` time complexity and is therefore not suitable for
large training set.

`HistGradientBoostingClassifier` and `HistGradientBoostingRegressor` use
histograms to approximate feature sorting to find the best feature split
thresholds and can therefore be trained efficiently on datasets with hundreds
of features and tens of millions of data points.

Furthermore they can benefit from running on machines with many CPU cores very
efficiently.

---

# Take away

**Bagging**  | **Boosting**
------------ | -------------
fit trees **independently** | fit trees **sequentially**
each **deep tree overfits** | each **shallow tree underfits**
averaging the tree predictions **reduces overfitting** | sequentially adding trees **reduces underfitting**

**Gradient boosting** tends to perform slightly better than **bagging** and
**random forest**. Furthermore, shallow trees predict faster.
