---
title: "Ensemble of tree-based models"
type: external-resource
status: imported-source
language: en
source_kind: markdown
source_commit: 0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6
---

> [!note] Original source material
> This page preserves [`slides/bagging.md`](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/slides/bagging.md) from the
> [scikit-learn MOOC](https://github.com/INRIA/scikit-learn-mooc/tree/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6) at commit `0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6`.
> Course material is licensed under [CC BY 4.0](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/LICENSE).
> Bookvar changed only publication markup, local asset paths, and characters
> required for safe rendering.

class: titlepage

.header[MOOC Machine learning with scikit-learn]

# Ensemble of tree-based models

Combine many decision trees into powerful models!

Bagging and Random Forests

Boosting and Gradient Boosting

For classification and regression

![[Assets/Sources/Scikit-learn MOOC/figures/scikit-learn-logo.svg|Source figure]]

---

# Part 1: bagging and random forests

---

# Bagging for classification

.pull-left[![[Assets/Sources/Scikit-learn MOOC/figures/bagging0.svg|Source figure]]]
.pull-right[![[Assets/Sources/Scikit-learn MOOC/figures/bagging.svg|Source figure]]]

???
Here we have a classification task: separating circles from squares.

---

# Bagging for classification

.pull-left[![[Assets/Sources/Scikit-learn MOOC/figures/bagging0.svg|Source figure]]]
.pull-right[![[Assets/Sources/Scikit-learn MOOC/figures/bagging_line.svg|Source figure]]]

.pull-right[![[Assets/Sources/Scikit-learn MOOC/figures/bagging_trees.svg|Source figure]]]


???

---

# Bagging for classification

.pull-left[![[Assets/Sources/Scikit-learn MOOC/figures/bagging0_cross.svg|Source figure]]]
.pull-right[![[Assets/Sources/Scikit-learn MOOC/figures/bagging_cross.svg|Source figure]]]

.pull-right[![[Assets/Sources/Scikit-learn MOOC/figures/bagging_trees_predict.svg|Source figure]]]

.pull-right[![[Assets/Sources/Scikit-learn MOOC/figures/bagging_vote.svg|Source figure]]]

--
.width65.shift-up-less.centered[
```python
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
```
]

---

# Bagging for classification

.pull-left[![[Assets/Sources/Scikit-learn MOOC/figures/bagging0_cross.svg|Source figure]]]
.pull-right[![[Assets/Sources/Scikit-learn MOOC/figures/bagging_cross.svg|Source figure]]]

.pull-right[![[Assets/Sources/Scikit-learn MOOC/figures/bagging_trees_predict.svg|Source figure]]]

.width65.shift-up-less.centered[
```python
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
```
]


---

# Bagging for regression

![[Assets/Sources/Scikit-learn MOOC/figures/bagging_reg_data.svg|Source figure]]

---
class: split-50
# Bagging for regression

.shift-up-less[
![[Assets/Sources/Scikit-learn MOOC/figures/bagging_reg_grey.svg|Source figure]]
]

.column1[
- Select multiple random subsets of the data
]

---
class: split-50
# Bagging for regression

.shift-up-less[
![[Assets/Sources/Scikit-learn MOOC/figures/bagging_reg_grey_fitted.svg|Source figure]]
]

.column1[
- Select multiple random subsets of the data
- Fit one model on each
]

---
class: split-50
# Bagging for regression

.shift-up-less[
![[Assets/Sources/Scikit-learn MOOC/figures/bagging_reg_grey_fitted.svg|Source figure]]
]

.column1[
- Select multiple random subsets of the data
- Fit one model on each
- Average predictions
]

.column2.center[
![[Assets/Sources/Scikit-learn MOOC/figures/bagging_reg_blue.svg|Source figure]]
]

???

In bagging, we will construct deep trees independently of one another.

Each tree will be fitted on a sub-sampling from the initial data. i.e. we will
only consider a random part of the data to build each model.

When we have to classify a new point, we will aggregate the predictions of all
models in the ensemble with a voting scheme.

Each deep tree overfits, but voting makes it possible to cancel out some of the
training set noise. The ensemble overfits less than the individual models.

---
# Bagging versus Random Forests

**Bagging** is a general strategy
- Can work with any base model (linear, trees...)

--

**Random Forests** are bagged *randomized* decision trees
- At each split: a random subset of features are selected
--

- The best split is taken among the restricted subset

--
- Extra randomization decorrelates the prediction errors

--
- Uncorrelated errors make bagging work better

???

It's fine to use deep trees (`max_depth=None`) in random forests because of the
reduced overfitting effect of prediction averaging.

The more trees the better, typical to use 100 trees or more.

Diminishing returns when increasing the number of trees.

More trees: longer to fit, slower to predict and bigger models to deploy.

---

# Take away

**Bagging** and **random forests** fit trees **independently**
- each **deep tree overfits** individually
- averaging the tree predictions **reduces overfitting**
