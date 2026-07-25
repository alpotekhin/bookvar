---
title: "📝 Exercise M1.03"
type: external-resource
status: imported-source
language: en
source_kind: jupytext-notebook
source_commit: 0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6
---

> [!note] Original source material
> This page preserves [`python_scripts/02_numerical_pipeline_ex_01.py`](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/python_scripts/02_numerical_pipeline_ex_01.py) from the
> [scikit-learn MOOC](https://github.com/INRIA/scikit-learn-mooc/tree/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6) at commit `0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6`.
> Course material is licensed under [CC BY 4.0](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/LICENSE).
> Bookvar changed only publication markup, local asset paths, and characters
> required for safe rendering.

The goal of this exercise is to compare the performance of our classifier in
the previous notebook (roughly 81% accuracy with `LogisticRegression`) to some
simple baseline classifiers. The simplest baseline classifier is one that
always predicts the same class, irrespective of the input data.

- What would be the score of a model that always predicts `' >50K'`?
- What would be the score of a model that always predicts `' <=50K'`?
- Is 81% or 82% accuracy a good score for this problem?

Use a `DummyClassifier` and do a train-test split to evaluate its accuracy on
the test set. This
[link](https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators)
shows a few examples of how to evaluate the generalization performance of
these baseline models.

```python
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")
```

We first split our dataset to have the target separated from the data used to
train our predictive model.

```python
target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=target_name)
```

We start by selecting only the numerical columns as seen in the previous
notebook.

```python
numerical_columns = ["age", "capital-gain", "capital-loss", "hours-per-week"]

data_numeric = data[numerical_columns]
```

Split the data and target into a train and test set.

```python
from sklearn.model_selection import train_test_split

# Write your code here.
```

Use a `DummyClassifier` such that the resulting classifier always predict the
class `' >50K'`. What is the accuracy score on the test set? Repeat the
experiment by always predicting the class `' <=50K'`.

Hint: you can set the `strategy` parameter of the `DummyClassifier` to achieve
the desired behavior.

```python
from sklearn.dummy import DummyClassifier

# Write your code here.
```
