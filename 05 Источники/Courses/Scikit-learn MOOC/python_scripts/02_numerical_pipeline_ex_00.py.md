---
title: "📝 Exercise M1.02"
type: external-resource
status: imported-source
language: en
source_kind: jupytext-notebook
source_commit: 0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6
---

> [!note] Original source material
> This page preserves [`python_scripts/02_numerical_pipeline_ex_00.py`](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/python_scripts/02_numerical_pipeline_ex_00.py) from the
> [scikit-learn MOOC](https://github.com/INRIA/scikit-learn-mooc/tree/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6) at commit `0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6`.
> Course material is licensed under [CC BY 4.0](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/LICENSE).
> Bookvar changed only publication markup, local asset paths, and characters
> required for safe rendering.

The goal of this exercise is to fit a similar model as in the previous
notebook to get familiar with manipulating scikit-learn objects and in
particular the `.fit/.predict/.score` API.

Let's load the adult census dataset with only numerical variables

```python
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census-numeric.csv")
data = adult_census.drop(columns="class")
target = adult_census["class"]
```

In the previous notebook we used `model = KNeighborsClassifier()`. All
scikit-learn models can be created without arguments. This is convenient
because it means that you don't need to understand the full details of a model
before starting to use it.

One of the `KNeighborsClassifier` parameters is `n_neighbors`. It controls the
number of neighbors we are going to use to make a prediction for a new data
point.

What is the default value of the `n_neighbors` parameter?

**Hint**: Look at the documentation on the [scikit-learn
website](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
or directly access the description inside your notebook by running the
following cell. This opens a pager pointing to the documentation.

```python
from sklearn.neighbors import KNeighborsClassifier

# KNeighborsClassifier?
```

Create a `KNeighborsClassifier` model with `n_neighbors=50`

```python
# Write your code here.
```

Fit this model on the data and target loaded above

```python
# Write your code here.
```

Use your model to make predictions on the first 10 data points inside the
data. Do they match the actual target values?

```python
# Write your code here.
```

Compute the accuracy on the training data.

```python
# Write your code here.
```

Now load the test data from `"../datasets/adult-census-numeric-test.csv"` and
compute the accuracy on the test data.

```python
# Write your code here.
```
