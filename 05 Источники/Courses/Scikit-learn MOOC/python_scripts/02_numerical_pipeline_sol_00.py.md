---
title: "📃 Solution for Exercise M1.02"
type: external-resource
status: imported-source
language: en
source_kind: jupytext-notebook
source_commit: 0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6
---

> [!note] Original source material
> This page preserves [`python_scripts/02_numerical_pipeline_sol_00.py`](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/python_scripts/02_numerical_pipeline_sol_00.py) from the
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

# %% [markdown] tags=["solution"]
# We can see that the default value for `n_neighbors` is 5.
```

Create a `KNeighborsClassifier` model with `n_neighbors=50`

```python
# solution
model = KNeighborsClassifier(n_neighbors=50)
```

Fit this model on the data and target loaded above

```python
# solution
model.fit(data, target)
```

Use your model to make predictions on the first 10 data points inside the
data. Do they match the actual target values?

```python
# solution
first_data_values = data.iloc[:10]
first_predictions = model.predict(first_data_values)
first_predictions

# %% tags=["solution"]
first_target_values = target.iloc[:10]
first_target_values

# %% tags=["solution"]
number_of_correct_predictions = (
    first_predictions == first_target_values
).sum()
number_of_predictions = len(first_predictions)
print(
    f"{number_of_correct_predictions}/{number_of_predictions} "
    "of predictions are correct"
)
```

Compute the accuracy on the training data.

```python
# solution
model.score(data, target)
```

Now load the test data from `"../datasets/adult-census-numeric-test.csv"` and
compute the accuracy on the test data.

```python
# solution
adult_census_test = pd.read_csv("../datasets/adult-census-numeric-test.csv")

data_test = adult_census_test.drop(columns="class")
target_test = adult_census_test["class"]

model.score(data_test, target_test)

# %% [markdown] tags=["solution"]
# Looking at the previous notebook, the accuracy seems slightly higher with
# `n_neighbors=50` than with `n_neighbors=5` (the default value).
```
