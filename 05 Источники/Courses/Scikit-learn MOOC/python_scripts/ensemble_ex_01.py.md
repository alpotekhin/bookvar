---
title: "📝 Exercise M6.01"
type: external-resource
status: imported-source
language: en
source_kind: jupytext-notebook
source_commit: 0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6
---

> [!note] Original source material
> This page preserves [`python_scripts/ensemble_ex_01.py`](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/python_scripts/ensemble_ex_01.py) from the
> [scikit-learn MOOC](https://github.com/INRIA/scikit-learn-mooc/tree/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6) at commit `0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6`.
> Course material is licensed under [CC BY 4.0](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/LICENSE).
> Bookvar changed only publication markup, local asset paths, and characters
> required for safe rendering.

The aim of this notebook is to investigate if we can tune the hyperparameters
of a bagging regressor and evaluate the gain obtained.

We will load the California housing dataset and split it into a training and a
testing set.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

data, target = fetch_california_housing(as_frame=True, return_X_y=True)
target *= 100  # rescale the target in k$
data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=0, test_size=0.5
)
```

> [!note] If you want a deeper overview regarding this dataset, you can refer to the
> Appendix - Datasets description section at the end of this MOOC.

Create a `BaggingRegressor` and provide a `DecisionTreeRegressor` to its
parameter `estimator`. Train the regressor and evaluate its generalization
performance on the testing set using the mean absolute error.

```python
# Write your code here.
```

Now, create a `RandomizedSearchCV` instance using the previous model and tune
the important parameters of the bagging regressor. Find the best parameters
and check if you are able to find a set of parameters that improve the default
regressor still using the mean absolute error as a metric.

> [!tip] You can list the bagging regressor's parameters using the `get_params` method.

```python
# Write your code here.
```
