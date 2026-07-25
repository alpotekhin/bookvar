---
title: "📝 Exercise M3.02"
type: external-resource
status: imported-source
language: en
source_kind: jupytext-notebook
source_commit: 0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6
---

> [!note] Original source material
> This page preserves [`python_scripts/parameter_tuning_ex_03.py`](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/python_scripts/parameter_tuning_ex_03.py) from the
> [scikit-learn MOOC](https://github.com/INRIA/scikit-learn-mooc/tree/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6) at commit `0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6`.
> Course material is licensed under [CC BY 4.0](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/LICENSE).
> Bookvar changed only publication markup, local asset paths, and characters
> required for safe rendering.

The goal is to find the best set of hyperparameters which maximize the
generalization performance on a training set.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

data, target = fetch_california_housing(return_X_y=True, as_frame=True)
target *= 100  # rescale the target in k$

data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42
)
```

In this exercise, we progressively define the regression pipeline and later
tune its hyperparameters.

Start by defining a pipeline that:
* uses a `StandardScaler` to normalize the numerical data;
* uses a `sklearn.neighbors.KNeighborsRegressor` as a predictive model.

```python
# Write your code here.
```

Use `RandomizedSearchCV` with `n_iter=20` and
`scoring="neg_mean_absolute_error"` to tune the following hyperparameters
of the `model`:

- the parameter `n_neighbors` of the `KNeighborsRegressor` with values
  `np.logspace(0, 3, num=10).astype(np.int32)`;
- the parameter `with_mean` of the `StandardScaler` with possible values
  `True` or `False`;
- the parameter `with_std` of the `StandardScaler` with possible values `True`
  or `False`.

The `scoring` function is expected to return higher values for better models,
since grid/random search objects **maximize** it. Because of that, error
metrics like `mean_absolute_error` must be negated (using the `neg_` prefix)
to work correctly (remember lower errors represent better models).

Notice that in the notebook "Hyperparameter tuning by randomized-search" we
pass distributions to be sampled by the `RandomizedSearchCV`. In this case we
define a fixed grid of hyperparameters to be explored. Using a `GridSearchCV`
instead would explore all the possible combinations on the grid, which can be
costly to compute for large grids, whereas the parameter `n_iter` of the
`RandomizedSearchCV` controls the number of different random combination that
are evaluated. Notice that setting `n_iter` larger than the number of possible
combinations in a grid (in this case 10 x 2 x 2 = 40) would lead to repeating
already-explored combinations.

Once the computation has completed, print the best combination of parameters
stored in the `best_params_` attribute.

```python
# Write your code here.
```
