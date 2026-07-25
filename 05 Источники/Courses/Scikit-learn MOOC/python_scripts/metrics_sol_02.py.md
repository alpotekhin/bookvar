---
title: "📃 Solution for Exercise M7.03"
type: external-resource
status: imported-source
language: en
source_kind: jupytext-notebook
source_commit: 0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6
---

> [!note] Original source material
> This page preserves [`python_scripts/metrics_sol_02.py`](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/python_scripts/metrics_sol_02.py) from the
> [scikit-learn MOOC](https://github.com/INRIA/scikit-learn-mooc/tree/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6) at commit `0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6`.
> Course material is licensed under [CC BY 4.0](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/LICENSE).
> Bookvar changed only publication markup, local asset paths, and characters
> required for safe rendering.

As with the classification metrics exercise, we will evaluate the regression
metrics within a cross-validation framework to get familiar with the syntax.

We will use the Ames house prices dataset.

```python
import pandas as pd
import numpy as np

ames_housing = pd.read_csv("../datasets/house_prices.csv")
data = ames_housing.drop(columns="SalePrice")
target = ames_housing["SalePrice"]
data = data.select_dtypes(np.number)
target /= 1000
```

> [!note] If you want a deeper overview regarding this dataset, you can refer to the
> Appendix - Datasets description section at the end of this MOOC.

The first step will be to create a linear regression model.

```python
# solution
from sklearn.linear_model import LinearRegression

model = LinearRegression()
```

Then, use the `cross_val_score` to estimate the generalization performance of
the model. Use a `KFold` cross-validation with 10 folds. Make the use of the
$R^2$ score explicit by assigning the parameter `scoring` (even though it is
the default score).

```python
# solution
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, data, target, cv=10, scoring="r2")
print(f"R2 score: {scores.mean():.3f} ± {scores.std():.3f}")
```

Then, instead of using the $R^2$ score, use the mean absolute error (MAE). You
may need to refer to the documentation for the `scoring` parameter.

```python
# solution
scores = cross_val_score(
    model, data, target, cv=10, scoring="neg_mean_absolute_error"
)
errors = -scores
print(f"Mean absolute error: {errors.mean():.3f} k$ ± {errors.std():.3f}")

# %% [markdown] tags=["solution"]
# The `scoring` parameter in scikit-learn expects score. It means that the
# higher the values, and the smaller the errors are, the better the model is.
# Therefore, the error should be multiplied by -1. That's why the string given
# the `scoring` starts with `neg_` when dealing with metrics which are errors.
```

Finally, use the `cross_validate` function and compute multiple scores/errors
at once by passing a list of scorers to the `scoring` parameter. You can
compute the $R^2$ score and the mean absolute error for instance.

```python
# solution
from sklearn.model_selection import cross_validate

scoring = ["r2", "neg_mean_absolute_error"]
cv_results = cross_validate(model, data, target, scoring=scoring)

# %% tags=["solution"]
import pandas as pd

scores = {
    "R2": cv_results["test_r2"],
    "MAE": -cv_results["test_neg_mean_absolute_error"],
}
scores = pd.DataFrame(scores)
scores

# %% [markdown] tags=["solution"]
# In the Regression Metrics notebook, we introduced the concept of loss function,
# which is the metric optimized when training a model. In the case of
# `LinearRegression`, the fitting process consists in minimizing the mean squared
# error (MSE). Some estimators, such as `HistGradientBoostingRegressor`, can
# use different loss functions, to be set using the `loss` hyperparameter.
#
# Notice that the evaluation metrics and the loss functions are not necessarily
# the same. Let's see an example:
```

```python
# solution
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingRegressor

scoring = ["neg_mean_squared_error", "neg_mean_absolute_error"]
loss_functions = ["squared_error", "absolute_error"]
scores = defaultdict(list)

for loss_func in loss_functions:
    model = HistGradientBoostingRegressor(loss=loss_func)
    cv_results = cross_validate(model, data, target, scoring=scoring)
    mse = -cv_results["test_neg_mean_squared_error"]
    mae = -cv_results["test_neg_mean_absolute_error"]
    scores["loss"].append(loss_func)
    scores["MSE"].append(f"{mse.mean():.1f} ± {mse.std():.1f}")
    scores["MAE"].append(f"{mae.mean():.1f} ± {mae.std():.1f}")
scores = pd.DataFrame(scores)
scores.set_index("loss")

# %% [markdown] tags=["solution"]
# Even if the score distributions overlap due to the presence of outliers in the
# dataset, it is true that the average MSE is lower when `loss="squared_error"`,
# whereas the average MAE is lower when `loss="absolute_error"` as expected.
# Indeed, the choice of a loss function is made depending on the evaluation
# metric that we want to optimize for a given use case.
#
# If you feel like going beyond the contents of this MOOC, you can try different
# combinations of loss functions and evaluation metrics.
#
# Notice that there are some metrics that cannot be directly optimized by
# optimizing a loss function. This is the case for metrics that evolve in a
# discontinuous manner with respect to the internal parameters of the model, as
# learning solvers based on gradient descent or similar optimizers require
# continuity (the details are beyond the scope of this MOOC).
#
# For instance, classification models are often evaluated using metrics computed
# on hard class predictions (i.e. whether a sample belongs to a given class)
# rather than from continuous values such as
# [`predict_proba`](https://scikit-learn.org/stable/glossary.html#term-predict_proba)
# (i.e. the estimated probability of belonging to said given class). Because of
# this, classifiers are typically trained by optimizing a loss function computed
# from some continuous output of the model. We call it a "surrogate loss" as it
# substitutes the metric of interest. For instance `LogisticRegression`
# minimizes the `log_loss` applied to the `predict_proba` output of the model.
# By minimizing the surrogate loss, we maximize the accuracy. However
# scikit-learn does not provide surrogate losses for all possible classification
# metrics.
```
