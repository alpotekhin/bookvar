---
title: "📝 Exercise M7.03"
type: external-resource
status: imported-source
language: en
source_kind: jupytext-notebook
source_commit: 0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6
---

> [!note] Original source material
> This page preserves [`python_scripts/metrics_ex_02.py`](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/python_scripts/metrics_ex_02.py) from the
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
# Write your code here.
```

Then, use the `cross_val_score` to estimate the generalization performance of
the model. Use a `KFold` cross-validation with 10 folds. Make the use of the
$R^2$ score explicit by assigning the parameter `scoring` (even though it is
the default score).

```python
# Write your code here.
```

Then, instead of using the $R^2$ score, use the mean absolute error (MAE). You
may need to refer to the documentation for the `scoring` parameter.

```python
# Write your code here.
```

Finally, use the `cross_validate` function and compute multiple scores/errors
at once by passing a list of scorers to the `scoring` parameter. You can
compute the $R^2$ score and the mean absolute error for instance.

```python
# Write your code here.
```
