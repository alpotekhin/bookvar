---
title: "📝 Exercise M6.04"
type: external-resource
status: imported-source
language: en
source_kind: jupytext-notebook
source_commit: 0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6
---

> [!note] Original source material
> This page preserves [`python_scripts/ensemble_ex_04.py`](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/python_scripts/ensemble_ex_04.py) from the
> [scikit-learn MOOC](https://github.com/INRIA/scikit-learn-mooc/tree/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6) at commit `0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6`.
> Course material is licensed under [CC BY 4.0](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/LICENSE).
> Bookvar changed only publication markup, local asset paths, and characters
> required for safe rendering.

The aim of the exercise is to get familiar with the histogram
gradient-boosting in scikit-learn. Besides, we will use this model within a
cross-validation framework in order to inspect internal parameters found via
grid-search.

We will use the California housing dataset.

```python
from sklearn.datasets import fetch_california_housing

data, target = fetch_california_housing(return_X_y=True, as_frame=True)
target *= 100  # rescale the target in k$
```

First, create a histogram gradient boosting regressor. You can set the trees
number to be large, and configure the model to use early-stopping.

```python
# Write your code here.
```

We will use a grid-search to find some optimal parameter for this model. In
this grid-search, you should search for the following parameters:

* `max_depth: [3, 8]`;
* `max_leaf_nodes: [15, 31]`;
* `learning_rate: [0.1, 1]`.

Feel free to explore the space with additional values. Create the grid-search
providing the previous gradient boosting instance as the model.

```python
# Write your code here.
```

Finally, we will run our experiment through cross-validation. In this regard,
define a 5-fold cross-validation. Besides, be sure to shuffle the data.
Subsequently, use the function `sklearn.model_selection.cross_validate` to run
the cross-validation. You should also set `return_estimator=True`, so that we
can investigate the inner model trained via cross-validation.

```python
# Write your code here.
```

Now that we got the cross-validation results, print out the mean and standard
deviation score.

```python
# Write your code here.
```

Then inspect the `estimator` entry of the results and check the best
parameters values. Besides, check the number of trees used by the model.

```python
# Write your code here.
```

Inspect the results of the inner CV for each estimator of the outer CV.
Aggregate the mean test score for each parameter combination and make a box
plot of these scores.

```python
# Write your code here.
```
