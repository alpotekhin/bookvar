---
title: "📝 Exercise M6.03"
type: external-resource
status: imported-source
language: en
source_kind: jupytext-notebook
source_commit: 0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6
---

> [!note] Original source material
> This page preserves [`python_scripts/ensemble_ex_03.py`](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/python_scripts/ensemble_ex_03.py) from the
> [scikit-learn MOOC](https://github.com/INRIA/scikit-learn-mooc/tree/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6) at commit `0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6`.
> Course material is licensed under [CC BY 4.0](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/LICENSE).
> Bookvar changed only publication markup, local asset paths, and characters
> required for safe rendering.

The aim of this exercise is to:

* verifying if a random forest or a gradient-boosting decision tree overfit if
  the number of estimators is not properly chosen;
* use the early-stopping strategy to avoid adding unnecessary trees, to get
  the best generalization performances.

We use the California housing dataset to conduct our experiments.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

data, target = fetch_california_housing(return_X_y=True, as_frame=True)
target *= 100  # rescale the target in k$
data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=0, test_size=0.5
)
```

> [!note] If you want a deeper overview regarding this dataset, you can refer to the
> Appendix - Datasets description section at the end of this MOOC.

Create a gradient boosting decision tree with `max_depth=5` and
`learning_rate=0.5`.

```python
# Write your code here.
```

Also create a random forest with fully grown trees by setting `max_depth=None`.

```python
# Write your code here.
```

For both the gradient-boosting and random forest models, create a validation
curve using the training set to assess the impact of the number of trees on
the performance of each model. Evaluate the list of parameters `param_range =
np.array([1, 2, 5, 10, 20, 50, 100, 200])` and score it using
`neg_mean_absolute_error`. Remember to set `negate_score=True` to recover the
right sign of the Mean Absolute Error.

```python
# Write your code here.
```

Random forest models improve when increasing the number of trees in the
ensemble. However, the scores reach a plateau where adding new trees just
makes fitting and scoring slower.

Now repeat the analysis for the gradient boosting model.

```python
# Write your code here.
```

Gradient boosting models overfit when the number of trees is too large. To
avoid adding a new unnecessary tree, unlike random-forest gradient-boosting
offers an early-stopping option. Internally, the algorithm uses an
out-of-sample set to compute the generalization performance of the model at
each addition of a tree. Thus, if the generalization performance is not
improving for several iterations, it stops adding trees.

Now, create a gradient-boosting model with `n_estimators=1_000`. This number
of trees is certainly too large as we have seen above. Change the parameter
`n_iter_no_change` such that the gradient boosting fitting stops after adding
5 trees to avoid deterioration of the overall generalization performance.

```python
# Write your code here.
```

Estimate the generalization performance of this model again using the
`sklearn.metrics.mean_absolute_error` metric but this time using the test set
that we held out at the beginning of the notebook. Compare the resulting value
with the values observed in the validation curve.

```python
# Write your code here.
```
