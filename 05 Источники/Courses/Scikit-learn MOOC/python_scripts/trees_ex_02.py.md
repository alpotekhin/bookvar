---
title: "📝 Exercise M5.02"
type: external-resource
status: imported-source
language: en
source_kind: jupytext-notebook
source_commit: 0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6
---

> [!note] Original source material
> This page preserves [`python_scripts/trees_ex_02.py`](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/python_scripts/trees_ex_02.py) from the
> [scikit-learn MOOC](https://github.com/INRIA/scikit-learn-mooc/tree/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6) at commit `0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6`.
> Course material is licensed under [CC BY 4.0](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/LICENSE).
> Bookvar changed only publication markup, local asset paths, and characters
> required for safe rendering.

The aim of this exercise is to find out whether a decision tree model is able
to extrapolate.

By extrapolation, we refer to values predicted by a model outside of the range
of feature values seen during the training.

We first load the regression data.

```python
import pandas as pd

penguins = pd.read_csv("../datasets/penguins_regression.csv")

feature_name = "Flipper Length (mm)"
target_name = "Body Mass (g)"
data_train, target_train = penguins[​[feature_name]​], penguins[target_name]
```

> [!note] If you want a deeper overview regarding this dataset, you can refer to the
> Appendix - Datasets description section at the end of this MOOC.

First, create two models, a linear regression model and a decision tree
regression model, and fit them on the training data. Limit the depth at 3
levels for the decision tree.

```python
# Write your code here.
```

Create a synthetic dataset containing all possible flipper length from the
minimum to the maximum of the training dataset. Get the predictions of each
model using this dataset.

```python
# Write your code here.
```

Create a scatter plot containing the training samples and superimpose the
predictions of both models on the top.

```python
# Write your code here.
```

Now, we check the extrapolation capabilities of each model. Create a dataset
containing a broader range of values than your previous dataset, in other
words, add values below and above the minimum and the maximum of the flipper
length seen during training.

```python
# Write your code here.
```

Finally, make predictions with both models on this new interval of data.
Repeat the plotting of the previous exercise.

```python
# Write your code here.
```
