---
title: "📝 Exercise M3.01"
type: external-resource
status: imported-source
language: en
source_kind: jupytext-notebook
source_commit: 0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6
---

> [!note] Original source material
> This page preserves [`python_scripts/parameter_tuning_ex_02.py`](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/python_scripts/parameter_tuning_ex_02.py) from the
> [scikit-learn MOOC](https://github.com/INRIA/scikit-learn-mooc/tree/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6) at commit `0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6`.
> Course material is licensed under [CC BY 4.0](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/LICENSE).
> Bookvar changed only publication markup, local asset paths, and characters
> required for safe rendering.

The goal is to write an exhaustive search to find the best parameters
combination maximizing the model generalization performance.

Here we use a small subset of the Adult Census dataset to make the code faster
to execute. Once your code works on the small subset, try to change
`train_size` to a larger value (e.g. 0.8 for 80% instead of 20%).

```python
import pandas as pd

from sklearn.model_selection import train_test_split

adult_census = pd.read_csv("../datasets/adult-census.csv")

target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=[target_name, "education-num"])

data_train, data_test, target_train, target_test = train_test_split(
    data, target, train_size=0.2, random_state=42
)
```

```python
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OrdinalEncoder

categorical_preprocessor = OrdinalEncoder(
    handle_unknown="use_encoded_value", unknown_value=-1
)
preprocessor = make_column_transformer(
    (categorical_preprocessor, selector(dtype_include=object)),
    remainder="passthrough",
)

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

model = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", HistGradientBoostingClassifier(random_state=42)),
    ]
)
```

Use the previously defined model (called `model`) and using two nested `for`
loops, make a search of the best combinations of the `learning_rate` and
`max_leaf_nodes` parameters. In this regard, you need to train and test the
model by setting the parameters. The evaluation of the model should be
performed using `cross_val_score` on the training set. Use the following
parameters search:
- `learning_rate` for the values 0.01, 0.1, 1 and 10. This parameter controls
  the ability of a new tree to correct the error of the previous sequence of
  trees
- `max_leaf_nodes` for the values 3, 10, 30. This parameter controls the depth
  of each tree.

```python
# Write your code here.
```

Now use the test set to score the model using the best parameters that we
found using cross-validation. You will have to refit the model over the full
training set.

```python
# Write your code here.
```
