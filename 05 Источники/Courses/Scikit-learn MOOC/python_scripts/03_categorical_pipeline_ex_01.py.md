---
title: "📝 Exercise M1.04"
type: external-resource
status: imported-source
language: en
source_kind: jupytext-notebook
source_commit: 0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6
---

> [!note] Original source material
> This page preserves [`python_scripts/03_categorical_pipeline_ex_01.py`](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/python_scripts/03_categorical_pipeline_ex_01.py) from the
> [scikit-learn MOOC](https://github.com/INRIA/scikit-learn-mooc/tree/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6) at commit `0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6`.
> Course material is licensed under [CC BY 4.0](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/LICENSE).
> Bookvar changed only publication markup, local asset paths, and characters
> required for safe rendering.

The goal of this exercise is to evaluate the impact of using an arbitrary
integer encoding for categorical variables along with a linear classification
model such as Logistic Regression.

To do so, let's try to use `OrdinalEncoder` to preprocess the categorical
variables. This preprocessor is assembled in a pipeline with
`LogisticRegression`. The generalization performance of the pipeline can be
evaluated by cross-validation and then compared to the score obtained when
using `OneHotEncoder` or to some other baseline score.

First, we load the dataset.

```python
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")
```

```python
target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=[target_name, "education-num"])
```

In the previous notebook, we used `sklearn.compose.make_column_selector` to
automatically select columns with a specific data type (also called `dtype`).
Here, we use this selector to get only the columns containing strings (column
with `object` dtype) that correspond to categorical features in our dataset.

```python
from sklearn.compose import make_column_selector as selector

categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(data)
data_categorical = data[categorical_columns]
```

Define a scikit-learn pipeline composed of an `OrdinalEncoder` and a
`LogisticRegression` classifier.

Because `OrdinalEncoder` can raise errors if it sees an unknown category at
prediction time, you can set the `handle_unknown="use_encoded_value"` and
`unknown_value` parameters. You can refer to the [scikit-learn
documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html)
for more details regarding these parameters.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression

# Write your code here.
```

Your model is now defined. Evaluate it using a cross-validation using
`sklearn.model_selection.cross_validate`.

> [!note] Be aware that if an error happened during the cross-validation,
> `cross_validate` would raise a warning and return NaN (Not a Number) as scores.
> To make it raise a standard Python exception with a traceback, you can pass
> the `error_score="raise"` argument in the call to `cross_validate`. An
> exception would be raised instead of a warning at the first encountered problem
> and `cross_validate` would stop right away instead of returning NaN values.
> This is particularly handy when developing complex machine learning pipelines.

```python
from sklearn.model_selection import cross_validate

# Write your code here.
```

Now, we would like to compare the generalization performance of our previous
model with a new model where instead of using an `OrdinalEncoder`, we use a
`OneHotEncoder`. Repeat the model evaluation using cross-validation. Compare
the score of both models and conclude on the impact of choosing a specific
encoding strategy when using a linear model.

```python
from sklearn.preprocessing import OneHotEncoder

# Write your code here.
```
