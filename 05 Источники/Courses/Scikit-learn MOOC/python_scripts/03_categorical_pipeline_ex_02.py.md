---
title: "📝 Exercise M1.05"
type: external-resource
status: imported-source
language: en
source_kind: jupytext-notebook
source_commit: 0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6
---

> [!note] Original source material
> This page preserves [`python_scripts/03_categorical_pipeline_ex_02.py`](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/python_scripts/03_categorical_pipeline_ex_02.py) from the
> [scikit-learn MOOC](https://github.com/INRIA/scikit-learn-mooc/tree/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6) at commit `0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6`.
> Course material is licensed under [CC BY 4.0](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/LICENSE).
> Bookvar changed only publication markup, local asset paths, and characters
> required for safe rendering.

The goal of this exercise is to evaluate the impact of feature preprocessing
on a pipeline that uses a decision-tree-based classifier instead of a logistic
regression.

- The first question is to empirically evaluate whether scaling numerical
  features is helpful or not;
- The second question is to evaluate whether it is empirically better (both
  from a computational and a statistical perspective) to use integer coded or
  one-hot encoded categories.

```python
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")
```

```python
target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=[target_name, "education-num"])
```

As in the previous notebooks, we use the utility `make_column_selector` to
select only columns with a specific data type. Besides, we list in advance all
categories for the categorical columns.

```python
from sklearn.compose import make_column_selector as selector

numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)
numerical_columns = numerical_columns_selector(data)
categorical_columns = categorical_columns_selector(data)
```

## Reference pipeline (no numerical scaling and integer-coded categories)

First let's time the pipeline we used in the main notebook to serve as a
reference:

```python
import time

from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier

categorical_preprocessor = OrdinalEncoder(
    handle_unknown="use_encoded_value", unknown_value=-1
)
preprocessor = make_column_transformer(
    (categorical_preprocessor, categorical_columns),
    remainder="passthrough",
)


model = make_pipeline(preprocessor, HistGradientBoostingClassifier())

start = time.time()
cv_results = cross_validate(model, data, target)
elapsed_time = time.time() - start

scores = cv_results["test_score"]

print(
    "The mean cross-validation accuracy is: "
    f"{scores.mean():.3f} ± {scores.std():.3f} "
    f"with a fitting time of {elapsed_time:.3f} seconds"
)
```

## Scaling numerical features

Let's write a similar pipeline that also scales the numerical features using
`StandardScaler` (or similar):

```python
# Write your code here.
```

## One-hot encoding of categorical variables

We observed that integer coding of categorical variables can be very
detrimental for linear models. However, it does not seem to be the case for
`HistGradientBoostingClassifier` models, as the cross-validation score of the
reference pipeline with `OrdinalEncoder` is reasonably good.

Let's see if we can get an even better accuracy with `OneHotEncoder`.

Hint: `HistGradientBoostingClassifier` does not yet support sparse input data.
You might want to use `OneHotEncoder(handle_unknown="ignore",
sparse_output=False)` to force the use of a dense representation as a
workaround.

```python
# Write your code here.
```

## Which encoder should I use?

|                  | Meaningful order              | Non-meaningful order |
| ---------------- | ----------------------------- | -------------------- |
| Tree-based model | `OrdinalEncoder`              | `OrdinalEncoder` with reasonable depth    |
| Linear model     | `OrdinalEncoder` with caution | `OneHotEncoder`      |

> [!important] - `OneHotEncoder`: always does something meaningful, but can be unnecessary
>   slow with trees.
> - `OrdinalEncoder`: can be detrimental for linear models unless your category
>   has a meaningful order and you make sure that `OrdinalEncoder` respects this
>   order. Trees can deal with `OrdinalEncoder` fine as long as they are deep
>   enough. However, when you allow the decision tree to grow very deep, it might
>   overfit on other features.

Next to one-hot-encoding and ordinal encoding categorical features,
scikit-learn offers the [`TargetEncoder`](https://scikit-learn.org/stable/modules/preprocessing.html#target-encoder).
This encoder is well suited for nominal, categorical features with high
cardinality. This encoding strategy is beyond the scope of this course,
but the interested reader is encouraged to explore this encoder.
