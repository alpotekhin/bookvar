---
title: "Visualizing scikit-learn pipelines in Jupyter"
type: external-resource
status: imported-source
language: en
source_kind: jupytext-notebook
source_commit: 0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6
---

> [!note] Original source material
> This page preserves [`python_scripts/03_categorical_pipeline_visualization.py`](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/python_scripts/03_categorical_pipeline_visualization.py) from the
> [scikit-learn MOOC](https://github.com/INRIA/scikit-learn-mooc/tree/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6) at commit `0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6`.
> Course material is licensed under [CC BY 4.0](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/LICENSE).
> Bookvar changed only publication markup, local asset paths, and characters
> required for safe rendering.

## First we load the dataset

We need to define our data and target. In this case we build a classification
model

```python
import pandas as pd

ames_housing = pd.read_csv("../datasets/house_prices.csv", na_values="?")

target_name = "SalePrice"
data, target = (
    ames_housing.drop(columns=target_name),
    ames_housing[target_name],
)
target = (target > 200_000).astype(int)
```

We inspect the first rows of the dataframe

```python
data
```

For the sake of simplicity, we can cherry-pick some features and only retain
this arbitrary subset of data:

```python
numeric_features = ["LotArea", "FullBath", "HalfBath"]
categorical_features = ["Neighborhood", "HouseStyle"]
data = data[numeric_features + categorical_features]
```

## Then we create the pipeline

The first step is to define the preprocessing steps

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)

categorical_transformer = OneHotEncoder(handle_unknown="ignore")
```

The next step is to apply the transformations using `ColumnTransformer`

```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)
```

Then we define the model and join the steps in order

```python
from sklearn.linear_model import LogisticRegression

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression()),
    ]
)
model
```

Let's fit it!

```python
model.fit(data, target)
```

Notice that the diagram changes color once the estimator is fit.

So far we used `Pipeline` and `ColumnTransformer`, which allows us to custom
the names of the steps in the pipeline. An alternative is to use
`make_column_transformer` and `make_pipeline`, they do not require, and do not
permit, naming the estimators. Instead, their names are set to the lowercase
of their types automatically.

```python
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

numeric_transformer = make_pipeline(
    SimpleImputer(strategy="median"), StandardScaler()
)
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = make_column_transformer(
    (numeric_transformer, numeric_features),
    (categorical_transformer, categorical_features),
)
model = make_pipeline(preprocessor, LogisticRegression())
model.fit(data, target)
```

## Finally we can score the model using cross-validation:

```python
from sklearn.model_selection import cross_validate

cv_results = cross_validate(model, data, target, cv=5)
scores = cv_results["test_score"]
print(
    "The mean cross-validation accuracy is: "
    f"{scores.mean():.3f} ± {scores.std():.3f}"
)
```

> [!note] In this case, around 86% of the times the pipeline correctly predicts whether
> the price of a house is above or below the 200_000 dollars threshold. But be
> aware that this score was obtained by picking some features by hand, which is
> not necessarily the best thing we can do for this classification task. In this
> example we can hope that fitting a complex machine learning pipelines on a
> richer set of features can improve upon this performance level.
>
> Reducing a price estimation problem to a binary classification problem with a
> single threshold at 200_000 dollars is probably too coarse to be useful in in
> practice. Treating this problem as a regression problem is probably a better
> idea. We will see later in this MOOC how to train and evaluate the performance
> of various regression models.
