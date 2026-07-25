---
title: "Hyperparameter tuning by grid-search"
type: external-resource
status: imported-source
language: en
source_kind: jupytext-notebook
source_commit: 0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6
---

> [!note] Original source material
> This page preserves [`python_scripts/parameter_tuning_grid_search.py`](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/python_scripts/parameter_tuning_grid_search.py) from the
> [scikit-learn MOOC](https://github.com/INRIA/scikit-learn-mooc/tree/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6) at commit `0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6`.
> Course material is licensed under [CC BY 4.0](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/LICENSE).
> Bookvar changed only publication markup, local asset paths, and characters
> required for safe rendering.

In the previous notebook, we saw that hyperparameters can affect the
generalization performance of a model. In this notebook, we show how to
optimize hyperparameters using a grid-search approach.

## Our predictive model

Let us reload the dataset as we did previously:

```python
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")
```

We extract the column containing the target.

```python
target_name = "class"
target = adult_census[target_name]
target
```

We drop from our data the target and the `"education-num"` column which
duplicates the information from the `"education"` column.

```python
data = adult_census.drop(columns=[target_name, "education-num"])
data
```

Once the dataset is loaded, we split it into a training and testing sets.

```python
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42
)
```

We define a pipeline as seen in the first module, to handle both numerical and
categorical features.

The first step is to select all the categorical columns.

```python
from sklearn.compose import make_column_selector as selector

categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(data)
```

Here we use a tree-based model as a classifier (i.e.
`HistGradientBoostingClassifier`). That means:

* Numerical variables don't need scaling;
* Categorical variables can be dealt with an `OrdinalEncoder` even if the
  coding order is not meaningful;
* For tree-based models, the `OrdinalEncoder` avoids having high-dimensional
  representations.

We now build our `OrdinalEncoder` by passing it the known categories.

```python
from sklearn.preprocessing import OrdinalEncoder

categorical_preprocessor = OrdinalEncoder(
    handle_unknown="use_encoded_value", unknown_value=-1
)
```

We then use `make_column_transformer` to select the categorical columns and apply
the `OrdinalEncoder` to them.

```python
from sklearn.compose import make_column_transformer

preprocessor = make_column_transformer(
    (categorical_preprocessor, categorical_columns),
    remainder="passthrough",
)
```

Finally, we use a tree-based classifier (i.e. histogram gradient-boosting) to
predict whether or not a person earns more than 50 k$ a year.

```python
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

model = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "classifier",
            HistGradientBoostingClassifier(random_state=42, max_leaf_nodes=4),
        ),
    ]
)
model
```

## Tuning using a grid-search

In the previous exercise (M3.01) we used two nested `for` loops (one for each
hyperparameter) to test different combinations over a fixed grid of
hyperparameter values. In each iteration of the loop, we used
`cross_val_score` to compute the mean score (as averaged across
cross-validation splits), and compared those mean scores to select the best
combination. `GridSearchCV` is a scikit-learn class that implements a very
similar logic with less repetitive code. The suffix `CV` refers to the
cross-validation it runs internally (instead of the `cross_val_score` we
"hard" coded).

The `GridSearchCV` estimator takes a `param_grid` parameter which defines all
hyperparameters and their associated values. The grid-search is in charge of
creating all possible combinations and testing them.

The number of combinations is equal to the product of the number of values to
explore for each parameter. Thus, adding new parameters with their associated
values to be explored rapidly becomes computationally expensive. Because of
that, here we only explore the combination learning-rate and the maximum
number of nodes for a total of 4 x 3 = 12 combinations.

```python
# %%time
from sklearn.model_selection import GridSearchCV

param_grid = {
    "classifier__learning_rate": (0.01, 0.1, 1, 10),  # 4 possible values
    "classifier__max_leaf_nodes": (3, 10, 30),  # 3 possible values
}  # 12 unique combinations
model_grid_search = GridSearchCV(model, param_grid=param_grid, n_jobs=2, cv=2)
model_grid_search.fit(data_train, target_train)
```

You can access the best combination of hyperparameters found by the grid
search using the `best_params_` attribute.

```python
print(f"The best set of parameters is: {model_grid_search.best_params_}")
```

Once the grid-search is fitted, it can be used as any other estimator, i.e. it
has `predict` and `score` methods. Internally, it uses the model with the
best parameters found during `fit`.

Let's get the predictions for the 5 first samples using the estimator with the
best parameters:

```python
model_grid_search.predict(data_test.iloc[0:5])
```

Finally, we check the accuracy of our model using the test set.

```python
accuracy = model_grid_search.score(data_test, target_test)
print(
    f"The test accuracy score of the grid-search pipeline is: {accuracy:.2f}"
)
```

The accuracy and the best parameters of the grid-search pipeline are similar
to the ones we found in the previous exercise, where we searched the best
parameters "by hand" through a double `for` loop.

## The need for a validation set

In the previous section, the selection of the best hyperparameters was done
using the train set, coming from the initial train-test split. Then, we
evaluated the generalization performance of our tuned model on the left out
test set. This can be shown schematically as follows:

![[Assets/Sources/Scikit-learn MOOC/figures/cross_validation_train_test_diagram.png|Cross-validation tuning
diagram]]

> [!note] This figure shows the particular case of **K-fold** cross-validation strategy
> using `n_splits=5` to further split the train set coming from a train-test
> split. For each cross-validation split, the procedure trains a model on all
> the red samples, evaluates the score of a given set of hyperparameters on the
> green samples. The best combination of hyperparameters `best_params` is selected
> based on those intermediate scores.
>
> Then a final model is refitted using `best_params` on the concatenation of the
> red and green samples and evaluated on the blue samples.
>
> The green samples are sometimes referred as the **validation set** to
> differentiate them from the final test set in blue.

In addition, we can inspect all results which are stored in the attribute
`cv_results_` of the grid-search. We filter some specific columns from these
results.

```python
cv_results = pd.DataFrame(model_grid_search.cv_results_).sort_values(
    "mean_test_score", ascending=False
)
cv_results
```

Let us focus on the most interesting columns and shorten the parameter names
to remove the `"param_classifier__"` prefix for readability:

```python
# get the parameter names
column_results = [f"param_{name}" for name in param_grid.keys()]
column_results += ["mean_test_score", "std_test_score", "rank_test_score"]
cv_results = cv_results[column_results]
```

```python
def shorten_param(param_name):
    if "__" in param_name:
        return param_name.rsplit("__", 1)[1]
    return param_name


cv_results = cv_results.rename(shorten_param, axis=1)
cv_results
```

Given that we are tuning only 2 parameters, we can visualize the results as a
heatmap. To do so, we first need to reshape the `cv_results` into a dataframe
where:

- the rows correspond to the learning-rate values;
- the columns correspond to the maximum number of leaf;
- the content of the dataframe is the mean test scores.

```python
pivoted_cv_results = cv_results.pivot_table(
    values="mean_test_score",
    index=["learning_rate"],
    columns=["max_leaf_nodes"],
)

pivoted_cv_results
```

Now that we have the data in the right format, we can create the heatmap as
follows:

```python
import seaborn as sns

ax = sns.heatmap(
    pivoted_cv_results,
    annot=True,
    cmap="YlGnBu",
    vmin=0.7,
    vmax=0.9,
    cbar_kws={"label": "mean test accuracy"},
)
ax.invert_yaxis()
```

The heatmap above shows the mean test accuracy (i.e., the average over
cross-validation splits) for each combination of hyperparameters, where darker
colors indicate better performance. However, notice that using colors only
allows us to visually compare the mean test score, but does not carry any
information on the standard deviation over splits, making it difficult to say
if different scores coming from different combinations lead to a significantly
better model or not.

The above tables highlights the following things:

* for too high values of `learning_rate`, the generalization performance of
  the model is degraded and adjusting the value of `max_leaf_nodes` cannot fix
  that problem;
* outside of this pathological region, we observe that the optimal choice of
  `max_leaf_nodes` depends on the value of `learning_rate`;
* in particular, we observe a "diagonal" of good models with an accuracy close
  to the maximal of 0.87: when the value of `max_leaf_nodes` is increased, one
  should decrease the value of `learning_rate` accordingly to preserve a good
  accuracy.

The precise meaning of those two parameters will be explained later.

For now we note that, in general, **there is no unique optimal parameter
setting**: 4 models out of the 12 parameter configurations reach the maximal
accuracy (up to small random fluctuations caused by the sampling of the
training set).

In this notebook we have seen:

* how to optimize the hyperparameters of a predictive model via a grid-search;
* that searching for more than two hyperparameters is too costly;
* that a grid-search does not necessarily find an optimal solution.
