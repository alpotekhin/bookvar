---
title: "📃 Solution for Exercise M1.01"
type: external-resource
status: imported-source
language: en
source_kind: jupytext-notebook
source_commit: 0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6
---

> [!note] Original source material
> This page preserves [`python_scripts/01_tabular_data_exploration_sol_01.py`](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/python_scripts/01_tabular_data_exploration_sol_01.py) from the
> [scikit-learn MOOC](https://github.com/INRIA/scikit-learn-mooc/tree/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6) at commit `0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6`.
> Course material is licensed under [CC BY 4.0](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/LICENSE).
> Bookvar changed only publication markup, local asset paths, and characters
> required for safe rendering.

Imagine we are interested in predicting penguins species based on two of their
body measurements: culmen length and culmen depth. First we want to do some
data exploration to get a feel for the data.

What are the features? What is the target?

%% [markdown] tags=["solution"]
The features are `"culmen length"` and `"culmen depth"`. The target is the
penguin species.

The data is located in `../datasets/penguins_classification.csv`, load it with
`pandas` into a `DataFrame`.

```python
# solution
import pandas as pd

penguins = pd.read_csv("../datasets/penguins_classification.csv")
```

Show a few samples of the data.

How many features are numerical? How many features are categorical?

%% [markdown] tags=["solution"]
Both features, `"culmen length"` and `"culmen depth"` are numerical. There are
no categorical features in this dataset.

```python
# solution
penguins.head()
```

What are the different penguins species available in the dataset and how many
samples of each species are there? Hint: select the right column and use the
[`value_counts`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.value_counts.html)
method.

```python
# solution
penguins["Species"].value_counts()
```

Plot histograms for the numerical features

```python
# solution
_ = penguins.hist(figsize=(8, 4))
```

Show features distribution for each class. Hint: use
[`seaborn.pairplot`](https://seaborn.pydata.org/generated/seaborn.pairplot.html)

```python
# solution
import seaborn

pairplot_figure = seaborn.pairplot(penguins, hue="Species")

# %% [markdown] tags=["solution"]
# We observe that the labels on the axis are overlapping. Even if it is not the
# priority of this notebook, one can tweak them by increasing the height of each
# subfigure.

# %% tags=["solution"]
pairplot_figure = seaborn.pairplot(penguins, hue="Species", height=4)
```

Looking at these distributions, how hard do you think it would be to classify
the penguins only using `"culmen depth"` and `"culmen length"`?

%% [markdown] tags=["solution"]
Looking at the previous scatter-plot showing `"culmen length"` and `"culmen
depth"`, the species are reasonably well separated:
- low culmen length -> Adelie
- low culmen depth -> Gentoo
- high culmen depth and high culmen length -> Chinstrap

There is some small overlap between the species, so we can expect a
statistical model to perform well on this dataset but not perfectly.
