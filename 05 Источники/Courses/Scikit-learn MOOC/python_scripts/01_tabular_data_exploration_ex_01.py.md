---
title: "📝 Exercise M1.01"
type: external-resource
status: imported-source
language: en
source_kind: jupytext-notebook
source_commit: 0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6
---

> [!note] Original source material
> This page preserves [`python_scripts/01_tabular_data_exploration_ex_01.py`](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/python_scripts/01_tabular_data_exploration_ex_01.py) from the
> [scikit-learn MOOC](https://github.com/INRIA/scikit-learn-mooc/tree/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6) at commit `0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6`.
> Course material is licensed under [CC BY 4.0](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/LICENSE).
> Bookvar changed only publication markup, local asset paths, and characters
> required for safe rendering.

Imagine we are interested in predicting penguins species based on two of their
body measurements: culmen length and culmen depth. First we want to do some
data exploration to get a feel for the data.

What are the features? What is the target?

The data is located in `../datasets/penguins_classification.csv`, load it with
`pandas` into a `DataFrame`.

```python
# Write your code here.
```

Show a few samples of the data.

How many features are numerical? How many features are categorical?

```python
# Write your code here.
```

What are the different penguins species available in the dataset and how many
samples of each species are there? Hint: select the right column and use the
[`value_counts`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.value_counts.html)
method.

```python
# Write your code here.
```

Plot histograms for the numerical features

```python
# Write your code here.
```

Show features distribution for each class. Hint: use
[`seaborn.pairplot`](https://seaborn.pydata.org/generated/seaborn.pairplot.html)

```python
# Write your code here.
```

Looking at these distributions, how hard do you think it would be to classify
the penguins only using `"culmen depth"` and `"culmen length"`?
