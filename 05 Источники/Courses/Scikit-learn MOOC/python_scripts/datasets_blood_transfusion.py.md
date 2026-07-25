---
title: "The blood transfusion dataset"
type: external-resource
status: imported-source
language: en
source_kind: jupytext-notebook
source_commit: 0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6
---

> [!note] Original source material
> This page preserves [`python_scripts/datasets_blood_transfusion.py`](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/python_scripts/datasets_blood_transfusion.py) from the
> [scikit-learn MOOC](https://github.com/INRIA/scikit-learn-mooc/tree/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6) at commit `0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6`.
> Course material is licensed under [CC BY 4.0](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/LICENSE).
> Bookvar changed only publication markup, local asset paths, and characters
> required for safe rendering.

In this notebook, we will present the "blood transfusion" dataset. This
dataset is locally available in the directory `datasets` and it is stored as a
comma separated value (CSV) file. We start by loading the entire dataset.

```python
import pandas as pd

blood_transfusion = pd.read_csv("../datasets/blood_transfusion.csv")
```

We can have a first look at the at the dataset loaded.

```python
blood_transfusion.head()
```

In this dataframe, we can see that the last column correspond to the target to
be predicted called `"Class"`. We will create two variables, `data` and
`target` to separate the data from which we could learn a predictive model and
the `target` that should be predicted.

```python
data = blood_transfusion.drop(columns="Class")
target = blood_transfusion["Class"]
```

Let's have a first look at the `data` variable.

```python
data.head()
```

We observe four columns. Each record corresponds to a person that intended to
give blood. The information stored in each column are:

* `Recency`: the time in months since the last time a person intended to give
  blood;
* `Frequency`: the number of time a person intended to give blood in the past;
* `Monetary`: the amount of blood given in the past (in cm³);
* `Time`: the time in months since the first time a person intended to give
  blood.

Now, let's have a look regarding the type of data that we are dealing in these
columns and if any missing values are present in our dataset.

```python
data.info()
```

Our dataset is made of 748 samples. All features are represented with integer
numbers and there is no missing values. We can have a look at each feature
distributions.

```python
_ = data.hist(figsize=(12, 10), bins=30, edgecolor="black")
```

There is nothing shocking regarding the distributions. We only observe a high
value range for the features `"Recency"`, `"Frequency"`, and `"Monetary"`. It
means that we have a few extreme high values for these features.

Now, let's have a look at the target that we would like to predict for this
task.

```python
target.head()
```

```python
import matplotlib.pyplot as plt

target.value_counts(normalize=True).plot.barh()
plt.xlabel("Number of samples")
_ = plt.title("Class distribution")
```

We see that the target is discrete and contains two categories: whether a
person `"donated"` or `"not donated"` his/her blood. Thus the task to be
solved is a classification problem. We should note that the class counts of
these two classes is different.

```python
target.value_counts(normalize=True)
```

Indeed, ~76% of the samples belong to the class `"not donated"`. It is rather
important: a classifier that would predict always this `"not donated"` class
would achieve an accuracy of 76% of good classification without using any
information from the data itself. This issue is known as class imbalance. One
should take care about the generalization performance metric used to evaluate
a model as well as the predictive model chosen itself.

Now, let's have a naive analysis to see if there is a link between features
and the target using a pair plot representation.

```python
import seaborn as sns

_ = sns.pairplot(blood_transfusion, hue="Class")
```

Looking at the diagonal plots, we don't see any feature that individually
could help at separating the two classes. When looking at a pair of feature,
we don't see any striking combinations as well. However, we can note that the
`"Monetary"` and `"Frequency"` features are perfectly correlated: all the data
points are aligned on a diagonal.

As a conclusion, this dataset would be a challenging dataset: it suffer from
class imbalance, correlated features and thus very few features will be
available to learn a model, and none of the feature combinations were found to
help at predicting.
