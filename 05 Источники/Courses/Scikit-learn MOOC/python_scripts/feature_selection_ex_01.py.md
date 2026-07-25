---
title: "📝 Exercise 01"
type: external-resource
status: imported-source
language: en
source_kind: jupytext-notebook
source_commit: 0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6
---

> [!note] Original source material
> This page preserves [`python_scripts/feature_selection_ex_01.py`](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/python_scripts/feature_selection_ex_01.py) from the
> [scikit-learn MOOC](https://github.com/INRIA/scikit-learn-mooc/tree/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6) at commit `0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6`.
> Course material is licensed under [CC BY 4.0](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/LICENSE).
> Bookvar changed only publication markup, local asset paths, and characters
> required for safe rendering.

The aim of this exercise is to highlight caveats to have in mind when using
feature selection. You have to be extremely careful regarding the set of data
on which you will compute the statistic that helps your feature selection
algorithm to decide which feature to select.

On purpose, we will make you program the wrong way of doing feature selection
to gain insights.

First, you will create a completely random dataset using NumPy. Using the
function `np.random.randn`, generate a matrix `data` containing 100 samples
and 100,000 features. Then, using the function `np.random.randint`, generate a
vector `target` with 100 samples containing either 0 or 1.

This type of dimensionality is typical in bioinformatics when dealing with
RNA-seq. However, we will use completely randomized features such that we
don't have a link between the data and the target. Thus, the generalization
performance of any machine-learning model should not perform better than the
chance-level.

```python
import numpy as np

# Write your code here.
```

Now, create a logistic regression model and use cross-validation to check the
score of such a model. It will allow use to confirm that our model cannot
predict anything meaningful from random data.

```python
# Write your code here.
```

Now, we will ask you to program the **wrong** pattern to select feature.
Select the feature by using the entire dataset. We will choose ten features
with the highest ANOVA F-score computed on the full dataset. Subsequently,
subsample the dataset `data` by selecting the features' subset. Finally, train
and test a logistic regression model.

You should get some surprising results.

```python
from sklearn.feature_selection import SelectKBest, f_classif

# Write your code here.
```

Now, we will make you program the **right** way to do the feature selection.
First, split the dataset into a training and testing set. Then, fit the
feature selector on the training set. Then, transform both the training and
testing sets before you train and test the logistic regression.

```python
from sklearn.model_selection import train_test_split

# Write your code here.
```

However, the previous case is not perfect. For instance, if we were asking to
perform cross-validation, the manual `fit`/`transform` of the datasets will
make our life hard. Indeed, the solution here is to use a scikit-learn
pipeline in which the feature selection will be a pre processing stage before
to train the model.

Thus, start by creating a pipeline with the feature selector and the logistic
regression. Then, use cross-validation to get an estimate of the uncertainty
of your model generalization performance.

```python
from sklearn.pipeline import make_pipeline

# Write your code here.
```
