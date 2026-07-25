---
title: "Working with numerical data"
type: external-resource
status: imported-source
language: en
source_kind: jupytext-notebook
source_commit: 0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6
---

> [!note] Original source material
> This page preserves [`python_scripts/02_numerical_pipeline_hands_on.py`](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/python_scripts/02_numerical_pipeline_hands_on.py) from the
> [scikit-learn MOOC](https://github.com/INRIA/scikit-learn-mooc/tree/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6) at commit `0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6`.
> Course material is licensed under [CC BY 4.0](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/LICENSE).
> Bookvar changed only publication markup, local asset paths, and characters
> required for safe rendering.

In the previous notebook, we trained a k-nearest neighbors model on some data.

However, we oversimplified the procedure by loading a dataset that contained
exclusively numerical data. Besides, we used datasets which were already split
into train-test sets.

In this notebook, we aim at:

* identifying numerical data in a heterogeneous dataset;
* selecting the subset of columns corresponding to numerical data;
* using a scikit-learn helper to separate data into train-test sets;
* training and evaluating a more complex scikit-learn model.

We start by loading the adult census dataset used during the data exploration.

## Loading the entire dataset

As in the previous notebook, we rely on pandas to open the CSV file into a
pandas dataframe.

```python
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")
# drop the duplicated column `"education-num"` as stated in the first notebook
adult_census = adult_census.drop(columns="education-num")
adult_census
```

The next step separates the target from the data. We performed the same
procedure in the previous notebook.

```python
data, target = adult_census.drop(columns="class"), adult_census["class"]
```

```python
data
```

```python
target
```

> [!note] Here and later, we use the name `data` and `target` to be explicit. In
> scikit-learn documentation, `data` is commonly named `X` and `target` is
> commonly called `y`.

At this point, we can focus on the data we want to use to train our predictive
model.

## Identify numerical data

Numerical data are represented with numbers. They are linked to measurable
(quantitative) data, such as age or the number of hours a person works a week.

Predictive models are natively designed to work with numerical data. Moreover,
numerical data usually requires very little work before getting started with
training.

The first task here is to identify numerical data in our dataset.

> [!warning] Numerical data are represented with numbers, but numbers do not always
> represent numerical data. Categories could already be encoded with
> numbers and you may need to identify these features.

Thus, we can check the data type for each of the column in the dataset.

```python
data.dtypes
```

We seem to have only two data types: `int64` and `object`. We can make sure by
checking for unique data types.

```python
data.dtypes.unique()
```

Indeed, the only two types in the dataset are integer `int64` and `object`. We
can look at the first few lines of the dataframe to understand the meaning of
the `object` data type.

```python
data
```

We see that the `object` data type corresponds to columns containing strings.
As we saw in the exploration section, these columns contain categories and we
will see later how to handle those. We can select the columns containing
integers and check their content.

```python
numerical_columns = ["age", "capital-gain", "capital-loss", "hours-per-week"]
data[numerical_columns]
```

Now that we limited the dataset to numerical columns only, we can analyse
these numbers to figure out what they represent. We can identify two types of
usage.

The first column, `"age"`, is self-explanatory. We can note that the values
are continuous, meaning they can take up any number in a given range. Let's
find out what this range is:

```python
data["age"].describe()
```

We can see the age varies between 17 and 90 years.

We could extend our analysis and we would find that `"capital-gain"`,
`"capital-loss"`, and `"hours-per-week"` are also representing quantitative
data.

Now, we store the subset of numerical columns in a new dataframe.

```python
data_numeric = data[numerical_columns]
```

## Train-test split the dataset

In the previous notebook, we loaded two separate datasets: a training one and
a testing one. However, having separate datasets in two distincts files is
unusual: most of the time, we have a single file containing all the data that
we need to split once loaded in the memory.

Scikit-learn provides the helper function
`sklearn.model_selection.train_test_split` which is used to automatically
split the dataset into two subsets.

```python
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data_numeric, target, random_state=42, test_size=0.25
)
```

> [!tip] In scikit-learn setting the `random_state` parameter allows to get
> deterministic results when we use a random number generator. In the
> `train_test_split` case the randomness comes from shuffling the data, which
> decides how the dataset is split into a train and a test set).

When calling the function `train_test_split`, we specified that we would like
to have 25% of samples in the testing set while the remaining samples (75%)
are assigned to the training set. We can check quickly if we got what we
expected.

```python
print(
    f"Number of samples in testing: {data_test.shape[0]} => "
    f"{data_test.shape[0] / data_numeric.shape[0] * 100:.1f}% of the"
    " original set"
)
```

```python
print(
    f"Number of samples in training: {data_train.shape[0]} => "
    f"{data_train.shape[0] / data_numeric.shape[0] * 100:.1f}% of the"
    " original set"
)
```

In the previous notebook, we used a k-nearest neighbors model. While this
model is intuitive to understand, it is not widely used in practice. Now, we
use a more useful model, called a logistic regression, which belongs to the
linear models family.

> [!note] In short, linear models find a set of weights to combine features linearly
> and predict the target. For instance, the model can come up with a rule such
> as:
> * if `0.1 * age + 3.3 * hours-per-week - 15.1 > 0`, predict `high-income`
> * otherwise predict `low-income`
>
> Linear models, and in particular the logistic regression, will be covered
> more in detail in the "Linear models" module later in this course. For now the
> focus is to use this logistic regression model in scikit-learn rather than
> understand how it works in details.

To create a logistic regression model in scikit-learn you can do:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
```

Now that the model has been created, you can use it exactly the same way as we
used the k-nearest neighbors model in the previous notebook. In particular, we
can use the `fit` method to train the model using the training data and
labels:

```python
model.fit(data_train, target_train)
```

We can also use the `score` method to check the model generalization
performance on the test set.

```python
accuracy = model.score(data_test, target_test)
print(f"Accuracy of logistic regression: {accuracy:.3f}")
```

## Notebook recap

In scikit-learn, the `score` method of a classification model returns the
accuracy, i.e. the fraction of correctly classified samples. In this case,
around 8 / 10 of the times the logistic regression predicts the right income
of a person. Now the real question is: is this generalization performance
relevant of a good predictive model? Find out by solving the next exercise!

In this notebook, we learned to:

* identify numerical data in a heterogeneous dataset;
* select the subset of columns corresponding to numerical data;
* use the scikit-learn `train_test_split` function to separate data into a
  train and a test set;
* train and evaluate a logistic regression model.
