---
title: "✅ Quiz M4.01"
type: external-resource
status: imported-source
language: en
source_kind: markdown
source_commit: 0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6
---

> [!note] Original source material
> This page preserves [`jupyter-book/linear_models/linear_models_quiz_m4_01.md`](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/jupyter-book/linear_models/linear_models_quiz_m4_01.md) from the
> [scikit-learn MOOC](https://github.com/INRIA/scikit-learn-mooc/tree/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6) at commit `0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6`.
> Course material is licensed under [CC BY 4.0](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/LICENSE).
> Bookvar changed only publication markup, local asset paths, and characters
> required for safe rendering.

> [!note] Question
> What is a linear regression?
>
> - a) a model that outputs a continuous prediction as the sum of the values of a
>   **limited** subset of the input features
> - b) a model that outputs a binary prediction based on a linear combination
>   of the values of the input features
> - c) a model that outputs a continuous prediction as a weighted sum of the input
>   features
>
> _Select a single answer_

+++

> [!note] Question
> Is it possible to get a perfect fit (zero prediction error on the training set)
> with a linear classifier **by itself** on a non-linearly separable dataset?
>
> - a) yes
> - b) no
>
> _Select a single answer_

+++

> [!note] Question
> If we fit a linear regression where `X` is a single column vector, how many
> parameters our model will be made of?
>
> - a) 1
> - b) 2
> - c) 3
>
> _Select a single answer_

+++

> [!note] Question
> If we train a scikit-learn `LinearRegression` with `X` being a single column
> vector and `y` a vector, `coef_` and `intercept_` will be respectively:
>
> - a) an array of shape (1, 1) and a number
> - b) an array of shape (1,) and an array of shape (1,)
> - c) an array of shape (1, 1) and an array of shape (1,)
> - d) an array of shape (1,) and a number
>
> _Select a single answer_

+++

> [!note] Question
> The decision boundaries of a logistic regression model:
>
> - a) split classes using only one of the input features
> - b) split classes using a combination of the input features
> - c) often have curved shapes
>
> _Select a single answer_

+++

> [!note] Question
> For a binary classification task, what is the shape of the array returned by the
> `predict_proba` method for 10 input samples?
>
> - a) (10,)
> - b) (10, 2)
> - c) (2, 10)
>
> _Select a single answer_

+++

> [!note] Question
> In logistic regression's `predict_proba` method in scikit-learn, which of the
> following statements is true regarding the predicted probabilities?
>
> - a) The sum of probabilities across different classes for a given sample is always equal to 1.0.
> - b) The sum of probabilities across all samples for a given class is always equal to 1.0.
> - c) The sum of probabilities across all features for a given class is always equal to 1.0.
>
> _Select a single answer_
