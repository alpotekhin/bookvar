---
title: "✅ Quiz M4.03"
type: external-resource
status: imported-source
language: en
source_kind: markdown
source_commit: 0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6
---

> [!note] Original source material
> This page preserves [`jupyter-book/linear_models/linear_models_quiz_m4_03.md`](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/jupyter-book/linear_models/linear_models_quiz_m4_03.md) from the
> [scikit-learn MOOC](https://github.com/INRIA/scikit-learn-mooc/tree/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6) at commit `0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6`.
> Course material is licensed under [CC BY 4.0](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/LICENSE).
> Bookvar changed only publication markup, local asset paths, and characters
> required for safe rendering.

> [!note] Question
> Which of the following estimators can solve linear regression problems?
>
> - a) sklearn.linear_model.LinearRegression
> - b) sklearn.linear_model.LogisticRegression
> - c) sklearn.linear_model.Ridge
>
> _Select all answers that apply_

+++

> [!note] Question
> Regularization allows:
>
> - a) to create a model robust to outliers (samples that differ widely from
>   other observations)
> - b) to reduce overfitting by forcing the weights to stay close to zero
> - c) to reduce underfitting by making the problem linearly separable
>
> _Select a single answer_

+++

> [!note] Question
> A ridge model is:
>
> - a) the same as linear regression with penalized weights
> - b) the same as logistic regression with penalized weights
> - c) a linear model
> - d) a non linear model
>
> _Select all answers that apply_

+++

> [!note] Question
> Assume that a data scientist has prepared a train/test split and plans to use
> the test for the final evaluation of a `Ridge` model. The parameter `alpha` of
> the `Ridge` model:
>
> - a) is internally tuned when calling `fit` on the train set
> - b) should be tuned by running cross-validation on a **train set**
> - c) should be tuned by running cross-validation on a **test set**
> - d) must be a positive number
>
> _Select all answers that apply_

+++

> [!note] Question
> Scaling the data before fitting a model:
>
> - a) is often useful for regularized linear models
> - b) is always necessary for regularized linear models
> - c) may speed-up fitting
> - d) has no impact on the optimal choice of the value of a regularization parameter
>
> _Select all answers that apply_

+++

> [!note] Question
> The effect of increasing the regularization strength in a ridge model is to:
>
> - a) shrink all weights towards zero
> - b) make all weights equal
> - c) set a subset of the weights to exactly zero
> - d) constrain all the weights to be positive
>
> _Select all answers that apply_

+++

> [!note] Question
> By default, a [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) in scikit-learn applies:
>
> - a) no penalty
> - b) a penalty that shrinks the magnitude of the weights towards zero (also called "l2 penalty")
> - c) a penalty that ensures all weights are equal
>
> _Select a single answer_

+++

> [!note] Question
> The parameter `C` in a logistic regression is:
>
> - a) similar to the parameter `alpha` in a ridge regressor
> - b) similar to `1 / alpha` where `alpha` is the parameter of a ridge regressor
> - c) not controlling the regularization
>
> _Select a single answer_

+++

> [!note] Question
> In logistic regression, increasing the regularization strength (by
> decreasing the value of `C`) makes the model:
>
> - a) more likely to overfit to the training data
> - b) more confident: the values returned by `predict_proba` are closer to 0 or 1
> - c) less complex, potentially underfitting the training data
>
> _Select a single answer_
