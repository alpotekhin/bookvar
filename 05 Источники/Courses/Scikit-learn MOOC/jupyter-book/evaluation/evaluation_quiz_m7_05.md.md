---
title: "✅ Quiz M7.05"
type: external-resource
status: imported-source
language: en
source_kind: markdown
source_commit: 0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6
---

> [!note] Original source material
> This page preserves [`jupyter-book/evaluation/evaluation_quiz_m7_05.md`](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/jupyter-book/evaluation/evaluation_quiz_m7_05.md) from the
> [scikit-learn MOOC](https://github.com/INRIA/scikit-learn-mooc/tree/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6) at commit `0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6`.
> Course material is licensed under [CC BY 4.0](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/LICENSE).
> Bookvar changed only publication markup, local asset paths, and characters
> required for safe rendering.

> [!note] Question
> What is the default score in scikit-learn when using a regressor?
>
> - a) $R^2$
> - b) mean absolute error
> - c) median absolute error
>
> _Select a single answer_

+++

> [!note] Question
> If we observe that the values returned by
> `cross_val_scores(model, X, y, scoring="r2")` increase after changing the model
> parameters, it means that the latest model:
>
> - a) generalizes better
> - b) generalizes worse
>
> _Select a single answer_

+++

> [!note] Question
> If all the values returned by
> `cross_val_score(model_A, X, y, scoring="neg_mean_squared_error")`
> are strictly lower than those returned by
> `cross_val_score(model_B, X, y, scoring="neg_mean_squared_error")`
> it means that `model_B` generalizes:
>
> - a) better than `model_A`
> - b) worse than `model_A`
>
> Hint: Remember that `"neg_mean_squared_error"` is an alias for the negative of
> the Mean Squared Error.
>
> _Select a single answer_

+++

> [!note] Question
> Values returned by `cross_val_scores(model, X, y, scoring="neg_mean_squared_error")`
> are:
>
> - a) guaranteed to be positive or zero
> - b) guaranteed to be negative or zero
> - c) can be either positive or negative depending on the data
>
> _Select a single answer_
