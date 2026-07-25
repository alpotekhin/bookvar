---
title: "✅ Quiz"
type: external-resource
status: imported-source
language: en
source_kind: markdown
source_commit: 0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6
---

> [!note] Original source material
> This page preserves [`jupyter-book/interpretation/interpretation_quiz.md`](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/jupyter-book/interpretation/interpretation_quiz.md) from the
> [scikit-learn MOOC](https://github.com/INRIA/scikit-learn-mooc/tree/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6) at commit `0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6`.
> Course material is licensed under [CC BY 4.0](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/LICENSE).
> Bookvar changed only publication markup, local asset paths, and characters
> required for safe rendering.

> [!note] Question
> With a same dataset, feature importance might differs if:
>
> - a) we use two different models
> - b) we use two different train/test split with a same model
> - c) we use a same model with a different set of hyper-parameters
> - d) we use a same model with the same set of hyper-parameters but a different
>   random_state

+++

> [!note] Question
> In linear model, the feature importance:
>
> - a) might be infer from the coefficients
> - b) might be infer by `importance_permutation`
> - c) need a regularization to infer the importance
> - d) is a built-in attribute

+++

> [!note] Question
> If two feature are the same (thus correlated)
>
> - a) their feature importance will be the same
> - b) their feature importance will be divided by 2
> - c) only one will receive all the feature importance, the second one will be 0
> - d) it depends

+++

> [!note] Question
> The feature importance provided by the scikit-learn random forest:
>
> - a) has bias for categorical feature
> - b) has bias for continuous (high cardinality) feature
> - c) is independent from the train/test split
> - d) is independent from the hyper-parameters

+++

> [!note] Question
> To evaluate the feature importance for a specific model, one could:
>
> - a) drop a column and compare the score
> - b) shuffle a column and compare the score
> - c) put all column to 0 and compare the score
> - d) change a column value to random number and compare the score
