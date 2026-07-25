---
title: "Main take-away"
type: external-resource
status: imported-source
language: en
source_kind: markdown
source_commit: 0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6
---

> [!note] Original source material
> This page preserves [`jupyter-book/ensemble/ensemble_module_take_away.md`](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/jupyter-book/ensemble/ensemble_module_take_away.md) from the
> [scikit-learn MOOC](https://github.com/INRIA/scikit-learn-mooc/tree/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6) at commit `0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6`.
> Course material is licensed under [CC BY 4.0](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/LICENSE).
> Bookvar changed only publication markup, local asset paths, and characters
> required for safe rendering.

## Wrap-up

<!-- Quick wrap-up for the module -->

So in this module, we discussed ensemble learners which are a type of
learner that combines simpler learners together. We saw two strategies:

- one based on bootstrap samples allowing learners to be fit in a parallel
  manner;
- the other called boosting which fit learners sequentially.

From these two families, we mainly focused on giving intuitions regarding the
internal machinery of the random forest and gradient-boosting models which
are state-of-the-art methods.

## To go further

<!-- Some extra links of content to go further -->

You can refer to the following scikit-learn examples which are related to
the concepts approached in this module:

- [Early-stopping in gradient-boosting](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_early_stopping.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-early-stopping-py)
- [Combining predictors using stacking](https://scikit-learn.org/stable/auto_examples/ensemble/plot_stack_predictors.html#sphx-glr-auto-examples-ensemble-plot-stack-predictors-py)
