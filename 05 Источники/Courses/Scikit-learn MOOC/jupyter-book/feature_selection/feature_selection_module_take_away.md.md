---
title: "Main take-away"
type: external-resource
status: imported-source
language: en
source_kind: markdown
source_commit: 0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6
---

> [!note] Original source material
> This page preserves [`jupyter-book/feature_selection/feature_selection_module_take_away.md`](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/jupyter-book/feature_selection/feature_selection_module_take_away.md) from the
> [scikit-learn MOOC](https://github.com/INRIA/scikit-learn-mooc/tree/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6) at commit `0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6`.
> Course material is licensed under [CC BY 4.0](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/LICENSE).
> Bookvar changed only publication markup, local asset paths, and characters
> required for safe rendering.

## Wrap-up

<!-- Quick wrap-up for the module -->

In this module, we presented the principle of feature selection. In short,
feature selection is not a magical tool to get marginal gains. We tackle
the following aspects:

- you should use feature selection to speed-up training and testing rather
  than seeking for marginal performance gains;
- you should be careful regarding the framework and how to include a feature
  selector within your pipeline;
- you should be aware of the limitation of a feature selector based on
  machine-learning models.

## To go further

<!-- Some extra links of content to go further -->

You can refer to the following scikit-learn examples which are related to
the concepts approached during this module:

- [Recursive feature selection using cross-validation](https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html#sphx-glr-auto-examples-feature-selection-plot-rfe-with-cross-validation-py)
