---
title: "Module overview"
type: external-resource
status: imported-source
language: en
source_kind: markdown
source_commit: 0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6
---

> [!note] Original source material
> This page preserves [`jupyter-book/overfit/overfit_module_intro.md`](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/jupyter-book/overfit/overfit_module_intro.md) from the
> [scikit-learn MOOC](https://github.com/INRIA/scikit-learn-mooc/tree/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6) at commit `0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6`.
> Course material is licensed under [CC BY 4.0](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/LICENSE).
> Bookvar changed only publication markup, local asset paths, and characters
> required for safe rendering.

## What you will learn

<!-- Give in plain English what the module is about -->

This module gives an intuitive introduction to the very **fundamental
concepts** of overfitting and underfitting in machine learning.

Machine learning models can never make perfect predictions: the test error is
never exactly zero. This failure comes from a **fundamental trade-off** between
**modeling flexibility** and the **limited size of the training dataset**.

The first presentation will define those problems and characterize how and why
they arise.

Then we will present a methodology to quantify those problems by **contrasting
the train error with the test error** for various choice of the model family,
model parameters. More importantly, we will emphasize the **impact of the size
of the training set on this trade-off**.

Finally we will relate overfitting and underfitting to the concepts of
statistical variance and bias.

## Before getting started

<!-- Give the required skills for the module -->

The required technical skills to carry on this module are:

- skills acquired during the "The Predictive Modeling Pipeline" module with
  basic usage of scikit-learn.

<!-- Point to resources to learning these skills -->

## Objectives and time schedule

<!-- Give the learning objectives -->

The objective in the module are the following:

- understand the concept of overfitting and underfitting;
- understand the concept of generalization;
- understand the general cross-validation framework used to evaluate a model.

<!-- Give the investment in time -->

The estimated time to go through this module is about 3 hours.
