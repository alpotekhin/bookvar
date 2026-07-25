---
title: "✅ Quiz Intro.01"
type: external-resource
status: imported-source
language: en
source_kind: markdown
source_commit: 0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6
---

> [!note] Original source material
> This page preserves [`jupyter-book/ml_concepts/quiz_intro_01.md`](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/jupyter-book/ml_concepts/quiz_intro_01.md) from the
> [scikit-learn MOOC](https://github.com/INRIA/scikit-learn-mooc/tree/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6) at commit `0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6`.
> Course material is licensed under [CC BY 4.0](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/LICENSE).
> Bookvar changed only publication markup, local asset paths, and characters
> required for safe rendering.

Given a case study: pricing apartments based on a real estate website. We have
thousands of house descriptions with their price. Typically, an example of a
house description is the following:

"Great for entertaining: spacious, updated 2 bedroom, 1 bathroom apartment in
Lakeview, 97630. The house will be available from May 1st. Close to nightlife
with private backyard. Price ~$1,000,000."

We are interested in predicting house prices from their description. One
potential use case for this would be, as a buyer, to find houses that are cheap
compared to their market value.

> [!note] Question
> What kind of problem is it?
>
> - a) a supervised problem
> - b) an unsupervised problem
> - c) a classification problem
> - d) a regression problem
>
> _Select all answers that apply_

+++

> [!note] Question
> What are the features?
>
> - a) the number of rooms might be a feature
> - b) the post code of the house might be a feature
> - c) the price of the house might be a feature
>
> _Select all answers that apply_

+++

> [!note] Question
> What is the target variable?
>
> - a) the full text description is the target
> - b) the price of the house is the target
> - c) only house description with no price mentioned are the target
>
> _Select a single answer_

+++

> [!note] Question
> What is a record (a sample)?
>
> - a) each house description is a record
> - b) each house price is a record
> - c) each kind of description (as the house size) is a record
>
> _Select a single answer_
