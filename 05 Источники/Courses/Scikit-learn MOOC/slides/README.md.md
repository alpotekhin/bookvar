---
title: "View slides"
type: external-resource
status: imported-source
language: en
source_kind: markdown
source_commit: 0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6
---

> [!note] Original source material
> This page preserves [`slides/README.md`](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/slides/README.md) from the
> [scikit-learn MOOC](https://github.com/INRIA/scikit-learn-mooc/tree/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6) at commit `0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6`.
> Course material is licensed under [CC BY 4.0](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/LICENSE).
> Bookvar changed only publication markup, local asset paths, and characters
> required for safe rendering.

## On the .github.io website

The general pattern is `https://inria.github.io/scikit-learn-mooc/slides/?file=[FILENAME].md`

Example for ML concepts slides:
https://inria.github.io/scikit-learn-mooc/slides/?file=ml_concepts.md

## Locally

Useful when working on the slides:

```py
# on the root repo folder
python -m http.server

# open your browser with the right port (from previous command) using the right md file
firefox 'http://localhost:8000/slides/index.html?file=../slides/ml_concepts.md'
```
