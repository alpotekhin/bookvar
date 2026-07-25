---
title: "✅ Quiz M3.01"
type: external-resource
status: imported-source
language: en
source_kind: markdown
source_commit: 0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6
---

> [!note] Original source material
> This page preserves [`jupyter-book/tuning/parameter_tuning_manual_quiz_m3_01.md`](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/jupyter-book/tuning/parameter_tuning_manual_quiz_m3_01.md) from the
> [scikit-learn MOOC](https://github.com/INRIA/scikit-learn-mooc/tree/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6) at commit `0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6`.
> Course material is licensed under [CC BY 4.0](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/LICENSE).
> Bookvar changed only publication markup, local asset paths, and characters
> required for safe rendering.

> [!note] Question
> Which parameters below are hyperparameters of `HistGradientBoostingClassifier`?
> Remember we only consider hyperparameters to be those that potentially impact
> the result of the learning procedure and subsequent predictions.
>
> - a) `C`
> - b) `max_leaf_nodes`
> - c) `verbose`
> - d) `classes_`
> - e) `learning_rate`
>
> _Select all answers that apply_

+++

`> [!note] Question
> Given an instance named `model` as defined by:python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
```

how do you get the value of the `C` parameter?
- a) `model.get_parameters()['C']`
- b) `model.get_params()['C']`
- c) `model.get_params('C')`
- d) `model.get_params['C']`

_Select a single answer_
````

+++

`> [!note] Question
> Given `model` defined by:python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
```

how do you set the value of the `C` parameter to `5`?
- a) `model.set_params('C', 5)`
- b) `model.set_params({'C': 5})`
- c) `model.set_params()['C'] = 5`
- d) `model.set_params(C=5)`

_Select a single answer_
````

+++

`> [!note] Question
> Given `model` defined by:python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])
```

how do you set the value of the `C` parameter of the `LogisticRegression` component to 5:
- a) `model.set_params(C=5) `
- b) `model.set_params(logisticregression__C=5)`
- c) `model.set_params(classifier__C=5) `
- d) `model.set_params(classifier--C=5)`

_Select a single answer_
````
