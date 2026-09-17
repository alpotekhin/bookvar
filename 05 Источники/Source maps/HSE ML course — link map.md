---
title: HSE ML course — current material map
type: source-note
status: active
last_verified: 2026-09-15
---

# HSE ML course — current material map

Repository:
[`esokolov/ml-course-hse`](https://github.com/esokolov/ml-course-hse/tree/4b21051531fb72dc9eef58632332ad971c92d006),
commit `4b21051531fb72dc9eef58632332ad971c92d006`.

The [[05 Источники/Courses/HSE ML course|local course archive]] now contains
67 artifacts from the two Spring 2026 tracks: 31 notebooks, 26 PDFs, and 10
Python files. They retain the source language. The import records a permission
basis separately from a repository license; no repository-wide license was
present in the inspected snapshot.

This table maps source topics to intended textbook sections. It is not a
completed-integration ledger: most destination cells below are topic labels,
not evidence that every explanation or exercise has been incorporated. Use
[[00 Учебник/01 Классическое машинное обучение/00 Карта модуля|the classical ML module]]
for the reading sequence and the local archive for the original lectures and
seminars. The [[00 Учебник/01 Классическое машинное обучение/06a Спектральная кластеризация и графовый Laplacian|spectral clustering chapter]]
already links the relevant HSE seminar and assignment; broader chapter-by-chapter
coverage still needs a separate content audit.

The repository contains several historical editions. The table below prefers
`ml1-2026-spring` and `ml2-2026-spring`; older editions are used only when they
contain a topic absent from the current folders.

## ML foundations and classical machine learning

| Topic | Theory | Seminar or implementation | Bookvar destination |
|---|---|---|---|
| Problem setting and empirical risk | [lecture01-intro.pdf](https://github.com/esokolov/ml-course-hse/blob/4b21051531fb72dc9eef58632332ad971c92d006/ml1-2026-spring/lecture-notes/lecture01-intro.pdf) | [sem01-data.ipynb](https://github.com/esokolov/ml-course-hse/blob/4b21051531fb72dc9eef58632332ad971c92d006/ml1-2026-spring/seminars/sem01-data.ipynb) | ML foundations: task, target, loss, generalization |
| Linear regression | [lecture02](https://github.com/esokolov/ml-course-hse/blob/4b21051531fb72dc9eef58632332ad971c92d006/ml1-2026-spring/lecture-notes/lecture02-linregr.pdf), [lecture03](https://github.com/esokolov/ml-course-hse/blob/4b21051531fb72dc9eef58632332ad971c92d006/ml1-2026-spring/lecture-notes/lecture03-linregr.pdf) | [sklearn linear regression](https://github.com/esokolov/ml-course-hse/blob/4b21051531fb72dc9eef58632332ad971c92d006/ml1-2026-spring/seminars/sem02-sklearn-linreg.ipynb), [gradient descent](https://github.com/esokolov/ml-course-hse/blob/4b21051531fb72dc9eef58632332ad971c92d006/ml1-2026-spring/seminars/sem03-gd.ipynb) | losses, optimization, regularization |
| Vector differentiation | [homework-theory-02](https://github.com/esokolov/ml-course-hse/blob/4b21051531fb72dc9eef58632332ad971c92d006/ml1-2026-spring/homework-theory/homework-theory-02-derivatives.pdf) | [NumPy seminar](https://github.com/esokolov/ml-course-hse/blob/4b21051531fb72dc9eef58632332ad971c92d006/ml1-2026-spring/seminars/sem03-numpy.ipynb) | derivatives, gradients, matrix calculus |
| Feature engineering and leakage | — | [sem04-features.ipynb](https://github.com/esokolov/ml-course-hse/blob/4b21051531fb72dc9eef58632332ad971c92d006/ml1-2026-spring/seminars/sem04-features.ipynb), [advanced homework](https://github.com/esokolov/ml-course-hse/blob/4b21051531fb72dc9eef58632332ad971c92d006/ml1-2026-spring/homework-practice/homework-practice-03-features/homework-practice-03-advanced.ipynb) | train/validation/test and leakage |
| Linear classification | [lectures 04–06](https://github.com/esokolov/ml-course-hse/tree/4b21051531fb72dc9eef58632332ad971c92d006/ml1-2026-spring/lecture-notes) | [classification metrics](https://github.com/esokolov/ml-course-hse/blob/4b21051531fb72dc9eef58632332ad971c92d006/ml1-2026-spring/seminars/sem05-linclass-metrics.pdf) | logistic regression, SVM, multiclass classification |
| Calibration and decision thresholds | — | [sem06-calibration.ipynb](https://github.com/esokolov/ml-course-hse/blob/4b21051531fb72dc9eef58632332ad971c92d006/ml1-2026-spring/seminars/sem06-calibration.ipynb) | evaluation: probabilities, calibration, ROC/PR |
| Decision trees | [lecture09-trees.pdf](https://github.com/esokolov/ml-course-hse/blob/4b21051531fb72dc9eef58632332ad971c92d006/ml1-2026-spring/lecture-notes/lecture09-trees.pdf) | [sem09-trees.ipynb](https://github.com/esokolov/ml-course-hse/blob/4b21051531fb72dc9eef58632332ad971c92d006/ml1-2026-spring/seminars/sem09-trees.ipynb) | classical ML: trees |
| Bias–variance decomposition | — | [sem10-bvd.pdf](https://github.com/esokolov/ml-course-hse/blob/4b21051531fb72dc9eef58632332ad971c92d006/ml1-2026-spring/seminars/sem10-bvd.pdf) | generalization and regularization |
| Bagging and boosting | [lectures 10–12](https://github.com/esokolov/ml-course-hse/tree/4b21051531fb72dc9eef58632332ad971c92d006/ml1-2026-spring/lecture-notes) | [boosting homework](https://github.com/esokolov/ml-course-hse/blob/4b21051531fb72dc9eef58632332ad971c92d006/ml1-2026-spring/homework-practice/homework-practice-06-boosting/homework_practice_06_boosting.ipynb) | classical ML: ensembles and boosting |

## Neural networks

| Topic | Theory | Seminar or assignment | Bookvar destination |
|---|---|---|---|
| MLP, computation graph, backpropagation | [lecture07-deeplearning.pdf](https://github.com/esokolov/ml-course-hse/blob/4b21051531fb72dc9eef58632332ad971c92d006/ml1-2026-spring/lecture-notes/lecture07-deeplearning.pdf) | [sem07-torch-mlp.ipynb](https://github.com/esokolov/ml-course-hse/blob/4b21051531fb72dc9eef58632332ad971c92d006/ml1-2026-spring/seminars/sem07-torch-mlp.ipynb) | neural networks: MLP and backpropagation |
| CNN | [homework-theory-04-cnn.pdf](https://github.com/esokolov/ml-course-hse/blob/4b21051531fb72dc9eef58632332ad971c92d006/ml1-2026-spring/homework-theory/homework-theory-04-cnn.pdf) | [sem08-torch-cnn.ipynb](https://github.com/esokolov/ml-course-hse/blob/4b21051531fb72dc9eef58632332ad971c92d006/ml1-2026-spring/seminars/sem08-torch-cnn.ipynb) | neural networks: convolution and CNN |
| End-to-end PyTorch practice | — | [homework_practice_04_dl.ipynb](https://github.com/esokolov/ml-course-hse/blob/4b21051531fb72dc9eef58632332ad971c92d006/ml1-2026-spring/homework-practice/homework_practice_04_dl.ipynb) | practice track after MLP/CNN |

## Unsupervised learning, kernels and recommendations

| Topic | Source | Practice | Bookvar destination |
|---|---|---|---|
| Clustering | [sem12-clustering.pdf](https://github.com/esokolov/ml-course-hse/blob/4b21051531fb72dc9eef58632332ad971c92d006/ml1-2026-spring/seminars/sem12-clustering.pdf) | [sem12-clustering.ipynb](https://github.com/esokolov/ml-course-hse/blob/4b21051531fb72dc9eef58632332ad971c92d006/ml1-2026-spring/seminars/sem12-clustering.ipynb) | classical ML: clustering |
| PCA and t-SNE | [sem13-pca.pdf](https://github.com/esokolov/ml-course-hse/blob/4b21051531fb72dc9eef58632332ad971c92d006/ml1-2026-spring/seminars/sem13-pca.pdf) | [sem13-pca-tsne.ipynb](https://github.com/esokolov/ml-course-hse/blob/4b21051531fb72dc9eef58632332ad971c92d006/ml1-2026-spring/seminars/sem13-pca-tsne.ipynb) | representation and dimensionality reduction |
| Graph clustering | [sem03-graph-clustering.pdf](https://github.com/esokolov/ml-course-hse/blob/4b21051531fb72dc9eef58632332ad971c92d006/ml2-2026-spring/seminars/sem03-graph-clustering.pdf) | [sem04-graph.ipynb](https://github.com/esokolov/ml-course-hse/blob/4b21051531fb72dc9eef58632332ad971c92d006/ml2-2026-spring/seminars/sem04-graph.ipynb) | classical ML: spectral and graph methods |
| Recommender systems | — | [homework-practice-10](https://github.com/esokolov/ml-course-hse/blob/4b21051531fb72dc9eef58632332ad971c92d006/ml2-2026-spring/homeworks-practice/homework-practice-10-recommendations/homework-practice-10-recommendations.ipynb) | recommendation systems |
| Random Fourier features and kernel regression | — | [homework-practice-11](https://github.com/esokolov/ml-course-hse/blob/4b21051531fb72dc9eef58632332ad971c92d006/ml2-2026-spring/homeworks-practice/homework-practice-11-random-features/homework-practice-11-random-features.ipynb) | kernels and scalable approximations |
| EM | — | [homework-practice-12](https://github.com/esokolov/ml-course-hse/blob/4b21051531fb72dc9eef58632332ad971c92d006/ml2-2026-spring/homeworks-practice/homework-practice-12-em/homework-practice-12-em.ipynb) | latent-variable models |
| Imbalanced learning | — | [homework-practice-09](https://github.com/esokolov/ml-course-hse/blob/4b21051531fb72dc9eef58632332ad971c92d006/ml2-2026-spring/homeworks-practice/homework-practice-09-imbalanced.ipynb) | evaluation and class imbalance |

## Editorial use

The HSE material should provide the classical-ML spine that precedes the
neural-network and NLP sections. It should not be scattered as isolated links
inside unrelated LLM chapters. Each future Bookvar chapter should reference
the theory document, a worked seminar, and an assignment together.
