---
title: "✅ Quiz M4.01"
type: external-resource
status: imported-source
language: en
source_kind: markdown
source_commit: 0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6
---

> [!note] Original source material
> This page preserves [`jupyter-book/clustering/clustering_quiz_m4_01.md`](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/jupyter-book/clustering/clustering_quiz_m4_01.md) from the
> [scikit-learn MOOC](https://github.com/INRIA/scikit-learn-mooc/tree/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6) at commit `0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6`.
> Course material is licensed under [CC BY 4.0](https://github.com/INRIA/scikit-learn-mooc/blob/0cc70c41c640578c28eb7fd38d8c1c19e09c8aa6/LICENSE).
> Bookvar changed only publication markup, local asset paths, and characters
> required for safe rendering.

> [!note] Question
> Imagine you work for a music streaming platform that hosts a vast library of
> songs, playlists, and podcasts. You have access to detailed listening data from
> millions of users. For each user, you know their most-listened genres, the
> devices they use, their average session length, and how often they explore new
> content.
>
> You want to segment users based on their listening patterns to improve
> personalized recommendations, without relying on rigid, predefined labels like
> "pop fan" or "casual listener" which may fail to capture the complexity of
> their behavior.
>
> What kind of problem are you dealing with?
>
> - a) a supervised task
> - b) an unsupervised task
> - c) a classification task
> - d) a clustering task
>
> _Select all answers that apply_

+++

> [!note] Question
> The plots below show the cluster labels as found by k-means with 3 clusters, only
> differing in the scaling step. Based on this, which conclusions can be obtained?
>
> ![[Assets/Sources/Scikit-learn MOOC/figures/clustering_quiz_kmeans_not_scaled.svg|K-means on original features]]
> ![[Assets/Sources/Scikit-learn MOOC/figures/clustering_quiz_kmeans_scaled.svg|K-means on scaled features]]
>
> - a) without scaling, cluster assignment is dominated by the feature in the vertical axis
> - b) without scaling, cluster assignment is dominated by the feature in the horizontal axis
> - c) without scaling, both features contribute equally to cluster assignment
>
> _Select a single answer_

+++

> [!note] Question
> Which of the following statements correctly describe factors that affect the
> stability of k-means clustering across different resampling iterations of the data?
>
> - a) K-means can produce different results on resampled datasets due to
>   sensitivity to initialization.
> - b) If data is unevenly distributed, the stability improves when increasing the
>   parameter `n_init` in the "k-means++" initialization.
> - c) Stability under resampling is guaranteed after feature scaling.
> - d) Increasing the number of clusters always reduces the variability of
>   results across resamples.
>
> _Select all answers that apply_

+++

> [!note] Question
> Which of the following statements correctly describe how WCSS (within-cluster
> sum of squares, or inertia) behaves in k-means clustering?
>
> - a) For a fixed number of clusters, WCSS is lower when clusters are compact.
> - b) For a fixed number of clusters, WCSS is lower for wider clusters.
> - c) For a fixed number of clusters, lower WCSS implies lower computational cost
>   during training.
> - d) Assuming `n_init` is large enough to ensure convergence, WCSS always
>   decreases as the number of clusters increases.
>
> _Select all answers that apply_

+++

> [!note] Question
> Which of the following statements correctly describe differences between
> supervised and unsupervised clustering metrics?
>
> - a) Supervised clustering metrics such as ARI and AMI require access to ground
>   truth labels to evaluate clustering performance.
> - b) WCSS and the silhouette score evaluate internal cluster structure without
>   needing reference labels.
> - c) V-measure is zero when labels are assigned completely at random.
> - d) Supervised clustering metrics are not useful if the number of clusters does
>   not match the number of predefined classes.
>
> _Select all answers that apply_
