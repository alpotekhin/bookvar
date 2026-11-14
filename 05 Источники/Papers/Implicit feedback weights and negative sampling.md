---
title: Implicit feedback weights and negative sampling
type: source-note
status: canonical
source_language: en
last_verified: 2026-11-09
source_url: https://doi.org/10.1145/2911451.2911489
primary_sources:
  - https://yifanhu.net/PUB/cf.pdf
  - https://arxiv.org/abs/1205.2618v1
  - https://doi.org/10.1145/2484028.2484126
  - https://doi.org/10.1145/2911451.2911489
---

# Implicit feedback weights and negative sampling

- Yifan Hu, Yehuda Koren, and Chris Volinsky, [Collaborative Filtering for Implicit Feedback Datasets](https://yifanhu.net/PUB/cf.pdf), ICDM 2008. Section 4 defines the confidence-weighted objective and the reusable Gram matrix for alternating updates. The conclusion identifies item availability as one reason to distinguish different kinds of missing feedback.
- Xiangnan He, Hanwang Zhang, Min-Yen Kan, and Tat-Seng Chua, [Fast Matrix Factorization for Online Recommendation with Implicit Feedback](https://hexiangnan.github.io/papers/sigir16-eals-cm.pdf#page=4), SIGIR 2016. Sections 4.1–4.2 assign item-specific weights to missing pairs. The algorithm reuses a weighted Gram matrix, but updates coordinates one at a time. The textbook uses the Gram identity in a full-vector solve; it does not implement eALS.
- Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme, [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/pdf/1205.2618v1#page=4), UAI 2009. Sections 4.1–4.2 define the objective over triples and LearnBPR's bootstrap sampling with replacement.
- Weinan Zhang, Tianqi Chen, Jun Wang, and Yong Yu, [Optimizing Top-N Collaborative Filtering via Dynamic Negative Item Sampling](https://wnzhang.net/papers/lambdarankcf-sigir.pdf#page=3), SIGIR 2013. Algorithms 1–2 favor high-scoring unobserved items from a small sampled pool. This intentionally changes which comparisons receive updates. The textbook's two-draw example uses the highest-scoring item and separately derives the importance correction for a *different*, uniform target.

Worked examples: [[00 Учебник/02 Рекомендательные системы/02 Collaborative filtering и matrix factorization#als-availability-weights|an ALS update with an unavailable item]] and [[00 Учебник/02 Рекомендательные системы/03 Ranking и нейронные рекомендательные модели#bpr-hard-negative-probabilities|the expected BPR gradient under two samplers]]. Both use synthetic inputs, not paper benchmark results.
