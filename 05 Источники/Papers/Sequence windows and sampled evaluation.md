---
title: Sequence windows and sampled recommendation evaluation
type: source-note
status: canonical
source_language: en
last_verified: 2026-11-09
source_url: https://research.google/pubs/on-sampled-metrics-for-item-recommendation/
primary_sources:
  - https://arxiv.org/abs/1808.09781v1
  - https://research.google/pubs/on-sampled-metrics-for-item-recommendation/
  - https://arxiv.org/abs/2106.10621v1
  - https://arxiv.org/abs/2211.15743v1
---

# Sequence windows and sampled recommendation evaluation

- Wang-Cheng Kang and Julian McAuley, [Self-Attentive Sequential Recommendation, §III-A](https://arxiv.org/html/1808.09781v1#S3.SS1), ICDM 2018. Left padding places the most recent actions in a fixed-width window. The [original position lookup](https://github.com/kang205/SASRec/blob/e3738967fddab206d6eeb4fda433e7a7034dd8b1/model.py#L28-L42) indexes the learned position table by the current slot. The textbook derives a cache consequence of that input convention; it does not claim that the original code implements a KV cache.
- Walid Krichene and Steffen Rendle, [On Sampled Metrics for Item Recommendation](https://research.google/pubs/on-sampled-metrics-for-item-recommendation/), KDD 2020. A naive sampled metric can change the relative order of methods even in expectation. Rendle's [earlier analysis, §§4.4–4.5](https://arxiv.org/html/1912.02263v1#S4.SS4) derives the distribution of the sampled rank when exactly one item is relevant.
- Dong Li, Ruoming Jin, Jing Gao and Zhi Liu, [On Sampling Top-K Recommendation Evaluation, §2.2](https://arxiv.org/html/2106.10621v1#S2.SS2), KDD 2020. Uniform sampling without replacement gives a hypergeometric count of items that outrank the relevant item. The paper studies approximating a possibly different full-catalog cutoff; it does not make unadjusted sampled and full-catalog Hit@1 interchangeable.
- Dong Li, Ruoming Jin, Zhenming Liu, Bin Ren, Jing Gao and Zhi Liu, [Towards Reliable Item Sampling for Recommendation Evaluation, §4.1 and Figure 1](https://arxiv.org/html/2211.15743v1#S4.F1), preprint v1, 2022. The original plot shows rank concentration and estimation error in one EASE / MovieLens 20M experiment. The [original PNG](https://arxiv.org/html/2211.15743v1/figure/p1.png) is reused unchanged under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Worked examples: [[00 Учебник/02 Рекомендательные системы/04 Последовательные рекомендации и признаки#sasrec-rolling-window-cache|position changes and KV reuse]] and [[00 Учебник/02 Рекомендательные системы/05 Оценивание и эксплуатация#sampled-ranking-order-reversal|an exact reversal under sampled Hit@1]]. The examples use synthetic inputs, not scores from these papers.
