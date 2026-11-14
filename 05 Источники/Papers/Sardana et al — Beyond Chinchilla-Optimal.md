---
title: "Beyond Chinchilla-Optimal: accounting for inference"
type: source-note
status: canonical
last_verified: 2026-11-14
source_url: https://proceedings.mlr.press/v235/sardana24a.html
year: 2024
---

# Beyond Chinchilla-Optimal: accounting for inference

[Sardana, Portes, Doubov and Frankle, ICML 2024](https://proceedings.mlr.press/v235/sardana24a.html)
fix a target pretraining loss and forecast inference demand. They compare
the combined cost of training and serving models capable of reaching that
loss. Equal pretraining loss is their proxy for equal quality, not a
guarantee of equivalent behavior on specific tasks.

| Where to read | What the result uses |
| --- | --- |
| [§2 and Figures 1–2](https://arxiv.org/html/2401.00448v3#S2) | A loss model, a fixed demand forecast and an estimate of training and inference FLOPs. |
| [§§3–4](https://arxiv.org/html/2401.00448v3#S3) | The models trained to test scaling far beyond typical tokens-per-parameter ratios. |
| [§5](https://arxiv.org/html/2401.00448v3#S5) and [published Figure 5, p. 6](https://raw.githubusercontent.com/mlresearch/v235/main/assets/sardana24a/sardana24a.pdf#page=6) | How fitting on progressively longer runs changes the extrapolated curves; the short-run fit can overstate gains. |
| [§6](https://arxiv.org/html/2401.00448v3#S6) | A cost model that separates training, input and output utilization. Latency is not part of its objective. |

The [[00 Учебник/11 Pre-training и Scaling/43 Scaling laws#scaling-iso-loss-lifetime|scaling chapter]] works through a candidate search at equal loss
and reads the original ICML Figure 5. The
[[00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура#training-and-inference-budget|LLaMA chapter]] compares that choice with
the fixed training budget of the original Chinchilla experiment and reads
Figures 1 and 2(b). Figures are from the published version under the
[PMLR publication agreement, CC BY 4.0](https://proceedings.mlr.press/pmlr-license-agreement.html).
