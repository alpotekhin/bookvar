---
title: Delayed conversion feedback
type: source-note
status: canonical
source_language: en
last_verified: 2026-11-09
source_url: https://arxiv.org/abs/2012.03245v1
primary_sources:
  - https://doi.org/10.1145/2623330.2623634
  - https://arxiv.org/abs/2012.03245v1
---

# Delayed conversion feedback

- Olivier Chapelle, [Modeling Delayed Feedback in Display Advertising](https://doi.org/10.1145/2623330.2623634), KDD 2014, pp. 1097–1105. In §3, a click with no observed conversion contributes a mixture of non-conversion and conversion after the observation time (Eq. 9). The model uses a logistic conversion probability and an exponential conditional delay distribution. It does not fix a seven-day target window.
- Jia-Qi Yang, Xiang Li, Shuguang Han, Tao Zhuang, De-Chuan Zhan, Xiaoyi Zeng, and Bin Tong, [Capturing Delayed Feedback in Conversion Rate Prediction via Elapsed-Time Sampling](https://arxiv.org/html/2012.03245v1), preprint v1, 2020; AAAI 2021. In Background, [Figure 1](https://arxiv.org/html/2012.03245v1#Sx3.F1) distinguishes click, conversion, and observation times. Eqs. 2–3 separate the observed status from the eventual outcome. The proposed method samples observation ages, inserts a positive record when a previously missed conversion arrives, and uses importance weights to account for the changed training distribution. Those weights are estimated, so the paper also analyzes their bias.

The textbook uses the [original preprint PNG](https://arxiv.org/html/2012.03245v1/figures/toxi.png) without changes under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as stated for [arXiv v1](https://arxiv.org/abs/2012.03245v1). The [[00 Учебник/02 Рекомендательные системы/01 Задача рекомендации, данные и pipeline#recommendation-delayed-labels|worked example]] fixes a seven-day purchase target per impression and treats event delivery separately. It does not reproduce either paper’s estimator or experiments.
