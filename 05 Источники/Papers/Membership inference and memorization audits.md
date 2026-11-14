---
title: Membership inference and memorization audits
type: source-note
status: canonical
source_language: en
last_verified: 2026-11-09
source_url: https://arxiv.org/abs/2112.03570v2
primary_sources:
  - https://arxiv.org/abs/2112.03570v2
  - https://proceedings.mlr.press/v235/zarifzadeh24a.html
  - https://arxiv.org/abs/1802.08232v3
---

# Membership inference and memorization audits

- Nicholas Carlini et al., [Membership Inference Attacks From First Principles](https://arxiv.org/html/2112.03570v2#S3.SS2), IEEE S&P 2022. Section III-B contrasts balanced accuracy and overall ROC AUC with TPR at low FPR. Sections IV-A–IV-C derive LiRA from per-example IN and OUT score distributions. Figure 2 shows the same baseline ROC on linear and logarithmic scales.
- Sajjad Zarifzadeh, Philippe Liu, Reza Shokri, [Low-Cost High-Power Membership Inference Attacks](https://proceedings.mlr.press/v235/zarifzadeh24a.html), ICML 2024, PMLR 235:58244–58282. Section 2 sets up the membership test. Figure 1 (PDF p. 2) compares attacks on CIFAR-100 with a single reference model. The textbook crop preserves the complete plot under [CC BY 4.0](https://proceedings.mlr.press/pmlr-license-agreement.html).
- Nicholas Carlini et al., [The Secret Sharer](https://arxiv.org/html/1802.08232v3#S4), USENIX Security 2019. Sections 4.1–4.3 define the search space, likelihood rank and exposure. Section 5 describes canary insertion into the training corpus and evaluation; Section 8 tests extraction separately.

The chapter works through
[[00 Учебник/19 Deployment, Reliability и MLOps/04 Безопасность и приватность ML-систем#security-membership-audit|membership thresholds and base rates]]
and [[00 Учебник/19 Deployment, Reliability и MLOps/04 Безопасность и приватность ML-систем#security-canary-exposure|a canary exposure calculation]].
These examples use synthetic numbers, not the papers' experimental results.
