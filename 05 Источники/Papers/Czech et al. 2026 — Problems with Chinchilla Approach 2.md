---
title: "Problems with Chinchilla Approach 2: Systematic Biases in IsoFLOP Parabola Fits"
type: source-note
status: canonical
last_verified: 2026-11-09
source_url: https://arxiv.org/abs/2603.22339v1
source_language: en
year: 2026
---

# Problems with Chinchilla Approach 2

Eric Czech, Zhiwei Xu, Yael Elmatad, Yixin Wang and William Held,
[arXiv:2603.22339v1](https://arxiv.org/html/2603.22339v1), CC BY 4.0.

Sections 4 and 5 compare fits to noiseless IsoFLOP samples. With the
same sampling grid centered on each true minimum, the symmetric loss
surface recovers both the exponent and the coefficient. For an
asymmetric surface the fitted minimum moves by the same factor at every
budget. This changes the coefficient even when the exponent is exact.
Section 6 shows what happens when the center of the grid drifts with
the budget: now the exponent can change too.

[Figure 3](https://arxiv.org/html/2603.22339v1#S4.F3) shows the symmetric
control; [Figure 4](https://arxiv.org/html/2603.22339v1#S5.F4) compares two
asymmetric surfaces. These panels come from simulations, not repeated
training runs. Section 8 describes a fit that solves for three linear
coefficients at each candidate pair of exponents. The paper evaluates
the method on simulations and on data digitized from published plots;
the resulting allocation estimates depend on the fitted loss model.

[[00 Учебник/11 Pre-training и Scaling/43 Scaling laws.md#isoflop-parabola-bias|Textbook (in Russian): a three-point fit with a known minimum]].
