---
title: Label shift and certified robustness
type: source-note
status: canonical
source_language: en
last_verified: 2026-11-09
source_url: https://proceedings.mlr.press/v97/cohen19c.html
primary_sources:
  - https://proceedings.mlr.press/v80/lipton18a.html
  - https://proceedings.mlr.press/v97/cohen19c.html
---

# Label shift and certified robustness

- Zachary C. Lipton, Yu-Xiang Wang, Alexander J. Smola,
  [Detecting and Correcting for Label Shift with Black Box Predictors](https://proceedings.mlr.press/v80/lipton18a.html),
  ICML 2018, PMLR 80:3122–3130. Sections 3–4 and equation (1) separate
  the source confusion matrix from the target prediction marginal. A fixed
  classifier can estimate a target label prior under the label-shift and
  identifiability assumptions; a change in predictions alone does not establish
  those assumptions.
- Jeremy Cohen, Elan Rosenfeld, J. Zico Kolter,
  [Certified Adversarial Robustness via Randomized Smoothing](https://proceedings.mlr.press/v97/cohen19c.html),
  ICML 2019, PMLR 97:1310–1320. Section 3.1 gives the radius for the
  smoothed classifier; §3.2.2 and the CERTIFY pseudocode separate class
  selection from estimation. Figure 1 (PDF p. 1) shows the vote
  distribution and Figure 3 (PDF p. 5) shows the worst-case half-space.
  Both images were cropped from the published PDF without redrawing under
  the [PMLR CC BY 4.0 publication agreement](https://proceedings.mlr.press/pmlr-license-agreement.html).

The textbook works through [[00 Учебник/19 Deployment, Reliability и MLOps/05 Устойчивость ML-систем#robust-label-shift|binary label-shift estimation]]
and [[00 Учебник/19 Deployment, Reliability и MLOps/05 Устойчивость ML-систем#robust-randomized-smoothing|a smoothing certificate from vote counts]].
The numbers in both worked examples are synthetic.
