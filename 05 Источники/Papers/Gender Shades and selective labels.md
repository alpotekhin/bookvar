---
title: Gender Shades and selective labels
type: source-note
status: canonical
source_language: en
last_verified: 2026-11-09
source_url: https://proceedings.mlr.press/v81/buolamwini18a.html
primary_sources:
  - https://proceedings.mlr.press/v81/buolamwini18a.html
  - https://doi.org/10.1145/3097983.3098066
---

# Gender Shades and selective labels

Joy Buolamwini and Timnit Gebru, [Gender Shades](https://proceedings.mlr.press/v81/buolamwini18a.html),
FAT* 2018, PMLR 81:77–91. Sections 3.3–3.5 describe how the authors assembled
and annotated the Pilot Parliaments Benchmark. Figure 3 compares the subgroup
composition of three datasets; Table 4 separately reports classification errors
on PPB. The printed Figure 3 caption assigns 4.4% to Adience, but the plot and
§3.5 assign that figure to IJB-A. The
[[00 Учебник/18 Evaluation и методология/59a Responsible systems#responsible-intersectional-audit|textbook comparison]]
reuses the original plot under the [PMLR CC BY 4.0 agreement](https://proceedings.mlr.press/pmlr-license-agreement.html).

Himabindu Lakkaraju et al.,
[The Selective Labels Problem](https://www.kdd.org/kdd2017/papers/view/the-selective-labels-problem-evaluating-algorithmic-predictions-in-the-pres),
KDD 2017, pp. 275–284. [Section 3 and Figure 1](https://www.cs.cornell.edu/home/kleinber/kdd17-selective.pdf)
show how existing decisions determine which outcomes can be observed.
The [[00 Учебник/18 Evaluation и методология/59a Responsible systems#responsible-selective-labels-bounds|textbook loan example]]
uses a separate elementary calculation: it bounds the group true positive rates
by varying the unknown outcomes. It does not implement the paper's contraction
method or reuse its illustration.
