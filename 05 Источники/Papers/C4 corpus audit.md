---
title: "C4: auditing the corpus and what it excludes"
type: source-note
status: canonical
last_verified: 2026-11-14
source_url: https://aclanthology.org/2021.emnlp-main.98/
year: 2021
---

# C4: auditing the corpus and what it excludes

[Dodge et al., EMNLP 2021](https://aclanthology.org/2021.emnlp-main.98/)
study the English portion of C4 using records that survived corpus
construction and alternatives produced by changing individual rules.
They distinguish the composition of the retained corpus from the effect
of the blocklist and the presence of benchmark text.

| Where to read | What the authors measure |
| --- | --- |
| [§§2–3](https://arxiv.org/html/2104.08758v2#S2) | How the C4 variants were constructed; domains, dates and geographic coverage. |
| [§4.2, Table 2](https://arxiv.org/html/2104.08758v2#S4.SS2) | Exact normalized overlap with separate fields of evaluation examples, not a model's measured score gain. |
| [§5](https://arxiv.org/html/2104.08758v2#S5) and [Appendix A.6](https://arxiv.org/html/2104.08758v2#A1.SS6) | The topics of excluded pages, association with identity mentions, and model-estimated varieties of English. |

The textbook uses [[00 Учебник/11 Pre-training и Scaling/41a Дедупликация, PII и контроль качества корпуса#c4-blocklist-audit|Figure 5 to explain the blocklist audit]] and [[00 Учебник/11 Pre-training и Scaling/41a Дедупликация, PII и контроль качества корпуса#contamination|the QNLI example to explain field-level overlap]].
Figure 5 is reproduced unchanged under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
