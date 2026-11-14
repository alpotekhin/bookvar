---
title: "Dolma: filtering, deduplication and ablations"
type: source-note
status: canonical
last_verified: 2026-11-14
source_url: https://aclanthology.org/2024.acl-long.840/
year: 2024
---

# Dolma: filtering, deduplication and ablations

[Soldaini et al., ACL 2024](https://aclanthology.org/2024.acl-long.840/)
describe the three-trillion-token Dolma release. The paper separates the
construction of the web, code and social subsets from experiments on
intermediate data variants. The latter use smaller models; they do not
report the cost or outcome of retraining every released model.

| Where to read | What the authors compare |
| --- | --- |
| [§4](https://arxiv.org/html/2402.00159v2#S4) | Applying filter annotations when building a mixture; model and evaluation settings for the data experiments. |
| [§5.3](https://arxiv.org/html/2402.00159v2#S5.SS3) and [Appendix I](https://arxiv.org/html/2402.00159v2#A9) | Content filters and the decision to mask detected spans or drop the document. |
| [§5.4](https://arxiv.org/html/2402.00159v2#S5.SS4) | Exact URL, document and paragraph deduplication; rates use different denominators. |
| [Appendix J](https://arxiv.org/html/2402.00159v2#A10) | Document-level overlap between filter decisions, split by perplexity bucket. |
| [Appendix O.3](https://arxiv.org/html/2402.00159v2#A15.SS3) | Perplexity and downstream task scores for alternative PII policies; no direct measurement of personal information left in the corpus. |

The [[00 Учебник/11 Pre-training и Scaling/41a Дедупликация, PII и контроль качества корпуса#dolma-pii-policy|textbook explanation]] reads the C4 panel of Figure 31 alongside
the masking rule in the text. The source panel is reproduced unchanged
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The [dataset card](https://huggingface.co/datasets/allenai/dolma) and
[toolkit](https://github.com/allenai/dolma) set out separate terms.
