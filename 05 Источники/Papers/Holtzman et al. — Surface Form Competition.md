---
title: Surface Form Competition
type: source-note
status: canonical
source_language: en
source_url: https://aclanthology.org/2021.emnlp-main.564/
year: 2021
last_verified: 2026-11-13
---

# Surface Form Competition

Ari Holtzman, Peter West, Vered Shwartz, Yejin Choi, and Luke Zettlemoyer.
[EMNLP 2021, pp. 7038–7051](https://aclanthology.org/2021.emnlp-main.564/);
[arXiv v9, CC BY 4.0](https://arxiv.org/abs/2104.08315v9);
[authors' evaluation code](https://github.com/peterwestuw/surface-form-competition).

[Sections 3.1–3.3](https://arxiv.org/pdf/2104.08315v9#page=3) compare
full-string probability, mean log-likelihood per token, and a correction
using a short domain-specific prefix. [Table 1](https://arxiv.org/pdf/2104.08315v9#page=6)
reports the zero-shot comparison. [Table 4](https://arxiv.org/pdf/2104.08315v9#page=7)
uses four demonstrations, a separate setting.

[Figure 3 and §5.2](https://arxiv.org/pdf/2104.08315v9#page=8) swap the
cause and effect in COPA so that different prefixes predict the same
string. The example below the dotted line is a wording check, not a
third choice in the dataset. In Figure 2, the displayed Avg numerator
omits the logarithm; §3.1 gives the log-likelihood formula.

[[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning.md#gpt3-surface-form-competition|Textbook: when wording changes the winning answer]].
