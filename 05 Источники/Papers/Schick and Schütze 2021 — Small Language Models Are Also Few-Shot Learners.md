---
title: Small Language Models Are Also Few-Shot Learners
type: source-note
status: canonical
last_verified: 2026-11-13
source_url: https://aclanthology.org/2021.naacl-main.185/
year: 2021
---

# Small Language Models Are Also Few-Shot Learners

Timo Schick and Hinrich Schütze, NAACL 2021, pp. 2339–2352.
[Proceedings paper](https://aclanthology.org/2021.naacl-main.185.pdf), [CC BY 4.0](https://aclanthology.org/faq/copyright/).

[Sections 3 and 3.1](https://aclanthology.org/2021.naacl-main.185.pdf#page=2) define PET and its extension to
multi-token answers. [Section 4 and Table 1](https://aclanthology.org/2021.naacl-main.185.pdf#page=6) compare
FewGLUE fine-tuning with frozen GPT-3 across SuperGLUE. The number
of labeled examples per task matches; the sets of examples and the
use of unlabeled data do not. [Figure 5](https://aclanthology.org/2021.naacl-main.185.pdf#page=8) compares one
and 32 demonstrations within GPT-3 configurations, rather than
comparing different adaptation methods.

[[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning.md#gpt3-few-shot-pet|Textbook: 32 examples in the prompt or 32 examples for fine-tuning]].
