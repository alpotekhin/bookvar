---
title: "Why do LLMs attend to the first token?"
type: source-note
status: canonical
last_verified: 2026-11-14
source_url: https://arxiv.org/html/2504.02732v4
year: 2025
---

# Why do LLMs attend to the first token?

[Barbero et al., arXiv v4](https://arxiv.org/html/2504.02732v4) study why
some attention heads put much of their probability mass on the first
position. They test a proposed explanation involving the amount of
information mixed across positions and change how the first token is
introduced during pretraining. The packing experiments are controlled
runs with roughly 120M parameters and 30B tokens, not the Llama 3 training recipe.

| Section | What to look for |
| --- | --- |
| [§3](https://arxiv.org/html/2504.02732v4#S3) | The over-mixing hypothesis and measurements of attention outputs in Gemma 7B. |
| [§5](https://arxiv.org/html/2504.02732v4#S5) | What changes when a beginning-of-sequence token appears at document boundaries or is fixed at the start of the window. |
| [Appendix A.3, Figure 8](https://arxiv.org/html/2504.02732v4#A1.F8) | The four attention masks used in the packing experiments. A shared beginning-of-window token is an extra column even in the isolated-document variant. |

The [[00 Учебник/11 Pre-training и Scaling/42 Next-token prediction#ntp-packed-document-boundary|textbook explanation]] uses Figure 8
to distinguish accessible keys from the targets included in next-token
loss. The figure is reproduced unchanged under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
