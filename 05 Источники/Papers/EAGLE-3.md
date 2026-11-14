---
title: "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"
type: source-note
status: canonical
source_language: en
source_url: https://arxiv.org/abs/2503.01840v3
year: 2025
last_verified: 2026-11-09
---

# EAGLE-3

Yuhui Li, Fangyun Wei, Chao Zhang and Hongyang Zhang.
[Paper, arXiv v3](https://arxiv.org/html/2503.01840v3);
[authors' NeurIPS version on OpenReview](https://openreview.net/forum?id=4exx1hUffq);
[official implementation, commit `cb7e0841`](https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/README.md).

[Figure 5](https://arxiv.org/html/2503.01840v3#S3.F5) follows a draft through
three inference steps. [Figure 6](https://arxiv.org/html/2503.01840v3#S3.F6)
shows how the training mask separates continuations of different prefixes.
Both original figures are reproduced in the textbook under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as recorded on the authors' OpenReview submission.
For the actual loss and inputs, see
[`dataprepare` and `forward`](https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/eagle/traineagle3/cnets.py#L713-L866):
the code returns the drafter's previous states to its input and matches
the teacher's restricted token distribution at each retained position.

The [authors' repository chart](https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/figs/eagle3r.jpg)
is reproduced unchanged under [Apache-2.0](/licenses/eagle-cb7e0841-Apache-2.0.txt).
For numerical comparisons, distinguish
[Table 1](https://arxiv.org/html/2503.01840v3#S4.T1),
which uses vanilla decoding as its denominator, from the
[SGLang throughput run](https://arxiv.org/html/2503.01840v3#S4.SS3).

- [[00 Учебник/14 Inference и оптимизация/58 Спекулятивное декодирование.md#eagle3-inference-figure|Textbook: the three inference steps in Figure 5]].
- [[00 Учебник/14 Inference и оптимизация/58 Спекулятивное декодирование.md#eagle3-training-mask|Textbook: the attention masks in Figure 6]].
- [[00 Учебник/14 Inference и оптимизация/58 Спекулятивное декодирование.md#eagle3-training-targets|Textbook: the training inputs and soft target]].
- [[00 Учебник/14 Inference и оптимизация/58 Спекулятивное декодирование.md#eagle3-benchmark-reading|Textbook: reading the speedup chart]].
