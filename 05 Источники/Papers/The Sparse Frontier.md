---
title: "The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs"
type: source-note
status: canonical
last_verified: 2026-11-09
source_url: https://arxiv.org/abs/2504.17768v2
year: 2026
---

# The Sparse Frontier

Piotr Nawrot, Robert Li, Renjie Huang, Sebastian Ruder, Kelly Marchisio
and Edoardo M. Ponti. [Preprint v2](https://arxiv.org/html/2504.17768v2),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

[Appendix B](https://arxiv.org/html/2504.17768v2#A2.F12) uses FLOP estimates for prefill and memory traffic
estimates for decoding. Figure 12 compares 16K, 64K and 128K input lengths
for Qwen 2.5 models from 7B to 72B. The bars compare estimates across model
sizes, not GPU timings. The analytical count includes linear attention
projections and the vocabulary head; masking, a different head, or a
different set of kernel assumptions changes the count.

In the paper's quality evaluation, reducing the number of computed
query–key pairs need not preserve performance on every task. The prefill
FLOP chart alone gives neither a latency estimate nor a quality guarantee.

[[00 Учебник/10 ML Systems/04 Арифметика Transformer и MoE.md#arith-attention-share|Textbook: compare context length and model width]].
