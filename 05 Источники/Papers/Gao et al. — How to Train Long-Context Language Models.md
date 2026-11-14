---
title: How to Train Long-Context Language Models (Effectively)
type: source-note
status: canonical
source_language: en
last_verified: 2026-11-10
source_url: https://aclanthology.org/2025.acl-long.366/
authors: [Tianyu Gao, Alexander Wettig, Howard Yen, Danqi Chen]
year: 2025
license: CC BY 4.0
---

# How to Train Long-Context Language Models (Effectively)

[Gao et al., ACL 2025, pp. 7376–7399](https://aclanthology.org/2025.acl-long.366/).
[Author repository and checkpoints](https://github.com/princeton-nlp/ProLong).

[§2.1 and Figure 1](https://aclanthology.org/2025.acl-long.366.pdf#page=2)
start from Llama-3-8B-Base and compare intermediate checkpoints of
long-context continued pretraining. At each checkpoint, the authors
evaluate directly and after a separate UltraChat SFT run. In the RAG
panel, the score rises with additional pretraining tokens after SFT;
the same trend is much less visible when evaluating the base
checkpoints. The lines and shaded regions summarize two training runs.
The figure tests how the evaluation protocol affects the conclusions
one draws during development. It is not an ablation of Meta's Llama 3.1
training recipe.

[§3.2 and Figure 2](https://aclanthology.org/2025.acl-long.366.pdf#page=4)
vary the fraction of long documents with the total held at 5B tokens.
The fraction that works best depends on the task and on whether the
model has received SFT. [§5](https://aclanthology.org/2025.acl-long.366.pdf#page=6)
then compares instruction datasets and tests whether long SFT examples
are necessary in this setting.

The original Figure 1, its axis reading, and the distinction between
sequence length and measured performance are in
[[00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура#llama31-long-context|the LLaMA chapter]].
The proceedings figure is reproduced under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
([ACL Anthology policy](https://aclanthology.org/faq/copyright/)).
