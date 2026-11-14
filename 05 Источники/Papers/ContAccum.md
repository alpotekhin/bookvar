---
title: "A Gradient Accumulation Method for Dense Retriever under Memory Constraint"
type: source-note
status: canonical
last_verified: 2026-11-09
source_url: https://arxiv.org/abs/2406.12356v3
authors: [Jaehee Kim, Yukyung Lee, Pilsung Kang]
year: 2024
source_language: en
license: CC BY 4.0
---

# Contrastive Accumulation (ContAccum)

[Kim, Lee and Kang, arXiv v3](https://arxiv.org/abs/2406.12356v3)
and the [NeurIPS 2024 proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/15ba84c1e19b0eb75f96922f5da0a021-Abstract-Conference.html).
The [comparison figure](https://arxiv.org/html/2406.12356v3#S2.F1)
comes from arXiv v3, which identifies its license as
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

With ordinary gradient accumulation, §3.1 applies the contrastive loss to
each microbatch, so a query is compared only with documents in that microbatch.
GradCache uses scores from the whole current batch, caches gradients at the
representation boundary and recomputes encoder activations. The conditions
for equivalence to full-batch differentiation are covered in
[[00 Учебник/15 Embeddings, Retrieval и RAG/60 Embeddings и metric learning#emb-gradcache|the chapter's GradCache example]].

ContAccum (§§3.2–3.3) keeps two FIFO queues, one for query vectors and one
for document vectors. Both queues are detached when their entries are
concatenated with current representations. An old query can still affect
the gradient of a current document, and vice versa; only the current
representations connect to the encoder parameters. Old vectors may have been
computed before a weight update. Enlarging this set of candidates changes
the objective rather than reconstructing a larger current batch.

Figure 1 shows the three score matrices. Figure 2 follows an enqueue and
dequeue in both queues. The retrieval and timing measurements in §5 apply
to the paper's DPR setup, not to arbitrary corpora, encoders or hardware.
