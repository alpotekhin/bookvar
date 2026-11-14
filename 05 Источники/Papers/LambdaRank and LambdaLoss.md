---
title: "LambdaRank and the LambdaLoss framework"
type: source-note
status: canonical
last_verified: 2026-11-06
source_url: https://marc.najork.org/papers/cikm2018.pdf
authors: [Xuanhui Wang, Cheng Li, Nadav Golbandi, Michael Bendersky, Marc Najork]
year: 2018
source_language: en
---

# LambdaRank and the LambdaLoss framework

[The LambdaLoss Framework for Ranking Metric Optimization, CIKM 2018](https://marc.najork.org/papers/cikm2018.pdf),
DOI: 10.1145/3269206.3271784.
Section 3.4 defines the change in nDCG used to weight document pairs;
§4.3 describes the ranked-list assignment and weighted optimization step.
The paper distinguishes these objectives from differentiating nDCG through sorting.

The integrated explanation and executable example are in
[[00 Учебник/15 Embeddings, Retrieval и RAG/62 Reranking — cross-encoder и late interaction#rerank-lambdarank|LambdaRank: weighting pairs by a change in nDCG]].
They derive the swap weights for three documents, the weighted pairwise
gradient and the effect of a top-k cutoff.

[Sentence Transformers implementation](https://github.com/huggingface/sentence-transformers/blob/8a880ec7f1524640619fe382940ee319144f74ca/sentence_transformers/cross_encoder/losses/lambda_loss.py)
is pinned to revision `8a880ec7`. Its `LambdaRankScheme`, `LambdaLoss.forward`, pair mask,
normalizer, default weighting scheme and logarithm convention are checked
separately from an exact truncated-nDCG swap calculation. The full source file
and its Apache-2.0 license are retained as public test fixtures.

The paper is linked, not republished. No new paper figure or page image is copied.
