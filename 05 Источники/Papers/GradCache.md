---
title: "Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup"
type: source-note
status: canonical
last_verified: 2026-11-06
source_url: https://aclanthology.org/2021.repl4nlp-1.31/
authors: [Luyu Gao, Yunyi Zhang, Jiawei Han, Jamie Callan]
year: 2021
source_language: en
---

# Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup

[Published paper, RepL4NLP 2021](https://aclanthology.org/2021.repl4nlp-1.31/),
DOI: 10.18653/v1/2021.repl4nlp-1.31.
[Official implementation, revision 906f0383](https://github.com/luyug/GradCache/tree/906f03835fbc183132a9db32612a9e8f180ca3b4).

The integrated explanation is in
[[00 Учебник/15 Embeddings, Retrieval и RAG/60 Embeddings и metric learning#emb-gradcache|GradCache: full-batch negatives with chunked encoder passes]].
It retains the published §3.3 prose in English, then works through a two-pair
example, representation derivatives, recomputation and the shared encoder's
parameter gradient. The chapter also separates encoder-activation memory
from cache and score-matrix memory and explains the stochastic/stateful-layer
conditions for gradient equivalence.

The paper's §3.1–3.4 develops the method; §4 reports DPR experiments on Natural
Questions. Its runtime and retrieval results concern those experiments, not
a universal overhead or guarantee of better embeddings. The early arXiv v1
has an algorithm illustration absent from the published version; that image
has not been republished. The chapter uses the separately licensed
[[05 Источники/Papers/ContAccum|comparison from Kim, Lee and Kang]]
to show which query–document scores each method includes.

The selected published prose is reproduced under CC BY 4.0 according to the
[ACL Anthology copyright policy](https://aclanthology.org/faq/copyright/).
The implementation's Apache-2.0 license is separate. The local fixture retains
the original `GradCache` and `RandContext` files unchanged so that recomputation
can be checked without downloading model weights or contacting an inference API.
