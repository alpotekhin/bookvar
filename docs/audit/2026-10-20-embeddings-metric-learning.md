# Embeddings and metric learning — local editorial acceptance

Date: 2026-10-20.

Chapter: `00 Учебник/15 Embeddings, Retrieval и RAG/60 Embeddings и metric learning.md`.
SHA-256: `ddf06790a50b026ab2480f28b12360453e881ad945895277a86bfe9c71a5caf3`.
Route: `/textbook/retrieval/embeddings-metric-learning/`.
Worktree branch: `editorial-route-alignment`.

This pass covers this chapter, its five new illustrations, the provenance of
three existing DPR slides, and narrow responsive styles for those slides and
SimCSE. It does not claim completion of the textbook, ingestion of entire
courses, an embedding-model benchmark, or English parity. No commit, push,
deployment or translation was performed.

## Sources actually read

Frozen sources and hashes are in
`05 Источники/Originals/embeddings-2026-10-20/references.json` (27 files).

- Sentence-BERT: abstract/introduction, related work, §3 and §3.1, table 1 and
  §4.1; original architecture diagrams on PDF page 3 visually inspected.
  This is selected-section review, not a claim of reading the entire paper.
- DPR: §3.1–3.2 (encoders, objective, inference, negative choices), §4.1
  positive-context selection and preprocessing, §5.1 and the negative-sampling
  ablation/table 3 in §5.2. Cross-dataset analysis outside the selected passage
  was not used to support a new claim.
- Danqi Chen, ACL 2020 OpenQA tutorial, part 5: slides 20–22 and their surrounding
  discussion. The frozen PDF is pinned to
  `3c82cf3bfa7f7f26aa4016cf4c48b83f1a4eaad0`.
- Sentence Transformers: complete selected README explanations for MS MARCO,
  cross-encoder distillation and Matryoshka; Loss Overview; Semantic Search
  background, symmetric/asymmetric distinction, manual search and ANN discussion.
  Code review covered the MNRL constructor/docstring and loss computation,
  MarginMSE default scoring and loss computation, and Pooling mask/prompt,
  mean, CLS, max and last-token paths. Unrelated API options were not exhaustively
  documented or executed. Repository pin:
  `8a880ec7f1524640619fe382940ee319144f74ca`.
- SimCSE README method overview, original two-panel diagram and complete MIT
  license, pin `49fa580a853752ede55b8c76d9debf748e214d3f`.
- multilingual-e5-base model-card usage: prefix roles, masked mean pooling,
  normalization and truncation. The downloaded snapshot is hashed; its remote
  `main` URL is not presented as an immutable model revision.
- Introduction to Information Retrieval, §8.4 ranked-result evaluation, including
  the exponential-gain nDCG convention; MTEB abstract, not a current leaderboard
  survey or model recommendation.
- Complete Apache-2.0 and MIT license texts; Hugging Face documentation-images
  dataset license metadata/README at
  `1bade87e4c4e41d80d5f72ba2d7b38ab3194f044`.

## Explanation review

The entire chapter was read before editing, then reread in successive complete
ranges after expansion. Subsequent local additions were reviewed in their full
sections. Existing seven section headings, the worked-batch heading and source
footer anchors were retained.

| Mechanism | Worked reasoning and limitations now present |
| --- | --- |
| Pooling | Three token states; padding changes `(1,1)` into `(25.75,.75)` if included; CLS/last/max comparison; empty-mask rejection. |
| Similarity | A dot-product ranking reversed by cosine; unit-vector distance identity; zero vectors and non-calibrated scores. |
| Encoder setup | Independent encodings versus shared weights; SBERT NLI classifier and discarded classification head; DPR separate encoders; all-pairs count and candidate/reranker boundary. |
| Contrastive training | B-by-B scores, row normalization, numeric probability/loss, softmax derivatives, temperature scaling and finite-difference verification. |
| Multiple positives | Log of set probability contrasted with mean negative log probability; distinct demands on positive coverage. |
| Negative selection | Random, other-question and BM25 candidates; weak answer-string labels; false negatives; teacher filtering example; historical DPR ablation interpreted with its own metric. |
| Batch mechanics | B-by-2B candidate accounting; two singleton microbatches versus one joint batch; cross-device gradients and Cached/GradCache costs. |
| SimCSE | Two dropout views, supervised NLI positive/negative roles, and relevance changing with the retrieval task. |
| Objectives | Pair MSE, pair contrastive and triplet losses with numeric active/inactive margins; scale conventions not treated as interchangeable. |
| Distillation | Original diagram correctly identified as cross-encoder-to-cross-encoder; separate bi-encoder MarginMSE derivation, offset invariance and incompatible score scales. |
| Geometry | Shared-component cosine counterexample; centering caveat; explicit covariance/whitening example; no claim that a UMAP picture proves retrieval quality. |
| Dimension | Byte arithmetic, prefix-truncation counterexample, MRL nested losses, separate prefix normalization, STSB correlation versus accuracy, unchanged encoder computation. |
| Compatibility | Joint versus query-only rotation; consistent document/query processing and reindexing conditions. |
| Evaluation | Five-document exact search, graded qrels, Recall/Hit/RR/nDCG with intermediate calculations, second ranking with identical Recall but worse order; ANN overlap versus relevance. |

Examples are explicitly hypothetical, except historical source results marked as
such. No newly authored synthetic example is attributed to a course experiment.
No actual model was trained or downloaded for an empirical quality claim.

## Editorial review

Applied no-ai-slop SKILL and eval with project qualifications for academic prose:
preserve mathematical reasoning, necessary terms, useful contrasts, citations and
normal Russian punctuation. The expansion request authorizes new worked examples;
this is not a voice-preserving copy edit of an external author.

The review removed a broken sentence about the loss overview, the claim that
one million pairs necessarily mean one million separate forward calls, a generic
STS preface, and editorial process remarks in reader-facing captions. It kept
English licensed excerpts verbatim rather than translating or restyling them.
Captions distinguish the mechanism in a figure from the mechanism discussed next.
The style checklist does not constitute proof of authorship or technical quality;
the source, arithmetic and UI checks below are separate evidence.

## Illustrations and provenance

Eight original illustrations appear in the chapter. Five new PNGs are unchanged
byte copies with per-asset metadata in
`00 Учебник/Assets/Figures/curated/embeddings-review-2026-10/assets.json`:

- bi-/cross-encoder, MNRL, cross-encoder distillation: Apache-2.0, Sentence Transformers;
- SimCSE: MIT, Princeton NLP;
- Matryoshka results: **CC BY-NC-SA 4.0**, the Hugging Face documentation-images
  dataset, not the Apache license of the code linking to that image.

Attribution and source links are adjacent to each figure. Public Apache and MIT
notices match the frozen upstream license files. Quotes from the licensed README
sources are checked against the originals.

The three older DPR files were not redrawn or cropped. Re-rendering the pinned
PDF pages 20, 21 and 22 at 180 dpi produced 2560×1920 images with **identical RGB
pixels** to all three existing PNGs (mean absolute difference 0). Their registry
entries now point to the real frozen PDF and page numbers rather than missing
ignored `raw` PNG paths and an inaccurate crop description.

Rights note: the pre-existing user-confirmed reuse statement for those three
slides was retained. This pass established source identity, not independent
copyright permission for those previously imported slides. It does not infer
redistribution permission merely from open access. No new publication occurred.

## Executed checks

- `check_embeddings_examples.py`: 456 grouped checks, including three Python
  blocks executed standalone, random masked pooling/metric identities,
  one-/multi-positive finite-difference gradients, all 120 permutations of five
  documents, frozen-source hashes, image decoding, licenses, quotes and built
  anchors/figures/formulas. These are synthetic correctness checks, not a model
  benchmark or 456 independent pedagogical acceptance tests.
- Asset-registry suite: **47 passed**. New negative cases reject pin, image URL,
  hash, license, evidence, source-page, crop/derivation and author drift.
- Final site build: **1961 pages**, completed successfully at logged 17:41:39.
- Site-output suite after the final build: **12 passed**.
- Built-link check after the final build: **0** broken routes, fragments or files.
- Source-coverage checker passed. This is structural coverage, not evidence that
  every source topic has been explained in the textbook.
- Targeted `git diff --check`: no errors.

## Visual acceptance

All eight figures were inspected in the actual local site at 1365×900 and
390×844. Whole original images were also inspected. The final page was reloaded
and checked for the added NLI and whitening passages, eight loaded images,
eleven display formulas and no KaTeX errors.

Mobile inspection found two actual layout problems and led to changes:

1. The Recall/Hit aligned formula exceeded the 343 px content width (406 px).
   It was changed to independently centered lines. Final inspection: all eleven
   display formulas had `clientWidth == scrollWidth == 343`.
2. Full DPR slides and the two-panel SimCSE image became too small when reduced
   to the phone column. A narrow CSS rule keeps these four unchanged originals
   at 672 px inside local horizontal scrollers at widths up to 40 rem. Both
   SimCSE panels and the DPR loss/table were traversed with keyboard input;
   `scrollLeft` reached 329 inside the 343 px container. The whole page did not
   become horizontally scrollable. Captions explain the scrolling behavior.

Native original-image opening and zoom were also exercised for SimCSE; this is
available in addition to the in-page scroller, not a substitute for it. Desktop
images remain complete and responsive. Smaller bi-/cross-encoder, MNRL,
distillation and Matryoshka figures retain the overview layout on mobile.

Proof screenshots are under `/tmp/bookvar-embeddings-*-2026-10-20.jpg`, including
`bi-cross-desktop-final`, `simcse-mobile-pan-left`, `simcse-mobile-pan-right`,
`dpr-loss-mobile-pan-right`, `dpr-batch-mobile-pan-right` and `recall-mobile`.
Temporary screenshots are inspection evidence, not source illustrations.

## Remaining work

Next canonical chapter: `61 Retrieval — от BM25 до dense и hybrid.md`.
Other textbook chapters, English synchronization, empirical embedding selection
and publication remain outside this completed local chapter pass. The active
whole-textbook goal is not complete.
