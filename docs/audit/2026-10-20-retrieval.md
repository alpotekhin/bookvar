# Retrieval chapter — local editorial acceptance

Date: 2026-10-20.

Chapter: `00 Учебник/15 Embeddings, Retrieval и RAG/61 Retrieval — от BM25 до dense и hybrid.md`.
SHA-256: `c2f49438a49ecdf7aac70a44605763ebdcf3311781fb4b10c4514470ceab25bc`.
Route: `/textbook/retrieval/sparse-dense-hybrid/`.
Worktree branch: `editorial-route-alignment`.

This pass covers the Russian retrieval chapter, six new source illustrations,
provenance corrections for two existing OpenQA slides, and a source-image zoom
viewer. It does not establish completion of the textbook, full-course ingestion,
English content parity, a production retrieval system or a model benchmark.
No translation, commit, push or deployment was performed. Unrelated dirty files
were preserved.

## Sources actually read

Frozen source identities are recorded in
`05 Источники/Originals/retrieval-2026-10-20/references.json`: 34 collected
files and two legacy figure hashes. Repository tree inventories are explicitly
collection-only; their presence is not evidence of reading repository contents.
Research snapshots are not automatically licensed publication assets.

- Jurafsky and Martin, SLP3 chapter 11, draft dated 19 August 2026:
  introduction and §11.1.1–11.1.5 on vector representation, TF–IDF, BM25 and
  inverted indexes; the bi-/cross-encoder passage in §11.3. The chapter is not
  cited as deriving BM25 from a probabilistic ranking model. No SLP figure was
  copied. This was selected-section review, not review of the whole textbook.
- Danqi Chen, ACL 2020 OpenQA tutorial, part 4, slides 4–5 and their complete
  figures. Frozen PDF pin `3c82cf3bfa7f7f26aa4016cf4c48b83f1a4eaad0`.
  Slide 4 is the historical DrQA retriever/reader system; slide 5 is TF–IDF,
  not BM25. Both source slides were independently re-rendered for comparison.
- The Faiss Library, arXiv v4, 23 October 2025: complete selected HTML sections
  §3.1, §3.2, §5.1 and §5.2; the PQ/ADC/SDC and OPQ/rotation passages in
  §4.1–4.2 checked in PDF text. Unrelated benchmark plots were not reproduced.
- Faiss official Python tutorials `1-Flat.py` and `2-IVFFlat.py`, both complete,
  and complete MIT license. Pin `42aac2fb4e17a66a6b17e7a99e3df229bcf757b8`.
  The adapted example specifies `METRIC_L2` explicitly rather than relying on
  a possibly stale default-metric comment in the source tutorial.
- Milvus HNSW overview, search procedure and build/search parameter explanations;
  IVF_FLAT overview and build/search explanation; IVF_PQ overview, PQ, ADC and
  combined IVF/PQ explanation. All five selected figures and the complete
  Apache-2.0 license were inspected. This does not claim exhaustive reading of
  every SDK option or parameter table. Documentation pin
  `9acdc99959e188a54be1df512ea3806c3110ef59`.
- Yusuke Matsui, nanopq tutorial: complete tutorial RST and MIT license,
  pin `898fad8226a4fcc1cc96ca584fc1da7749ed2a94`. Its English explanation
  of the ADC distance table is retained verbatim with attribution and notice.
  The nanopq library itself was not installed or benchmarked.
- HyDE, published ACL 2023 paper: method §3.1–3.2 and the complete Figure 1 on
  PDF page 2. The mean of generated-document representations follows the method
  passage; the optional original-query inclusion expression is not reproduced.
  The published ACL version, rather than its earlier arXiv upload, supplies the
  figure and license basis. ACL copyright policy was read separately.
- Anserini BEIR fusion experiment: introductory settings and the complete
  nDCG@10 table. The `nq` and `trec-covid` values are historical results from
  that configuration, not newly measured scores or a current recommendation.
  Pin `07d380635f1719736c6042c6e971fd7d08ee125e`.
- Hugging Face Advanced RAG: selected chunking section §1.1, including structure,
  character/token length discussion and code; not the whole cookbook.
- Elasticsearch kNN documentation: complete selected pre-/post-filter discussion
  and example, with adjacent hybrid/nested-query context; not the entire API page.
- Lucene 10.3.1 BM25Similarity class documentation, including its positive IDF
  and parameter definitions. Only that IDF convention is equated with Lucene;
  the chapter does not claim bit-for-bit equality with its entire scorer.
- Original HNSW abstract was checked. No figure was copied from that paper.
  The existing BEIR further-reading link was retained without claiming a new
  exhaustive review of BEIR experiments.

## Explanation review

The original chapter was read in full before editing. The expanded chapter was
reread in complete successive ranges through line 1063. Later changes to the
nanopq quotation, ANN/fusion diagnosis and mobile BM25 formula were reviewed in
their full local explanations. All eight numbered section anchors, the existing
RRF subsection anchor and source footer were preserved.

| Mechanism | Reasoning and worked case |
| --- | --- |
| Candidate retrieval | Two relevant IDs versus a two-document result; Recall differs from Hit, and repeated overlapping chunks do not constitute two independent evidence sources. Retriever, reranker and final context budgets are distinct. |
| Inverted index | Three-document corpus, document frequencies, posting frequencies, union/intersection and the need for positional information for phrases. |
| TF–IDF and BM25 | Sparse representation, document frequency versus occurrence count, saturation at 2.2, length normalization, positive IDF convention, numeric contribution 5.89779 versus 7.16160 for different document lengths. |
| Exact dense versus ANN | Exact vector neighbors are not human relevance labels; exact top-k overlap and relevance recall can disagree. |
| IVF | Centers 0 and 10; query 4.9 misses the true neighbor at 5.1 when only one list is visited. `nprobe=2` recovers it; all-list IVFFlat differs from all-list IVFPQ. |
| HNSW | Upper-layer greedy search versus a base-layer candidate queue; a synthetic chain illustrates the local-minimum problem without pretending to reproduce HNSW graph construction. M, construction width and search width have different roles. |
| PQ/ADC | Two explicit codebooks, encoding, reconstruction error .07, distance table, approximate .04 versus exact .07/.01, and a collision that loses the order of two documents. Coordinate compression is not whole-service compression. |
| OPQ | Shared orthogonal rotation preserves exact L2 before coding; quantization changes approximations. This differs from whitening, which generally changes the metric itself. |
| Faiss API | Actual train/add/search calls on 512 synthetic eight-dimensional vectors; training leaves the database empty; visiting every IVFFlat list matches Flat under the same metric. |
| Chunking | Twelve tokens with a four-token condition crossing a boundary; overlap preserves the condition but still lacks service name/version. Offsets, duplicated token count, structure, parents and ACL constraints are explained. |
| Query processing | Negation/version preservation, domain expansion, decomposition versus paraphrase variants, HyDE generated documents versus real evidence, and a two-vector averaging example. |
| Filtering | Global top-2 followed by ACL filtering returns nothing; top-2 among permitted documents returns two candidates. ANN filtering cost and authorization are separate concerns. |
| Fusion | Score-scale reversal; explicit RRF ranks/scores and deterministic ties; source truncation, duplicate channels, weighted lists, union recall ceiling and final-cut losses. Historical Anserini results show both a gain and a loss. |
| Versioning | Corpus, preprocessing, encoders, metric, centers/codebooks, statistics and cache identity; a consistent encoder/index/metadata bundle is selected per request. Alias replacement alone does not update in-flight requests or caches. |
| Diagnosis | A source found at ANN rank 18 but cut after fusion is traced separately from a source absent from ANN or lost during parsing. Changed ANN ranks can still affect fusion; the text no longer claims that search width could never affect that case. |

The chapter is a coherent explanation with intermediate arithmetic and six
standalone executable examples, not a set of source-link annotations. Synthetic
examples are explicitly identified as such. No quality claim is inferred from
synthetic distances or API checks.

## Editorial review

Applied no-ai-slop SKILL and eval with the repository's academic qualifications.
Necessary terminology, notation, substantive limitations, worked reasoning and
source excerpts were retained. The new nanopq quotation remains in English.
This is an authorized substantive expansion, not a stylistic rewrite of an
external author's text; the checklist's prohibition on inventing details is not
used to suppress clearly labeled, checked teaching examples.

Removed process commentary about a whitespace analyzer and an admonition about
changing formulas. Replaced an overstrong diagnosis about `efSearch` with the
actual observed loss stage and a separate, testable hypothesis. The prose begins
with exact identifiers and paraphrased queries, not a generic claim about the
importance of retrieval. The style check is not evidence of authorship or
technical correctness; source and example checks are separate.

## Figures and provenance

Eight source figures are used. The six new ones have individual hashes,
dimensions, derivation and license evidence in
`00 Учебник/Assets/Figures/curated/retrieval-review-2026-10/assets.json`:

- Five official Milvus diagrams: IVF construction, IVF search, HNSW, PQ
  codebooks and PQ distance computation. Byte-identical PNG copies, 2560 pixels
  wide, no authored replacement diagrams, no recoloring or label changes.
  Documentation revision is pinned; the separate mutable S3 URLs are fixed by
  downloaded image hashes, not falsely described as Git-versioned image bytes.
- HyDE Figure 1 with its complete legend: crop `[200,200,1630,590]` from the
  1819×2573 rendering of published ACL PDF page 2 at 220 dpi. Output 1430×390.
  The crop is pixel-identical to that region. Caption discloses cropping and
  identifies generated examples as hypothetical, not factual medical answers.
  The ACL policy for 2016-and-later proceedings supplies CC BY 4.0 evidence.

The two historical OpenQA figures now point to the pinned PDF and actual page
numbers rather than nonexistent ignored PNG source paths. RGB comparisons with
independent full-page PDF renders pass. Their existing user-confirmed reuse
status was retained; independent publication permission was not re-established
or silently promoted to an open-license claim in this pass.

Full Milvus, Faiss and nanopq notices are present under `site/public/licenses/`
and match the frozen license files. Course images were not treated as reusable
merely because they were publicly accessible.

## Large-image viewer

Added `BookFooter.astro`, `FigureViewer.astro` and `figure-viewer.ts`, with a
single Starlight footer override preserving the default footer. The original
image links remain ordinary file links. An unmodified primary click or keyboard
activation opens a native dialog; modifier/middle clicks and PDF wrappers are
not intercepted. Controls provide 1×–4× zoom relative to fit width, reset, original
file and close. Zoom is a CSS display change, not a rewritten source image.

The viewport scrolls in both directions. The dialog retains source links and
KaTeX caption markup, locks background scrolling while open, and restores focus
on close. The first QA pass found duplicated mathematical caption text from
`textContent`; cloning the existing caption nodes fixed the visible duplication.
Explicit automatic margins fixed off-center dialog placement. UI labels follow
the document language; no English chapter content was edited.

## Current checks

- `pnpm --dir site check`: 0 errors, 0 warnings, 0 hints across 12 Astro files.
- `pnpm --dir publishing test:unit`: 250 tests in 18 files pass, including 48
  asset registry tests and 3 new viewer geometry/click-eligibility tests. These
  subset counts are not additional tests on top of 250.
- `pnpm --dir site build`: 1961 pages built; completed 18:26:02 local log time.
- `publishing/tools/check_retrieval_examples.py`: 1172 counted grouped parameter
  cases pass; this includes standalone/shared execution of six blocks, random
  BM25 versus document-scan reference, IVF versus exact distance, PQ table versus
  reconstructed L2, orthogonal identity, chunk span coverage, fusion/filters,
  frozen-source hashes, figure pixel checks, notices, quote and built anchors.
  Faiss CPU version 1.15.1 ran in a temporary venv. No dependency was added to
  the publication project for this check.
- `pnpm --dir publishing test:output`: 13 tests pass, including the new single
  accessible dialog and preserved source-file link assertion.
- `pnpm --dir publishing check:links`: 0 broken internal routes, fragments or files.
- `pnpm --dir publishing check:source-coverage`: passes as a structural check;
  its course/artifact counts are not semantic completeness claims.
- Scoped `git diff --check`: passes. No unrelated edits were staged or reverted.

Site check and site build must run sequentially: both regenerate derived content.
An earlier concurrent run failed while generated files were being replaced;
the sequential check/build and final build above succeeded. That transient
failure was not hidden or interpreted as a source-image viewer failure.

## Rendered QA

Actual in-app browser inspection at 1365×900 and 390×844, not just HTML parsing:

- All eight figures opened with the expected original file and attribution,
  without body-width overflow. All six new originals had already been inspected
  separately before embedding. Tall figures remain fully available by scrolling,
  rather than being cropped by a fixed-height image container.
- PQ zoom, two-axis viewport scrolling, Escape/button close and restored source
  focus were checked on desktop. Mobile 4× IVF/PQ views make their small labels
  readable; both axes scroll and fit-width resets position and scale.
- Formula QA found an overwide BM25 expression on mobile. Factoring out its
  saturated term weight preserved the definition and calculation. Final inspection
  shows eight display formulas within the 343-pixel article width, no KaTeX errors
  and no horizontal body overflow.
- The existing mobile SimCSE image in chapter 60 opens in the viewer and returns
  correctly. Its original source and chapter text were not changed. English UI
  labels were checked on the existing English chapter 60 page without translating it.

Evidence files under `/tmp/` include
`bookvar-retrieval-{retriever-reader,tfidf,ivf-build,ivf-search,hnsw,pq-search,hyde}-{desktop,mobile}-2026-10-20.jpg`,
`bookvar-retrieval-pq-codebooks-mobile-2026-10-20.jpg`,
`bookvar-retrieval-pq-viewer-desktop-2026-10-20.jpg`,
`bookvar-retrieval-pq-viewer-pan-desktop-2026-10-20.jpg`,
`bookvar-retrieval-ivf-mobile-zoom-2026-10-20.jpg`,
`bookvar-retrieval-pq-mobile-pan-2026-10-20.jpg`,
`bookvar-retrieval-bm25-mobile-final-2026-10-20.jpg` and
`bookvar-retrieval-viewer-embeddings-regression-mobile-2026-10-20.jpg`.
Check logs use the same `bookvar-retrieval-…-2026-10-20.log` prefix.

No unresolved explanatory or rendering issue was identified in this chapter's
selected scope. This is not a claim to cover every ANN index or query rewriting
method. The next editorial chapter is `62 Reranking — cross-encoder и late interaction.md`.
