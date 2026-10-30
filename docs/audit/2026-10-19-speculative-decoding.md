# Chapter 58: speculative decoding — editorial and visual review

Date: 2026-10-19

## Deliverable and scope

Canonical chapter: `00 Учебник/14 Inference и оптимизация/58 Спекулятивное декодирование.md`

SHA-256: `4f60e570288e3dd4ec403f160a396f683f7a69e5db95433a73e0a1b4b8c73097`

Local route: <http://127.0.0.1:4330/textbook/inference/speculative-decoding/>

The complete original 333-line chapter was read before editing. The replacement was reread in order; the subsequent equation layout and source-figure caption changes were reviewed separately. This is a source-based explanatory revision of this chapter, not completion of the book or integration of every source course.

Canonical RU Markdown, source snapshots, six original figure assets, shared provenance, scoped image sizing, a CPU checker and this audit are delivered locally. English chapters, commits, pushes and deployment were not changed.

## Explanation and sources actually read

- Leviathan, Kalman, Matias, arXiv:2211.17192v2: sections 2.1–2.3, 3.1–3.3 and Appendix A.1 in full; Figure 1 and Figure 5, including the source caption. The chapter keeps the Chen/CS336 convention draft p, target q and explicitly maps the reversed notation in Leviathan's original material.
- Chen et al., arXiv:2302.01318v1: Auto-regressive Sampling, Conditional Scoring, Modified Rejection Sampling, Choice of Draft Models, Results/Table 1, lookahead discussion/Figure 1 and Theorem 1. The final source file selected by the paper is speculative_sampling_results_2.pdf; the unused earlier file in the archive was not substituted for it.
- Pinned Stanford CS336 lecture 10, speculative_sampling(), lines 507–550: original algorithm, binary-vocabulary proof, source figures and draft variants.
- EAGLE arXiv:2401.15077v3: section 3 and Appendix A.2 in full. The published ICML 2024 page 4 and Figure 6 were also inspected. Cross-entropy uses the teacher distribution induced by its true feature, not merely a hard observed token ID.
- EAGLE-2 arXiv:2406.16858v2: sections 3–4 in full; published EMNLP version sections 4.1–4.2 and complete Figure 7. Expansion, global reranking, path-product approximation, tie handling and ancestor-only attention remain distinct.
- EAGLE-3 arXiv:2503.01840v3: section 3 in full. The published NeurIPS section 3/Figures 5–6 and section 4.3/Table 3 were checked. The 1.38x throughput result used a length-3 chain, not a tree, with SGLang v0.4.4, one H100, LLaMA 3.1 8B, batch 64 and MT-bench.
- Hugging Face Universal Assisted Generation article: tokenizer conversion, prefix alignment and KV synchronization sections. No claim that heterogeneous vocabularies can be inserted directly into the single-vocabulary ratio is made.
- The existing Efficient DL Systems practice/lecture links are retained. Its full lecture and homework were not executed or claimed as newly integrated.

Mechanisms now have explicit intermediate states rather than just definitions:

1. Greedy cycle A, B, C → A, B, D and invalidation of later conditional logits.
2. Three-token sampling example with accepted mass (0.2, 0.3, 0.1), rejection mass 0.4, residual (0, 0.5, 0.5) and the wrong direct-resampling counterexample (0.28, 0.5, 0.22).
3. Full sampled cycle B, A, C → B, B on rejection, and B, A, C, C on full acceptance.
4. Target KV transaction with an explicitly pending last token; token selection is not confused with KV materialization. Draft and target caches need not have equal lengths at every intermediate instant.
5. Per-position acceptance versus total progress, total-variation identity, the capped-geometric simplification and a latency table with stated hypothetical costs.
6. EAGLE shifted-token input, soft teacher training target, dynamic tree selection, depth-based position IDs, greedy tree verification, and the current residual/actual proposal required for sampling siblings.
7. EAGLE-3 multi-level feature fusion, feedback of its own unconstrained states and branch-specific training-time attention.

## Style and source fidelity

The no-ai-slop review was separate from the mechanism review. Artificial framing and ambiguous statements were replaced with concrete state transitions, calculations and limitations. Necessary technical contrasts, notation and normal Russian punctuation were retained. Two attributed English passages remain verbatim; no translation was substituted for those excerpts.

Snapshots and exact selectors are under `05 Источники/Originals/speculative-2026-10-19/`. Seventeen source-file hashes are checked. The ACL quotation uses a left-column PDF extraction to avoid interleaving the adjacent figure's labels into the source paragraph.

The previous route, four source-unit IDs and old heading fragments are preserved; renamed acceptance and worked-example headings retain explicit legacy anchors.

## Figures and provenance

Eight original illustrations are embedded:

- Leviathan Figure 1, original raster from the versioned source archive.
- Leviathan Figure 5, full render of the standalone vector time trace.
- Existing Chen Algorithm 2 and Table 1 source images.
- Chen Figure 1, full render of the final standalone vector source.
- EAGLE ICML Figure 6, complete shifted-input network artwork extracted from page 4.
- EAGLE original repository overview, unchanged at commit cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b.
- EAGLE-2 Figure 7, complete artwork extracted from published PDF page 5.

The new manifests record author, title, direct source, license evidence, hashes, page/render sizes and exact crop coordinates. CC BY 4.0 is verified for Leviathan, Chen and the ACL publication. PMLR's publication agreement, clauses 2–3, verifies the CC BY 4.0 grant and required original-publication link for the ICML figure. The repository overview retains its Apache-2.0 provenance. Four PDF-derived images are byte-identical to fresh Poppler renders with the recorded parameters.

No new EAGLE-3 art is embedded. Its published Figures 5–6 are linked directly to PDF page 5. A permissive artwork license was not established; arXiv's non-exclusive distribution license was not relabeled as CC BY. This is an explicit remaining illustration-ingestion item, not a missing textual mechanism explanation.

## Checks on the delivered revision

`publishing/tools/check_speculative_examples.py` passes:

- Seven Python examples extracted and executed from the actual chapter.
- Exact agreement for all 81 length-four sequences at each K=1,2,3: 243 sequence probabilities in total, using prefix-dependent toy models and every rejection/all-accepted branch of the displayed verifier.
- 502 accepted/residual-mass cases including disjoint support and p=q.
- 500 fixed-logit greedy checks.
- KV prefix/pending-tail checks across n=1..19, K=1..9 and every accepted depth.
- Capped-geometric expectation checks including alpha=0 and alpha=1.
- 300 ancestor-mask attention checks against separate path evaluation.
- 300 randomly generated reranking trees with every node budget and shallower-first ties.
- The second-sibling residual example, exact excerpt/source hashes and figure rerenders.

These are CPU algebra, finite-probability and provenance tests. No neural drafter, EAGLE model, real KV kernel, quality benchmark, GPU serving or measured speedup was executed.

Final build: successful, 1961 pages.

Post-build checks:
- Asset registry: 37 passed.
- Built links: 0 broken internal routes, fragments or files.
- Site-output tests: 12 passed.
- Targeted git diff --check: clean.

## Actual browser review

Desktop 1365 × 900 and mobile 390 × 844 were checked. All eight images load and preserve aspect ratio. The two more vertical EAGLE figures use a scoped 28rem maximum width: shifted-input diagram 448 × 419 px, tree/mask 448 × 802 px on desktop, both at the 343 px content width on mobile. No source image was stretched or clipped.

The overall repository overview was too small for its detailed labels; the separate ICML shifted-input diagram was therefore added and inspected at both widths. Wide time traces, algorithm screenshots and the overview still need their full originals for small labels on a phone. Clicking the shifted-input diagram was verified to open its actual 700 × 655 image.

Two mobile display equations originally exceeded the content width. They were split into aligned lines. Final result: 11 display equations, zero KaTeX errors, zero display-equation overflows; mobile document width 375 px within a 390 px viewport. Wide tables/code retain their own scroll containers.

Proof screenshots:
- /tmp/bookvar-speculative-review-desktop-2026-10-19.png
- /tmp/bookvar-speculative-review-mobile-2026-10-19.png (complete EAGLE-2 tree/mask)
- /tmp/bookvar-speculative-review-mobile-eagle-2026-10-19.png (shifted-input network)

## Remaining work

This version is local only. The English page remains unchanged under the current task constraint. EAGLE-3 figures remain direct source links, pending verified reproduction rights or a suitable licensed original illustration.

Next canonical chapter: 58a, distributed inference. It was not revised in this turn.

