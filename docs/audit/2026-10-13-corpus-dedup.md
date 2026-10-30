# Chapter 41a editorial and numerical audit

Date: 2026-10-13. Scope: `00 Учебник/11 Pre-training и Scaling/41a Дедупликация, PII и контроль качества корпуса.md`.
No commit, push, deployment, English edit or shared registry modification.

## Content

Read the original chapter in full, the applicable editorial instructions,
CS336 Lecture 14 deduplication code, Assignment 4 sections 2.4/2.5/3/4 and
its PII, toxicity and deduplication tests. Read the relevant algorithm/results
sections of Lee et al. (ACL 2022), and Presidio's original documentation.

The chapter now develops:

- detection vs transformation of PII, contextual ambiguities, synthetic-only
  examples, span errors, idempotence, and avoiding PII in logs;
- toxicity labels vs quality, source-domain transfer, threshold trade-offs,
  and why sanity tests do not estimate classifier quality;
- exact equality of normalized strings, versioned normalization, collision
  buckets with a real equality check, deterministic representatives;
- the distinction between keeping one copy and Assignment 4's removal of
  **all** occurrences of a repeated line;
- document vs substring deduplication and coherence/boundary failure modes;
- word shingles, Jaccard, empty-set handling, an explicit MinHash argument,
  estimator variance, same-position LSH bands, every probability in a worked
  table, exact verification, bucket size and candidate-recall limits;
- nontransitive similarity, connected components vs direct representative
  matching, and actual distribution of component sizes in the paper;
- contamination protocols, fixed evaluation versions, development/validation/
  final-test separation, and interpreting perplexity changes on duplicate data;
- document accounting vs byte/token accounting and fixed-budget model tests.

All 20 existing `source_unit_id` comments retained. Original explicit anchors
retained; compatibility anchors added for changed section titles. Next chapter
now points to `42 Next-token prediction`; chapter 53 remains a related topic.

## Visuals and provenance

Three original figures are present: Presidio's analyze/anonymize diagram and
Lee et al. Figures 1 and 2. Each full source and final crop was inspected.
Complete asset URLs, authors, licenses, pinned revision, checksums and crop
commands are in:

`00 Учебник/Assets/Figures/curated/corpus-dedup-review-2026-10/NOTICE.md`.

The ACL source PDF was obtained from its publisher using HTTP byte ranges
after full downloads stalled, reconstructed to exactly 634490 bytes, validated
with `pdfinfo`, and rendered. It is retained in the tracked source directory:

`05 Источники/Papers/Source PDFs/lee-deduplicating-acl2022.pdf`.

Remove **only this chapter's current used_in reference** from these old assets;
do not delete their files or other chapters' uses:

- `curated/stanford-cs336-2026/data-corpus/minhash-characteristic-matrix-trace.png`
- `curated/stanford-cs336-2026/data-corpus/minhash-agreement-trace.png`
- `curated/stanford-cs336-2026/data-corpus/lsh-banding-probability-trace.png`

Reason: screenshot text is less readable than typeset prose/tables; the old
LSH trace also displayed globals b=3,r=4 alongside a return value calculated
with b=5,r=10, making the visible example ambiguous. The numerical content is
retained as text and independently checked.

## Executed CPU checks

Command:

```sh
/Users/aleksan.potekhin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  /tmp/corpus-dedup-validation.py
```

Result: all checks passed, exit 0.

1. Executes the actual chapter Python block, including all six input orderings
   and a constant hash function forcing every record into one bucket.
2. Confirms normalized duplicates, distinct strings surviving collisions,
   Unicode equivalence, empty input and punctuation preservation.
3. Confirms keep-one vs remove-all line behavior and 32/64-bit collision values.
4. Confirms the two-sentence shingle example and explicit empty-set rejection.
5. Enumerates all 120 permutations of the five-item union: 72 matching minima,
   exactly 0.6; verifies the finite-signature standard deviation.
6. Confirms band positions matter, candidates below threshold remain, and all
   eight values of the LSH probability table are correct.
7. Confirms the nontransitive A-B-C example and two direct-match representatives.
8. Checks a fictitious masking fixture's idempotence, document counts, source
   IDs, explicit anchors, all three local image paths and the next-chapter link.

The masking fixture is a transformation-contract check, **not** a production
PII detector. No classifier benchmark, corpus-scale precision/recall estimate,
GPU training run, or claim of complete privacy/contamination removal is made.

## Remaining integration evidence

The main editor owns asset-registry integration, common tests/build/link checks
and rendered browser review. This audit records source/content/image/CPU work
only; file existence is not a claim of publication or UI correctness.
