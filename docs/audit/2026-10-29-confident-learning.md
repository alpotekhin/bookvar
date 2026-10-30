# Confident Learning: explanation and source review

29 October 2026. Local `editorial-route-alignment`; no publication.

## Scope and revision

The label-error gap in Harvard Data Selection is developed inside existing
chapter 41b, at `/textbook/pretraining/data-selection-scores/#confident-learning`.
The chapter remains source-language English. No EN body, other accepted chapter,
reader-route edge or historical source archive is rewritten.

Current chapter SHA-256:
`18aa3e0c623deceaf7bd475e03f045c7a5e5e8b68ccc7921297e106fa4b04fe4`.
The bounded 411-line addition and two frontmatter date changes restore exactly
`4457573c808536ae88f021c72f3d8be98d9ba9866b7f2b23af99da5c322acff2`.
The previous 615 lines, five programs, two figures and navigation are preserved.
The new total is 1026 lines, eight programs, four source figures.
The restoration guard rejects changes inside the new block and does not erase
unreviewed changes elsewhere. Historical acceptance hashes remain unchanged.

## Source comparison

Primary article: Northcutt, Jiang and Chuang, *Confident Learning: Estimating
Uncertainty in Dataset Labels*, JAIR 70 (2021), 1373–1411,
doi:10.1613/jair.1.12125. The complete publisher PDF was already archived;
no duplicate course or paper download was needed.

Read in this pass: extracted lines 1–950 (abstract, introduction, framework,
methods, complete main theory section, CIFAR evaluation and ImageNet human
validation; the Figure 4 caption ends outside this range); lines 1930–2110
(end of proof discussion, complete Appendix B algorithms and opening of C).
Appendix A is not claimed fully read or independently proved. Source PDF pages
6 and 14 inspected as complete pages, then both selected panels inspected.
The official cross-validation guide's complete explanatory body and pseudocode
were read. Pinned Cleanlab `count.py` functions `get_confident_thresholds`,
`compute_confident_joint`, and `calibrate_confident_joint` were read in full
at commit `750625747de1b26d8530954f51f0530bd0b51d3c`; the module as a whole
and multilabel helpers are not claimed read.

| Reader mechanism | Source | Retained reasoning and qualification |
| --- | --- | --- |
| Given versus latent label and class-conditional noise | §2, pp. 1375–1376 | Joint, row/column meanings, dependence on model, instance-dependent counterexample |
| Out-of-fold predictions | §3 opening; official complete CV guide | Fresh fits, held-out rows, stable ordering, preprocessing/groups, executable 60-record CPU example |
| Class thresholds | §3.1 Eq. 2, pp. 1378–1379; threshold function | Mean uses given labels, not true labels; complete original English explanation excerpt |
| Confident counts | §3.1 Eq. 1; Algorithm 1; counting function | Full 12-record table, assigned/unassigned, ordinary argmax comparison, collision discrepancy |
| Joint calibration | §3.1 Eq. 3; Algorithm 2; calibration function | 10 counted versus 12 original records, row rescaling, fractional counts, both conditional directions, zero-row limit |
| Rank and review | Complete §3.2 | Direct candidates versus class/noise-rate budgets; concrete margins and stable tie order |
| Source/software difference | Eq. 1, collision paragraph, Algorithm 1, pinned source | Restricted versus global argmax counterexample, tolerance, diagonal floor and integer rounding |
| Review limits | §5.2, pp. 1387–1388 | Original issue/error wording, 500-record review and 58% scope, valid rare C counterexample, review log |
| Theory and evaluation | Complete §4; §5.1, Figure 2 and Table 4 | Sufficient conditions not guaranteed by softmax; joint accuracy distinct from candidate precision and training quality |

Harvard pinned source lines 1386–1403 were read in full. Its broad claim that
confident disagreement is more commonly an annotation mistake is not imported
as a universal fact. The explanation follows the primary algorithm instead.
Data cleaning is not extended to sequence-level LLM targets without a task
definition. Existing EL2N/GraNd/forgetting and cost explanations are retained.

## Executed examples

`publishing/tools/check_confident_learning_examples.py` executes the actual
three new blocks, not copies in a separate notebook. It verifies 60 out-of-fold
rows from three fresh logistic regressions, threshold vector (.49,.475,.4875),
raw joint [[2,1,0],[1,2,1],[0,1,2]], row-calibrated joint, total one, marginal
identities, estimated off-diagonal mass 7/18, candidate order [3,7,11,8] and
unassigned IDs [4,12]. Independent checks include 30 input permutations,
argmax's extra candidate 12, both conditional directions, the collision
counterexample and a valid rare example falsely flagged by the same predictions.
NumPy 2.3.5, scikit-learn 1.9.1. No real-dataset precision is inferred from
assigned labels/probabilities.

The full previous numerical checker now executes all eight actual programs;
its EL2N/gradient/forgetting/ID checks are unchanged. A dedicated test guards
all previously accepted bytes. Cleanlab package execution, GPU work, paper
benchmark reproduction and a real annotation audit are not claimed.

## Illustrations and rights

Two complete panels from the publisher PDF, Figure 1 (300 dpi, 1325×1100)
and Figure 2 (450 dpi, 2725×794). The second retains all three panels, all axes,
the shared colour scale and subpanel labels. Neither figure is redrawn, translated
or recoloured. Adjacent captions explain what the matrices do and do not show;
one cell is worked through in text. Both renders are reproducible from recorded
page/crop coordinates and checked against the saved PDF SHA.

The 2021 article uses **JAIR License Version 1.0**, not the journal's later CC BY
policy. Full bibliographic citations, DOI, article link, AI Access Foundation
copyright notice and saved terms are adjacent to the figures/excerpts. Figure 3
is not copied: its ImageNet photographs raise separate third-party provenance
questions. The Cleanlab CV graphic is linked through its guide, not reused
without asset-specific permission. Asset entries distinguish the PDF bytes
from the rendered PNG bytes and are registered centrally.

## Final acceptance

The complete final 1026-line chapter was reread in order. The no-ai-slop
checklist prompted three corrections within the addition: a concrete valid
subgroup replaces anthropomorphic wording; the OOF example distinguishes
stored labels from the hidden corruption indicator; an unnecessary editorial
aside was removed. Existing explanations, source quotations and technical
qualifications were retained. This is an editorial review, not an authorship
detector or a claim that a checklist proves teaching quality.

Observed checks on the saved revision:

- The three new programs and independent numerical checks pass on the final
  chapter. All eight programs and the unchanged earlier checks also passed
  before the last three prose corrections; those corrections changed no code
  or numerical inputs. Logs: `/var/tmp/bookvar-cl-final-examples-2026-10-29.log`
  and `/var/tmp/bookvar-cl-all-examples-2026-10-29.log`.
- The full unit suite initially passed 657 tests and rejected the new PDF's
  `Originals` archive path. The archive-path whitelist was extended while
  retaining hash, rights, dimension and reproduction checks. The affected
  registry and Confident Learning suites then passed all 152 tests, including
  exact regeneration of both saved crops. Logs:
  `/var/tmp/bookvar-cl-final-unit-2026-10-29.log` and
  `/var/tmp/bookvar-cl-final-assets-2026-10-29.log`.
- The final build produced 1989 pages. All 165 built-output tests pass after
  updating this chapter's explicit expectations from five to eight programs
  and from eleven to seventeen display equations, and checking both new
  images and the license file. Both locale routes retain the existing explicit
  EN fallback; no English body was created. Logs:
  `/var/tmp/bookvar-cl-final-build-2026-10-29.log` and
  `/var/tmp/bookvar-cl-final-output-2026-10-29.log`.
- Internal route, fragment and file checks report zero broken links. Course
  coverage checks pass; this is structural evidence, not full course coverage.
  Logs: `/var/tmp/bookvar-cl-final-links-2026-10-29.log` and
  `/var/tmp/bookvar-cl-final-coverage-2026-10-29.log`.
- All 149 current chapter hashes match the revision ledger. This verifies
  identities; it is not a fresh reread of the other 148 chapters. Scoped
  `git diff --check` passes. The whole dirty checkout still reports pre-existing
  trailing spaces in imported Harvard labs/slides, outside this change.

The final local page was inspected at desktop and 360-pixel mobile widths.
Both figures load; neither image panel is cropped. The matrix figure's three
panels and colour scale are visible together on desktop. On mobile, zoom and
horizontal panning expose both ends with readable labels. Formulas render
without KaTeX errors; document width stays within the viewport. The wide table
scrolls inside its container. A transient image load during concurrent asset
replacement was rechecked after the build and resolved. Screenshots:

- `/var/tmp/bookvar-cl-desktop-2026-10-29.png`
- `/var/tmp/bookvar-cl-desktop-matrices-2026-10-29.png`
- `/var/tmp/bookvar-cl-mobile-formulas-2026-10-29.png`

The label-error explanation gap is closed in the source map with this exact
chapter revision and evidence. Image deduplication and cold embedding-ID gaps
remain open. Nothing was staged, committed, pushed or deployed.
