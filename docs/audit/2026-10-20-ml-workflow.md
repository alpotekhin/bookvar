# ML workflow: chapter acceptance

Date: 2026-10-20.
Canonical chapter:
`00 Учебник/19 Deployment, Reliability и MLOps/01 ML workflow.md`.
Route: <http://127.0.0.1:4330/textbook/deployment/ml-workflow/>.
Final chapter SHA-256:
`498a80237b35db4bf8376d2bc0dade25d6065f955a71958ce53d4ec9c515115e`.

## Scope and reading

The entire original 297-line chapter was read before editing. The revised
635-line chapter was reread sequentially in ranges 1-220, 221-440 and 441-end.
The final correction replaces a misleading statement that future/late feature
records were already available at prediction time; the example now explicitly
uses the stored history and checks both event and availability times.

This accepts this chapter's narrative, examples, selected source reuse and
local rendering. It does not accept the entire textbook, transfer of every
linked course, English parity, interactive practice or production behavior.
No English file, commit, push, deployment or infrastructure mutation was made.
The broader page-by-page task remains active. The next canonical chapter is
`02 MLOps.md`.

The deliverables are the rewritten canonical chapter, six unchanged original
course figures, eleven source snapshots, exact original-language excerpts,
provenance/license records, a CPU/source-fidelity checker and targeted registry
tests. Route, previous/next links and nine explicit legacy heading anchors
are retained.

## Explanation review

| Subject | Source basis inspected | Explanation and boundary |
|---|---|---|
| Prediction and update cycles | Complete Google ML Pipelines article body, exercises, challenges and Figures 4-10 | Serving is separated from data, training and candidate validation; update cadence depends on labels, task change, cost and candidate results |
| First working pipeline | Complete Rules of ML 1-6 | Original Rule 4 passage plus a TF-IDF/rule baseline; model complexity is compared on the same task and full path |
| Task, action and resource constraints | Complete selected Harvard problem-definition sections and Google first-pipeline context | First-message support routing; A/B/C confusion matrices, cost 5FN+FP, recall requirement and engineer-capacity constraint; only C passes the two validation conditions |
| Point-in-time inputs | Rules of ML 29, 31 and 32, within complete selected rules 23-35 | Event time and actual feature availability are both enforced; late historical values cannot be treated as already observed |
| Delayed and selected labels | Google feedback/skew discussion and explicit worked example | A 72-hour label horizon excludes immature cohorts; old routing decisions are not assumed to be independent semantic truth |
| Experiment and artifact identity | Harvard reproducible-artifact sections and EDLS week 7 original files | Data, transformations, tokenizer, weights, runtime and threshold are dependencies; manifest identity is distinguished from actual file verification, trusted signing and numerical repeatability |
| Change propagation | Complete Harvard Systems Thinking section and subsections | A concrete dependency table replaces a synthetic exponential rework law; prior evidence remains evidence about its original conditions |
| Validation and release | Complete Harvard Evaluation and Validation section; original Google Rule 5 | Fixed-input pipeline tests, statistical quality, replay, shadow, limited release and designed A/B experiments have different evidentiary roles |
| Online/offline inference | Complete Google serving sections and Figures 5-7 | Request-time prediction versus precomputation is independent of training cadence; cached results retain input/model identity and freshness rules |
| Export versus training | Entire pinned EDLS week 7 README and train_model.py; complete imported slide text and Python-server README | The file named train_model.py traces a pretrained VGG16 rather than running optimizer/backward training; imported source code is not executed or certified as a valid release |
| Feedback selection | Rules of ML 28-35 and a finite stratified example | Auditing 100/100 predicted positives and 100/900 predicted negatives gives naive recall 81.8%, weighted estimate 33.3%; known nonzero inclusion probabilities and sampling uncertainty are explicit |

Source scopes and hashes are recorded in
`05 Источники/Originals/ml-workflow-2026-10-20/references.json`.
The pinned Harvard QMD is commit
`45ecc8d82fcae70c149cdce550d3b3d3411df913`; the existing complete imported
course is a different historical edition, not claimed to be byte-identical.
Selected Harvard sections were read completely: lines 55-201, 990-1023,
1632-1782, 1783-1859 and 2101-2164. Selected complete Google Rules of ML
are 1-6 and 23-35, including the training-serving skew introduction.
No whole-book or whole-course reading claim is made.

Google's automatic best-model and daily-retraining recommendations are not
presented as universal guarantees. Harvard's synthetic exponential rework
formula, generic traditional-software/ML binary, unsourced scenario numbers
and universal determinism language are not retained as established laws.
The source archive preserves the actual wording rather than silently editing
the originals.

## Original text and figures

Two English excerpts from Google Rules of ML, rules 4 and 5, retain their
original wording and language. Only Markdown blockquote/list layout changes.
Each has adjacent author/source attribution and CC BY 4.0. The checker strips
only Markdown markers and normalizes whitespace, then confirms that each
entire excerpt occurs in the archived original HTML.

Six original Google ML Pipelines PNGs are embedded with nearby source links,
figure numbers and CC BY 4.0 attribution:

| Source figure | File | Original pixels |
|---|---|---|
| 4, four pipelines | google-pipes.png | 742 x 906 |
| 8, data pipeline | google-data-process.png | 622 x 262 |
| 9, training pipeline | google-train.png | 622 x 262 |
| 5, online/offline predictions | google-offline-online.png | 750 x 273 |
| 6, pre/postprocessing | google-prediction-process.png | 504 x 161 |
| 7, prediction/application logs | google-prediction-log.png | 622 x 262 |

All six are complete byte-for-byte copies, not crops, redraws, translations,
bitmap upscales or generated illustrations. Original labels and language are
unchanged. The official archived article retains each exact image URL and
the CC BY 4.0 license link. Local assets.json, NOTICE.md and the global asset
registry identify sources, output/source hashes and license-evidence hashes.
The existing EDLS MIT notice is preserved and hash-checked.

The older Harvard dual-pipeline and exponential-cost illustrations are no
longer embedded in this narrative. Their shared original files are preserved.
No new license clearance for unrelated historical assets is claimed.

## Separate editorial review

The no-ai-slop skill and eval were read before substantial editing. The whole
chapter was reviewed for empty setup, definition-only fragments, unsupported
generalizations, vague attribution and mixed-language filler. A continuous
support-routing example supplies the reasoning rather than a list of stage
definitions. Necessary technical terms, mathematical notation, caveats,
quotations, source links and figures are retained.

The final source comparison and mechanism review are separate from this
wording review. Passing a style checklist is not proof of teaching completeness
or human authorship.

## Executed checks

- `check_ml_workflow_examples.py` executes all four actual chapter Python blocks.
  It checks 200 independent cases each for confusion-matrix metrics,
  point-in-time feature choice, release identity and inverse-inclusion count
  arithmetic: 800 additional cases total. Invalid counts, missing/late inputs,
  NaN/infinite manifest values and invalid inclusion probabilities are tested.
  These are CPU teaching examples, not experiments on a trained model.
- Eleven immutable source hashes, the EDLS MIT notice, six source/output image
  identities and dimensions, two exact English excerpts, nine legacy anchors
  and previous/next links pass the same checker.
- `pnpm --dir publishing exec vitest run tests/asset-registry.test.ts`:
  41 tests passed. The new strict branch checks all six official Google assets
  and rejects missing attribution, incorrect source/license/hash evidence,
  a crop derivation and unverified metadata.
- `pnpm --dir site build`: success; 1961 pages built and search index completed.
- `pnpm --dir publishing check:links`: zero broken internal routes, fragments
  or files.
- `pnpm --dir publishing test:output`: 12 tests passed.
- Targeted `git diff --check`: clean.

Build and check logs:

- `/tmp/bookvar-workflow-build-2026-10-20.log`
- `/tmp/bookvar-workflow-links-2026-10-20.log`
- `/tmp/bookvar-workflow-output-2026-10-20.log`

## Actual browser inspection

The built route was opened in the existing in-app browser. All six figures
were visually inspected in their chapter context at 1365 x 900 and 390 x 844.
The tall four-pipeline diagram was inspected across consecutive vertical
views; its original file was opened from the image link and verified as
742 x 906. No crop is applied to these files or their layout.

On desktop the document is 1350 px wide including its normal scrollbar space;
content is 687 px. All six figures load with their correct natural dimensions.
On mobile the document is 375 px with a 343 px content column, without
horizontal page overflow. All figures preserve their aspect ratios. Tables
and four code blocks scroll within their own 343 px boxes. Six formula
containers fit; KaTeX errors are zero. The sampled-recall formulas were also
visually inspected on mobile rather than accepted solely from DOM dimensions.

Small labels in the original overall and online/offline diagrams need
enlargement on a phone; scaling an unchanged course image to 343 px cannot
make all of its text comfortably readable. The image link opens the complete
original. This is an acknowledged readability limit, not a claim that tiny
labels are fully legible in the inline mobile view.

Proof screenshots:

- `/tmp/bookvar-workflow-desktop-2026-10-20.jpg`
- `/tmp/bookvar-workflow-mobile-2026-10-20.jpg`

The desktop proof was embedded in commentary, not only returned by a tool.
Temporary viewport overrides are reset after inspection. This is visual QA
of the local static output, not validation of model training, online serving,
resource limits, product causality or a deployed release.
