# Synthetic data and curricula: page review

Date: 2026-09-22. Scope: one canonical Russian chapter and its source/asset
records. This is not acceptance of the whole textbook. No commit, push or
deployment was performed. No English translation was authored.

## Reviewed text

- Chapter: `00 Учебник/11 Pre-training и Scaling/53 Синтетические данные и учебные программы.md`.
- SHA-256: `e95096e9930265b930fd7ed967a742dfadb8e4d32bedacd952dba7020c40a93d`.
- The original chapter and rewritten page were read in full. Three independent
  source reviewers read the complete draft; their mathematical and factual
  corrections were checked and incorporated by the main editor.
- Preserved anchors: synthetic-transformations, teacher-data,
  posttraining-environments, curricula-replay, synthetic-evidence.

The rewrite distinguishes generated instructions, responses, judgments and
environments; hard gates from ranking; question from response filtering;
task creation from solution collection; available corpus size from token
exposure; mixture from curriculum, replay and LR annealing; replacement
from accumulation. Examples include acceptance bias, repeat counts,
document/token proportions, finite-sample rare-category loss and generation
cost per accepted response.

## Source-specific checks

- Self-Instruct: fixed generator during pool growth, input/output-first,
  heuristic filtering and the limited 200-instance manual audit.
- Nemotron-CC: separate labeling and rewriting models, maximum quantile-rank
  ensemble, real/synthetic branches, unverified factual fidelity.
- OpenThoughts3: math/science exact dedup but no code dedup; 16 responses
  retained per question record; final response filtering omitted. Unequal
  63,200/31,600-row comparison is not compute-controlled.
- SWE-Smith/Rebench/Zero: bug injection versus real PR mining; separate
  solution/test patches; runtime validation versus protocol filtering;
  evaluation-time Execution column is not a training-data ablation.
- OLMo 2: 832.6B/10.7B are available corpora, not consumed-token counts;
  independent late branches and weight averaging. Inconsistent exact
  fractions from Table 13 were not repeated.
- Collapse: limited empirical-frequency example separated from LLM claims;
  Shumailov panels use different epoch budgets; accumulation is not a
  universal guarantee and TinyStories itself is synthetic.

## Executed real example

SWE-smith task:
`un33k__python-slugify.872b3750.func_basic__nis2b6y4`.
Upstream commit:
`872b37509399a7f02e53f46ad9881f63f66d334b`.
Bug patch SHA-256:
`cd80f21f8aa8c6eb77aabbe215bfdaab66924a44010c1489cbabd65e405b92d3`.

Executed the 11 upstream CLI tests against extracted original functions:
clean 11/11, mutated 9/11, delimiter-only repair 11/11, full repair 11/11.
Additional input A->B->C detects the remaining split-limit bug after partial
repair. Also checked that the generated issue command needs -- to terminate
the replacement arguments. No Docker run, full 81-test suite, teacher
trajectory or training experiment was claimed.

## Visuals and checks

Nine original figures/tables are embedded. New licensed source assets:
Nemotron Figure 3; high-resolution OpenThoughts Figure 4; full Rebench PNG;
corrected Shumailov Figure 1. Existing licensed Self-Instruct illustration
is reused without duplication. PDF and PNG hashes/crops/licenses are in the
asset registry. Gerstgrasser's figure was not imported because its reuse
license was not established.

Desktop 1360px and mobile 390px: all nine images decoded, kept aspect ratios,
fit the article and link to full-size assets. The original OpenThoughts
palette is pale; it was not recolored. Dense source labels need zoom on
mobile. Screenshots of each image were personally inspected.

Final verification:

- Site build: 1961 pages, exit 0.
- Publisher unit tests: 210/210.
- Generated-output tests: 12/12.
- Stanford/Berkeley Python importer tests: 46/46.
- Stanford snapshot check and both course-ledger checks pass.
- Built links: zero broken internal routes, fragments or files.
- Browser: 18 image checks; zero KaTeX errors, wide display formulas,
  unresolved wiki syntax, page overflow or page JavaScript errors.
- git diff --check: clean.

Local evidence: `/private/tmp/bookvar-synthetic-visual-review-20260922/`
and `/private/tmp/bookvar-synthetic-*-20260922.log`. These temporary files
are diagnostic evidence, not permanent published artifacts.
