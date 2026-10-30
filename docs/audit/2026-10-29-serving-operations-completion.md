# Serving, benchmarking and MLOps: three additions

Historical first-stage report. The later full source reading, three additional
sections and final checks are recorded in
[2026-10-29-serving-full-source-reconciliation.md](2026-10-29-serving-full-source-reconciliation.md).
The outline-only boundary below describes the earlier stage, not the current
source map. Its earlier verification counts are preserved as stage evidence.

Date: 2026-10-29. Scope: additive explanations in three existing chapters. This report does **not** accept the three complete Harvard units as transferred or fully audited. English source passages keep their original language; the surrounding explanations distinguish source claims, derived results and invented examples.

## Saved content

| Destination | New section | Addition |
| --- | --- | --- |
| `00 Учебник/14 Inference и оптимизация/58b Benchmarking, SLO и эксплуатация inference.md` | `energy-per-result-measurement-boundary-and-time-window` | 162 lines: power versus energy; accounting boundary; interval integration; energy per SLO-passing result; idle subtraction; sampling and counter limitations; PUE and TDP caveats. One executable example and one original Harvard figure. |
| `00 Учебник/14 Inference и оптимизация/58c Queueing и capacity planning.md` | `fan-out-stragglers-and-hedged-requests` | 148 lines: fan-out tail amplification; independence and correlation; partial results; hedging timelines; conditional tails; duplicated work; streaming winner selection and side effects. Two executable examples and timing tables. |
| `00 Учебник/19 Deployment, Reliability и MLOps/02 MLOps.md` | `ml-test-score-check-the-four-areas-separately` | 160 lines: four readiness areas; concrete feature/model/infrastructure/monitoring checks; manual versus automated evidence; the minimum-section scoring rule; version-bound evidence; additional LLM gates. One executable example and one original Harvard figure. |

The 470 added lines include tables, source attribution and four Python blocks, not 470 lines of new prose. Existing material was preserved. The marker-delimited additions and their original/current chapter hashes are recorded in `2026-10-29-serving-operations-additions.json`.

## Source reading boundaries

The complete pre-edit destination chapters were read: benchmarking lines 1–750, queueing 1–827, MLOps 1–2360. A truncated part of the MLOps read was recovered by rereading lines 320–370. A matching section title was not treated as proof of coverage.

The heading inventory in `2026-10-29-serving-source-section-map.json` includes all 246 headings from the three imported Markdown units, with their parent headings and explicit reading labels. `outline-only` means the heading was inventoried, not that its prose was read, accepted or found missing. A unit-level `destination_candidate` is a routing aid, not a claim that every section belongs in that chapter.

Imported Markdown snapshot: Harvard commit `2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd`.

| Source | Complete explanatory sections read in this pass |
| --- | --- |
| `05 Источники/Courses/Harvard ML Systems/vol1/model_serving.md` | 272–366: serving architecture; 731–804: tail latency, fan-out and tail-tolerant techniques; 850–1012: model lifecycle, export, versioning and deployment. |
| `05 Источники/Courses/Harvard ML Systems/vol1/benchmarking.md` | 1488–1649: power measurement, accounting boundaries, computational efficiency, standardized power measurement and the MLPerf example. The initially truncated 1586–1597 range was reread. |
| `05 Источники/Courses/Harvard ML Systems/vol1/ml_ops.md` | 431–449: development infrastructure introduction, stack table and figure caption; 1680–1759: ML Test Score, technical debt, representative checklist and following checkpoint. |

Quizzes were not transferred. The infrastructure introduction is not a full reading of the larger Development Infrastructure chapter section.

Pinned QMD comparisons: Harvard commit `45ecc8d82fcae70c149cdce550d3b3d3411df913`.

- `05 Источники/Originals/inference-benchmark-2026-10-20/harvard-benchmarking.qmd`: 3045–3080 (power-boundary introduction and figure source start); 3372–3382 (complete standardized-power explanation). The original rendered SVG was inspected instead of claiming to read every TikZ helper line.
- `05 Источники/Originals/inference-queueing-2026-10-20/harvard-model-serving.qmd`: 2408–2438 (tail model); 2547–2585 (tail-tolerant explanation). The transferred paragraph is checked byte-for-byte against the pinned source.
- `05 Источники/Originals/mlops-2026-10-20/harvard-ml-ops.qmd`: the complete ML Test Score passage and representative checklist beginning at line 3171. The original four-category passage is checked against this source.
- Breck et al., [The ML Test Score](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/), §VI-A: verified seven tests in each of four sections, manual documented execution worth 0.5, automatic repeated execution worth 1, and final score equal to the minimum section sum. No figure or complete paper section was copied.
- Dean and Barroso, [The Tail at Scale](https://research.google/pubs/the-tail-at-scale/): publication record and the primary-author PDF fan-out example were checked. This was not a complete paper reading.

## What was added, already covered or rejected

Within the read serving architecture/lifecycle material, the existing destinations already explain release artifacts, tokenizer/configuration identity, cold starts, canaries, rollback, feature consistency, queues and overload. These were not duplicated under a new course heading. The implemented gaps were energy accounting, fan-out/hedging and a faithful readiness-scoring explanation.

Several source claims were not imported as general facts:

- REST statelessness does not by itself require repeated full-context serialization, and HTTP/2 is not exclusive to gRPC.
- GPU layout preference cannot be reduced to a universal NCHW rule, nor CPU preference to a universal NHWC rule.
- A transformation mismatch does not imply accuracy degradation proportional to the numerical difference; fixed accuracy-loss percentages require an actual experiment.
- Mutable shared-storage propagation is not a substitute for a pinned release and an observed rollout.
- M/M/1 is not a universal conservative bound for a dynamically batched service. Per-request service times and batched saturation throughput cannot be interchanged without a model.
- A compliant MLPerf load point is not the same quantity as dropping failed requests from an arbitrary offered-rate measurement and calling the remainder valid QPS.
- A 5% increase in request attempts is not necessarily a 5% increase in GPU work.
- Harvard's representative ML Test Score table is not the original complete inventory of 28 tests. The new chapter explicitly labels its eight selected checks.

The MLOps figure's old caption describes vertical tiers, but the actual SVG is a five-component horizontal layout. The new explanation follows the image rather than repeating the mismatch.

## Figures and permissions

Scoped asset directory: `00 Учебник/Assets/Figures/curated/serving-operations-2026-10-29/`.

| Local asset | Exact source asset | SHA-256 |
| --- | --- | --- |
| `power-measurement-boundaries.svg` | `Assets/Sources/Harvard ML Systems/vol1/benchmarking/68403b70d9a611957fc377cc27f4711283a8df2f.svg` | `be6580529d52eae2ab6d124b7960a9392d7d10a410bb23ee936af5ca1d783543` |
| `mlops-stack-layers.svg` | `Assets/Sources/Harvard ML Systems/vol1/ml_ops/3665b88dc7480bf5936afba8999575d6c9eb2f29.svg` | `bec070b7a36e9343c8bb8a675427ea233a749a44fa096f7ffb8de9992115f669` |

Both are unchanged original SVGs. `assets.json` and `NOTICE.md` record URLs, author, snapshot, source import manifest, attribution, licence evidence and hashes. The pinned Harvard licence was fetched and matched the local licence file: CC BY-NC-SA 4.0, SHA-256 `b6f4689d6fccf48d82d0871a3b994a122427fcb0b278971f6ba008be44e2c926`.

Both SVGs were rasterized and visually inspected with all source labels and arrows intact. The power diagram is wide and needs browser zoom/full-size access on narrow screens; this check alone does not establish responsive page rendering. The Harvard blast-radius SVG was inspected and rejected: its unlabeled circles and arrows do not explain the new fan-out example. That section uses worked tables instead. No redrawn or AI-generated figure was added.

The IEEE ML Test Score paper is a primary reference, not an asset licence source. Its figures and full text were not redistributed.

## Verification

New checker: `publishing/tools/check_serving_operations_completion.py`.

Passed on the saved chapters:

- execution of all 37 current Python blocks: 8 benchmarking, 12 queueing, 17 MLOps;
- 250 independently checked energy cases, 24 enumerated fan-out cases, 1,331 hedge timelines, 250 readiness-rubric cases and invalid-input checks;
- all three exact source passages, both unchanged SVG identities, XML parsing and pinned licence identity;
- all three bounded block removals reproduce the original chapter hashes.

Historical benchmarking checker passed its seven prior blocks and independent timestamp/cohort/quantile/queue/context/phase/burn-rate tests, source hashes and figure checks. Historical MLOps checker passed its sixteen prior blocks, independent cases, sources and figures. Its built-anchor check used the existing build and is not evidence for the new sections.

Historical queueing checker passed its mathematical, source and figure checks but reached an old footer assertion requiring the previous ML-workflow navigation link. The parent is integrating the existing bounded navigation restoration; this report does not claim that final assertion passed here.

`git diff --check` passed for the three chapter files and three modified historical checker files. No global build or responsive browser check was performed by this child agent. No hardware benchmark, measured power run, production operation, commit, push or deployment was performed.

## Integration files

New files owned by this pass:

- the four scoped asset files listed above;
- this report, the section hierarchy JSON and the additions/hash journal;
- `publishing/tools/check_serving_operations_completion.py`;
- `publishing/tools/serving-operations-baseline.mjs`;
- `publishing/tools/serving_operations_baseline.py`.

The three existing historical checkers import the Python bounded restoration helper before checking their previous accepted chapter content. Historical golden hashes were not changed. The parent has connected the JS helper to the shared reader-route chain and owns central asset registration, the navigation restoration, global build, rendered-page checks and final corpus status.

Remaining acceptance boundary: these three additions are implemented and locally checked; the outline-only Harvard sections remain unaccepted by this pass. A complete-source transfer claim would require reading and reconciling those sections, not merely this heading map.
