# Serving, benchmarking and MLOps: completed source reconciliation

Date: 2026-10-29. This report supersedes the partial-reading boundary in
`2026-10-29-serving-operations-completion.md`. No source chapter was accepted
merely because its headings appeared in the first inventory.

## Source reading and decisions

All explanatory prose was read in the three imported Harvard units at commit
`2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd`:

| Source under `05 Источники/Courses/Harvard ML Systems/vol1/` | Read range | Hierarchy |
| --- | --- | --- |
| `model_serving.md` | 1–2171 | 97 prose headings plus answer appendix |
| `benchmarking.md` | 1–2214 | 88 prose headings plus answer appendix |
| `ml_ops.md` | 1–2230 | 101 prose headings plus answer appendix |

The ranges include the complete sections, not just paragraphs selected for
copying. Truncated tool outputs were reread in smaller ranges. Embedded quiz
material was not transferred. Self-check answer appendices begin at lines
2172, 2215 and 2231 and were excluded from transfer. Rendering helpers and
TikZ code were not counted as prose reading; the reused figures were inspected
as rendered originals.

`2026-10-29-serving-source-section-map.json` now records **289 headings**,
including H5/H6 subsections missing from the earlier 246-heading inventory.
It has 286 substantive-prose heading entries, three excluded answer appendices,
59 manually authored thematic decisions and **zero outline-only entries**.
Every entry retains its parent, body/subtree range, disposition, destination
anchors and applicable cautions. The generator refuses a source whose hash
has changed since this reading.

The map does not claim that every source sentence was copied. It separates:

- mechanisms already explained in the canonical chapters;
- the six confirmed gaps closed in this pass;
- material in another existing thematic chapter, such as SHAP/LIME and edge rollback;
- sibling-owned compression/hardware scope, without claiming this child reread those complete destination chapters;
- historical suite lists, maturity taxonomies and application case narratives retained as attributed source reading;
- unsupported generalizations not promoted to textbook facts.

Complete destination reads: the pre-edit benchmarking chapter 1–750, queueing
1–827, MLOps 1–2360, system introduction 1–416 and serving engines 1–611.
Additional complete sections read: Responsible systems 202–535 (calibration,
group uncertainty, feedback, SHAP, LIME and counterfactuals), roofline 703–1080
(MFU/HFU, scaling, retained progress, time-to-quality and energy). The hardware
agent independently read Edge 998–1049 (safe update and rollback).

## Six implemented additions

| Canonical destination | New anchor | Explanation and evidence |
| --- | --- | --- |
| `14 Inference и оптимизация/58b Benchmarking, SLO и эксплуатация inference.md` | `energy-per-result-measurement-boundary-and-time-window` | Watts versus joules; same-window integration; device/server/facility boundary; energy per acceptable result; idle subtraction, sampling, PUE and TDP. Executable example and original Harvard boundary diagram. |
| `14 Inference и оптимизация/58c Queueing и capacity planning.md` | `fan-out-stragglers-and-hedged-requests` | Tail amplification with independent versus correlated components; hedge/cancellation timelines; attempts versus GPU work; partial results and side effects. Two executable examples and worked timing tables. |
| `19 Deployment, Reliability и MLOps/02 MLOps.md` | `ml-test-score-check-the-four-areas-separately` | Original four-area scoring rule, 0.5/1 manual/automated evidence, minimum-section score, selected versus complete test inventory. Executable example and original Harvard stack figure. |
| Same MLOps chapter | `retraining-frequency-stale-predictions-and-update-cost` | Staleness integral, update cost per unit time, small-drift approximation versus exact optimum, operational/label constraints. Exact source passage, original retraining-policy plot, executable solver and independent integration checks. |
| Same MLOps chapter | `ab-validation-effect-size-sample-size-and-decision-time` | Assignment unit, absolute MDE, power, variance, sample size, enrollment duration, delayed outcomes, interference and stopping rules. Two complete source paragraphs and executable calculation. |
| `14 Inference и оптимизация/55c Serving engines — vLLM, SGLang, TensorRT-LLM и FlashInfer.md` | `host-placement-thread-pools-and-loading-weights` | NUMA CPU/memory placement, thread-pool oversubscription, mapping versus residency/transfer, meta initialization and `assign=True`, Safetensors layout. Pinned source passages and a real CPU output/storage check. Existing approved engine excerpts remain unchanged. |

These are six bounded additions in four existing chapters, not six new course
introductions. They contain seven new Python blocks. Parent read all six
complete additions and accepted their explanations before browser QA.

## Figures and source files

Three unchanged original Harvard assets are used:

| Scoped directory under `00 Учебник/Assets/Figures/curated/` | Asset | Identity |
| --- | --- | --- |
| `serving-operations-2026-10-29/` | `power-measurement-boundaries.svg` | SHA `be6580529d52eae2ab6d124b7960a9392d7d10a410bb23ee936af5ca1d783543` |
| Same | `mlops-stack-layers.svg` | SHA `bec070b7a36e9343c8bb8a675427ea233a749a44fa096f7ffb8de9992115f669` |
| `serving-decisions-2026-10-29/` | `retraining-policies.png` | SHA `aff285a0ee3e5c7cdaaf6c84fb25e377e8e269d5d228eb7389af58170ae9f64a`, 5190×2077 |

Each directory has `assets.json` and `NOTICE.md` with source paths, URLs,
licence evidence and hashes. Harvard's CC BY-NC-SA 4.0 licence is retained
locally. No image was redrawn. The unlabeled source fan-out illustration was
rejected; the actual new timing example is shown as a table.

New source snapshots in `05 Источники/Originals/serving-decisions-2026-10-29/`:
PyTorch tuning guide, module-loading tutorial and BSD-3-Clause licence at
`5672cd98a7671eda4d9122b10f8eec580f74d03e`; Safetensors README and Apache-2.0
licence at `e246a2560645b7525f5775669ed816eb57c5bcc8`. Snapshot hashes are in
`2026-10-29-serving-decisions-sources.json`. The source excerpts are checked
against these local files. Performance figures from the tutorials were not
reused as measurements of this repository.

## Claims corrected or withheld

The substantive map records source-specific cautions. In particular:

- M/M/1 is not a universal upper bound for batched GPU serving; p99 cannot replace mean residence time in Little's law.
- A batch-formation deadline does not bound waiting behind a busy accelerator. Batch throughput and request service time must not be interchanged.
- Resident KV capacity is not proof of TPOT-feasible concurrency. Weight quantization does not automatically quantize KV.
- Utilization counters, useful model FLOPs, advertising peaks, TDP and measured energy are different quantities.
- The square-root retraining interval is an approximation; the chapter derives and solves the exact stationarity condition under the stated model.
- Running an A/B test until significance is not a fixed-horizon design. A lack of significant improvement is not equivalence.
- Schema agreement is not feature-value agreement; a feature store does not guarantee zero skew or zero maintenance cost.
- Drift is not measured quality loss. A Prometheus scrape interval does not reduce bytes of per-request logs by that interval.
- Unverified fixed speedup/accuracy percentages, the source Llama serving scenario and clinical/device case claims remain attributed source context, not canonical measurements or clinical advice.

## Verification on the final saved content

`check_serving_operations_completion.py` passed: the first three additions,
37 Python blocks in their reconstructed revision, 250 energy cases, 24
enumerated fan-out cases, 1,331 hedge timelines, 250 rubric cases, exact source
passages and two byte-identical figures.

`check_serving_decisions_examples.py` passed with cached offline PyTorch
2.14.1: all 19 current MLOps Python blocks, 300 independent retraining cases
(including numerical integration), 300 A/B power cases, real mmap/meta/assign
output-and-storage checks, five exact source passages and the unchanged PNG.

All eight historical checkers passed after exact bounded restoration:
feature store, platform resources, FinOps, MIG visuals, ops-scale,
queueing/capacity, inference benchmarking and MLOps. No accepted historical
chapter hash was changed. Queueing uses the existing exact navigation
restoration, rather than restoring an obsolete footer on the live chapter.

The six relevant Vitest files passed, **15 tests**: reading paths, ops-scale,
MIG visuals, FinOps, platform resources and feature stores. The ops-scale
public JSON was regenerated against the current full chapter, practice and
checker hashes. The practice worker's restoration hook remains intact.

The source map regenerates byte-for-byte from its authored decisions, verifies
the three reviewed source hashes and resolves all 68 distinct destination
anchors. Scoped `git diff --check` passed for the chapter and legacy-checker
changes. These checks validate the map structure, not a claim of verbatim
transfer of every source section.

Parent browser QA found escaped-dollar math failures and two narrow-screen
equations. Currency is now written as USD outside math; the two displays use
line breaks. The addition journals retain previous QA hashes and final block
hashes. The arithmetic/source checks passed again after those edits. Parent
owns the subsequent rebuild and rendered-page recheck; this report does not
claim that CPU checks establish visual correctness.

Current MLOps SHA:
`16e7174177cbe112cc35562f557554368f4d23eb3bc74e5651ac70a6ea4360bb`.
Current benchmarking SHA:
`1d453c6e9f77e4aee10c81305bc406fe0bf3a50e69fc4edd22748a555e5a0b9c`.
The bounded additions journals contain all four chapter identities.

Not executed by this child: optional DuckDB feature-store SQL validation,
optional MIG PDF rerender, NUMA/GPU performance measurements, measured power,
live serving or production changes. No commit, push, deployment, EN edit or
practice-body edit was made by this child. Central figure registration and
global build/browser acceptance remain with the parent/integration workers.
