# Task 4 report — computational foundations

## Status

DONE. Seven chapters, their dependency-ordered routes, and the textbook index
were added locally. No push, PR, deploy, or merge was performed.

## Pinned inputs

- Harvard CS249r source tree:
  `/private/tmp/bookvar-cs249r-source`, commit
  `45ecc8d82fcae70c149cdce550d3b3d3411df913`.
- Efficient DL Systems source tree:
  `/private/tmp/bookvar-efficient-dl-systems-source`, commit
  `e632aa89ca9e6638d52e1b686095e7442faffbb0`.
- Canonical Bookvar matrices:
  `Harvard ML Systems — complete transfer matrix.md`,
  `Efficient DL Systems — complete transfer matrix.md`, and
  `ML systems — Bookvar gap matrix.md`.

## Chapter coverage and provenance

| Chapter | Source sections used | Transfer labels | Figures | Words |
|---|---|---|---|---:|
| 01 Модель как часть системы | Harvard Vol. I Introduction: D·A·M Taxonomy, Iron Law of ML Systems, energy tax/hierarchy; ML Systems: Purpose and Deployment Paradigm Framework/physical constraints | English source excerpts retained; Russian connecting and worked-example prose is editorial | `introduction_iron_law_bars.svg`, `introduction_energy_hierarchy.svg` | 544 |
| 02 GPU, CUDA и иерархия памяти | EDLS week 1 lecture: GPU/CUDA execution and timing; Harvard Hardware Acceleration: acceleration fundamentals, parallel computing, memory/energy hierarchy | Imported course concepts in source language where quoted; Russian derivation and bridge prose is editorial/adapted | `hw_acceleration_energy_ladder.svg` | 592 |
| 03 Измерение производительности и roofline | EDLS week 1 lecture: bandwidth, FLOP/s, CUDA measurement; Harvard Benchmarking: ML Benchmarking Framework, measurement challenges, tail latency; Harvard Hardware Acceleration: roofline | Source equations and claims adapted with device-context caveats; measurement procedure is editorial synthesis | `hw_acceleration_roofline_elbow.svg`, `benchmarking_tail_latency_gap.svg` | 378 |
| 04 Арифметика Transformer и MoE | complete EDLS week 6 lecture; Harvard Neural Computation: matrix operations and memory accounting | EDLS calculations adapted and re-derived; Bookvar connective prose is editorial | none | 654 |
| 05 Численные форматы и mixed precision | EDLS week 2 lecture: FP formats, AMP, FP8/MXFP8, utilization; Harvard training context through the transfer matrix | Device/version-specific numerical claims are attributed; derivations and decision rules are editorial | none | 540 |
| 06 Data pipeline, padding и packing | EDLS week 2 lecture; week 2 dynamic-padding homework task 2; Harvard data-pipeline context through the gap matrix | Course mechanisms adapted; examples and causal bridges are editorial | none | 427 |
| 07 Profiling ML-нагрузки | EDLS week 2 lecture and profiler practice; EDLS week 6 seminar context; Harvard Frameworks: execution problem and dispatch tax | Tool workflows adapted with explicit scope; diagnostic ladder is editorial synthesis | `frameworks_dispatch_tax_divergence.svg` | 519 |

All Harvard figures are pre-registered under
`00 Учебник/Assets/Figures/curated/ml-systems/harvard/`, retain visible
captions/attribution, and use publishing-compatible canonical vault links.

## Navigation and interfaces

- Seven flat routes were inserted after modern architecture/MoE and before
  pre-training.
- Sidebar order is S1 through S7 in the same dependency sequence.
- `_index.md` has a dedicated computational-foundations reading block.
- Every chapter has explicit previous/next links and pinned source links.

## Verification

Commands and results:

```text
cd publishing
npm test
  10 test files passed; 98/98 tests passed
npm run build
  passed
npm run check:links
  passed; 0 broken internal routes, fragments, or files

cd ../site
npm run build
  passed; Astro built 574 pages, including all seven /textbook/ml-systems/* routes
```

The first full Astro build found that ordinary relative Markdown figure paths
did not survive the publishing adapter. They were replaced with canonical
Obsidian asset links; the second full build passed and emitted all images.

Headless Chrome opened all seven generated routes at 1440×1200 and 390×844 and
captured fourteen screenshots under `/private/tmp/*-desktop.png` and
`/private/tmp/*-narrow.png`. Desktop inspection confirmed readable body text,
KaTeX formulas, sidebar order, and SVG rendering. Narrow captures confirmed the
responsive mobile chrome and readable text; the Retina headless capture uses a
2× bitmap scale, so apparent screenshot pixel width is not CSS viewport width.
No source-level fixed-width container or broken asset was found.

Additional checks:

```text
git diff --check
  passed
all six referenced Harvard SVG files exist
required-topic scan
  covered DAM/Iron Law, CUDA/SIMT/divergence/PCIe/timing, roofline and
  distributions, dense/MoE arithmetic and PP schedules, FP formats and
  MFU/HFU, padding/bucketing/packing, and the four profiler layers
```

## Self-review

- The sequence is causal: system constraint → device execution → measurement
  model → workload arithmetic → numerical representation → input supply →
  diagnosis.
- Claims with hardware- or software-version dependence are not presented as
  universal constants.
- Chapter 01 keeps the Harvard systems framing in English and limits Russian
  additions to connective explanation and one worked calculation.
- The pages are calculation-led rather than definition-only.
- Existing unrelated modification to
  `docs/superpowers/plans/2026-07-23-course-material-ingestion.md` was preserved
  and excluded from the Task 4 commit.

## Commit

Main Task 4 commit:
`9ebb1bd7865b900192b23a08f1706f4b3ab5e930`
(`Add computational foundations for ML systems`).
