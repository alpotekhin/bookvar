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
| 01 Модель как часть системы | Harvard Vol. I Introduction and ML Systems, with exact named sections/anchors in the chapter | Russian adaptation; PPDF is explicitly labeled a Bookvar mnemonic, not a Harvard term; two short verbatim excerpts are labeled | `introduction_iron_law_bars.svg`, `introduction_energy_hierarchy.svg` | 911 |
| 02 GPU, CUDA и иерархия памяти | EDLS week 1 lecture; Harvard Hardware Acceleration “Acceleration Fundamentals” and hardware/memory hierarchy | Source concepts adapted; text SM diagram is an editorial execution map and registered Harvard diagram is unchanged | `gpu-memory-hierarchy.svg` | 732 |
| 03 Измерение производительности и roofline | EDLS week 1 lecture; Harvard Benchmarking framework/measurement challenges and Hardware Acceleration roofline | Source equations adapted; harness, A/B and anti-gaming procedure are editorial synthesis | `benchmarking_tail_latency_gap.svg`, `roofline-model.svg` | 678 |
| 04 Арифметика Transformer и MoE | complete EDLS week 6 slides 4–146; Harvard Neural Computation and Model Training | EDLS calculations re-derived with slide locators; custom schedule diagrams are explicitly explanatory text diagrams | `training_optimizer_memory.svg` | 1969 |
| 05 Численные форматы и mixed precision | EDLS week 2 named slide sections; FP8 paper; PyTorch AMP | Device/version-specific claims and adaptation status are explicit | `hw_acceleration_energy_ladder.svg` | 903 |
| 06 Data pipeline, padding и packing | EDLS week 2 named slide sections and dynamic-padding homework task 2 | Course mechanisms adapted; padding diagram is a derived quantitative ledger with inputs and efficiency shown | derived padding ledger | 770 |
| 07 Profiling ML-нагрузки | EDLS week 2 named slide sections/practice; Harvard Frameworks | Tool workflow and end-to-end case are editorial synthesis with source locators | `profiling-hierarchy.svg`, `frameworks_dispatch_tax_divergence.svg` | 730 |

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
- Chapter 01 now distinguishes Harvard concepts from the Bookvar PPDF mnemonic;
  no adapted prose is presented as a verbatim source excerpt.
- The pages are calculation-led rather than definition-only.
- Existing unrelated modification to
  `docs/superpowers/plans/2026-07-23-course-material-ingestion.md` was preserved
  and excluded from the Task 4 commit.

## Commit

Main Task 4 commit:
`9ebb1bd7865b900192b23a08f1706f4b3ab5e930`
(`Add computational foundations for ML systems`).

## Review remediation

The first review returned `CHANGES REQUIRED`; the compact first drafts were not
accepted as textbook-depth. The correction pass:

- fixes activation dimensionality to `O(LNH) = O(LBSH)` and adds a per-layer
  attention/MLP/norm activation ledger, FlashAttention and checkpointing;
- distinguishes one MoE projection (`2kHI`) from full SwiGLU (`6kHI`);
- covers the complete EDLS week 6 logistics sequence, Llama 7B/70B,
  Qwen 235B-A32B, DeepSeek-V3, FSDP/NCCL/overlap, fusion/Liger, TP/EP and
  GPipe/1F1B/ZeroBubble/DualPipeV;
- expands benchmarking, GPU execution, numerical formats, input pipeline and
  profiling to the requested mechanisms and worked examples;
- creates a distinct sidebar module VI and renumbers training/operations to
  VII/VIII; the manifest regression was updated to require eight modules;
- corrects chapter 01's previous link to the immediately preceding architecture
  chapter and adds honest section/slide locators throughout.

Final correction verification:

```text
publishing npm test: 98/98 passed
publishing build: passed
publishing check:links: 0 broken internal routes, fragments, or files
Astro production build: 574 pages; no ERROR/parse diagnostics
render inspection: 14 corrected desktop/narrow captures opened
```

Formula self-check used direct arithmetic, independently of the prose:

```text
Llama 7B block:
  4H^2 = 67,108,864 attention parameters
  3HI = 135,266,304 SwiGLU parameters
  total = 202,375,168 per layer; 6,476,005,376 over 32 layers
  forward linear work at N=8192 = 3,315,714,752,512 FLOP

activation ledger at B=1,S=8192,H=4096,I=11008,n_h=32:
  residual = 67,108,864 bytes
  two MLP intermediates = 360,710,144 bytes
  one materialized attention matrix = 4,294,967,296 bytes

MoE example S=8192,k=8,H=7168,I=2048:
  full SwiGLU = 5,772,436,045,824 FLOP = 7.216 ms at 800 TFLOP/s
  one projection = 1,924,145,348,608 FLOP = 2.405 ms
```

Review-fix implementation commit:
`c3a85bebe0c50767d479fc2bc92260b8bde838c5`
(`Deepen ML systems computational foundations`).
