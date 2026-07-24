# Section VI visual audit report

Date: 2026-07-24
Scope: `00 Учебник/10 ML Systems/*.md`
Pinned Harvard source: `/private/tmp/bookvar-cs249r-source` at
`45ecc8d82fcae70c149cdce550d3b3d3411df913`.

## Coverage

| Page | Before | After | Added teaching anchors and why |
|---|---:|---:|---|
| 01 Модель как часть системы | 2 | 4 | `vol1/ml_systems/images/svg/ml_systems_memory_wall_divergence.svg` makes the compute/bandwidth divergence visible; `vol1/ml_systems/images/svg/ml_systems_deployment_span.svg` puts the four deployment envelopes on one logarithmic power scale. |
| 02 GPU, CUDA и иерархия памяти | 1 | 3 | `vol1/hw_acceleration/images/svg/hw_acceleration_roofline_elbow.svg` distinguishes byte-starved from compute-bound execution; `vol2/performance_engineering/images/svg/operator-fusion.svg` shows which intermediate HBM round trips fusion removes. |
| 03 Измерение производительности и roofline | 2 | 3 | `vol1/benchmarking/images/svg/benchmarking_confidence_detectability.svg` connects sample size to the smallest defensible regression claim. |
| 04 Арифметика Transformer и MoE | 1 | 4 | `vol2/distributed_training/images/svg/distributed_training_memory_budget.svg` separates persistent state from dynamic activations; `vol2/distributed_training/images/svg/moe-all-to-all-routing.svg` exposes both dispatch and combine shuffles; `vol2/distributed_training/images/svg/pipeline-parallelism.svg` connects layer placement, microbatches, and activation transfers. |
| 05 Численные форматы и mixed precision | 1 | 3 | `vol2/performance_engineering/images/svg/block-quantization.svg` shows why a local block scale contains outlier damage; reused `vol1/training/images/svg/training_optimizer_memory.svg` prevents confusing narrow forward tensors with total training-state memory. |
| 06 Data pipeline, padding и packing | 0 | 3 | `vol1/data_engineering/images/svg/data_engineering_storage_latency_hierarchy.svg` ties shard/format choice to locality and latency. `bookvar/padding-bucketing-packing-ledger.svg` performs the same slot accounting across three layouts. `bookvar/packed-causal-mask-ledger.svg` performs the leakage test by comparing ordinary and segment-isolated causal masks. |
| 07 Profiling ML-нагрузки | 2 | 4 | `vol2/performance_engineering/images/svg/diagnostic-flow.svg` enforces the drill-down order before kernel diagnosis; reused `vol2/performance_engineering/images/svg/optimization-decision-tree.svg` maps a measured bottleneck to an intervention class. |

Total: 9 → 24 figures. Every chapter now has at least three.

## Original Bookvar figures

- `padding-bucketing-packing-ledger.svg`: original slot ledger derived from the
  chapter equations and EDLS week 2 “Optimal sequence processing”, pinned
  `e632aa89ca9e6638d52e1b686095e7442faffbb0`; it is not a copy of the slide.
- `packed-causal-mask-ledger.svg`: original attention-permission ledger derived
  from causal masking and segment isolation; EDLS supplies the pedagogical
  problem, not the artwork.

Both SVGs have `viewBox`, `role=img`, `<title>`, `<desc>`, text labels in
addition to color, and SHA-256 records in both registries.

## Verification

- Asset registry: 98/98 tests passed.
- Full publishing suite: 98/98 tests passed.
- Astro check: 0 errors, 0 warnings, 0 hints.
- Astro build: 594 pages.
- Published-output suite: 8/8 tests passed.
- Scoped `git diff --check` passed for authored files. A repository-wide
  `git diff --check` reports trailing whitespace inside the imported
  `diagnostic-flow.svg`; `cmp` confirms that file remains byte-for-byte equal
  to pinned upstream, so its source bytes were intentionally not normalized.
- Built HTML contains all new image URLs; output tests found no broken internal
  routes, fragments, or files.
- Figure counts verified from the seven Markdown sources: `4/3/3/4/3/3/4`.

## Render inspection

The first desktop inspection found
`data_engineering_storage_latency_hierarchy.svg` at approximately 97 px inside
a 617 px article column. The imported SVG intentionally retains its upstream
`72.63pt` intrinsic width; a source-specific site rule now presents it at
`min(100%, 28rem)` (448 px desktop maximum). The remaining Section VI Harvard
and Bookvar figures use `min(100%, 36rem)` (576 px desktop maximum). At a
narrow viewport both rules collapse to the content width and cannot overflow.

The corrected build therefore changes the expected rendered widths from
intrinsic values as low as roughly 97–108 px to 448 px for the storage ladder
and up to 576 px for the other anchors. The Codex browser runtime was not
available to this worker for a second screenshot pass; desktop/narrow
screenshots remain an external verification item, while CSS/output structure
and responsive bounds were verified locally.

## Review correction

- The padding/bucketing/packing ledger now omits EOS in all three panels.
  For document lengths `[8, 7, 3, 2]`, useful tokens stay constant at 20:
  padding uses 32 slots (62.5%), bucketing 22 (90.9%), and packing 20 (100%).
  Packed document boundaries are drawn as metadata/mask separators and consume
  no token slot in this comparison.
- The GPU chapter's roofline elbow now follows the arithmetic-intensity bridge
  `$I=F/Q_{\mathrm{HBM}}`, directly adjacent to the concept it explains.
- Seven reused assets now have matching `used_in` and SHA-256 fields in the
  local ML Systems manifest and the global asset registry.

## Final metadata audit

The final focused audit covered the seven reused Section VI anchors plus the
Bookvar padding/bucketing/packing ledger. For every `used_in` value it checked
that the Markdown file exists and contains the asset basename; it also compared
sorted `used_in` and SHA-256 fields between the local and global registries.
All eight assets passed.

Three phantom references were removed from both registries:

- `hw_acceleration_roofline_elbow.svg` → chapter 03;
- `operator-fusion.svg` → chapter 14/56;
- `optimization-decision-tree.svg` → chapter 14/56.

Their actual Section VI references remain. The editorial ledger global ID is
now `curated-bookvar-padding-bucketing-packing-ledger-svg-c01bab8fa293`,
matching the first 12 characters of its current SHA-256.

## Provenance notes

No EDLS raster/slides were copied, so no unresolved third-party figure
provenance was introduced. Harvard figures are exact copies from the pinned
paths above under the repository's CC BY-NC-SA 4.0 license. Visible captions
carry section locators, commit/license, and derivation status.
