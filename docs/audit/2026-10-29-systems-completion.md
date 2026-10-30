# Systems additions and Harvard Model Compression reconciliation

## Scope and reading evidence

The canonical compression chapter was read in full before editing (1,092
lines); the GPU chapter was read in full before editing (910 lines).
The new additions were read again after writing. Original source language
is retained. Neither chapter is presented as a translation of the entire
Harvard unit.

The pinned source
`05 Источники/Originals/model-compression-2026-10-19/harvard-model-compression.qmd`
was read through its end, source lines 1–8139. Reading excluded unpublished
include/echo-false figure generators and TikZ implementation, but retained all
published prose, captions, formulas, tables and displayed code. Truncated
tool outputs were repaired by explicit rereads of source lines 251–455 and
5064–5282. Captions are not evidence of visual inspection: the added source
SVGs and published pages have separate visual checks.

This completes a substantive comparison of this pinned Model Compression
unit, not all Harvard chapters, future upstream revisions, all HSE material,
or model-quality experiments.

## Where the source knowledge is taught

Destinations below refer to canonical chapter headings, not merely imported
course archives. The JSON section map enumerates every source heading with
its line range and destination.

| Source range | Teaching material | Destination and disposition |
| --- | --- | --- |
| 1–414 | Deployment constraints, memory/latency/energy, Pareto choices | Compression: “Четыре разных преобразования”, “Как выбирать метод”; benchmarking. Already explained; generic speed/accuracy ranges are not imported as measurements. |
| 415–1814 | Masks, units to prune, criteria, one-shot/iterative, LTH, dynamic sparsity | Compression: existing mask/Wanda/structured examples plus new “Prune, recover, and reset”. New numerical schedule and initialization comparison. |
| 1815–1828 | Mask versus deployable artifact | Compression: mask versus physically smaller matrices, sparse formats; retained original distinction between efficient architecture and pruning. |
| 1829–2033 | Teacher/student signal, temperature, KL, training expense | Compression: complete temperature/gradient example and “Какие сигналы доступны у языковой модели”. No new duplicate definition. |
| 2034–2414 | SVD, CP, Tucker, TT, direct execution of factors | Compression: SVD and existing expanded tensor-factorization section with executable convolution/TT examples. |
| 2415–2645 | Search space, nested optimization, RL/evolution/DARTS, search amortization | New search section: 27 candidates, continuous versus discrete result, validation/test distinction; existing latency-constrained choice. |
| 2646–3985 | Precision, memory traffic, energy, numeric formats | Quantization: coding/scales/packing and compute paths; numerical-formats chapter; GPU memory hierarchy. Reject automatic speedups and universal low-bit accuracy cliffs. |
| 3986–4917 | PTQ calibration, zero point, granularity, weights/activations, dynamic scales, QAT/STE | Quantization: complete scalar examples, affine formula, metadata overhead, static/dynamic calibration, GPTQ/AWQ/SmoothQuant and QAT. These destination sections were inspected, not merely title-matched. |
| 4918–4987 | Binary/ternary arithmetic and runtime restrictions | New binary-dot-product explanation with agreement table, bit packing, executable XNOR/popcount identity and ternary storage limit. |
| 4988–5076 | Hardware-aware architecture, compound scaling, operator structure, memory reuse | New compound-scaling example; CNN grouped/depthwise section already demonstrates restricted kernels and 432→75 weights; activation checkpointing stays in ML Systems. |
| 5077–5282 | Fusion and traffic, Conv–BN–ReLU | Existing fusion example plus new fixed-statistics BN folding algebra; distinguishes parameter folding, kernel fusion and training-mode BN. |
| 5283–5690 | Early exits, routing, MoE, overhead/quality | New exit-cost and static-batch example; link to full MoE chapter. No arbitrary decoder-layer skipping guarantee. |
| 5691–6859 | Sparse layouts, block/N:M patterns, metadata and combinations | Existing CSR and 2:4 treatment, small dense structured model and sparse-kernel caveats; MoE handles routed blocks. |
| 6860–7178 | Constraint-led method choice and composition | Existing “Как выбирать метод” and validation workflow; no fabricated BERT mobile pipeline numbers. |
| 7179–7382 | Profiling, Amdahl, layer sensitivity, quality/latency/energy | Existing roofline and inference-benchmark chapters, plus new damage-diagnosis paragraph. |
| 7383–8139 | Tool boundaries, reproducibility, diagnostic plots and pitfalls | Existing export/runtime requirements and new diagnostic interpretation. Source checklists, repeated summaries and decorative metaphors are not duplicated. |

## New material

Compression receives 6 displayed programs: iterative removal/reset,
DARTS relaxation, early-exit cost, compound scaling, BN folding and binary
dot product. Three unchanged original Harvard SVGs explain lottery-ticket
iteration, NAS and early exits. Existing SVD, teacher/student and tensor
factorization material remains intact. Total: 17 displayed checked programs
and 12 figure embeds.

GPU receives two original Google TPU animations and the complete original
Harvard systolic-array drawing. The additions derive the 2×2 execution
schedule and compare weight/output/input-stationary scalar storage costs.
Two CPU programs verify those examples.

Google animation provenance and license evidence are in
`00 Учебник/Assets/Figures/curated/hardware-completion-2026-10-29/assets.json`.
The source's simplification about GPU memory accesses is explicitly qualified.
No image was generated or redrawn.

## Source claims deliberately not copied

These are corrections to this source snapshot, not assertions that every
version of the course contains the same problem.

- L1115 conflates gradual magnitude pruning with regrowth. These are separate
  choices in the new explanation.
- L1143–1360 supplies precise iterative/one-shot accuracy trajectories without
  a reproducible run. They are not used as universal measurements.
- L1990/1999 overstates calibration transfer and assumes temperature 1 yields
  nearly one-hot probabilities. The existing chapter defines temperature
  through logits, with no such universal premise.
- L2097 calls low rank extra arithmetic, whereas direct factor application
  can reduce MACs; current chapter counts both factors and the intermediate.
- L2708/3655–3658/6831 give generic accuracy cliffs, energy categories and
  sparsity thresholds. No universal 3-bit cliff or 90–95% threshold is added.
- L2838 argues that energy savings explain numerical robustness. They do not:
  numerical error and energy are evaluated separately.
- L4204 describes asymmetric quantization as different positive/negative
  scales. The inspected canonical chapter uses one affine scale and a zero point.
- L4728 excludes other PTQ remedies. Current GPTQ/AWQ/SmoothQuant explanations
  already show why “only higher precision” is not the only recourse.
- L4838 calls the STE correct in magnitude and wrong in direction without
  a specified objective. The chapter instead identifies it as a surrogate.
- L4904 prescribes permanently adapting scales/BN statistics. Actual QAT
  recipes differ, including frozen observers; no universal prescription copied.
- L5064 gives a square-root checkpoint bound in terms of memory bytes, without
  layer-count/activation-size assumptions. The ML Systems chapter retains its
  explicit checkpoint schedule rather than importing this bound.
- L5181 equates not storing the full score matrix with linear total attention
  HBM traffic. FlashAttention's tiled traffic is treated separately.
- L5279 assumes all ML graphs are declared before execution and have no alias
  issues. The source's pseudocode is not a real fused kernel.
- L7173/7236/7363–7377/8078–8094 combine illustrative metrics into apparent
  measured profiles. No fabricated accuracy, energy or speed benchmark added.
- L8082 claims LayerNorm/softmax always need at least FP16; that is not a
  universal statement about integer approximations or all runtimes.

## Acceptance

Exact additions and reversible preservation of prior reviewed text:
`2026-10-29-systems-completion-additions.json`.
Regression tests: `publishing/tests/systems-completion.test.ts`.
CPU checks: `publishing/tools/check_systems_completion_examples.py`
and `publishing/tools/check_model_compression_examples.py`.

The final combined pass records build, links and visual evidence separately.
CPU arithmetic does not validate GPU speed, trained-model accuracy, NAS search,
or deployment on a physical TPU. No stage/commit/push/deploy performed.
