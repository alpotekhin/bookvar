# Expert / hybrid parallelism audit - 2026-10-13

Scope: canonical RU chapter 44f, its CPU arithmetic/gradient checker, and its
own illustration/provenance directory. No shared registry, navigation, EN,
build, commit or publication was performed by this subtask.

## Delivered explanation

- Four-token top-2 worked example; 8 assignment IDs, uneven source/destination
  message counts, packed and inverse maps, expert outputs and weighted result.
- Expert/input/router gradients, selected-softmax derivative, fixed top-k
  branch caveat, empty expert and optimizer-state distinction.
- Separate expert/rank load ratios with a counterexample: the same expert
  maximum can give either [8,0] or [4,4] rank loads.
- Capacity as a pool-specific limit; assignment dropping versus sequence-token
  deletion; padding, dropless variable buffers and GroupedGEMM.
- TP versus EP and an explicit 1024-rank map. Dense DP=16 and expert DP=2
  represent different replica sets. Non-TP-sharded router/norm parameters are
  excluded from the simplified P_d/8 formula.
- Pinned Megatron A2A -> AG(TP) -> experts -> RS(TP) -> A2A layout, with SP
  token ownership and logical bytes distinguished from physical per-rank
  sends. Per-source TP group: 1024 unique tokens; physical rank: 128 tokens,
  256 assignments. EP remote send: 1.75 MiB, TP AG/RS: 14 MiB each under the
  stated ring model. Forward total: 31.5 MiB outgoing activation bytes/rank.
- Expert gradients receive contributions from eight dense-DP source groups
  before EDP reduction. With the stated local-mean convention, correct scale
  is /16, not /2. Both Megatron SUM and AVG paths are explained.
- Parallel Folding as a distinct supported layout, not a reinterpretation of
  the worked map; DeepSeek-V3 and dense Llama 3 as primary-source case studies.

Existing `source_unit_id` values and canonical filename are unchanged. Old
renamed-heading slugs are retained as explicit anchors. Next/previous and
course links remain in the chapter.

## Sources checked

- EDLS commit `e632aa89ca9e6638d52e1b686095e7442faffbb0`, Week 6 PDF:
  MoE/GroupedGEMM and TP/EP arithmetic. Text extraction read alongside rendered
  pages; final figures 129/130 inspected before and after cropping.
- Harvard `vol2/distributed_training`: Expert parallelism and Hybrid
  Parallelism sections read, including the worked configurations. Some
  absolute network claims/diagram cost labels are not carried into 44f.
- Megatron `core_v0.19.2` commit
  `4b4acac9a1d28ea6829c8d4f566d75698a21249d`:
  `megatron/core/transformer/moe/token_dispatcher.py`, especially class
  workflow lines 372-383 and dispatch/combine code;
  `megatron/core/distributed/distributed_data_parallel.py` lines 181-247;
  MoE README token-dispatch, capacity, GroupedGEMM and Parallel Folding sections.
- DeepSeek-V3 report `https://arxiv.org/html/2412.19437v2`, section 3.
- Llama 3 report `https://arxiv.org/html/2407.21783v3`, section 3.3.2/table 4.
  Corrected a attempted v4 lookup: v3 is the actual reviewed revision.

## CPU verification

Command, executed successfully on the current script:

```sh
/Users/aleksan.potekhin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  publishing/tools/check_expert_parallel_examples.py
```

NumPy 2.3.5; float64 arithmetic. Seven PASS reports:

1. Manual outputs, explicit inverse, uneven messages, empty expert.
2. Dense all-expert reference versus packed per-expert execution with TP=1/2/3:
   outputs plus input, both expert-matrix and router gradients. Also checks
   relabeling physical expert owners, wrong inverse mapping, nondivisible TP.
3. Central finite differences for x/W1/W2/router. Observed maximum absolute
   errors: 4.5832e-10 / 1.3679e-9 / 5.8440e-10 / 8.5104e-10. Every perturbed
   call asserts that it has not crossed a top-k selection boundary.
4. Expert/rank imbalance and capacity arithmetic, including the NVIDIA image.
5. Bijection of 1024 coordinates, complete disjoint partitions for each group
   type, rank750 and primary-report GPU-count products.
6. Actual 16384 assignment identities and per-endpoint counts in one simulated
   EPxTP domain; token rows before/after EP and TP AG plus exact byte ledger.
7. Real linear-regression global-loss reference versus expert aggregation;
   naive EDP mean differs by 8x; Megatron AVG prescale and SUM prescale agree.

`git diff --check` passed for the edited chapter/script.
The checker does not launch torch.distributed, CUDA, NCCL, or hardware timing.
It proves algebraic equivalence, not the behavior/performance of a running
Megatron cluster. Site generation and actual page visual QA remain the parent
task's integration work.

## Asset registry proposals (parent integration)

Common fields for all three:

- `used_in`: `00 Учебник/11 Pre-training и Scaling/44f Expert и hybrid parallelism.md` only;
- provenance: `source-license-verified` for EDLS crops;
  `pinned-upstream-repository` for the unchanged NVIDIA GitHub image;
- `metadata_status: verified`;
- folder prefix: `00 Учебник/Assets/Figures/curated/expert-hybrid-review-2026-10/`;
- preserve the co-located complete license files and NOTICE.

### EDLS pair

- Author: Efficient DL Systems course authors.
- `license: MIT`;
- `license_url: https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/LICENSE`;
- `source_url: https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week06_dl_arithmetic/lecture.pdf`;
- `source_asset: 05 Источники/Papers/Source PDFs/edls-week06-e632aa89.pdf`;
- `source_asset_sha256: 64b6f4285f9fe9bc62bf8293f3585eb26e738d493723b22c2301c58a5a191215`;
- `commit: e632aa89ca9e6638d52e1b686095e7442faffbb0`;
- `derivation: pdf-page-render-crop`;
- `render_page_size_px: [2400, 1350]`;
- `crop_box_px: [70, 230, 2240, 1220]`;
- PNG dimensions `[2170, 990]`;
- modification: rendered exact PDF page, removed presentation title/outer
  margins only; no redrawing, translation or diagram content modification.

| asset | source_page | sha256 |
|---|---|---|
| edls-moe-tensor-parallel.png | 129 | 5118bb2f49454b22504c4915a2b9bedf002b4215c3764a16a6744939b5b32e08 |
| edls-moe-expert-parallel.png | 130 | f74a6f8fd2d83d702bb21082fd6b17bd730dbfbdaab96b6e2b21af239eefc7ce |

The parent chose to move the exact source PDF to the accepted tracked source
location above. `assets.json` beside NOTICE contains the three complete records
with this new path. This subtask did not weaken/modify the validator or change
the shared registry.

### NVIDIA capacity original

- `asset: .../mcore-token-capacity.png`;
- author: NVIDIA CORPORATION;
- `license: BSD-3-Clause`;
- `license_url: https://github.com/NVIDIA/Megatron-LM/blob/4b4acac9a1d28ea6829c8d4f566d75698a21249d/LICENSE`;
- `source_url: https://github.com/NVIDIA/Megatron-LM/blob/4b4acac9a1d28ea6829c8d4f566d75698a21249d/docs/images/moe/token_drop.png`;
- `source_image_url: https://raw.githubusercontent.com/NVIDIA/Megatron-LM/4b4acac9a1d28ea6829c8d4f566d75698a21249d/docs/images/moe/token_drop.png`;
- `commit: 4b4acac9a1d28ea6829c8d4f566d75698a21249d`;
- `derivation: source-raster-copy`, no separate local `source_asset` needed;
- `provenance_confirmation: pinned-upstream-repository` to use the registry's
  GitHub-original route (the no-local-source license-verified route is
  specifically for arXiv originals);
- `sha256: d5200aa91f34f4a09b6f5cce988dea53586e30965f3b4cecf3cc878c83ee2206`;
- dimensions `[3104, 1510]`; no modifications.

### Remove stale RU usage, do not delete source assets

Remove this chapter from `used_in` for:

- `curated/ml-systems/harvard/distributed/moe-all-to-all-routing.svg`;
- `curated/ml-systems/harvard/distributed/3d-parallelism-cube.svg`;
- `curated/ml-systems/harvard/distributed/parallelism-decision-tree.svg`;
- `curated/stanford-cs336-2026/systems/l08-p66.png`.

Retain other consumers, including EN if it still uses those assets.
The source matrix still has historical `integrate` rows H-SIM-09/14/15/16/17
targeting 44f. This chapter is not a full port of five Harvard simulator
tutorials; parent should avoid representing it as such. The present change
adds no claim of full tutorial migration and does not edit shared matrix.
