# Task 6 report — reconciled inference sequence

Date: 2026-07-24

Base: `e84f55837cfde7aa91d1c1b047a52635e73409ac`

Pinned sources:

- Harvard ML Systems: `45ecc8d82fcae70c149cdce550d3b3d3411df913`
- Efficient DL Systems: `e632aa89ca9e6638d52e1b686095e7442faffbb0`

Scope: local worktree only; no push, PR, deploy, or merge.

## Canonical-definition and duplication table

| Term | Canonical chapter | Treatment elsewhere |
|---|---|---|
| prefill | 55a | 55/55c/56/58a2/58b link to the phase definition and discuss only their mechanism-specific consequence |
| decode | 55a | same boundary; 54 remains canonical for token selection rather than systems execution |
| roofline | `10 ML Systems/03` for the general model; 55a for inference application | later chapters name the limiting resource and link instead of re-deriving roofline |
| KV-cache | 55 for semantics, paging and allocation; 55a owns the byte formula | 57a owns compression/offload; 58a2 owns transfer |
| batching | 55 for memory/utilization motivation; 55b for scheduler state and continuous batching | 58c compares static/continuous batching only as a queueing/capacity trade-off |
| TTFT | 58b | 58c consumes it as an SLO constraint |
| TPOT | 58b | 58c consumes it as an SLO constraint |

## Old-to-new passage map and preservation

The existing 55a–58b sequence was read before editing. No chapter was replaced.
All edits are additive bridges or expansions.

| Existing passage family | Current location | Preservation |
|---|---|---|
| Aleksa Gordić vLLM latency, engine, chunked-prefill and prefix-cache walkthroughs | 55a, 55b, 55c | original images, captions and prose retained; canonical links added around them |
| vLLM/SGLang/TensorRT-LLM/FlashInfer runtime comparison | 55c | retained verbatim; orientation paragraph added |
| FlashAttention online softmax and FA2 partitioning | 56 | retained; memory hierarchy, fusion, compilation and prefill/decode kernel distinction added |
| GPTQ/AWQ/SmoothQuant/NF4 verification | 57 | retained; FP8/FP4/MXFP4/NVFP4 and rotation methods added |
| Leviathan exact-sampling figures and acceptance discussion | 58 | retained; distribution proof, greedy/sampling distinction, transaction rollback and EAGLE variants added |
| NCCL collectives and DP/TP/PP/CP/EP explanation | 58a | retained; pinned Harvard TP/PP/EP routing visuals added |
| DistServe/Splitwise/Mooncake/Dynamo and LMCache-related transfer discussion | 58a2 | retained; pinned Harvard disaggregation visual added |
| benchmarking/goodput/open-loop/overload operational prose | 58b | retained; EDLS Qwen3-4B protocol and epistemic status added |

The protected original-language Aleksa Gordić, vLLM and LMCache material was not
rewritten or translated. New prose links into it.

## Source coverage

| Destination | Coverage |
|---|---|
| 55a | Harvard prefill/decode roofline; dense forward and KV/GQA byte arithmetic |
| 55 | Harvard KV fragmentation; block table, last-block waste and copy-on-write |
| 55b | Harvard continuous batching; explicit scheduler state machine and token/KV budget |
| 55c | non-duplicative runtime orientation preserving existing articles |
| 56 | Harvard memory hierarchy; EDLS week 6 fused QKV, RoPE, CE, RMSNorm and SwiGLU linked to the pinned notebook |
| 57 | EDLS week 9 PTQ/QAT, GPTQ, outliers, SmoothQuant, FP8/FP4 and rotations |
| 57a | EDLS three-level KV taxonomy; GQA/MLA, KV quantization, token eviction, sparse attention, prefix sharing and offload |
| 58 | exact speculative sampling proof; accept/reject/rollback; EAGLE/EAGLE-2/EAGLE-3 |
| 58a | Harvard tensor/pipeline/expert routing plus retained collective/topology treatment |
| 58a2 | Harvard disaggregation plus retained KV transfer, routing, queues and backpressure |
| 58b | EDLS week 8 Qwen3-4B measurement protocol; benchmark outputs correctly marked absent |
| 58c | Harvard Model Serving and Inference: Little's Law, utilization knee, queueing, replicas, cost and capacity experiment |

## Visual inventory

All new figures are original registered Harvard SVGs, CC BY-NC-SA 4.0, with
exact commit-pinned locators in captions:

- `performance/prefill-vs-decode-roofline.svg` — 55a
- `inference/kv-cache-fragmentation.svg` — 55
- `inference/continuous-batching.svg` — 55b
- `performance/gpu-memory-hierarchy.svg` — 56
- `inference/tensor-parallel-routing.svg` — 58a
- `inference/pipeline-parallel-routing.svg` — 58a
- `inference/expert-parallel-routing.svg` — 58a
- `inference/disaggregated-serving.svg` — 58a2
- `inference/ttft-tpot-timeline.svg` — 58c
- `inference/queuing-hockey-stick.svg` — 58c
- `inference/batching-strategies.svg` — 58c

No substitute Mermaid/SVG was created.

## Worked formulas and configurations

- 55a: `$2P$` forward approximation; decode arithmetic-intensity argument;
  `2BLH_kvDSb` and 128 KiB/token, 4 GiB/32k worked example.
- 57a: GQA factor; EDLS 94-layer 3.05-GB/request ledger; 8×H200/8×H100 batch
  bounds; FP8 1.52-GB estimate; 4-GiB PCIe transfer lower bound.
- 58: local proof that acceptance plus residual correction sums to target
  probability; explicit KV commit/rollback sequence.
- 58c: end-to-end latency decomposition; Little's Law at 8 RPS/2.5 s; 10-GiB
  KV concurrency ledger; five-replica failure headroom and 32-USD/hour cost;
  token-weighted chat/summarization mix; 2.5-GiB/s KV-transfer requirement.

## Qwen3-4B source finding

EDLS week 8 fixes Qwen3-4B, `transformers==4.53.0`, FP16, warmup/run counts,
synchronization, greedy 256-token generation and FlashAttention-2 comparison.
The pinned notebook contains no stored outputs and no concrete GPU name.
Accordingly 58b records the exact protocol and states that no source-derived
throughput numbers can be published. This distinguishes a reproducible planned
measurement from an observed result.

## Verification and render inspection

- `publishing/npm test`: 10 files, 98 tests passed.
- `publishing/pnpm build`: passed.
- `publishing/npm run check:links`: 0 broken routes, fragments or files.
- `publishing/npm run test:output`: 1 file, 8 tests passed.
- `site/pnpm check`: 0 errors, 0 warnings, 0 hints.
- `site/pnpm build`: 594 pages built; both new routes generated; Pagefind
  indexed 595 HTML files.
- `git diff --check`: exit 0.

Headless Chrome inspected:

- desktop 1440×1100:
  `/textbook/inference/queueing-capacity-planning/`; layout, formula, Harvard
  image, sidebar and page outline rendered;
- narrow 412×915:
  `/textbook/inference/queueing-capacity-planning/` and the KV page; mobile
  header and typography rendered. The existing site theme has horizontal
  overflow at this narrow width, visible as clipped right-side prose; this is a
  pre-existing site-level responsive concern outside Task 6 chapter/nav
  ownership, not a missing route or Markdown failure.

## Scope audit

Task changes are limited to inference chapters, navigation and this report.
The pre-existing modified shared plan file remains excluded. Review remediation
also adds four pinned EDLS slide renders and their provenance registry entries.
No English fallback page, protected imported article, deploy state or remote was
modified.

## Review remediation

Review found one mathematical defect and four depth/visual gaps. The remediation
changes the earlier scope statement only for provenance: four original EDLS
slide renders and their asset-registry entries are now included.

### Speculative-decoding notation

Chapter 58 now uses one convention everywhere: draft distribution $p$, target
distribution $q$, candidate $x\sim p$, acceptance
$\min(1,q(x)/p(x))$, residual proportional to $(q-p)_+$. The erroneous later
paragraph that swapped $p,q$ was replaced by an unconditional-mass proof:
accepted mass `min(p,q)` plus correction `(q-p)+` equals `q`. Greedy
verification remains explicitly distinct.

### Quantization depth

Chapter 57 now gives:

- E4M3FN and E5M2 bit layouts, finite ranges, normal/subnormal minima;
- scalar E2M1 value set and saturation;
- OCP MXFP4: 32-element blocks, E8M0 scale and 4.25 effective bits;
- NVIDIA NVFP4: 16-element blocks, E4M3 block scale, FP32 global scale,
  1D activation and 16×16 2D weight layout;
- a 4096×4096 W4A16/W8A8 execution and byte ledger;
- orthogonal-rotation algebra, offline absorption versus online fusion,
  attention/RoPE constraints and an explicit verification ablation.

Exact locators are EDLS week 9 slides 29, 36–55; OCP MX v1.0 Table 1/§5.3;
CUDA `__nv_fp4_e2m1`; NVIDIA Transformer Engine NVFP4; and SpinQuant.

### EAGLE depth and visual

Chapter 58 separates EAGLE-1 feature regression plus token loss and shifted
token input, EAGLE-2 confidence-weighted dynamic tree, and EAGLE-3 direct token
loss, multi-layer feature fusion and training-time test. Proposal,
tree-attention verification and KV rollback are explicit. A worked
10-ms-target example shows 2.71× at accepted length 4.2 but slowdown at 1.4,
making non-universality quantitative.

The original EDLS slide 64 is registered and embedded:

- `eagle-method-slide-64.png`
- source SHA `e632aa89...`, page 64
- rendered full slide, no crop/edit
- PNG SHA-256 `c88f42fc89fc3e689ed751916854c70c40e9b128127a1a665bed31d31a8157f7`

### KV visual layer

Chapter 57a now embeds the three original EDLS taxonomy slides:

- token-level slide 71, SHA-256 `0e6d59e080984086b425ad3ac2d073ef3aed70d9546455fe2c61865c62b8f7fd`;
- model-level slide 72, SHA-256 `2f929ef94a51ff5bcbc6f83dae1aee643de4d6115ddc6ed98df073de7d849878`;
- system-level slide 73, SHA-256 `b48e2be8523e45859a1f81312b2fdb1a25448ded6f6e23d2357bc1c820f282dc`.

It also reuses the protected original LMCache MP transfer image and adds an
explicit `HBM -> DRAM -> SSD` residency/evict/fetch state sequence, tier table
and quality-versus-latency comparison. All four imported EDLS PNGs are
registered with exact commit/page, SHA, license and `used_in`.

### EDLS Qwen3-4B exhaustive audit

The audit covered week 8 lecture, README, seminar and homework notebook:

- lecture: 37 pages; qualitative prefill/decode on pages 11–12 and
  memory hierarchy on page 21; no Qwen measurements or hardware identity;
- seminar: zero saved output cells;
- homework: zero saved output cells;
- README: links only.

Thus the requested stored measurements are absent, not merely overlooked.
Chapter 58b marks the requirement unresolved, records both prompt-shape suites,
warmup/run count, synchronization, metric formulas, the `eager` versus actual
`flex_attention` code mismatch, and a copy-pasteable pinned notebook execution
protocol. Closing it requires explicit authorization to download Qwen3-4B and
an external CUDA run with `nvidia-smi`, software versions and raw samples; no
model download/run occurred in Task 6.

### Remediation verification

- `publishing/npm test`: 10 files, 98 tests passed.
- `publishing/pnpm build`: passed.
- `publishing/npm run check:links`: 0 broken links.
- `publishing/npm run test:output`: 1 file, 8 tests passed.
- `site/pnpm check`: 0 errors, 0 warnings, 0 hints.
- `site/pnpm build`: 594 pages and 595 indexed HTML files built.
- all four imported PNG SHA-256 values match both manifests.
- desktop render of speculative decoding is clean.
- 412-pixel render of KV-cache offload reproduces the pre-existing,
  site-wide horizontal overflow in the shared layout; the new figures are not
  the source of that overflow.

## P1 provenance remediation

This section supersedes the earlier statements that the four EDLS full-slide
PNG renders were safe reusable assets. Inspection showed that slide 64 embeds
paper artwork from EAGLE and slides 71–73 embed taxonomy figures from a KV-cache
survey. The MIT license of the course repository does not relicense those
embedded figures.

The upstream records were audited as follows:

- **EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty**,
  Yuhui Li, Fangyun Wei, Chao Zhang and Hongyang Zhang,
  <https://arxiv.org/abs/2401.15077>, §3 and Figure 4.
- **A Survey on Large Language Model Acceleration based on KV Cache
  Management**, Haoyang Li, Yiming Li, Anxin Tian, Tianhao Tang, Zhanchao Xu,
  Xuejia Chen, Nicole Hu, Wei Dong, Qing Li and Lei Chen,
  <https://arxiv.org/abs/2412.19442>, §§4–6 and Figures 4–6.

Both arXiv records expose the arXiv non-exclusive distribution license, not a
license granting downstream reuse of paper artwork. Therefore all four
full-slide PNGs were deleted rather than attributed as reusable.

They were replaced with four project-authored SVGs:

- `editorial/eagle-feature-drafting.svg`;
- `editorial/kv-token-level.svg`;
- `editorial/kv-model-level.svg`;
- `editorial/kv-system-level.svg`.

Each SVG uses an original Bookvar composition drawn from factual method or
taxonomy descriptions; none copies external artwork. Visible captions now give
the upstream title, authors or author group, canonical URL, section and figure
number, and explain the licensing decision. Both manifests record the pinned
EDLS slide as the discovery/factual source required by the curated-asset
contract, the canonical paper and license evidence in derivation metadata,
`modified: true`, and the final SVG hash.

Fresh verification after replacement:

- `publishing/npm test`: 10 files, 98 tests passed;
- `publishing/pnpm build`: passed;
- `publishing/npm run check:links`: 0 broken links;
- `publishing/npm run test:output`: 1 file, 8 tests passed;
- `site/pnpm check`: 0 errors, 0 warnings, 0 hints;
- `site/pnpm build`: 594 pages and 595 indexed HTML files;
- direct Chrome render of the EAGLE SVG and in-page render of the KV taxonomy
  are clean.

The Qwen3-4B external measurement remains unresolved. No model download or GPU
run was performed during this remediation.
