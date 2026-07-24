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
The pre-existing modified shared plan file remains excluded. No source registry,
English fallback page, protected imported article, deploy state or remote was
modified.
