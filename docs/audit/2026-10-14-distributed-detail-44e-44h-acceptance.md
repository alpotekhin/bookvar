# Distributed training details 44e–44h and inference physics 55a

Local RU editorial acceptance, 2026-10-14. Canonical Obsidian Markdown;
Astro publication is derived. No EN source edits, commit, push or deployment.
Unrelated AGENTS.md and .agents changes were preserved.

This note closes this editorial batch, not the full textbook goal. Prior
accepted batches 42–44 and 44a–44d are not reclassified as unfinished here.

## Primary-agent review and final artifacts

The primary agent read each chapter in full, reviewed source explanations,
personally inspected selected original illustrations, and reviewed delegated
changes rather than accepting agent completion messages as proof.

Paths below are relative to the canonical worktree.

| Chapter | Final SHA-256 |
|---|---|
| 00 Учебник/11 Pre-training и Scaling/44e ZeRO, FSDP2, DeviceMesh и DTensor.md | 7c6cf1b6423256b7c5e3aebdbab4c5c71e79c0bf088d4e3bada5a6eeef1a343f |
| 00 Учебник/11 Pre-training и Scaling/44f Expert и hybrid parallelism.md | 964eb85deb64acfbeacab741b019d907a7346d3746c1e18db0faa17233c8509c |
| 00 Учебник/11 Pre-training и Scaling/44g Network, storage и distributed checkpoints.md | dea2ccd14e020dd7649065d2ab8405a603b19ba5c00d11ae67476a277f5d7c77 |
| 00 Учебник/11 Pre-training и Scaling/44h Fault tolerance и fleet orchestration.md | 2a77c85a000990600b323c0bf5298126aabee94d32041734f486924cfe032c42 |
| 00 Учебник/14 Inference и оптимизация/55a Физика LLM inference — prefill, decode и roofline.md | 6c459a9f5153b9eaaa35e437411bb14f4301b2e49d251c6abbe278c0c4257fd5 |

Individual reasoning/source records:

- 2026-10-14-fsdp.md
- 2026-10-13-expert-hybrid.md
- 2026-10-13-network-checkpoints.md, including the primary-editor follow-up
- 2026-10-14-fault-tolerance.md
- 2026-10-14-inference-physics.md

## Substance

44e distinguishes conditional 12-byte accounting, 16-byte ZeRO master-copy
accounting and actual FSDP2 dtypes; explains lifetimes, DeviceMesh/DTensor,
HSDP groups and valid-token normalization including zero-local/global-empty
cases. 44f follows routed tokens through dispatch, expert computation,
return and weighted combination; distinguishes EP from TP and EDP gradient
normalization, capacity drops and communication volume.

44g now follows bytes through GPU/NIC/network/storage, separates bits from
bytes and logical ring traffic from network hops, then develops full training
state, staging, async writing, resharding and publication of completed
checkpoint generations. Queue stability at exact capacity is not confused
with strict headroom. 44h covers failure domains, silent corruption,
checkpoint intervals, restart semantics, data cursors, elastic batch size
and gang placement. Arithmetic examples are not GPU performance claims.

55a now counts separate/fused MLP memory transfers explicitly; distinguishes
model allocation from compulsory linear-weight reads; recalculates the
Llama example and states the GQA architecture change; separates optimistic
attention input/output traffic from real tiled kernels; explains latency
timestamps and client chunk granularity.

## Original figures and provenance

44e/f/g/h use 3/3/3/4 figures respectively; 55a retains five originals.
Ten new reviewed assets in the four distributed-training review folders
are covered by the registry regression check: three FSDP, three expert,
two network/checkpoint and two fault-tolerance figures.

Sources: EDLS, the PyTorch FSDP paper, Megatron Core, OpenACC/NVIDIA
N-Ways to Multi-GPU, PyTorch's asynchronous checkpoint article and Harvard
ML Systems. Exact source revisions, hashes and full license notices are
retained in the folders and central registry. No custom diagram was drawn.

Retired from revised RU consumers, not deleted:

- earlier FSDP/EP figures superseded by reviewed course originals;
- Harvard gpudirect-data-path.svg and complete-data-path.svg:
  misleading 700 GB and 0.3 s annotations;
- Harvard young-daly-optimization.svg: invalid placement of total-cost curve.

The new network originals were copied byte-for-byte, not cropped or redrawn.
Their 1726×517 and 1600×286 aspect ratios are preserved. The wide PyTorch
diagram needs full-size viewing for small labels; clicking it was verified
to open the original. Transparent RDMA labels have a white page background.

Public EDLS, Megatron and PyTorch-website license files were fetched over
HTTP from the preview and compared byte-for-byte with source and dist copies:
all equal. Existing unverified rights of Gordić illustrations in 55a remain
explicitly unverified; no named license was invented.

## Executed examples and boundaries

| Checker | Observed evidence | Does not prove |
|---|---|---|
| check_fsdp_examples.py | PASS; actual five Markdown blocks; four CPU/Gloo processes; FSDP2 and 2×2 HSDP gradients/Adam states versus a global FP64 reference; DTensor, accumulation, BF16 CPU forward and default parameter lifetimes | CUDA/NCCL, GPU peak memory or performance |
| check_expert_parallel_examples.py | PASS; routing outputs/gradients, finite differences, capacity, topology and byte arithmetic | Real all-to-all implementation or GPU speed |
| check_network_checkpoint_examples.py | PASS; actual five Markdown blocks; exact CPU next-step restore with negative controls; delayed async writer; real 2→3 CPU/Gloo resharding; deterministic queue equality; local generation protocol | Power-loss durability, object-store atomicity, GPU/FSDP optimizer resharding or network throughput |
| check_fault_tolerance_examples.py | PASS; exact CPU restore/negative controls; Young/Daly and batch arithmetic; real two-worker torchrun group replacement after worker exit 23 | Launcher test is not a Gloo or end-to-end checkpoint-recovery test; no hardware failure or cluster scheduler |
| check_inference_physics_examples.py | PASS; actual two Markdown blocks; independent MLP/parameter/embedding/KV ledgers, attention, peak roofline ratio and timestamp arithmetic | Actual HBM transactions, fused kernel or server benchmark |

FSDP and network checkers were executed by the primary editor on 2026-10-14;
expert/fault results from the preceding part of this same work were reused
because their executable blocks/checkers were unchanged. Later network
edits changed prose/figures only. PyTorch checks used 2.14.1 on CPU.

## Build and actual-page checks

- Astro check: 0 errors, warnings or hints in nine checked source files.
- Final local build at 14:22:00: 1961 pages. Existing duplicate /404 route
  warning remains; no formula warning was emitted.
- Unit tests: 232 passed, including the ten-asset hash/license check.
- Final-dist output tests: 12 passed. Built-link check: zero broken
  internal routes, fragments or files.
- Source-coverage checker passed: 210 matrix rows, 59 destinations,
  215 MOOC pages, 212 figures, seven canonical linked pages, 34 Harvard labs,
  35 slide decks and 67 HSE artifacts. These counts establish connectivity,
  not full pedagogical integration.
- Browser inspection at 1280×900 and 390×844: all selected chapter images
  load, no KaTeX errors, no page-level horizontal overflow, original aspect
  ratios retained. Long equations/tables have local horizontal scrolling.
  Desktop/mobile screenshots were personally inspected; they were not
  persisted as repository image assets.
- The last 55a build was reloaded; relocated latency figure and corrected
  caption observed. Temporary viewport override reset afterward.
- git diff --check passed; en/ remains unchanged.

## Remaining work

The active full-book revision continues with KV-cache/PagedAttention and
serving scheduling, then the remaining planned pages. English alignment is
intentionally deferred. No completion claim is made for all knowledge from
all courses; historical source-matrix rows (including H-SIM rows) require
separate content evidence, not merely a working destination link.

No version was pushed or deployed in this batch.
