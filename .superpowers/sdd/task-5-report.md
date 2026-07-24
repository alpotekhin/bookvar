# Task 5 report — distributed-training sequence

Date: 2026-07-24

Base: `1b85e84097c2204fc09f40a4e12588c8e7dc81ce`

Harvard snapshot: `45ecc8d82fcae70c149cdce550d3b3d3411df913`

Scope: local worktree only; no push, PR, deploy, or merge.

## Legacy 44 audit and passage migration

The old chapter was read in full before modification. Its unique passages map as follows:

| Old passage | New canonical location | Preservation |
|---|---|---|
| opening distinction: fit state vs reduce time-to-result | 44 opening | retained as the orientation problem |
| Adam 16–20 B/parameter and activation memory | 44 ledger; 44b activation ledger; 44e sharded ledger | retained and expanded with 7B and FSDP examples |
| CS336 DP memory figure and explanation | 44; 44b | retained with original slide locator |
| DP, ZeRO/FSDP, TP, PP, context/sequence, EP taxonomy | 44 links; mechanisms in 44a/44c/44d/44e/44f | expanded, not replaced by definitions |
| topology rule: TP intra-node, slower links for less-frequent exchange | 44, 44c, 44f, 44g, 44h | retained and derived from collective frequency |
| alpha/bandwidth communication model and overlap caveat | 44a ring/butterfly derivation; 44g network path | expanded with byte/time examples |
| FP16 loss scaling, BF16, FP8, FP32-sensitive operations | 44 `Смешанная точность` | retained under the old stable anchor |
| five-step launch plan | 44 `Инвариант проверки`; per-chapter verification sections | retained and made mechanism-specific |
| pipeline bubble formula | 44d GPipe/1F1B | retained with worked 4-stage examples and schedules |
| typical errors | 44 plus corresponding 44b–44h checks | every unique failure mode retained |
| system/learning verification and strong/weak scaling | 44; 44a/44c/44d/44f | retained |
| roofline explanation | 44 | retained with ASCII-only LaTeX variables for renderer compatibility |
| CS336 parallelism matrix and decision table | 44; Harvard decision tree in 44f | retained and complemented |
| Megatron sequence-parallel compatibility invariant | 44f | retained |
| checkpoint as testable protocol | 44; 44e DCP; 44g atomic storage; 44h recovery runbook | retained and expanded |
| source list | distributed across each chapter’s exact mechanism sources | retained primary sources; Harvard pinned locators added |

The incoming question-index anchors `#Оси параллелизма` and `#Смешанная точность` remain valid.

## Source coverage by chapter

| Chapter | Main source coverage |
|---|---|
| 44 | legacy material; Stanford CS336 lecture 8 slides 17 and 55; Harvard Distributed Training |
| 44a | EDLS week 3; Harvard Vol II Collective Communication and Distributed Training; PowerSGD paper |
| 44b | EDLS week 4; Chen et al.; PyTorch activation checkpointing; CS336 memory slide |
| 44c | EDLS week 4; Harvard Distributed Training model-parallel section; Ulysses and Ring Attention papers |
| 44d | EDLS week 4; GPipe; Megatron/PipeDream-Flush; Harvard pipeline explanation |
| 44e | EDLS week 5; ZeRO; PyTorch FSDP2/DTensor/Distributed Checkpoint; Harvard memory partitioning |
| 44f | EDLS week 6; Harvard distributed/hybrid parallelism; Megatron Core constraints; link to canonical MoE architecture chapter |
| 44g | Harvard Vol II Network Fabrics (`sec-network-fabrics-transport`, topology) and Data Storage (training path/checkpoint storm) |
| 44h | Harvard Vol II Fault Tolerance (failure models, SDC, checkpoint optimization) and Fleet Orchestration (gang, placement, Slurm/Kubernetes); Daly 2006 |

All Harvard links pin commit `45ecc8d82fcae70c149cdce550d3b3d3411df913`.

## Figure inventory

| Chapter | Figure | Exact origin |
|---|---|---|
| 44 | `44-cs336-dp-memory.png` | CS336 lecture 8 slide 17 |
| 44 | `44-cs336-parallelism-table.png` | CS336 lecture 8 slide 55 |
| 44a | `collective-primitives-overview.svg` | Harvard Collective Communication |
| 44a | `ring-allreduce.svg` | Harvard Collective Communication |
| 44a | `data-parallel-flow.svg` | Harvard Distributed Training |
| 44b | `44-cs336-dp-memory.png` | CS336 lecture 8 slide 17 |
| 44c | `tensor-parallel-split.svg` | Harvard Distributed Training |
| 44d | `pipeline-parallelism.svg` | Harvard Distributed Training |
| 44e | `zero-memory-partitioning.svg` | Harvard Distributed Training |
| 44f | `moe-all-to-all-routing.svg` | Harvard Distributed Training |
| 44f | `3d-parallelism-cube.svg` | Harvard Distributed Training |
| 44f | `parallelism-decision-tree.svg` | Harvard Distributed Training |
| 44g | `five-level-model.svg` | Harvard Network Fabrics |
| 44g | `gpudirect-data-path.svg` | Harvard Network Fabrics |
| 44g | `complete-data-path.svg` | Harvard Data Storage |
| 44h | `failure-domains.svg` | Harvard Fault Tolerance |
| 44h | `young-daly-optimization.svg` | Harvard Fault Tolerance |
| 44h | `topology-placement.svg` | Harvard Fleet Orchestration |

Harvard SVGs were pre-registered in the curated asset registry with commit, license, and `used_in`; no substitute diagrams were created. The two text schedule diagrams in 44d are semantic timelines, not replacements for the original pipeline figure.

## Derivations and worked examples

- 44: 7B Adam-state ledger and sharded lower bound; roofline bound.
- 44a: ring AllReduce volume `2(N-1)M/N`; 8-rank 1-GiB time; butterfly latency crossover; DDP overlap; PowerSGD 4096² rank-8 compression.
- 44b: 48-layer activation ledger; PCIe 16-GiB offload lower bound and uncovered stall.
- 44c: TP per-layer communication; 32k-context sequence-shard memory and transient AllGather peak.
- 44d: GPipe bubble for `p=4,m=8/32`; GPipe and 1F1B schedules; stage imbalance; 128-MiB activation transfer.
- 44e: ZeRO-1/2/3 state table; two AllGather + ReduceScatter volume; synchronized peak-memory example; DTensor local shape; 1.12-TB checkpoint lower bound.
- 44f: EP dispatch/combine payload; imbalance ratio; GroupedGEMM; 1024-GPU 3D mesh.
- 44g: 175B BF16 ring volume/time; 2-TB checkpoint storm and async staging ledger.
- 44h: job MTBF aggregation; Young–Daly derivation and 120-s/12-h example; recovery ledger; compact-placement completion-time example.

## Navigation and rendered inspection

`publishing/navigation.yml` places the eight new routes immediately after orientation 44 in both flat pages and sidebar. Local links are linear from 43 → 44 → 44a … 44h → SFT. Rendered inspection covered all nine Task 5 routes:

- each emitted a Russian HTML page with one `h1`;
- each has `rel=prev` and `rel=next`;
- all 18 figure uses rendered as `<img>`;
- formula-bearing pages rendered KaTeX;
- no raw `[[wiki links]]`, `TODO/TBD/FIXME`, `file://`, or workstation paths appeared.

## Verification evidence

Red/green navigation evidence:

- first `npm test`: failed exactly because eight page routes were absent from the sidebar;
- sidebar updated;
- second `npm test`: **10 files, 98 tests passed**.

Build/link/output evidence:

- first publication build found two unresolved legacy anchors; both stable headings were restored;
- `pnpm build` in `site`: **exit 0, 590 pages built, Pagefind completed** after fixing the math fence;
- `npm run check:links`: **0 broken internal routes, fragments, or files**;
- `npm run test:output`: **1 file, 8 tests passed**.

## Self-review

- Scope is limited to chapter 44, 44a–44h, navigation, and this report. The pre-existing modified plan file was not touched.
- All brief checklist topics are present by term and in explanatory context.
- Old passages and incoming anchors were preserved.
- Source figures are original registered course/book assets with attribution.
- No deploy, remote operation, PR, push, or merge was performed.

Content commit: `27c16bb68ba9442599ba046aa3a175bc4b3bb232`.

## Review remediation — standalone depth and schedule proof

The compact first pass was expanded after review without reverting the inherited
44c/44d corrections. Current source counts are:

| Page | Lines | Words |
|---|---:|---:|
| 44 | 95 | 947 |
| 44a | 89 | 674 |
| 44b | 88 | 821 |
| 44c | 138 | 942 |
| 44d | 166 | 1177 |
| 44e | 86 | 664 |
| 44f | 71 | 585 |
| 44g | 74 | 597 |
| 44h | 92 | 782 |

Review finding → remediation map:

- ambiguous Adam ledger → named BF16/FP32 dtype table and explicit
  `2+4+4+4+4=18 B/param`;
- lost legacy operational material → restored constraint→tool→cost→check table
  and the five launch gates;
- activation formula hid naive attention → separated
  `O(LBHS^2)` saved score/probability state from FlashAttention's
  `O(LBHSD)` saved-state contract;
- mixed GiB/GB arithmetic → `16 GiB = 17.18 GB`,
  `17.18/24 = 0.716 s`, with `0.316 s` uncovered after 0.4 s compute;
- compact chapter summaries → added goals/prerequisites, state transitions,
  pseudocode, worked ledgers/configurations, trade-offs and verification;
- vague source references → added mechanism locators for selective activation
  checkpointing, FlashAttention, FSDP2/fully_shard/DCP, EP compatibility,
  RDMA/RoCE/PFC/storage, gang/topology/Slurm/Kubernetes;
- multiline display math triggered the non-ASCII math guard → made the two
  affected formulas single-line without changing their semantics.

Schedule proof used literal 4-stage, 4-microbatch GPipe and 1F1B tables. The
validator asserted: 32 unique operations per table; exactly one `Fμ` and `Bμ`
per `(stage,microbatch)`; `Fμ@Si > Fμ@S(i-1)`; `Bμ@Si >
Bμ@S(i+1)`; and `Bμ@Si > Fμ@Si`. Output:

```text
GPipe PASS: 32 ops; uniqueness, F chain, B chain, and B-after-F dependencies hold
1F1B PASS: 32 ops; uniqueness, F chain, B chain, and B-after-F dependencies hold
```

Fresh verification after remediation:

```text
cd publishing && npm test
Test Files 10 passed (10); Tests 98 passed (98)

cd site && pnpm check && pnpm build
astro check: 0 errors, 0 warnings, 0 hints
astro build: 590 pages built; Pagefind completed; exit 0

cd publishing && pnpm build && npm run check:links && npm run test:output
0 broken internal routes, fragments, or files
Test Files 1 passed (1); Tests 8 passed (8)

git diff --check
exit 0
```

The pre-existing modified plan file remained excluded from Task 5 staging.
