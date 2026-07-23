# Course Material Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transfer every useful non-duplicate explanation, example, visual, calculation, and exercise from the pinned Harvard ML Systems and Efficient Deep Learning Systems curricula into the main Bookvar learning route.

**Architecture:** Preserve source material in its original language and connect it with short Bookvar editorial bridges. First build a complete source-to-destination matrix and provenance validation, then import content in dependency order: computational foundations, distributed training, inference, lifecycle/operations, and practice. Existing Bookvar chapters are extended or split; the project does not create a competing second handbook.

**Tech Stack:** Obsidian Markdown, YAML frontmatter, Astro/Starlight publishing adapter, TypeScript/Vitest validation, original SVG/PNG/PDF/IPYNB assets, GitHub Pages.

## Global Constraints

- Selection rule: if a course contains useful information absent from Bookvar, import it.
- If Bookvar already covers the information, do not duplicate it unless the source explanation is materially clearer; in that case replace or restructure the weaker passage.
- Imported material remains in the language of its source.
- Label imported fragments as `original excerpt`, `adaptation`, or `editorial bridge`.
- Preserve source author, title, pinned commit URL, license, local path, modification status, and destination pages.
- Pin Harvard to commit `45ecc8d82fcae70c149cdce550d3b3d3411df913` and mark two-volume `dev` content as preview where applicable.
- Pin Efficient DL Systems to commit `e632aa89ca9e6638d52e1b686095e7442faffbb0`.
- Harvard `book/`, `labs/`, and `slides/` content is CC BY-NC-SA 4.0.
- Harvard MLSys·im is Apache 2.0. Verify TinyTorch's directory-level license at the pinned revision before importing it.
- Efficient DL Systems is MIT; separately preserve credits for third-party figures embedded in slides.
- Prefer original Harvard SVGs and original course assets to newly drawn diagrams.
- Do not delete an existing chapter until its unique material has been mapped and migrated.
- Every batch must pass unit tests, full build, built-link checks, and rendered-page inspection.

---

### Task 1: Build the complete course coverage matrix

**Files:**
- Create: `05 Источники/Source maps/Harvard ML Systems — complete transfer matrix.md`
- Create: `05 Источники/Source maps/Efficient DL Systems — complete transfer matrix.md`
- Create: `05 Источники/Source maps/ML systems — Bookvar gap matrix.md`
- Modify: `05 Источники/Community repositories.md`
- Modify: `publishing/navigation.yml`

**Interfaces:**
- Consumes: the two pinned repository revisions and the current Bookvar navigation.
- Produces: one row per source unit with `decision`, `destination`, `asset candidates`, and `status`; all later tasks select work from these rows.

- [ ] **Step 1: Inventory every Harvard unit**

Create rows for all Vol I/II chapters, Marimo labs, TinyTorch modules,
MLSys·im exercises, slide decks, and figure directories:

```markdown
| ID | Source path | Material | Adds to Bookvar | Decision | Destination | Assets | Status |
|---|---|---|---|---|---|---|---|
| H-V1-01 | `book/quarto/contents/vol1/introduction/introduction.qmd` | DAM, Iron Law, energy hierarchy | yes | integrate | `00 Учебник/10 ML Systems/01 Модель как часть системы.md` | `introduction_iron_law_bars.svg` | audited |
```

Use only `integrate`, `cross-link`, `source-only`, or `exclude`. An `exclude`
row must contain a reason in `Adds to Bookvar`.

- [ ] **Step 2: Inventory every Efficient DL Systems unit**

Create separate rows for each lecture, seminar, homework, code project, and
image directory from weeks 1–9:

```markdown
| ID | Source path | Material | Adds to Bookvar | Decision | Destination | Assets | Status |
|---|---|---|---|---|---|---|---|
| E-06-L | `week06_dl_arithmetic/lecture.pdf` | Transformer/MoE arithmetic | yes | integrate | `00 Учебник/10 ML Systems/04 Арифметика Transformer и MoE.md` | pages selected during PDF review | audited |
```

- [ ] **Step 3: Compare the matrices against Bookvar**

In `ML systems — Bookvar gap matrix.md`, organize missing material by learning
dependency:

```markdown
| Bookvar concept | Current page | Missing material | Best source IDs | Action |
|---|---|---|---|---|
| CUDA timing | — | async launch, warmup, synchronization | E-01-L, E-01-S | new chapter + lab |
| KV-cache | `00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention.md` | already substantial; missing measured Qwen example | E-08-S | add worked example |
```

- [ ] **Step 4: Publish the three matrices**

Add the following routes after the existing seven-repository audit:

```yaml
- source: 05 Источники/Source maps/Harvard ML Systems — complete transfer matrix.md
  route: sources/source-maps/harvard-ml-systems-transfer
- source: 05 Источники/Source maps/Efficient DL Systems — complete transfer matrix.md
  route: sources/source-maps/efficient-dl-systems-transfer
- source: 05 Источники/Source maps/ML systems — Bookvar gap matrix.md
  route: sources/source-maps/ml-systems-bookvar-gaps
```

- [ ] **Step 5: Verify and commit**

Run:

```bash
cd publishing
npm test
npm run build
npm run check:links
```

Expected: 87 or more tests pass; build succeeds; `0 broken internal routes,
fragments, or files`.

Commit:

```bash
git add '05 Источники/Source maps' '05 Источники/Community repositories.md' publishing/navigation.yml
git commit -m "Map Harvard and Efficient DL Systems into Bookvar"
```

---

### Task 2: Make asset provenance enforceable

**Files:**
- Create: `00 Учебник/Assets/Figures/curated/ml-systems/assets.yml`
- Modify: `publishing/tests/asset-registry.test.ts`
- Modify: `05 Источники/Визуальные материалы и лицензии.md`

**Interfaces:**
- Consumes: curated files under `00 Учебник/Assets/Figures/curated/ml-systems/`.
- Produces: a manifest checked by Vitest; later tasks cannot add an unregistered course asset.

- [ ] **Step 1: Write a failing registry test**

Extend `asset-registry.test.ts` with:

```ts
import fs from "node:fs";
import path from "node:path";
import YAML from "yaml";

const systemsAssetRoot = path.resolve(
  vaultRoot,
  "00 Учебник/Assets/Figures/curated/ml-systems",
);

it("registers every curated ML systems asset with pinned provenance", () => {
  const manifestPath = path.join(systemsAssetRoot, "assets.yml");
  const manifest = YAML.parse(fs.readFileSync(manifestPath, "utf8")) as {
    assets: Array<{
      file: string;
      author: string;
      source_url: string;
      commit: string;
      license: string;
      modified: boolean;
      used_in: string[];
    }>;
  };
  const registered = new Set(manifest.assets.map((asset) => asset.file));
  const actual = fs
    .readdirSync(systemsAssetRoot, { recursive: true })
    .map(String)
    .filter((file) => /\.(svg|png|jpe?g|gif|webp)$/i.test(file));
  expect([...registered].sort()).toEqual(actual.sort());
  for (const asset of manifest.assets) {
    expect(asset.source_url).toMatch(/^https:\/\/github\.com\/.+\/blob\/[0-9a-f]{40}\//);
    expect(asset.commit).toMatch(/^[0-9a-f]{40}$/);
    expect(asset.used_in.length).toBeGreaterThan(0);
  }
});
```

- [ ] **Step 2: Run the focused test and observe failure**

Run:

```bash
cd publishing
npm test -- asset-registry.test.ts
```

Expected: FAIL because `assets.yml` does not yet exist.

- [ ] **Step 3: Create the manifest**

Start with:

```yaml
assets: []
```

Document the manifest fields and the source-language rule in
`Визуальные материалы и лицензии.md`.

- [ ] **Step 4: Run the focused test**

Run:

```bash
cd publishing
npm test -- asset-registry.test.ts
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add '00 Учебник/Assets/Figures/curated/ml-systems/assets.yml' \
  '05 Источники/Визуальные материалы и лицензии.md' \
  publishing/tests/asset-registry.test.ts
git commit -m "Enforce provenance for imported ML systems assets"
```

---

### Task 3: Import the Harvard visual spine

**Files:**
- Create: `00 Учебник/Assets/Figures/curated/ml-systems/harvard/`
- Modify: `00 Учебник/Assets/Figures/curated/ml-systems/assets.yml`

**Interfaces:**
- Consumes: original SVG files at pinned Harvard commit.
- Produces: locally rendered, lossless figures referenced by Tasks 4–8.

- [ ] **Step 1: Import foundation figures**

Copy without modification:

```text
introduction_iron_law_bars.svg
introduction_energy_hierarchy.svg
ml_systems_memory_wall_divergence.svg
frameworks_dispatch_tax_divergence.svg
training_optimizer_memory.svg
hw_acceleration_roofline_elbow.svg
hw_acceleration_energy_ladder.svg
benchmarking_tail_latency_gap.svg
```

Use the exact source directories recorded in the Harvard matrix. Do not infer a
path from the filename.

- [ ] **Step 2: Import distributed-system figures**

Copy without modification:

```text
five-level-model.svg
gpudirect-data-path.svg
complete-data-path.svg
collective-primitives-overview.svg
ring-allreduce.svg
data-parallel-flow.svg
tensor-parallel-split.svg
pipeline-parallelism.svg
moe-all-to-all-routing.svg
zero-memory-partitioning.svg
3d-parallelism-cube.svg
parallelism-decision-tree.svg
failure-domains.svg
young-daly-optimization.svg
topology-placement.svg
```

- [ ] **Step 3: Import performance and inference figures**

Copy without modification:

```text
profiling-hierarchy.svg
roofline-model.svg
gpu-memory-hierarchy.svg
operator-fusion.svg
flashattention-tiling.svg
prefill-vs-decode-roofline.svg
optimization-decision-tree.svg
batching-strategies.svg
continuous-batching.svg
kv-cache-fragmentation.svg
disaggregated-serving.svg
queuing-hockey-stick.svg
tensor-parallel-routing.svg
pipeline-parallel-routing.svg
expert-parallel-routing.svg
ttft-tpot-timeline.svg
```

- [ ] **Step 4: Register every figure**

Each entry follows:

```yaml
- file: harvard/distributed/ring-allreduce.svg
  author: Harvard Edge ML Systems Book contributors
  source_url: https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/collective_communication/images/svg/ring-allreduce.svg
  commit: 45ecc8d82fcae70c149cdce550d3b3d3411df913
  license: CC BY-NC-SA 4.0
  modified: false
  used_in:
    - 00 Учебник/10 ML Systems/06 Collective operations и AllReduce.md
```

- [ ] **Step 5: Verify SVG integrity and registry**

Run:

```bash
find '00 Учебник/Assets/Figures/curated/ml-systems/harvard' -name '*.svg' -print0 |
  xargs -0 -n1 xmllint --noout
cd publishing
npm test -- asset-registry.test.ts
```

Expected: every SVG parses; registry test passes.

- [ ] **Step 6: Commit**

```bash
git add '00 Учебник/Assets/Figures/curated/ml-systems'
git commit -m "Import Harvard ML systems visual spine"
```

---

### Task 4: Add the computational-foundation chapters

**Files:**
- Create: `00 Учебник/10 ML Systems/01 Модель как часть системы.md`
- Create: `00 Учебник/10 ML Systems/02 GPU, CUDA и иерархия памяти.md`
- Create: `00 Учебник/10 ML Systems/03 Измерение производительности и roofline.md`
- Create: `00 Учебник/10 ML Systems/04 Арифметика Transformer и MoE.md`
- Create: `00 Учебник/10 ML Systems/05 Численные форматы и mixed precision.md`
- Create: `00 Учебник/10 ML Systems/06 Data pipeline, padding и packing.md`
- Create: `00 Учебник/10 ML Systems/07 Profiling ML-нагрузки.md`
- Modify: `publishing/navigation.yml`
- Modify: `00 Учебник/_index.md`

**Interfaces:**
- Consumes: Harvard Vol I introduction/frameworks/training/hardware and EDLS weeks 1, 2, and 6.
- Produces: prerequisites used by distributed training and inference chapters.

- [ ] **Step 1: Create consistent frontmatter**

Every page starts with:

```yaml
---
title: GPU, CUDA и иерархия памяти
type: textbook-chapter
status: draft
last_verified: 2026-07-23
source_language: mixed
---
```

- [ ] **Step 2: Write chapter 01 from Harvard systems framing**

Preserve the original English excerpts that introduce Data–Algorithm–Machine,
the Iron Law, energy hierarchy, and deployment constraints. Add only a short
Russian bridge from model architecture to systems constraints. Include the
registered Iron Law and energy hierarchy figures with visible source captions.

- [ ] **Step 3: Write chapters 02–03 from EDLS week 1 and Harvard hardware**

Include host/device, warps, SIMT, divergence, memory hierarchy, PCIe, async
launch, bandwidth, FLOP/s, arithmetic intensity, roofline, warmup,
synchronization, caching, and measurement distributions. Reuse original English
course text where it is already explanatory.

- [ ] **Step 4: Write chapter 04 from the complete EDLS week 6 lecture**

Include parameter, FLOP, activation, optimizer-state, and communication
calculations for dense Transformer and MoE; attention and FFN arithmetic;
GroupedGEMM; dispatch/all-to-all; TP versus EP; PP schedules including 1F1B,
ZeroBubble, and DualPipeV.

- [ ] **Step 5: Write chapters 05–07 from EDLS week 2**

Cover FP32/TF32/FP16/BF16/FP8/MXFP8, master weights, loss scaling, MFU/HFU,
storage/loading, tokenization, dynamic padding, bucketing, packing, py-spy,
PyTorch Profiler, Memory Snapshot, and Nsight Systems. Preserve device- and
version-specific context for numerical claims.

- [ ] **Step 6: Add navigation in dependency order**

Place the seven chapters after modern architecture/MoE and before pre-training
systems. Add previous/next links and source links.

- [ ] **Step 7: Build and inspect**

Run:

```bash
cd publishing
npm test
npm run build
npm run check:links
```

Open the seven generated routes at desktop and narrow width. Expected: formulas
render, SVG labels remain readable, no horizontal overflow.

- [ ] **Step 8: Commit**

```bash
git add '00 Учебник/10 ML Systems' '00 Учебник/_index.md' publishing/navigation.yml
git commit -m "Add computational foundations for ML systems"
```

---

### Task 5: Replace the distributed-training overview with a sequence

**Files:**
- Modify: `00 Учебник/11 Pre-training и Scaling/44 Distributed training и mixed precision.md`
- Create: `00 Учебник/11 Pre-training и Scaling/44a Processes, collectives и DDP.md`
- Create: `00 Учебник/11 Pre-training и Scaling/44b Gradient checkpointing и offload.md`
- Create: `00 Учебник/11 Pre-training и Scaling/44c Tensor и sequence parallelism.md`
- Create: `00 Учебник/11 Pre-training и Scaling/44d Pipeline parallelism.md`
- Create: `00 Учебник/11 Pre-training и Scaling/44e ZeRO, FSDP2, DeviceMesh и DTensor.md`
- Create: `00 Учебник/11 Pre-training и Scaling/44f Expert и hybrid parallelism.md`
- Create: `00 Учебник/11 Pre-training и Scaling/44g Network, storage и distributed checkpoints.md`
- Create: `00 Учебник/11 Pre-training и Scaling/44h Fault tolerance и fleet orchestration.md`
- Modify: `publishing/navigation.yml`

**Interfaces:**
- Consumes: EDLS weeks 3–6 and Harvard Vol II network, storage, collectives, distributed training, fault tolerance, and orchestration.
- Produces: canonical explanations used by training, MoE, inference, and operations.

- [ ] **Step 1: Convert chapter 44 into an orientation page**

Retain unique existing material and replace its overloaded taxonomy with links
to 44a–44h. Its central question becomes: which state is replicated, sharded,
moved, recomputed, or persisted?

- [ ] **Step 2: Build 44a from EDLS week 3**

Include process/rank/world size, point-to-point messages, broadcast/reduce/
all-gather/reduce-scatter/all-to-all, Ring and Butterfly AllReduce, DDP,
communication cost, gossip, error feedback, and PowerSGD.

- [ ] **Step 3: Build 44b–44d from EDLS week 4**

Include activation memory, selective checkpointing, CPU/disk offload, async
prefetch, TP for MLP and attention, Ulysses/Ring Attention, PP, GPipe, 1F1B,
pipeline bubble, and verification of logits and gradients.

- [ ] **Step 4: Build 44e from EDLS week 5**

Explain AllGather and ReduceScatter lifetimes, FSDP unit boundaries, forward and
backward prefetch, peak-memory synchronization, ZeRO-1/2/3, hybrid sharding,
FSDP2 hooks, DeviceMesh, DTensor placements, and Distributed Checkpoint.

- [ ] **Step 5: Build 44f from EDLS week 6 and Harvard**

Explain expert dispatch, token imbalance, GroupedGEMM, EP all-to-all, TP versus
EP, and hybrid/3D parallelism. Link back to the MoE architecture chapter rather
than repeating routing theory.

- [ ] **Step 6: Build 44g–44h from Harvard Vol II**

Cover RDMA/GPUDirect, InfiniBand/RoCE, topology, storage paths, checkpoint
storms, failure domains, SDC, Young-Daly checkpoint interval, elastic recovery,
topology-aware placement, gang scheduling, Slurm/Kubernetes, and autoscaling.

- [ ] **Step 7: Verify and commit**

Run full tests/build/link check and inspect all new routes.

```bash
git add '00 Учебник/11 Pre-training и Scaling' publishing/navigation.yml
git commit -m "Expand distributed training into a systems sequence"
```

---

### Task 6: Reconcile and expand the inference sequence

**Files:**
- Modify: all published files `00 Учебник/14 Inference и оптимизация/55a*` through `58b*`
- Create: `00 Учебник/14 Inference и оптимизация/57a KV-cache compression и offload.md`
- Create: `00 Учебник/14 Inference и оптимизация/58c Queueing и capacity planning.md`
- Modify: `publishing/navigation.yml`

**Interfaces:**
- Consumes: Harvard performance/inference/model-serving chapters and EDLS weeks 6, 8, and 9.
- Produces: one non-repetitive inference line grounded in the computational-foundation chapters.

- [ ] **Step 1: Build a duplication table before editing**

For `prefill`, `decode`, `roofline`, `KV-cache`, `batching`, `TTFT`, and `TPOT`,
record the single canonical chapter and replace repeated definitions elsewhere
with links.

- [ ] **Step 2: Add Harvard visual and quantitative examples**

Use the registered prefill/decode roofline, batching, KV fragmentation,
disaggregation, routing, and queueing figures. Add the Qwen3-4B measurements
from EDLS week 8 with the original hardware/software context.

- [ ] **Step 3: Expand FlashAttention and kernels**

Connect the IO argument to memory hierarchy, fusion, compilation, and the
prefill/decode distinction. Preserve the EDLS examples of fused QKV, RoPE,
cross-entropy, RMSNorm, and SwiGLU as linked worked examples.

- [ ] **Step 4: Expand quantization and create KV compression**

Cover PTQ/QAT, weight-only versus weight-activation, GPTQ, outliers,
SmoothQuant, FP8/FP4/MXFP4/NVFP4, rotation methods, token/model/system-level KV
compression, prefix sharing, sparse attention, and offload.

- [ ] **Step 5: Expand speculative decoding**

Import the original distribution-preservation proof and the practical
accept/reject/KV-rollback sequence. Distinguish greedy verification from
sampling and cover EAGLE variants without treating speedup as universal.

- [ ] **Step 6: Add queueing and capacity planning**

Use Harvard Little's Law, utilization knee, tail-latency, static versus
continuous batching, replica, and cost examples. Connect them to TTFT, TPOT,
throughput, RPS, queue wait, and SLOs.

- [ ] **Step 7: Verify and commit**

Run full tests/build/link check and inspect all inference routes.

```bash
git add '00 Учебник/14 Inference и оптимизация' publishing/navigation.yml
git commit -m "Reconcile and expand the LLM inference sequence"
```

---

### Task 7: Import the practical course line

**Files:**
- Create: `06 Практика/06 Измерить CUDA правильно.md`
- Create: `06 Практика/07 Mixed precision и loss scaling.md`
- Create: `06 Практика/08 Padding, packing и profiler.md`
- Create: `06 Практика/09 Реализовать Ring AllReduce.md`
- Create: `06 Практика/10 Checkpointing и offload.md`
- Create: `06 Практика/11 Tensor и sequence parallelism.md`
- Create: `06 Практика/12 Собрать FSDP из collectives.md`
- Create: `06 Практика/13 Оптимизировать Transformer step.md`
- Create: `06 Практика/14 Собрать mini inference engine.md`
- Create: `06 Практика/15 W8A8 и SmoothQuant в Triton.md`
- Create: `06 Практика/16 Speculative decoding и KV rollback.md`
- Create: `06 Практика/17 Упаковать модель в наблюдаемый сервис.md`
- Create: `06 Практика/Source/efficient-dl-systems/`
- Modify: `06 Практика/_index.md`
- Modify: `publishing/navigation.yml`

**Interfaces:**
- Consumes: original notebooks/code from all EDLS weeks and selected Harvard Marimo labs/TinyTorch modules.
- Produces: runnable source-preserving labs and site pages explaining prerequisites, environment, expected artifacts, and textbook links.

- [ ] **Step 1: Vendor original notebooks and code**

Preserve original directory names beneath
`06 Практика/Source/efficient-dl-systems/`, include the MIT license, pinned
commit, and an `IMPORT.md` listing unchanged and modified files. Do not flatten
the mini-edlang package.

- [ ] **Step 2: Create one site page per independent experiment**

Each page includes:

```markdown
## Original course material

- Author/course: Efficient Deep Learning Systems
- Source: <pinned URL>
- Language: English or Russian
- License: MIT
- Local material: <relative path>
- Hardware assumptions: <from source>
- Expected artifacts: profiler trace, plot, benchmark table, or tests
```

- [ ] **Step 3: Preserve the original exercise language**

Do not translate notebook prose. Add only Bookvar prerequisites, links to
answers in the textbook, environment pinning, and explicit hardware/version
caveats.

- [ ] **Step 4: Add Harvard interactive exercises**

Initially link/embed the original Marimo lab. Fork it only when Bookvar needs a
stable local version. Preserve prediction → controls → result → governing
equation → Design Ledger behavior.

- [ ] **Step 5: Verify and commit**

Check every local source link and published practice route, then run the full
test/build/link suite.

```bash
git add '06 Практика' publishing/navigation.yml
git commit -m "Import the ML systems practical course line"
```

---

### Task 8: Import Harvard lifecycle and governance material

**Files:**
- Create or expand focused chapters under:
  - `00 Учебник/11 Pre-training и Scaling/`
  - `00 Учебник/18 Evaluation и методология/`
  - `00 Учебник/19 Deployment, Reliability и MLOps/`
  - `01 Справочник/Security и Robustness/`
- Modify: `publishing/navigation.yml`
- Modify: `00 Учебник/_index.md`

**Interfaces:**
- Consumes: all Harvard matrix rows not integrated by Tasks 4–7.
- Produces: coverage of data engineering, selection, frameworks, compression, deployment, MLOps, reliability, security, robustness, sustainability, and responsible AI.

- [ ] **Step 1: Extend data and selection chapters**

Import missing material on acquisition, labeling, quality, versioning, active
learning, data gravity, data debt, scaling saturation, importance, coverage,
and deployment constraints.

- [ ] **Step 2: Add frameworks and compilation**

Add the missing bridge from computation graphs and automatic differentiation to
eager/static execution, dispatch overhead, ONNX/interoperability, compilers,
and hardware execution.

- [ ] **Step 3: Broaden compression**

Separate pruning/sparsity, low-rank/tensor decomposition, distillation, graph
optimization, and architecture optimization from LLM quantization.

- [ ] **Step 4: Add deployment and MLOps**

Combine Harvard serving lifecycle with EDLS week 7: service boundaries,
REST/gRPC, containers, health endpoints, observability, monitoring, drift,
registry, shadow/canary deployment, CI/CD/CT, and feedback loops.

- [ ] **Step 5: Add security, robustness, and sustainability**

Import missing threat models, poisoning, extraction, inversion, membership
inference, secure aggregation/enclaves, distribution shift, anomaly detection,
hardware corruption, energy/carbon/water accounting, and inference energy.

- [ ] **Step 6: Add responsible-system material**

Connect fairness impossibility, interpretability cost, governance, reward
hacking, unlearning, and monitoring to the existing alignment and evaluation
sections. Do not present them as detached ethical summaries.

- [ ] **Step 7: Verify and commit**

Run full tests/build/link check and inspect the new section at desktop and
narrow width.

```bash
git add '00 Учебник' '01 Справочник' publishing/navigation.yml
git commit -m "Add ML lifecycle reliability and governance material"
```

---

### Task 9: Complete source-language publishing and attribution

**Files:**
- Modify: every imported chapter and practice page from Tasks 4–8
- Modify: `publishing/adapter/build.ts`
- Modify: `publishing/tests/publication-policy.test.ts`
- Modify: `publishing/tests/site-output.test.ts`

**Interfaces:**
- Consumes: fragment labels and asset manifests.
- Produces: visible source attribution and unambiguous mixed-language publishing.

- [ ] **Step 1: Add a failing attribution policy test**

Require any page containing `source_fragment:` metadata to include author,
pinned source URL, license, language, and transformation:

```ts
expect(fragment).toMatchObject({
  author: expect.any(String),
  source_url: expect.stringMatching(/\/blob\/[0-9a-f]{40}\//),
  license: expect.any(String),
  language: expect.stringMatching(/^(en|ru)$/),
  transformation: expect.stringMatching(/^(original excerpt|adaptation|editorial bridge)$/),
});
```

- [ ] **Step 2: Render source blocks visibly**

The generated page must show “Original course material” with author, course,
language, source link, license, and modification status. Do not translate an
English block for the Russian route.

- [ ] **Step 3: Test desktop and narrow output**

Assert that source blocks, figure captions, formulas, SVGs, and code cells do
not overflow the article container at 1440 px and 390 px.

- [ ] **Step 4: Verify and commit**

```bash
cd publishing
npm test
npm run build
npm run check:links
npm run test:output
git add publishing
git commit -m "Publish source-language course material with attribution"
```

---

### Task 10: Editorial and deployment gate

**Files:**
- Modify: `05 Источники/Source maps/ML systems — Bookvar gap matrix.md`
- Modify: affected chapters after editorial review

**Interfaces:**
- Consumes: all integrated rows and built pages.
- Produces: a verified dev deployment and a closed gap matrix.

- [ ] **Step 1: Reconcile the matrix**

Change every completed `integrate` row to `integrated` and add the exact Bookvar
route. No row may remain implicitly skipped.

- [ ] **Step 2: Read every changed chapter in full**

Reject and rewrite pages that are definition catalogues, slide bullet dumps, or
unconnected excerpts. Verify that each chapter has a causal progression and a
clear transition to the next dependency.

- [ ] **Step 3: Inspect every imported visual**

At desktop and narrow width, verify legibility, aspect ratio, caption,
attribution, source link, and absence of clipping. Do not mark a page complete
when its principal figure requires opening the raw file to read it.

- [ ] **Step 4: Run the complete gate**

```bash
cd publishing
npm test
npm run build
npm run check:links
npm run test:output
cd ../site
npm run build
```

Expected: all commands exit 0; no broken routes/assets/fragments.

- [ ] **Step 5: Commit and push only `dev`**

```bash
git add .
git commit -m "Complete the ML systems course ingestion"
git push origin dev
```

Do not merge or push `main`.

- [ ] **Step 6: Verify GitHub Pages**

Wait for the dev deployment workflow, open the deployed home page, one
foundation chapter, one distributed-training chapter, one inference chapter,
one practice page, and one lifecycle page. Record the deployment commit and
URLs in the handoff.
