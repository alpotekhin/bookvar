# Stanford CS336 Textbook Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the complete educational substance of Stanford CS336 Spring 2026 into the Bookvar narrative and turn Assignments 1–5 into five connected, verifiable capstones.

**Architecture:** Existing strong Bookvar chapters remain canonical and are expanded in place. New chapters are added only for missing conceptual homes: shape-driven Transformer construction, practical kernel engineering, corpus engineering, and reproducible capstone routes. Course text stays in its original language where that is stronger; Bookvar adds only the bridge needed to preserve narrative continuity. Every edit closes named rows in the Stanford coverage and visual ledgers.

**Tech Stack:** Obsidian Markdown, original Stanford Python/PDF/figure assets, YAML provenance ledgers, Python/PyTorch/Triton exercise contracts, Astro/Starlight, Vitest.

**Spec:** `docs/superpowers/specs/2026-09-04-stanford-berkeley-ingestion-design.md`

## Global Constraints

- Complete `2026-09-04-course-source-ledgers.md` Tasks 1–2 before this plan.
- Cite the pinned Stanford artifact and relevant primary paper at the point of use.
- Use original figures and slide sequences; do not redraw them with Mermaid or generated SVG.
- Introduce notation before formulas and tensor dimensions before code.
- Preserve the difference between model architecture, training objective, training system, inference system, and observed behavior.
- Do not publish Stanford solution code. Exercises specify interfaces, tests, measurements, and evidence bundles.
- Close each coverage row only after the destination text and rendered visual have been read in context.
- Before every batch commit, perform the same editorial gate on every changed destination: read the rendered page end to end; verify the six-part sequence `problem → source argument → mechanism/example → evidence boundary → primary source → transition/exercise`; search for summary-like or machine-written phrasing; verify each course claim against the pinned fragment; inspect all changed figures at desktop and narrow widths.
- Every practice has a machine-readable contract with pinned starter source, exact adapter/test interfaces, commands, expected tests, CPU/local mode, GPU/full mode, evidence schema, and pass/fail criteria. A Markdown brief alone is not a verifiable capstone.
- Focused commands in individual tasks supplement the mandatory full gate: `pnpm --dir publishing test`, `pnpm --dir publishing build`, `pnpm --dir publishing check:links`, `pnpm --dir site check`, `pnpm --dir site build`, and `pnpm --dir publishing test:output`.
- Before every commit, stage exact files individually, never a module directory. Confirm scope with `git diff --cached --name-only`; no pre-existing user file may enter the commit merely because it shares a directory.
- Keep `en/` untouched. Do not push or deploy.
- Stage only named files; unrelated VLM and navigation edits already exist in the worktree.

---

### Task 1: Build the language model from tokens to a trainable Transformer

**Files:**
- Modify: `00 Учебник/02 Представление текста и токенизация/02 BPE, WordPiece и Unigram.md`
- Modify: `00 Учебник/03 Языковое моделирование/01 Вероятность текста и next-token prediction.md`
- Modify: `00 Учебник/05 Attention и Transformer/Masking, multi-head и формы тензоров.md`
- Modify: `00 Учебник/05 Attention и Transformer/03 Полный Transformer.md`
- Modify: `00 Учебник/07 Анатомия современной LLM/02 Современный decoder block.md`
- Modify: `00 Учебник/07 Анатомия современной LLM/03 Pre-norm, RMSNorm, SwiGLU и residual.md`
- Modify: `00 Учебник/01 Основы нейронных сетей/02 Оптимизация и стабильность обучения.md`
- Create: `00 Учебник/07 Анатомия современной LLM/05 Transformer с нуля — формы, параметры и стоимость.md`
- Create: `06 Практика/20 Собрать языковую модель с нуля.md`
- Create: `06 Практика/Contracts/stanford-cs336-a1.yml`
- Create: `00 Учебник/Assets/Figures/curated/stanford-cs336-2026/foundations/`
- Modify: `05 Источники/Courses/Stanford CS336 Spring 2026/coverage.yml`
- Modify: `05 Источники/Courses/Stanford CS336 Spring 2026/visuals.yml`
- Modify: `05 Источники/asset-registry.yml`
- Modify: `publishing/navigation.yml`

**Interfaces:**
- Covers Lecture 1, Lecture 2's `einops`/shape/resource-accounting portion, Lecture 3, and Assignment 1.
- Practice output: tokenizer implementation, decoder-only Transformer, optimizer/training loop, checkpoint, tests, resource estimate, loss curve, and generated samples.

- [ ] **Step 1: Audit existing prose against the pinned lecture sections**

Mark each Stanford unit as `covered-existing`, `integrated`, or requiring new text. Remove claims whose only support is an obsolete Spring 2025 legacy note.

- [ ] **Step 2: Expand tokenization as an implementation story**

Explain bytes and Unicode, pre-tokenization, byte-level BPE training, encode/decode invariants, special tokens, compression ratio, multilingual and code effects, and the difference between BPE, WordPiece, and Unigram. Insert Stanford's strongest worked trace and its original figure sequence with captions telling the reader what changes after each merge.

- [ ] **Step 3: Add shape-driven construction**

Use `einops`-style named dimensions to trace batch, sequence, head, and feature axes through embeddings, RoPE, Q/K/V projections, causal masking, attention, SwiGLU, residual paths, logits, and cross-entropy. Derive parameter count, activation memory, and leading FLOPs from the same shapes.

- [ ] **Step 4: Reconcile the modern decoder block**

Integrate Stanford's architecture and hyperparameter discussion into the existing LLaMA-derived narrative. Keep the existing stronger explanations of RMSNorm, RoPE, and SwiGLU; add course material only where it supplies a clearer derivation, experiment, or implementation constraint.

- [ ] **Step 5: Integrate the optimizer and training-recipe units**

Route initialization/scale, AdamW, Muon/SOAP, learning-rate schedules including
WSD, batch size/critical batch size, and stability to the existing optimization
chapter. Distinguish an optimizer definition from a measured training recipe
and record unsupported or time-sensitive comparisons as source-only.

- [ ] **Step 6: Write Capstone 1 as a connected contract**

Specify byte-level BPE, tokenizer tests, model components, stable cross-entropy, AdamW, learning-rate schedule, gradient clipping, data loading, checkpointing, a small training run, and a required evidence bundle. In `stanford-cs336-a1.yml`, pin the starter repository and list exact adapters/tests, local and GPU commands, expected outputs, and evidence files. Link narrow Bookvar labs as prerequisites rather than duplicating them.

- [ ] **Step 7: Import and register the visual spine**

Include orienting, mechanism, and comparison visuals for tokenization, tensor shapes, the decoder block, and training/resource accounting. Preserve multi-frame traces as sequences and verify label legibility at narrow width.

- [ ] **Step 8: Render, read, and close ledger rows**

Read every changed page from beginning to end in the rendered site. Reject paragraphs that read as a list of lecture topics or disconnected definitions.

- [ ] **Step 9: Verify and commit**

```bash
cd publishing
npm run check:course-ingestion
npm test
npm run build
npm run check:links
```

Expected: zero orphaned Lecture 1–3/A1 rows, no unregistered visual, and no broken route.

```bash
git commit -m "docs: integrate CS336 language model foundations"
```

---

### Task 2: Integrate attention alternatives and mixture of experts

**Files:**
- Modify: `00 Учебник/08 Эффективный Attention и длинный контекст/01 MHA, MQA и GQA.md`
- Modify: `00 Учебник/08 Эффективный Attention и длинный контекст/02 MLA и сжатие KV-cache.md`
- Modify: `00 Учебник/09 Dense FFN и Mixture of Experts/02 Mixture of Experts — routing, capacity и serving.md`
- Modify: `00 Учебник/09 Dense FFN и Mixture of Experts/03 Mamba, RWKV, RetNet и гибридные архитектуры.md`
- Create: `00 Учебник/Assets/Figures/curated/stanford-cs336-2026/architectures/`
- Modify: `05 Источники/Courses/Stanford CS336 Spring 2026/coverage.yml`
- Modify: `05 Источники/Courses/Stanford CS336 Spring 2026/visuals.yml`
- Modify: `05 Источники/asset-registry.yml`

**Interfaces:**
- Covers Lecture 4.
- Keeps the Bookvar MoE chapter canonical and uses Stanford as a source for concrete tradeoffs and experiments.

- [ ] **Step 1: Map every Lecture 4 architecture to the current chapters**

Separate attention-state compression, attention-compute changes, recurrent/state-space alternatives, and conditional FFN computation. Do not imply that they solve the same bottleneck.

- [ ] **Step 2: Integrate the course's comparative explanation**

For each architecture, state the tensor or routing change, asymptotic and practical resource effect, training/serving constraint, and evidence shown in the lecture.

- [ ] **Step 3: Strengthen the MoE mechanism**

Use the original routing and load-balancing visuals to explain top-k selection, auxiliary losses, expert capacity, dropped/overflow tokens, expert parallel communication, and the difference between total and active parameters.

- [ ] **Step 4: Verify model-family claims against primary papers**

Remove the legacy note's inconsistent DeepSeek expert counts and unsupported GPT claims. Keep lecture-level comparisons labelled as course examples rather than universal rules.

- [ ] **Step 5: Render and commit**

```bash
cd publishing
npm run check:course-ingestion
npm test
npm run build
npm run check:links
```

```bash
git commit -m "docs: integrate CS336 efficient architectures and MoE"
```

---

### Task 3: Turn GPU execution, Triton, FlashAttention, and parallelism into one systems route

**Files:**
- Modify: `00 Учебник/10 ML Systems/02 GPU, CUDA и иерархия памяти.md`
- Modify: `00 Учебник/10 ML Systems/03 Измерение производительности и roofline.md`
- Modify: `00 Учебник/10 ML Systems/04 Арифметика Transformer и MoE.md`
- Modify: `00 Учебник/10 ML Systems/05 Численные форматы и mixed precision.md`
- Modify: `00 Учебник/10 ML Systems/07 Profiling ML-нагрузки.md`
- Create: `00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению.md`
- Modify: `00 Учебник/14 Inference и оптимизация/56 FlashAttention.md`
- Modify: `00 Учебник/11 Pre-training и Scaling/44a Processes, collectives и DDP.md`
- Modify: `00 Учебник/11 Pre-training и Scaling/44b Gradient checkpointing и offload.md`
- Modify: `00 Учебник/11 Pre-training и Scaling/44c Tensor и sequence parallelism.md`
- Modify: `00 Учебник/11 Pre-training и Scaling/44d Pipeline parallelism.md`
- Modify: `00 Учебник/11 Pre-training и Scaling/44e ZeRO, FSDP2, DeviceMesh и DTensor.md`
- Modify: `00 Учебник/11 Pre-training и Scaling/44f Expert и hybrid parallelism.md`
- Modify: `00 Учебник/11 Pre-training и Scaling/44 Distributed training и mixed precision.md`
- Create: `06 Практика/21 Профилировать и ускорить Transformer kernel.md`
- Create: `06 Практика/Contracts/stanford-cs336-a2.yml`
- Create: `00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/`
- Modify: `05 Источники/Courses/Stanford CS336 Spring 2026/coverage.yml`
- Modify: `05 Источники/Courses/Stanford CS336 Spring 2026/visuals.yml`
- Modify: `05 Источники/asset-registry.yml`
- Modify: `publishing/navigation.yml`

**Interfaces:**
- Covers Lectures 5–8 and Assignment 2.
- Practice requires correctness tests before timing and records device, dtype, shapes, warmup, synchronization, profiler traces, peak memory, and throughput.

- [ ] **Step 1: Add the missing TPU comparison without diluting the GPU route**

Explain execution and memory hierarchy differences only to the depth supported by Lecture 5. Use the original comparison figures and avoid treating vendor peak FLOPs as measured training throughput.

- [ ] **Step 2: Build a kernel-development narrative**

Move from eager PyTorch dispatch and memory traffic to fusion, tiled programs, Triton's program IDs and masks, autotuning, numerical tolerances, and profiler-guided iteration.

- [ ] **Step 3: Integrate FlashAttention forward and backward**

Connect online softmax and IO complexity to the actual tiled forward pass, saved statistics, recomputation, backward dependencies, and correctness tests. Use Stanford's stepwise kernel visuals where stronger than existing ones.

- [ ] **Step 4: Reconcile parallelism lectures with existing chapters**

Map data, tensor, sequence, pipeline, expert, optimizer-state, parameter, activation, and gradient sharding to the resource being partitioned. Add the course's worked communication/memory arithmetic and preserve Bookvar's stronger multi-dimensional treatment.

- [ ] **Step 5: Write Capstone 2**

Specify profiling, fused RMSNorm in Triton, FlashAttention forward/backward,
DDP, FSDP, activation checkpointing, optimizer-state sharding, correctness
gates, benchmarks, and an evidence bundle. The contract defines exact official
adapters/tests, CPU correctness mode, single-GPU kernel mode, multi-GPU full
mode, hardware metadata, and pass/fail thresholds. Do not provide completed
implementation.

- [ ] **Step 6: Inspect desktop and narrow renders**

Check roofline, memory hierarchy, GPU/TPU, tiling, FlashAttention, and parallelism figures at their rendered size; recrop from originals if labels are illegible, recording every crop.

- [ ] **Step 7: Verify and commit**

```bash
cd publishing
npm run check:course-ingestion
npm test
npm run build
npm run check:links
```

```bash
git commit -m "docs: integrate CS336 systems kernels and parallelism"
```

---

### Task 4: Make scaling laws a reproducible experiment rather than a slogan

**Files:**
- Modify: `00 Учебник/11 Pre-training и Scaling/43 Scaling laws.md`
- Create: `06 Практика/22 Провести scaling-law campaign.md`
- Create: `06 Практика/Contracts/stanford-cs336-a3.yml`
- Create: `00 Учебник/Assets/Figures/curated/stanford-cs336-2026/scaling/`
- Modify: `05 Источники/Courses/Stanford CS336 Spring 2026/coverage.yml`
- Modify: `05 Источники/Courses/Stanford CS336 Spring 2026/visuals.yml`
- Modify: `05 Источники/asset-registry.yml`
- Modify: `publishing/navigation.yml`

**Interfaces:**
- Covers Lectures 9 and 11 and Assignment 3.
- Practice output: experiment ledger, FLOPs budget, fitted alternatives, residual plots, uncertainty, out-of-budget prediction, and revealed-result evaluation.

- [ ] **Step 1: Audit current scaling claims**

Remove categorical rules such as a universal fixed tokens-per-parameter ratio. Distinguish empirical fit, compute-optimal frontier, data constraint, inference-aware optimum, and extrapolation assumption.

- [ ] **Step 2: Integrate both scaling lectures as one argument**

Explain why a campaign samples model/data/compute configurations, how IsoFLOP curves are constructed, which functional forms compete, how residuals expose misspecification, and why uncertainty grows outside the observed range.

- [ ] **Step 3: Preserve the lecture's empirical sequences**

Use full curve-building sequences instead of isolated final plots. Captions must tell the reader which axis or residual invalidates a tempting conclusion.

- [ ] **Step 4: Write Capstone 3**

Specify experiment selection under a budget, API observations, model fitting,
uncertainty, held-out extrapolation, and a post-reveal error analysis. The
contract pins the starter/API interface, supplies an offline fixture for local
mode, and defines the experiment-ledger/evidence schema without exposing
Stanford answers.

- [ ] **Step 5: Verify and commit**

```bash
cd publishing
npm run check:course-ingestion
npm test
npm run build
npm run check:links
```

```bash
git commit -m "docs: integrate CS336 scaling campaign"
```

---

### Task 5: Integrate inference and evaluation as measurement disciplines

**Files:**
- Modify: `00 Учебник/14 Inference и оптимизация/55a Физика LLM inference — prefill, decode и roofline.md`
- Modify: `00 Учебник/14 Inference и оптимизация/55b Scheduling — continuous batching, chunked prefill и prefix caching.md`
- Modify: `00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention.md`
- Modify: `00 Учебник/14 Inference и оптимизация/58 Спекулятивное декодирование.md`
- Modify: `00 Учебник/14 Inference и оптимизация/58b Benchmarking, SLO и эксплуатация inference.md`
- Modify: `00 Учебник/08 Эффективный Attention и длинный контекст/01 MHA, MQA и GQA.md`
- Modify: `00 Учебник/08 Эффективный Attention и длинный контекст/02 MLA и сжатие KV-cache.md`
- Modify: `00 Учебник/14 Inference и оптимизация/55c Serving engines — vLLM, SGLang, TensorRT-LLM и FlashInfer.md`
- Modify: `00 Учебник/14 Inference и оптимизация/57 Квантизация языковых моделей.md`
- Modify: `00 Учебник/14 Inference и оптимизация/57b Сжатие моделей — pruning, distillation и low-rank.md`
- Modify: `00 Учебник/18 Evaluation и методология/59 Оценивание моделей и контаминация.md`
- Create: `00 Учебник/Assets/Figures/curated/stanford-cs336-2026/inference-evaluation/`
- Modify: `05 Источники/Courses/Stanford CS336 Spring 2026/coverage.yml`
- Modify: `05 Источники/Courses/Stanford CS336 Spring 2026/visuals.yml`
- Modify: `05 Источники/asset-registry.yml`

**Interfaces:**
- Covers Lectures 10 and 12.
- Bookvar's current serving material is broader; Stanford supplies a coherent derivation and evaluation discipline, not replacement summaries.

- [ ] **Step 1: Rebuild the prefill/decode derivation from tensor shapes**

Derive compute, memory traffic, KV growth, batching opportunity, and latency metrics separately for prefill and decode. Connect the derivation to PagedAttention and continuous batching.

- [ ] **Step 2: Integrate the inference algorithm comparisons**

Use course examples for KV caching, speculative decoding, and scheduling only where assumptions are explicit. Keep framework-specific claims tied to measured hardware and versions.

Close the lecture units on MQA/GQA/MLA, quantization, pruning/distillation,
kernel/runtime boundaries, and serving-engine assumptions against the named
existing chapters. Use `covered-existing` only when the chapter contains a
specific reciprocal source-unit link.

- [ ] **Step 3: Expand evaluation construct validity**

Distinguish task definition, dataset, prompt/protocol, decoder, metric, statistical uncertainty, contamination, model evaluation, system evaluation, and agent evaluation.

- [ ] **Step 4: Replace the legacy inference-note role**

Migrate any unique, supported claim from `Courses/Stanford CS336/CS336 — Inference.md`; mark the page as a redirect or source-only historical note and remove it from search/navigation.

- [ ] **Step 5: Render and commit**

```bash
cd publishing
npm run check:course-ingestion
npm test
npm run build
npm run check:links
```

```bash
git commit -m "docs: integrate CS336 inference and evaluation"
```

---

### Task 6: Build the pretraining-corpus route and capstone

**Files:**
- Modify: `00 Учебник/11 Pre-training и Scaling/01 Данные и pre-training.md`
- Modify: `00 Учебник/11 Pre-training и Scaling/41 Сбор, очистка и смеси данных.md`
- Modify: `00 Учебник/11 Pre-training и Scaling/53 Синтетические данные и учебные программы.md`
- Create: `00 Учебник/11 Pre-training и Scaling/41a Дедупликация, PII и контроль качества корпуса.md`
- Create: `06 Практика/23 Собрать воспроизводимый pretraining corpus.md`
- Create: `06 Практика/Contracts/stanford-cs336-a4.yml`
- Create: `00 Учебник/Assets/Figures/curated/stanford-cs336-2026/data/`
- Modify: `05 Источники/Courses/Stanford CS336 Spring 2026/coverage.yml`
- Modify: `05 Источники/Courses/Stanford CS336 Spring 2026/visuals.yml`
- Modify: `05 Источники/asset-registry.yml`
- Modify: `publishing/navigation.yml`

**Interfaces:**
- Covers Lectures 13–14 and Assignment 4.
- Corpus capstone input is a reproducibly identified Common Crawl subset; output includes retained documents, audit logs, mixture, tokenizer output, and fixed-budget validation evidence.

- [ ] **Step 1: Explain the corpus as a sequence of lossy decisions**

Cover source selection, WARC/WET extraction, language identification, heuristic and learned quality filters, toxicity policy, PII detection/masking, exact deduplication, MinHash/LSH fuzzy deduplication, contamination checks, mixture design, tokenization, and dataset versioning.

- [ ] **Step 2: Add failure analysis at every filter**

For each stage, state what is removed, likely false positives/negatives, what metadata must be logged, and which downstream measurement can reveal damage.

- [ ] **Step 3: Integrate synthetic data and mixture evidence**

Separate synthetic generation, transformation, filtering, curriculum, and replay. Use course plots and tables to show measured effects rather than treating synthetic data as one technique.

- [ ] **Step 4: Write Capstone 4**

Require retention/diversity statistics, manual audits, PII and deduplication
error analysis, contamination checks, a mixture manifest, tokenizer statistics,
and a small fixed-token training comparison. The contract pins source objects,
defines a tiny local WET fixture and a full-data mode, and enumerates output
hashes, audit tables, and pass/fail checks.

- [ ] **Step 5: Render and commit**

```bash
cd publishing
npm run check:course-ingestion
npm test
npm run build
npm run check:links
```

```bash
git commit -m "docs: integrate CS336 pretraining data pipeline"
```

---

### Task 7: Integrate post-training, RLVR, and the reasoning capstone

**Files:**
- Modify: `00 Учебник/12 Post-training и Alignment/01 SFT и instruction data.md`
- Modify: `00 Учебник/12 Post-training и Alignment/03 Reward modeling.md`
- Modify: `00 Учебник/12 Post-training и Alignment/04 Policy gradient и PPO для LLM.md`
- Modify: `00 Учебник/12 Post-training и Alignment/05 DPO.md`
- Modify: `00 Учебник/12 Post-training и Alignment/06 RLVR и verifiers.md`
- Modify: `00 Учебник/12 Post-training и Alignment/07 GRPO и DeepSeek-R1.md`
- Modify: `00 Учебник/12 Post-training и Alignment/08 Reasoning distillation.md`
- Create: `06 Практика/24 Post-training и RLVR для математического reasoning.md`
- Create: `06 Практика/Contracts/stanford-cs336-a5.yml`
- Create: `00 Учебник/Assets/Figures/curated/stanford-cs336-2026/posttraining/`
- Modify: `05 Источники/Courses/Stanford CS336 Spring 2026/coverage.yml`
- Modify: `05 Источники/Courses/Stanford CS336 Spring 2026/visuals.yml`
- Modify: `05 Источники/asset-registry.yml`
- Modify: `publishing/navigation.yml`

**Interfaces:**
- Covers Lectures 15–16, Assignment 5, and its optional safety-alignment branch.
- Practice compares prompting, SFT, expert iteration, GRPO, MaxRL, and GSPO with response masks, log probabilities, verifiable rewards, multiple seeds, and verifier audit; DPO is a separate preference branch.

- [ ] **Step 1: Reconcile objectives and data flows**

For SFT, reward modelling, PPO, DPO, and RLVR, specify the data, model(s), sampled objects, loss, reference policy role, online/offline boundary, and failure modes. Do not compress each method to one paragraph.

- [ ] **Step 2: Integrate the course's runnable RLVR pathway**

Explain response masks, token log-probabilities, group-normalized advantages, on-policy/off-policy distinctions, policy-gradient loss, KL control, verifiable reward, sampling throughput, and reward hacking.

- [ ] **Step 3: Compare current reasoning objectives carefully**

Present GRPO, MaxRL, and GSPO as concrete objectives with assumptions and empirical context. Keep DeepSeek-R1 family claims tied to its paper rather than the course alone.

- [ ] **Step 4: Write Capstone 5**

Require baseline prompts, SFT, expert iteration, at least three random seeds in
full mode, comparable sampling budgets, curves, ablations, DPO safety branch,
verifier false-positive/false-negative audit, and an evidence bundle. The
contract defines a deterministic tiny local smoke mode, a budgeted single-GPU
mode, the full multi-seed mode, exact tests/adapters, and evidence JSON so the
exercise remains runnable without pretending every reader has the full budget.

- [ ] **Step 5: Render and commit**

```bash
cd publishing
npm run check:course-ingestion
npm test
npm run build
npm run check:links
```

```bash
git commit -m "docs: integrate CS336 post-training and RLVR"
```

---

### Task 8: Integrate the multimodal lecture into the existing VLM route

**Files:**
- Modify: `00 Учебник/16 Multimodal Models/64 Мультимодальные модели.md`
- Modify: `00 Учебник/16 Multimodal Models/64a Connectors и fusion.md`
- Modify: `00 Учебник/16 Multimodal Models/64b Разрешение, tiling и пространственные позиции.md`
- Modify: `00 Учебник/16 Multimodal Models/64c Обучение VLM — alignment, instruction tuning и данные.md`
- Modify: `00 Учебник/16 Multimodal Models/64e Видео, аудио и omni-модели.md`
- Create: `00 Учебник/16 Multimodal Models/64h Early fusion и Chameleon.md`
- Create: `00 Учебник/Assets/Figures/curated/stanford-cs336-2026/multimodality/`
- Modify: `05 Источники/Courses/Stanford CS336 Spring 2026/coverage.yml`
- Modify: `05 Источники/Courses/Stanford CS336 Spring 2026/visuals.yml`
- Modify: `05 Источники/asset-registry.yml`

**Interfaces:**
- Covers Lecture 17: CLIP, SigLIP, LLaVA, LLaVA-OneVision, Qwen-VL/Qwen2-VL/Qwen3-VL, and Chameleon.
- The current VLM worktree changes are user-owned; reconcile them in place and never overwrite them wholesale.

- [ ] **Step 1: Diff the current VLM edits before touching them**

Record which paragraphs and visuals are already stronger than the Stanford lecture. Apply narrow patches only where Lecture 17 adds a missing mechanism, experiment, or architecture.

- [ ] **Step 2: Integrate the representation and alignment progression**

Connect contrastive encoders, SigLIP's objective, frozen/fine-tuned vision towers, projection connectors, instruction tuning, variable resolution, and video/omni extensions.

- [ ] **Step 3: Add early-fusion generative multimodality**

Use Chameleon to explain tokenizing multiple modalities into one autoregressive vocabulary, how this differs from a vision-encoder connector architecture, and what training/inference consequences follow.

- [ ] **Step 4: Preserve the executable lecture's meaningful figures**

Import original visual assets and step sequences with pinned provenance. Do not screen-capture code when the executable source itself can be linked and rendered.

- [ ] **Step 5: Close Stanford coverage**

Confirm every Lecture 1–17 and Assignment 1–5 unit has a final disposition. Guest lectures 18–19 must remain explicitly `source-only` or `excluded` until official materials are available.

- [ ] **Step 6: Verify and commit**

```bash
cd publishing
npm run check:course-ingestion
npm test
npm run build
npm run check:links
```

```bash
git commit -m "docs: integrate CS336 multimodality"
```

---

### Task 9: Retire parallel legacy summaries

**Files:**
- Modify: `Courses/Stanford CS336/_index.md`
- Modify: `Courses/Stanford CS336/CS336 — Tokenization.md`
- Modify: `Courses/Stanford CS336/CS336 — Mixture of Experts.md`
- Modify: `Courses/Stanford CS336/CS336 — Scaling Laws.md`
- Modify: `Courses/Stanford CS336/CS336 — Inference.md`
- Modify: `publishing/navigation.yml`
- Modify: `publishing/tests/publication-policy.test.ts`

**Interfaces:**
- Every legacy page redirects to the canonical Bookvar chapter and the pinned Spring 2026 course hub.
- Redirect pages are excluded from search and primary navigation.

- [ ] **Step 1: Audit unique supported claims**

Move only claims that survive primary-source verification and are absent from the integrated chapters.

- [ ] **Step 2: Convert all four pages to minimal redirects**

Keep historical provenance, canonical destination, and course-hub link. Remove standalone explanatory prose that would compete with the textbook.

- [ ] **Step 3: Add a regression test**

Require all four pages to have redirect status and forbid their old routes in visible navigation.

- [ ] **Step 4: Verify and commit**

```bash
cd publishing
npm test -- publication-policy.test.ts
npm run build
npm run check:links
```

```bash
git commit -m "docs: retire legacy CS336 summaries"
```
