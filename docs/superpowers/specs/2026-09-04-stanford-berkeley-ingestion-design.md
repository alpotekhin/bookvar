# Stanford CS336 and Berkeley Advanced LLM Agents — Ingestion Design

## Status

Approved direction from the user on 2026-09-04. This document defines the
editorial architecture before implementation. It covers Stanford CS336 Spring
2026 and Berkeley Advanced Large Language Model Agents Spring 2025.

## Goal

Integrate the complete educational substance of both courses into the existing
Bookvar narrative. Course boundaries must not become the primary reading
structure: the textbook remains a coherent route through language modelling,
ML systems and agents, while course hubs preserve provenance and expose full
coverage.

The result must retain the courses' meaningful visual explanations, runnable or
verifiable exercises, primary references and distinctions between established
results, experimental evidence and open research questions.

## Non-goals

- Do not create or update the English publication route in this project.
- Do not translate course prose merely to make it Russian. Strong original
  English passages may remain in English with a short editorial bridge.
- Do not publish solution code for Stanford or Berkeley assignments.
- Do not replace existing course figures with Mermaid, generated SVG or
  AI-created diagrams.
- Do not create one short Bookvar page per lecture merely to summarize what the
  lecturer discussed.
- Do not push, merge or deploy without a separate user instruction.

## Editorial principle

Bookvar owns the learning sequence; Stanford and Berkeley supply source
material. A reader following the textbook should encounter each explanation at
the point where it answers the current technical question, rather than leaving
the narrative to read a detached course note.

Every integrated section should have this shape:

1. a concrete problem or observation;
2. the course's full explanatory development of the mechanism;
3. the original formula, example, experiment or figure that carries the idea;
4. an explanation of what the evidence establishes and what it does not;
5. links to the primary papers, implementation and course artifact;
6. a transition to the next concept or a verifiable exercise.

Course hubs remain in `05 Источники` as provenance and coverage maps. They are
not substitutes for textbook chapters.

## Definition of complete coverage

"Transfer all knowledge" does not mean copying every administrative slide or
duplicating the same definition in several chapters. It means that every
educational unit in the two courses is explicitly accounted for.

An educational unit is any of the following:

- a concept, mechanism or architectural distinction;
- a mathematical derivation or resource calculation;
- an empirical result, caveat or failure mode;
- a worked example or case study;
- an assignment contract, evaluation method or reproducibility requirement;
- a meaningful figure, table, animation or sequence of slides;
- a primary source or implementation that changes the interpretation of the
  material.

Each unit must appear in a course coverage ledger with exactly one disposition:

- `integrated` — incorporated into a named Bookvar chapter;
- `covered-existing` — already explained adequately, with the course linked as
  corroborating or alternative material;
- `source-only` — preserved in the course hub because it is useful reference
  material but would interrupt the main narrative;
- `excluded` — administrative, duplicated or obsolete material, with a written
  reason.

No lecture, lecture section, assignment section or meaningful visual may be
absent from the ledger. Completion is measured against these ledgers, not
against the number of new pages.

## Source snapshots and provenance

Before editorial work begins, record immutable source snapshots:

- Stanford course page, lecture repository, recordings and Assignment 1–5
  repositories for Spring 2026;
- Berkeley course page, all twelve official slide decks, recordings and listed
  readings for Spring 2025;
- official lab or project artifacts when available;
- the exact commit SHA, file path, slide/page number and retrieval date for each
  reused artifact.

The source repository or downloaded snapshot is immutable. Editorial pages and
asset metadata point back to it.

Every reused fragment records:

- author or lecturer;
- course and offering;
- lecture or assignment;
- source URL pinned to a revision when possible;
- page, slide, figure or code location;
- language;
- transformation: original excerpt, crop, adaptation or editorial bridge;
- license or the course-level usage rule confirmed by the user.

## Visual coverage

Images are part of the teaching content, not decoration. A separate visual
ledger must enumerate every figure, table, animation and multi-slide sequence
that carries an explanation.

For each visual, record:

- the question it helps the reader answer;
- its source location and attribution;
- whether it is used as one frame, a careful crop or a sequence;
- its destination chapter and position in the argument;
- whether labels remain legible at desktop and narrow viewport widths;
- whether a caption explains what to inspect and what conclusion follows.

A visual can be excluded only if it is administrative, redundant, illegible
with no better source, or factually obsolete. The reason remains in the ledger.

Each complex textbook topic should, where the source material supports it,
contain:

1. an orienting figure;
2. a mechanism or step-by-step figure;
3. a comparison or empirical figure.

Several consecutive slides that build one argument must be preserved as a
sequence rather than reduced to one arbitrary screenshot.

## Stanford CS336 integration

Stanford CS336 forms the practical spine of the language-model and ML-systems
parts of Bookvar.

### Narrative placement

1. Text representation: bytes, Unicode, BPE training and tokenization effects.
2. Tensor shapes and PyTorch mechanics: `einops`, dimensions and resource
   accounting.
3. Transformer construction: embeddings, RMSNorm, RoPE, SwiGLU, attention,
   cross-entropy and AdamW.
4. Architecture choices and hyperparameters.
5. Attention alternatives and mixture of experts.
6. GPU and TPU execution, memory hierarchy and arithmetic intensity.
7. Kernel composition, Triton, fused RMSNorm and FlashAttention.
8. Data, tensor, sequence, pipeline and expert parallelism; DDP and FSDP.
9. Scaling-law experiment design and compute-optimal training.
10. Inference: prefill/decode, KV cache, batching, PagedAttention, speculative
    decoding and serving economics.
11. Evaluation: construct validity, contamination, model versus system versus
    agent evaluation.
12. Pretraining data: sources, extraction, filtering, PII, deduplication,
    mixture design and synthetic data.
13. SFT, preference optimization, RLHF and RLVR.
14. Multimodality: contrastive encoders, connectors, native multimodal models
    and omni systems.

### Practical spine

The five Stanford assignments become five connected Bookvar capstones while
retaining existing narrow laboratories as prerequisites or sub-experiments.

#### Capstone 1 — Language model from scratch

Byte-level BPE, tokenizer tests, embeddings, RMSNorm, SwiGLU, RoPE, causal
attention, Transformer blocks, stable cross-entropy, AdamW, schedules, clipping,
data loading, checkpointing and a small training run. The final evidence bundle
contains tests, configuration, loss curves, samples and a resource estimate.

#### Capstone 2 — Systems and kernels

Profiling, resource accounting, Triton execution, fused RMSNorm,
FlashAttention forward and backward, distributed data parallelism, FSDP,
activation checkpointing and optimizer-state sharding. Correctness tests precede
benchmarks; profiles and hardware configuration are retained.

#### Capstone 3 — Scaling campaign

Select experiments under a FLOPs budget, fit competing scaling models, inspect
residuals and uncertainty, make an out-of-budget prediction and evaluate it
after the result is revealed. The deliverable is an experiment ledger rather
than a single fitted equation.

#### Capstone 4 — Pretraining corpus

Process a reproducible Common Crawl subset through extraction, language and
quality filters, toxicity checks, PII masking, exact and MinHash
deduplication, mixture construction and tokenization. Measure retention,
diversity, false positives, contamination and downstream validation loss under
a fixed token budget.

#### Capstone 5 — Post-training and reasoning

Build a prompting baseline, SFT and expert-iteration baseline, then compare
GRPO, MaxRL and GSPO using response masks, log probabilities, verifiable reward
and several random seeds. DPO and safety alignment form a separate preference
branch. The deliverable includes curves, ablations and verifier failure audit.

## Berkeley Advanced LLM Agents integration

Berkeley forms the advanced spine of the agents part of Bookvar. It begins only
after the reader understands language models, tool calls, harness state and
basic agent evaluation.

### Thematic modules

#### Reasoning, search and planning

Integrate inference-time search, self-correction limits, learning to reason,
agent-first versus LLM-first views, internal reasoning as an action, formal and
open-world planning, tree search, irreversible actions and model-based planning.

#### Memory and world models

Distinguish parametric memory, working context and external long-term memory;
explain associative retrieval, HippoRAG, structured memory, knowledge conflict,
continual learning and the use of learned transition models for safer planning.

#### Coding, web and GUI agents

Connect repository navigation, hypothesis testing, compiler and analyzer
feedback, vulnerability discovery, Mind2Web, WebArena, VisualWebArena, OSWorld
and pure-vision GUI agents. Perception, action space, environment state and
grader are treated as separate system interfaces.

#### Formal reasoning and scientific discovery

Cover mathematical data, verified rewards, Lean, LeanDojo, premise retrieval,
interactive theorem proving, autoformalization, AlphaProof-style search,
learned abstractions, concept libraries and symbolic regression.

#### Agent safety and privilege control

Treat the agent as a compound system. Cover direct and indirect prompt
injection, tool and supply-chain attacks, memory and knowledge-base poisoning,
model-level versus end-to-end evaluation, policy enforcement, least privilege
and programmable privilege control.

### Berkeley practice

The practical route uses a verifiable coding-agent contract:

- a model-visible task description and signature;
- hidden functional tests;
- a Lean implementation and specification skeleton;
- a bounded generate, compile, diagnose, repair and verify loop;
- separate compilation, test, specification and proof outcomes;
- `pass@k`, attempts, tokens, cost and wall-clock time;
- a failure taxonomy that separates implementation, specification and proof
  errors.

Only official artifacts may be described as Berkeley assignments. If an
official lab file is unavailable, Bookvar may create an original exercise based
on the verified interface, but it must be labelled as a Bookvar adaptation and
must not attribute unverified details to Berkeley.

The final agent capstone follows the useful AgentX lifecycle: proposal,
milestone, evaluation plan, reproducible artifact or demonstration, evidence
bundle and final report.

## Existing Bookvar content

Integration must begin with a chapter-level audit. Existing strong chapters are
expanded in place; they are not replaced by weaker course summaries. New pages
are created only when the curriculum lacks a stable conceptual home.

The four legacy Stanford notes under `Courses/Stanford CS336` must not remain
parallel summaries. After their claims and sources are audited, they become
redirects, source-map entries or are absorbed into stronger textbook chapters.

Existing Bookvar subjects that need explicit reconciliation include:

- tokenization and the modern decoder block;
- MoE and efficient attention;
- GPU arithmetic, profiling and distributed training;
- scaling laws and pretraining data;
- inference and serving;
- SFT, RLHF, DPO, GRPO and RLVR;
- multimodal models;
- tool use, harness, memory, planning and agent evaluation;
- the `Agentic learning` research overview.

## Navigation

The primary sidebar remains concept- and curriculum-oriented. Stanford and
Berkeley names do not become top-level textbook chapters.

Course hubs appear under Sources and provide:

- a complete lecture and assignment inventory;
- the coverage ledger;
- links from every source unit to its Bookvar destination;
- the visual ledger;
- offering, version and retrieval metadata;
- unresolved provenance or editorial gaps.

Practices may show a secondary badge such as `Based on Stanford CS336 A2`, but
their titles describe the skill or artifact produced.

## Quality standard for a completed chapter

A chapter is complete only if it:

- develops a causal explanation in connected prose;
- contains enough detail to reconstruct the mechanism or experiment;
- introduces notation before using it;
- separates model architecture, training, systems, inference and observed
  product behavior;
- includes the strongest relevant course visuals with explanatory captions;
- links the course source and the relevant primary papers;
- states important assumptions and limitations;
- ends with a meaningful transition, experiment or verification task;
- contains no unsupported model claims or unlabelled editorial inference;
- renders correctly in Obsidian and in the published site.

A page that consists primarily of definitions, bullets or source annotations
does not pass this standard.

## Verification gates

### Content completeness

- Every Stanford lecture and Assignment 1–5 is represented in the coverage
  ledger.
- Every Berkeley lecture, reading cluster and confirmed practical artifact is
  represented in the coverage ledger.
- Every row points to a destination or carries an explicit exclusion reason.
- A reverse audit from the course source finds no unaccounted educational unit.

### Visual completeness

- Every meaningful source visual is in the visual ledger.
- Every reused visual has attribution and a stable source location.
- No visual is clipped, illegible or wider than the reading column.
- Multi-step visual arguments retain their sequence.
- Desktop and narrow viewport screenshots are reviewed for all changed chapter
  types.

### Technical publication

- frontmatter and navigation validation pass;
- source coverage and asset-registry tests pass;
- the adapter and Astro site build without errors;
- internal routes, anchors and files have zero broken links;
- legacy redirects are excluded from search;
- generated output is inspected, not inferred from source Markdown alone.

### Editorial review

- Each changed chapter is read from beginning to end after rendering.
- A second pass explicitly searches for summary-like, promotional or
  machine-generated phrasing.
- Course facts are checked against the pinned source rather than memory.
- New material is rejected or rewritten if it still reads as lecture notes
  pasted between unrelated Bookvar paragraphs.

## Delivery sequence

The work is split into independently reviewable batches:

1. immutable source inventory, coverage ledgers and visual ledgers;
2. Stanford foundations and Transformer construction;
3. Stanford systems, kernels and distributed training;
4. Stanford scaling, data, evaluation and inference;
5. Stanford post-training and multimodality;
6. Berkeley reasoning, memory and planning;
7. Berkeley coding, web and GUI agents;
8. Berkeley formal reasoning and scientific discovery;
9. Berkeley agent safety and privilege control;
10. capstone practices, navigation reconciliation and removal of legacy course
    summaries;
11. full content, visual and publication audit.

Each batch receives its own local commit after validation. No batch is pushed or
deployed without the user's explicit approval.

