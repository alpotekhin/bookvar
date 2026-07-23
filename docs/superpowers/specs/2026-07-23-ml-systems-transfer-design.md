# ML Systems Transfer Design

Date: 2026-07-23  
Status: approved direction, awaiting written-spec review  
Scope: Bookvar system foundations, distributed training, LLM inference, and
corresponding practice

## Purpose

Bookvar already contains several substantial inference chapters, but the
reader reaches them without a continuous account of GPU execution, memory
traffic, distributed communication, and the arithmetic of a Transformer.
Consequently, terms such as roofline, tensor parallelism, KV-cache pressure, and
disaggregated serving appear locally but do not form one causal model.

The transfer will turn the existing material into a continuous ML systems
course. It will use the Harvard ML Systems Book for its linear systems
narrative, worked examples, and native SVG illustrations, and Efficient Deep
Learning Systems for quantitative theory, implementation details, experiments,
and laboratories. Primary papers and official documentation remain the source
of truth for individual mechanisms.

## Editorial principle

The project does not benefit from paraphrasing a clear existing explanation
into weaker synthetic prose. Source material is therefore retained in the
language in which the course, book, slide, laboratory, or notebook was written.
Bookvar does not translate a source merely to make every paragraph of a Russian
route Russian.

Every imported fragment is marked as one of:

- `original excerpt`: source wording retained;
- `adaptation`: source argument or exercise reworked for Bookvar;
- `editorial bridge`: new text connecting several sources.

An excerpt stays in its source language. If a short explanation is required for
navigation, terminology, or connection to the preceding chapter, it is added as
an `editorial bridge` and is not presented as a translation of the source.

For every imported text, image, table, animation, or code fragment, the
repository records the author, title, direct URL, pinned commit where practical,
license when stated, local path, transformation type, and destination chapters.
Captions on published pages provide visible attribution rather than hiding it
only in a global bibliography.

Harvard textbook, slide, and laboratory derivatives retain the
CC BY-NC-SA 4.0 conditions. Harvard MLSys·im is Apache 2.0. TinyTorch's
directory-level license is checked at the pinned revision before ingestion;
the repository summary and the current directory license have not always
agreed, and the directory license is authoritative. Efficient DL Systems code
and original material retain the MIT notice. Third-party images embedded inside
slide decks are checked at the level of the original asset before local
publication.

## Learning sequence

The system line will follow the dependency order below.

1. GPU execution and the memory hierarchy.
2. Arithmetic intensity and the roofline model.
3. Transformer arithmetic and memory accounting.
4. Numerical formats, mixed precision, and stability.
5. Data parallelism and collective communication.
6. ZeRO and FSDP.
7. Tensor, pipeline, sequence, expert, and hybrid parallelism.
8. Autoregressive inference: prefill, decode, and KV-cache.
9. FlashAttention, fusion, compilation, and kernels.
10. Scheduling, continuous batching, and PagedAttention.
11. Distributed and disaggregated serving.
12. Quantization, speculative decoding, and KV compression.
13. Metrics, queueing, SLOs, capacity planning, and operations.

This is a line inside the main textbook, not a separate ML systems handbook.
The reader should be able to move from model architecture to training and
serving without encountering a second competing table of contents.

## Chapter changes

### New foundations before distributed training

Add focused chapters covering:

- GPU, SIMT, Tensor Cores, memory hierarchy, tiling, and data movement;
- latency, bandwidth, FLOP/s, arithmetic intensity, and roofline;
- parameter, activation, gradient, optimizer-state, and communication memory;
- operation-level arithmetic for attention, FFN, normalization, and
  cross-entropy;
- numerical formats, accumulation, loss scaling, and mixed precision.

The hardware chapter will use selected Harvard hardware SVGs. The quantitative
chapters will use the calculations and exercises across the complete Efficient
DL Systems course. Weeks 1, 2, and 6 form the computational foundation, but
they are not the limit of the transfer.

### Replace the single distributed-training overview

The current chapter 44 is retained as an entry point and split into a sequence:

- data parallelism and AllReduce;
- ZeRO and FSDP;
- tensor and sequence parallelism;
- pipeline parallelism and bubble;
- expert parallelism and all-to-all;
- hybrid or 3D parallelism;
- checkpointing, offload, failure recovery, and distributed checkpoints.

The sequence uses the Harvard distributed-training chapter and SVG suite for
orientation. Efficient DL Systems weeks 3–5 supply the derivations,
communication arithmetic, failure modes, and implementable exercises.

### Reconcile the existing inference chapters

The existing chapters 55a through 58b are not discarded. They are edited
against the new foundations:

- repeated definitions of prefill, decode, KV-cache, and roofline are
  consolidated;
- each chapter assumes only concepts introduced earlier;
- Harvard inference and serving SVGs replace weaker or redundant diagrams;
- week 8 measurements become worked examples;
- week 9 material expands quantization, speculative decoding, and KV
  compression;
- model-serving queueing and capacity examples extend benchmarking and SLOs.

Legacy overview files remain unpublished or become redirects after their useful
content has been migrated and verified.

## Visual system

Each complex mechanism receives three visual roles when the sources provide
them:

1. an orientation diagram showing components and boundaries;
2. a step-by-step diagram showing data, memory, or communication movement;
3. a comparison diagram showing alternatives and trade-offs.

Priority Harvard assets include:

- memory hierarchy, systolic-array dataflow, and kernel fusion;
- DP, TP, PP, EP, ZeRO, collectives, the 3D parallelism cube, and the
  parallelism decision tree;
- continuous batching, KV fragmentation, PagedAttention, disaggregated serving,
  the decode roofline, the KV-cache ladder, and queueing curves;
- TTFT/TPOT and static-versus-continuous batching.

Assets are stored locally at original quality when permitted. Crops preserve
labels and contextual meaning. CSS constrains their displayed width without
downsampling the source. Every figure has a descriptive caption, source link,
author, and license record.

## Practice line

Practice runs beside the textbook rather than appearing as detached notebook
links.

### First vertical slice

The first laboratory asks the learner to measure a small open model and explain:

- the memory occupied by parameters and KV-cache;
- the difference between prefill and decode time;
- how prompt length, generated length, batch size, and padding affect results;
- when prefix reuse changes latency;
- why an optimization may help prefill but not decode.

The laboratory adapts the Efficient DL Systems week 8 seminar. It provides a
reproducible environment, pinned model and dependency versions, CPU-safe
explanatory fallback, expected plots, and questions whose answers link back to
the relevant textbook sections.

### Subsequent laboratories

- implement and measure collective communication;
- implement checkpointing and parameter offload;
- compare tensor and pipeline parallelism;
- build the Efficient DL Systems mini inference engine;
- add an asynchronous request queue and continuous batching;
- implement W8A8 and SmoothQuant in Triton;
- implement speculative decoding with correct rejection and KV rollback;
- tune replicas and batching against TTFT, TPOT, throughput, and cost SLOs.

Interactive browser exercises will be extracted from the Harvard Marimo serving
laboratory where the calculation can run without a GPU.

## Source-language and site-language policy

The source language is canonical for imported material:

- an English Harvard chapter, caption, slide, or laboratory remains English;
- the English-language Efficient DL Systems materials remain English;
- a Russian source remains Russian;
- code comments and notebook prose remain in their original language unless an
  adaptation requires changes.

Russian remains the canonical language for Bookvar's own navigation and
editorial bridges, but it does not force translation of imported educational
material. The English route reuses the same English originals. Russian source
fragments may remain Russian there with an English editorial bridge when they
cannot be replaced by an equally strong English source.

Figures, formulas, routes, provenance, and factual boundaries remain aligned
across both site locales. The interface must clearly distinguish Bookvar
editorial text from imported source material.

## Complete-course coverage

Transfer is based on a coverage matrix, not a shortlist of convenient chapters.
Every textbook chapter, lecture, seminar, homework, lab, simulator, and visual
collection in the two main repositories is inspected and assigned one of four
outcomes:

- `integrate`: adds a missing explanation, example, visual, or exercise;
- `cross-link`: overlaps with stronger existing Bookvar material but improves
  further study;
- `source-only`: supports claims or editorial verification without being copied;
- `exclude`: unrelated to the Bookvar scope, obsolete, or weaker than an
  already selected source; the reason is recorded.

### Efficient Deep Learning Systems

All available 2026 course weeks enter the matrix:

1. GPU architecture, CUDA, PyTorch CUDA operations, and benchmarking;
2. profiling, mixed precision, storage/loading, dynamic padding, JPEG decode,
   PyTorch Profiler, memory snapshots, and Nsight Systems;
3. distributed training, data parallelism, collectives, and AllReduce;
4. tensor, pipeline, and sequence parallelism, checkpointing, and offload;
5. DeviceMesh, DTensor, FSDP2, distributed checkpoints, and communication
   overlap;
6. deep-learning arithmetic, fusion, `torch.compile`, GPU memory hierarchy,
   Triton, and Liger kernels;
7. web-service deployment and production-ready service foundations;
8. inference metrics, KV-cache, continuous batching, FlashAttention,
   PagedAttention, serving frameworks, prefill/decode measurements, and the
   mini inference engine;
9. architecture optimization, quantization, speculative decoding, KV
   compression, distillation, Triton matrix multiplication, and profiling with
   Nsight Systems.

Past course iterations are searched only for valuable material absent from the
2026 edition. A newer file does not automatically replace a clearer older
explanation; differences are recorded.

### Harvard ML Systems curriculum

The complete Harvard curriculum enters the matrix rather than only its
inference and distributed-training chapters:

- both textbook volumes;
- interactive Marimo laboratories;
- TinyTorch's progressive implementation modules;
- MLSys·im quantitative infrastructure simulations;
- chapter slide decks and their native visual assets;
- hardware kits when they teach portable deployment constraints;
- assessment material when it adds systems-design exercises not present in the
  textbook.

The source audit begins from a pinned `dev` revision because `main` contains the
older stable single-volume edition while `dev` contains the materially richer
two-volume replacement. Preview status is retained for Vol II and early-release
labs; stable `main` is used for comparison, not assumed to be the most complete
source.

The expected coverage is deliberately broad:

- systems framing through the Data–Algorithm–Machine relationship, the Iron
  Law, deployment constraints, and energy hierarchy;
- ML workflow, data engineering, active learning, data quality, versioning,
  data gravity, data debt, and data selection;
- neural-network computation and architecture viewed through matrix
  multiplication, activation memory, arithmetic intensity, and inductive bias;
- framework internals: tensors, computation graphs, eager/static execution,
  automatic differentiation, dispatch overhead, interoperability, compilers,
  and the hardware boundary;
- training, profiling, mixed precision, memory accounting, and accelerator
  choice;
- compression beyond quantization: pruning, sparsity, low-rank methods,
  distillation, and architecture optimization;
- hardware acceleration, benchmarking methodology, model serving, and MLOps;
- compute infrastructure, network fabrics, storage paths, collective
  communication, distributed training, fault tolerance, and fleet
  orchestration;
- performance engineering and inference at scale;
- edge/on-device and federated learning as an optional deployment extension;
- security, privacy, robustness, sustainability, responsible engineering, and
  responsible AI as measurable system properties rather than detached ethics
  notes.

The practical audit includes all Vol I and Vol II Marimo labs, not only serving
labs. TinyTorch's progressive tensor-to-Transformer-to-KV-cache modules are
mapped onto Bookvar practice. The native slide decks and original SVG
collections are searched chapter by chapter; imported or legacy raster images
are not assumed to be reusable until their figure-level credit is checked.

Likely Bookvar destinations therefore include foundations of ML systems, data
and input pipelines, ML/DL foundations, training, performance engineering,
hardware acceleration, compression, deployment, serving, distributed training,
network fabrics, storage, reliability, inference at scale, fleet operation,
security, governance, sustainability, and interactive practice. The final
inclusion decision is made per source object, not per repository directory.

## Provenance records

A machine-readable or consistently structured registry entry accompanies every
local asset:

- `asset_id`;
- original author and resource;
- source URL and pinned revision;
- retrieved date;
- license and required notice;
- original filename and local filename;
- whether the object is original, cropped, translated, or annotated;
- pages that use it.

The build should fail when a newly referenced curated asset lacks a provenance
record. This validation may be introduced after the first import batch, but all
first-batch assets are recorded in the intended format.

## Implementation batches

### Batch 0: complete source matrix

- inventory every chapter, lecture, seminar, homework, lab, simulator, and
  reusable visual in both repositories;
- assign `integrate`, `cross-link`, `source-only`, or `exclude`;
- map each retained item to a Bookvar destination and provenance record;
- compare 2026 Efficient DL Systems materials with older iterations where the
  current course references or omits a useful artifact;
- compare Harvard's pinned `dev` two-volume curriculum with stable `main`,
  retaining preview labels for material not yet in the live edition;
- identify Bookvar chapters that must be created, split, merged, or reordered.

### Batch 1: computational foundation and input pipeline

- create the GPU, roofline, Transformer arithmetic, and mixed-precision
  chapters;
- cover benchmarking, profiling, storage/loading, dynamic padding, data
  decoding, and input-pipeline bottlenecks;
- import the relevant Harvard SVGs with provenance;
- connect the sequence to model architecture and distributed training;
- create the week 8 measurement laboratory.

### Batch 2: distributed training

- split chapter 44;
- import DP/TP/PP/EP/ZeRO/3D SVGs;
- incorporate Efficient DL Systems weeks 3–5;
- add collectives and memory-efficiency practice.

### Batch 3: inference reconciliation

- revise chapters 55a–58b;
- incorporate Harvard inference/serving and Efficient DL Systems weeks 8–9;
- remove duplicated explanations;
- add the mini inference-engine project.

### Batch 4: optimization and operations

- expand Triton, quantization, speculative decoding, compression, queueing,
  SLOs, and capacity planning;
- incorporate deployment, MLOps, reliability, network, storage, fleet,
  security, robustness, sustainability, and responsible-system material from
  the Harvard curriculum where it extends the ML/LLM route;
- add interactive serving controls;
- complete the English route.

## Verification

Every batch is complete only after:

- sources and licenses have been checked;
- imported text is labelled by transformation type;
- images render at desktop and narrow widths without clipping;
- formulas render in the built site;
- previous/next links follow the intended learning sequence;
- Russian and English routes are either aligned or explicitly marked as
  untranslated;
- unit tests, full site build, built-link checks, and rendered-page inspection
  pass;
- an editorial reread confirms that the pages form a textbook argument rather
  than a catalogue of definitions.

## Deliberate exclusions

- no automatic bulk copy of entire repositories;
- no restriction to a preselected subset of course weeks or book chapters;
- no recreation of existing strong figures as Mermaid or synthetic SVG;
- no silent benchmark claims without hardware and software context;
- no assumption that a repository license relicenses third-party slide images;
- no independent second table of contents for ML systems;
- no deletion of existing chapters until their unique content has been mapped
  and migrated.
