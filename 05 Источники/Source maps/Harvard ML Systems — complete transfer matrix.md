---
title: Harvard ML Systems — complete transfer matrix
type: source-note
status: editorial
last_updated: 2026-07-23
---

# Harvard ML Systems — complete transfer matrix

Pinned source: [harvard-edge/cs249r_book](https://github.com/harvard-edge/cs249r_book/tree/45ecc8d82fcae70c149cdce550d3b3d3411df913), `dev` commit `45ecc8d82fcae70c149cdce550d3b3d3411df913`. Source language is English. `book/`, `labs/`, and `slides/` are CC BY-NC-SA 4.0; MLSys·im is Apache-2.0; TinyTorch requires directory-level license verification before import. Vol II rows are preview.

| ID | Source path | Material | Adds to Bookvar | Decision | Destination | Assets | Status |
|---|---|---|---|---|---|---|---|
| H-V1-01 | `book/quarto/contents/vol1/introduction/introduction.qmd` | DAM, Iron Law, energy hierarchy | yes | integrate | `00 Учебник/10 ML Systems/01 Модель как часть системы.md` | `images/svg/*` | audited |
| H-V1-02 | `.../vol1/ml_systems/ml_systems.qmd` | workload and constraints | yes | integrate | `00 Учебник/10 ML Systems/01 Модель как часть системы.md` | `images/svg/*` | audited |
| H-V1-03 | `.../vol1/ml_workflow/ml_workflow.qmd` | lifecycle and constraints | yes | integrate | `00 Учебник/19 Deployment, Reliability и MLOps/01 ML workflow.md` | `images/svg/*` | audited |
| H-V1-04 | `.../vol1/data_engineering/data_engineering.qmd` | quality, debt, pipelines | yes | integrate | `00 Учебник/11 Pre-training и Scaling/41 Сбор, очистка и смеси данных.md` | `images/svg/*` | audited |
| H-V1-05 | `.../vol1/nn_computation/nn_computation.qmd` | compute and memory | yes | cross-link | `00 Учебник/10 ML Systems/04 Арифметика Transformer и MoE.md` | `images/svg/*` | audited |
| H-V1-06 | `.../vol1/nn_architectures/nn_architectures.qmd` | architecture trade-offs | yes | cross-link | `00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура.md` | `images/svg/*` | audited |
| H-V1-07 | `.../vol1/frameworks/frameworks.qmd` | graphs, autodiff, compilers | yes | integrate | `00 Учебник/10 ML Systems/07 Profiling ML-нагрузки.md` | `images/svg/*` | audited |
| H-V1-08 | `.../vol1/training/training.qmd` | memory, mixed precision, DP | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44a Processes, collectives и DDP.md` | `images/svg/*` | audited |
| H-V1-09 | `.../vol1/data_selection/data_selection.qmd` | selection and curricula | yes | integrate | `00 Учебник/11 Pre-training и Scaling/53 Синтетические данные и учебные программы.md` | `images/svg/*` | audited |
| H-V1-10 | `.../vol1/model_compression/model_compression.qmd` | pruning, distillation | yes | integrate | `00 Учебник/14 Inference и оптимизация/57 Квантизация языковых моделей.md` | `images/svg/*` | audited |
| H-V1-11 | `.../vol1/hw_acceleration/hw_acceleration.qmd` | GPU, memory, roofline | yes | integrate | `00 Учебник/10 ML Systems/02 GPU, CUDA и иерархия памяти.md` | `images/svg/*` | audited |
| H-V1-12 | `.../vol1/benchmarking/benchmarking.qmd` | methodology, percentiles | yes | integrate | `00 Учебник/10 ML Systems/03 Измерение производительности и roofline.md` | `images/svg/*` | audited |
| H-V1-13 | `.../vol1/model_serving/model_serving.qmd` | batching, queues, TTFT/TPOT | yes | integrate | `00 Учебник/14 Inference и оптимизация/58c Queueing и capacity planning.md` | `images/svg/*` | audited |
| H-V1-14 | `.../vol1/ml_ops/ml_ops.qmd` | monitoring and rollout | yes | integrate | `00 Учебник/19 Deployment, Reliability и MLOps/02 MLOps.md` | `images/svg/*` | audited |
| H-V1-15 | `.../vol1/responsible_engr/responsible_engr.qmd` | fairness, carbon, governance | yes | integrate | `00 Учебник/18 Evaluation и методология/60 Responsible systems.md` | `images/svg/*` | audited |
| H-V1-16 | `.../vol1/conclusion/conclusion.qmd` | systems synthesis | no — recap only | cross-link | `00 Учебник/10 ML Systems/01 Модель как часть системы.md` | `images/svg/*` | audited |
| H-V2-01…17 | `book/quarto/contents/vol2/{introduction,compute_infrastructure,network_fabrics,data_storage,distributed_training,collective_communication,fault_tolerance,fleet_orchestration,performance_engineering,inference,edge_intelligence,ops_scale,security_privacy,robust_ai,sustainable_ai,responsible_ai,conclusion}/*.qmd` | infrastructure through responsible AI | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44a…h`, `14 Inference`, `19 Deployment`, `01 Справочник/Security и Robustness` | `images/svg/*` | preview-audited |
| H-L1-00…16 | `labs/vol1/lab_{00_introduction…16_ml_conclusion}.py` | Marimo investigations for Vol I | yes | cross-link | matching H-V1 destination | interactive controls | audited |
| H-L2-01…17 | `labs/vol2/lab_{01_introduction…17_fleet_synthesis}.py` | Marimo node-to-fleet investigations | yes | cross-link | matching H-V2 destination | interactive controls | preview-audited |
| H-TT-01…08 | `tinytorch/src/{01_tensor…08_training}/` | progressive foundations | yes | cross-link | `06 Практика/` foundational exercises | code; license check | audited |
| H-TT-09…13 | `tinytorch/src/{09_convolutions…13_transformers}/` | model implementation | no — existing Bookvar coverage | source-only | chapters 13–40 | code; license check | audited |
| H-TT-14…20 | `tinytorch/src/{14_profiling…20_capstone}/` | profiling, quantization, KV, benchmark | yes | integrate | `06 Практика/13 Оптимизировать Transformer step.md` | code; license check | audited |
| H-SIM-00…04 | `mlsysim/docs/tutorials/{00_hello_roofline…02_two_phases}.qmd` | roofline, memory, phases | yes | integrate | `00 Учебник/10 ML Systems/03 Измерение производительности и roofline.md` | calculator outputs | audited |
| H-SIM-05…10 | `mlsysim/docs/tutorials/{03_kv_cache…10_gpu_vs_wafer}.qmd` | KV, quantization, scale, cost | yes | integrate | `00 Учебник/14 Inference и оптимизация/58c Queueing и capacity planning.md` | calculator outputs | audited |
| H-SIM-11…17 | `mlsysim/docs/tutorials/{11_training_memory_capacity_moe…distributed}.qmd` | MoE/DSE/distributed design | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44f Expert и hybrid parallelism.md` | calculator outputs | audited |
| H-SL1-00…16 | `slides/vol1/{00_course_overview…16_conclusion}/` | each deck and its figure directory | yes | integrate | matching H-V1 destination | original SVGs; credit check | audited |
| H-SL2-00…17 | `slides/vol2/{00_course_overview…17_conclusion}/` | each deck and its figure directory | yes | integrate | matching H-V2 destination | original SVGs; credit check | preview-audited |
| H-F1-01…16 | `book/quarto/contents/vol1/*/images/{svg,png,webp,jpg,jpeg}/` | native figure directories | yes | integrate | matching H-V1 destination | original SVG preferred | audited |
| H-F2-01…17 | `book/quarto/contents/vol2/*/images/{svg,png,webp,jpg,jpeg}/` | native figure directories | yes | integrate | matching H-V2 destination | original SVG preferred | preview-audited |

Inclusive ID ranges expand to one row per named source unit (for example, `H-L1-00` through `H-L1-16`); no item is implicitly selected for copying.
