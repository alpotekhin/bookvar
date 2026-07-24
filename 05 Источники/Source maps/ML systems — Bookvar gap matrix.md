---
title: ML systems — Bookvar gap matrix
type: source-note
status: editorial
last_updated: 2026-07-24
---

# ML systems — Bookvar gap matrix

Эта таблица описывает текущее состояние, а не первоначальные намерения.
`integrated` означает: каноническая страница существует, содержит прямые ссылки
на полные локальные оригиналы и включена в проверяемую карту назначения.

| Bookvar concept | Canonical destination | Best source IDs | State |
|---|---|---|---|
| System framing | `00 Учебник/10 ML Systems/01 Модель как часть системы.md` | H-V1-01, H-V1-02, H-SL1-01 | integrated |
| GPU and CUDA | `00 Учебник/10 ML Systems/02 GPU, CUDA и иерархия памяти.md`; `06 Практика/06 Измерить CUDA без самообмана.md` | H-V1-11, H-V2-02, E-01-L, E-01-S | integrated |
| Measurement and roofline | `00 Учебник/10 ML Systems/03 Измерение производительности и roofline.md`; chapter 55a | H-V1-12, H-SIM-01, H-SIM-02, E-01-S | integrated |
| Transformer and MoE arithmetic | `00 Учебник/10 ML Systems/04 Арифметика Transformer и MoE.md` | H-V1-05, E-06-L | integrated |
| Numerical formats | `00 Учебник/10 ML Systems/05 Численные форматы и mixed precision.md` | H-V1-08, E-02-L, E-02-A1 | integrated |
| Input pipeline | `00 Учебник/10 ML Systems/06 Data pipeline, padding и packing.md`; `06 Практика/08 Ускорить data pipeline.md` | H-V1-04, E-02-S, E-02-H1, E-02-H2 | integrated |
| Profiling | `00 Учебник/10 ML Systems/07 Profiling ML-нагрузки.md` | H-V1-07, E-02-H3, E-06-S | integrated |
| Data parallelism and collectives | chapter 44a; practice 09 | H-V2-06, E-03-L, E-03-S, E-03-H | integrated |
| Checkpointing and offload | chapter 44b; practice 10 | E-04-L, E-04-S1 | integrated |
| Tensor, sequence, and pipeline parallelism | chapters 44c–44d; practice 11 | H-V2-05, H-SL2-05, E-04-L, E-04-S2 | integrated |
| ZeRO and FSDP | chapter 44e; practice 12 | E-05-L, E-05-S, E-05-H | integrated |
| Expert and hybrid parallelism | chapter 44f and the MoE architecture chapter | H-V2-05, H-SIM-14, E-06-L | integrated |
| Networks, storage, and checkpoints | chapter 44g | H-V2-03, H-V2-04, H-F2-03, H-F2-04 | integrated |
| Fault tolerance and orchestration | chapter 44h | H-V2-07, H-V2-08, H-SL2-07, H-SL2-08 | integrated |
| KV cache, scheduling, and mini-engine | chapters 55–55c; practice 14 | H-V2-10, E-08-L, E-08-S, E-08-H | integrated |
| Kernels and compilation | chapter 56; practice 13 | H-V2-09, E-06-S, E-06-A | integrated |
| Quantization and KV compression | chapters 57–57a; practice 15 | H-V1-10, H-SIM-06, E-09-L, E-09-S | integrated |
| Speculative decoding | chapter 58; practice 16 | E-09-L, E-09-H | integrated |
| Queueing, SLO, and capacity | chapters 58b–58c | H-V1-13, H-V2-10, H-SIM-10, E-08-S | integrated |
| Workflow, deployment, and MLOps | deployment chapters 01–03; practice 17 | H-V1-03, H-V1-14, H-V2-11, H-V2-12, E-07 | integrated |
| Security, robustness, responsibility, and sustainability | chapter 59a; two Security and Robustness reference pages | H-V1-15, H-V2-13, H-V2-14, H-V2-15, H-V2-16 | integrated |

Автоматическая проверка выполняется командой
`pnpm --dir publishing check:source-coverage`. Она не оценивает качество прозы,
но не позволяет снова назвать destination готовым, если файл отсутствует или
не ведёт к соответствующему полному источнику.
