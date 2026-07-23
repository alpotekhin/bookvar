---
title: ML systems — Bookvar gap matrix
type: source-note
status: editorial
last_updated: 2026-07-23
---

# ML systems — Bookvar gap matrix

| Bookvar concept | Current page | Missing material | Best source IDs | Action |
|---|---|---|---|---|
| System framing | — | DAM, Iron Law, constraints | H-V1-01, H-V1-02 | new chapter 01 |
| CUDA timing | — | async launch, warmup, synchronization | E-01-L/S, H-V1-11 | new chapters 02–03 + lab |
| Transformer/MoE arithmetic | — | FLOPs, memory, all-to-all | E-06-L, H-V1-05 | new chapter 04 |
| Numerical formats | `44 Distributed training и mixed precision.md` | FP8/MXFP8, accumulation, loss scaling | E-02-L, H-V1-08 | new chapter 05; split 44 |
| Input pipeline/profiling | `41 Сбор, очистка и смеси данных.md` | I/O, decode, padding, snapshots | E-02-L/S/H1…3 | new chapters 06–07 |
| DP and collectives | `44 Distributed training и mixed precision.md` | ring/tree, overlap, cost | E-03-L/S/H, H-V2-06 | 44a + lab |
| TP/PP/checkpointing | `44 Distributed training и mixed precision.md` | placement, bubble, offload | E-04-L/S1…2, H-V2-05 | 44b–d |
| ZeRO/FSDP | `44 Distributed training и mixed precision.md` | DeviceMesh, DTensor, DCP | E-05-L/S/H | 44e + lab |
| EP/hybrid, fabric, recovery | `02 Mixture of Experts — routing, capacity и serving.md` | GroupedGEMM, RDMA, Young-Daly | E-06-L, H-V2-03…08 | 44f–h |
| KV-cache/scheduling | `55 KV-cache, пакетирование и PagedAttention.md` | Qwen measurements, mini-engine | E-08-L/S/H/P | worked example + lab |
| Kernels | `56 FlashAttention.md` | compile, fusion, Liger profiles | E-06-S, H-V2-09 | expand 56 |
| Quantization/speculation | `57 Квантизация языковых моделей.md`; `58 Спекулятивное декодирование.md` | W8A8, KV rollback/compression | E-09-L/S/H | 57a; practices 15–16 |
| Queueing and capacity | `58b Benchmarking, SLO и эксплуатация inference.md` | utilization, replica/cost model | H-V1-13, H-V2-10, E-08-S | new 58c |
| Deployment/MLOps | — | health, observability, rollout | E-07-L/P1…7, H-V1-14 | lifecycle chapter + lab |
| Security/robustness/sustainability | — | threats, SDC, energy | H-V2-13…16 | Task 8 focused pages |
