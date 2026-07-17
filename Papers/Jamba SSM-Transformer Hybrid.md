---
title: "Jamba: A Hybrid Transformer-Mamba Language Model"
url: https://arxiv.org/abs/2403.19887
authors: "AI21 Labs"
year: 2024
date_reviewed: 2026-04-08
type: source-note
status: legacy
category: paper
raw: "[[02 Areas/ML & DL/raw/papers/jamba/source]]"
concepts: [Mamba, State Space Models, Mixture of Experts, Transformer, KV Cache, Long Context, Hybrid Architecture]
---
# Jamba: A Hybrid Transformer-Mamba Language Model

## TL;DR

Jamba -- первая production-grade **hybrid Transformer-Mamba** модель с MoE. Interleaving Transformer и Mamba layers (1:7 ratio) даёт: 8x меньше KV cache (4GB vs 32GB на 256K context), 3x throughput vs Mixtral, поддержка 256K context -- при сопоставимом quality. 52B total / 12B active parameters, помещается в одну 80GB GPU.

## Problem

Transformer -- dominant architecture, но две проблемы при длинных контекстах: (1) KV cache растёт линейно, ограничивая context length, (2) каждый generated token требует computation на всём контексте → low throughput. SSM (Mamba) решают оба, но отстают по quality. Можно ли совместить benefits обоих?

## Method

### Архитектура (Jamba Block)
- **Hybrid layers**: чередование Attention и Mamba layers в ratio `a:m`
- **MoE**: на каждом втором MLP layer, 16 experts, top-2 routing
- Конфигурация: 4 Jamba blocks, каждый:
  - `l = 8` layers
  - `a:m = 1:7` (1 attention на 7 Mamba layers)
  - `e = 2` (MoE every other layer)
  - `n = 16` experts, `K = 2` top experts

### Ключевые design decisions
- **1:7 ratio** (Attention:Mamba): chosen через ablations как most compute-efficient среди best-performing variants
- **RMSNorm в Mamba layers**: critical для stability при large-scale training. Без нормализации training diverges.
- **No positional embeddings**: с Mamba layers позиционная информация не нужна (implicit через recurrence)
- GQA, SwiGLU, 64K vocabulary, BPE tokenizer

### Memory Efficiency
| Model | Total params | Active params | KV cache (256K, 16bit) |
|---|---|---|---|
| LLaMA-2 7B | 6.7B | 6.7B | 128GB |
| Mistral 7B | 7.2B | 7.2B | 32GB |
| Mixtral 8x7B | 46.7B | 12.9B | 32GB |
| **Jamba** | **52B** | **12B** | **4GB** |

- 8x reduction в KV cache vs vanilla Transformer
- Fits single 80GB GPU с context до 140K tokens (vs 20K для LLaMA-2 70B)

### Training
- NVIDIA H100 GPUs, in-house framework (FSDP, tensor/sequence/expert parallelism)
- In-house dataset: web, books, code (last update March 2024)
- Successfully trained on context lengths up to 1M tokens

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/jamba/fig1.png]]

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/jamba/fig3.png]]

## Key Results

### Standard Benchmarks
| Benchmark | Jamba | Mixtral 8x7B | LLaMA-2 70B |
|---|---|---|---|
| MMLU (5-shot) | 67.4 | 70.6 | 69.8 |
| HellaSwag (10-shot) | 87.1 | 86.7 | 85.3 |
| WinoGrande (5-shot) | 82.5 | 81.2 | 80.2 |
| GSM8K (3-shot CoT) | 59.9 | 60.4 | 55.3 |
| HumanEval | 29.3 | 34.8 | 29.9 |

### Long Context
- Needle-in-a-haystack: excellent performance до 256K tokens
- Few-shot classification (128K context): Jamba >= Mixtral на Trec-Fine, Banking77
- Long-context QA: Jamba outperforms Mixtral on average (0.44 vs 0.43 F1)

### Throughput
- Single GPU, batch=16: Jamba 3x throughput vs Mixtral (Mixtral doesn't fit)
- 4 GPUs, context 128K: Jamba 3x throughput vs Mixtral

## My notes

- **Architectural proof-of-concept**: hybrid Attention-SSM works at scale. 1:7 ratio -- aggressive, но ablations показывают: pure Mamba struggles с in-context learning, даже 1 attention layer на 7 Mamba решает проблему.
- KV cache 4GB vs 32GB на 256K context -- это game-changer для deployment. Long-context inference feasible на consumer hardware.
- Quality несколько ниже Mixtral на MMLU (67.4 vs 70.6) и HumanEval (29.3 vs 34.8). Это цена за architectural novelty -- оптимизация ещё не на уровне mature Transformer stack.
- RMSNorm в Mamba layers -- crucial insight для community. Без этого large-scale Mamba training нестабилен.
- No positional embeddings -- Mamba's recurrence inherently captures position. Это означает, что hybrid model полагается на attention layers для explicit positional reasoning, а Mamba -- для sequential patterns.
- Training на 1M tokens, release на 256K -- gap указывает на degradation при extreme lengths, но 256K -- уже best-in-class для open models.
- MoE on top of hybrid -- three-way combination (Attention + Mamba + MoE). Flexible design space с 5 degrees of freedom (l, a:m, e, n, K). Много room для optimization.
- Base model only (no instruction tuning) -- честно, но ограничивает practical use. Community fine-tunes нужны для fair comparison с Mixtral-Instruct.
