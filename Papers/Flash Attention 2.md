---
title: "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
url: https://arxiv.org/abs/2307.08691
authors: [Tri Dao]
year: 2023
date_reviewed: 2026-04-07
type: source-note
status: legacy
category: paper
tags:
  - attention
  - efficiency
  - GPU
  - systems
Organization: Princeton, Stanford
concepts:
  - "[[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]]"
  - "[[02 Areas/ML & DL/Concepts/Architectures/Attention|Attention]]"
raw: "[[02 Areas/ML & DL/raw/papers/flash-attention-2/paper.txt]]"
---

# FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning

**Authors:** Tri Dao (Princeton, Stanford)
**Published:** 2023 (arXiv:2307.08691)
**URL:** https://arxiv.org/abs/2307.08691

## TL;DR

**FlashAttention-2** — оптимизация FlashAttention с фокусом на лучшую параллелизацию и распределение работы между GPU thread blocks и warps. Три ключевых улучшения: (1) снижение non-matmul FLOPs, (2) параллелизация по sequence length dimension, (3) устранение "split-K" схемы между warps. Результат: **2x ускорение** vs FlashAttention, достижение 50-73% от теоретического максимума TFLOPs/s на A100 (до 230 TFLOPs/s forward, 225 TFLOPs/s end-to-end training GPT).

## Problem

FlashAttention (v1) достигает только 25-40% от теоретического максимума FLOPs/s устройства, тогда как оптимизированный GEMM — 80-90%. Через профилирование выявлены причины:

1. **Много non-matmul FLOPs**: на A100 matmul throughput (Tensor Cores) = 312 TFLOPs/s, non-matmul FP32 = 19.5 TFLOPs/s. Каждый non-matmul FLOP в 16x дороже matmul FLOP.
2. **Низкая occupancy**: FlashAttention параллелизует только по batch * num_heads. При длинных последовательностях (маленький batch) — SMs простаивают.
3. **Лишние shared memory reads/writes**: "split-K" схема в forward pass — warps делят K/V, потом синхронизируются через shared memory для агрегации результата.

## Method

### 1. Снижение non-matmul FLOPs (Section 3.1)

Два изменения в online softmax trick:

**a) Отложенное масштабирование:** вместо нормализации O на каждом шаге через diag(l)^{-1}, поддерживается "un-scaled" версия O_tilde. Нормализация применяется только один раз в конце цикла:
```
O_tilde^(j) = diag(e^{m^(j-1) - m^(j)}) * O_tilde^(j-1) + e^{S^(j) - m^(j)} * V^(j)
O = diag(l^(last))^{-1} * O_tilde^(last)
```

**b) Хранение logsumexp вместо m и l:** для backward pass достаточно хранить L = m + log(l) — один скаляр вместо двух.

### 2. Параллелизация по sequence length (Section 3.2)

**Forward pass:** внешний цикл идет по row blocks Q (не по column blocks K/V как в v1). Row blocks независимы — каждый назначается отдельному thread block. Параллелизация по batch * num_heads * T_r (количество row blocks).

**Backward pass:** параллелизация по column blocks K/V. Обновление dQ между column blocks — через atomic adds.

Идея swap outer/inner loops и параллелизации по seq length предложена Phil Tillet (Triton implementation).

### 3. Warp partitioning без split-K (Section 3.3)

**FlashAttention (v1):** K и V разделяются между 4 warps, Q доступна всем. Warps вычисляют slice QK^T, потом нужна синхронизация через shared memory для агрегации V slices.

**FlashAttention-2:** Q разделяется между 4 warps, K и V доступны всем. Каждый warp вычисляет свой slice QK^T и сразу умножает на общее V — нет нужды в межwarp коммуникации. Результат: значительное снижение shared memory reads/writes.

### Causal masking оптимизация

- Блоки, где все column indices > row indices — пропускаются целиком (~50% блоков для длинных последовательностей -> 1.7-1.8x speedup)
- Causal mask применяется только к 1 блоку на строку (при квадратных блоках)

### Multi-query / Grouped-query attention

Поддержка MQA/GQA без дублирования K/V heads — через манипуляцию индексов. В backward pass — суммирование градиентов dK, dV через неявно дублированные heads.

## Key Results

### Attention Benchmarks (A100 80GB)

| Setting | FlashAttention-2 vs FlashAttention | vs PyTorch |
|---------|-----------------------------------|-----------|
| Fwd + Bwd, head dim 64, no mask | **1.7-2.0x** | **3-10x** |
| Fwd + Bwd, head dim 128, no mask | **1.7-2.0x** | **3-10x** |
| Fwd + Bwd, causal, head dim 64 | **1.7-2.0x** | **5-15x** |
| Forward only, head dim 128, no mask | до **230 TFLOPs/s** (73% max) | - |

FlashAttention-2 также 1.3-2.5x быстрее FlashAttention в Triton.

### End-to-end GPT Training (8xA100 80GB)

| Model | Seq len | FlashAttention-2 | vs FlashAttention | vs baseline |
|-------|---------|-------------------|-------------------|-------------|
| GPT 1.3B | 2K | 189 TFLOPs/s | 1.08x | 2.20x |
| GPT 1.3B | 8K | 197 TFLOPs/s | 1.22x | 2.72x |
| GPT 2.7B | 2K | 207 TFLOPs/s | 1.07x | 2.34x |
| GPT 2.7B | 8K | **225 TFLOPs/s** | **1.29x** | **2.83x** |

72% model FLOPs utilization (MFU) — близко к максимуму для training.

### H100

На H100 без специальных оптимизаций: до 335 TFLOPs/s. Ожидается 1.5-2x дополнительное ускорение с TMA и 4th-gen Tensor Cores.

## My notes

- FlashAttention-2 — это про GPU systems engineering, не про ML алгоритмы. Алгоритмический вычисление то же самое (exact attention), отличия в том, как работа распределяется между thread blocks и warps.
- Ключевой insight: non-matmul FLOPs в 16x дороже на Tensor Core GPU. Оптимизация должна максимизировать долю времени на matmul.
- Swap inner/outer loops + параллелизация по seq dim — простая идея с большим эффектом. Автор кредитует Phil Tillet (OpenAI/Triton).
- На практике FlashAttention-2 используется повсеместно: PyTorch, Hugging Face Transformers, vLLM, TGI. С 2023 года это default в большинстве фреймворков.
- 72% MFU — очень высокий показатель. Для сравнения, PaLM достигал ~46-57% MFU. Это значит, что attention перестает быть bottleneck, и ограничения смещаются к другим компонентам модели (MLP, all-reduce в distributed training).
