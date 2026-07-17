---
title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
url: https://arxiv.org/abs/2205.14135
authors: [Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Re]
year: 2022
date_reviewed: 2026-04-07
type: paper-review
category: paper
tags:
  - attention
  - efficiency
  - GPU
  - systems
Organization: Stanford, University at Buffalo
concepts:
  - "[[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]]"
  - "[[02 Areas/ML & DL/Concepts/Architectures/Attention|Attention]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]]"
raw: "[[02 Areas/ML & DL/raw/papers/flash-attention/paper.txt]]"
---

# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

**Authors:** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Re (Stanford, University at Buffalo)
**Published:** 2022 (arXiv:2205.14135)
**URL:** https://arxiv.org/abs/2205.14135

## TL;DR

**FlashAttention** — IO-aware алгоритм exact attention, который использует tiling для минимизации обращений к GPU HBM (high bandwidth memory). Вместо материализации N*N матрицы внимания в HBM, FlashAttention блочно вычисляет attention в on-chip SRAM с использованием online softmax. Результат: до 7.6x ускорение attention на GPT-2, 15% ускорение BERT-large vs MLPerf record, 3x ускорение GPT-2 vs HuggingFace, линейное потребление памяти по длине последовательности.

## Problem

Стандартный self-attention квадратичен по длине последовательности N — O(N^2) по времени и памяти. Approximate attention методы (Linformer, Performer, Reformer и др.) снижают FLOPs, но:
- Не дают wall-clock speedup, т.к. не учитывают memory access overhead
- Фокусируются на FLOP reduction, который не коррелирует с реальной скоростью
- На современных GPU compute >> memory bandwidth (A100: 19 TB/s SRAM vs 1.5 TB/s HBM)
- Attention — memory-bound операция: softmax, dropout, masking — все elementwise, bottleneck на чтении/записи HBM

Стандартная реализация attention материализует матрицы S = QK^T и P = softmax(S) размера N*N в HBM, что дает O(Nd + N^2) HBM accesses.

## Method

### IO-aware подход

Ключевая идея: вместо снижения FLOPs — снизить количество обращений к медленной HBM, выполняя вычисления в быстрой on-chip SRAM (19 TB/s, ~20MB на A100).

### Tiling + Online Softmax

1. **Tiling**: Q, K, V разбиваются на блоки размера Br x d и Bc x d
2. **Outer loop**: итерация по блокам K, V — загрузка в SRAM
3. **Inner loop**: итерация по блокам Q — загрузка в SRAM, вычисление блочного attention
4. **Online softmax**: softmax вычисляется инкрементально, без доступа ко всему входу. Для каждого блока поддерживается текущий max m(x) и сумма экспонент l(x), которые обновляются при обработке нового блока:
   - m_new = max(m_old, rowmax(S_ij))
   - l_new = e^{m_old - m_new} * l_old + e^{m_ij - m_new} * l_ij_local

### Recomputation в backward pass

Вместо хранения N*N матрицы P для backward pass, сохраняются только O и статистики softmax (m, l). В backward pass attention пересчитывается из блоков Q, K, V в SRAM. Несмотря на дополнительные FLOPs, это быстрее из-за уменьшения HBM accesses.

### Kernel fusion

Все операции attention (matmul, softmax, mask, dropout, matmul) объединены в один CUDA kernel — загрузка из HBM один раз, результат обратно один раз.

### IO Complexity

- Стандартный attention: Theta(Nd + N^2) HBM accesses
- FlashAttention: Theta(N^2 * d^2 * M^-1), где M — размер SRAM
- Для типичных d=64-128 и M ~ 100KB: FlashAttention делает до 9x меньше HBM accesses
- Доказана нижняя граница: никакой exact attention алгоритм не может асимптотически улучшить IO FlashAttention

### Block-Sparse FlashAttention

Расширение для sparse attention: пропускаются нулевые блоки по маске M, IO complexity уменьшается пропорционально sparsity ratio s. С butterfly sparsity pattern: O(N * sqrt(N)) или O(N * log N).

## Key Results

### Training Speed

| Model | vs Baseline | Speedup |
|-------|------------|---------|
| BERT-large (seq 512) | vs MLPerf 1.1 record | **15% быстрее** (17.4 vs 20.0 min) |
| GPT-2 small (seq 1K) | vs HuggingFace | **3.5x** |
| GPT-2 small (seq 1K) | vs Megatron-LM | **1.7x** |
| GPT-2 medium (seq 1K) | vs HuggingFace | **3.0x** |
| LRA benchmark (seq 1K-4K) | vs standard attention | **2.4x** |

### Longer Context = Better Quality

| Model | Context | Perplexity | Speed |
|-------|---------|-----------|-------|
| GPT-2 small Megatron | 1K | 18.2 | 4.7 days (1.0x) |
| GPT-2 small FlashAttention | 1K | 18.2 | 2.7 days (1.7x) |
| GPT-2 small FlashAttention | 4K | **17.5** | 3.6 days (1.3x) |

GPT-2 с context 4K + FlashAttention быстрее, чем Megatron с context 1K, и при этом на 0.7 лучше по perplexity.

### Long Document Classification

- MIMIC-III: +4.3 F1 (52.8 -> 57.1) при увеличении seq length с 512 до 16K
- ECtHR: +8.5 F1 (72.2 -> 80.7) при увеличении seq length с 512 до 8K

### Path-X / Path-256

Первый Transformer, решающий Path-X (seq 16K): 61.4% accuracy. Block-sparse FlashAttention — первая модель на Path-256 (seq 64K): 63.1%. Все предыдущие модели = random.

### Memory

- Линейная по N (вместо квадратичной)
- До 20x меньше памяти vs стандартный attention
- Работает на seq 64K на одной A100 40GB

## My notes

- FlashAttention стал де-факто стандартом для attention в production. Используется в PyTorch 2.0+ (torch.nn.functional.scaled_dot_product_attention), в xformers, Triton.
- Главный insight: FLOP count != wall-clock time. На современных GPU bottleneck — memory bandwidth, а не compute. Attention — memory-bound, поэтому IO-aware подход дает реальное ускорение, даже при увеличении FLOPs (из-за recomputation).
- Tiling + online softmax — классические приемы, но их комбинация с CUDA kernel fusion и анализом IO complexity — это вклад работы.
- Ограничение: каждая новая вариация attention требует отдельного CUDA kernel. Это инженерно тяжело и не переносимо между GPU архитектурами. FlashAttention-2 и FlashAttention-3 продолжают решать эту проблему.
- Block-sparse FlashAttention — proof of concept, но на практике мало используется: большинство production моделей работают с dense attention или causal masking.
