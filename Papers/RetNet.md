---
title: "Retentive Network: A Successor to Transformer for Large Language Models"
url: https://arxiv.org/abs/2307.08621
authors: "Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, Furu Wei"
year: 2023
date_reviewed: 2026-04-07
type: paper-review
category: paper
raw: "[[02 Areas/ML & DL/raw/papers/retnet/source]]"
concepts:
  - "[[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/RNN/RNN|RNN]]"
  - "[[02 Areas/ML & DL/Concepts/Architectures/Decoder-only|Decoder-only]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Scaling Laws|Scaling Laws]]"
---

# Retentive Network: A Successor to Transformer for Large Language Models

**Authors:** Yutao Sun, Li Dong, Shaohan Huang et al. (Microsoft Research, Tsinghua University)
**Published:** 2023 (arXiv:2307.08621v4, Aug 2023)
**URL:** https://arxiv.org/abs/2307.08621

## TL;DR

RetNet предлагает **retention mechanism** как замену self-attention, который поддерживает три режима вычислений: parallel (для training), recurrent (для O(1) inference), и chunkwise recurrent (для эффективного long-sequence modeling). На моделях 1.3B--6.7B RetNet достигает comparable perplexity с Transformer, при этом на 6.7B модели при длине 8k: 8.4x быстрее decoding, 70% экономии памяти, и latency, нечувствительная к batch size.

## Problem

Авторы формулируют "impossible triangle": одновременно нельзя получить (1) training parallelism, (2) low-cost O(1) inference, (3) good performance. Transformer решает (1) и (3), но не (2). RNN решает (2), но не (1). Linear Transformers жертвуют (3). S4/Hyena решают частично. RetNet претендует на все три.

## Method

### Retention Mechanism

Выводится математически из рекуррентного отображения `v(n) -> o(n)`:
- Состояние: `s_n = A * s_{n-1} + K_n^T * v_n`
- Выход: `o_n = Q_n * s_n`

Матрица A диагонализуется как `Lambda * (gamma * e^{i*theta})^{n-m} * Lambda^{-1}`, где gamma -- scalar decay rate, theta -- positional angle (аналог xPos/RoPE).

### Три представления

1. **Parallel** (training): `Retention(X) = (QK^T * D) V`, где D -- causal mask с exponential decay `D_{nm} = gamma^{n-m}` для n >= m, 0 иначе. Похоже на attention, но без softmax.

2. **Recurrent** (inference): `S_n = gamma * S_{n-1} + K_n^T * V_n`, `Retention(X_n) = Q_n * S_n`. O(1) complexity per step, нет KV cache.

3. **Chunkwise recurrent** (long-sequence training): внутри chunk -- parallel representation, между chunks -- recurrent. Линейная memory complexity.

### Multi-Scale Retention (MSR)

- h = d_model / d голов, каждая с разным gamma decay rate: `gamma = 1 - 2^{-5-arange(0,h)}`.
- GroupNorm вместо LayerNorm (разные головы имеют разную variance).
- Swish gate для нелинейности: `MSR(X) = (swish(X * W_G) * Y) * W_O`.

### Архитектура

Стандартный stack L блоков с residual connections и pre-LayerNorm:
- `Y^l = MSR(LN(X^l)) + X^l`
- `X^{l+1} = FFN(LN(Y^l)) + Y^l`
- FFN: `gelu(X * W1) * W2`

Parameter allocation отличается от Transformer: 8d^2 в retention (W_Q, W_K в R^{d*d}, W_G, W_V в R^{d*2d}, W_O в R^{2d*d}), FFN intermediate dimension = 2d (вместо 4d). Head dimension = 256 (queries/keys), 512 (values).

## Key Results

### Language Modeling (1.3B, 2.7B, 6.7B на 100B токенов)

- Perplexity comparable с Transformer. RetNet начинает превосходить Transformer при размере > 2B.

### Zero/Few-Shot (6.7B)

| | HS | BoolQ | COPA | PIQA | Winograd | Winogrande | SC | Avg |
|---|---|---|---|---|---|---|---|---|
| Transformer 0-shot | 55.9 | 62.0 | 69.0 | 74.6 | 69.5 | 56.5 | 75.0 | 66.07 |
| RetNet 0-shot | 60.7 | 62.2 | 77.0 | 75.4 | 77.2 | 58.1 | 76.0 | **69.51** |
| Transformer 4-shot | 55.8 | 58.7 | 71.0 | 75.0 | 71.9 | 57.3 | 75.4 | 66.44 |
| RetNet 4-shot | 60.5 | 60.1 | 78.0 | 76.0 | 77.9 | 59.9 | 75.9 | **69.76** |

### Training cost (sequence length 8192)

- RetNet vs Transformer+FlashAttention: сопоставимая скорость и меньше памяти.
- RetNet без kernel fusion уже конкурентоспособен с highly-optimized FlashAttention.

### Inference cost (6.7B, A100-80GB)

- **Memory**: RetNet -- почти constant (~3% overhead сверх model weights), Transformer растёт линейно с длиной.
- **Throughput**: RetNet -- length-invariant, Transformer падает с длиной.
- **Latency**: RetNet нечувствителен к batch size и input length; Transformer latency растёт с обоими.

### Сравнение с вариантами (200M, 16 layers)

RetNet perplexity = 26.05 (in-domain) vs RWKV 30.92, H3 29.97, Hyena 32.08, Linear Transformer 40.24.

### Ablations

- Swish gate: убирает +1.79 perplexity
- GroupNorm: убирает +1.49
- gamma decay: убирает +1.81
- Multi-scale decay: убирает +0.97
- Reduced head dim (256->64): +1.63

## My notes

