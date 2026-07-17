---
title: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
url: https://arxiv.org/abs/2312.00752
authors: "Albert Gu, Tri Dao"
year: 2023
date_reviewed: 2026-04-07
type: paper-review
category: paper
raw: "[[02 Areas/ML & DL/raw/papers/mamba/source]]"
concepts:
  - "[[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/RNN|RNN]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Scaling Laws|Scaling Laws]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]]"
---

# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

**Authors:** Albert Gu (Carnegie Mellon University), Tri Dao (Princeton University)
**Published:** 2023
**URL:** https://arxiv.org/abs/2312.00752

## TL;DR

Mamba -- первая линейная по времени sequence model, которая достигает качества Transformer на language modeling. Ключевая идея: selective state space models (S6) -- параметры SSM (Delta, B, C) делаются зависимыми от входа, что позволяет модели селективно запоминать или забывать информацию. Это ломает time-invariance (нельзя использовать свёртки), но авторы решают проблему hardware-aware parallel scan алгоритмом. Mamba-3B превосходит Transformer-3B и соответствует Transformer-6B+ на zero-shot evaluations. Inference throughput 5x выше Transformer при линейном scaling по длине последовательности.

## Problem

Transformer -- доминирующая архитектура, но quadratic scaling attention по длине последовательности. Subquadratic альтернативы (linear attention, SSMs, Hyena, RWKV, RetNet) не достигали качества attention на языковых задачах. Причина: Linear Time-Invariant (LTI) модели не могут выполнять content-based reasoning -- их параметры (A, B, C) фиксированы для всех timesteps, и они не могут селективно фильтровать или запоминать информацию в зависимости от содержания входа.

Два ключевых synthetic tasks демонстрируют проблему:
1. **Selective Copying** -- LTI модели (свёртки) решают vanilla Copying (фиксированные позиции), но не Selective Copying (рандомные позиции между значимыми токенами)
2. **Induction Heads** -- ассоциативный recall ("Harry Potter" -> при следующем "Harry" предсказать "Potter"), требует content-aware reasoning

## Method

### Background: Structured State Space Models (S4)

SSM определяется параметрами (Delta, A, B, C):
- Continuous: h'(t) = Ah(t) + Bx(t), y(t) = Ch(t)
- Discretized: h_t = A_bar * h_{t-1} + B_bar * x_t, y_t = C * h_t
- Discretization (ZOH): A_bar = exp(Delta*A), B_bar = (Delta*A)^{-1}(exp(Delta*A) - I) * Delta*B

В LTI режиме параметры фиксированы -- можно вычислять как recurrence (inference) или global convolution (training, parallel).

### Selection Mechanism (Algorithm 2 -- S6)

Ключевое изменение: B, C, Delta становятся функциями входа x:
```
B : (B, L, N) <- s_B(x) = Linear_N(x)
C : (B, L, N) <- s_C(x) = Linear_N(x)
Delta : (B, L, D) <- softplus(Parameter + s_Delta(x))
```

где `s_Delta(x) = Broadcast_D(Linear_1(x))` -- проекция в 1 измерение, затем broadcast (low-rank).

Это добавляет length dimension L к параметрам -- модель становится time-varying. Теряется эквивалентность со свёрткой, можно только recurrence (scan).

### Hardware-aware Algorithm (Section 3.3)

Проблема: наивная рекуррентная реализация требует O(BLDN) FLOPs и материализацию hidden state (B,L,D,N) -- огромный memory footprint.

Решение (три классических техники):
1. **Kernel fusion:** загрузка параметров (Delta, A, B, C) из HBM в SRAM, discretization + recurrence в SRAM, запись результата (B,L,D) обратно в HBM
2. **Parallel scan:** несмотря на sequential nature recurrence, можно параллелизовать с work-efficient scan (Blelloch 1990)
3. **Recomputation:** промежуточные states не сохраняются для backprop, а перевычисляются при backward pass (аналог FlashAttention)

Результат: memory requirements как у оптимизированного Transformer с FlashAttention. До 3x быстрее предыдущих SSM на A100 GPU.

### Mamba Architecture (Section 3.4, Figure 3)

Объединение H3 block + MLP block в один гомогенный блок:
1. Input projection: x -> два branch через Linear (expansion factor E=2)
2. Branch 1: Conv1D -> SiLU activation -> Selective SSM
3. Branch 2: SiLU activation (gate)
4. Elementwise multiply branches -> output projection

Без attention, без отдельного MLP block. Стек одинаковых Mamba blocks с RMSNorm + residual connections. Для matching параметров Transformer (12D^2 per layer) используются 2 Mamba block стека с E=2.

### Connection to RNN Gating (Theorem 1)

При N=1, A=-1, B=1, selective SSM редуцируется к gated RNN:
```
g_t = sigma(Linear(x_t))
h_t = (1 - g_t) * h_{t-1} + g_t * x_t
```

Большой Delta -> reset state, focus on current input (select); маленький Delta -> persist state, ignore input.

## Key Results

### Synthetic Tasks

**Selective Copying (Table 1):** S6 (selection mechanism) решает задачу с 97-99.8% accuracy, в то время как S4 -- 18.3%, Hyena -- 28-30%.

**Induction Heads (Table 2):** Mamba решает perfect и **экстраполирует до 2^20 = 1M** токенов (обучение на 256). Ни один другой метод не экстраполирует дальше 2x.

### Language Modeling Scaling Laws (Figure 4)

На Pile dataset, модели 125M--1.3B параметров: **Mamba -- первая attention-free модель, matching Transformer++ (LLaMA recipe)** по perplexity, особенно на длинных последовательностях. Превосходит Hyena, RWKV, RetNet, vanilla Transformer.

### Zero-shot Evaluations (Table 3)

Mamba-3B vs baselines (все 300B tokens training, NeoX tokenizer):

| Model | LAMBADA acc | HellaSwag | PIQA | Arc-E | WinoGrande | Average |
|-------|-------------|-----------|------|-------|------------|---------|
| Pythia-1.4B | 61.7 | 52.1 | 71.0 | 60.5 | 57.2 | 55.2 |
| RWKV-1.5B | 56.4 | 52.5 | 72.4 | 60.5 | 54.6 | 54.3 |
| **Mamba-1.4B** | **64.9** | **59.1** | **74.2** | **65.5** | **61.5** | **59.7** |
| Pythia-2.8B | 64.7 | 59.3 | 74.0 | 64.1 | 59.7 | 59.1 |
| RWKV-3B | 63.9 | 59.6 | 73.7 | 67.8 | 59.6 | 59.6 |
| **Mamba-2.8B** | **69.2** | **66.1** | **75.2** | **69.7** | **63.5** | **63.3** |
| GPT-J-6B | 68.3 | 66.3 | 75.4 | 67.0 | 64.1 | 63.0 |

Mamba-1.4B best-in-class в своём размере. Mamba-2.8B превосходит все 3B модели и соответствует GPT-J-6B (2x размер).

### DNA Modeling (Figures 5, 6)

HG38 human genome dataset: Mamba масштабируется лучше HyenaDNA и Transformer++ по model size (3-4x parameter efficiency). Perplexity улучшается монотонно с context length до 1M (в отличие от HyenaDNA, которая ухудшается). Great Apes DNA classification: Mamba достигает 90%+ accuracy при длине 1M (random chance ~20%).

### Audio (Figure 7)

YouTubeMix piano: превосходит SaShiMi (prior SOTA на S4), perplexity улучшается до minute-long sequences (1M samples).

### Speed and Memory (Section 4.5)

- **Inference throughput:** 5x выше Transformer при sequence length 2048+ (generation throughput ~500 tokens/s на A100 для Mamba-2.8B vs ~100 для Transformer)
- **Training:** линейный scaling по sequence length (vs quadratic для attention)
- **Memory:** O(BLDN) -> O(BLD) через selective scan (state не материализуется)

## My notes

