---
title: "Attention Is All You Need"
url: https://arxiv.org/abs/1706.03762
authors: [Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin]
year: 2017
date_reviewed: 2026-04-07
type: paper-review
category: paper
tags:
  - arch
  - LLM
  - Transformer
  - attention
Organization: Google
concepts:
  - "[[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/Self-Attention|Self-Attention]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/Positional Encoding|Positional Encoding]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/Feed-Forward Network|Feed-Forward Network]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/Layer Normalization|Layer Normalization]]"
raw: "[[02 Areas/ML & DL/raw/papers/attention-is-all-you-need/paper.txt]]"
---

# Attention Is All You Need

**Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (Google Brain / Google Research)
**Published:** 2017 (NeurIPS 2017, arXiv:1706.03762)
**URL:** https://arxiv.org/abs/1706.03762

## TL;DR

Авторы предложили **Transformer** — первую архитектуру трансляции последовательностей, основанную **исключительно на механизме внимания**, без RNN или свёрток. Transformer значительно более параллелизуем, обучается быстрее и достигает SOTA на WMT 2014 EN-DE (28.4 BLEU) и EN-FR (41.8 BLEU) за долю вычислительной стоимости предыдущих моделей. Стала фундаментом для BERT, GPT-3 и практически всех современных LLM.

## Problem

Доминирующие модели sequence transduction (MT, языковое моделирование) — это RNN (LSTM, GRU) с механизмом внимания. Два ключевых недостатка:
1. **Последовательность вычислений:** скрытое состояние h_t зависит от h_{t-1} → невозможна параллелизация в рамках одного примера → узкое место при длинных последовательностях
2. **Ограниченные дальние зависимости:** путь сигнала O(n) через рекуррентные слои → градиенты затухают на длинных последовательностях

Параллельные подходы (ByteNet, ConvS2S) сокращают последовательность до O(1) / O(log n), но число операций для связи позиций растёт с расстоянием.

## Method

### Архитектура Transformer (§3)

Encoder-Decoder структура (N=6 identical layers в каждом):

**Encoder layer:**
1. Multi-Head Self-Attention (все позиции видят друг друга)
2. Position-wise Feed-Forward Network
(каждый подслой: Add & Norm = Residual + LayerNorm)

**Decoder layer:**
1. Masked Multi-Head Self-Attention (causal mask — позиция i видит только ≤ i)
2. Multi-Head Cross-Attention (Q из decoder, K,V из encoder output)
3. Position-wise Feed-Forward Network

**Гиперпараметры base model:** d_model=512, d_ff=2048, h=8 heads, N=6 layers, d_k=d_v=64, P_drop=0.1, ε_ls=0.1

### Scaled Dot-Product Attention (§3.2.1)

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) V
```

Деление на √d_k: при больших d_k dot-product растёт, softmax попадает в насыщение с малыми градиентами → scaling исправляет это.

### Multi-Head Attention (§3.2.2)

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) Wᴼ
headi = Attention(Q Wᵢᴼ, K Wᵢᴷ, V Wᵢᵛ)
```

h=8 параллельных heads, d_k=d_v=64. Общая вычислительная стоимость ≈ single-head attention с полной размерностью.

Три применения в Transformer:
- **Encoder self-attention:** каждая позиция → все позиции
- **Decoder masked self-attention:** позиция i → позиции ≤ i (causal)
- **Cross-attention:** decoder queries → encoder keys/values

### Position-wise Feed-Forward (§3.3)

```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

d_model=512 → 2048 → 512. ReLU activation. Параметры независимы между позициями, но одинаковы внутри одного слоя.

### Positional Encoding (§3.5)

Синусоидальные PE (не обучаемые):
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Гипотеза: PE_pos+k линейно выражается через PE_pos → модель может обобщаться на длины больше обучающих. Эксперименты (Table 3 row E) показывают: learned PE ≈ sinusoidal PE по качеству.

### Training (§5)

- **Данные:** WMT 2014 EN-DE: 4.5M пар (BPE, vocab 37K); WMT 2014 EN-FR: 36M пар (word-piece, vocab 32K)
- **Hardware:** 8 NVIDIA P100 GPUs
- **Base model:** 100K steps, ~12 часов; **Big model:** 300K steps, 3.5 дня
- **Optimizer:** Adam (β₁=0.9, β₂=0.98, ε=10⁻⁹) с warmup scheduler: lr = d_model^(-0.5) · min(step^(-0.5), step · warmup_steps^(-1.5)), warmup_steps=4000
- **Regularization:** Dropout 0.1 + Label Smoothing ε=0.1

## Key Results

### Machine Translation (Table 2)

| Model | EN-DE BLEU | EN-FR BLEU | FLOPs |
|-------|-----------|-----------|-------|
| ConvS2S Ensemble | 26.36 | 41.29 | 7.7·10¹⁹ |
| **Transformer (base)** | **27.3** | **38.1** | **3.3·10¹⁸** |
| **Transformer (big)** | **28.4** | **41.8** | **2.3·10¹⁹** |

Transformer (big) превышает лучший предыдущий результат (**включая ансамбли**) на +2.0 BLEU (EN-DE) при примерно той же стоимости обучения.

### English Constituency Parsing (Table 4)

Transformer (4 layers) на WSJ: 91.3 F1 (WSJ only), 92.7 F1 (semi-supervised) — конкурентоспособен с SOTA без task-specific tuning.

## Ablation Studies (Table 3)

| Вариация | PPL (dev) | BLEU (dev) |
|----------|-----------|-----------|
| base | 4.92 | 25.8 |
| h=1 (single head) | 5.29 | 24.9 (-0.9) |
| h=32 (too many) | 5.01 | 25.4 |
| d_k=16 (small) | 5.16 | 25.1 |
| N=2 layers | 6.11 | 23.7 |
| N=8 layers | 4.88 | 25.5 |
| big | 4.33 | 26.4 |
| sinusoidal → learned PE | 4.92 | 25.7 ≈ |

Вывод: h=8 оптимально; уменьшение d_k критично; больше слоёв = лучше; PE вид почти неважен.

### Computational complexity (Table 1)

| Layer Type | Complexity | Sequential Ops | Max Path |
|-----------|-----------|----------------|---------|
| Self-Attention | O(n²·d) | O(1) | O(1) |
| Recurrent | O(n·d²) | O(n) | O(n) |
| Convolutional | O(k·n·d²) | O(1) | O(log_k n) |

Self-Attention быстрее Recurrent когда n < d (обычный случай для предложений в MT).

## My notes

- Transformer стал фундаментом всего современного NLP: BERT (encoder), GPT-2/3 (decoder), T5 (enc-dec). Почти все LLM 2020-2026 — это Transformer-based.
- Ключевой инсайт: параллелизация attention позволяет обучаться на порядки быстрее при том же или лучшем качестве.
- Sinusoidal PE → к 2023 практически заменены на RoPE (LLaMA, Mistral) для лучшего экстраполяции на длинные контексты.
- Multi-head attention → к 2023 эволюция: Grouped Query Attention (Mistral, LLaMA 2) для эффективности inference.
- Warmup + inverse square root schedule — стандартный рецепт до появления cosine schedule.
- Label smoothing ε=0.1 — до сих пор широко применяется для регуляризации при классификации.
