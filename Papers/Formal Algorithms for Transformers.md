---
title: "Formal Algorithms for Transformers"
url: https://arxiv.org/abs/2207.09238
authors: [Mary Phuong, Marcus Hutter]
year: 2022
date_reviewed: 2026-04-06
type: paper-review
category: paper
tags:
  - Review
  - arch
Date: 2022-07-19
Organization: DeepMind
Status: Done
concepts:
  - "[[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/Self-Attention|Self-Attention]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/Positional Encoding|Positional Encoding]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/Feed-Forward Network|Feed-Forward Network]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/Layer Normalization|Layer Normalization]]"
  - "[[02 Areas/ML & DL/Concepts/Architectures/Encoder-Decoder|Encoder-Decoder]]"
  - "[[02 Areas/ML & DL/Concepts/Architectures/Encoder-only|Encoder-only]]"
  - "[[02 Areas/ML & DL/Concepts/Architectures/Decoder-only|Decoder-only]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Masked Language Modeling|MLM]]"
raw: "[[02 Areas/ML & DL/raw/papers/formal-algorithms-for-transformers/paper.txt]]"
---

# Formal Algorithms for Transformers

**Authors:** Mary Phuong, Marcus Hutter (DeepMind)
**Published:** 2022 (arXiv:2207.09238v1, Jul 2022)
**URL:** https://arxiv.org/abs/2207.09238

## TL;DR

Технический отчёт DeepMind, цель которого — дать **математически точное, самодостаточное описание** трансформерных архитектур в виде формальных алгоритмов (псевдокода). Не статья с результатами, а **справочник**: покрывает токенизацию, компоненты, три архитектуры (Encoder-Decoder, Encoder-only/BERT, Decoder-only/GPT), обучение и инференс. Весь ключевой функционал — ~50 строк псевдокода. Целевая аудитория: теоретики, нуждающиеся в точных формулировках, и практики, желающие реализовать трансформер с нуля.

## Motivation

DL-сообщество не публикует псевдокод для нейросетевых архитектур. Многие статьи с ≥100 страницами описывают модели неформально — только диаграммы и проза. Проблема: нет точной точки отсчёта для последующих вариаций, чтения теоретических работ и воспроизводимых реализаций.

Авторы утверждают: открытый код != псевдокод. 2000+ строк C — не эквивалент 50 строк хорошо структурированного псевдокода.

## Tasks (Section 3)

Трансформеры решают три класса задач:

**1. Sequence Modelling (Decoder):**
`P̂(x) = ∏_t P̂_θ(x[t] | x[1:t-1])`
Примеры: language modelling, RL policy distillation, music generation.

**2. Sequence-to-Sequence (Encoder-Decoder):**
`P̂(x|z) = ∏_t P̂_θ(x[t] | x[1:t-1], z)`
Примеры: translation, QA, text-to-speech.

**3. Classification (Encoder):**
Estimate `P(c|x)` для меток `c ∈ [N_C]`.
Примеры: sentiment classification, spam filtering, toxicity.

## Tokenization (Section 4)

Три уровня токенизации:

| Тип | Словарь V | Длина | Проблемы |
|-----|-----------|-------|----------|
| Character-level | алфавит | длинные последовательности | медленно |
| Word-level | все слова | короткие | большой словарь, OOV |
| **Subword** (BPE) | субслова | оптимально | **стандарт** |

Специальные токены: `mask_token = N_V - 2`, `bos_token = N_V - 1`, `eos_token = N_V`.

## Architectural Components (Section 5)

Представлены в виде нумерованных алгоритмов:

**Algorithm 1: Token Embedding**
```
e = W_e[:, v]    # W_e ∈ R^(d_e × N_V)
```

**Algorithm 2: Positional Embedding**
```
e_p = W_p[:, l]  # W_p ∈ R^(d_e × l_max), learned
```
Оригинальный Transformer [Vaswani 2017] использует **синусоидальные** (hardcoded) positional embeddings:
```
W_p[2i-1, t] = sin(t / l_max^(2i/d_e))
W_p[2i, t]   = cos(t / l_max^(2i/d_e))
```
Итоговое embedding: `e = W_e[:, x[t]] + W_p[:, t]` — eq.(1)

**Algorithm 3–5: Attention**

*Single-head attention (Alg. 3–4):*
```
Q = W_q X + b_q    # queries
K = W_k Z + b_k    # keys (from context Z)
V = W_v Z + b_v    # values
S = K^T Q          # scores ∈ R^(l_z × l_x)
if not Mask[t_z, t_x]: S[t_z, t_x] = -∞
Ṽ = V · softmax(S / √d_attn)   # eq.2,3,4
```

Масочная функция (eq.3):
- `Mask[t_z, t_x] = 1` — bidirectional (BERT-style)
- `Mask[t_z, t_x] = [[t_z ≤ t_x]]` — unidirectional/causal (GPT-style)

*Multi-head attention (Alg. 5):*
```
For h in [H]:
    Y^h ← Attention(X, Z | W^h_qkv, Mask)
Y ← [Y^1; Y^2; ...; Y^H]      # concat
Ṽ = W_o Y + b_o 1^T
```

**Algorithm 6: Layer Normalization**
```
m = Σ_i e[i] / d_e
v = Σ_i (e[i] - m)^2 / d_e
ê = (e - m) / √v ⊙ γ + β
```
Примечание: RMSNorm = упрощённая версия с `m = β = 0`.

**Algorithm 7: Unembedding**
```
p = softmax(W_u e)    # W_u ∈ R^(N_V × d_e)
```
Иногда `W_u = W_e^T` (tied embeddings).

## Three Architectures (Section 6)

### EDT: Encoder-Decoder Transformer [Vaswani 2017] (Alg. 8)

Оригинальная архитектура для seq2seq:
1. Encoder: bidirectional self-attention → кодирует контекст z
2. Decoder: unidirectional self-attention + cross-attention к encoder output → генерирует x

Uses: ReLU activation, post-layer-norm (LayerNorm после sub-layer).

### ETransformer: BERT / Encoder-only (Alg. 9)

Bidirectional трансформер для classification/understanding:
- Uses: **GELU** activation, post-layer-norm
- `[CLS]` token → распределение по классам (aggregate representation)
- Masking: специальный (не через Mask параметр) — 15% токенов → `mask_token` во время pre-training

**Training: MLM (Alg. 12)**
```
For each token x[t] with prob p_mask:
    replace x[t] with mask_token
    predict original x[t] from context
```

### DTransformer: GPT / Decoder-only (Alg. 10)

Унидиректиональная авторегрессивная модель:
- Uses: **GELU** activation, **pre-layer-norm** (LayerNorm до sub-layer)
- Causal Mask: `Mask[t_z, t_x] = [[t_z ≤ t_x]]`

**Training: CLM (Alg. 13)**
```
For each token x[t]:
    predict next token x[t+1]
    loss = -log P(x[t+1] | x[1:t])
```

**Inference / Prompting (Alg. 14):**
```
Given prompt p = x[1:T]:
    sample x[T+1] ~ P_θ(· | x[1:T])
    append x[T+1] to p
    repeat
```

### Gato: Multi-modal Decoder-only [Reed et al. 2022]

Decoder-only transformer для нескольких модальностей (Atari, robotics, image captioning, dialogue). Каждая модальность конвертируется в sequence через отдельный tokenizer (изображения: 16×16 patches → ResNet block → vector). Архитектурно = GPT (Alg. 10) с модальность-специфичным embedding вместо Line 2.

## Training (Section 7)

Обобщённый алгоритм:
```
Алгоритм 11/12/13: Pre-training
Initialize parameters θ
For each minibatch:
    compute loss L(θ) = -Σ log P_θ(x[t] | context)
    update θ via gradient descent
```

Fine-tuning = та же процедура с меньшим LR на задача-специфичных данных.

## Practical Considerations (Section 8, brief)

- **Context length** ограничен max_l (fixed at training time для learned positional embeddings)
- **Chunking**: документы длиннее max_l разбиваются на chunks
- **Tied embeddings**: W_u = W_e^T — часто используется для уменьшения параметров
- **Different norms**: RMSNorm = m = β = 0 (более эффективный)

## Notation

- V = vocabulary (set of tokens)
- N_V = |V| (vocabulary size)
- d_e = embedding dimension
- d_attn = attention dimension per head
- H = number of attention heads
- l = sequence length (l_x, l_z для primary/context)
- l_max = maximum sequence length

## My notes

- Это не исследовательская статья в обычном смысле — это **референсный документ** для Transformer архитектур. Полезен именно как точная, самодостаточная спецификация.
- Ключевые различия BERT vs GPT: (1) bidirectional vs unidirectional masking, (2) GELU activations в обоих, (3) **post-LayerNorm** у BERT vs **pre-LayerNorm** у GPT. Pre-LayerNorm улучшает стабильность обучения больших моделей.
- **Синусоидальные positional embeddings**: формула eq для 2i-1 (sin) и 2i (cos) позиций при разных частотах — позволяет теоретически неограниченную длину; learned embeddings ограничены max_l.
- RMSNorm (γ только, без bias β и mean m) стал стандартом в LLaMA и современных LLM — алгоритм 6 с m=β=0.
- Статья дала официальную нотацию для многих теоретических работ по трансформерам.
