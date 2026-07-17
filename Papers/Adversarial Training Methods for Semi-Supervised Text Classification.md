---
title: "Adversarial Training Methods for Semi-Supervised Text Classification"
url: https://arxiv.org/abs/1605.07725
authors: [Takeru Miyato, Andrew M. Dai, Ian Goodfellow]
year: 2017
date_reviewed: 2026-04-06
type: source-note
status: legacy
category: paper
tags:
  - Feature
Date: 2016-05-25
Organization: Google Brain / OpenAI
Status: Done
concepts:
  - "[[02 Areas/ML & DL/Concepts/Training/Adversarial Training|Adversarial Training]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Semi-supervised Learning|Semi-supervised Learning]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/Text Classification|Text Classification]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]]"
raw: "[[02 Areas/ML & DL/raw/papers/adversarial-training-methods-for-semi-supervised-text-classi/paper.txt]]"
---

# Adversarial Training Methods for Semi-Supervised Text Classification

**Authors:** Takeru Miyato (Preferred Networks/Kyoto), Andrew M. Dai (Google Brain), Ian Goodfellow (OpenAI)
**Published:** ICLR 2017 (arXiv:1605.07725v4, Nov 2021)
**URL:** https://arxiv.org/abs/1605.07725

## TL;DR

Работа расширяет Adversarial Training (AT) и Virtual Adversarial Training (VAT) на текстовую область: вместо perturbations исходного дискретного input применяются perturbations к **word embeddings** в LSTM. Сочетание LM pre-training + VAT даёт SOTA на нескольких полу-supervised и supervised бенчмарках (IMDB 5.91%, Elec 5.40%, RCV1 6.71%). Дополнительный вывод: adversarial training улучшает качество word embeddings, семантически разделяя слова со схожей грамматической, но разной смысловой ролью.

## Problem

Adversarial training в компьютерном зрении добавляет небольшие perturbations к вещественным входным векторам. Для текста это невозможно: входы — **дискретные one-hot векторы**, не допускающие инфинитезимальных изменений. Пространство one-hot векторов не имеет метрики для "малого" возмущения.

Решение: перенести perturbations на **word embeddings** — непрерывные векторы в ℝ^D, которые являются первым дифференцируемым слоем модели. Это новое применение AT/VAT, первое для текстовых задач.

## Method

### Модель

LSTM-based text classifier (рис. 1a):
- Вход: последовательность слов w(1)...w(T) → word embeddings v(1)...v(T)
- LSTM → softmax → class distribution P(y|s; θ)
- Дополнительно: bidirectional LSTM (revers + forward concatenated)

**Нормализация embeddings** (eq.1): для предотвращения патологического решения (модель учится делать embeddings очень большими, чтобы perturbations стали незначимыми):
```
v̄_k = (v_k - E[v]) / √Var(v)
```
где E[v] и Var(v) считаются по частотам слов f_j.

### Adversarial Training (AT)

Добавляет к loss функции (eq.2):
```
L_adv(θ) = -1/N Σ log p(y_n | s_n + r_adv,n; θ)
```
где adversarial perturbation (eq.5):
```
r_adv = -ε g / ||g||_2,   g = ∇_s log p(y | s; θ̂)
```
— градиент loss по embedding. Worst-case perturbation в направлении, максимально увеличивающем loss.

Требует меток y (supervised only).

### Virtual Adversarial Training (VAT)

Не требует меток — может использовать unlabeled данные (semi-supervised). Loss (eq.3,8):
```
L_v-adv(θ) = 1/N' Σ KL[p(·|s_n'; θ̂) || p(·|s_n' + r_v-adv,n'; θ)]
```
Virtual adversarial perturbation (eq.7):
```
r_v-adv = ε g / ||g||_2,   g = ∇_{s+d} KL[p(·|s;θ̂) || p(·|s+d;θ̂)]
```
где d — маленький случайный вектор. Это 2nd-order Taylor expansion + одна итерация power method.

Модель обучается быть устойчивой к perturbations без знания истинного label — только на основе KL divergence между оригинальным и возмущённым output.

### Pre-training

LM pre-training (Dai & Le, 2015): unidirectional single-layer LSTM, 1024 hidden units, 256/512-d embeddings, sampled softmax (1024 кандидатов), Adam (lr=0.001), 100K steps. Значительно улучшает classification на всех датасетах.

### Training

После pre-training — fine-tuning с adversarial/virtual adversarial loss:
- Adam (lr=0.0005), LR decay 0.9998
- Embedding dropout (основной гиперпараметр)
- Один дополнительный гиперпараметр ε (norm constraint)
- Gradient clipping, truncated backprop ≤400 слов

## Key Results

### IMDB Sentiment (Table 2)

| Method | Test error rate |
|--------|----------------|
| Baseline (pretrain + emb. dropout) | 7.39% |
| Random perturbation (labeled) | 7.20% |
| Random perturbation (labeled+unlabeled) | 6.78% |
| Adversarial | 6.21% |
| **Virtual Adversarial** | **5.91%** |
| One-hot bi-LSTM (prev SOTA, Johnson & Zhang 2016) | 5.94% |

**VAT: 5.91%** — сопоставимо со SOTA (bidirectional LSTM), при использовании unidirectional LSTM. VAT < Adversarial, т.к. использует unlabeled данные.

Adversarial > Random perturbation: подтверждает, что смысл в направленности perturbation, а не просто шуме.

### Elec & RCV1 (Table 4)

| Method | Elec | RCV1 |
|--------|------|------|
| Baseline | 6.24% | 7.40% |
| Adversarial | 5.61% | 7.12% |
| Virtual Adversarial | 5.54% | 7.05% |
| **Adv + VAT** | **5.40%** | 6.97% |
| **VAT (bi-LSTM)** | 5.55% | **6.71%** |
| Prev SOTA (CNN+bi-LSTM) | 5.42% | 6.64% |

Достигнут SOTA на Elec (5.40%) с unidirectional LSTM. На RCV1 — bi-LSTM VAT (6.71%) близок к SOTA.

### DBpedia (Table 5, supervised)

| Method | Test error rate |
|--------|----------------|
| Baseline | 0.84% |
| Adversarial | 0.60% |
| LSTM | 1.3% (Zhang et al.) |
| CNN | 0.84% (Zhang et al.) |

**Adversarial: 0.60%** — значительно лучше baselines.

### Word Embedding Analysis (Table 3)

Adversarial training семантически разделяет слова с похожей грамматической, но разной семантической ролью:

- Baseline/Random: 'bad' — 3-й ближайший сосед к 'good' (обе — прилагательные, модифицирующие одинаковые существительные → LM предсказывает их в одинаковых контекстах)
- **Adversarial/VAT**: 'bad' опускается до **19-го/21-го** соседа 'good'

Это происходит потому, что adversarial training требует, чтобы смысл предложения нельзя было инвертировать малым изменением embedding → слова с противоположным значением должны быть далеко в пространстве.

## My notes

- Работа исторически важна: первое применение AT/VAT к тексту. Концепция "perturbation на embedding, а не на input" стала стандартной для NLP adversarial robustness.
- VAT принципиально отличается от AT: не требует label, может использовать unlabeled corpora. Это делает его мощным инструментом для semi-supervised NLP.
- Дополнительный гиперпараметр всего один (ε) — практически очень важно для adoption.
- Визуализация embeddings (Table 3) элегантно показывает, что adversarial training улучшает семантику: грамматически похожие, но семантически противоположные слова (good/bad) расходятся.
- SiFT (DeBERTa), используемый для fine-tuning крупных моделей, является прямым потомком этой работы — те же принципы, адаптированные для Transformer эпохи.
