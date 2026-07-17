---
title: "DeBERTa: Decoding-enhanced BERT with Disentangled Attention"
url: https://arxiv.org/abs/2006.03654
authors: [Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen]
year: 2021
date_reviewed: 2026-04-06
type: paper-review
category: paper
tags:
  - BERT
  - arch
Date: 2020-05-06
Organization: Microsoft
Parent item:
  - "[[BERT]]"
Status: Done
Sub-item:
  - "[[02 Areas/ML & DL/Papers/DeBERTa|DeBERTa v3]]"
concepts:
  - "[[02 Areas/ML & DL/Concepts/Architectures/DeBERTa|DeBERTa]]"
  - "[[02 Areas/ML & DL/Concepts/Architectures/Encoder-only|Encoder-only]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Disentangled Attention|Disentangled Attention]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/Positional Encoding|Positional Encoding]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/GLUE|GLUE]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Masked Language Modeling|MLM]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/Transfer Learning|Transfer Learning]]"
raw: "[[02 Areas/ML & DL/raw/papers/deberta/paper.txt]]"
---

# DeBERTa: Decoding-enhanced BERT with Disentangled Attention

**Authors:** Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen (Microsoft)
**Published:** ICLR 2021 (arXiv:2006.03654v6, Oct 2021)
**URL:** https://arxiv.org/abs/2006.03654

## TL;DR

DeBERTa улучшает BERT и RoBERTa двумя архитектурными нововведениями: (1) **Disentangled Attention** — каждый токен представлен двумя векторами (контент + позиция), attention вычисляется диагонально по четырём компонентам; (2) **Enhanced Mask Decoder (EMD)** — абсолютные позиции добавляются только в слой декодирования маски. Обученная на **половине данных RoBERTa** (78GB vs 160GB), DeBERTa превосходит её на всех задачах. 1.5B-параметровая версия впервые **превышает человеческий baseline** на SuperGLUE (89.9 vs 89.8).

## Problem

Все существующие подходы к positional encoding в Transformer либо используют абсолютные позиции (BERT, GPT), добавляемые к word embeddings в input layer, либо относительные position bias (XLNet, Shaw et al.). Ни один не разделяет **content** и **position** как независимые сигналы при вычислении attention.

Проблема: attention weight между двумя токенами зависит и от их содержания, и от их взаимного расположения — но в существующих методах эти сигналы смешиваются. DeBERTa предлагает "disentangle" (разделить) их в явную формулу с отдельными компонентами.

Вторая проблема: диагональный attention с relative positions не учитывает absolute position при предсказании masked tokens. Например, в "a new store opened beside the new mall" слова "store" и "mall" имеют одинаковый локальный контекст (оба следуют за "new") — различить их можно только по абсолютной позиции.

## Method

### 1. Disentangled Attention (Section 3.1)

Каждый токен в позиции i представлен двумя векторами:
- `H_i` — content embedding
- `P_{i|j}` — relative position embedding от i до j

Attention score между i и j декомпозируется на **3 (из 4)** компонентов:
```
Ã_{i,j} = Q^c_i K^c_j^T         # (a) content-to-content
         + Q^c_i K^r_{δ(i,j)}^T  # (b) content-to-position
         + K^c_j Q^r_{δ(j,i)}^T  # (c) position-to-content
```
Четвёртый term (position-to-position) удалён как несущественный при relative encoding.

Масштабирующий фактор `1/√(3d)` вместо стандартного `1/√d` (три слагаемых, а не одно).

**Реализация:** relative position embeddings `P ∈ R^{2k×d}` shared across layers (k = максимальная relative distance = 512). Сложность: `O(kd)` вместо `O(N²d)` — существенная экономия памяти.

### 2. Enhanced Mask Decoder (EMD) (Section 3.2)

В стандартном BERT абсолютные позиции добавляются к input embeddings в первом слое. Результат: информация о relative positions хуже усваивается в промежуточных слоях.

DeBERTa **откладывает** добавление абсолютных позиций до softmax слоя для предсказания masked tokens. Все Transformer layers работают только с relative positions. EMD = "decoding-enhanced" компонент, добавляющий абсолютные позиции как дополнительный bias непосредственно перед финальным softmax.

Ablation (Table 4 в статье): EMD даёт +0.3 MNLI-m, +0.4 SQuAD v2 F1 при удалении.

### 3. SiFT: Scale-invariant Fine-Tuning (Section 4)

Virtual adversarial training для fine-tuning: perturbations применяются к **normalized** word embeddings (не к исходным). Нормализация устраняет нестабильность у больших моделей. Применяется только для DeBERTa-1.5B на SuperGLUE.

### Pre-training Details

- **Архитектура (Large):** L=24, H=1024, A=16 — идентично BERT-Large/RoBERTa-Large. BPE-50K (как RoBERTa).
- **Данные:** Wikipedia (12GB) + BookCorpus (6GB) + OpenWebText (38GB) + STORIES (31GB) = **78GB** после дедупликации. (RoBERTa: 160GB)
- **Training:** batch=2K, 1M steps, 6 DGX-2 machines (96 V100), ~20 дней
- **Данных вдвое меньше**, чем у RoBERTa/XLNet (2B vs 4B samples)

## Key Results

### GLUE Dev Set (Table 1)

| Модель | MNLI-m/mm | CoLA | SST-2 | STS-B | QNLI | RTE | MRPC | Avg |
|--------|-----------|------|-------|-------|------|-----|------|-----|
| BERT_L | 86.6/- | 60.6 | 93.2 | 90.0 | 92.3 | 70.4 | 88.0 | 84.05 |
| RoBERTa_L | 90.2/90.2 | 68.0 | 96.4 | 92.4 | 93.9 | 86.6 | 90.9 | 88.82 |
| XLNet_L | 90.8/90.8 | 69.0 | 97.0 | 92.5 | 94.9 | 85.9 | 90.8 | 89.15 |
| ELECTRA_L | 90.9/- | 69.1 | 96.9 | 92.6 | 95.0 | 88.0 | 90.8 | 89.46 |
| **DeBERTa_L** | **91.1/91.1** | **70.5** | **96.8** | **92.8** | **95.3** | **88.3** | **91.9** | **90.00** |

DeBERTa устанавливает SOTA на всём GLUE (avg 90.0), при **78GB данных vs 160GB** у конкурентов.

### Extended NLU Benchmarks (Table 2)

| Модель | SQuAD v1.1 F1/EM | SQuAD v2.0 F1/EM | RACE | ReCoRD F1/EM | SWAG |
|--------|-----------------|-----------------|------|-------------|------|
| RoBERTa_L | 94.6/88.9 | 89.4/86.5 | 83.2 | 90.6/90.0 | 89.9 |
| XLNet_L | 95.1/89.7 | 90.6/87.9 | 85.4 | – | – |
| **DeBERTa_L** | **95.5/90.1** | **90.7/88.0** | **86.8** | **91.4/91.0** | **90.8** |

Улучшения над RoBERTa: MNLI +0.9%, SQuAD v2.0 +2.3%, RACE **+3.6%**.

### Base Models (Table 3)

| Модель | MNLI-m/mm | SQuAD v1.1 F1/EM | SQuAD v2.0 F1/EM |
|--------|-----------|-----------------|-----------------|
| RoBERTa_Base | 87.6/- | 91.5/84.6 | 83.7/80.5 |
| XLNet_Base | 86.8/- | –/– | –/80.2 |
| **DeBERTa_Base** | **88.8/88.5** | **93.1/87.2** | **86.2/83.1** |

Base model (78GB) превышает RoBERTa Base (160GB) на всех задачах. MNLI: +1.2% над RoBERTa, +2% над XLNet.

### SuperGLUE: 1.5B Model — First Human Parity

DeBERTa с 48 слоями и **1.5 млрд параметров**:
- **Single model: 89.9** (Human: 89.8) — **впервые превышает human baseline**
- **Ensemble: 90.3** (первое место на leaderboard на Jan 6, 2021)
- T5-11B достигает только 89.3 (на 0.6% ниже DeBERTa-1.5B, при ~7x большем числе параметров)

### Ablation: Disentangled Attention components (Table 4)

| Вариант | MNLI-m/mm | SQuAD v1.1 | SQuAD v2.0 | RACE |
|---------|-----------|-----------|-----------|------|
| DeBERTa_Base | 86.3/86.2 | 92.1/86.1 | 82.5/79.3 | 71.7 |
| -EMD | 85.4/85.1 | 91.5/85.4 | 81.1/77.8 | 70.2 |
| -C2P (no content-to-position) | 85.4/85.2 | 91.5/85.0 | 81.3/77.8 | 70.0 |
| -P2C (no position-to-content) | 85.6/85.5 | 91.6/85.4 | 80.4/77.4 | 70.3 |

Все три компонента (EMD, C2P, P2C) вносят значительный вклад.

### NLG: WikiText-103

DeBERTa снижает perplexity с **21.6** (RoBERTa-based) до **19.5** на WikiText-103.

## My notes

- DeBERTa — **лучшая encoder-only архитектура** своего времени. DeBERTa v3 (с ELECTRA-style training + MDeBERTa) стал де-факто стандартом для NLU задач в 2022–2024.
- Диагональный attention — элегантная идея: вместо единого вектора `H_i + P_i` (BERT) разделить на `(H_i, P_i)` и вычислять attention по 3 компонентам. Это позволяет модели явно учитывать "где я смотрю относительно кого я смотрю" vs "что я вижу".
- **EMD** — практически важный инсайт: абсолютные позиции нужны только для предсказания masked token, а не для "понимания" контекста. Аналогичный принцип используется в T5 (relative embeddings only).
- **1.5B > 11B (T5) на SuperGLUE** — демонстрирует, что архитектурные улучшения важнее чистого scale.
- SiFT adversarial training = виртуальная аугментация только для крупных моделей. В практике часто не применяется — слишком дорого.
