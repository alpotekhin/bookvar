---
title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
url: https://arxiv.org/abs/1810.04805
authors: [Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova]
year: 2018
date_reviewed: 2026-04-06
type: paper-review
category: paper
tags:
  - BERT
  - LLM
  - arch
Date: 2018-01-10
Organization: Google
Status: Done
Sub-item:
  - "[[RoBERTa]]"
  - "[[DeBERTa]]"
  - "[[Revisiting Few-sample BERT Fine-tuning]]"
concepts:
  - "[[02 Areas/ML & DL/Concepts/Architectures/BERT|BERT]]"
  - "[[02 Areas/ML & DL/Concepts/Architectures/Encoder-only|Encoder-only]]"
  - "[[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Masked Language Modeling|MLM]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Next Sentence Prediction|NSP]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Fine-tuning|Fine-tuning]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/Transfer Learning|Transfer Learning]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/GLUE|GLUE]]"
raw: "[[02 Areas/ML & DL/raw/papers/bert/paper.txt]]"
---

# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

**Authors:** Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova (Google AI Language)
**Published:** 2018 (arXiv:1810.04805v2, May 2019)
**URL:** https://arxiv.org/abs/1810.04805

## TL;DR

BERT — Bidirectional Encoder Representations from Transformers — первая модель, которая делает **глубокое двунаправленное** предобучение для языковых представлений. В отличие от GPT (слева направо) и ELMo (конкатенация двух однонаправленных LSTM), BERT видит контекст с обеих сторон на каждом слое. Это позволяет достичь SOTA на 11 NLP задачах с одним дообученным классификационным слоем поверх.

## Problem

До BERT стратегии применения предобученных языковых представлений делились на два лагеря:
- **Feature-based** (ELMo): используют задача-специфичные архитектуры, в которые предобученные представления добавляются как дополнительные признаки.
- **Fine-tuning** (OpenAI GPT): вводят минимальные задача-специфичные параметры и дообучают все веса.

Общая проблема обоих подходов — **однонаправленность**: стандартные LM могут обрабатывать только слева направо (GPT) или использовать мелкое объединение двух однонаправленных моделей (ELMo). Для задач уровня токенов (NER, QA) это критично — нельзя учитывать правый контекст при предсказании каждого токена.

## Method

### Архитектура

BERT — это многослойный двунаправленный [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] encoder (Vaswani et al., 2017). Два варианта:
- **BERT\_{BASE}**: L=12, H=768, A=12, **110M параметров**
- **BERT\_{LARGE}**: L=24, H=1024, A=16, **340M параметров**

BERT_{BASE} намеренно сделан того же размера, что OpenAI GPT, для честного сравнения. Ключевое отличие: BERT использует **двунаправленный self-attention**, GPT — только левосторонний.

### Input representation

Вход BERT — сумма трёх эмбеддингов (Figure 2 в статье):
1. **Token embeddings** — WordPiece словарь из 30,000 токенов
2. **Segment embeddings** — принадлежность к предложению A или B (EA / EB)
3. **Position embeddings** — позиция токена в последовательности

Специальные токены: `[CLS]` в начале каждого примера (его финальный hidden state = агрегированное представление для классификации), `[SEP]` — разделитель предложений.

### Pre-training: два задания

**Task 1: Masked Language Model (MLM)**

Случайно маскируется 15% WordPiece токенов. Из них:
- 80% заменяются на `[MASK]`
- 10% заменяются на случайный токен
- 10% остаются без изменений

Цель — предсказать оригинальный токен по контексту с обеих сторон. Такая схема (а не 100% `[MASK]`) нужна для уменьшения rассогласования между pre-training и fine-tuning, поскольку `[MASK]` не появляется при дообучении.

**Task 2: Next Sentence Prediction (NSP)**

50% времени предложение B является реальным следующим после A (IsNext), 50% — случайным из корпуса (NotNext). Финальный вектор `[CLS]` используется для предсказания. Задание критично для QA и NLI, где важно понимать отношения между предложениями.

### Pre-training data

BooksCorpus (800M слов) + English Wikipedia (2,500M слов). Важно использовать корпус уровня документов (не перемешанные предложения) для получения длинных контекстов.

**Параметры обучения:**
- Batch size: 256 последовательностей × 512 токенов = 128,000 токенов/батч
- 1,000,000 шагов (~40 эпох по 3.3B слов корпуса)
- Adam (lr=1e-4, β₁=0.9, β₂=0.999), L2 weight decay=0.01
- Dropout 0.1 на всех слоях, активация GELU
- BERT_{BASE}: 4 Cloud TPU, BERT_{LARGE}: 16 Cloud TPU, по 4 дня

### Fine-tuning

Fine-tuning прост: для каждой задачи подставляются соответствующие inputs/outputs, все параметры дообучаются end-to-end. Минимальные задача-специфичные параметры (только classification layer W ∈ ℝ^{K×H}).

Примеры адаптации:
- Классификация предложений: вектор `[CLS]` → softmax
- QA (SQuAD): добавляются start-vector S и end-vector E, вероятность i-го токена как начала ответа: P_i = exp(S·T_i) / Σ_j exp(S·T_j)
- NER: token-level предсказания по T_i

## Key results

### GLUE (Table 1)

| Система | MNLI-m/mm | Average |
|---------|-----------|---------|
| Pre-OpenAI SOTA | 80.6/80.1 | 74.0 |
| OpenAI GPT | 82.1/81.4 | 75.1 |
| **BERT_{BASE}** | **84.6/83.4** | **79.6** |
| **BERT_{LARGE}** | **86.7/85.9** | **82.1** |

**BERT_{LARGE} на официальном GLUE leaderboard: 80.5%** (GPT: 72.8%). Улучшение: +7.7 пп абсолютных.

### SQuAD v1.1 (Table 2)

| Система | Test F1 |
|---------|---------|
| Human | 91.2 |
| BERT_{LARGE} Ens.+TriviaQA | **93.2** |
| BERT_{LARGE} Single | 91.8 |

**+1.5 F1 над лучшим ансамблем** в leaderboard (одна модель BERT обходит все ансамбли).

### SQuAD v2.0 (Table 3)

BERT_{LARGE}: Test F1 = **83.1** (+5.1 F1 над предыдущим SOTA).

### SWAG (Table 4)

BERT_{LARGE}: **86.3%** (+27.1% над ESIM+ELMo baseline авторов, +8.3% над OpenAI GPT).

### NER CoNLL-2003 (Table 7)

Fine-tuning: **BERT_{LARGE} 92.8 F1** (SOTA). Feature-based (concat last 4 layers): 96.1 F1 на dev — всего -0.3 от fine-tuning.

## Ablation Studies

**Важность двунаправленности (Table 5):**

| Модель | MNLI-m | SQuAD F1 |
|--------|--------|----------|
| BERT_{BASE} | 84.4 | 88.5 |
| No NSP | 83.9 | 87.9 |
| LTR & No NSP | 82.1 | 77.8 |
| LTR + BiLSTM | 82.1 | 84.9 |

Вывод: убрать NSP — небольшое падение; перейти на LTR — существенное падение на SQuAD (с 88.5 до 77.8). Даже добавление BiLSTM поверх LTR не восстанавливает до уровня MLM.

**Эффект размера модели (Table 6):**

Больший размер → стабильное улучшение на всех задачах, включая маленькие датасеты (MRPC: 3,600 примеров). BERT впервые убедительно показывает, что scaling дает прирост даже на небольших задачах.

**MLM masking strategy (Table 8):**

Схема 80% MASK / 10% random / 10% unchanged оптимальна. 100% MASK — проблемы для feature-based NER. 100% random — заметно хуже.

## My notes

- BERT открыл эру "pre-train + fine-tune" как доминирующей парадигмы NLP. До него — задача-специфичные архитектуры.
- Ключевой инсайт: двунаправленность критична для токен-уровневых задач (QA, NER). Задачи вроде SQuAD падают на 10+ пунктов F1 без неё.
- NSP оказался полезным для QA/NLI, но позже RoBERTa показала, что он переоценён и можно обойтись без него.
- MLM принуждает модель держать распределённое контекстное представление каждого токена, т.к. encoder не знает, какие токены замаскированы.
- Fine-tuning на 1 Cloud TPU за ~1 час — революционно доступно по сравнению с обучением с нуля.
