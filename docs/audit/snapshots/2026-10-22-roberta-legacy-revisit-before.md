---
title: "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
url: https://arxiv.org/abs/1907.11692
authors: [Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov]
year: 2019
date_reviewed: 2026-04-06
type: source-note
status: legacy
category: paper
tags:
  - BERT
  - arch
Date: 2019-07-26
Organization: Meta
Parent item:
  - "[[BERT]]"
Status: Done
concepts:
  - "[[02 Areas/ML & DL/Concepts/Architectures/RoBERTa|RoBERTa]]"
  - "[[02 Areas/ML & DL/Concepts/Architectures/Encoder-only|Encoder-only]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Masked Language Modeling|MLM]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/GLUE|GLUE]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Fine-tuning|Fine-tuning]]"
raw: "[[02 Areas/ML & DL/raw/papers/roberta/paper.txt]]"
---

# RoBERTa: A Robustly Optimized BERT Pretraining Approach

**Authors:** Yinhan Liu, Myle Ott, Naman Goyal et al. (Facebook AI Research / UW)
**Published:** 2019 (arXiv:1907.11692v1, Jul 2019)
**URL:** https://arxiv.org/abs/1907.11692

## TL;DR

RoBERTa — репликационное исследование BERT, выявившее, что оригинальная модель **значительно недообучена**. Четыре ключевых изменения в рецептуре предобучения (без изменения архитектуры!) дают SOTA на GLUE (88.5), SQuAD v2 (89.4 F1) и RACE (83.2%) — превосходя XLNet и все предыдущие методы. Главный вывод: тщательная настройка гиперпараметров, больше данных и дольше обучение важнее архитектурных инноваций.

## Problem

В 2019 году появилось много post-BERT методов (XLNet, ALBERT, XLMR), претендующих на улучшение через архитектурные изменения. Однако:
- Обучение BERT дорого, что ограничивает ablation studies
- Разные методы используют разные private данные разного размера
- Гиперпараметры сильно влияют на результат, но редко публикуются детально

**Авторы задают вопрос:** если мы тщательно воспроизведём BERT с оптимальными гиперпараметрами и данными — насколько хорош BERT сам по себе?

## Method

### 4 ключевых улучшения над BERT (Section 4)

**1. Dynamic Masking (Section 4.1)**

Оригинальный BERT: маски генерируются один раз при предобработке → данные дублируются 10× (каждая последовательность видится с одной из 10 масок за 40 эпох).

RoBERTa: маска генерируется **заново каждый раз**, когда последовательность подаётся в модель. Критично при обучении на большем числе шагов/данных.

Результат (Table 1): dynamic masking немного лучше или сопоставимо со static. SQuAD F1: 78.7 → 78.7 (dynamic sightly better on MNLI-m: 84.3 → 84.0, SST-2: 92.8 → 92.9).

**2. Удаление NSP + Full-Sentences формат (Section 4.2)**

Протестированы 4 варианта:
- SEGMENT-PAIR+NSP: оригинальный BERT (пары сегментов + NSP loss)
- SENTENCE-PAIR+NSP: пары отдельных предложений + NSP
- FULL-SENTENCES: последовательности до 512 токенов из любых документов, **без NSP**
- DOC-SENTENCES: только из одного документа, **без NSP**

Результаты (Table 2):
- SENTENCE-PAIR ухудшает — модель не учит long-range зависимости
- **FULL-SENTENCES лучше BERT с NSP** на SQuAD (90.4/79.1 vs 88.5/76.3), MNLI-m (84.7 vs 84.3)
- DOC-SENTENCES чуть лучше FULL-SENTENCES, но batch size варьируется → авторы выбирают FULL-SENTENCES для удобства

**Вывод: NSP не помогает и может мешать.** (Противоречит оригинальной BERT статье — там NSP убирали неправильно, сохраняя SEGMENT-PAIR формат.)

**3. Большие batches (Section 4.3)**

| Batch | Steps | LR | PPL | MNLI-m | SST-2 |
|-------|-------|-----|-----|--------|-------|
| 256 | 1M | 1e-4 | 3.99 | 84.7 | 92.7 |
| 2K | 125K | 7e-4 | 3.68 | 85.2 | 92.9 |
| **8K** | **31K** | **1e-3** | **3.77** | **84.6** | **92.8** |

Большие батчи улучшают PPL для MLM и end-task точность. RoBERTa обучается с batch=8K.

**4. Byte-level BPE (Section 4.4)**

Оригинальный BERT: character-level BPE, 30K токенов. RoBERTa: **byte-level BPE** (Radford et al., 2019), 50K субword единиц без дополнительной preprocessing/токенизации. +15M параметров (BASE) / +20M (LARGE). Небольшая деградация на некоторых задачах, но универсальность покрытия важнее.

### Dataset (Section 3.2)

**160GB** текста из 5 источников:
- BookCorpus + Wikipedia (оригинал BERT): 16GB
- **CC-NEWS** (CommonCrawl News, Sep 2016 — Feb 2019): 76GB после фильтрации
- **OpenWebText** (репликация WebText): 38GB
- **STORIES** (CommonCrawl, filtered by Winograd-style): 31GB

### Training: RoBERTa (Section 5, Table 4)

Архитектура: BERT_LARGE (L=24, H=1024, A=16, 355M параметров). 1024 V100 GPU, ~1 день.

| Данные | Batch | Шаги | SQuAD v1.1/v2.0 | MNLI-m | SST-2 |
|--------|-------|------|-----------------|--------|-------|
| 16GB (Books+Wiki) | 8K | 100K | 93.6/87.3 | 89.0 | 95.3 |
| 160GB (+CC-NEWS+...) | 8K | 100K | 94.0/87.7 | 89.3 | 95.6 |
| 160GB | 8K | 300K | 94.4/88.7 | 90.0 | 96.1 |
| 160GB | 8K | **500K** | **94.6/89.4** | **90.2** | **96.4** |

Каждый шаг аккумулирует улучшения от предыдущего.

## Key Results

### GLUE (Table 5)

| Модель | MNLI-m/mm | RTE | SST | CoLA | STS | Avg |
|--------|-----------|-----|-----|------|-----|-----|
| BERT_LARGE | 86.6/- | 70.4 | 93.2 | 60.6 | 90.0 | - |
| XLNet_LARGE | 89.8/- | 83.8 | 95.6 | 63.6 | 91.8 | - |
| **RoBERTa** | **90.2/90.2** | **86.6** | **96.4** | **68.0** | **92.4** | - |

Ensembles test (leaderboard):
- RoBERTa: **88.5** avg (XLNet: 88.4, MT-DNN: 87.6)
- SOTA на 4/9 GLUE задач: MNLI, QNLI, RTE, STS-B
- Без multi-task fine-tuning (большинство топ-конкурентов используют MTL)

### SQuAD (Table 6)

| Модель | v1.1 EM/F1 | v2.0 EM/F1 |
|--------|----------|----------|
| BERT_LARGE | 84.1/90.9 | 79.0/81.8 |
| XLNet_LARGE | 89.0/94.5 | 86.1/88.8 |
| **RoBERTa** | **88.9/94.6** | **86.5/89.4** |

Test set: RoBERTa 86.8/89.8 — top среди моделей **без аугментации данных** (XLNet test использует доп. данные).

### RACE (Table 7)

| Модель | Accuracy | Middle | High |
|--------|----------|--------|------|
| BERT_LARGE | 72.0 | 76.6 | 70.1 |
| XLNet_LARGE | 81.7 | 85.4 | 80.2 |
| **RoBERTa** | **83.2** | **86.5** | **81.3** |

Новый SOTA на RACE.

## Key Takeaways

1. **BERT was undertrained** — оригинал: 1M шагов с batch=256; RoBERTa при тех же данных и дольше обучении значительно лучше
2. **NSP вредит или не помогает** — удаление NSP улучшает SQuAD и стабилизирует MNLI
3. **Данные имеют значение** — 16GB → 160GB даёт +0.4 SQuAD F1, +0.3 MNLI
4. **Dynamic masking** — лучше или эквивалентно статическому при дольшем обучении
5. **Масштаб training steps** — 100K → 300K → 500K шагов = постоянный прирост, даже на 500K не наблюдается переобучения

## My notes

- RoBERTa — **эталонная ablation study** для encoder-only моделей. Показала, что многие "улучшения" XLNet и других методов объяснялись не архитектурными инновациями, а большими данными и дольшим обучением.
- Вывод про NSP стал стандартной мудростью: большинство последующих BERT-like моделей (ALBERT, DeBERTa, RoBERTa-Large) убирают NSP.
- Byte-level BPE → стал стандартом (GPT-2, GPT-3, LLaMA используют его вариации).
- RoBERTa остаётся актуальной базовой моделью для NLU задач (fine-tuning на GLUE/SQuAD). В 2024+ DeBERTa-v3 и ModernBERT занял её место, но RoBERTa — conceptual baseline.
- Практически важное: fine-tuning RoBERTa на RTE/STS/MRPC → начинать с MNLI-checkpoint, не с базового. Это значительно улучшает результаты для малых датасетов.
