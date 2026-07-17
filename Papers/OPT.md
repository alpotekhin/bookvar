---
title: "OPT: Open Pre-trained Transformer Language Models"
url: https://arxiv.org/abs/2205.01068
authors: [Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, Luke Zettlemoyer]
year: 2022
date_reviewed: 2026-04-06
type: paper-review
category: paper
tags:
  - LLM
  - arch
Date: 2022-01-05
Organization: Meta
Status: Done
concepts:
  - "[[02 Areas/ML & DL/Concepts/Architectures/OPT|OPT]]"
  - "[[02 Areas/ML & DL/Concepts/Architectures/Decoder-only|Decoder-only]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Scaling Laws|Scaling Laws]]"
  - "[[02 Areas/ML & DL/Concepts/Inference/In-context Learning|In-context Learning]]"
raw: "[[02 Areas/ML & DL/raw/papers/opt/paper.txt]]"
---

# OPT: Open Pre-trained Transformer Language Models

**Authors:** Susan Zhang, Stephen Roller, Naman Goyal et al. (Meta AI)
**Published:** 2022 (arXiv:2205.01068v4, Jun 2022)
**URL:** https://arxiv.org/abs/2205.01068

## TL;DR

OPT (Open Pre-trained Transformers) — серия открытых decoder-only LLM от Meta (125M–175B параметров), нацеленная на воспроизведение GPT-3 class моделей. OPT-175B сравним с GPT-3 на большинстве бенчмарков, при этом требует **в 7× меньше углеродного следа** при разработке. Ключевой вклад — открытость: полные веса, код (metaseq), и **logbook со всеми проблемами при обучении** (hardware failures, loss divergences) для исследователей. Первый полностью открытый аналог GPT-3.

## Problem

К 2022 году крупные LLMs (GPT-3, Gopher, PaLM) доступны только через платные API без доступа к весам. Это ограничивает:
- **Воспроизводимость** исследований
- **Изучение** bias, toxicity, механизмов работы
- **Разнообразие голосов** в обсуждении этических вопросов AI

Существующие открытые модели ограничены 20B (EleutherAI GPT-NeoX). OPT предоставляет модели до 175B для академических и некоммерческих исследователей.

## Method

### Архитектура (Table 1)

Стандартный decoder-only Transformer (следует GPT-3). 9 размеров:

| Размер | Слои | Heads | d_model | LR | Batch (tokens) |
|--------|------|-------|---------|-----|----------------|
| 125M | 12 | 12 | 768 | 6e-4 | 0.5M |
| 350M | 24 | 16 | 1024 | 3e-4 | 0.5M |
| 1.3B | 24 | 32 | 2048 | 2e-4 | 1M |
| 2.7B | 32 | 32 | 2560 | 1.6e-4 | 1M |
| 6.7B | 32 | 32 | 4096 | 1.2e-4 | 2M |
| 13B | 40 | 40 | 5120 | 1e-4 | 4M |
| 30B | 48 | 56 | 7168 | 1e-4 | 4M |
| 66B | 64 | 72 | 9216 | 0.8e-4 | 2M |
| **175B** | **96** | **96** | **12288** | **1.2e-4** | **2M** |

ReLU activation (не GeLU, в отличие от BERT). Sequence length = 2048. Все bias terms = 0.

### Optimizer & Training

AdamW (β₁=0.9, β₂=0.95), weight decay=0.1, linear LR schedule, gradient clip=1.0 (снижался до 0.3 при нестабильности). Adam state в FP32, weights в FP16, dynamic loss scaling.

OPT-175B обучался на **992 A100 80GB GPU** через:
- FSDP (Fully Sharded Data Parallel)
- Megatron-LM Tensor Parallelism
- Утилизация: **147 TFLOP/s per GPU**

### Pre-training Corpus (Section 2.3)

Объединение трёх источников (~**180B токенов**):
- **RoBERTa subset:** BookCorpus + Stories + CCNews v2
- **The Pile subset:** CommonCrawl, DM Mathematics, Project Gutenberg, HackerNews, OpenSubtitles, OpenWebText2, USPTO, Wikipedia (другие subsets удалены из-за нестабильностей при обучении)
- **PushShift.io Reddit:** longest chains из threads (~66% сокращение)

Дедупликация через MinHashLSH (Jaccard ≥ 0.95). GPT-2 BPE токенизатор.

⚠️ **The Pile был особенно полон дубликатов** — авторы рекомендуют доп. дедупликацию.

### Training Process: Challenges (Section 2.5)

OPT авторы полностью раскрывают процесс обучения — редкость для LLM:

**Hardware Failures:**
- ≥35 manual restarts за 2 месяца
- >100 хостов заменено
- 70+ автоматических restarts

**Loss Divergences:**
- Решение: снижение LR + restart с более ранним checkpoint
- Симптомы: dynamic loss scalar → 0, activation norms резко растут
- Другие меры: временный переход на SGD (не помог, откатились), reset dynamic loss scalar, upgrade версии Megatron

**Эмпирический LR schedule (Figure 1):** сложный, с ручными снижениями в нескольких точках.

## Key Results

### NLP Benchmarks: Zero-shot (Figure 3)

Средний zero-shot по 14 задачам:
- OPT **сопоставим с GPT-3** по всему диапазону размеров (125M–175B)
- На 10 из 14 задач: результаты совпадают
- Отстаёт на ARC Challenge, MultiRC
- Превосходит GPT-3 на WIC

Задачи с непредсказуемым поведением у обеих моделей (мало примеров в validation): CB (56), BoolQ (277), WSC (104).

### Few-shot (Figure 4)

One- и few-shot OPT в среднем **отстаёт от GPT-3**. Паттерн per-task непоследователен — возможны различия в evaluation setup.

### Dialogue (Table 2)

OPT-175B в unsupervised режиме vs supervised BlenderBot 1 на 5 датасетах:

| Датасет | Reddit 2.7B | BlenderBot 1 (sup) | OPT-175B (unsup) |
|---------|-------------|-------------------|------------------|
| ConvAI2 PPL | 18.9 | 10.2 | 10.8 |
| Wizard PPL | 21.0 | 12.5 | 13.3 |

OPT-175B unsupervised **конкурентоспособен** с supervised BlenderBot 1. Авторы исключают data leakage через hidden test set проверку.

## Bias & Toxicity Analysis (Section 4)

### Hate Speech Detection (Table 3, ETHOS dataset)

OPT-175B **значительно превосходит GPT-3/Davinci** в few-shot settings:
- Few-shot multiclass F1: OPT 0.812 vs Davinci 0.672

Причина: большое количество немодерированных Reddit данных → модель знакома с токсичным языком → лучше детектирует его.

### CrowS-Pairs Stereotypes (Table 4)

OPT-175B демонстрирует **более высокий stereotype bias**, чем GPT-3 Davinci в большинстве категорий (кроме религии). Объяснение: Reddit corpus содержит больше дискриминирующих текстов.

### StereoSet (Table 5)

OPT-175B и Davinci — **схожие результаты** по aggregate ICAT score. Davinci лучше в profession/race; OPT лучше в gender/religion.

### RealToxicityPrompts (Figure 5)

OPT-175B генерирует **более токсичный контент**, чем Davinci или PaLM при любом уровне prompt toxicity. Объяснение: то же — Reddit данные.

## Limitations

- Fewer-shot performance отстаёт от GPT-3
- Более токсичен чем GPT-3/PaLM из-за Reddit данных
- Сложный training process с частыми failures — не исключает повторных подобных проблем
- 175B доступна только для исследователей (не свободно, по запросу)

## My notes

- OPT — **важный исторический шаг** в open-source LLM. Первая полностью открытая GPT-3-comparable модель. Дала исследователям базу для изучения, что внутри LLM такого класса.
- **Logbook** — уникальный вклад OPT. Детальное описание hardware failures, loss divergences, mid-flight changes — practical guide по обучению large LLM, которого раньше не было публично.
- ReLU вместо GeLU — OPT сохраняет оригинальный GPT-3 стек. Последующие открытые LLM (LLaMA) перешли на SwiGLU.
- **Toxicity/bias trade-off:** Reddit данные улучшают hate speech detection, но ухудшают stereotype scores. Нет хорошего решения без тщательного data curation.
- OPT устарел после LLaMA (2023): LLaMA-13B лучше GPT-3 175B. OPT представляет ценность как исторический объект и infrastructure reference, а не как current SOTA.
