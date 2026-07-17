---
title: "Scaling Instruction-Finetuned Language Models (Flan-T5 / Flan-PaLM)"
url: https://arxiv.org/abs/2210.11416
authors: [Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, et al.]
year: 2022
date_reviewed: 2026-04-06
type: paper-review
category: paper
tags:
  - LLM
  - T5
Date: 2022-01-10
Organization: Google
Parent item:
  - "[[T5]]"
Status: Done
concepts:
  - "[[02 Areas/ML & DL/Concepts/Architectures/Flan-T5|Flan-T5]]"
  - "[[02 Areas/ML & DL/Concepts/Architectures/T5|T5]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Instruction Tuning|Instruction Tuning]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/Transfer Learning|Transfer Learning]]"
  - "[[02 Areas/ML & DL/Concepts/Inference/Chain of Thought|Chain of Thought]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Scaling Laws|Scaling Laws]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/GLUE|GLUE]]"
raw: "[[02 Areas/ML & DL/raw/papers/flan-t5-palm/paper.txt]]"
---

# Scaling Instruction-Finetuned Language Models (Flan-T5 / Flan-PaLM)

**Authors:** Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, Jason Wei et al. (Google)
**Published:** 2022 (arXiv:2210.11416v5, Dec 2022)
**URL:** https://arxiv.org/abs/2210.11416

## TL;DR

Авторы систематически исследуют instruction finetuning с тремя осями масштабирования: (1) число задач (1.8K), (2) размер модели (до 540B), (3) добавление [[02 Areas/ML & DL/Concepts/Inference/Chain of Thought|Chain of Thought]] данных. Результаты: **Flan-PaLM 540B** достигает SOTA на MMLU (75.2% с CoT+SC), BBH, TyDiQA, MGSM. Публично выпущены **Flan-T5 checkpoints** (80M–11B), обгоняющие T5 на двузначные величины и даже конкурирующие с PaLM 62B.

## Problem

Instruction finetuning (FLAN, T0, InstructGPT) улучшает generalization к unseen задачам, но остаются открытые вопросы:
1. Масштабируется ли instruction finetuning с числом задач и размером модели?
2. Как совместить instruction finetuning с CoT? Прошлые подходы без CoT **ухудшали** CoT performance при fine-tuning.
3. Нужны ли публичные checkpoints для исследований?

## Method

### Flan Finetuning Data (Figure 2)

Комбинация 4 task mixtures — **473 датасета, 146 категорий задач, 1,836 задач**:

| Mixture | Датасеты | Категории | Задачи |
|---------|---------|-----------|-------|
| Muffin | 80 | ~25 | 80 |
| T0-SF | ~55 | 14 | 193 |
| NIV2 | ~372 | 108 | 1,554 |
| **CoT** | **9** | **1** | **9** |

**CoT mixture** — 9 датасетов с human-written chain-of-thought аннотациями: arithmetic reasoning (GSM8K), multi-hop reasoning (StrategyQA), natural language inference (e-NLI) и другие. **10 instruction templates** написаны вручную для каждого.

**Held-out tasks** (не в finetuning): MMLU (57 tasks), BBH (23 tasks), TyDiQA (8 languages), MGSM (10 languages).

### Форматы данных (Figure 3)

4 комбинации:
1. Instruction without exemplars
2. Instruction with few-shot exemplars
3. Chain-of-thought without exemplars
4. Chain-of-thought with exemplars

### Модели (Table 2)

Instruction finetuning применяется к нескольким семействам:

| Размер | Модель | Архитектура | Finetune % compute |
|--------|--------|-------------|-------------------|
| 80M–11B | **Flan-T5** | encoder-decoder | 0.2–1.6% |
| 8B, 62B, 540B | **Flan-PaLM** | decoder-only | 0.2–0.4% |
| 540B | Flan-U-PaLM | decoder-only | 0.2% |

**Finetuning = tiny fraction pre-training compute.** Например, Flan-PaLM 540B: ~512 v4 TPU chips, **37 часов** (0.2% от pre-training compute).

Процедура: constant LR, Adafactor optimizer, packing (несколько примеров в одну последовательность с EOS separator, masking across boundaries). Одиночный checkpoint для всех evaluations.

## Key Results

### Scaling: Tasks & Model Size (Table 3, Figure 4)

Нормализованный avg по MMLU+BBH+TyDiQA+MGSM:

**540B модель:**
- No finetuning: **49.1%**
- +9 CoT tasks: 52.6% (+3.5%)
- +89 tasks: 57.0% (+7.9%)
- +282 tasks: 57.5% (+8.4%)
- **+1,836 tasks: 58.5% (+9.4%)**

**Ключевые наблюдения:**
1. Основной прирост приходится на первые 282 задачи; после — небольшой incremental gain
2. Scale модели (8B→62B→540B) значительно улучшает результаты
3. CoT tasks (9 датасетов) дают большой прирост на CoT-evaluations при минимальном числе задач

### MMLU: State of the Art (Table 1)

| Модель | MMLU (5-shot) |
|--------|---------------|
| GPT-3 5-shot | 43.9% |
| Chinchilla 5-shot | 67.6% |
| PaLM 5-shot | 69.3% |
| **Flan-PaLM 5-shot** | **72.2%** |
| **Flan-PaLM CoT + SC** | **75.2% ← SOTA** |
| Human expert | 89.8% |

Flan-PaLM 540B с CoT+Self-Consistency: **75.2%** — SOTA на момент публикации (Oct 2022).

### BBH (BIG-Bench Hard)

| Модель | BBH Direct | BBH CoT |
|--------|-----------|---------|
| PaLM 540B | 49.1% | 63.7% |
| **Flan-PaLM 540B** | **58.8%** | **65.6%** |

Flan-PaLM улучшает PaLM на BBH при обоих prompting режимах.

### TyDiQA (Multilingual)

| Модель | TyDiQA Direct (1-shot) |
|--------|------------------------|
| PaLM 62B | 52.9% |
| **Flan-PaLM 540B** | **67.4%** |
| Absolute improvement | **+14.5%** |

### MGSM (Math in 10 languages)

| Модель | MGSM CoT |
|--------|----------|
| PaLM 540B | 45.9% |
| **Flan-PaLM 540B** | **61.3%** |
| Improvement | **+15.4 ppts** |

### CoT vs no-CoT Finetuning (Section 4)

Критический результат: instruction finetuning **без CoT данных** значительно деградирует CoT performance. Добавление всего **9 CoT датасетов** решает эту проблему — CoT performance улучшается на обоих: CoT-evaluation (+3.5% при 9 CoT tasks) и non-CoT evaluations (прирост умеренный, не деградация).

Это означает: instruction finetuning и CoT prompting **синергируют**, но только если в finetuning data включены CoT примеры.

### Flan-T5 vs PaLM 62B (Table 5)

Flan-T5-XL (3B) превосходит PaLM 62B на нескольких challenging BIG-Bench задачах. Flan-T5-XXL (11B) превосходит PaLM 62B по средним метрикам.

**Flan-T5 outperforms T5 on double-digit improvements** на всех бенчмарках.

### Responsible AI

Instruction finetuning также улучшает performance на RAI бенчмарках (BBQ bias, ToxiGen). Flan-PaLM реже генерирует токсичный контент при zero-shot.

## My notes

- Flan-T5 — **стандартный открытый encoder-decoder baseline** для NLU/seq2seq задач в 2022–2024. Его checkpoints используются в огромном числе работ.
- **"9 CoT datasets are enough"** — ключевой практический вывод. Не нужны тысячи CoT примеров, достаточно 9 хорошо охватывающих датасетов.
- **0.2% compute** — instruction finetuning дёшево. Это делает Flan подход очень доступным.
- Scaling plateau: после 282 задач прирост незначительный. Важнее разнообразие задач, а не их количество.
- Работа вдохновила Alpaca (instruction-tuning LLaMA) и Vicuna — они использовали Flan методологию на open-source моделях.
- MMLU 75.2% (с CoT+SC) был SOTA несколько месяцев, пока GPT-4 не установил 86.4%.
