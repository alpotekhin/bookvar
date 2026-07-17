---
title: "PaLM 2 Technical Report"
url: https://arxiv.org/abs/2305.10403
authors: [Google]
year: 2023
date_reviewed: 2026-04-06
type: paper-review
category: paper
tags:
  - LLM
  - arch
Date: 2023-01-05
Organization: Google
Parent item:
  - "[[02 Areas/ML & DL/Papers/Flan-T5-PaLM|PaLM / Flan-PaLM]]"
Status: Done
concepts:
  - "[[02 Areas/ML & DL/Concepts/Architectures/PaLM 2|PaLM 2]]"
  - "[[02 Areas/ML & DL/Concepts/Architectures/Decoder-only|Decoder-only]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Scaling Laws|Scaling Laws]]"
  - "[[02 Areas/ML & DL/Concepts/Inference/Chain of Thought|Chain of Thought]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Instruction Tuning|Instruction Tuning]]"
raw: "[[02 Areas/ML & DL/raw/papers/palm-2/paper.txt]]"
---

# PaLM 2 Technical Report

**Authors:** Google (large number of contributors)
**Published:** 2023 (arXiv:2305.10403v3, Sep 2023)
**URL:** https://arxiv.org/abs/2305.10403

## TL;DR

PaLM 2 — преемник PaLM от Google, улучшенный по трём ключевым осям: **compute-optimal scaling** (Chinchilla-style 1:1 data/model), **более мультилингвальный и диверсный датасет** (сотни языков), **смешанный pre-training objective** (не только CLM). Ключевые результаты: улучшение на BIG-Bench Hard (+26% direct, +20% CoT over PaLM), GSM8K competitive с GPT-4 (91.0/92.0), MGSM SOTA 87.0%, прохождение профессиональных языковых экзаменов C2 на всех оцениваемых языках. PaLM 2 меньше по параметрам, чем PaLM, но превосходит его по большинству задач — демонстрируя что data quality и architecture > raw scale.

## Problem

PaLM (540B параметров) следовал старой парадигме: больше параметров → лучше, при этом:
- Тренировочных данных пропорционально меньше (не Chinchilla-optimal)
- Датасет преимущественно английский (~78% non-code текста)
- Единственный pre-training objective (causal LM)

PaLM 2 систематически исправляет все три ограничения, показывая что меньшая модель с лучшими данными и objective может обходить большую.

## Method

### Архитектура

Технический отчёт **намеренно не раскрывает** детали архитектуры и размер PaLM 2. Известно:
- Transformer-based decoder-only model
- Три продуктовых варианта: PaLM 2-S (Small), PaLM 2-M (Medium), PaLM 2-L (Large)
- Coding-specific вариант: PaLM 2-S* (fine-tuned для кода)
- PaLM 2-L значительно **меньше** PaLM-540B, но превосходит его

### Compute-Optimal Scaling (Section 2)

**Scaling law study:** серия IsoFLOP экспериментов (как Chinchilla) на 4 compute budgets (10^19 – 10^22 FLOPs). Quadratic fit оптимальных N (параметры) vs D (токены).

**Вывод:** D и N должны расти пропорционально (1:1), совпадает с Hoffmann et al. (2022). Это подтверждено на ещё более крупных масштабах.

Ключевые значения из Table 1:
- 10^22 FLOPs: оптимум ≈ **10.7B параметров** с ≈10^11 токенов (Chinchilla предсказывает ~10B)
- 10^21 FLOPs: ≈3.35B параметров

**Downstream оговорка:** минимум training loss ≠ лучший downstream. Например, 9.5B модель (минимум loss) немного уступает 16.1B на задачах. Training loss = прокси, но не точно.

### Training Dataset (Section 3)

Значительно более диверсный и мультилингвальный по сравнению с PaLM:
- Web documents, books, code, mathematics, conversational data
- Параллельные данные для сотен языков (source-target pairs с English)
- Специальные control tokens (toxicity markers) для inference-time control
- Canary tokens для memorization measurement
- Дедупликация + PII-фильтрация

**Языки:** top-50 языков по доле перечислены в Table 21. Никакой фильтрации для удаления языков. Больший контекст, чем у PaLM.

### Pre-training Objective

Смесь objectives (инспирировано UL2, Tay et al. 2023) — не только causal LM. Конкретный состав не раскрывается.

## Key Results

### Language Proficiency Exams (Figure 1)

PaLM 2 оценивался на профессиональных экзаменах C2 (CEFR mastery level):

| Язык | PaLM 2 | PaLM | Результат |
|------|--------|------|-----------|
| Chinese (K7-9 Writing) | 82 | 62 | ✓ Pass |
| Chinese (HS Writing) | 81 | 46 | ✓ Pass |
| Japanese (A-level) | 94 | 33 | Pass* (C1) |
| Italian (CELI) | 87 | 70 | ✓ Pass |
| French (TCF) | 82 | 77 | ✓ Pass (C1) |
| French (C2 Writing) | 94 | 83 | ✓ Pass |
| Spanish (DELE) | 67 | 69 | ✓ Pass |

**PaLM 2 проходит все экзамены, PaLM — несколько.**

### English QA & Classification (Table 2, 1-shot)

| Задача | PaLM-540B | PaLM 2-S | PaLM 2-M | PaLM 2-L |
|--------|-----------|----------|----------|----------|
| TriviaQA EM | 81.4 | 75.2 | 81.7 | **86.1** |
| HellaSwag | 83.6 | 82.0 | 84.0 | **86.8** |
| RACE-H | 52.1 | 53.3 | 57.2 | **62.3** |
| ANLI-R1 | 52.6 | 53.1 | 58.1 | **73.1** |
| Average | 70.4 | 69.9 | 72.0 | **76.9** |

PaLM 2-**M** (меньше PaLM-540B) уже **превосходит** PaLM-540B в среднем. PaLM 2-L намного лучше.

### Multilingual QA: TyDi QA (Table 3, 1-shot)

No-context (знание только из параметров):
- PaLM: 31.5 avg → PaLM 2-L: **40.3 avg** (+8.8 ppts)
- Особенно сильный прирост: Telugu +2.6, Swahili +10.6, Indonesian +10.9

### Reasoning (Table 5)

| Задача | SOTA | GPT-4 | PaLM | **PaLM 2** |
|--------|------|-------|------|------------|
| WinoGrande | 87.5 | 87.5 | 85.1 | **90.9** |
| ARC-C | 96.3 | 96.3 | 88.7 | **95.1** |
| DROP | 88.4 | 80.9 | 70.8 | **85.0** |
| StrategyQA | 81.6 | – | 81.6 | **90.4** |
| CSQA | 91.2 | – | 80.7 | **90.4** |
| BIG-Bench Hard | 65.2 | – | 65.2 | **78.1** |

**BIG-Bench Hard (Table 6):** PaLM 2 превосходит PaLM на каждой из 23 задач. Особенно:
- temporal_sequences: PaLM CoT 78.8% → PaLM 2 CoT **100.0%** (+27%)
- multistep_arithmetic: PaLM CoT 19.6% → PaLM 2 CoT **75.6%** (+286%)
- dyck_languages CoT: 28.0% → **63.6%** (+127%)
- Average Direct: 52.3% → 65.7% (+13.4 ppts); Average CoT: 65.2% → 78.1% (+12.9 ppts)

### Math (Table 7)

| Задача | SOTA | GPT-4 | PaLM | PaLM 2 |
|--------|------|-------|------|--------|
| MATH | 50.3 | 42.5 | 8.8 | **48.8/34.3** (SC/direct) |
| GSM8K | 92.0 | 92.0 | 56.5/74.4 | **80.7/91.0** |
| MGSM | 72.0 | – | 45.9/57.9 | **72.2/87.0** |

GSM8K: PaLM 2 competitive с GPT-4 (91.0% vs 92.0%). MGSM: **новый SOTA** 87.0% (предыдущий: 72.0 с SC).

### Coding

PaLM 2-S* (code-specific fine-tune на PaLM 2-S) оценивался на HumanEval и многоязычной оценке. Детальные результаты в Appendix A.4 — отчёт фокусируется на coding как на ключевом use case (GitHub Copilot-like продукты).

## Inference-time Toxicity Control (Section 5.1)

Специальные **control tokens** добавляются в pre-training data (фракция с помеченной токсичностью через Perspective API). При inference: добавление "low toxicity" control token значительно снижает токсичность без ухудшения других способностей — **без дополнительных затрат** (в отличие от RLHF).

## My notes

- PaLM 2 — первый крупный техотчёт Google, где явно применяется **Chinchilla-optimal scaling** для действительно крупных моделей.
- "Меньше, но лучше" — PaLM 2-L < PaLM-540B по параметрам, но превосходит его везде. Это подтверждает тезис LLaMA: inference budget и data quality важнее raw scale.
- **BIG-Bench Hard** стал стандартным бенчмарком для измерения reasoning improvements. Результаты PaLM 2 здесь особенно показательны.
- Control tokens для toxicity — элегантное решение: дёшево, не ломает другие способности.
- Мультилингвальность — отличительная черта PaLM 2 от большинства современников. Параллельные данные для сотен языков → явное преимущество на TyDi QA и MGSM.
- Отчёт намеренно не раскрывает размер модели — тренд после GPT-4 TR на "closed" LLM.
