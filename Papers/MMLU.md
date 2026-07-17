---
title: "Measuring Massive Multitask Language Understanding"
url: https://arxiv.org/abs/2009.03300
authors: [Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt]
year: 2021
date_reviewed: 2026-04-07
type: source-note
status: legacy
category: paper
tags:
  - Benchmark
  - Evaluation
  - LLM
Date: 2021-01-12
Organization: UC Berkeley, Columbia University, UChicago, UIUC
Status: Done
concepts:
  - "[[02 Areas/ML & DL/Concepts/Evaluation/MMLU|MMLU]]"
  - "[[02 Areas/ML & DL/Concepts/Evaluation/Benchmarks|Benchmarks]]"
  - "[[02 Areas/ML & DL/Concepts/Inference/Few-shot Learning|Few-shot Learning]]"
  - "[[02 Areas/ML & DL/Concepts/Inference/Zero-shot Learning|Zero-shot Learning]]"
raw: "[[02 Areas/ML & DL/raw/papers/mmlu/paper.txt]]"
---

# Measuring Massive Multitask Language Understanding (MMLU)

**Authors:** Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt
**Published:** 2021 (ICLR 2021; arXiv:2009.03300v3)
**URL:** https://arxiv.org/abs/2009.03300

## TL;DR

MMLU — бенчмарк из **57 задач** (15,908 вопросов multiple choice) по STEM, гуманитарным, социальным наукам и другим областям, от elementary до professional уровня. Тестирует **знания и problem-solving**, полученные при pretraining. На момент публикации GPT-3 175B достигает 43.9% (random = 25%), UnifiedQA 11B — 48.9%. Expert-level accuracy оценивается в ~89.8%. Ни одна модель не достигает экспертного уровня ни в одной из 57 задач.

## Problem

Существующие NLP бенчмарки имеют критические недостатки:

1. **GLUE / SuperGLUE** — сверхчеловеческий уровень достигнут за ~1 год; тестируют лингвистические навыки, а не понимание
2. **Commonsense benchmarks** (HellaSwag, PIQA, CosmosQA) — оценивают базовые способности, которыми обладает почти каждый ребенок
3. **QA benchmarks** (ARC, OpenBookQA) — покрывают простые темы школьного уровня

Проблема: модели видят огромный объём специализированных знаний при pretraining (Wikipedia, учебники, веб), но **бенчмарки не тестируют усвоение этих знаний**.

## Method

### Дизайн бенчмарка

- **57 задач** по 4 категориям: Humanities, Social Science, STEM, Other
- **Multiple choice** (4 варианта) — простая автоматическая оценка
- **Источники:** практические экзаменационные вопросы (GRE, USMLE, bar exam), вопросы из university courses, книги OUP
- **Уровни сложности:** Elementary, High School, College, Professional
- **15,908 вопросов:** 5 few-shot dev на предмет, 1,540 validation, 14,079 test (min 100 на предмет)

### Оценка

- **Zero-shot и few-shot** (до 5 примеров) — без fine-tuning на train set
- Промпт: "The following are multiple choice questions (with answers) about [subject]."
- Модель генерирует вероятности для токенов "A", "B", "C", "D"

### Человеческий baseline

- Amazon Mechanical Turk (неспециалисты): 34.5%
- Expert-level (95-й перцентиль реальных экзаменов): ~89.8%

## Key Results

### Основные результаты

| Модель | Humanities | Social Sci | STEM | Other | Avg |
|--------|-----------|------------|------|-------|-----|
| Random | 25.0 | 25.0 | 25.0 | 25.0 | 25.0 |
| GPT-2 | 32.8 | 33.3 | 30.2 | 33.1 | 32.4 |
| UnifiedQA 11B | 45.6 | 56.6 | 40.2 | 54.6 | 48.9 |
| GPT-3 Small (2.7B) | 24.4 | 30.9 | 26.0 | 24.1 | 25.9 |
| GPT-3 Medium (6.7B) | 26.1 | 21.6 | 25.6 | 25.5 | 24.9 |
| GPT-3 Large (13B) | 27.1 | 25.6 | 24.3 | 26.5 | 26.0 |
| **GPT-3 X-Large (175B)** | **40.8** | **50.4** | **36.7** | **48.8** | **43.9** |

### Ключевые наблюдения

1. **Порог масштаба:** GPT-3 до 13B = random chance (~25%). Только 175B выходит за случайный уровень — **эмерджентное** появление multitask knowledge
2. **Fine-tuning помогает:** UnifiedQA 11B (fine-tuned на QA datasets) = 48.9% vs GPT-3 175B few-shot = 43.9%, при 15x меньше параметров
3. **Lopsided performance:** GPT-3 от 69% (US Foreign Policy) до 26% (College Chemistry). Модели не достигают экспертного уровня ни в одном предмете
4. **Procedural vs declarative knowledge:** GPT-3 знает PEMDAS (порядок операций), но не может его последовательно применять. 9 из 10 худших предметов — STEM с вычислениями
5. **Нелогичный порядок обучения:** GPT-3 лучше на College Medicine (47.4%) чем на Elementary Mathematics (29.9%)
6. **Калибрация:** GPT-3 **плохо калиброван** — разница между confidence и accuracy до 24%. Модель не знает, чего не знает

### Проблемные области

Особенно низкие результаты на:
- **Moral Scenarios** — важно для alignment
- **Professional Law** — критично для следования правилам
- **Elementary Mathematics** — требует procedural knowledge

Попытка fine-tune RoBERTa на ~2000 law-примеров + pretrain на 1.6M legal case summaries дала только 36.1% — дополнительные данные помогают, но недостаточно.

## My notes

- MMLU стал **де-факто стандартом** оценки LLM. Почти каждая новая модель сообщает MMLU score. К 2026 году GPT-4, Claude, Gemini значительно превзошли 90% average
- Бенчмарк оказался точен в предсказании: multitask knowledge — ключевое свойство, развившееся с масштабированием
- Методологическое новшество: тестирование в zero/few-shot setting без train set. Это предвосхитило парадигму "pretraining IS training"
- Слабость бенчмарка: multiple choice формат. Модели могут использовать elimination strategy, surface cues и т.д. Open-ended evaluation (как в [[02 Areas/ML & DL/Papers/HumanEval|HumanEval]]) более информативна
- Калибрация остается проблемой: даже современные модели плохо знают, чего не знают
- MMLU теперь "решен" (>90% для frontier models), что привело к созданию MMLU-Pro, GPQA и других более сложных бенчмарков
