---
title: "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
url: https://arxiv.org/abs/2203.11171
authors: [Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, Denny Zhou]
year: 2023
date_reviewed: 2026-04-07
type: paper-review
category: paper
tags:
  - Reasoning
  - Decoding
  - LLM
Date: 2023-03-07
Organization: Google Research Brain
Status: Done
concepts:
  - "[[02 Areas/ML & DL/Concepts/Inference/Chain of Thought|Chain of Thought]]"
  - "[[02 Areas/ML & DL/Concepts/Inference/Self-Consistency|Self-Consistency]]"
  - "[[02 Areas/ML & DL/Concepts/Inference/Prompting|Prompting]]"
  - "[[02 Areas/ML & DL/Concepts/Inference/Decoding Strategies|Decoding Strategies]]"
raw: "[[02 Areas/ML & DL/raw/papers/self-consistency/paper.txt]]"
---

# Self-Consistency Improves Chain of Thought Reasoning in Language Models

**Authors:** Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, Denny Zhou
**Published:** 2023 (ICLR 2023; arXiv:2203.11171v4)
**URL:** https://arxiv.org/abs/2203.11171

## TL;DR

Self-Consistency — decoding strategy для CoT prompting: вместо одного greedy path модель **сэмплирует множество reasoning paths**, а финальный ответ определяется **majority vote**. Идея: правильные рассуждения, даже разные, чаще сходятся к одному ответу, чем неправильные. Метод unsupervised, не требует обучения. На GSM8K: +17.9%, SVAMP: +11.0%, AQuA: +12.2% абсолютного прироста accuracy на PaLM-540B.

## Problem

CoT prompting значительно улучшает reasoning в LLM, но использует **greedy decoding** — один детерминированный путь рассуждения. Проблемы:

1. Greedy decoding склонен к **repetitiveness** и **local optima**
2. Одна ошибка в reasoning chain ведет к неправильному ответу
3. Для сложных задач обычно существует **множество валидных путей** к правильному ответу

Альтернативы (verifier/re-ranker) требуют **дополнительного обучения** и аннотаций.

## Method

### Три шага

1. **Prompt:** Стандартный CoT prompt с manually written exemplars
2. **Sample:** Сэмплировать m diverse reasoning paths из decoder (temperature sampling, top-k, nucleus sampling)
3. **Marginalize:** Маргинализовать по reasoning paths, выбрать ответ с **majority vote**:

```
arg max_a Σ_{i=1}^{m} 1(a_i = a)
```

### Агрегация ответов

Авторы сравнивают несколько стратегий на PaLM-540B:

| Стратегия | GSM8K | MultiArith |
|-----------|-------|------------|
| Greedy decode | 56.5 | 94.7 |
| Weighted avg (unnormalized) | 56.3 | 90.5 |
| **Weighted sum (normalized)** | **74.1** | **99.3** |
| **Majority vote (unweighted)** | **74.4** | **99.3** |

Majority vote и normalized weighted sum дают практически одинаковые результаты — нормализованные вероятности разных paths очень близки друг к другу.

### Свойства

- **Unsupervised** — не нужны аннотации, дополнительные модели или fine-tuning
- **Self-ensemble** — работает поверх одной модели (не model ensemble)
- Применима только к задачам с **фиксированным answer space** (но в принципе расширяема через метрику согласованности)

## Key Results

### Arithmetic Reasoning

| Модель | Метод | GSM8K | SVAMP | AQuA |
|--------|-------|-------|-------|------|
| PaLM-540B | CoT | 56.5 | 79.0 | 35.8 |
| PaLM-540B | **Self-Consistency** | **74.4 (+17.9)** | **86.6 (+7.6)** | **48.3 (+12.5)** |
| GPT-3 (code-002) | CoT | 60.1 | 75.8 | 39.8 |
| GPT-3 (code-002) | **Self-Consistency** | **78.0 (+17.9)** | **86.8 (+11.0)** | **52.0 (+12.2)** |

Self-Consistency с PaLM-540B/GPT-3 достигает **новый SOTA** на почти всех задачах — без fine-tuning, обгоняя подходы с обучением verifier на тысячах примеров.

### Commonsense Reasoning

| Модель | Метод | StrategyQA | ARC-c |
|--------|-------|------------|-------|
| PaLM-540B | CoT | 75.3 | 85.2 |
| PaLM-540B | **SC** | **81.6 (+6.3)** | **88.7 (+3.5)** |

### Масштабирование

- Прирост **увеличивается** с ростом модели: +3-6% на UL2-20B, +9-23% на LaMDA-137B
- 40 sampled paths оптимальны; дальнейшее увеличение дает diminishing returns
- Робастна к sampling strategy (temperature, top-k, nucleus) и imperfect prompts

### Self-Consistency vs другие подходы

| Подход | GSM8K (LaMDA-137B) |
|--------|-------------------|
| CoT | 17.1 |
| Sample-and-rank (40 paths) | ~19 |
| Ensemble (40 prompt permutations) | 19.2 |
| Ensemble (3 sets of prompts) | 18.6 |
| **Self-Consistency (40 paths)** | **27.7** |

SC значительно превосходит sample-and-rank, beam search и prompt ensembles.

### Дополнительные результаты

- **CoT hurts performance:** На некоторых NLP задачах (ANLI-R1, e-SNLI, RTE) CoT ухудшает результаты vs standard prompting. Self-Consistency робастно восстанавливает и превосходит baseline
- **Uncertainty estimation:** Уровень согласованности (% paths, сходящихся к одному ответу) **сильно коррелирует с accuracy** — модель может "знать, когда не знает"
- **Zero-shot CoT:** SC работает и с zero-shot CoT (Kojima et al.) — +26.2% на GSM8K с PaLM-540B

## My notes

- Удивительно простая и мощная идея. "Sample diverse paths + majority vote" — это по сути **Monte Carlo approximation** правильного ответа через marginalization
- Принципиальное наблюдение: правильные рассуждения сходятся, неправильные — расходятся. Это интуитивно верно и хорошо работает эмпирически
- Стоимость: 40x inference cost для 40 paths. Для production: можно начинать с малого k и останавливаться при высокой consistency
- Weighted sum (normalized) ~ majority vote потому что LLM плохо калиброваны: нормализованные вероятности разных paths примерно равны
- Связь с [[02 Areas/ML & DL/Papers/Tree of Thoughts|Tree of Thoughts]]: CoT-SC = ToT с breadth k, depth 1, majority vote агрегацией. ToT обобщает, добавляя промежуточную evaluation и search
- Связь с [[02 Areas/ML & DL/Papers/COT|CoT]]: Self-Consistency — прямое улучшение CoT decoding strategy, полностью совместимое с любыми CoT промптами
