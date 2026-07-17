---
title: "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
url: https://arxiv.org/abs/2201.11903
authors: [Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, Denny Zhou]
year: 2022
date_reviewed: 2026-04-06
type: paper-review
category: paper
tags:
  - Feature
  - LLM
Date: 2022-01-01
Organization: Google
Status: Done
concepts:
  - "[[02 Areas/ML & DL/Concepts/Inference/Chain of Thought|Chain of Thought]]"
  - "[[02 Areas/ML & DL/Concepts/Inference/Prompting|Prompting]]"
  - "[[02 Areas/ML & DL/Concepts/Inference/Zero-shot Learning|Zero-shot Learning]]"
  - "[[02 Areas/ML & DL/Concepts/Inference/In-context Learning|In-context Learning]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/Emergent Abilities|Emergent Abilities]]"
raw: "[[02 Areas/ML & DL/raw/papers/cot/paper.txt]]"
---

# Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

**Authors:** Jason Wei, Xuezhi Wang, Dale Schuurmans et al. (Google Research, Brain Team)
**Published:** 2022 (NeurIPS 2022; arXiv:2201.11903v6)
**URL:** https://arxiv.org/abs/2201.11903

## TL;DR

Chain-of-Thought (CoT) Prompting — техника, при которой в few-shot exemplars добавляются **промежуточные шаги рассуждения** (chain of thought) перед ответом. Без переобучения, только через изменение prompts, PaLM 540B достигает SOTA на GSM8K (math word problems), превосходя fine-tuned GPT-3. Ключевые открытия: CoT является **эмерджентной способностью**, проявляющейся только у моделей ≥100B параметров, и особенно полезна для многошаговых задач.

## Problem

Масштабирование LLM улучшает множество задач, но **сложные reasoning задачи** (математика, multi-hop commonsense, символьные манипуляции) остаются труднодостижимы. Два существующих подхода имеют ограничения:

1. **Rationale-augmented training/finetuning** — требует дорогостоящих высококачественных аннотаций с пошаговыми объяснениями
2. **Standard few-shot prompting** (Brown et al., GPT-3) — плохо работает на reasoning задачах; не улучшается с масштабом

Нужен подход, который сочетает силу few-shot промптинга с пошаговым рассуждением без дорогостоящего fine-tuning.

## Method

### Chain-of-Thought Prompting

Идея проста: вместо exemplars вида `(вопрос, ответ)`, использовать `(вопрос, шаги рассуждения, ответ)`.

**Standard prompting:**
```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
   Each can has 3 tennis balls. How many tennis balls does he have now?
A: The answer is 11.
```

**Chain-of-Thought prompting:**
```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
   Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls.
   5 + 6 = 11. The answer is 11.
```

**Промпт:** 8 exemplars с chains of thought, написанных вручную одним аннотатором (Annotator A). Один и тот же набор использовался для всех math benchmarks (кроме AQuA — 4 exemplars).

### Модели

Эксперименты на 5 семействах LLM:
- GPT-3: text-ada (350M), babbage (1.3B), curie (6.7B), **text-davinci (175B)**
- LaMDA: 422M, 2B, 8B, 68B, **137B**
- **PaLM**: 8B, 62B, **540B**
- UL2 20B
- Codex (code-davinci-002)

Greedy decoding (температура 0).

## Key Results

### Arithmetic Reasoning (Section 3)

**GSM8K** (math word problems, Figure 4):

| Модель | Standard | Chain-of-Thought |
|--------|----------|-----------------|
| LaMDA 137B | 18% | 33% |
| GPT-3 175B | 15% | 46% |
| Finetuned GPT-3+verifier | – | 55% |
| **PaLM 540B** | 33% | **57%** ← **новый SOTA** |

PaLM 540B+CoT **обходит fine-tuned GPT-3 с верификатором** (55%), используя только 8 промпт-примеров.

**Другие math benchmarks:**

| Датасет | GPT-3 CoT | PaLM CoT |
|---------|-----------|----------|
| SVAMP | 63% | 79% |
| MAWPS | 92% | 93% |
| ASDiv | 71% | 80% |

### Commonsense Reasoning (Section 4, Figure 7)

PaLM 540B + CoT:
- **StrategyQA: 75.6%** (prior SOTA: 69.4%) — превышает SOTA
- **Sports Understanding: 95.4%** (unaided sports enthusiast: 84%) — превышает human baseline
- **Date Understanding:** значительное улучшение

### Symbolic Reasoning (Section 5)

**Last Letter Concatenation** (in-domain, 2 слова): PaLM 540B+CoT 99%, standard 47%

**Out-of-Distribution** (4+ слова, не видел в exemplars):
- PaLM 540B+CoT: **67%** (standard: 10%)
- CoT демонстрирует **compositional generalization** — переносит паттерн рассуждения на более длинные входы

**Coin Flip** (state tracking): PaLM 540B+CoT 97.5%, standard 33.8%

### Ключевые наблюдения

**1. Эмерджентность (Figure 4):**
- CoT НЕ помогает при размере < 100B параметров — маленькие модели генерируют "беглые, но нелогичные" chains of thought
- Резкое улучшение начинается с ~100B параметров — это [[02 Areas/ML & DL/Concepts/NLP/Emergent Abilities|Emergent Abilities]]

**2. Больший выигрыш на сложных задачах:**
- GSM8K (самый сложный, низкий baseline) — самый большой прирост (например, GPT-3: 15%→46%)
- SingleOp (1-шаговые задачи) — минимальный или нулевой прирост

## Ablation Studies (Section 3.3, Figure 5)

Контрольные эксперименты на GSM8K (LaMDA 137B + PaLM 540B):

| Вариант | Результат |
|---------|-----------|
| Standard prompting | baseline |
| **Chain-of-thought** | **+значительный прирост** |
| Equation only | ~= baseline |
| Variable compute only (...) | ~= baseline |
| Chain of thought after answer | ~= baseline |

**Вывод:** Улучшение от CoT не объясняется:
- только добавлением математического уравнения
- только дополнительными токенами (variable compute)
- только активацией знаний (chain after answer → не помогает)

Именно **последовательное рассуждение до ответа** является ключевым.

## Robustness (Section 3.4, Figure 6)

- Три разных аннотатора написали независимые chains of thought → все значительно лучше baseline, несмотря на стилистические различия
- Concise vs verbose styles — оба работают
- Случайные exemplars из GSM8K training set → сопоставимо с ручными exemplars
- Разные порядки exemplars — устойчиво

CoT не зависит от конкретного лингвистического стиля.

## Error Analysis (Section 3.2)

Из 50 верных ответов LaMDA 137B на GSM8K:
- 48/50 chains of thought логически и математически корректны
- 2/50 пришли к верному ответу случайно

Из 50 неверных ответов:
- 46% chains of thought — почти верные (малые ошибки: calculator error, symbol mapping, один пропущенный шаг)
- 54% — серьёзные ошибки в semantic understanding или coherence

PaLM 62B→540B: scaling исправляет большую часть "missing step" и "semantic understanding" ошибок.

## My notes

- CoT — один из самых влиятельных промпт-инжиниринговых результатов. Открыл "zero-shot CoT" (Kojima et al., "Let's think step by step") и majority voting / self-consistency (Wang et al., 2022).
- **Эмерджентность** CoT как функция scale — важнейший практический вывод: 7B модели не получат CoT-бенефит так же, как 70B+ модели.
- Ключевая инсайт: intermediate steps = дополнительное compute + структурирование задачи. Это предшественник "thinking tokens" (OpenAI o1, DeepSeek-R1).
- CoT является few-shot методом — но существует также zero-shot CoT: просто добавление "Let's think step by step" в промпт (Kojima et al., 2022). Это отдельная работа.
- Для простых (1-2 шага) задач — CoT не помогает и может ухудшать (overhead на генерацию лишних токенов).
