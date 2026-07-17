---
title: "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
url: https://arxiv.org/abs/2305.10601
authors: [Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, Karthik Narasimhan]
year: 2023
date_reviewed: 2026-04-07
type: paper-review
category: paper
tags:
  - Reasoning
  - Search
  - LLM
  - Planning
Date: 2023-12-03
Organization: Princeton University, Google DeepMind
Status: Done
concepts:
  - "[[02 Areas/ML & DL/Concepts/Inference/Chain of Thought|Chain of Thought]]"
  - "[[02 Areas/ML & DL/Concepts/Inference/Tree of Thoughts|Tree of Thoughts]]"
  - "[[02 Areas/ML & DL/Concepts/Inference/Prompting|Prompting]]"
  - "[[02 Areas/ML & DL/Concepts/Agents/Self-Evaluation|Self-Evaluation]]"
raw: "[[02 Areas/ML & DL/raw/papers/tree-of-thoughts/paper.txt]]"
---

# Tree of Thoughts: Deliberate Problem Solving with Large Language Models

**Authors:** Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, Karthik Narasimhan
**Published:** 2023 (NeurIPS 2023; arXiv:2305.10601v2)
**URL:** https://arxiv.org/abs/2305.10601

## TL;DR

Tree of Thoughts (ToT) — фреймворк для инференса LLM, который обобщает Chain of Thought: вместо одного линейного пути рассуждения модель **исследует дерево мыслей**, где каждый узел — промежуточное состояние решения. LLM сама **генерирует** кандидатные мысли, **оценивает** их перспективность и **использует алгоритмы поиска** (BFS/DFS) с lookahead и backtracking. На Game of 24 ToT решает 74% задач vs 4% у CoT (GPT-4).

## Problem

LLM ограничены **token-level left-to-right** генерацией — они принимают решения последовательно, без возможности:
1. **Исследовать альтернативы** — нет разветвлений в процессе рассуждения
2. **Планировать наперед** — нет lookahead
3. **Возвращаться назад** — нет backtracking при ошибке

Аналогия с dual process theory: стандартный LLM inference = System 1 (быстрый, автоматический). Нужен System 2 (медленный, deliberate).

Существующие подходы:
- **CoT** — один линейный путь, нет exploration
- **CoT-SC** — множество путей, но нет локальной exploration и backtracking; работает только для задач с ограниченным answer space

## Method

### Формализация

ToT фреймирует задачу как **поиск по дереву**:
- **Состояние** s = [x, z_1...i] — вход + последовательность мыслей
- **Мысль** z — когерентная языковая последовательность (от пары слов до параграфа)

4 ключевых вопроса:

### 1. Thought decomposition
Как разбить задачу на промежуточные шаги. Зависит от задачи:
- Game of 24: одно уравнение = одна мысль
- Creative Writing: план текста = одна мысль
- Crosswords: одно слово = одна мысль

### 2. Thought generator G(p_θ, s, k)
Два подхода для генерации k кандидатов:
- **i.i.d. sampling** из CoT prompt — для богатых пространств мыслей (параграфы)
- **Propose prompt** — для ограниченных пространств (слово, уравнение), чтобы избежать дупликатов

### 3. State evaluator V(p_θ, S)
Два подхода для оценки:
- **Value** — независимая оценка каждого состояния (sure/likely/impossible)
- **Vote** — сравнение состояний между собой (какое самое перспективное)

LLM сама выступает эвристикой для поиска — это новый подход по сравнению с programmed (DeepBlue) или learned (AlphaGo) эвристиками.

### 4. Search algorithm
- **BFS** — для неглубоких деревьев (Game of 24: depth 3, breadth 5)
- **DFS** — для глубоких деревьев с pruning (Crosswords: depth 10)

IO, CoT, CoT-SC — **частные случаи ToT** (деревья с ограниченной глубиной и шириной).

## Key Results

### Game of 24

| Метод | Success Rate |
|-------|-------------|
| IO prompt | 7.3% |
| CoT prompt | 4.0% |
| CoT-SC (k=100) | 9.0% |
| IO + Refine (k=10) | 27% |
| CoT (best of 100) | 49% |
| **ToT (b=1)** | **45%** |
| **ToT (b=5)** | **74%** |

60% CoT-сэмплов ошибаются уже на первом шаге (первые 3 слова). Left-to-right decoding — критическое ограничение.

### Creative Writing

- GPT-4 coherency scores: IO 6.19, CoT 6.93, **ToT 7.56**
- Human evaluation: ToT > CoT в 41/100 случаев, CoT > ToT в 21/100

### Mini Crosswords (5x5)

| Метод | Letter % | Word % | Game % |
|-------|----------|--------|--------|
| IO | 38.7 | 14 | 0 |
| CoT | 40.6 | 15.6 | 1 |
| **ToT** | **78** | **60** | **20** |

DFS с pruning критичен: без pruning performance падает, без backtrack — до 20% words.

## My notes

- ToT — концептуально красивое обобщение: IO/CoT/CoT-SC/Self-Refine = частные случаи деревьев с разными ограничениями на depth и breadth
- Ключевое ограничение: **стоимость inference**. Для Game of 24 с b=5 каждый шаг = 5 генераций + 15 оценок (5 thoughts * 3 samples). Это в ~100x дороже одного CoT
- LLM как heuristic для поиска — элегантно, но неидеально: в Crosswords оценщик иногда отвергает правильные слова (редкие/устаревшие, которые GPT-4 не знает)
- Практическая применимость ограничена задачами, где: (a) можно чётко decompose мысли, (b) LLM может оценить промежуточное состояние, (c) стоимость inference оправдана
- Связь с [[02 Areas/ML & DL/Papers/Self-Consistency|Self-Consistency]]: CoT-SC = tree с breadth k и depth 1, агрегация через majority vote. ToT обобщает на произвольную глубину и ширину
- Направление развития: MCTS (как в AlphaGo) вместо простых BFS/DFS, learned value functions вместо LLM-based evaluation
