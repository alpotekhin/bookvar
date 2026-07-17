---
title: "Toolformer: Language Models Can Teach Themselves to Use Tools"
url: https://arxiv.org/abs/2302.04761
authors: [Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom]
year: 2023
date_reviewed: 2026-04-07
type: paper-review
category: paper
tags:
  - Tools
  - Agents
  - LLM
  - Self-Supervised
Date: 2023-02-09
Organization: Meta AI Research
Status: Done
concepts:
  - "[[02 Areas/ML & DL/Concepts/Agents/Tool Use|Tool Use]]"
  - "[[02 Areas/ML & DL/Concepts/Inference/In-context Learning|In-context Learning]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Self-Supervised Learning|Self-Supervised Learning]]"
raw: "[[02 Areas/ML & DL/raw/papers/toolformer/paper.txt]]"
---

# Toolformer: Language Models Can Teach Themselves to Use Tools

**Authors:** Timo Schick, Jane Dwivedi-Yu, Roberto Dessi et al.
**Published:** 2023 (arXiv:2302.04761)
**URL:** https://arxiv.org/abs/2302.04761

## TL;DR

Toolformer обучает LLM (GPT-J 6.7B) самостоятельно решать, **когда и какой инструмент вызвать**, какие аргументы передать и как использовать результат — полностью self-supervised, без задачеспецифичных аннотаций. Модель обучается на текстовом корпусе, аугментированном API-вызовами, которые сама модель сгенерировала и отфильтровала по критерию "помогает ли вызов предсказывать следующие токены". Toolformer 6.7B превосходит GPT-3 175B на math benchmarks и LAMA.

## Problem

LLM имеют фундаментальные ограничения, которые не решаются простым масштабированием:
- Неспособность к точным арифметическим вычислениям
- Нет доступа к актуальной информации (hallucination)
- Неосведомленность о текущей дате/времени
- Плохая работа с low-resource языками

Существующие подходы к tool use:
1. Требуют **массивных человеческих аннотаций** (Komeili et al., Thoppilan et al.)
2. Ограничены **конкретными задачами** с task-specific промптами (Gao et al., Parisi et al.)

## Method

### 3-шаговый pipeline

1. **Sample API calls:** Для каждой позиции в тексте модель M (GPT-J) с in-context промптом генерирует кандидатные API-вызовы. Используется порог τ_s для выбора позиций
2. **Execute API calls:** Вызовы выполняются соответствующими инструментами (другие нейросети, Python-скрипты, retrieval-системы)
3. **Filter API calls:** Оставляем только те вызовы, где L^- - L^+ >= τ_f, т.е. API-вызов с результатом **снижает loss на следующих токенах** по сравнению с отсутствием вызова или вызовом без результата

### API-вызовы в тексте

Линеаризованный формат: `<API> ac(ic) -> r </API>`

Пример: "Out of 1400 participants, 400 (or `[Calculator(400/1400) -> 0.29]` 29%) passed the test."

### Инструменты

| API | Описание | Модель/Система |
|-----|----------|---------------|
| QA | Factoid Q&A | Atlas (retrieval-augmented LM) |
| Calculator | Арифметика (+, -, *, /) | Python |
| WikiSearch | Поиск по Wikipedia | BM25 retriever (KILT) |
| MT | Перевод на английский | NLLB 600M (200 языков) |
| Calendar | Текущая дата | Системный вызов |

### Fine-tuning

Модель M дообучается на C* (корпус с вставленными API-вызовами) стандартным LM objective. Критически: C* содержит **те же тексты**, что и оригинальный C, только с добавленными API-вызовами — модель не теряет general language modeling abilities.

### Инференс

Модифицированный greedy decoding: `<API>` генерируется не только когда это top-1 токен, но когда входит в top-k (k=10). Максимум один API-вызов на вход.

## Key Results

### LAMA (factual knowledge)

| Модель | SQuAD | Google-RE | T-REx |
|--------|-------|-----------|-------|
| GPT-J | 17.8 | 4.9 | 31.9 |
| **Toolformer** | **33.8** | **11.5** | **53.5** |
| GPT-3 (175B) | 26.8 | 7.0 | 39.8 |

### Math (ASDiv / SVAMP / MAWPS)

| Модель | ASDiv | SVAMP | MAWPS |
|--------|-------|-------|-------|
| GPT-J | 7.5 | 5.2 | 9.9 |
| **Toolformer** | **40.4** | **29.4** | **44.0** |
| GPT-3 (175B) | 14.0 | 10.0 | 19.8 |

Toolformer использует калькулятор в 97.9% math-примеров. Модель в 6.7B параметров **в 3 раза превосходит GPT-3 175B** на math tasks.

### QA (WebQS / NQ / TriviaQA)

Toolformer превосходит все модели того же размера, но уступает GPT-3 175B. Причина: простота search engine (BM25 без переформулирования запросов).

### Language Modeling

Perplexity на WikiText/CCNet **не ухудшается** при добавлении API-вызовов — модель сохраняет core LM abilities.

### Scaling Laws

Способность эффективно использовать инструменты **эмерджентно появляется** на ~775M параметров. Меньшие модели не извлекают пользы из API-вызовов.

## My notes

- Ключевая инновация — self-supervised критерий фильтрации: L^- - L^+ >= τ_f. Модель сама определяет, какие API-вызовы ей полезны. Человеку не нужно указывать, когда использовать инструмент
- Ограничения текущей реализации: max 1 API-вызов на вход; нет цепочек вызовов (calendar -> QA); нет переформулирования запросов
- Toolformer (disabled) уже лучше GPT-J на math — dine-tuning на примерах с API-вызовами улучшает внутренние math capabilities модели
- Шум в фильтрации полезен: заставляет модель не слепо доверять результатам API
- Связь с [[02 Areas/ML & DL/Papers/ReAct|ReAct]]: ReAct через промптинг, Toolformer через обучение. ReAct позволяет цепочки вызовов и рефлексию, Toolformer — нет
- Для production: подход можно обобщить на любые API с текстовым I/O — потенциально очень мощная парадигма для интеграции LLM с внешними системами
