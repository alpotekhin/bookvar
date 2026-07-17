---
title: "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
url: https://arxiv.org/abs/2310.11511
authors: [Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi]
year: 2023
date_reviewed: 2026-04-07
type: source-note
status: legacy
category: paper
tags:
  - RAG
  - Retrieval
  - Agents
  - Self-Reflection
Date: 2023-10-17
Organization: University of Washington, Allen AI, IBM Research
Status: Done
concepts:
  - "[[02 Areas/ML & DL/Concepts/Retrieval/Retrieval-Augmented Generation|RAG]]"
  - "[[02 Areas/ML & DL/Concepts/Inference/Self-Reflection|Self-Reflection]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Reinforcement Learning from Human Feedback|RLHF]]"
raw: "[[02 Areas/ML & DL/raw/papers/self-rag/paper.txt]]"
---

# Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection

**Authors:** Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi
**Published:** 2023 (Preprint; arXiv:2310.11511)
**URL:** https://arxiv.org/abs/2310.11511

## TL;DR

Self-RAG обучает одну LLM (7B/13B) адаптивно извлекать документы **по требованию**, генерировать ответ и **критиковать собственные выходы** с помощью специальных reflection tokens. В отличие от классического RAG, который всегда извлекает фиксированное количество документов, Self-RAG сама решает, нужен ли retrieval, оценивает релевантность документов и степень поддержки ответа цитатами. Результат: Self-RAG 7B/13B превосходит ChatGPT и retrieval-augmented Llama2-chat на задачах QA, fact verification и long-form generation.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/self-rag/fig1.png]]
*Рисунок 1: Сравнение стандартного RAG (слева) и Self-RAG (справа). Self-RAG адаптивно решает, когда извлекать, обрабатывает несколько документов параллельно и критикует качество генерации.*

## Problem

LLM часто генерируют фактически неверные ответы из-за опоры только на параметрическое знание. Существующие RAG-подходы имеют три ключевые проблемы:

1. **Неразборчивый retrieval** — извлекают фиксированное число документов для всех запросов, даже когда retrieval не нужен (например, для эссе о летнем отпуске)
2. **Нерелевантные документы** — шумные или off-topic пассажи ухудшают качество генерации
3. **Отсутствие проверки** — нет гарантии, что ответ модели согласуется с извлеченными документами; нет атрибуции

## Method

### Reflection Tokens

Self-RAG расширяет словарь LLM четырьмя типами специальных токенов:

| Тип | Вход | Выход | Назначение |
|-----|------|-------|------------|
| **Retrieve** | x или (x, y) | yes / no / continue | Решает, нужен ли retrieval |
| **IsRel** | (x, d) | relevant / irrelevant | Оценивает релевантность документа d |
| **IsSup** | (x, d, y) | fully / partially / no support | Проверяет, поддержан ли ответ документом |
| **IsUse** | (x, y) | 5..1 | Общая полезность ответа |

### Обучение

1. **Critic model C:** Обучается на данных, размеченных GPT-4, предсказывать reflection tokens. Согласие с GPT-4 > 90% на большинстве категорий
2. **Generator model M:** Обучается на корпусе, аугментированном retrieved passages и reflection tokens, вставленными offline критиком C. Стандартный next-token prediction на расширенном словаре. Потери по retrieved passages маскируются

### Инференс

1. Модель генерирует `Retrieve` токен — решает, нужен ли retrieval
2. Если да — retriever R извлекает K документов, модель генерирует K параллельных продолжений
3. Каждое продолжение оценивается через `IsRel`, `IsSup`, `IsUse`
4. Segment-level beam search выбирает лучший вариант по взвешенной сумме critique scores

Ключевое отличие от RLHF: критика вычисляется offline и вставляется в тренировочный корпус, что значительно снижает стоимость обучения по сравнению с PPO.

### Кастомизация на инференсе

Веса для `IsRel`, `IsSup`, `IsUse` настраиваются без переобучения:
- Для задач с требованиями к цитированию — увеличивается вес `IsSup`
- Для открытых задач — снижается частота retrieval через порог на `Retrieve`

## Key Results

| Модель | PopQA | TriviaQA | PubHealth | ARC-C | Bio (FS) | ASQA (prec) |
|--------|-------|----------|-----------|-------|----------|-------------|
| ChatGPT | 29.3 | 74.3 | 70.1 | 75.3 | 71.8 | — |
| Ret-ChatGPT | 50.8 | 65.7 | 54.7 | 75.3 | — | 65.1 |
| Llama2-FT 7B + ret | 48.7 | 57.3 | 64.3 | 65.8 | 78.2 | 5.0 |
| **Self-RAG 7B** | **54.9** | 66.4 | **72.4** | 67.3 | **81.2** | **66.9** |
| **Self-RAG 13B** | **55.8** | **69.3** | **74.5** | **73.1** | 80.2 | **70.3** |

- Self-RAG 7B/13B превосходит ChatGPT на PopQA, PubHealth, Bio, ASQA
- Превосходит CoVE 65B (итеративный prompt engineering на Llama2 65B) на Bio generation
- Citation precision Self-RAG выше ChatGPT (70.3 vs 65.1 на ASQA)

Ablation studies:
- Без Critic C — падение до 18.1 str-em на ASQA (вместо 32.1)
- Retrieve top 1 (как обычный RAG) — падение на PopQA и ASQA
- Увеличение веса `IsSup` повышает citation precision, но снижает fluency (MAUVE)
- Emergent threshold: модель сама калибрует, когда вызывать retrieval

## My notes

- Элегантное решение проблемы "всегда извлекать vs никогда не извлекать". Модель сама определяет необходимость retrieval через специальные токены
- Подход к обучению через offline critique + standard LM objective намного дешевле PPO/RLHF
- Кастомизация весов на инференсе без переобучения — сильная практическая фича: один и тот же чекпоинт для разных юзкейсов
- Critic model обучен на GPT-4 разметке (4k-20k примеров на тип токена) — дешевая дистилляция
- Ограничения: генерация по сегментам (предложениям) — медленнее стандартного inference; beam search по K документам * B beams = K*B параллельных генераций
- По сути, Self-RAG — это первый серьезный шаг к тому, чтобы модель стала своим собственным критиком в RAG pipeline, без внешних NLI/summarization моделей
- Связь с [[02 Areas/ML & DL/Papers/ReAct|ReAct]]: обе работы дают модели agency над внешними действиями, но Self-RAG делает это через learned special tokens, а не промптинг
