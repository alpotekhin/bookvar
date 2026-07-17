---
title: "ReAct: Synergizing Reasoning and Acting in Language Models"
url: https://arxiv.org/abs/2210.03629
authors: [Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao]
year: 2023
date_reviewed: 2026-04-07
type: source-note
status: legacy
category: paper
tags:
  - Agents
  - Reasoning
  - LLM
Date: 2023-03-10
Organization: Princeton University, Google Research Brain
Status: Done
concepts:
  - "[[02 Areas/ML & DL/Concepts/Inference/Chain of Thought|Chain of Thought]]"
  - "[[02 Areas/ML & DL/Concepts/Agents/ReAct|ReAct]]"
  - "[[02 Areas/ML & DL/Concepts/Inference/Prompting|Prompting]]"
  - "[[02 Areas/ML & DL/Concepts/Inference/In-context Learning|In-context Learning]]"
raw: "[[02 Areas/ML & DL/raw/papers/react/paper.txt]]"
---

# ReAct: Synergizing Reasoning and Acting in Language Models

**Authors:** Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao
**Published:** 2023 (ICLR 2023; arXiv:2210.03629v3)
**URL:** https://arxiv.org/abs/2210.03629

## TL;DR

ReAct — парадигма промптинга, в которой LLM генерирует **чередующиеся reasoning traces (мысли) и действия** в интерактивной среде. Мысли помогают модели планировать, отслеживать прогресс и обрабатывать исключения; действия позволяют получать информацию из внешних источников. На HotpotQA и FEVER ReAct побеждает Act-only и конкурирует с CoT, а комбинация ReAct + CoT-SC дает лучший результат. На ALFWorld и WebShop 1-2-shot ReAct превосходит RL/IL агентов, обученных на 10^3-10^5 примерах.

## Problem

Два направления использования LLM развивались изолированно:

1. **Reasoning (CoT)** — модель рассуждает, но в замкнутом пространстве; не может получить новую информацию извне. Результат: галлюцинации и ошибки накапливаются
2. **Acting (action generation)** — модель выполняет действия, но без явного рассуждения не может планировать, отслеживать подцели и обрабатывать ошибки

Люди же естественно объединяют "inner speech" (рассуждение) с действиями: между шагами готовки мы думаем, отслеживаем прогресс, корректируем план.

## Method

### Ключевая идея

Расширить пространство действий агента: A_hat = A ∪ L, где L — пространство языка. "Мысль" (thought) — действие в языковом пространстве, которое **не влияет на среду**, но обновляет контекст для дальнейших решений.

### Типы мыслей
- Декомпозиция цели: "I need to search X, find Y, then find Z"
- Извлечение информации из наблюдений: "X was started in 1844"
- Commonsense reasoning: "X is not Y, so Z must be..."
- Отслеживание прогресса: "Now I need to..."
- Обработка ошибок: "Maybe I can search X instead"
- Синтез ответа: "So the answer is X"

### Промптинг

Для каждой задачи вручную аннотируется 1-6 траекторий с thoughts, actions и observations. В reasoning-задачах мысли чередуются с действиями на каждом шаге; в decision-making задачах мысли появляются спарсенно в ключевых моментах.

### Среды

**Knowledge-intensive:** Wikipedia API с тремя действиями: `search[entity]`, `lookup[string]`, `finish[answer]`

**Decision making:** ALFWorld (текстовая среда для embodied tasks), WebShop (онлайн-шопинг на 1.18M товаров)

## Key Results

### HotpotQA / FEVER (PaLM-540B)

| Метод | HotpotQA (EM) | FEVER (Acc) |
|-------|---------------|-------------|
| Standard | 28.7 | 57.1 |
| CoT | 29.4 | 56.3 |
| CoT-SC (21 samples) | 33.4 | 60.4 |
| Act-only | 25.7 | 58.9 |
| **ReAct** | 27.4 | **60.9** |
| **CoT-SC -> ReAct** | 34.2 | **64.6** |
| **ReAct -> CoT-SC** | **35.1** | 62.0 |

### Анализ ошибок (HotpotQA, 200 примеров)

| Тип | ReAct | CoT |
|-----|-------|-----|
| Галлюцинации (success) | 6% | 14% |
| Галлюцинации (failure) | 0% | **56%** |
| Reasoning errors | **47%** | 16% |
| Search errors | 23% | — |

CoT страдает от галлюцинаций (56% ошибок), а ReAct — от reasoning errors и неинформативного поиска. Комбинация дает лучший результат.

### ALFWorld / WebShop

| Метод | ALFWorld (success %) | WebShop (success rate) |
|-------|---------------------|----------------------|
| Act-only (best of 6) | 45% | 30.1% |
| **ReAct (best of 6)** | **71%** | **40.0%** |
| BUTLER (IL, 10^5 traj) | 37% | — |
| IL+RL | — | 28.7% |

### Fine-tuning

При fine-tuning на 3000 траекториях ReAct становится лучшим методом: PaLM-8B finetuned ReAct превосходит все PaLM-62B prompting методы.

## My notes

- Основополагающая работа для LLM-агентов. ReAct паттерн (Thought -> Action -> Observation) стал стандартом в LangChain, AutoGPT и т.д.
- Trade-off между factuality (ReAct) и flexibility (CoT): ReAct заземлена в фактах, но ограничена структурой thought-action-observation; CoT свободнее в рассуждениях, но галлюцинирует
- Лучшая стратегия — **комбинация**: если ReAct не может найти ответ за N шагов -> fallback на CoT-SC (или наоборот при низкой уверенности CoT)
- Повторяющиеся мысли и действия — частая проблема ReAct (47% ошибок — reasoning errors). Авторы предполагают, что beam search может помочь
- Промпты минимальны: 1-6 human-annotated trajectories. Это делает подход масштабируемым
- Связь с [[02 Areas/ML & DL/Papers/Toolformer|Toolformer]]: оба дают LLM доступ к инструментам, но ReAct через промптинг, а Toolformer через обучение вставлять API calls
- Связь с [[02 Areas/ML & DL/Papers/Self-RAG|Self-RAG]]: Self-RAG решает ту же проблему (когда и как извлекать), но через learned reflection tokens вместо промптинга
