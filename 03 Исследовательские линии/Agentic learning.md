---
title: Agentic learning
type: research-line
status: emerging
started: 2022
last_updated: 2026-07-16
last_verified: 2026-07-16
key_concepts: [ReAct, Tool Use, RLVR, Memory, Planning]
key_models: [Toolformer]
primary_sources:
  - https://arxiv.org/abs/2210.03629
  - https://arxiv.org/abs/2302.04761
---

# Agentic learning

## Тезис

Агент отличается от одиночной генерации наличием цикла «действие →
наблюдение → обновление состояния»:

```mermaid
flowchart LR
    G["цель"] --> P["план / policy"]
    P --> A["действие или tool call"]
    A --> O["наблюдение"]
    O --> M["память / состояние"]
    M --> P
    P -->|готово| R["результат"]
```

Agentic learning учит не только формат tool call, но и политику: когда вызвать
инструмент, какой выбрать, как восстановиться после ошибки и когда остановиться.

## Лестница обучения

| Уровень | Что обучается | Ограничение |
|---|---|---|
| prompt-only | формат ReAct и few-shot pattern | хрупкость |
| SFT trajectories | имитация успешных действий | нет exploration |
| synthetic trajectories | масштабирование сценариев | simulator gap |
| outcome RL / RLVR | оптимизация конечного успеха | credit assignment |
| process/step rewards | качество отдельных действий | локальная оптимальность |

## Что должен проверять benchmark

- выбор правильного инструмента;
- корректность аргументов и schema;
- multi-step dependency;
- обработка ошибок и изменившегося состояния;
- эффективность числа вызовов;
- корректный отказ или уточнение;
- безопасность побочных эффектов.

Точный финальный ответ недостаточен: агент мог угадать его, проигнорировав
инструмент. Нужна проверка trajectory и состояния среды.

## Главные риски

- reward hacking в симуляторе;
- расхождение synthetic arena и production API;
- недетерминированные и необратимые действия;
- prompt injection через tool output;
- отсутствие наблюдаемости длинной trajectory;
- оптимизация benchmark-specific workflows.

## Открытые вопросы

- Как распределять credit на длинном горизонте?
- Как учить безопасной остановке и запросу подтверждения?
- Что хранить в памяти и как предотвращать её отравление?
- Как переносить policy между меняющимися API?
- Где заканчивается модель и начинается обучаемая внешняя orchestration system?

## Связанные страницы

[[02 Areas/ML & DL/Concepts/Reasoning/ReAct|ReAct]] ·
[[02 Areas/ML & DL/Concepts/Reasoning/Tool Use|Tool Use]] ·
[[02 Areas/ML & DL/Concepts/Reasoning/Toolformer|Toolformer]] ·
[[02 Areas/ML & DL/Concepts/Training/RLVR|RLVR]]
