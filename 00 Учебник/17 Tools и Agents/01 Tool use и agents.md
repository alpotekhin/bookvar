---
title: Tool use и agents
type: textbook-chapter
status: canonical
last_updated: 2026-07-16
---

# Tool use и agents

> [!info] Углублённый модуль
> Harness, context construction, reminders, compaction, memory, permissions,
> budgets и trajectory evaluation разобраны в
> [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/00 Agent Harness и Context Engineering — карта модуля]].

> [!abstract] Идея главы
> Tool call — это структурированный ответ модели, который исполняет внешняя
> система. Agent появляется тогда, когда результат действия возвращается модели
> и цикл продолжается до проверяемого результата или условия остановки.

Tool use превращает generation в structured action. Agent добавляет loop,
состояние, среду и критерий остановки.

## Один цикл

```text
goal + state
→ model proposes answer or action
→ permission/policy check
→ tool executes
→ observation is recorded
→ model continues or stops
```

## Компоненты

- policy/model;
- tools с typed schemas;
- observation и state;
- memory;
- planner или implicit reasoning;
- permission boundary;
- verifier/evaluator;
- stop conditions.

ReAct чередует reasoning и actions. Toolformer обучает self-supervised вызовы.
Современный agentic RL использует environments с проверяемыми outcomes.

![[02 Areas/ML & DL/raw/papers/react/images/react-fig1.png]]

*ReAct: reasoning traces чередуются с actions и observations среды.*

Framework не является архитектурой агента. Сначала описывается нейтральный loop,
затем реализации вроде smolagents или LangGraph.

## Где чаще всего ломается

- tool schema допускает неоднозначные arguments;
- observation слишком длинное или недоверенное;
- модель повторяет неудачное действие;
- нет idempotency для side effects;
- stop condition зависит только от уверенного текста модели;
- prompt injection из tool output получает лишние permissions.

Надёжный agent ограничивает действия policy layer, журналирует state и проверяет
результат независимо от самооценки модели.

- [[02 Areas/ML & DL/Concepts/Reasoning/ReAct]]
- [[02 Areas/ML & DL/Concepts/Reasoning/Tool Use]]
- [[02 Areas/ML & DL/Papers/Toolformer]]
- [[02 Areas/ML & DL/Papers/ToolRL]]
- [[02 Areas/ML & DL/Papers/ASTRA]]
