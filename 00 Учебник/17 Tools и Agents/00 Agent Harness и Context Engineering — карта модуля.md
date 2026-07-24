---
title: Agent Harness и Context Engineering — карта модуля
type: textbook-chapter
status: redirect
last_updated: 2026-07-20
redirect_to: "[[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/65 Tool use — от вызова функции к действию]]"
---

# Agent Harness и Context Engineering

Агентный раздел начинается не с перечня frameworks, а с четырёх вопросов:

1. как ответ модели превращается в проверяемое действие;
2. какая программа собирает контекст и исполняет цикл;
3. где живёт состояние между шагами и как координируются несколько участников;
4. как доказать, что весь процесс решает задачу устойчиво, а не случайно получил
   правдоподобный финальный текст.

## Линейный маршрут

| Глава | Главный объект | Что читатель должен уметь восстановить |
|---|---|---|
| [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/65 Tool use — от вызова функции к действию|65. Tool use]] | schema, tool call, executor | полный путь от logits/JSON до изменения внешнего состояния |
| [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/66 Agent harness и context engineering|66. Harness и context]] | agent loop и context builder | состав каждого model input, права, ошибки и stopping rule |
| [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/67 Память, планирование и оркестрация агентов|67. Память и orchestration]] | persisted state и coordination | границы памяти, маршрутизацию и восстановление workflow |
| [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/68 Оценивание агентных систем|68. Evaluation]] | task, trajectory и environment | воспроизводимый evaluation harness и failure taxonomy |

Порядок существенен. Без главы 65 «планирование» остаётся текстом без
исполнения. Без главы 66 невозможно указать, кто добавил reminder, сжал историю
или повторил упавший инструмент. Без главы 67 успешный длинный запуск нельзя
возобновить. Без главы 68 демонстрация не превращается в инженерное
утверждение.

## Практический корпус

В Bookvar полностью импортирована коллекция
[[02 Areas/ML & DL/05 Источники/Courses/GenAI Agents|GenAI Agents]]: 53
оригинальных notebooks с кодом, результатами выполнения и иллюстрациями.
Начальная последовательность построена так, чтобы на каждом шаге появлялся один
новый элемент агентной системы:

1. [[02 Areas/ML & DL/05 Источники/Courses/GenAI Agents/all_agents_tutorials/simple_question_answering_agent.ipynb|Simple Question Answering Agent]] — исходная точка: один вызов `prompt → model → answer`, без инструментов и цикла;
2. [[02 Areas/ML & DL/05 Источники/Courses/GenAI Agents/all_agents_tutorials/task_oriented_agent.ipynb|Task-Oriented Agent]] — две функции превращаются в типизированные инструменты, которыми управляет `AgentExecutor`;
3. [[02 Areas/ML & DL/05 Источники/Courses/GenAI Agents/all_agents_tutorials/langgraph-tutorial.ipynb|LangGraph Tutorial]] — состояние и переходы между узлами записаны явно;
4. [[02 Areas/ML & DL/05 Источники/Courses/GenAI Agents/all_agents_tutorials/memory-agent-tutorial.ipynb|Memory Agent Tutorial]] — к графу добавляются семантическая, эпизодическая и процедурная память;
5. [[02 Areas/ML & DL/05 Источники/Courses/GenAI Agents/all_agents_tutorials/e2e_testing_agent.ipynb|E2E Testing Agent]] — длинный процесс включает построение плана, генерацию Playwright-кода, исполнение, проверку и отчёт.

Остальные notebooks не нужно читать подряд. Индекс коллекции позволяет выбрать
реализацию под механизм главы: MCP, multi-agent routing, document intake,
research workflow, recovery или domain-specific tool use.
