---
title: "CS224N — Lecture 14: Reasoning and Agents"
course: "Stanford CS224N"
lecture: 14
type: source-note
status: legacy
raw: "[[02 Areas/ML & DL/raw/courses/Stanford CS224N/slides/cs224n-spr2024-lecture14-agents-shikhar-updated]]"
concepts: ["[[02 Areas/ML & DL/Concepts/Inference/Chain of Thought|Chain-of-Thought]]", "[[02 Areas/ML & DL/Concepts/NLP/LLM Agents|LLM Agents]]", "[[02 Areas/ML & DL/Concepts/NLP/Tool Use|Tool Use]]", "[[02 Areas/ML & DL/Concepts/Reasoning/ReAct|ReAct]]"]
---

# Lecture 14: Reasoning and Agents

> *"Disclaimer: Content for today is an active area of research and still emerging."* -- Shikhar Murty

Лектор: Shikhar Murty.

## Что такое Reasoning?

Reasoning -- использование **фактов и логики** для вывода ответа.

### Три типа рассуждений

| Тип | Описание | Пример |
|-----|----------|--------|
| **Deductive** | Из посылок → твёрдое заключение | Все млекопитающие имеют почки + Все киты -- млекопитающие → Все киты имеют почки |
| **Inductive** | Из наблюдений → вероятное заключение | Существо с крыльями обычно птица + Видим крылья → Скорее всего птица |
| **Abductive** | Из наблюдения → наиболее вероятное объяснение | Машина не заводится + лужа под двигателем → Вероятно, утечка в радиаторе |

В лекции фокус на **informal deductive reasoning** в несколько шагов.

## [[02 Areas/ML & DL/Concepts/Inference/Chain of Thought|Chain-of-Thought]] Prompting

### Стандартный CoT (Wei et al. 2023)

Вместо прямого ответа модель генерирует **промежуточные шаги** решения:

**Standard prompting**: Q: "Roger has 5 tennis balls. He buys 2 cans of 3." → A: "11" (часто ошибка)

**CoT prompting**: добавляем examples с цепочкой рассуждений в prompt → модель генерирует: "Roger started with 5. 2 cans of 3 = 6. 5 + 6 = 11." → A: "11" (правильно)

**Ключевой эффект**: значительное улучшение на арифметике, word problems, reasoning задачах. Работает **только для больших моделей** (>100B параметров).

### Zero-shot CoT (Kojima et al. 2023)

Достаточно добавить **"Let's think step by step"** в конец промпта -- модель начинает генерировать цепочку рассуждений без few-shot примеров.

### Self-Consistency (Wang et al. 2023)

Проблема greedy decoding: одна цепочка рассуждений может быть неверной. Решение:

1. Сэмплировать **несколько** цепочек рассуждений (diverse decoding)
2. Извлечь финальный ответ из каждой цепочки
3. Выбрать ответ с **максимальным consensus** (majority voting)

**Insight**: правильные рассуждения имеют **большее agreement**, чем неправильные. Self-consistency outperforms обычный CoT на многих бенчмарках и делает **больше, чем простое ансамблирование**.

### Least-to-Most Prompting (Zhou et al. 2023)

Problem decomposition: разбить сложную задачу на **подзадачи**, решить последовательно.

1. **Decompose**: "Какие подзадачи нужно решить?"
2. **Solve**: решить каждую подзадачу, используя результаты предыдущих

**Преимущество**: обобщает на задачи с **большим числом шагов**, чем в in-context examples (OOD generalization по числу шагов). Но с достаточным prompt engineering, CoT $\approx$ Least-to-Most.

## Reasoning через дистилляцию

### Orca (Mukherjee et al. 2023)

Можно ли получить reasoning-поведение в **маленьких** моделях, обучив их имитировать большие?

1. Собрать разнообразные instructions из FLAN-v2
2. Промптить GPT-4/ChatGPT с system message для генерации **CoT rationales**
3. Finetune Llama-13B на этих outputs

**Результат**: Orca outperforms Vicuna-13B и ChatGPT на BigBench-Hard (23 задачи с фокусом на multi-step reasoning).

### ReSTEM (Singh et al. 2024)

Reasoning by finetuning LMs на **собственных** outputs:
1. **E-step (Generate)**: сэмплировать несколько решений для reasoning задачи
2. **M-step (Improve)**: отфильтровать правильные (по answer correctness), finetune модель на них

Итеративный процесс: генерация → фильтрация → обучение → repeat.

## Могут ли LLM действительно рассуждать?

### Проблема faithfulness

Lanham et al. 2023: **CoT rationales не всегда faithful** (верны причине ответа):
- Модели иногда отвечают правильно **без полного rationale** → rationale может быть post-hoc
- Иногда правильный ответ даже с **неверным** rationale

### Reasoning vs Memorization

**Counterfactual experiments** (Wu et al. 2024, Hodel et al. 2024):
- Заменяем задачи на **counterfactual** варианты (сохраняя структуру, меняя facts)
- GPT-4 показывает **значительное падение** performance → evidence of spurious reasoning / memorization
- Люди **не** показывают такого падения

**Вывод**: LLMs демонстрируют impressive reasoning, но часть performance может быть объяснена memorization patterns из training data, а не genuine logical inference.

## [[02 Areas/ML & DL/Concepts/NLP/LLM Agents|LLM Agents]]

### Терминология

Agent -- модель, которая **взаимодействует с окружением**:

```
                    Action
                      ↓
Environment    ←→    π(·|g)    ←    Language Instruction
                      ↑
                 Observation
```

- **Action**: Type X, Click on Y, Move mouse to Z
- **Observation**: HTML DOM, screenshots, text
- **Instruction (goal)**: "Book a flight from SFO to NYC"

### Pre-LLM подходы

1. **Machine Translation approach**: map instructions → action sequences напрямую (Zettlemoyer et al. 2012). Работает для простых grounded environments (text2sql)
2. **Structured plans**: infer executable plans из (instruction, trajectory) pairs (Chen & Mooney 2011)
3. **RL**: reinforcement learning для mapping instructions → actions (Branavan et al. 2009)

### LLM-era подходы

С 2024: **LLM как policy** -- generative trajectory modeling через causal [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]].

### [[02 Areas/ML & DL/Concepts/Reasoning/ReAct|ReAct]] (Yao et al. 2023)

Reasoning + Acting в одном LLM:

```
System: You are an agent capable of actions: Type, Click, Move mouse...
Instruction: {goal}
Previous actions and observations: o1, a1, o2, a2, ...
Current HTML state: <state>

Thought: [model generates reasoning]
Action: [model generates next action]
```

По сути: **CoT prompting в цикле** -- модель чередует reasoning (Thought) и acting (Action), получает observation, и повторяет.

### [[02 Areas/ML & DL/Concepts/NLP/Tool Use|Tool Use]] и Function Calling

LLM взаимодействует с **внешними инструментами**:
- Калькулятор, поисковик, API, интерпретатор кода
- **Function calling**: модель генерирует structured JSON с именем функции и аргументами
- Примеры: Toolformer (Schick et al. 2023), WebGPT

### Компоненты Agent-а

| Компонент | Описание |
|-----------|----------|
| **Planning** | Разбиение задачи на подзадачи |
| **Tool use** | Вызов внешних инструментов |
| **Memory** | Context window + external storage (RAG) |
| **Reflection** | Self-evaluation и коррекция ошибок |

## Бенчмарки для LLM Agents

### MiniWoB++ (Shi et al. 2017)

Sandboxed browser interactions (social media, email). Короткий horizon. Zero-shot performance далеко от perfect.

### WebArena (Zhou et al. 2024)

Sandboxed approximations **реальных сайтов** (e-commerce, social media). Multi-tab browsing, long-horizon задачи. Evaluates functional correctness.

### WebLINX (Lu et al. 2024)

Interactions на **реальных** веб-сайтах. Conversational: "say" action для коммуникации с человеком. Turn-level метрики.

## BAGEL: Exploration + Synthetic Data для Agents

Проблема: few-shot demonstrations от humans **не масштабируются**. Решение:

1. **Explore**: LLM автономно исследует website (click, type, navigate)
2. **Label**: из записанных trajectories генерировать **natural language instructions**
3. **Filter**: проверить, что trajectory действительно выполняет сгенерированную instruction
4. **Train**: finetune LLM-agent на (instruction, trajectory) парах

**Ключевая идея**: использовать natural language как bridge между exploration и training.

## Concepts covered

- [[02 Areas/ML & DL/Concepts/Inference/Chain of Thought|Chain-of-Thought]] -- промпт с цепочкой рассуждений, zero-shot CoT, self-consistency
- [[02 Areas/ML & DL/Concepts/NLP/LLM Agents|LLM Agents]] -- модели как агенты с planning, tool use, memory
- [[02 Areas/ML & DL/Concepts/NLP/Tool Use|Tool Use]] -- function calling, external tools
- [[02 Areas/ML & DL/Concepts/Reasoning/ReAct|ReAct]] -- Reasoning + Acting в одном LLM
