---
title: "RLVR"
aliases: [RLVR, Reinforcement Learning with Verifiable Rewards, RL with Verifiable Rewards]
type: concept
status: legacy
category: Training
papers:
  - "[[02 Areas/ML & DL/Papers/DeepSeek-R1 Reasoning via RL|DeepSeek-R1]]"
  - "[[02 Areas/ML & DL/Papers/ASTRA|ASTRA]]"
  - "[[02 Areas/ML & DL/Papers/ToolRL|ToolRL]]"
  - "[[02 Areas/ML & DL/Papers/Nemotron-Research-Tool-N1|Tool-N1]]"
courses: []
sources:
  - "[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL (2025)](https://arxiv.org/abs/2501.12948)"
  - "[ASTRA: Automated Synthesis of agentic Trajectories and Reinforcement Arenas (2026)](https://arxiv.org/abs/2601.21558)"
  - "[ToolRL: Reward is All Tool Learning Needs (2025)](https://arxiv.org/abs/2504.13958)"
---

# RLVR — Reinforcement Learning with Verifiable Rewards

## Зачем это нужно: проблема reward signal

В стандартном [[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]] reward приходит от **обученной нейросети** (Reward Model). Две фундаментальные проблемы:

1. **Reward hacking.** Модель находит баги в RM — генерирует ответы, которые RM оценивает высоко, но которые на самом деле плохие (длинные, уклончивые, стилистически "приятные" но неправильные)
2. **Шум.** RM ошибается, особенно на out-of-distribution примерах. Шумный reward → шумный gradient → нестабильное обучение

**RLVR** решает обе проблемы: reward вычисляется **детерминированно** через внешнюю систему, которая **проверяет**, а не аппроксимирует.

## Как работает RLVR

### Идея

Заменить нейросетевой reward model на **верификатор** — систему, которая проверяет правильность ответа объективно:

```
RLHF:  model output → Reward Model (нейросеть) → score (шумный)
RLVR:  model output → Verifier (код/правила)   → score (детерминированный)
```

### Типы верификаторов

| Домен | Верификатор | Reward |
|---|---|---|
| **Математика** | Сравнение с ground truth | `1 if answer == expected else 0` |
| **Код** | Запуск unit-тестов | `passed_tests / total_tests` |
| **Формат** | Regex / JSON parser | `1 if valid_format else 0` |
| **Tool-use** | Sandbox execution | `1 if correct_output else 0` |

### Пример: math RLVR (DeepSeek-R1)

```
Prompt: "Сколько будет 17 × 23?"
Model output: "<think>17 × 20 = 340, 17 × 3 = 51, итого 391</think><answer>391</answer>"

Verifier: extract_answer("391") == 391   → reward = 1.0
```

Никакой RM. Никакого judge. Код либо прошёл, либо нет.

### Пример: tool-use RLVR (ASTRA)

```
Prompt: "Найди email пользователя 123"
Agent: → tool_call: get_user(id=123)
Tool:  → {"name": "Иванов", "email": "iv@x5.ru"}
Agent: → "Email пользователя: iv@x5.ru"

Verifier: "iv@x5.ru" in agent_final_answer   → reward = 1.0
```

Тул — реальная Python-функция с ground-truth данными. Детерминированная проверка.

## Reward design в RLVR

### Binary vs Continuous

**Binary (0 или 1)** — простейший, работает для math/code:
- Проблема: sparse signal. Для сложных задач модель почти всегда получает 0 → нулевой gradient

**Continuous** — более информативный, но сложнее построить:
- [[02 Areas/ML & DL/Papers/ASTRA|ASTRA]]: F1-style `reward = 2rp/(r+p)` где `r = solved/total`, `p = solved/calls`
- [[02 Areas/ML & DL/Papers/ToolRL|ToolRL]]: step-level granular (-3..+3) по tool name, param names, param values

### Trajectory-level vs Step-level

**Trajectory-level** (ASTRA): один reward за всю цепочку tool calls.
- (+) Учит глобальную стратегию
- (−) Размытый credit assignment — модель не знает какой именно шаг был ошибкой
- (−) Чем длиннее цепочка, тем слабее сигнал на каждый шаг

**Step-level** (ToolRL): отдельный reward за каждый tool call.
- (+) Точный credit assignment
- (−) Может быть greedy — оптимизирует шаги по отдельности, теряет глобальный план

### F1 reward: баланс recall и precision

ASTRA показал что форма reward критична для стабильности RL:
- **Recall-only** (`r = solved/total`) → agent спамит tool calls бесконечно → training collapse
- **Precision-only** (`p = solved/calls`) → agent слишком консервативен → collapse
- **F1** (`2rp/(r+p)`) → баланс, стабильный training

## RLVR и [[02 Areas/ML & DL/Concepts/Training/GRPO|GRPO]]

RLVR естественно сочетается с GRPO:
- GRPO не требует value model — только reward function
- RLVR даёт детерминированный reward → более стабильные z-scores внутри группы
- Нет reward model → нет reward hacking → можно убрать KL-penalty (как в ASTRA)

Но убирание KL-penalty — рискованно. KL — основной механизм антифоргеттинга. Без него модель может деградировать на general capabilities (T-Bank T-Pro 2.1 подтверждает это).

## Ограничения RLVR

### 1. Не всё верифицируемо

RLVR работает когда есть **один правильный ответ**. Не работает для:
- Creative writing ("напиши эссе")
- Subjective tasks ("переформулируй вежливее")
- Open-ended planning ("подготовь отчёт")
- Safety/refusal ("модель должна отказать")

### 2. Shortcutting

Модель может обойти тулы — угадать ответ из промпта без реального использования инструментов. Reward = 1.0 (ответ-то правильный), но мета-навык не развивается. Чем однообразнее арены, тем выше риск.

### 3. Нет reward за правильный отказ

```
Нерешаемая задача:
- try & fail   → reward = 0
- refuse       → reward = 0    ← ТАКОЙ ЖЕ
- нулевая дисперсия → batch выброшен (Adaptive Batch Filling)
```

Модель **никогда не учится** правильно отказывать. В production ~15-25% запросов требуют отказа или уточнения.

### 4. Reality gap

Синтетические арены ≠ реальные API. Не покрыты: latency, rate limits, pagination, auth, versioning, concurrent writes, statefulness.

### 5. Coverage problem

Арены покрывают узкое множество сценариев. Production = long tail. Модель отлично работает in-distribution и непредсказуемо за пределами.

### 6. Inference-time gap

RLVR улучшает training-time signal, но **на инференсе нет ground truth**. Модель может галлюцинировать tool responses в production, и нет верификатора чтобы это поймать. Нужны дополнительные inference-time проверки: schema validation, grounding check, observability.

## Ландшафт RLVR для tool-use (2025-2026)

| Paper | SFT нужен? | Reward type | Horizon | Base model |
|---|---|---|---|---|
| [[02 Areas/ML & DL/Papers/ASTRA|ASTRA]] | Да | Trajectory F1 | Multi-turn | Qwen3-14B/32B |
| [[02 Areas/ML & DL/Papers/ToolRL|ToolRL]] | Нет (вредит) | Step granular | Single-turn | Qwen2.5-≤7B |
| [[02 Areas/ML & DL/Papers/Nemotron-Research-Tool-N1|Tool-N1]] | Нет | Binary | Multi-turn | 7B/14B |

Консенсуса нет. 2 из 3 работ говорят что SFT не нужен или вредит. Вопрос открыт — зависит от scale, horizon length, reward granularity.

## Related concepts

- [[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]] — предшественник с нейросетевым reward
- [[02 Areas/ML & DL/Concepts/Training/GRPO|GRPO]] — RL-алгоритм, естественно сочетающийся с RLVR
- [[02 Areas/ML & DL/Concepts/Training/DPO|DPO]] — off-policy альтернатива, не использует verifiable rewards
- [[02 Areas/ML & DL/Concepts/Reasoning/Toolformer|Toolformer]] — ранний подход к tool-use без RL
- [[02 Areas/ML & DL/Concepts/Reasoning/ReAct|ReAct]] — agent loop, который RLVR оптимизирует
