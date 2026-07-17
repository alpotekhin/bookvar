---
title: "ReAct"
aliases: [ReAct, Reasoning and Acting, ReAct Agent, Thought-Action-Observation]
type: concept
status: legacy
category: Reasoning
papers:
  - "[[02 Areas/ML & DL/Papers/ReAct|ReAct]]"
courses:
  - "[[02 Areas/ML & DL/Courses/Stanford CS224N/Lecture 14 — Reasoning and Agents|CS224N Lecture 14]]"
sources:
  - "[ReAct project page (Princeton)](https://react-lm.github.io/)"
  - "[IBM — What is a ReAct Agent?](https://www.ibm.com/think/topics/react-agent)"
  - "[Prompt Engineering Guide — ReAct](https://www.promptingguide.ai/techniques/react)"
  - "[Hugging Face — Agents Course: Thoughts](https://huggingface.co/learn/agents-course/unit1/thoughts)"
---

# ReAct: Synergizing Reasoning and Acting in Language Models

## Зачем это нужно: два изолированных мира

До 2022 года способности LLM к рассуждению (reasoning) и к действию (acting) изучались как **отдельные направления**. С одной стороны, [[02 Areas/ML & DL/Concepts/Inference/Chain of Thought|Chain of Thought]] (Wei et al., 2022) показал, что LLM могут генерировать цепочки рассуждений и решать задачи, требующие многошагового вывода. С другой стороны, работы вроде WebGPT и SayCan учили модели выполнять действия в среде -- искать информацию, управлять роботом.

Проблема в том, что эти подходы были **взаимоисключающими**:
- **CoT (Reason Only)** -- модель рассуждает, но в вакууме. Если внутренние знания модели неверны или устарели, она **галлюцинирует**: 56% ошибок CoT на HotpotQA связаны с выдуманными фактами (Table 2 в статье).
- **Act Only** -- модель действует (например, вызывает search API), но без рассуждений. Она не может декомпозировать задачу, отслеживать прогресс или обрабатывать ошибки. Агент в Figure 1(1c) не способен синтезировать финальный ответ из нескольких наблюдений.

Человек же естественно **чередует** мышление и действие: "У меня нет соли, поэтому использую соевый соус" (рассуждение) -> открыть холодильник (действие) -> увидеть, что соевого соуса тоже нет (наблюдение) -> "Тогда попробую лимонный сок" (новое рассуждение). Именно эту синергию воспроизводит ReAct.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/react/react-fig1.png]]
*Сравнение 4 paradigm: (a) Standard, (b) CoT (Reason Only), (c) Act Only, (d) ReAct (Reason+Act). ReAct чередует Thought, Action и Observation для решения задач с внешними инструментами (источник: Yao et al., 2023)*

## Ключевая идея: расширенное пространство действий

**ReAct** (Yao et al., 2023, Princeton/Google Brain, ICLR 2023) -- парадигма промптинга, в которой LLM генерирует **чередующиеся** reasoning traces (мысли) и actions (действия) при взаимодействии с внешней средой.

Формально, пространство действий агента расширяется:

$$\hat{A} = A \cup L$$

где $A$ -- пространство физических/API действий, а $L$ -- пространство языка. "Мысль" $\hat{a}_t \in L$ -- это действие в языковом пространстве, которое **не влияет на среду** (нет observation feedback), но обновляет контекст агента для будущих шагов.

## Паттерн Thought -> Action -> Observation

Каждая траектория решения задачи состоит из чередующихся шагов:

**Thought (Мысль)** -- свободный текст рассуждения. Типы мыслей из статьи:
- Декомпозиция задачи: *"I need to search Apple Remote and find the program it was originally designed to interact with"*
- Извлечение информации из наблюдений: *"Apple Remote was originally designed to control the Front Row media center program"*
- Commonsense reasoning: *"x is not y, so z must instead be..."*
- Отслеживание прогресса: *"Now I find a pepper shaker. Next, I need to put it in/on drawer 1"*
- Обработка ошибок и переформулирование: *"Front Row is not found. I need to search Front Row (software)"*

**Action (Действие)** -- взаимодействие со средой через фиксированный набор API:
- `search[entity]` -- первые 5 предложений из Wikipedia
- `lookup[string]` -- Ctrl+F по странице
- `finish[answer]` -- завершение с ответом

**Observation (Наблюдение)** -- результат из среды, не контролируемый моделью.

### Промптинг

ReAct использует **few-shot промптинг**: 1-6 вручную аннотированных траекторий с thoughts, actions, observations как in-context examples. Дизайн промптов интуитивен -- аннотатор просто записывает свои мысли поверх действий. Никаких специальных форматов, шаблонов или selection стратегий.

Для knowledge-intensive задач (HotpotQA, FEVER) используется **dense thought** -- мысль перед каждым действием. Для decision-making задач (ALFWorld, WebShop), где действий может быть много, мысли появляются **спарсивно** -- модель сама решает, когда думать.

## Эксперименты и результаты

### Knowledge-intensive задачи

| Метод | HotpotQA (EM) | FEVER (Acc) |
|-------|---------------|-------------|
| Standard | 28.7 | 57.1 |
| CoT | 29.4 | 56.3 |
| CoT-SC (21 samples) | 33.4 | 60.4 |
| Act Only | 25.7 | 58.9 |
| **ReAct** | **27.4** | **60.9** |
| ReAct -> CoT-SC | **35.1** | 62.0 |
| CoT-SC -> ReAct | 34.2 | **64.6** |
| *Supervised SoTA* | *67.5* | *89.5* |

*Таблица: Результаты PaLM-540B на HotpotQA и FEVER (Table 1 из статьи)*

Ключевые наблюдения:
- **ReAct outperforms Act** на обеих задачах -- рассуждения помогают синтезировать ответ
- **ReAct vs CoT**: ReAct лучше на FEVER (60.9 vs 56.3), чуть хуже на HotpotQA (27.4 vs 29.4)
- **Комбинация ReAct + CoT-SC** -- лучший результат: модель использует и внутренние знания (CoT), и внешние (ReAct)

### Анализ ошибок: factuality vs flexibility trade-off

| Тип | ReAct | CoT |
|-----|-------|-----|
| **Успех: правильное рассуждение** | 94% | 86% |
| **Успех: галлюцинация** | 6% | 14% |
| **Ошибка: reasoning error** | 47% | 16% |
| **Ошибка: галлюцинация** | 0% | **56%** |
| **Ошибка: плохой поиск** | 23% | - |

*Таблица: Анализ 200 примеров на HotpotQA (Table 2 из статьи)*

**Главный вывод**: галлюцинация -- основная проблема CoT (56% ошибок), но **нулевая** для ReAct. Зато ReAct чаще ошибается в reasoning (47%), особенно зацикливаясь на повторении предыдущих мыслей и действий.

### Decision-making задачи

| Задача | ReAct (1-2 shot) | Лучший baseline |
|--------|-------------------|-----------------|
| ALFWorld | **71%** | BUTLER 37% (10^5 trajs) |
| WebShop | **40.0%** | IL+RL 28.7% |

ReAct с 1-2 примерами в промпте **превосходит** методы, обученные на сотнях тысяч траекторий. На ALFWorld мысли помогают: (1) декомпозировать цель, (2) отслеживать прогресс подзадач, (3) определять вероятное расположение объектов через commonsense (desklamps -> desks/shelves/dressers).

### Fine-tuning: ReAct масштабируется лучше всех

При промптинге маленьких моделей (PaLM-8B/62B) ReAct -- худший метод (сложно выучить и reasoning, и acting из few-shot). Но при fine-tuning на 3,000 примеров ситуация кардинально меняется:

- **PaLM-8B finetuned ReAct** превосходит все PaLM-62B prompting методы
- **PaLM-62B finetuned ReAct** превосходит все PaLM-540B prompting методы

Причина: fine-tuning Standard/CoT учит модель **запоминать факты** (потенциально галлюцинированные). Fine-tuning ReAct/Act учит **навыку** обращения к Wikipedia -- более обобщаемому умению.

## Стратегии комбинирования ReAct и CoT

Статья предлагает два эвристических подхода:

1. **ReAct -> CoT-SC**: если ReAct не находит ответ за $n$ шагов (7 для HotpotQA, 5 для FEVER), fallback на CoT-SC. Лучший результат на HotpotQA (35.1 EM).

2. **CoT-SC -> ReAct**: если majority answer среди $n$ CoT-SC сэмплов встречается менее $n/2$ раз (модель не уверена), fallback на ReAct. Лучший результат на FEVER (64.6 acc).

Оба подхода достигают уровня CoT-SC с 21 сэмплом, используя всего 3-5 сэмплов -- значительная экономия compute.

## Почему ReAct стал фундаментом LLM-агентов

ReAct-паттерн (Thought-Action-Observation) стал **стандартной архитектурой** для LLM-агентов:

1. **LangChain** и **LangGraph** -- ReAct как основной тип агента
2. **AutoGPT**, **BabyAGI** -- развитие идеи автономных агентов с рассуждениями
3. **OpenAI Function Calling** -- формализация action space через JSON schema
4. **Hugging Face Agents Course** -- ReAct как первый паттерн, который изучают

**Свойства**, делающие ReAct практически универсальным:
- **Интуитивный дизайн промптов** -- просто записать мысли на естественном языке
- **Гибкость** -- работает для QA, fact verification, text games, web navigation
- **Интерпретируемость** -- человек может проверить reasoning и factual correctness
- **Контролируемость** -- можно корректировать поведение через thought editing (Figure 5)

## Ограничения

- **Зависимость от качества поиска**: 23% ошибок ReAct -- неинформативные результаты search
- **Repetitive loops**: модель может зацикливаться на повторении мыслей/действий
- **Prompting**: ограниченная поддержка сложных рассуждений в few-shot setup
- **Один API за шаг**: нет параллельных вызовов инструментов (в отличие от более поздних работ)

## Related concepts

- [[02 Areas/ML & DL/Concepts/Inference/Chain of Thought|Chain of Thought]] -- reasoning-only baseline, комплементарен ReAct
- [[02 Areas/ML & DL/Concepts/Reasoning/Self-Consistency|Self-Consistency]] -- majority voting по нескольким CoT, комбинируется с ReAct
- [[02 Areas/ML & DL/Concepts/Reasoning/Toolformer|Toolformer]] -- обучение tool use через self-supervised fine-tuning (в отличие от промптинга ReAct)
- [[02 Areas/ML & DL/Concepts/Reasoning/Tree of Thoughts|Tree of Thoughts]] -- tree search вместо линейной траектории
- [[02 Areas/ML & DL/Concepts/Inference/Prompting|Prompting]] -- ReAct как продвинутый вид промптинга
- [[02 Areas/ML & DL/Concepts/Retrieval/Self-RAG|Self-RAG]] -- selective retrieval с self-reflection

## Хронология

| Год | Milestone |
|-----|-----------|
| 2022 | CoT prompting (Wei et al.) -- reasoning в LLM |
| 2022 | WebGPT (Nakano et al.) -- acting в LLM для QA |
| **2023** | **ReAct (Yao et al.) -- объединение reasoning + acting, ICLR 2023** |
| 2023 | Toolformer -- self-supervised tool use |
| 2023 | LangChain ReAct Agent -- промышленная имплементация |
| 2024 | OpenAI Assistants API, LangGraph -- продвинутые агентные фреймворки |

## Дополнительные ресурсы

- [ReAct project page (Princeton)](https://react-lm.github.io/) -- код и промпты
- [IBM — What is a ReAct Agent?](https://www.ibm.com/think/topics/react-agent) -- хороший обзор
- [Prompt Engineering Guide — ReAct](https://www.promptingguide.ai/techniques/react) -- практическое руководство
- [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/unit1/thoughts) -- ReAct в контексте агентного курса
