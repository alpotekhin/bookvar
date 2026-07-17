---
title: "Chain of Thought"
aliases: [CoT, chain-of-thought prompting, CoT prompting, Chain-of-Thought]
type: concept
category: Inference
papers:
  - "[[02 Areas/ML & DL/Papers/COT|Chain-of-Thought Prompting (Wei et al., 2022)]]"
  - "[[02 Areas/ML & DL/Papers/Flan-T5-PaLM|Flan-T5/PaLM (Chung et al., 2022)]]"
sources:
  - "[Wei et al. — Chain-of-Thought Prompting (2022)](https://arxiv.org/abs/2201.11903)"
  - "[Wang et al. — Self-Consistency (2023)](https://arxiv.org/abs/2203.11171)"
  - "[Yao et al. — Tree of Thoughts (2023)](https://arxiv.org/abs/2305.10601)"
  - "[Kojima et al. — Zero-shot CoT (2022)](https://arxiv.org/abs/2205.11916)"
  - "[Mercity — Guide to Chain-of-Thought Prompting](https://www.mercity.ai/blog-post/guide-to-chain-of-thought-prompting/)"
courses:
  - "[[02 Areas/ML & DL/Courses/SHAD LLM/Week 1 — Intro to LLMs|SHAD LLM — Week 1]]"
---

# Chain of Thought (CoT)

![[02 Areas/ML & DL/raw/papers/cot/images/cot-prompting-fig1.png]]
*Standard prompting vs Chain-of-Thought prompting: CoT позволяет модели разбить сложную задачу на шаги, значительно повышая точность (источник: Wei et al., 2022)*

## Зачем это нужно: LLM не умеют считать

Большие языковые модели генерируют текст **слева направо, по одному токену**. Для простых задач (перевод, саммаризация) это работает хорошо. Но для задач, требующих **многошагового рассуждения** (арифметика, логика, символьные манипуляции), прямая генерация ответа катастрофически ломается.

Пример:

```
Standard Prompting:
Q: В кафетерии было 23 яблока. Использовали 20 на обед и купили ещё 6. 
   Сколько яблок осталось?
A: The answer is 27.                    ← НЕПРАВИЛЬНО

Chain-of-Thought Prompting:
Q: В кафетерии было 23 яблока. Использовали 20 на обед и купили ещё 6.
   Сколько яблок осталось?
A: В кафетерии было 23 яблока. Использовали 20, осталось 23 - 20 = 3.
   Купили ещё 6, итого 3 + 6 = 9.
   The answer is 9.                     ← ПРАВИЛЬНО
```

Почему standard prompting ошибается? Модель пытается «выстрелить» ответ за один шаг — один forward pass на один токен. Но задача требует нескольких последовательных вычислений. CoT решает это, заставляя модель **разбить рассуждение на шаги**, где каждый шаг — это дополнительные токены (и, следовательно, дополнительные вычислительные ресурсы через дополнительные forward passes).

## Как это работает: few-shot CoT

**Chain-of-Thought prompting** (Wei et al., 2022, Google Research, NeurIPS 2022): в демонстрациях few-shot промпта вместо пар (question, answer) используются тройки (question, **reasoning chain**, answer).

### Формат промпта

Промпт содержит **8 вручную написанных** демонстраций с reasoning chains:

```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
   Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls.
   5 + 6 = 11. The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more,
   how many apples do they have?
A: The cafeteria had 23 apples originally. They used 20 to make lunch.
   So they had 23 - 20 = 3. They bought 6 more apples, so they have 3 + 6 = 9.
   The answer is 9.

[... ещё 6 демонстраций ...]

Q: [НОВЫЙ ВОПРОС]
A: [модель генерирует цепочку рассуждений + ответ]
```

Модель видит паттерн «вопрос → пошаговое рассуждение → ответ» и **имитирует** его для нового вопроса. Это [[02 Areas/ML & DL/Concepts/Inference/In-context Learning|in-context learning]] — модель не обучается, а выводит формат из примеров.

### Типы задач

Авторы тестировали на трёх категориях:

1. **Arithmetic reasoning** — GSM8K, SVAMP, AQuA, ASDiv, MAWPS (math word problems)
2. **Commonsense reasoning** — StrategyQA, Date Understanding, Sports Understanding
3. **Symbolic reasoning** — Last Letter Concatenation, Coin Flip (state tracking)

Для каждой категории — свой набор chain-of-thought демонстраций.

## Emergent ability: CoT работает только при масштабе

![[02 Areas/ML & DL/raw/papers/cot/images/cot-examples-fig3.png]]
*Примеры chain-of-thought рассуждений по трём типам задач: arithmetic, commonsense и symbolic reasoning (источник: Wei et al., 2022)*

**Критический результат**: CoT — **emergent ability**, появляющаяся только при размере модели $\geq \sim 100B$ параметров.

### GSM8K (math word problems):

| Модель | Standard | CoT | Delta |
|--------|----------|-----|-------|
| LaMDA 137B | 17.9% | 17.9% | **+0** (нет улучшения!) |
| GPT-3 175B | ~18% | ~46% | **+28%** |
| PaLM 540B | ~33% | **57%** | **+24%** |
| PaLM 540B + Self-Consistency | — | **74.4%** | SOTA |

**Для моделей <100B параметров CoT не помогает** — маленькие модели генерируют fluent but illogical цепочки рассуждений, которые приводят к **худшим** результатам, чем standard prompting.

Почему? Маленькие модели не могут поддерживать **логическую когерентность** на протяжении длинной цепочки: каждый шаг выглядит правдоподобно, но связь между шагами ломается.

### Сложность задачи

CoT помогает **больше** для сложных задач:
- **GSM8K** (multi-step math): +24-28% improvement
- **SingleOp** (one-step math, easiest subset of MAWPS): **негативное или нулевое** improvement

Интуитивно: если задача решается за один шаг, «промежуточные» рассуждения — лишний шум.

## Zero-shot CoT: «Let's think step by step»

Kojima et al. (2022) обнаружили поразительно простой трюк: достаточно добавить к вопросу фразу **«Let's think step by step.»** — и модель начинает генерировать reasoning chain без единого few-shot примера.

```
Zero-shot Standard:
Q: [вопрос]
A:                                 ← модель сразу даёт ответ

Zero-shot CoT:
Q: [вопрос]
A: Let's think step by step.      ← trigger phrase
   [модель генерирует reasoning]
   Therefore, the answer is X.
```

**Двухэтапный процесс:**
1. Модель генерирует reasoning chain после trigger phrase
2. Из reasoning chain извлекается финальный ответ

Zero-shot CoT слабее few-shot CoT, но **не требует написания демонстраций** — огромное практическое преимущество. Вариации trigger phrases: «Let me solve this step by step», «Think carefully about this problem», «Break this down into steps».

## Self-Consistency: sampling + majority vote

Self-Consistency (Wang et al., 2023, ICLR 2023) — мощное расширение CoT. Проблема greedy decoding: модель генерирует **одну** цепочку, которая может содержать ошибку. Self-Consistency:

1. **Sample** K diverse reasoning paths при temperature > 0 (обычно K=40)
2. Из каждого пути извлечь финальный ответ
3. **Majority vote** — выбрать наиболее частый ответ

```
Path 1: 23 - 20 = 3, 3 + 6 = 9.   Answer: 9  ✓
Path 2: 23 + 6 = 29, 29 - 20 = 9.  Answer: 9  ✓
Path 3: 20 - 6 = 14, 23 - 14 = 9.  Answer: 9  ✓
Path 4: 23 - 20 = 3, 3 - 6 = -3.   Answer: -3 ✗

Majority vote: 9 (3 из 4 путей)
```

Интуиция: **правильный ответ обычно достижим несколькими путями**, а ошибочные пути дают разные неправильные ответы. Majority vote фильтрует outlier-ошибки.

### Результаты Self-Consistency vs CoT

| Dataset | CoT (greedy) | + Self-Consistency | Improvement |
|---------|-------------|-------------------|-------------|
| GSM8K | 56.5% | **74.4%** | **+17.9%** |
| SVAMP | 79.0% | **90.0%** | **+11.0%** |
| AQuA | 35.8% | **48.0%** | **+12.2%** |
| StrategyQA | 73.4% | **79.8%** | **+6.4%** |

Self-Consistency — **бесплатное** улучшение (не требует дополнительного обучения, только больше inference compute). Трейдофф: K inference calls вместо одного.

## Tree of Thoughts (ToT): поиск в пространстве рассуждений

Tree of Thoughts (Yao et al., 2023, NeurIPS 2023) — обобщение CoT. Если CoT — это **линейная** цепочка рассуждений, то ToT — **дерево**, где модель может:

1. **Генерировать** несколько кандидатов на каждом шаге (branching)
2. **Оценивать** промежуточные состояния (self-evaluation)
3. **Backtrack** — откатываться при обнаружении тупика

### Четыре компонента ToT

| Компонент | Роль | Реализация |
|-----------|------|-----------|
| Thought decomposer | Разбивает задачу на шаги | Task-specific decomposition |
| Thought generator | Генерирует кандидатов для каждого шага | LLM sampling / propose |
| State evaluator | Оценивает перспективность состояния | LLM «Is this path promising?» |
| Search algorithm | Обходит дерево | **BFS** или **DFS** |

### Пример: Game of 24

Задача: из четырёх чисел с помощью +, -, *, / получить 24.

```
Input: 4 9 10 13

CoT (линейная цепочка):
  4 + 9 = 13, 13 + 10 = 23, 23 + 13 = 36 ← тупик

ToT (дерево с backtracking):
  Branch 1: 13 - 9 = 4 → 4 * 4 = 16 → тупик → backtrack
  Branch 2: 13 - 10 = 3 → 9 - 3 = 6 → 6 * 4 = 24 → РЕШЕНИЕ ✓
```

**Результаты Game of 24:**
- GPT-4 + standard CoT: **4%** success rate
- GPT-4 + ToT: **74%** success rate

ToT превосходит CoT на задачах, требующих **планирования** и **отката** (backtracking), но значительно дороже: десятки LLM calls на одну задачу.

## Почему CoT работает: четыре свойства

Из оригинальной статьи (Wei et al., 2022):

1. **Decomposition** — CoT позволяет модели разбить сложную задачу на подзадачи. Каждый шаг рассуждения — отдельная подзадача, для которой достаточно «одного вычислительного шага».

2. **Additional compute** — цепочка рассуждения — это дополнительные токены, а значит дополнительные forward passes. Модель получает **больше вычислительных ресурсов** для сложных задач (adaptive computation).

3. **Interpretability** — reasoning chain = окно в процесс «мышления» модели. Можно debug'ить, на каком шаге ошибка: «модель правильно вычислила 23-20=3, но ошиблась на 3+6».

4. **Generality** — CoT применим к любой задаче, которую человек может решить через рассуждение на естественном языке: математика, логика, commonsense, символьные манипуляции.

## CoT в instruction tuning

Flan-T5/PaLM (Chung et al., 2022) включает CoT данные в обучающий mix для instruction tuning. Модель обучается на примерах с reasoning chains → CoT reasoning **трансферится** даже на задачи, для которых не было CoT training data.

Это показывает, что CoT — не только prompting trick, но и **learnable skill**, который можно усилить обучением.

## Ограничения CoT

1. **Faithfulness** — reasoning chain может быть **post-hoc rationalization**, а не реальный процесс «мышления». Модель может прийти к ответу другим путём, а chain — лишь правдоподобная «отговорка». (Turpin et al., 2023: biased features в промпте могут менять ответ, не влияя на reasoning chain.)

2. **Scale requirement** — CoT работает только при $\geq 100B$ параметров. Для моделей 7-13B нужны специальные подходы (distillation of CoT data, fine-tuning на CoT traces).

3. **Simple tasks hurt** — на задачах, решаемых за один шаг, CoT добавляет шум и может ухудшить результат.

4. **Cost** — reasoning chain = больше токенов = дороже inference. Self-Consistency умножает это на K. ToT — ещё дороже.

## Хронология

| Год | Milestone | Статья |
|-----|-----------|--------|
| 2020 | GPT-3: few-shot prompting без reasoning | Brown et al. |
| 2021 | Scratchpad: reasoning через промежуточные вычисления | Nye et al. |
| **2022** | **Chain-of-Thought prompting** | **Wei et al. (NeurIPS)** |
| 2022 | Zero-shot CoT: «Let's think step by step» | Kojima et al. |
| 2023 | Self-Consistency: sample + majority vote | Wang et al. (ICLR) |
| 2023 | Tree of Thoughts: deliberate problem solving | Yao et al. (NeurIPS) |
| 2023 | Least-to-Most: decompose → solve sub-problems | Zhou et al. |
| 2024 | OpenAI o1: internal CoT с RL training | OpenAI |
| 2025 | «Thinking» models: CoT как часть training, не prompting | DeepSeek-R1, Claude |

## Эволюция: от prompting к training

Оригинальный CoT (2022) — чисто prompting technique. К 2024-25 CoT эволюционировал:

- **Process Reward Models (PRM)**: обучение reward model оценивать каждый шаг рассуждения (не только финальный ответ)
- **o1-style reasoning**: модель обучена через RL генерировать **внутренние** рассуждения, не показывая их пользователю
- **Inference-time compute scaling**: quality ∝ compute at inference (больше tokens of thought → лучше ответ)

## Related concepts

- [[02 Areas/ML & DL/Concepts/Inference/In-context Learning|In-context Learning]] — CoT — частный случай ICL с reasoning chains
- [[02 Areas/ML & DL/Concepts/Inference/Prompting|Prompting]] — CoT — техника промптинга
- [[02 Areas/ML & DL/Concepts/NLP/Emergent Abilities|Emergent Abilities]] — CoT работает только при $\geq 100B$
- [[02 Areas/ML & DL/Concepts/Inference/Prompt Engineering|Prompt Engineering]] — практическое применение CoT

## Дополнительные ресурсы

- [Wei et al. — Chain-of-Thought Prompting (2022)](https://arxiv.org/abs/2201.11903) — оригинальная статья
- [Kojima et al. — Zero-shot Reasoners (2022)](https://arxiv.org/abs/2205.11916) — «Let's think step by step»
- [Wang et al. — Self-Consistency (2023)](https://arxiv.org/abs/2203.11171) — sample + majority vote
- [Yao et al. — Tree of Thoughts (2023)](https://arxiv.org/abs/2305.10601) — BFS/DFS в пространстве мыслей
- [LearnPrompting — Chain of Thought Guide](https://learnprompting.org/blog/guide-to-chain-of-thought-part-one) — практический гайд
