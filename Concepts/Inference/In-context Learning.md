---
title: "In-context Learning"
aliases: [ICL, in-context learning, few-shot learning, few-shot prompting]
type: concept
category: Inference
papers:
  - "[[02 Areas/ML & DL/Papers/GPT 3.0|GPT-3 (Brown et al., 2020)]]"
  - "[[02 Areas/ML & DL/Papers/Pre-train, Prompt, and Predict- A Systematic Survey of Prompting Methods in Natural Language Processing|Prompting Survey (Liu et al., 2021)]]"
sources:
  - "[Brown et al. — GPT-3 (2020)](https://arxiv.org/abs/2005.14165)"
  - "[Stanford SAIL — Understanding In-Context Learning](https://ai.stanford.edu/blog/understanding-incontext/)"
  - "[Xie et al. — ICL as Implicit Bayesian Inference (2022)](https://arxiv.org/abs/2111.02080)"
  - "[Olsson et al. — In-context Learning and Induction Heads (2022)](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)"
  - "[von Oswald et al. — Transformers learn in-context by gradient descent (2023)](https://arxiv.org/abs/2212.07677)"
courses: []
---

# In-context Learning (ICL)

## Зачем это нужно: обучение без обучения

Традиционный ML workflow: собрать датасет → обучить модель → задеплоить. Каждая новая задача требует нового цикла fine-tuning с новыми данными, GPU-часами и инженерными усилиями.

**In-context learning** (ICL) — радикально другой подход: модель «обучается» новой задаче **прямо в момент inference**, используя только несколько примеров в промпте. Никаких обновлений весов. Никакого gradient descent. Только forward pass.

```
Промпт (контекст = "обучающие данные"):
────────────────────────────────────────
Input: "La maison est grande."
Output: "The house is big."

Input: "Bonjour le monde."
Output: "Hello world."

Input: "Le chat dort sur le canapé."
Output:                                    ← модель продолжает паттерн
────────────────────────────────────────
Модель: "The cat sleeps on the couch."
```

Модель **не обновляет веса** — она «выводит» задачу (перевод FR→EN) из паттерна примеров и применяет её к новому входу. Всё это происходит за один forward pass через трансформер.

## Открытие: GPT-3 (Brown et al., 2020)

GPT-3 систематизировал три режима inference без fine-tuning:

| Режим | Демонстрации | Описание |
|-------|-------------|----------|
| **Zero-shot** | 0 | Только описание задачи: «Translate French to English:» |
| **One-shot** | 1 | Одна пара input→output + новый input |
| **Few-shot** | K (10-100) | K пар input→output + новый input |

**ICL = few-shot режим** — модель получает $K$ демонстрационных пар $(x_1, y_1), \ldots, (x_K, y_K)$ и должна предсказать $y$ для нового $x$.

### Ключевые результаты GPT-3

**TriviaQA** (closed-book QA):

| Режим | GPT-3 175B | Улучшение |
|-------|-----------|-----------|
| Zero-shot | 64.3% | — |
| One-shot | 68.0% | +3.7 |
| Few-shot (K=64) | **71.2%** | **+6.9** |

Каждый дополнительный пример **систематически улучшает** результат. Это было удивительным: модель не обучается на примерах (нет gradient updates), но предсказание улучшается с их количеством.

### Scaling: ICL как emergent ability

ICL работает **только при достаточном масштабе**. Рисунок 1.2 из GPT-3:

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/in-context-learning/icl-scaling-model-size.png]]
*Larger models gain more from in-context examples: few-shot performance scales steeply with model size, while zero-shot improves more gradually (источник: Brown et al., 2020, Figure 1.2)*

Маленькие модели (125M-1.3B) **не используют** демонстрации эффективно — adding examples не помогает. Большие модели (13B+) — каждый пример существенно улучшает результат. Это одна из причин, почему ICL считается **emergent ability** (Wei et al., 2022).

## Как это работает: теоретические объяснения

ICL — одна из самых загадочных способностей LLM. Модель не обновляет веса, но «учится» из примеров. Несколько конкурирующих теорий:

### 1. Implicit Bayesian Inference (Xie et al., 2022)

Самая формальная теория. Идея: при pretraining на разнообразных текстах модель учится выводить **скрытый концепт** документа из контекста. Документ о медицине → модель «активирует» медицинский концепт. Документ о программировании → активирует другой.

ICL — тот же механизм: примеры в промпте — evidence для скрытого концепта (задачи). Модель делает **Bayesian inference**:

$$p(\text{concept} | \text{demonstrations}) \propto p(\text{demonstrations} | \text{concept}) \cdot p(\text{concept})$$

Prior $p(\text{concept})$ — из pretraining (какие задачи модель видела). Likelihood — из демонстраций. Posterior — «какую задачу нужно решить».

**Ключевой результат:** ICL emerges, когда pretraining distribution — **mixture of concepts** (mixture of HMMs в теоретической модели). Модель учится при pretraining различать концепты → при inference использует примеры для выбора нужного.

**Следствие для prompt engineering:** ICL работает за счёт сигнала в четырёх компонентах:
1. **Input distribution** — примеры из правильного домена
2. **Output space** — корректный формат/набор ответов
3. **Format** — consistent formatting между примерами
4. **Input-output mapping** — правильные соответствия

Удивительный факт: эксперименты Xie et al. показали, что замена правильных ответов на **случайные метки** (random labels) вызывает лишь 5-10% падение accuracy! Это значит, что input-output mapping — не главный сигнал. Главное — правильный **формат** и **домен** примеров.

### 2. Implicit Gradient Descent (von Oswald et al., 2023)

Вторая теория: Transformer при forward pass **имплицитно выполняет gradient descent**. Attention layers можно интерпретировать как один шаг GD на линейной модели, где:
- Примеры = training data
- Attention weights = gradient updates
- Новый вход = test input

Формально: для linear self-attention forward pass математически эквивалентен одному шагу GD на least-squares loss по демонстрациям.

$$\text{Transformer forward pass} \approx \text{GD step on } \mathcal{L}(\theta) = \sum_{k=1}^{K} \|f_\theta(x_k) - y_k\|^2$$

Это объясняет, почему ICL performance улучшается с числом примеров (больше «training data» → лучший «gradient step»).

### 3. Mesa-Optimization (Hubinger et al., 2019; von Oswald et al., 2023)

Более общая рамка: Transformer при pretraining становится **mesa-optimizer** — он обучает внутри себя subsidiary learning algorithm, который запускается при inference. Pretraining создаёт **optimizer**, а forward pass — это **optimization run** на данных из промпта.

Исследования (Uncovering mesa-optimization in Transformers, 2023) показали, что в Transformers, обученных на задачах sequence prediction, действительно возникает subsidiary gradient-based optimization в forward pass.

### 4. Induction Heads — механистическое объяснение (Olsson et al., 2022)

Самое конкретное, механистическое объяснение. **Induction heads** — пара attention голов в разных слоях, которые реализуют pattern completion:

**Голова 1 (previous-token head):** каждый токен attend'ит к предыдущему токену → копирует позиционную информацию.

**Голова 2 (induction head):** используя информацию от головы 1, ищет паттерн «[A][B]...[A]» и предсказывает [B].

```
Контекст: "Harry Potter is a wizard. The boy who lived..."
                                      ^
Когда модель видит "Harry Potter" второй раз, induction head 
находит: "Harry Potter" → "is" (паттерн из контекста)
и предсказывает следующий токен как "is".
```

**Phase change в training:** Olsson et al. обнаружили, что induction heads формируются в определённый момент обучения, и в **этот же момент** ICL ability резко улучшается (фазовый переход). Это первое causal evidence, что конкретный механизм (induction heads) ответственен за ICL.

## Факторы, влияющие на ICL

### Формат промпта (prompt formatting)

ICL **чувствителен** к formatting. Одна и та же задача с разным оформлением даёт разные результаты:

```
Формат A (хороший):          Формат B (плохой):
Input: "Bonjour"             Bonjour → Hello
Output: "Hello"              Merci → Thank you
                             
Input: "Merci"               Au revoir →
Output: "Thank you"

Input: "Au revoir"
Output:
```

### Порядок примеров (order sensitivity)

**Position bias**: последние примеры в промпте влияют сильнее, чем первые. Разный порядок тех же примеров может менять accuracy на 10-15%.

### Label bias (majority label)

Если 8 из 10 примеров — класс A и 2 — класс B, модель склоняется к A даже для примеров класса B. ICL **не** устойчив к дисбалансу классов в демонстрациях.

### Common token bias

Распространённые слова (yes, no, true, false) предсказываются чаще, чем редкие. Модель использует pretraining prior наряду с evidence из примеров.

### Число примеров (K)

ICL performance обычно растёт с $K$, но с убывающей отдачей. Оптимум зависит от задачи и context window. Для GPT-3 (2048 токенов): $K \leq 100$ демонстраций.

## ICL vs Fine-tuning: сравнение

| Критерий | ICL | Fine-tuning |
|----------|-----|-------------|
| Обновление весов | Нет | Да |
| Данные | K=1-100 примеров | 100-100K примеров |
| Время адаптации | Мгновенно | Часы-дни |
| Инфраструктура | Только inference | GPU для обучения |
| Качество | Уступает при K→∞ | SOTA при достаточных данных |
| Масштаб модели | >13B для reliability | Работает с малыми моделями |
| Multi-task | Один checkpoint → все задачи | Отдельная модель на задачу |

ICL не заменяет fine-tuning, но радикально снижает **порог входа**: для прототипа новой задачи достаточно 5-10 примеров и API call.

## Связь с другими концепциями

ICL — основа для целого дерева техник:

- **[[02 Areas/ML & DL/Concepts/Inference/Chain of Thought|Chain of Thought]]** — ICL с reasoning chains в примерах
- **Zero-shot prompting** — ICL с K=0, только task description
- **[[02 Areas/ML & DL/Concepts/Retrieval/Retrieval-Augmented Generation|RAG]]** — retrieved документы как «примеры» в контексте
- **Instruction tuning** (Flan, InstructGPT) — обучение модели лучше следовать ICL patterns

## Хронология

| Год | Milestone | Статья |
|-----|-----------|--------|
| 2019 | GPT-2 — первые наблюдения zero-shot capabilities | Radford et al. |
| **2020** | **GPT-3 — систематизация zero/one/few-shot ICL** | **Brown et al.** |
| 2021 | Survey: Prompt-based Learning taxonomy | Liu et al. |
| 2022 | ICL as Implicit Bayesian Inference | Xie et al. |
| 2022 | Induction Heads — mechanistic explanation | Olsson et al. |
| 2022 | Chain-of-Thought — ICL + reasoning | Wei et al. |
| 2023 | Transformers learn in-context by gradient descent | von Oswald et al. |
| 2023 | Uncovering mesa-optimization in Transformers | Bai et al. |
| 2024 | Many-shot ICL (>100 examples) с long context | Agarwal et al. |

## Открытые вопросы

1. **Почему induction heads достаточно?** — Induction heads реализуют pattern matching, но ICL работает на задачах, которые выходят за пределы simple copying.
2. **ICL vs task recognition** — модель «учится» новой задаче или «вспоминает» похожую задачу из pretraining?
3. **Scaling laws для ICL** — как ICL quality масштабируется с model size и number of examples?
4. **Adversarial ICL** — можно ли «обмануть» ICL специально подобранными misleading примерами?

## Related concepts

- [[02 Areas/ML & DL/Concepts/Inference/Chain of Thought|Chain of Thought]] — ICL + промежуточные шаги рассуждения
- [[02 Areas/ML & DL/Concepts/Inference/Prompting|Prompting]] — общая парадигма управления LLM через текст
- [[02 Areas/ML & DL/Concepts/NLP/Emergent Abilities|Emergent Abilities]] — ICL как одна из emergent abilities
- [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]] — induction heads как конкретный attention pattern
- [[02 Areas/ML & DL/Concepts/Architectures/GPT-3|GPT-3]] — модель, систематизировавшая ICL

## Дополнительные ресурсы

- [Brown et al. — GPT-3 (2020)](https://arxiv.org/abs/2005.14165) — оригинальная систематизация ICL
- [Stanford SAIL — How does ICL work?](https://ai.stanford.edu/blog/understanding-incontext/) — доступное объяснение Bayesian framework
- [Olsson et al. — Induction Heads (2022)](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — механистическое исследование
- [von Oswald et al. — ICL by gradient descent (2023)](https://arxiv.org/abs/2212.07677) — теоретическая работа
