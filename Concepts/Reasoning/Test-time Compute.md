---
title: "Test-time Compute"
aliases: [Test-time Compute, Inference-time Scaling, Test-time Scaling, TTS, Thinking Tokens]
type: concept
status: legacy
category: Reasoning
papers:
  - "[[02 Areas/ML & DL/Papers/DeepSeek-R1 Reasoning via RL|DeepSeek-R1]]"
courses: []
sources:
  - "[HuggingFace Blog — What is Test-time Compute and How to Scale It?](https://huggingface.co/blog/Kseniase/testtimecompute)"
  - "[Build ML — Test-Time Compute Scaling: A Practical Guide](https://buildml.substack.com/p/test-time-compute-scaling-a-practical)"
  - "[Introl Blog — Inference-Time Scaling Research](https://introl.com/blog/inference-time-scaling-research-reasoning-models-december-2025)"
  - "[Emerge Haus — Test-Time Compute in Generative AI](https://www.emerge.haus/blog/test-time-compute-generative-ai)"
---

# Test-time Compute

## Зачем это нужно: новая ось масштабирования

До 2024 года доминировала одна парадигма масштабирования LLM:

$$\text{Better AI} = \text{Bigger Model} + \text{More Data} + \text{More Training Compute}$$

[[02 Areas/ML & DL/Concepts/Training/Scaling Laws|Scaling Laws]] (Kaplan et al., 2020; Hoffmann et al., 2022) формализовали: качество растёт как power law от размера модели и объёма данных. Это привело к гонке: GPT-3 (175B) → GPT-4 (~1.8T MoE) → каждое следующее поколение больше и дороже.

**Проблема:** train-time scaling упирается в потолок — стоимость обучения растёт суперлинейно, а returns diminishing. Обучение GPT-4, по внешним оценкам, стоило около 100 миллионов долларов; следующее поколение может оказаться ещё дороже.

**Test-time compute** (inference-time scaling) — открытие **второй оси**:

$$\text{Better AI} = \text{Good Model} + \text{More Thinking at Inference}$$

Вместо обучения модели побольше — дать существующей модели **больше времени на размышление**.

## Ключевой эксперимент: o1

### Что сделал OpenAI (сентябрь 2024)

OpenAI o1 — первая модель, целенаправленно обученная **тратить inference compute на рассуждения**:

1. Модель получает вопрос
2. Генерирует **internal reasoning tokens** (скрытый «блокнот»)
3. Проходит через несколько шагов рассуждения: постановка задачи → разбор подзадач → попытки решения → проверка → финальный ответ
4. Пользователь видит только финальный ответ

**Результат:** o1 потратила 10-100x больше токенов, чем GPT-4o, но решила задачи, недоступные GPT-4o — олимпиадная математика (AIME: 83%), PhD-уровень science (GPQA: 78%).

### Scaling curve: thinking tokens vs accuracy

Ключевое наблюдение — **log-linear scaling**:

$$\text{Accuracy} \propto \log(\text{Test-time Compute})$$

При увеличении «бюджета» на размышление в 10x, accuracy на сложных задачах растёт на ~10-15 percentage points. Это **новый scaling law** — аналог scaling laws для training, но на оси inference.

## Механизмы test-time compute

### 1. Extended Chain-of-Thought

Модель генерирует длинную цепочку рассуждений перед ответом. Это наиболее распространённый подход:

```
Prompt: "Solve: Find all primes p such that p^2 + 2 is also prime."

<think>
Let me think about this systematically.
If p = 2: p^2 + 2 = 6 = 2×3, not prime.
If p = 3: p^2 + 2 = 11, prime! So p = 3 works.
If p > 3: p is odd, so p^2 is odd, so p^2 + 2 is odd. Good.
Wait, let me think modulo 3...
If p ≡ 1 (mod 3): p^2 ≡ 1, p^2 + 2 ≡ 0 (mod 3). Divisible by 3!
If p ≡ 2 (mod 3): p^2 ≡ 1, p^2 + 2 ≡ 0 (mod 3). Also divisible by 3!
So for any prime p > 3, p^2 + 2 is divisible by 3 and > 3, hence not prime.
Therefore only p = 3.
</think>

Answer: The only prime p such that p^2 + 2 is also prime is p = 3.
```

### 2. Self-Verification

Модель проверяет свои шаги и возвращается назад при обнаружении ошибки:

- **Forward pass:** решить задачу
- **Verification:** проверить каждый шаг
- **Backtrack:** если ошибка найдена — откатить и попробовать альтернативный путь

Это emergence из RL-обучения: модель «поняла», что self-verification увеличивает вероятность правильного ответа → увеличивает reward.

### 3. Majority Voting / Best-of-N

Простейший подход без специального обучения:
1. Сгенерировать $N$ ответов на один вопрос
2. Выбрать наиболее частый (majority voting) или лучший по reward model (best-of-N)

Масштабирование: $N$ растёт → accuracy растёт → compute растёт линейно.

### 4. Tree Search

Более структурированный подход:
- Модель исследует **дерево** рассуждений
- В каждом узле — несколько возможных следующих шагов
- Process Reward Model оценивает промежуточные шаги
- Beam search или MCTS выбирает лучший путь

## Как обучить модель «думать дольше»

### Подход DeepSeek-R1: pure RL

1. Pre-trained модель (без reasoning training)
2. [[02 Areas/ML & DL/Concepts/Training/GRPO|GRPO]] с reward = правильность ответа
3. Модель **сама** учится генерировать длинные reasoning traces
4. Длина CoT растёт органически в процессе RL

**Emergence:** модель открывает, что более длинные рассуждения → лучше reward → оптимизатор поощряет длинные рассуждения.

### Подход o1: (предположительно) RL + process supervision

1. SFT на human-written reasoning traces
2. RL с process reward model (оценка каждого шага)
3. Hidden thinking tokens (пользователь не видит reasoning)

### Подход Qwen3: thinking budget

Пользователь задаёт **максимальное число thinking tokens**. Модель адаптирует глубину рассуждения:
- Простой вопрос → 0 thinking tokens
- Средний вопрос → 500 thinking tokens
- Сложная олимпиада → 10,000+ thinking tokens

## Зачем это paradigm shift

### Train-time vs Test-time: сравнение

| Свойство | Train-time scaling | Test-time scaling |
|----------|-------------------|-------------------|
| Когда compute тратится | Один раз при обучении | На каждый запрос |
| Стоимость | Фиксированная, очень высокая | Variable, per-query |
| Benefit | Все задачи | Особенно сложные задачи |
| Diminishing returns | Да (power law) | Менее выражены |
| Flexibility | Фиксирована после training | Адаптируется к задаче |

### Экономический аргумент

**Train-time:** обучение модели оплачивается один раз целиком, после чего одна и та же модель обрабатывает и «2 + 2», и «докажи теорему».

**Test-time:** дешёвая модель + много думания на сложных задачах. На простых задачах — мгновенный ответ, на сложных — 10x compute. **Средняя** стоимость запроса может быть ниже, при **лучшем** качестве на сложных задачах.

### Compute-Optimal Inference

Аналогия с Chinchilla для training: для каждой задачи существует **оптимальное соотношение** model size и test-time compute. Маленькая модель с большим thinking budget может превзойти большую модель с малым budget.

$$\text{Quality}(M, T) = f(\text{model\_size}(M), \text{test\_compute}(T))$$

Frontier модели будущего — не самые большие, а **самые эффективно думающие**.

## Timeline

| Дата | Milestone | Модель |
|------|-----------|--------|
| 2022 | Chain-of-Thought prompting | Wei et al. |
| 2023 | Tree of Thoughts | Yao et al. |
| 2024 Sep | o1 — первая reasoning model | OpenAI |
| 2025 Jan | DeepSeek-R1 — open-weight reasoning | DeepSeek |
| 2025 Jan | Kimi k1.5 — multimodal reasoning | Moonshot |
| 2025 May | Qwen3 — unified thinking/non-thinking | Alibaba |

## Related concepts

- [[02 Areas/ML & DL/Concepts/Architectures/DeepSeek-R1|DeepSeek-R1]] — open-weight reasoning model
- [[02 Areas/ML & DL/Concepts/Architectures/Kimi|Kimi]] — multimodal reasoning через RL
- [[02 Areas/ML & DL/Concepts/Architectures/Qwen3|Qwen3]] — thinking budget mechanism
- [[02 Areas/ML & DL/Concepts/Training/GRPO|GRPO]] — RL алгоритм для обучения reasoning
- [[02 Areas/ML & DL/Concepts/Inference/Chain of Thought|Chain of Thought]] — базовая техника reasoning
- [[02 Areas/ML & DL/Concepts/Training/Scaling Laws|Scaling Laws]] — train-time аналог
- [[02 Areas/ML & DL/Concepts/Reasoning/Tree of Thoughts|Tree of Thoughts]] — structured test-time compute
