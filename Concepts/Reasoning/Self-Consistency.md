---
title: "Self-Consistency"
aliases: [Self-Consistency, CoT-SC, Majority Voting, self-consistency decoding]
type: concept
category: Reasoning
papers: ["[[02 Areas/ML & DL/Papers/Self-Consistency|Self-Consistency]]"]
courses: []
sources:
  - "[Wang et al. — Self-Consistency Improves Chain of Thought Reasoning in Language Models (ICLR 2023)](https://arxiv.org/abs/2203.11171)"
---

# Self-Consistency

## Зачем это нужно: одного рассуждения недостаточно

Chain-of-Thought (CoT) prompting научил LLM «думать вслух» — генерировать пошаговое рассуждение перед ответом. Но стандартный CoT использует **greedy decoding**: модель генерирует **один** reasoning path и даёт **один** ответ. Если этот единственный путь содержит ошибку — ответ неверный.

Представь, что ты решаешь сложную математическую задачу. Ты можешь решить её несколькими способами: через алгебру, через подбор, через визуализацию. Если три разных подхода дают ответ «42», а четвёртый — «38», скорее всего «42» правильно. **Правильные рассуждения, даже разные, сходятся к одному ответу. Неправильные — расходятся.**

Именно эту интуицию формализовали Wang et al. (Google Research, 2023) в **Self-Consistency**: вместо одного greedy path — **множество** sampled reasoning paths → **majority vote** по финальным ответам.

## Алгоритм: три шага

### Шаг 1: Prompt

Стандартный CoT prompt с few-shot примерами, включающими пошаговые рассуждения:

```
Q: If there are 3 cars in the parking lot and 2 more arrive, 
   how many cars are in the parking lot?
A: There are 3 cars already. 2 more arrive. Now there are 
   3 + 2 = 5 cars. The answer is 5.
...
Q: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast 
   and bakes muffins with 4. She sells the rest for $2 per egg. 
   How much does she make every day?
A:
```

### Шаг 2: Sample diverse reasoning paths

Вместо greedy decoding — **семплируем** $m$ independent reasoning paths из decoder (temperature/top-k/nucleus sampling):

**Path 1**: «She uses 3 + 4 = 7 eggs. She has 16 - 7 = 9 left. 9 * $2 = **$18**.»
**Path 2**: «She sells the remainder for $2 * (16 - 4 - 3) = $2 * 9 = **$18**.»
**Path 3**: «She uses 3 + 4 = 7 eggs. She sells 7 * $2 = **$14**.» (ошибка!)

Каждый path может использовать **разную стратегию** решения и прийти к **разному ответу**.

### Шаг 3: Marginalize — majority vote

$$\hat{a} = \arg\max_a \sum_{i=1}^{m} \mathbb{1}(a_i = a)$$

Просто считаем, какой ответ встречается **чаще всего** среди $m$ paths. В примере выше: $18 появляется 2 раза, $14 — 1 раз. Ответ: **$18**.

По сути — **Monte Carlo approximation** правильного ответа через маргинализацию по reasoning paths.

## Почему majority vote, а не weighted average?

Казалось бы, нужно учитывать вероятности каждого path:

$$P(r_i, a_i | \text{prompt, question})$$

Wang et al. протестировали несколько стратегий агрегации на PaLM-540B:

| Стратегия | GSM8K | AQuA | SVAMP |
|-----------|-------|------|-------|
| Greedy decode | 56.5 | 35.8 | 79.0 |
| Weighted avg (unnormalized) | 56.3 | 35.8 | 73.0 |
| Weighted sum (normalized) | 74.1 | 48.0 | 86.8 |
| **Majority vote (unweighted)** | **74.4** | **48.3** | **86.6** |

Majority vote работает **так же хорошо**, как normalized weighted sum. Причина: LLM **плохо откалиброваны** — normalized вероятности разных paths примерно равны. Модель считает все свои генерации «одинаково вероятными», поэтому weighting не помогает.

Это также объясняет, почему **отдельные verifier'ы** (Cobbe et al., 2021) обучались для ранжирования решений — LLM сами не могут надёжно отличить правильное решение от неправильного по probability.

## Результаты: поразительные улучшения

### Arithmetic Reasoning

| Модель | Метод | GSM8K | AQuA | SVAMP | MultiArith |
|--------|-------|-------|------|-------|------------|
| **PaLM-540B** | CoT-greedy | 56.5 | 35.8 | 79.0 | 94.7 |
| **PaLM-540B** | **Self-Consistency** | **74.4 (+17.9)** | **48.3 (+12.5)** | **86.6 (+7.6)** | **99.3 (+4.6)** |
| GPT-3 code-002 | CoT-greedy | 60.1 | 39.8 | 75.8 | 96.2 |
| GPT-3 code-002 | **Self-Consistency** | **78.0 (+17.9)** | **52.0 (+12.2)** | **86.8 (+11.0)** | **100.0 (+3.8)** |

+17.9% на GSM8K — это **огромный** прирост. Для сравнения: обучение отдельного verifier'а на 7.5K примерах (Cobbe et al.) давало comparable результаты. Self-Consistency достигает этого **без обучения**, **без дополнительных моделей**, **без аннотаций**.

### Commonsense Reasoning

| Модель | Метод | CSQA | StrategyQA | ARC-challenge |
|--------|-------|------|------------|---------------|
| PaLM-540B | CoT-greedy | 79.0 | 75.3 | 85.2 |
| PaLM-540B | **Self-Consistency** | **80.7 (+1.7)** | **81.6 (+6.3)** | **88.7 (+3.5)** |

Приросты поменьше, но **стабильные** на всех задачах.

### Новые SOTA

Self-Consistency достигла **новых SOTA** на почти всех задачах — несмотря на то, что это **unsupervised, task-agnostic** метод без fine-tuning. Результаты превосходят системы с task-specific обучением и тысячами training examples.

## Scale dependency: чем больше модель, тем больше прирост

Критическое наблюдение: прирост от self-consistency **растёт** с размером модели:

| Размер модели | Типичный прирост | Пример |
|---------------|------------------|--------|
| UL2-20B | +3-6% | GSM8K: 4.1 → 7.3 (+3.2) |
| LaMDA-137B | +9-23% | MultiArith: 51.8 → 75.7 (+23.9) |
| **PaLM-540B** | **+5-18%** | **GSM8K: 56.5 → 74.4 (+17.9)** |

Интуиция: маленькие модели генерируют **плохие** reasoning paths → большинство paths неправильные → majority vote не помогает. Большие модели генерируют **в основном правильные** paths с разной структурой → majority vote эффективно отфильтровывает редкие ошибки.

## Оптимальное число paths

Из экспериментов с LaMDA-137B:

- **1 path** = greedy decode (baseline)
- **5 paths** = значительный прирост
- **10 paths** = ещё лучше
- **20 paths** = почти оптимально
- **40 paths** = оптимально (используется в статье)
- **>40 paths** = diminishing returns

Рекомендация: **40 paths** — стандартный выбор. Для production можно останавливаться раньше, если высокая consistency (все paths дают одинаковый ответ → нет смысла семплировать больше).

## Robustness

Self-Consistency **робастна** к:

- **Sampling strategy**: temperature sampling, top-k, nucleus — всё работает. Авторы использовали $T = 0.5{-}0.7$, $k = 40$
- **Imperfect prompts**: даже с неоптимальными CoT примерами self-consistency улучшает результат
- **Different CoT exemplars**: разные наборы примеров дают сходные улучшения

## Uncertainty estimation: бонусный эффект

Уровень согласованности (какая доля paths сошлась на одном ответе) **сильно коррелирует** с accuracy:

- Если 38 из 40 paths дали одинаковый ответ — почти наверняка правильный
- Если paths разделились 15/14/11 — модель «не уверена»

Это даёт **бесплатную uncertainty estimation**: модель может «знать, когда не знает». В production это позволяет:
- Отправлять неуверенные случаи на human review
- Показывать confidence score пользователю
- Решать, нужен ли retrieval для уточнения

## Zero-shot CoT + Self-Consistency

Self-Consistency работает не только с few-shot CoT, но и с **zero-shot CoT** («Let's think step by step»):

PaLM-540B на GSM8K: zero-shot CoT-greedy 43.0% → zero-shot CoT + Self-Consistency **69.2%** (+26.2%).

+26.2% — **ещё больший прирост**, чем с few-shot CoT. Это делает self-consistency особенно ценной когда нет хороших few-shot примеров.

## Связь с другими методами

### Self-Consistency vs Tree of Thoughts (ToT)

ToT (Yao et al., 2023) можно рассматривать как обобщение self-consistency:
- **Self-Consistency** = ToT с breadth $k$, depth 1, majority vote (explore many paths, no backtracking)
- **ToT** = arbitrary depth, с self-evaluation и backtracking (explore tree of possibilities)

Self-Consistency проще и дешевле. ToT мощнее, но требует больше inference compute.

### Self-Consistency vs Ensemble

Self-Consistency — это **не ensemble** в классическом смысле:
- Ensemble: несколько моделей, агрегация outputs
- Self-Consistency: **одна модель**, разные sampling paths («self-ensemble»)

Не нужно обучать несколько моделей — только один чекпоинт + multiple samplings.

### Self-Consistency vs Verifier

Cobbe et al. (2021) обучали **отдельный verifier** (175B параметров) на 7.5K примерах для ранжирования CoT solutions. Self-Consistency достигает **comparable results без обучения verifier'а**.

## Ограничения

- **Compute cost**: $m$ reasoning paths = $m \times$ inference cost. При $m = 40$ и длинных CoT — дорого
- **Fixed answer set**: majority vote работает только для задач с **дискретным** множеством ответов. Для open-text generation нужна другая метрика consistency
- **Не работает на слабых моделях**: если модель генерирует в основном неправильные paths, majority vote не поможет (мусор на входе → мусор на выходе)
- **Не для всех задач**: на задачах, где CoT не помогает (простая classification), self-consistency тоже не помогает

## Key papers

- [[02 Areas/ML & DL/Papers/Self-Consistency|Self-Consistency]] — majority voting over CoT paths, +17.9% GSM8K без обучения (Wang et al., ICLR 2023)

## Related concepts

- [[02 Areas/ML & DL/Concepts/Inference/Chain of Thought|Chain of Thought]] — основа, на которой строится self-consistency
- [[02 Areas/ML & DL/Concepts/Reasoning/Tree of Thoughts|Tree of Thoughts]] — обобщение: tree search вместо flat sampling
- [[02 Areas/ML & DL/Concepts/Inference/Sampling|Sampling]] — стратегии генерации diverse paths (temperature, top-k, top-p)
- [[02 Areas/ML & DL/Concepts/Inference/Prompting|Prompting]] — CoT prompting как prerequisite

## Дополнительные ресурсы

- [Wang et al. — Self-Consistency (ICLR 2023)](https://arxiv.org/abs/2203.11171) — оригинальная статья
- [Wei et al. — Chain-of-Thought Prompting (NeurIPS 2022)](https://arxiv.org/abs/2201.11903) — CoT, на котором построена self-consistency
- [Cobbe et al. — Training Verifiers to Solve Math Word Problems (2021)](https://arxiv.org/abs/2110.14168) — альтернативный подход через обученный verifier
