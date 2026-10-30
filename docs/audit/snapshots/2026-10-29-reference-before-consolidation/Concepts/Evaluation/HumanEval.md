---
title: "HumanEval"
aliases: [HumanEval, pass@k, Codex, HumanEval+, Code Generation Benchmark]
type: concept
status: legacy
category: Evaluation
papers:
  - "[[02 Areas/ML & DL/Papers/HumanEval|HumanEval]]"
courses: []
sources:
  - "[Chen et al. (2021) — Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)"
  - "[GitHub — openai/human-eval](https://github.com/openai/human-eval)"
  - "[Deepgram — HumanEval LLM Benchmark](https://deepgram.com/learn/humaneval-llm-benchmark)"
  - "[Papers With Code — HumanEval SOTA](https://paperswithcode.com/sota/code-generation-on-humaneval)"
  - "[IBM — What is HumanEval?](https://www.ibm.com/think/topics/humaneval)"
---

# HumanEval: Evaluating Large Language Models Trained on Code

## Зачем это нужно: BLEU для кода не работает

До HumanEval генеративные модели для кода оценивались через **match-based метрики** (exact match, BLEU score). Проблема фундаментальна: пространство функционально эквивалентных программ **огромно и сложно**. Два совершенно разных куска кода могут делать одно и то же (другие имена переменных, другая структура, другой алгоритм), а два похожих по тексту -- разное.

Авторы показывают это на данных: программы, которые **не проходят** unit tests (функционально неэквивалентны), часто имеют **более высокий BLEU**, чем правильные решения. BLEU просто не подходит для оценки кода.

Решение из software engineering давно известно: **test-driven development** -- код правильный, если проходит тесты. HumanEval переносит эту идею на evaluation LLM.

## Бенчмарк: 164 рукописных задачи

### Дизайн

| Параметр | Значение |
|----------|----------|
| Количество задач | 164 |
| Среднее число unit tests | 7.7 на задачу |
| Язык | Python |
| Сложность | Simple software interview questions |
| Источник | Hand-written (не из интернета!) |

Каждая задача содержит:
- **Function signature** -- имя функции, параметры, типы
- **Docstring** -- описание задачи на естественном языке
- **Body** -- reference solution (не показывается модели)
- **Unit tests** -- для проверки корректности

Задачи тестируют: language comprehension, algorithms, simple mathematics. Критически важно, что задачи **написаны вручную** и не скопированы из существующих источников, так как модели обучены на GitHub, который содержит решения из Codeforces, LeetCode и т.д.

## Метрика pass@k: ключевой вклад

### Идея

Генерируем $k$ программ для каждой задачи. Задача считается решённой, если **хотя бы одна** из $k$ программ проходит все unit tests. Это отражает реальность: программист редко пишет правильный код с первой попытки.

### Unbiased Estimator

Наивный подсчёт pass@k имеет высокую дисперсию. Авторы предлагают unbiased estimator: генерируем $n \geq k$ сэмплов, считаем $c$ правильных:

$$\text{pass@k} = \mathbb{E}_{\text{Problems}} \left[ 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}} \right]$$

Numerically stable реализация:

```python
def pass_at_k(n, c, k):
    if n - c < k: return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
```

Это **не** наивная формула $1 - (1 - \hat{p})^k$, которая даёт biased оценку (доказательство в Appendix A статьи).

### Оптимальная температура зависит от k

| k | Оптимальная T |
|---|---------------|
| 1 | 0.2 (низкая, консервативная) |
| 100 | 0.8 (высокая, diverse) |

При низком $k$ нужны **надёжные** решения (low temperature). При высоком $k$ нужно **разнообразие** -- больше шансов найти хотя бы одно правильное среди разных подходов. Это наблюдение предвосхитило [[02 Areas/ML & DL/Concepts/Reasoning/Self-Consistency|Self-Consistency]].

## Codex: модель для генерации кода

### Архитектура и обучение

| Параметр | Значение |
|----------|----------|
| Базовая модель | GPT (до 12B параметров) |
| Fine-tuning data | 54M публичных GitHub репозиториев (159 GB Python) |
| Специальные токены | Дополнительные whitespace tokens (~30% сжатие кода) |
| Codex-S | + fine-tuning на standalone correct functions |

### Результаты

| Модель | pass@1 | pass@10 | pass@100 |
|--------|--------|---------|----------|
| GPT-3 | 0% | - | - |
| GPT-J 6B | 11.4% | 15.7% | 27.7% |
| Codex-300M | 13.2% | 20.4% | 36.3% |
| **Codex-12B** | **28.8%** | **46.8%** | **72.3%** |
| **Codex-S-12B** | **37.7%** | **55.5%** | **77.5%** |

**GPT-3 решает 0% задач** -- general LM без fine-tuning на коде бесполезна. Специализация критична: GPT-J 6B (~Codex-300M) -- модель в 40 раз меньше, чем GPT-3, но обученная на коде, уже лучше.

### Sample Selection без oracle

При k=100, oracle (знаем правильный ответ) даёт 77.5%. Без oracle, **mean log-probability ranking** даёт **44.5%** -- значительно лучше random (28.8%). Это показывает, что log-probability -- разумный proxy для корректности кода.

## Ограничения Codex и HumanEval

### Scaling complexity

Performance **экспоненциально падает** с числом chained building blocks. Codex может написать отдельную функцию, но плохо справляется с задачами, требующими **композиции** нескольких операций.

### Prompt sensitivity

Баги в промпте (неправильное описание в docstring) приводят к генерации кода, точно воспроизводящего баг. Модель следует описанию, а не intent.

### 164 задачи -- мало

HumanEval стал настолько популярным, что результаты могли быть contaminated через training data. Это мотивировало создание расширенных версий.

## Эволюция: от HumanEval к современным бенчмаркам

| Бенчмарк | Год | Задач | Особенности |
|----------|-----|-------|-------------|
| **HumanEval** | 2021 | 164 | Foundational, hand-written |
| MBPP | 2021 | 974 | Google, crowd-sourced basic Python |
| **HumanEval+** | 2023 | 164 | 80x больше тестов (avg 774), находит баги в "правильных" решениях |
| MBPP+ | 2023 | 399 | Расширенные тесты для MBPP |
| SWE-bench | 2024 | 2,294 | Real GitHub issues, full repository context |
| **BigCodeBench** | 2024 | 1,140 | Library usage, real-world programming |
| HumanEval-V | 2024 | - | Visual reasoning + code generation |
| LiveCodeBench | 2024 | Обновляется | Свежие задачи, исключает contamination |

### HumanEval+ (Liu et al., 2023)

Ключевое открытие: стандартные тесты HumanEval **недостаточны**. HumanEval+ добавляет в среднем 774 теста на задачу (vs 7.7 в оригинале) и обнаруживает, что **многие "правильные" решения на самом деле содержат баги**, не пойманные оригинальными тестами. Pass@1 моделей падает на 10-20% при переходе от HumanEval к HumanEval+.

## Прогрессия на HumanEval (pass@1)

| Модель | Год | pass@1 |
|--------|-----|--------|
| GPT-3 | 2021 | 0% |
| Codex-12B | 2021 | 28.8% |
| GPT-3.5 | 2022 | ~48% |
| GPT-4 | 2023 | ~67% |
| Claude 3 Opus | 2024 | ~85% |
| GPT-4o | 2024 | ~90% |
| Claude 3.5 Sonnet | 2024 | ~92% |
| Frontier models | 2025 | **>95%** |

**От 0% до 95% за 4 года** -- одна из самых впечатляющих кривых прогресса в AI.

## Практическое значение pass@k

Различие pass@1 vs pass@k имеет прямое практическое значение:

- **Production (k=1)**: нужна надёжная генерация, низкая температура, **pass@1** -- ключевая метрика
- **Agentic loops (k=many)**: можно генерировать много кандидатов и выбирать лучший (по тестам, по log-prob, по LLM-оценке), **pass@k** -- ключевая метрика
- **Human-in-the-loop**: разработчик просматривает несколько вариантов -- промежуточный случай

## Related concepts

- [[02 Areas/ML & DL/Concepts/Evaluation/MMLU|MMLU]] -- foundational бенчмарк для knowledge/reasoning (как HumanEval для code)
- [[02 Areas/ML & DL/Concepts/Reasoning/Self-Consistency|Self-Consistency]] -- repeated sampling и выбор лучшего (аналог pass@k для reasoning)
- [[02 Areas/ML & DL/Concepts/Inference/Sampling|Sampling]] -- температура и diversity при генерации

## Дополнительные ресурсы

- [GitHub — openai/human-eval](https://github.com/openai/human-eval) -- бенчмарк и framework
- [Papers With Code — HumanEval Leaderboard](https://paperswithcode.com/sota/code-generation-on-humaneval) -- актуальные результаты
- [EvalPlus — HumanEval+](https://evalplus.github.io/) -- расширенная версия с дополнительными тестами
- [BigCodeBench](https://bigcode-bench.github.io/) -- современный бенчмарк для real-world coding
