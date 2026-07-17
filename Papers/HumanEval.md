---
title: "Evaluating Large Language Models Trained on Code"
url: https://arxiv.org/abs/2107.03374
authors: [Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, Wojciech Zaremba]
year: 2021
date_reviewed: 2026-04-07
type: source-note
status: legacy
category: paper
tags:
  - Code-Generation
  - Benchmark
  - Evaluation
  - LLM
Date: 2021-07-14
Organization: OpenAI, Anthropic AI
Status: Done
concepts:
  - "[[02 Areas/ML & DL/Concepts/Evaluation/HumanEval|HumanEval]]"
  - "[[02 Areas/ML & DL/Concepts/Evaluation/pass@k|pass@k]]"
  - "[[02 Areas/ML & DL/Concepts/Code Generation/Code Generation|Code Generation]]"
  - "[[02 Areas/ML & DL/Concepts/Evaluation/Functional Correctness|Functional Correctness]]"
raw: "[[02 Areas/ML & DL/raw/papers/humaneval/paper.txt]]"
---

# Evaluating Large Language Models Trained on Code (Codex / HumanEval)

**Authors:** Mark Chen, Jerry Tworek, Heewoo Jun et al.
**Published:** 2021 (arXiv:2107.03374v2)
**URL:** https://arxiv.org/abs/2107.03374

## TL;DR

Статья представляет **Codex** — GPT модель, fine-tuned на публичном коде с GitHub (159GB Python), и **HumanEval** — бенчмарк из **164 рукописных задач** для оценки генерации кода по docstrings. Ключевая метрика — **pass@k** (функциональная корректность через unit tests). Codex-12B решает 28.8% задач (pass@1), а с 100 сэмплами — 72.3% (pass@100). Supervised fine-tuning (Codex-S) доводит pass@1 до 37.7%. BLEU score плохо коррелирует с функциональной корректностью.

## Problem

1. **Оценка генерации кода:** Существующие метрики (BLEU, exact match) не отражают **функциональную корректность** — множество функционально эквивалентных программ имеют разный синтаксис
2. **Отсутствие бенчмарка:** Нет стандартного набора задач для оценки code synthesis из docstrings. Существующие (APPS) содержат задачи из интернета, на которых модели могли обучаться
3. **Возможности LLM для кода:** GPT-3 показал зачатки программирования несмотря на отсутствие специализированного обучения — потенциал для fine-tuned моделей не исследован

## Method

### HumanEval

- **164 задачи**, написанные вручную (не скопированы из интернета)
- Каждая задача: function signature + docstring + body + unit tests (в среднем 7.7 тестов)
- Тестирует: language comprehension, алгоритмы, простую математику
- Сложность сопоставима с simple software interview questions

### Метрика pass@k

Генерируем n >= k сэмплов на задачу, считаем c правильных, вычисляем **unbiased estimator:**

```
pass@k = E[1 - C(n-c, k) / C(n, k)]
```

Это устраняет high variance проблему наивного подсчёта.

### Codex

- **Данные:** 54M public GitHub repos, 179GB Python files (после фильтрации — 159GB)
- **Архитектура:** GPT (до 12B параметров), fine-tuned from GPT-3
- **Tokenizer:** GPT-3 tokenizer + дополнительные whitespace tokens (~30% сжатие кода)
- **Обучение:** 100B tokens, cosine LR decay

### Codex-S (supervised fine-tuning)

Дополнительное fine-tuning на:
1. **Competitive programming** — 10,000 задач с сайтов соревнований
2. **Continuous integration** — ~40,000 функций из open source проектов с CI (tracing inputs/outputs через sys.setprofile)

### Codex-D (docstring generation)

Обратная задача: генерация docstring по коду. Используется для back-translation ranking.

### Sandbox

gVisor container runtime + eBPF firewall для безопасного исполнения сгенерированного кода.

## Key Results

### Scaling (HumanEval)

| Модель | pass@1 | pass@10 | pass@100 |
|--------|--------|---------|----------|
| GPT-Neo 2.7B | 6.41% | 11.27% | 21.37% |
| GPT-J 6B | 11.62% | 15.74% | 27.74% |
| Codex-300M | 13.17% | 20.37% | 36.27% |
| Codex-2.5B | 21.36% | 35.42% | 59.5% |
| **Codex-12B** | **28.81%** | **46.81%** | **72.31%** |
| **Codex-S-12B** | **37.7%** | — | **77.5%** |

- Test loss масштабируется как **power law** от размера модели: (N / 5.92e7)^(-0.13)
- GPT-J 6B ~ Codex-300M (20x меньше параметров) — специализация критична

### Температура и diversity

- Оптимальная температура зависит от k: T*=0.2 для pass@1, T*=0.8 для pass@100
- Высокая температура = больше diversity = лучше pass@k при больших k

### Sample selection (без oracle)

Когда нет unit tests для выбора лучшего сэмпла:
- **Mean log-probability ranking:** 44.5% (vs 28.8% random, vs 77.5% oracle) для Codex-12B при 100 сэмплах
- **Back-translation (Codex-D):** хуже log-prob ranking, но лучше random
- **Sum log-probability:** может быть **хуже** random (длинные ответы получают преимущество)

### BLEU vs Functional Correctness

Распределения BLEU scores для правильных и неправильных решений **сильно перекрываются** — BLEU ненадежный proxy для корректности.

### Ограничения Codex

1. **Длинные цепочки операций:** Performance экспоненциально падает с ростом числа chained building blocks (factor 2-3x на каждый шаг)
2. **Binding operations to variables:** Ошибки при привязке операций к переменным в сложных docstrings
3. **Prompt sensitivity:** Если prompt содержит баги, Codex генерирует худший код — эффект усиливается с ростом модели
4. **Sample efficiency:** Модель обучена на сотнях миллионов строк кода, в то время как студент CS-101 решает больше задач

## My notes

- HumanEval стал **стандартом** оценки code generation (наряду с MBPP). К 2026 году frontier models решают >90% pass@1
- Ключевой вклад — **pass@k метрика** и unbiased estimator. До этого код оценивали через BLEU, что бессмысленно для функциональной корректности
- Repeated sampling — "удивительно эффективная стратегия": pass@100 = 72.3% vs pass@1 = 28.8%. Это предвосхитило идеи [[02 Areas/ML & DL/Papers/Self-Consistency|Self-Consistency]]: diversity + selection > single best attempt
- Температура и k — взаимосвязаны. Это важный урок: для production (k=1) нужна низкая T; для agentic loops (k=many) — высокая T
- Codex-S показывает силу **distribution matching**: fine-tuning на задачах формата "signature + docstring -> body" вместо сырого кода дает +8.9% pass@1
- Ограничение HumanEval: 164 задачи — слишком мало. Привело к созданию HumanEval+, MBPP, SWE-bench и др.
- Safety concern: модель может генерировать вредоносный код, уязвимый код, или код, нарушающий авторские права. Sandbox для execution — необходимость
- Связь с [[02 Areas/ML & DL/Papers/MMLU|MMLU]]: оба — бенчмарки для оценки capabilities LLM, но MMLU = knowledge, HumanEval = coding skills. Вместе покрывают ключевые измерения
