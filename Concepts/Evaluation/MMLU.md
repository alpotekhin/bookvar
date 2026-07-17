---
title: "MMLU"
aliases: [MMLU, Massive Multitask Language Understanding, Massive Multitask Test]
type: concept
category: Evaluation
papers:
  - "[[02 Areas/ML & DL/Papers/MMLU|MMLU]]"
courses:
  - "[[02 Areas/ML & DL/Courses/Stanford CS224N/Lecture 11 — Benchmarking and Evaluation|CS224N Lecture 11]]"
sources:
  - "[Hendrycks et al. (2021) — Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)"
  - "[MMLU GitHub — hendrycks/test](https://github.com/hendrycks/test)"
  - "[MMLU Wikipedia](https://en.wikipedia.org/wiki/MMLU)"
  - "[Klu — MMLU Benchmark](https://klu.ai/glossary/mmlu-eval)"
---

# MMLU: Massive Multitask Language Understanding

## Зачем это нужно: разрыв между бенчмарками и реальными способностями

К 2020 году модели достигли superhuman performance на большинстве существующих бенчмарков:
- **GLUE** (Wang et al., 2018) -- побит за год
- **SuperGLUE** (Wang et al., 2019) -- побит за год
- **HellaSwag**, **Physical IQA**, **CosmosQA** -- commonsense бенчмарки, быстро решены

Но эти бенчмарки тестировали в основном **лингвистические навыки** (NLI, coreference, reading comprehension) или **базовый commonsense** -- вещи, которые умеет почти каждый ребёнок. При этом модели при pretraining видели **всю Wikipedia, тысячи книг, миллионы сайтов** -- огромный объём специализированных знаний, который никто не тестировал.

MMLU закрывает этот разрыв: тестирует знания и reasoning **от elementary до professional уровня** по 57 предметным областям.

## Дизайн бенчмарка

### 57 задач, 4 категории

| Категория | Примеры предметов |
|-----------|-------------------|
| **STEM** | Abstract Algebra, College Mathematics, Machine Learning, Computer Science, Physics, Chemistry |
| **Humanities** | Philosophy, World History, Jurisprudence, Formal Logic, Moral Scenarios |
| **Social Sciences** | Psychology, Economics, Sociology, Political Science |
| **Other** | Professional Medicine, Clinical Knowledge, Professional Law, Business Ethics, Nutrition |

### Уровни сложности

- **Elementary** -- базовые вопросы
- **High School** -- уровень AP экзаменов (e.g., AP Psychology)
- **College** -- университетский курс
- **Professional** -- уровень профессиональных экзаменов (USMLE, bar exam, EPPP)

### Формат: multiple choice (4 варианта)

```
The following are multiple choice questions (with answers) about high school mathematics.

How many numbers are in the list 25, 26, ..., 100?
(A) 75 (B) 76 (C) 22 (D) 23
Answer: B
```

Источники вопросов: practice exams (GRE, USMLE, bar exam), university courses, книги Oxford University Press.

### Статистика

| Параметр | Значение |
|----------|----------|
| Всего вопросов | 15,908 |
| Few-shot dev set | 5 на предмет |
| Validation set | 1,540 |
| Test set | 14,079 |
| Min questions per subject | 100 |

### Оценка

Zero-shot и few-shot (до 5 примеров) **без fine-tuning**. Промпт: *"The following are multiple choice questions (with answers) about [subject]."* Модель генерирует вероятности для токенов "A", "B", "C", "D" -- выбирается максимальная.

Методологическое новшество: **тестирование в zero/few-shot setting без train set**. Это предвосхитило парадигму "pretraining IS training" и ближе к тому, как мы тестируем людей.

## Human Baselines

| Тип | Accuracy |
|-----|----------|
| Amazon Mechanical Turk (неспециалисты) | **34.5%** |
| Random chance | 25% |
| Expert-level (95th percentile test-takers) | **~89.8%** |

Разрыв между неспециалистами (34.5%) и экспертами (~89.8%) огромен -- это не trivial commonsense, а **специализированные знания**.

## Прогрессия модельных результатов

| Модель | Год | Params | MMLU |
|--------|-----|--------|------|
| Random | - | - | 25.0% |
| GPT-3 (Small/Medium/Large) | 2020 | < 13B | ~25% (random) |
| **GPT-3 (XL)** | 2020 | **175B** | **43.9%** |
| UnifiedQA | 2020 | 11B | 48.9% |
| Chinchilla | 2022 | 70B | 67.5% |
| GPT-4 | 2023 | ~1.8T (MoE) | **86.4%** |
| Claude 3.5 Sonnet | 2024 | - | ~88% |
| GPT-4o | 2024 | - | **88.7%** |
| Llama 3.1 405B | 2024 | 405B | ~88% |
| GPT-4.1 | 2025 | - | ~90%+ |

### Ключевые наблюдения из оригинальной статьи

1. **Эмерджентное появление**: модели GPT-3 до 13B показывают **random chance** (25%). Только при 175B accuracy выходит за случайный уровень -- скачок на 20 процентных пунктов. Это одно из ранних наблюдений [[02 Areas/ML & DL/Concepts/NLP/Emergent Abilities|Emergent Abilities]].

2. **Лопсайдный performance**: GPT-3 достигает ~70% на лучшем предмете, но near-random на нескольких других. Модель не является экспертом **ни в одной** области.

3. **Проблемные области**:
   - **Moral Scenarios** -- критично для alignment
   - **Professional Law** -- сложные сценарии с неочевидными правовыми прецедентами
   - **Elementary Mathematics** -- procedural knowledge (GPT-3 плох в арифметике)

4. **Плохая калибровка**: разница между confidence и accuracy -- до **24%**. Модели не знают, чего они не знают.

5. **Fine-tuning помогает**: UnifiedQA 11B (48.9%) превосходит GPT-3 175B (43.9%) при 15x меньшем количестве параметров -- специализированное обучение на QA задачах компенсирует размер.

## Критика MMLU

### Формат multiple choice

Multiple choice позволяет **elimination strategy**: даже не зная правильный ответ, можно исключить заведомо неверные. Это отличается от open-ended generation, где модель должна сформулировать ответ сама.

### Contamination

Вопросы из публичных источников (practice exams) могут попадать в training data моделей. По мере роста pretraining корпусов эта проблема усиливается.

### Потолок достигнут

К 2024-2025 годам frontier models **"решили" MMLU** (>88-90%). Это привело к созданию более сложных бенчмарков:
- **MMLU-Pro** (2024) -- 10 вариантов вместо 4, более сложные вопросы, chain-of-thought reasoning
- **GPQA** (2024) -- вопросы, написанные PhD-экспертами, на которых даже другие эксперты ошибаются
- **ARC-AGI** -- тесты на обобщение

### Что MMLU не измеряет

- Способность генерировать связный текст
- Coding skills
- Многошаговое рассуждение (вопросы в основном однократные)
- Creativity и open-ended problem solving
- Real-world task completion

## Почему MMLU остаётся важным

Несмотря на критику, MMLU стал **де-факто стандартной метрикой** для сравнения LLM:
- Почти каждая новая модель сообщает MMLU score
- 57 предметов позволяют **диагностировать слабые области** модели
- Few-shot evaluation без fine-tuning -- чистый тест pretraining knowledge
- Предсказал важность **multitask knowledge** при масштабировании

## Related concepts

- [[02 Areas/ML & DL/Concepts/Evaluation/HumanEval|HumanEval]] -- аналогичный foundational бенчмарк для code generation
- [[02 Areas/ML & DL/Concepts/Inference/Few-shot Learning|Few-shot Learning]] -- MMLU тестирует в few-shot setting
- [[02 Areas/ML & DL/Concepts/Inference/Zero-shot Learning|Zero-shot Learning]] -- и в zero-shot setting
- [[02 Areas/ML & DL/Concepts/NLP/Emergent Abilities|Emergent Abilities]] -- MMLU стал ключевым бенчмарком для измерения emergence

## Дополнительные ресурсы

- [MMLU GitHub — hendrycks/test](https://github.com/hendrycks/test) -- код и данные
- [MMLU-Pro paper (2024)](https://arxiv.org/html/2406.01574v2) -- более сложная версия
- [MMLU Wikipedia](https://en.wikipedia.org/wiki/MMLU) -- актуальные результаты моделей
- [Klu — MMLU Benchmark](https://klu.ai/glossary/mmlu-eval) -- хороший обзор
