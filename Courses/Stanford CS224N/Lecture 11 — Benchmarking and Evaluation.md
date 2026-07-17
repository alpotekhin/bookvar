---
title: "CS224N — Lecture 11: Benchmarking and Evaluation"
course: "Stanford CS224N"
lecture: 11
type: course-note
raw: "[[02 Areas/ML & DL/raw/courses/Stanford CS224N/slides/cs224n-spr2024-lecture11-evaluation-yann]]"
concepts: ["[[02 Areas/ML & DL/Concepts/NLP/Benchmarking|Benchmarking]]", "[[02 Areas/ML & DL/Concepts/Evaluation/MMLU|MMLU]]", "[[02 Areas/ML & DL/Concepts/NLP/SuperGLUE|SuperGLUE]]", "[[02 Areas/ML & DL/Concepts/NLP/BLEU Score|BLEU]]", "[[02 Areas/ML & DL/Concepts/NLP/LLM-as-Judge|LLM-as-Judge]]"]
---

# Lecture 11: Benchmarking and Evaluation

> *"Benchmarks and how we drive the progress of the field."* -- Yann Dubois

Лектор: Yann Dubois.

## Зачем измерять performance?

Разные **цели** оценки предъявляют разные **требования**:

| Цель | Требования |
|------|-----------|
| **Train** | Супер быстро, дёшево, дифференцируемо (loss function), без shortcuts |
| **Develop** | Быстро, дёшево, без shortcuts |
| **Model selection** | Быстро, дёшево, стандартизировано, воспроизводимо |
| **Publish** | ~дёшево, crude metrics допустимы, fine-grained distinguishability |
| **Deploy** | Trustworthy, task-specific, абсолютные показатели |

## Close-ended evaluation (классификация)

### Что такое close-ended задачи

Ограниченное число возможных ответов, часто один правильный. Позволяют **автоматическую** оценку стандартными ML-метриками.

### Ключевые задачи и датасеты

| Задача | Датасеты |
|--------|----------|
| Sentiment analysis | SST, IMDB, Yelp |
| Entailment (NLI) | SNLI |
| Named Entity Recognition | CoNLL-2003 |
| Part-of-Speech tagging | PTB |
| Coreference resolution | WSC |
| Question Answering | SQuAD 2 |

### [[02 Areas/ML & DL/Concepts/NLP/SuperGLUE|SuperGLUE]]: multi-task NLU benchmark

Попытка измерить "general language capabilities". Включает:
- **BoolQ, MultiRC**: reading comprehension
- **CB, RTE**: entailment
- **COPA**: причинно-следственные связи
- **ReCoRD**: QA + reasoning
- **WiC**: значения слов в контексте
- **WSC**: coreference resolution

### Выбор метрик

Accuracy, Precision, Recall, F1, ROC AUC -- выбор зависит от задачи. Проблема **агрегации**: как объединить метрики по разным задачам? Простое среднее может быть misleading.

### Spurious correlations

Пример на SNLI (Gururangan et al. 2019):
- Premise: *"The economy could be still better"*
- Hypothesis: *"The economy has never been better"*
- Модель может использовать **negation** ("never") как shortcut для предсказания entailment, игнорируя смысл

SNLI сам по себе сложен, но в данных могут быть **ненайденные** корреляции, позволяющие модели "жульничать".

## Open-ended evaluation (генерация текста)

### Проблема

Длинные генерации с **слишком большим числом** правильных ответов, чтобы их перечислить. Стандартные ML-метрики неприменимы. Есть не "правильные/неправильные", а "лучшие/худшие" ответы.

### Content overlap метрики

Ref: *"They walked to the grocery store."*
Gen: *"The woman went to the hardware store."*

N-gram overlap (BLEU, ROUGE, METEOR, CIDEr) -- лексическое сходство с reference.

**Проблема**: нет понятия семантической близости! Пример failure case:
- *"Are you enjoying the CS224N lectures?"*
- "Heck yes!" vs "Yes!" → score 0.67 vs 0.25
- "Yup." → score **0** (false negative!)
- "Heck no!" → score **0.67** (false positive!)

### Model-based метрики

Используют **learned representations** для семантического сходства:

- **BERTScore** (Zhang et al. 2020): cosine similarity контекстуальных BERT-эмбеддингов. Matching words между candidate и reference через cosine similarity.
- **BLEURT** (Sellam et al. 2020): regression model на базе BERT, оценивает грамматичность и передачу смысла.

**Критическое ограничение**: reference-based метрики хороши только настолько, насколько хороши references. Если reference некачественный -- корреляция с human judgment пропадает.

### Human evaluation

Золотой стандарт для text generation. Оценка по измерениям:
- Fluency, coherence, consistency
- Factuality, correctness
- Commonsense, style, grammaticality, redundancy

**Проблемы** human evaluation:
- **Дорого и медленно**
- **Inter-annotator disagreement**: особенно для субъективных задач
- **Intra-annotator disagreement**: один человек оценивает по-разному в разное время
- **Невоспроизводимо**: лишь ~5% human evaluations реально повторяемы (20% при помощи авторов)
- **Biases**: если incentive -- $/час, annotators ищут shortcuts

## [[02 Areas/ML & DL/Concepts/NLP/LLM-as-Judge|LLM-as-Judge]]: Reference-free evaluation

### Мотивация

Human evaluation дорого и медленно. Идея: пусть **LLM оценивает** качество ответов.

### Chatbot Arena (LMSYS)

Side-by-side comparison: пользователь общается с двумя моделями одновременно, даёт thumbs up/down. Формирует **ELO рейтинг** моделей.

**Минусы**: требует масштабного community effort, новые модели долго бенчмаркятся.

### AlpacaEval

1. Для каждой инструкции: сгенерировать output от baseline и от модели
2. Попросить GPT-4 оценить вероятность, что output модели лучше
3. **AlpacaEval LC**: reweight по длине (контроль spurious correlation с длиной)
4. Усреднить win probability → **win rate**

Результаты: 98% корреляция с Chatbot Arena, <3 мин, <$10.

### MT-Bench

6 категорий multi-turn задач, GPT-4 оценивает по 10-балльной шкале.

### Осторожности с LLM-as-Judge

- **Spurious correlations**: длина ответа, позиция (хотя позицию рандомизируют)
- **Self-bias**: GPT-4 немного предпочитает свои ответы, но **удивительно слабо**
- Люди имеют **низкое agreement** между собой из-за variance -- LLM-as-Judge может быть **более консистентным**, чем humans!

## [[02 Areas/ML & DL/Concepts/Evaluation/MMLU|MMLU]]: Massive Multitask Language Understanding

Hendrycks et al. 2021: **57 knowledge-intensive задач** (от медицины до юриспруденции). Формат: multiple-choice questions. Стал де-факто стандартом для оценки LLM knowledge.

## Другие capability benchmarks

### Code evaluation
- **HumanEval**: human-written coding problems
- Метрика: **Pass@1** (один сгенерированный output проходит тесты)
- GPT-4: ~67%

### Agent evaluation
- LLM используются для управления агентами -- оценка в sandbox environments
- Сложнее стандартного text evaluation

## Текущее состояние оценки LLM

### Три уровня

| Этап | Метрика | Применение |
|------|---------|-----------|
| **Pretraining** | Perplexity | Коррелирует с downstream performance, но зависит от данных и tokenizer |
| **Finetuned** | HELM, Open LLM Leaderboard | Множество автоматически оцениваемых бенчмарков |
| **Chat/Deployment** | Arena-like (ELO) | Chatbot Arena, AlpacaEval |

### Проблемы бенчмарков

- **Data contamination**: тестовые данные могут попасть в training set. Решения: held-out data, canary strings, temporal splits
- **Benchmark saturation**: модели быстро достигают human-level (SuperGLUE "решён" за 2 года)
- **Goodhart's Law**: "когда метрика становится целью, она перестаёт быть хорошей метрикой"
- **Construct validity**: измеряем ли мы то, что хотим?

## Concepts covered

- [[02 Areas/ML & DL/Concepts/NLP/Benchmarking|Benchmarking]] -- стандартизированная оценка моделей, close-ended vs open-ended
- [[02 Areas/ML & DL/Concepts/Evaluation/MMLU|MMLU]] -- 57-task knowledge benchmark для LLM
- [[02 Areas/ML & DL/Concepts/NLP/SuperGLUE|SuperGLUE]] -- multi-task NLU benchmark
- [[02 Areas/ML & DL/Concepts/NLP/BLEU Score|BLEU]] -- n-gram overlap метрика и её ограничения
- [[02 Areas/ML & DL/Concepts/NLP/LLM-as-Judge|LLM-as-Judge]] -- AlpacaEval, MT-Bench, Chatbot Arena
