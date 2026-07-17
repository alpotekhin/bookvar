---
title: "Flan-T5"
aliases: [Flan T5, flan-t5]
type: concept
category: Architectures
papers:
  - "[[02 Areas/ML & DL/Papers/Flan-T5-PaLM]]"
  - "[[02 Areas/ML & DL/Papers/T5]]"
courses: []
sources:
  - "[Chung et al. — Scaling Instruction-Finetuned Language Models (2022)](https://arxiv.org/abs/2210.11416)"
  - "[Google AI Blog — Flan-T5](https://ai.googleblog.com/2023/02/the-flan-collection-advancing-open.html)"
  - "[HuggingFace Flan-T5 Collection](https://huggingface.co/google/flan-t5-xxl)"
---

# Flan-T5

## Зачем эта модель появилась

К 2022 году instruction tuning (обучение модели следовать инструкциям) становилось ключевым способом улучшения LLM. InstructGPT показал это для GPT-3, FLAN (2021) — для PaLM. Но оставались открытые вопросы:

1. **Масштабируется ли instruction tuning по числу задач?** (от сотен к тысячам)
2. **Масштабируется ли оно по размеру модели?** (от Small до 540B)
3. **Как chain-of-thought данные влияют на reasoning?**

Chung et al. (Google, 2022) в работе «Scaling Instruction-Finetuned Language Models» систематически ответили на все три вопроса. **Flan-T5** — это [[02 Areas/ML & DL/Concepts/Architectures/T5|T5]], instruction-tuned на **1836 задачах** по рецепту из этой работы.

## Ключевая идея: instruction tuning масштабируется по трём осям

### Ось 1: Число задач (282 → 1836)

Авторы объединили три источника задач:

| Источник | Число датасетов | Примечание |
|----------|-----------------|------------|
| FLAN (Wei et al., 2021) | ~62 | Оригинальные FLAN задачи |
| T0++ (Sanh et al., 2021) | ~55 | Promptsource templates |
| Super-Natural Instructions (Wang et al., 2022) | **~1600** | Crowd-sourced инструкции |
| CoT datasets | 9 | GSM8K, CommonsenseQA, CREAK, etc. |
| **Итого** | **~1836** | — |

Результат: performance **монотонно растёт** с числом задач от 282 до 1836. **Сатурация не достигнута** — больше задач = лучше.

### Ось 2: Размер модели

Instruction tuning помогает **больше для больших моделей**:

| Модель | Без IT | С IT | Дельта |
|--------|--------|------|--------|
| Flan-T5-Small (80M) | Baseline | Иногда хуже | **Нестабильно** |
| Flan-T5-Base (250M) | Baseline | Немного лучше | +small |
| Flan-T5-Large (780M) | Baseline | Значительно лучше | ++medium |
| Flan-T5-XL (3B) | Baseline | Значительно лучше | +++large |
| **Flan-PaLM (540B)** | PaLM | **+9.4% average** | **++++huge** |

**Важное наблюдение:** маленькие модели (Small, Base) могут **деградировать** от instruction tuning — capacity слишком мала, чтобы одновременно решать 1836 разных задач. Instruction tuning — техника **для больших моделей**.

### Ось 3: Chain-of-Thought данные

Включение 9 CoT датасетов в instruction tuning данные даёт **двойной эффект**:

1. **Улучшает CoT reasoning** (ожидаемо) — модель учится пошаговому reasoning
2. **Не вредит non-CoT tasks** (неожиданно) — стандартный performance не падает
3. **Без CoT данных** — reasoning на CoT бенчмарках **деградирует** даже по сравнению с базовой моделью

Это значит: для reasoning capabilities нужно **явно включать CoT данные** в instruction tuning, а не просто надеяться на prompting.

## Формат данных

Каждая задача представлена в нескольких форматах:

```
# Zero-shot формат:
"Classify the sentiment of the following review: 'Great movie!' → "

# Few-shot формат:
"Review: 'Terrible' → negative
 Review: 'Amazing' → positive
 Review: 'Great movie!' → "

# CoT формат:
"Q: If John has 3 apples and gives away 1, how many does he have?
 Let's think step by step.
 John starts with 3 apples. He gives away 1. 3 - 1 = 2.
 The answer is 2."
```

Модель обучается на **смеси всех форматов** — это учит её работать как в zero-shot, так и в few-shot режиме.

## Ключевые результаты

### Flan-PaLM 540B (headline results)

| Benchmark | PaLM 540B | Flan-PaLM 540B | Дельта |
|-----------|-----------|----------------|--------|
| **MMLU** (5-shot) | 69.3 | **75.2** | +5.9% |
| **TyDiQA** (multilingual, 1-shot) | — | **+14.9%** | +14.9% |
| **MGSM** (math, underrep. langs) | — | **+8.1%** | +8.1% |
| Average (across benchmarks) | Baseline | **+9.4%** | +9.4% |

MMLU 75.2% — **SOTA** на момент публикации (до GPT-4).

### Flan-T5: маленькие модели, большие результаты

| Модель | Params | vs GPT-3 (175B) |
|--------|--------|-----------------|
| Flan-T5-Large | 780M | Конкурентна на нескольких бенчмарках |
| **Flan-T5-XL** | **3B** | **Превосходит на большинстве MMLU задач** |
| Flan-T5-XXL | 11B | Значительно лучше |

**Flan-T5-XL (3B) > GPT-3 (175B)** на MMLU — демонстрация того, что instruction tuning может **компенсировать 50x разницу в масштабе**.

## Flan-T5 vs InstructGPT/ChatGPT

Обе модели используют instruction tuning, но с разными подходами:

| Аспект | Flan-T5 | InstructGPT/ChatGPT |
|--------|---------|---------------------|
| Base model | T5 (encoder-decoder) | GPT-3 (decoder-only) |
| Instruction tuning | Supervised only | SFT + **RLHF** |
| Число задач | **1836** | ~thousands (не раскрыто) |
| CoT данные | Да (9 датасетов) | Предположительно да |
| Human feedback | Нет | **Да (reward model + PPO)** |
| Открытость | Полностью открыта | Закрыта |

Flan-T5 показала, что **instruction tuning без RLHF** уже даёт значительное улучшение. RLHF добавляет ещё один уровень improvement, но не является обязательным для базовой instruction-following capability.

## Практическое использование Flan-T5

Flan-T5 открыла **демократизацию instruction-tuned моделей**:

| Размер | Use case | Hardware |
|--------|----------|----------|
| Flan-T5-Small (80M) | Прототипирование, тесты | CPU |
| Flan-T5-Base (250M) | Edge/mobile | CPU/small GPU |
| Flan-T5-Large (780M) | Production NLU/NLG | Single GPU |
| Flan-T5-XL (3B) | General-purpose assistant | Single GPU (16GB) |
| Flan-T5-XXL (11B) | Complex reasoning, research | Multi-GPU |

**Encoder-decoder преимущество**: Flan-T5 естественно работает с structured input→output задачами (summarization, translation, paraphrasing), где decoder-only модели требуют больше prompt engineering.

## Влияние на field

### 1. Instruction tuning как commodity

До Flan-T5 instruction tuning выглядел как «секретный sauce» OpenAI. После — стал **стандартной техникой**, доступной каждому. Alpaca, Vicuna, и тысячи других instruction-tuned моделей последовали этому пути.

### 2. Data curation > model size

Flan-T5-XL (3B) > GPT-3 (175B) — один из самых ярких примеров того, что **качество данных** важнее размера модели. Это предвосхитило Chinchilla и PaLM 2.

### 3. Мост между T5 и ChatGPT

Flan-T5 показала: T5 + instruction tuning = сильная модель. ChatGPT = GPT-3 + instruction tuning + RLHF. Разница — в RLHF, и Flan-T5 позволила количественно оценить **вклад каждого компонента**.

### 4. CoT in training data

До Flan-T5 CoT (chain-of-thought) рассматривался как **prompting technique**. Flan-T5 показала, что CoT нужно включать **в training data** — это фундаментально улучшает reasoning, а не просто активирует скрытые capabilities.

## Key papers

- [[02 Areas/ML & DL/Papers/Flan-T5-PaLM]] — оригинал «Scaling Instruction-Finetuned Language Models» (Chung et al., 2022)
- [[02 Areas/ML & DL/Papers/T5]] — базовая архитектура

## Related concepts

- [[02 Areas/ML & DL/Concepts/Architectures/T5|T5]] — базовая encoder-decoder архитектура
- [[02 Areas/ML & DL/Concepts/Architectures/Encoder-Decoder|Encoder-Decoder]] — архитектурный класс
- [[02 Areas/ML & DL/Concepts/Training/Instruction Tuning|Instruction Tuning]] — ключевая техника
- [[02 Areas/ML & DL/Concepts/Inference/Chain of Thought|Chain-of-Thought]] — CoT в training data
- [[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]] — дополнительный alignment (Flan-T5 его не использует)
- [[02 Areas/ML & DL/Concepts/Architectures/PaLM 2|PaLM 2]] — Flan-PaLM как companion модель

## Дополнительные ресурсы

- [Paper (arXiv)](https://arxiv.org/abs/2210.11416) — полная работа
- [The Flan Collection (Google Blog)](https://ai.googleblog.com/2023/02/the-flan-collection-advancing-open.html) — описание датасета
- [HuggingFace Flan-T5-XXL](https://huggingface.co/google/flan-t5-xxl) — крупнейший открытый чекпоинт
- [Flan Collection GitHub](https://github.com/google-research/FLAN) — код и данные
