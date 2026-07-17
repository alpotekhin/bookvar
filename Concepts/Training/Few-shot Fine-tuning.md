---
title: "Few-shot Fine-tuning"
aliases: [few-sample fine-tuning, low-resource fine-tuning, малоданковая дообучение]
type: concept
category: Training
papers:
  - "[[02 Areas/ML & DL/Papers/Revisiting Few-sample BERT Fine-tuning]]"
  - "[[02 Areas/ML & DL/Papers/BERT]]"
courses: []
sources:
  - "[Zhang et al. -- Revisiting Few-sample BERT Fine-tuning (2021)](https://arxiv.org/abs/2006.05987)"
  - "[Mosbach et al. -- On the Stability of Fine-tuning BERT (2021)](https://arxiv.org/abs/2006.04884)"
  - "[Dodge et al. -- Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping (2020)](https://arxiv.org/abs/2002.06305)"
---

# Few-shot Fine-tuning

## Зачем это нужно: реальность low-data режимов

В учебниках fine-tuning BERT выглядит просто: загрузи pretrained модель, обучи 3 эпохи на downstream данных, получи SOTA. На практике это работает **только при достаточном количестве данных** (десятки тысяч примеров).

Но многие реальные задачи имеют **мало размеченных данных**:
- Медицинские тексты: разметка требует врачей, 500-1000 примеров -- роскошь
- Редкие языки: аннотаторов мало
- Узкие домены: юридические, финансовые, научные задачи
- Новые задачи: быстрое прототипирование с минимумом данных

При <10K примерах (часто <1K) стандартный fine-tuning BERT **катастрофически нестабилен**: разные random seeds дают от **SOTA до near-random** результатов. Zhang et al. (2021) систематически исследовали эту проблему и нашли три конкретные причины.

## Проблема 1: баг в BERTAdam (bias correction)

### Симптом

На RTE (2.4K train examples): **48% запусков** дают accuracy <55% (close to random для бинарной классификации). То есть **почти половина** экспериментов -- degenerate.

### Причина: пропущенная bias correction в Adam

Оригинальный BERT (Devlin et al., 2019) и HuggingFace Transformers (до июля 2019) использовали кастомный оптимизатор **BERTAdam**, в котором **пропущена bias correction** из оригинального Adam (Kingma & Ba, 2015):

```
Incorrect (BERTAdam):
  m_hat_t = m_t                 # нет деления на (1 - beta_1^t)
  v_hat_t = v_t                 # нет деления на (1 - beta_2^t)
  theta = theta - lr * m_hat_t / (sqrt(v_hat_t) + eps)

Correct (Adam):
  m_hat_t = m_t / (1 - beta_1^t)    # bias correction
  v_hat_t = v_t / (1 - beta_2^t)    # bias correction
  theta = theta - lr * m_hat_t / (sqrt(v_hat_t) + eps)
```

**Почему это критично?** В начале обучения $m_t$ и $v_t$ инициализированы нулями. Без bias correction:
- $\beta_1 = 0.9$: на шаге 1, $m_1 = 0.1 \cdot g_1$ вместо корректного $\hat{m}_1 = g_1$. Момент **занижен в 10 раз**.
- Эффективный learning rate **непредсказуемо скачет** в первые шаги обучения.

При большом датасете (MNLI, 393K) начальная нестабильность "сглаживается" за много шагов. При маленьком датасете (RTE, 2.4K) -- **3 эпохи = ~96 шагов**, и первые нестабильные шаги составляют значительную долю обучения.

### Исправление

Замена на стандартный `torch.optim.AdamW` (debiased):

| Оптимизатор | RTE (degenerate runs) | RTE (mean acc) |
|-------------|----------------------|----------------|
| BERTAdam (biased) | **48%** | ~60% |
| AdamW (debiased) | **~5%** | **69.5%** |

Одна строчка кода: `optimizer = AdamW(...)` вместо `BertAdam(...)`.

### Масштаб проблемы

Этот баг **распространился по всей экосистеме NLP**: тысячи статей 2018-2019 использовали BERTAdam. Многие "методы стабилизации" (Mixout, LLRD) на самом деле **косвенно компенсировали** этот баг, а не решали реальную проблему.

## Проблема 2: плохая инициализация верхних слоёв

### Гипотеза

Верхние слои BERT **специализируются на MLM** (pre-training task). Для downstream задачи (classification, NER) эти слои плохо инициализированы -- они "помнят" паттерны MLM, а не семантику целевой задачи.

### Решение: Re-init

Re-инициализация top $L$ BERT blocks + pooler из $\mathcal{N}(0, 0.02^2)$. $L$ подбирается по validation.

**Результаты** (с debiased Adam):

| Задача | Baseline (no re-init) | + Re-init top L | Gain |
|--------|----------------------|----------------|------|
| RTE (acc) | 69.5% | **72.6%** | +3.1% |
| MRPC (F1) | 90.8 | **91.4** | +0.6 |
| CoLA (MCC) | 63.0 | **64.2** | +1.2 |

### Интуиция

```
BERT Layers (12 blocks):
  Layer 1-4:   базовые linguistic features    <-- сохраняем
  Layer 5-8:   синтаксис, семантика           <-- сохраняем
  Layer 9-12:  специализация на MLM            <-- re-init!
  Pooler:      тоже MLM-специфичный            <-- re-init!
```

Re-init не уничтожает полезные знания: нижние слои (syntax, semantics) сохраняются. Верхние слои "освобождаются" от MLM-specific паттернов и могут быстрее адаптироваться к downstream задаче.

## Проблема 3: стандартные 3 эпохи -- слишком мало

### Симптом

На 1K примерах: 3 эпохи = **~96 шагов** (batch size 32). Модель просто не успевает сойтись.

### Решение: longer training

Вместо фиксированных 3 эпох -- подбирать число шагов из {200, 400, 800, 1600, 3200}.

**Результаты** (longer training + Re-init):

| Задача (1K примеров) | 3 эпохи | Longer training | Gain |
|---------------------|---------|-----------------|------|
| MRPC (F1) | 80.5 +/- 3.3 | **86.0 +/- 1.2** | +5.5, -2.1 std |
| MNLI (acc) | 52.2 +/- 4.2 | **68.8 +/- 0.5** | +16.6, -3.7 std |

Обрати внимание: не только среднее качество растёт, но и **стандартное отклонение падает** -- модель стабилизируется.

## Сравнение с другими методами стабилизации

Zhang et al. провели критический ablation: при **правильном debiased Adam** большинство ранее предложенных методов **теряют эффект**:

| Метод | RTE acc (mean +/- std) | Помогает? |
|-------|----------------------|-----------|
| Baseline (debiased Adam) | 69.5 +/- 2.4 | -- |
| LLRD (Layer-wise LR Decay) | 69.7 +/- 3.2 | Нет |
| Mixout | 71.3 +/- 1.4 | Немного |
| Pre-trained WD | 69.6 +/- 2.1 | Нет |
| **Re-init** | **72.6 +/- 1.6** | **Да** |
| **Intermediate Task Transfer** | **81.8 +/- 1.7** | **Значительно** |

### Intermediate Task Transfer

Единственный метод, который даёт **значительный** прирост помимо Re-init: сначала fine-tune BERT на **большом родственном датасете** (MNLI, 393K), потом на целевом маленьком (RTE, 2.4K).

Это работает, потому что MNLI и RTE -- оба NLI задачи. Transfer помогает модели выучить task-specific features на большом датасете, прежде чем адаптироваться к маленькому.

## Рецепт: few-shot fine-tuning BERT

Собирая вместе все findings Zhang et al.:

```
1. Оптимизатор: AdamW (ОБЯЗАТЕЛЬНО debiased)
   lr: 2e-5 (стандартный BERT LR)
   weight_decay: 0.01
   warmup: 10% шагов

2. Re-init: top L BERT blocks + pooler
   L: подобрать по validation (обычно 1-3)
   init: N(0, 0.02^2)

3. Training: longer than 3 epochs
   Подобрать steps из {200, 400, 800, 1600, 3200}
   Early stopping по validation

4. Multiple runs: запустить 10-20 seeds
   Отфильтровать degenerate runs
   Сообщать mean +/- std

5. (Optional) Intermediate Task Transfer
   Если есть большой родственный датасет
```

## Современный контекст: few-shot в эпоху LLM

С приходом GPT-3+ появилась альтернатива few-shot fine-tuning -- **in-context learning**: дать модели несколько примеров прямо в prompt, без обновления весов.

| Подход | Плюсы | Минусы |
|--------|-------|--------|
| Few-shot FT (BERT) | Высокая точность на NLU, детерминизм | Нестабильность, нужен GPU |
| In-context learning (GPT) | Нет обучения, мгновенный deployment | Ниже точность, зависимость от prompt |
| Few-shot FT + PEFT (LoRA) | Стабильнее FT, дешевле | Всё ещё нужен GPU |

**Практический вывод**: для production NLU задач (NER, classification) с <1K примерами -- few-shot fine-tuning BERT/DeBERTa с правильным рецептом по-прежнему **лучше**, чем GPT-4 zero/few-shot.

## Key papers

- [[02 Areas/ML & DL/Papers/Revisiting Few-sample BERT Fine-tuning]] -- три источника нестабильности + fix (Zhang et al., 2021)
- [[02 Areas/ML & DL/Papers/BERT]] -- оригинальные рекомендации (3 эпохи, BERTAdam)

## Related concepts

- [[02 Areas/ML & DL/Concepts/Training/Fine-tuning|Fine-tuning]] -- general fine-tuning
- [[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]] -- base модели для few-shot FT
- [[02 Areas/ML & DL/Concepts/Training/PEFT|PEFT]] -- parameter-efficient альтернатива
- [[02 Areas/ML & DL/Concepts/Architectures/BERT|BERT]] -- основная модель для few-shot FT
- [[02 Areas/ML & DL/Concepts/Architectures/RoBERTa|RoBERTa]] -- улучшенная base модель
- [[02 Areas/ML & DL/Concepts/NLP/GLUE|GLUE]] -- бенчмарк для оценки few-shot FT

## Дополнительные ресурсы

- [Zhang et al. -- Revisiting Few-sample BERT Fine-tuning (2021)](https://arxiv.org/abs/2006.05987) -- must-read для любого, кто fine-tunes BERT на малых данных
- [Mosbach et al. -- On the Stability of Fine-tuning BERT (2021)](https://arxiv.org/abs/2006.04884) -- complementary work по стабильности
- [HuggingFace -- Fine-tuning with Trainer](https://huggingface.co/docs/transformers/training) -- tutorial с правильным AdamW
