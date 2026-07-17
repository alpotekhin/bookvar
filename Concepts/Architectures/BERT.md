---
title: "BERT"
aliases: [Bidirectional Encoder Representations from Transformers]
type: concept
category: Architectures
papers:
  - "[[02 Areas/ML & DL/Papers/BERT]]"
  - "[[02 Areas/ML & DL/Papers/RoBERTa]]"
  - "[[02 Areas/ML & DL/Papers/DeBERTa]]"
  - "[[02 Areas/ML & DL/Papers/Revisiting Few-sample BERT Fine-tuning]]"
courses: []
sources:
  - "[Jay Alammar — The Illustrated BERT](https://jalammar.github.io/illustrated-bert/)"
  - "[Lilian Weng — Generalized Language Models](https://lilianweng.github.io/posts/2019-01-31-lm/)"
  - "[d2l.ai — BERT](https://d2l.ai/chapter_natural-language-processing-pretraining/bert.html)"
---

# BERT — Bidirectional Encoder Representations from Transformers

## Зачем это нужно: проблема однонаправленности

До BERT (2018) существовало два подхода к pre-trained language representations:

1. **Feature-based (ELMo)**: обучаем bidirectional LSTM, используем скрытые состояния как фичи для downstream задач. Проблема: left-to-right и right-to-left LM обучаются **независимо**, потом просто конкатенируются. Это «поверхностная» двунаправленность.

2. **Fine-tuning (GPT)**: обучаем left-to-right Transformer decoder, потом fine-tune'им на задачу. Проблема: модель видит только **левый** контекст. Для token-level задач вроде QA и NER это катастрофически плохо — ответ на вопрос может зависеть от контекста **справа**.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/openai-transformer-1.png]]
*GPT (OpenAI Transformer): однонаправленный decoder — каждый токен видит только предшествующие (источник: Jay Alammar)*

**Ключевой вопрос**: можно ли обучить **глубоко двунаправленную** модель? Проблема: если каждый токен видит все остальные, при стандартном language modeling токен «видит сам себя» через multi-layer attention — это тривиальная утечка информации.

**Решение Devlin et al. (2019)**: **Masked Language Modeling** — случайно маскируем токены, предсказываем их по двунаправленному контексту. Просто и гениально.

## Архитектура

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/bert-base-bert-large-encoders.png]]
*BERT BASE (12 слоёв) vs BERT LARGE (24 слоя) — стеки encoder-блоков (источник: Jay Alammar)*

BERT — стек [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] encoder блоков с **bidirectional (unmasked) self-attention**. Две конфигурации:

| Конфигурация | Layers (L) | Hidden (H) | Heads (A) | FFN (4H) | Параметры |
|-------------|-----------|-----------|----------|---------|----------|
| BERT_BASE | 12 | 768 | 12 | 3072 | 110M |
| BERT_LARGE | 24 | 1024 | 16 | 4096 | 340M |

BERT_BASE специально выбран с таким же числом параметров, что и GPT-1 — для честного сравнения. Критическое отличие: BERT использует **bidirectional** self-attention, GPT — **constrained** (left-to-right only).

### Входные эмбеддинги: три компонента

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/bert-input-output.png]]
*Вход BERT: [CLS] + токены предложения A + [SEP] + токены предложения B + [SEP] (источник: Jay Alammar)*

Каждый токен получает **сумму** трёх эмбеддингов:

1. **Token Embedding** — WordPiece токенизация (словарь 30K). Слова разбиваются на подслова: «playing» → «play» + «##ing». Это решает проблему OOV (out-of-vocabulary) слов.

2. **Segment Embedding** — вектор A или B, указывающий к какому предложению принадлежит токен. Необходим для задач с парами предложений (NLI, QA).

3. **Position Embedding** — **обучаемые** позиционные векторы (не синусоидальные, как в оригинальном Transformer). Максимальная длина: 512 токенов.

**Специальные токены:**
- `[CLS]` — первый токен каждого входа. Его финальное представление используется как aggregate representation для classification задач.
- `[SEP]` — разделитель между предложениями A и B.

## Pre-training: два objective

### Task 1: Masked Language Modeling (MLM)

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/BERT-language-modeling-masked-lm.png]]
*MLM: 15% токенов маскируются, модель предсказывает оригинальные токены по двунаправленному контексту (источник: Jay Alammar)*

15% токенов случайно выбираются для предсказания. Но не все заменяются на `[MASK]` — иначе возникает mismatch между pre-training (где `[MASK]` есть) и fine-tuning (где его нет). Схема:

| Действие | Доля | Пример |
|----------|------|--------|
| Замена на `[MASK]` | 80% | «my dog is hairy» → «my dog is `[MASK]`» |
| Замена на случайный токен | 10% | «my dog is hairy» → «my dog is apple» |
| Без изменения | 10% | «my dog is hairy» → «my dog is hairy» |

**Зачем 10% случайных?** Чтобы модель не могла просто запомнить «если вижу `[MASK]`, нужно предсказывать» — она должна поддерживать хорошее представление для **каждого** токена.

**Зачем 10% без изменений?** Чтобы модель имела bias к реальному наблюдаемому слову.

Loss считается **только по маскированным позициям** — финальный hidden state соответствующей позиции пропускается через softmax по словарю.

**Почему MLM работает лучше left-to-right LM?** В стандартном LM токен «it» в предложении «The animal didn't cross the street because it was too tired» видит только «The animal didn't cross the street because» (левый контекст). В MLM он видит и левый контекст, и «was too tired» справа — может однозначно связать «it» с «animal», а не со «street».

### Task 2: Next Sentence Prediction (NSP)

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/bert-next-sentence-prediction.png]]
*NSP: бинарная классификация — предложение B реально следует за A (IsNext) или случайное (NotNext) (источник: Jay Alammar)*

Бинарная задача для пар предложений:
- 50% пар: B реально следует за A (**IsNext**)
- 50% пар: B — случайное предложение из корпуса (**NotNext**)

Предсказание по вектору `[CLS]`. Точность на NSP: ~97-98%.

**Важно**: позже RoBERTa (Liu et al., 2019) показала, что NSP **не улучшает** downstream performance и убрала его. ALBERT заменил на sentence order prediction. Вывод: NSP был полезной, но не обязательной частью BERT.

### Pre-training данные и параметры

- **BooksCorpus**: 800M слов (Zhu et al., 2015)
- **English Wikipedia**: 2,500M слов (только текст, без таблиц и списков)
- Document-level корпус (не sentence-level) — для длинных контекстов
- Batch size: 256 sequences × 512 tokens = 128K tokens/batch
- Training: 1M шагов ≈ 40 эпох на объединённых данных
- Оптимизатор: Adam, lr = 1e-4, $\beta_1 = 0.9$, $\beta_2 = 0.999$, warmup 10K шагов
- Training time: 4 дня на 4 Cloud TPUs (16 TPU chips) для BERT_BASE; 4 дня на 16 Cloud TPUs (64 TPU chips) для BERT_LARGE

## Fine-tuning: простота — сила BERT

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/bert-tasks.png]]
*Fine-tuning BERT на разные задачи: одна и та же архитектура, разные output layers (источник: Jay Alammar)*

Fine-tuning максимально прост — BERT не требует task-specific архитектур:

| Задача | Как fine-tune'ить | Input формат |
|--------|------------------|-------------|
| **Classification** (sentiment, NLI) | Linear layer поверх `[CLS]`: $\text{softmax}(C \cdot W^T)$ | `[CLS] text [SEP]` или `[CLS] sent_A [SEP] sent_B [SEP]` |
| **Token-level** (NER) | Linear layer поверх каждого $T_i$ | `[CLS] tokens [SEP]` |
| **QA (span extraction)** | Start vector $S$ + End vector $E$: $P(\text{start}=i) = \text{softmax}(S \cdot T_i)$ | `[CLS] question [SEP] paragraph [SEP]` |

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/bert-classifier.png]]
*Classification: вектор [CLS] → linear layer → softmax → класс (источник: Jay Alammar)*

**Гиперпараметры fine-tuning**: batch size = 32, эпохи = 3, lr $\in$ {5e-5, 4e-5, 3e-5, 2e-5}. Тюнится только learning rate на dev set.

## Ключевые результаты

### GLUE Benchmark (Table 1 из статьи)

| Задача | BERT_LARGE | GPT (prev SOTA) | Улучшение |
|--------|-----------|-----------------|-----------|
| MNLI | 86.7% | 82.1% | +4.6% |
| QQP | 72.1 F1 | 70.3 F1 | +1.8 |
| QNLI | 92.7% | 87.4% | +5.3% |
| SST-2 | 94.9% | 91.3% | +3.6% |
| **GLUE Average** | **80.5** | **72.8** | **+7.7 абс.** |

**+7.7 абсолютных процентов** над предыдущим SOTA (GPT) — взрывной скачок.

### SQuAD (Question Answering)

| Бенчмарк | BERT_LARGE single | Prev. best single | Prev. best ensemble |
|----------|------------------|-------------------|---------------------|
| SQuAD v1.1 F1 | 90.9 (dev) | — | 91.8 (ensemble) |
| SQuAD v2.0 F1 | 83.1 (test) | 78.0 | — |

BERT single model на SQuAD v2.0 побил предыдущий лучший результат на **+5.1 F1**.

## Ablation Studies: что важно, а что нет

### Bidirectionality критична (Table 5)

| Конфигурация | MNLI | SQuAD F1 |
|-------------|------|---------|
| BERT_BASE (bidirectional, MLM + NSP) | 84.4% | 88.5 |
| No NSP | 83.9% | 87.9 |
| LTR & No NSP (= GPT-like) | 82.1% | **77.8** |
| + BiLSTM на LTR | 82.1% | 84.8 |

**Left-to-right без NSP** (GPT-like): SQuAD F1 падает на **10.7 пунктов** — bidirectionality действительно критична для token-level задач. Даже добавление BiLSTM поверх LTR не спасает.

### Масштаб модели помогает всегда (Table 6)

| Конфигурация | MNLI | MRPC | SQuAD F1 |
|-------------|------|------|---------|
| L=3, H=768 | 77.9% | 79.8% | 81.6 |
| L=6, H=768 | 80.6% | 82.2% | 85.8 |
| L=12, H=768 (BASE) | 84.4% | 86.7% | 88.5 |
| L=24, H=1024 (LARGE) | 86.6% | 87.8% | 90.9 |

Строгое улучшение при увеличении L, H, A — даже MRPC (3.6K примеров) выигрывает от масштаба.

## Gotchas при fine-tuning (из практики)

Из [[02 Areas/ML & DL/Papers/Revisiting Few-sample BERT Fine-tuning]] (Zhang et al., 2021):

1. **BERTAdam без bias correction** — оригинальный код Google не имел bias correction в Adam → нестабильность при малых датасетах (48% degenerate runs на RTE). Fix: используйте `torch.optim.AdamW`.

2. **3 эпохи недостаточно** при 1K примерах — это только ~96 шагов. Тюнируйте число итераций.

3. **Re-init top layers** (+3.1% на RTE) — верхние слои BERT специализируются под MLM, re-initialization улучшает downstream.

## BERT как feature extractor

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/bert-feature-extraction-contextualized-embeddings.png]]
*BERT как feature extractor: выходы разных слоёв можно использовать как контекстуализированные эмбеддинги без fine-tuning (источник: Jay Alammar)*

BERT можно использовать и без fine-tuning — как source of contextualized embeddings. Ablation из статьи показывает: конкатенация последних 4 слоёв даёт NER F1 = 96.1 (всего -0.3 от fine-tuning F1 = 96.4). Это полезно когда:
- Нет GPU для fine-tuning
- Задача не вписывается в стандартные форматы BERT
- Нужны фичи для другой модели (SVM, CRF и т.д.)

## BERT vs ELMo vs GPT: сравнение парадигм

| Аспект | ELMo | GPT | BERT |
|--------|------|-----|------|
| Архитектура | BiLSTM | Transformer decoder | Transformer encoder |
| Направленность | Shallow bidirectional | Left-to-right | **Deep bidirectional** |
| Pre-training | LM (left + right concat) | Autoregressive LM | MLM + NSP |
| Применение | Feature extraction | Fine-tuning | Fine-tuning (или features) |
| Специальные токены | — | — | `[CLS]`, `[SEP]`, `[MASK]` |
| GLUE score | 70.0 | 72.8 | **80.5** |

## Наследие и потомки

BERT породил целое семейство моделей:
- **RoBERTa** (Liu et al., 2019) — оптимизированный pre-training: нет NSP, dynamic masking, больше данных (160GB), batch size 8K → GLUE 88.5
- **ALBERT** (Lan et al., 2020) — factorized embedding, cross-layer parameter sharing → 18x меньше параметров, comparable performance
- **DeBERTa** (He et al., 2021) — disentangled attention (раздельное внимание к контенту и позиции) → SuperGLUE SOTA
- **SpanBERT** (Joshi et al., 2020) — span masking вместо token masking → лучше для QA и co-reference

## Key papers

- [[02 Areas/ML & DL/Papers/BERT]] — оригинал (Devlin et al., 2019)
- [[02 Areas/ML & DL/Papers/RoBERTa]] — оптимизированный pre-training
- [[02 Areas/ML & DL/Papers/DeBERTa]] — disentangled attention
- [[02 Areas/ML & DL/Papers/Revisiting Few-sample BERT Fine-tuning]] — fix'ы нестабильного fine-tuning

## Related concepts

- [[02 Areas/ML & DL/Concepts/Training/Masked Language Modeling|Masked Language Modeling (MLM)]]
- [[02 Areas/ML & DL/Concepts/Training/Next Sentence Prediction|Next Sentence Prediction (NSP)]]
- [[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]]
- [[02 Areas/ML & DL/Concepts/Training/Fine-tuning|Fine-tuning]]
- [[02 Areas/ML & DL/Concepts/Training/Few-shot Fine-tuning|Few-shot Fine-tuning]]
- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]]
- [[02 Areas/ML & DL/Concepts/Architectures/Encoder-only|Encoder-only]]
- [[02 Areas/ML & DL/Concepts/Architectures/RoBERTa|RoBERTa]]
- [[02 Areas/ML & DL/Concepts/Architectures/DeBERTa|DeBERTa]]
- [[02 Areas/ML & DL/Concepts/NLP/GLUE|GLUE]]
- [[02 Areas/ML & DL/Concepts/NLP/Tokenization|Tokenization]]

## Дополнительные ресурсы

- [Jay Alammar — The Illustrated BERT](https://jalammar.github.io/illustrated-bert/) — лучшие визуализации
- [Lilian Weng — Generalized Language Models](https://lilianweng.github.io/posts/2019-01-31-lm/) — контекст ELMo → GPT → BERT
- [d2l.ai — BERT](https://d2l.ai/chapter_natural-language-processing-pretraining/bert.html) — учебник с кодом
- [Google Research BERT repo](https://github.com/google-research/bert) — оригинальный код
