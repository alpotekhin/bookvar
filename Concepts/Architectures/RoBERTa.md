---
title: "RoBERTa"
aliases: [Robustly Optimized BERT Pretraining Approach]
type: concept
category: Architectures
papers:
  - "[[02 Areas/ML & DL/Papers/RoBERTa]]"
  - "[[02 Areas/ML & DL/Papers/BERT]]"
courses: []
sources:
  - "[Liu et al. — RoBERTa: A Robustly Optimized BERT Pretraining Approach (2019)](https://arxiv.org/abs/1907.11692)"
  - "[Facebook AI Blog — RoBERTa](https://ai.meta.com/blog/roberta-an-optimized-method-for-pretraining-self-supervised-nlp-systems/)"
  - "[HuggingFace roberta-large](https://huggingface.co/roberta-large)"
---

# RoBERTa — Robustly Optimized BERT Pretraining Approach

## Зачем эта работа появилась

К середине 2019 года после BERT (Devlin et al., 2018) появилось множество моделей, заявлявших превосходство: XLNet, ERNIE, SpanBERT. Каждая предлагала архитектурные или training objective изменения. Но **справедливое сравнение** было невозможно: разные модели обучались на разных данных, с разными гиперпараметрами, разное время.

Liu et al. (Facebook AI, июль 2019) провели **самый важный ablation study** в истории NLP pre-training. Их вывод шокировал сообщество: **BERT был значительно недообучен**. Простое исправление training recipe — без единого архитектурного изменения — даёт модель, которая **превосходит все post-BERT модели**, включая те, что предлагали «улучшенные» архитектуры.

## Четыре открытия: что было не так с BERT

### 1. NSP не нужен и вредит (§4.2)

Next Sentence Prediction (NSP) — одна из двух целевых функций BERT. Авторы утверждали, что она улучшает downstream performance. RoBERTa **опровергла это**:

| Формат данных | NSP | SQuAD 2.0 F1 | MNLI-m | RACE |
|---------------|-----|--------------|--------|------|
| SEGMENT-PAIR (как BERT) | Да | 78.7 | 84.0 | 64.2 |
| SENTENCE-PAIR | Да | 78.4 | 84.0 | 63.0 |
| FULL-SENTENCES | Нет | 79.1 | 84.7 | 64.8 |
| **DOC-SENTENCES** | **Нет** | **79.7** | **84.7** | **65.6** |

**DOC-SENTENCES** (полные предложения из одного документа, без NSP) — лучший вариант по всем метрикам.

**Почему NSP вредит:** предположение авторов — NSP task слишком простой и «загрязняет» MLM сигнал. Модель тратит capacity на тривиальную задачу вместо изучения глубоких языковых репрезентаций. Кроме того, SEGMENT-PAIR формат (два коротких сегмента) ограничивает контекст — модель видит меньше текста за раз.

### 2. Dynamic Masking лучше Static (§4.1)

**BERT (static masking):** маскирование выполняется **один раз** при preprocessing. Данные дублируются 10 раз с разными масками, но каждая копия всегда видит одну и ту же маску. За 40 эпох обучения модель видит каждую маску ~4 раза.

**RoBERTa (dynamic masking):** маска генерируется **заново** каждый раз, когда последовательность подаётся в модель. Каждая эпоха — новые маски.

Разница невелика на коротком обучении (90.4 vs 90.4 F1 на SQuAD), но **становится значимой при длинном обучении** — модель видит больше разнообразных контекстов для каждого токена, что улучшает generalization.

### 3. Большие батчи ускоряют и улучшают (§4.3)

| Batch size | Steps | BERT PPL | SQuAD 1.1 F1 |
|------------|-------|----------|--------------|
| 256 (BERT) | 1M | — | 90.4 |
| 2K | 125K | — | 93.7 |
| **8K** | **31K** | — | **94.6** |

При правильном learning rate schedule больший batch size:
- **Ускоряет** обучение (меньше шагов при том же compute)
- **Улучшает** downstream performance (точнее gradient estimation)

Это согласуется с работой McCandlish et al. (2018) о critical batch size и демонстрирует, что BERT использовал субоптимальный batch size.

### 4. Больше данных + дольше обучение = стабильный рост (§4.4)

Систематический ablation на объёме данных и длительности обучения:

| Данные | Размер | Training Steps | SQuAD 2.0 F1 | MNLI-m |
|--------|--------|----------------|--------------|--------|
| Books + Wiki (как BERT) | 16GB | 100K | 87.3 | 89.0 |
| + CC-News + WebText + Stories | 160GB | 100K | 87.7 | 89.3 |
| то же | 160GB | 300K | 88.7 | 90.0 |
| то же | 160GB | **500K** | **89.4** | **90.2** |

**Данные (160GB):** BookCorpus + Wikipedia + CC-News (63M статей, ~76GB) + OpenWebText (Reddit 3+ karma, ~38GB) + Stories (~31GB).

Вывод: **performance не сатурирует** — даже на 500K шагов модель продолжает улучшаться. BERT с его 1M шагов на 16GB данных был катастрофически недообучен.

## Итоговая формула RoBERTa

RoBERTa = BERT + все четыре исправления:

| Аспект | BERT | RoBERTa |
|--------|------|---------|
| NSP loss | Да | **Нет** |
| Masking | Static (один раз) | **Dynamic** (каждый batch) |
| Batch size | 256 | **8000** |
| Данные | 16GB (Wiki + Books) | **160GB** (+ CC-News, WebText, Stories) |
| Training steps | 1M × 256 | **500K × 8K** (больше compute) |
| Short sequences | 90% обучения на 128 токенов | **Всегда 512 токенов** |
| Архитектура | L=24, H=1024, A=16, 355M | **Та же** |
| Optimizer | Adam β2=0.999 | Adam **β2=0.98** (стабильнее с большим batch) |

## Ключевые результаты

### GLUE Test (Table 5)

| Задача | BERT-Large | XLNet-Large | **RoBERTa** |
|--------|------------|-------------|-------------|
| MNLI-m | 86.7 | 89.8 | **90.2** |
| QNLI | 92.7 | 93.9 | **94.7** |
| SST-2 | 94.9 | 95.6 | **96.4** |
| RTE | 70.1 | 83.8 | **86.6** |
| **Average** | **82.1** | **87.6** | **88.5** |

RoBERTa: **SOTA на 4 из 9 задач GLUE** (MNLI, QNLI, RTE, STS-B). +6.4% average над BERT-Large.

### SQuAD 2.0

| Модель | F1 |
|--------|-----|
| BERT-Large | 83.1 |
| XLNet-Large | 88.8 |
| **RoBERTa** | **89.4** |

## Почему RoBERTa важна

### 1. Самый влиятельный ablation study в NLP

RoBERTa показала: прежде чем предлагать новые архитектуры, нужно **правильно обучить** существующие. Многие «улучшения» BERT (XLNet permutation LM, ERNIE entity masking) оказались **менее важны**, чем простое увеличение данных и правильные гиперпараметры.

### 2. Baseline для всей области (2019-2022)

RoBERTa стала **де-факто стандартным baseline** для encoder-only research. Если новая модель не сравнивается с RoBERTa — результаты считаются неполными. Именно поэтому [[02 Areas/ML & DL/Concepts/Architectures/DeBERTa|DeBERTa]] сравнивается с RoBERTa-Large, а не с BERT.

### 3. Рецепт, который масштабируется

Каждое открытие RoBERTa **оказалось применимым** к другим моделям:
- **Удаление NSP** → стандарт (DeBERTa, ALBERT, и т.д.)
- **Dynamic masking** → стандарт для MLM pre-training
- **Больше данных** → подтверждено Chinchilla scaling laws
- **Большие батчи** → стандарт для больших моделей

### 4. Установила методологический стандарт

RoBERTa показала, как правильно проводить **контролируемые эксперименты** в NLP pre-training: один параметр за раз, фиксированный compute, воспроизводимые результаты. Это повлияло на методологию всех последующих работ.

## Практическое использование

RoBERTa-large остаётся конкурентоспособным backbone для:
- **Text classification** (sentiment, topic, intent detection)
- **Named Entity Recognition** (NER)
- **Extractive QA** (SQuAD-style span extraction)
- **Sentence similarity** и retrieval (через mean pooling)
- **Feature extraction** для downstream models

Однако для production NLU задач рекомендуется [[02 Areas/ML & DL/Concepts/Architectures/DeBERTa|DeBERTa-v3-large]] — та же парадигма, но с архитектурными улучшениями.

## Key papers

- [[02 Areas/ML & DL/Papers/RoBERTa]] — оригинал (Liu et al., 2019)
- [[02 Areas/ML & DL/Papers/BERT]] — архитектурная основа

## Related concepts

- [[02 Areas/ML & DL/Concepts/Architectures/BERT|BERT]] — модель, которую RoBERTa «починила»
- [[02 Areas/ML & DL/Concepts/Architectures/Encoder-only|Encoder-only]] — архитектурный класс
- [[02 Areas/ML & DL/Concepts/Training/Masked Language Modeling|Masked Language Modeling (MLM)]] — pre-training objective
- [[02 Areas/ML & DL/Concepts/Training/Next Sentence Prediction|Next Sentence Prediction (NSP)]] — objective, который RoBERTa удалила
- [[02 Areas/ML & DL/Concepts/Architectures/DeBERTa|DeBERTa]] — архитектурный наследник

## Дополнительные ресурсы

- [RoBERTa paper (arXiv)](https://arxiv.org/abs/1907.11692) — оригинальная статья
- [fairseq GitHub](https://github.com/pytorch/fairseq) — официальная реализация
- [HuggingFace roberta-large](https://huggingface.co/roberta-large) — pretrained чекпоинт
