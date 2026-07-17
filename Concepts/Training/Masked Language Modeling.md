---
title: "Masked Language Modeling"
aliases: [MLM, masked LM, маскированное языковое моделирование]
type: concept
category: Training
papers:
  - "[[02 Areas/ML & DL/Papers/BERT]]"
  - "[[02 Areas/ML & DL/Papers/RoBERTa]]"
  - "[[02 Areas/ML & DL/Papers/DeBERTa]]"
  - "[[02 Areas/ML & DL/Papers/T5]]"
courses:
  - "[[02 Areas/ML & DL/Courses/Stanford CS224N/07 -- Attention|CS224N Lecture 7]]"
sources:
  - "[Devlin et al. -- BERT: Pre-training of Deep Bidirectional Transformers (2019)](https://arxiv.org/abs/1810.04805)"
  - "[Liu et al. -- RoBERTa: A Robustly Optimized BERT Pretraining Approach (2019)](https://arxiv.org/abs/1907.11692)"
  - "[Jay Alammar -- The Illustrated BERT](https://jalammar.github.io/illustrated-bert/)"
---

# Masked Language Modeling (MLM)

## Зачем это нужно: проблема однонаправленности

До BERT (2018) существовали два подхода к pre-training:

1. **Left-to-right LM** (GPT) -- модель видит только предыдущие токены. Хорошо для генерации, но для понимания текста критически не хватает **правого контекста**. Если нужно понять, к чему относится местоимение, или классифицировать sentiment -- нужны слова и слева, и справа.

2. **Отдельные left-to-right + right-to-left** (ELMo) -- два независимых LM, конкатенированных. Каждое направление обучается отдельно, не видя другое. Это не настоящая bidirectionality.

Проблема: **нельзя просто обучить bidirectional LM** стандартным способом, потому что каждое слово будет "видеть себя" через другие слои -- модель тривиально выучит identity function вместо полезных представлений.

**MLM** решает эту проблему элегантно: маскируем случайные токены и просим модель их восстановить, используя **весь окружающий контекст** одновременно. Это позволяет обучить по-настоящему bidirectional encoder.

## Как работает MLM: схема 80/10/10

Из оригинальной статьи BERT (Devlin et al., 2019, раздел 3.1):

### Шаг 1: выбор токенов для маскирования

Из входной последовательности случайно выбираются **15% токенов** для предсказания.

### Шаг 2: замена по правилу 80/10/10

Из выбранных 15% токенов:
- **80%** заменяются на специальный токен `[MASK]`
- **10%** заменяются на **случайный токен** из словаря
- **10%** остаются **неизменными**

```
Оригинальный текст:
  "The  quick  brown  fox  jumps  over  the  lazy  dog"
        |       |          |                   |
Выбраны для предсказания (15% токенов):
  "quick", "brown", "jumps", "lazy"

Замена по 80/10/10:
  "quick"  -->  [MASK]     (80%: маска)
  "brown"  -->  "green"    (10%: случайный токен)
  "jumps"  -->  "jumps"    (10%: без изменений)
  "lazy"   -->  [MASK]     (80%: маска)

Вход в BERT:
  "The  [MASK]  green  fox  [MASK]  over  the  jumps  dog"

BERT (bidirectional): каждый токен видит ВЕСЬ контекст
  --> предсказывает: "quick" (контекст слева И справа)
  --> предсказывает: "lazy"  (контекст слева И справа)

Loss: cross-entropy ТОЛЬКО по маскированным позициям
```

### Зачем 80/10/10, а не просто 100% [MASK]?

Это решает **mismatch между pre-training и fine-tuning**:

При fine-tuning модель никогда не видит токен `[MASK]`. Если бы все маскированные позиции заменялись на `[MASK]`, модель бы:
- Выучила, что предсказывать нужно только когда видишь `[MASK]`
- Не умела бы строить контекстные представления для обычных токенов

10% случайных замен + 10% без изменений **заставляют модель** строить качественные bidirectional представления для **каждого** токена, а не только для маскированных.

### Prediction head

Предсказание маскированного токена:

$$P(x_i | \text{context}) = \text{softmax}(W \cdot h_i + b)$$

где $h_i$ -- hidden state BERT на позиции маскированного токена, $W \in \mathbb{R}^{|V| \times d}$ -- linear projection на словарь.

**Loss** вычисляется **только по маскированным позициям** (не по всем токенам). Это замедляет convergence по сравнению с CLM (который учится на каждом токене), но даёт bidirectionality.

## Static vs Dynamic Masking

### BERT (static masking)

Маска генерируется **один раз** при preprocessing. Данные дублируются 10 раз с 10 разными масками. За 40 эпох обучения каждая последовательность видит каждую маску ~4 раза.

### RoBERTa (dynamic masking)

Маска **re-генерируется перед каждой подачей** последовательности в модель. Каждый раз модель видит одну и ту же последовательность с разными замаскированными позициями.

Из RoBERTa Table 2: dynamic masking **немного лучше** при длительном обучении:

| Маскирование | MNLI-m | SQuAD F1 |
|-------------|--------|----------|
| Static (BERT) | 84.3 | -- |
| Dynamic (RoBERTa) | 84.7 | +0.2 |

Разница небольшая, но dynamic masking ещё и проще в реализации (не нужно дублировать данные).

## Расширения MLM

### SpanBERT: маскирование span-ов

Вместо случайных отдельных токенов, SpanBERT (Joshi et al., 2020) маскирует **непрерывные span-ы** случайной длины (geometric distribution, mean=3.8). Дополнительно -- **Span Boundary Objective (SBO)**: предсказание маскированных токенов по hidden states на границах span-а.

Результат: значительное улучшение на extractive QA (SQuAD: +2.1 F1) и coreference resolution, где контекст span-а критически важен.

### T5: Span Corruption

T5 (Raffel et al., 2020) обобщает MLM в формат **text-to-text**: маскированные span-ы заменяются на sentinel tokens, и модель должна сгенерировать только маскированные span-ы:

```
Вход:  "The <X> fox <Y> the lazy dog"
Выход: "<X> quick brown <Y> jumps over"
```

Преимущество: целевая последовательность **короче** -- меньше compute при обучении. T5 показал, что span corruption (15% текста, средняя длина span 3) оптимальна среди множества вариантов.

### Whole Word Masking (WWM)

BERT использует WordPiece токенизацию: слово "playing" может быть разбито на "play" + "##ing". Если маскировать только "##ing", модель тривиально восстанавливает его по "play".

**Whole Word Masking**: если маскируется любой sub-word токен слова, маскируются **все** sub-word токены этого слова. Улучшает качество на Chinese BERT и DistilBERT.

## MLM vs CLM: фундаментальное сравнение

| Свойство | MLM (BERT) | CLM (GPT) |
|----------|-----------|-----------|
| Контекст | **Bidirectional** | Left-to-right |
| Генерация текста | Нет (не авторегрессивный) | Да |
| Loss | Только по ~15% токенов | По всем токенам |
| Скорость convergence | Медленнее (15% signal) | Быстрее (100% signal) |
| NLU задачи | **Лучше** при fine-tuning | Хуже |
| NLG задачи | Не применим | **Лучше** |
| Архитектура | Encoder-only | Decoder-only |

### Ablation из BERT (Table 5)

Devlin et al. провели критический ablation -- что будет, если BERT обучить как left-to-right LM (без MLM)?

| Модель | SQuAD v1.1 F1 | MNLI-m |
|--------|--------------|--------|
| BERT (bidirectional MLM) | **88.5** | **84.0** |
| LTR (left-to-right, no MLM) | **77.8** (-10.7!) | 82.1 |
| LTR + BiLSTM | 82.1 | 81.5 |

**-10.7 F1** на SQuAD при замене MLM на LTR -- это огромная разница. Даже добавление BiLSTM поверх LTR не спасает ситуацию. Bidirectionality через MLM -- ключевой фактор.

## Когда использовать MLM-модели

```
Задача NLU?
|-- Много размеченных данных?
|   |-- Да --> BERT/RoBERTa/DeBERTa fine-tuning (MLM pretrained)
|   |-- Нет --> Few-shot с LLM (CLM) может быть лучше
|
|-- Задача NLG?
    |-- Да --> CLM (GPT-family) или Seq2Seq (T5)
    |-- Нет --> MLM-модели -- лучший выбор
```

**Практический вывод** из Harnessing LLMs survey: MLM (fine-tuned) лучше CLM для NER (~2x), toxicity detection, span-extraction QA. CLM лучше для open-ended generation, few-shot/zero-shot, knowledge-intensive tasks.

## DeBERTa: Enhanced Mask Decoder

DeBERTa (He et al., 2020) улучшает MLM prediction: стандартный BERT использует только relative position information в attention, но для MLM задачи **абсолютная позиция** маскированного токена тоже важна (например, слово в начале предложения vs в конце).

**Enhanced Mask Decoder (EMD)** добавляет один дополнительный Transformer слой, который инжектирует absolute position embeddings перед prediction head -- только для маскированных позиций. Результат: DeBERTa Large > RoBERTa Large при том же compute.

## Хронология

| Год | Milestone | Модель/метод |
|-----|-----------|-------------|
| 2018 | Оригинальный MLM + NSP | BERT |
| 2019 | Dynamic masking, убран NSP | RoBERTa |
| 2020 | Span masking + SBO | SpanBERT |
| 2020 | Span corruption (text-to-text) | T5 |
| 2020 | Disentangled attention + EMD для MLM | DeBERTa |
| 2021 | Replaced Token Detection (альтернатива MLM) | ELECTRA |

## Key papers

- [[02 Areas/ML & DL/Papers/BERT]] -- оригинал MLM (80/10/10 scheme)
- [[02 Areas/ML & DL/Papers/RoBERTa]] -- dynamic masking ablation, убран NSP
- [[02 Areas/ML & DL/Papers/DeBERTa]] -- Enhanced Mask Decoder (absolute positions при prediction)
- [[02 Areas/ML & DL/Papers/T5]] -- span corruption как обобщение MLM

## Related concepts

- [[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]] -- MLM как основной pre-training objective для encoder-моделей
- [[02 Areas/ML & DL/Concepts/Training/Causal Language Modeling|Causal Language Modeling]] -- альтернативный objective для decoder-моделей
- [[02 Areas/ML & DL/Concepts/Training/Next Sentence Prediction|Next Sentence Prediction (NSP)]] -- вспомогательный objective BERT
- [[02 Areas/ML & DL/Concepts/Training/Disentangled Attention|Disentangled Attention]] -- Enhanced Mask Decoder в DeBERTa
- [[02 Areas/ML & DL/Concepts/Architectures/BERT|BERT]] -- модель, для которой MLM был создан
- [[02 Areas/ML & DL/Concepts/Architectures/RoBERTa|RoBERTa]] -- оптимизация MLM pre-training
- [[02 Areas/ML & DL/Concepts/Architectures/Encoder-only|Encoder-only]] -- класс архитектур, использующих MLM

## Дополнительные ресурсы

- [Jay Alammar -- The Illustrated BERT](https://jalammar.github.io/illustrated-bert/) -- лучшие визуализации MLM
- [Lena Voita -- Transfer Learning in NLP](https://lena-voita.github.io/nlp_course/transfer_learning.html) -- глубокое объяснение MLM vs CLM
- [d2l.ai -- BERT](https://d2l.ai/chapter_natural-language-processing-pretraining/bert.html) -- учебник с кодом
