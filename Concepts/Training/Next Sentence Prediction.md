---
title: "Next Sentence Prediction"
aliases: [NSP, предсказание следующего предложения]
type: concept
status: legacy
category: Training
papers:
  - "[[02 Areas/ML & DL/Papers/BERT]]"
  - "[[02 Areas/ML & DL/Papers/RoBERTa]]"
courses: []
sources:
  - "[Devlin et al. -- BERT (2019)](https://arxiv.org/abs/1810.04805)"
  - "[Liu et al. -- RoBERTa (2019)](https://arxiv.org/abs/1907.11692)"
  - "[Lan et al. -- ALBERT (2020)](https://arxiv.org/abs/1909.11942)"
  - "[Jay Alammar -- The Illustrated BERT](https://jalammar.github.io/illustrated-bert/)"
---

# Next Sentence Prediction (NSP)

## Зачем это придумали

BERT (Devlin et al., 2019) обучается на **двух** pre-training задачах одновременно:
1. **MLM** (Masked Language Modeling) -- восстановление маскированных токенов
2. **NSP** (Next Sentence Prediction) -- бинарная классификация: является ли предложение B реальным следующим за A

Мотивация NSP: многие downstream задачи работают с **парами предложений**:
- **QA**: вопрос + контекст (нужно понимать связь между ними)
- **NLI**: premise + hypothesis (нужно определить entailment/contradiction)
- **Paraphrase detection**: предложение A + предложение B

MLM учит понимать токены внутри одного контекста, но не учит явно понимать **отношения между предложениями**. NSP был задуман как дополнительный сигнал для inter-sentence reasoning.

## Как работает NSP

### Формирование обучающих пар

Из BERT (раздел 3.1):

```
Обучающий корпус разбивается на пары предложений (A, B):

50% -- IsNext (реальная пара):
  A: "The man went to the store."
  B: "He bought a gallon of milk."    --> label: IsNext

50% -- NotNext (случайная пара):
  A: "The man went to the store."
  B: "Penguins are flightless birds."  --> label: NotNext
```

### Вход в BERT

```
[CLS] The man went to the store . [SEP] He bought a gallon of milk . [SEP]
  |                                  |
  |-- Segment A embeddings           |-- Segment B embeddings
  |-- Token embeddings               |-- Token embeddings
  |-- Position embeddings             |-- Position embeddings
```

**Segment embeddings** -- дополнительный embedding layer, который маркирует каждый токен как принадлежащий предложению A или B.

### Предсказание

NSP prediction = **binary classifier** поверх `[CLS]` token:

$$P(\text{IsNext}) = \sigma(W_{NSP} \cdot h_{[CLS]} + b_{NSP})$$

где $h_{[CLS]}$ -- hidden state первого специального токена `[CLS]` после последнего Transformer layer.

**Точность** BERT на NSP: ~97-98%. Задача слишком проста.

### Совместное обучение с MLM

Итоговый loss BERT = сумма двух losses:

$$\mathcal{L} = \mathcal{L}_{MLM} + \mathcal{L}_{NSP}$$

Оба loss вычисляются на каждом batch одновременно.

## Почему NSP вредит: ablation от RoBERTa

Liu et al. (2019) провели тщательный ablation study, который показал, что NSP **не помогает и может вредить**.

### Четыре варианта формата данных

| Формат | NSP | Описание |
|--------|-----|----------|
| SEGMENT-PAIR | да | Как BERT: два сегмента, могут пересекать границы предложений |
| SENTENCE-PAIR | да | Строго два предложения (часто очень короткие) |
| FULL-SENTENCES | нет | Непрерывный текст из одного или нескольких документов, без NSP |
| **DOC-SENTENCES** | **нет** | Непрерывный текст из одного документа, без NSP |

### Результаты (RoBERTa Table 2)

| Формат | NSP | SQuAD 2.0 F1 | MNLI-m | SST-2 |
|--------|-----|-------------|--------|-------|
| SEGMENT-PAIR | да | 78.7 | 84.0 | 92.5 |
| SENTENCE-PAIR | да | 76.2 | 83.7 | 92.6 |
| FULL-SENTENCES | нет | 79.1 | 84.7 | 92.6 |
| **DOC-SENTENCES** | **нет** | **79.7** | **84.7** | **93.0** |

**Вывод 1**: Убирание NSP + обучение на document-level текстах **улучшает** все метрики.

**Вывод 2**: SENTENCE-PAIR (строгие предложения + NSP) -- **худший** вариант, потому что предложения слишком короткие и модель получает меньше контекста.

**Вывод 3**: DOC-SENTENCES (один документ, без NSP) -- **лучший** вариант.

### Почему NSP вредит: три гипотезы

**1. Задача слишком проста.** 97-98% accuracy означает, что NSP даёт **крайне мало обучающего сигнала**. Модель решает задачу через topic matching: если предложение A про магазин, а B про пингвинов -- это NotNext. Не нужно понимать логическую связь.

**2. Вредный побочный эффект на формат данных.** NSP требует пар предложений, что **ограничивает длину каждого сегмента**. Вместо 512 токенов контекста модель получает два коротких куска. Меньше контекста = хуже representations.

**3. Segment embeddings мешают.** Дополнительные segment embeddings добавляют complexity, но не несут полезной информации для большинства downstream задач.

## Альтернативы NSP

### RoBERTa: просто убрать NSP

Самый простой подход -- **не делать NSP**. Обучать только на MLM с длинными последовательностями из одного документа. Это стало стандартом.

### ALBERT: Sentence Order Prediction (SOP)

Lan et al. (2020) предложили заменить NSP на **SOP** -- предсказание **порядка** двух предложений:

```
SOP (Sentence Order Prediction):
  Positive: A, затем B (правильный порядок)
  Negative: B, затем A (инвертированный порядок)
```

**Почему SOP лучше NSP?** NSP можно решить через **topic matching** (разные документы -- разные топики). SOP **нельзя** решить через topic matching -- оба предложения из одного документа, одного топика. Нужно понимать **логический порядок**.

Из ALBERT ablations:

| Pre-training task | SQuAD 1.1 F1 | SQuAD 2.0 F1 | MNLI-m |
|-------------------|-------------|-------------|--------|
| NSP | 89.3 | 78.8 | 84.1 |
| **SOP** | **89.9** | **80.0** | **84.4** |
| Ничего (только MLM) | 89.1 | 78.0 | 83.7 |

SOP лучше NSP и лучше "ничего" -- это полезный inter-sentence objective, в отличие от NSP.

### XLNet: Permutation Language Modeling

XLNet (Yang et al., 2019) полностью отказался от NSP и использует **permutation language modeling**: предсказание токена в случайном порядке перестановки. Это позволяет моделировать bidirectional контекст без маскирования.

### DeBERTa: только MLM + Enhanced Mask Decoder

DeBERTa (He et al., 2020) тоже **убрал NSP**. Вместо inter-sentence objective использует Enhanced Mask Decoder для лучшего MLM prediction.

## Почему [CLS] token всё равно полезен

Несмотря на удаление NSP, **`[CLS]` token остаётся** в RoBERTa и DeBERTa. Его representation агрегирует информацию о всей последовательности и используется для classification при fine-tuning.

Но без NSP, `[CLS]` обучается **только через MLM** (attention к маскированным позициям), что оказывается достаточным.

## Влияние NSP на современные модели

| Модель | NSP | Альтернатива |
|--------|-----|-------------|
| BERT | да | -- |
| RoBERTa | **нет** | Только MLM |
| ALBERT | **нет** | SOP |
| XLNet | **нет** | Permutation LM |
| DeBERTa | **нет** | Enhanced Mask Decoder |
| ELECTRA | **нет** | Replaced Token Detection |
| GPT-2/3/4 | **нет** | CLM (decoder, нет пар) |
| T5 | **нет** | Span corruption |

**Практический вывод**: ни одна успешная модель после BERT не использует NSP. При fine-tuning на новых задачах -- **предпочитай RoBERTa/DeBERTa** (без NSP) вместо BERT (с NSP) как базовую модель.

## Историческое значение

NSP -- это **первая попытка** добавить inter-sentence reasoning в pre-training. Идея правильная (понимание связей между предложениями важно), но реализация оказалась неудачной:
- Задача слишком простая (topic matching вместо reasoning)
- Побочные эффекты на формат данных перевешивают пользу
- SOP от ALBERT -- правильная версия той же идеи

NSP -- хороший пример того, как **ablation study** (RoBERTa) может опровергнуть "очевидную" идею и привести к улучшению.

## Key papers

- [[02 Areas/ML & DL/Papers/BERT]] -- введён NSP как часть pre-training
- [[02 Areas/ML & DL/Papers/RoBERTa]] -- ablation study, показавший что NSP вреден

## Related concepts

- [[02 Areas/ML & DL/Concepts/Training/Masked Language Modeling|Masked Language Modeling (MLM)]] -- основной pre-training objective BERT
- [[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]] -- NSP как часть multi-task pre-training
- [[02 Areas/ML & DL/Concepts/Architectures/BERT|BERT]] -- модель, использующая NSP
- [[02 Areas/ML & DL/Concepts/Architectures/RoBERTa|RoBERTa]] -- улучшенная модель без NSP

## Дополнительные ресурсы

- [Jay Alammar -- The Illustrated BERT](https://jalammar.github.io/illustrated-bert/) -- визуализация NSP и MLM
- [Lan et al. -- ALBERT (2020)](https://arxiv.org/abs/1909.11942) -- SOP как улучшение NSP
- [Yang et al. -- XLNet (2019)](https://arxiv.org/abs/1906.08237) -- Permutation LM вместо NSP
