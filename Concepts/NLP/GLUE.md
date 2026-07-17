---
title: "GLUE / SuperGLUE"
aliases: [GLUE, SuperGLUE, General Language Understanding Evaluation]
type: concept
status: legacy
category: NLP
papers:
  - "[[02 Areas/ML & DL/Papers/BERT]]"
  - "[[02 Areas/ML & DL/Papers/RoBERTa]]"
  - "[[02 Areas/ML & DL/Papers/DeBERTa]]"
  - "[[02 Areas/ML & DL/Papers/T5]]"
  - "[[02 Areas/ML & DL/Papers/Revisiting Few-sample BERT Fine-tuning]]"
  - "[[02 Areas/ML & DL/Papers/GPT 3.0]]"
courses: []
sources:
  - "[GLUE Benchmark (gluebenchmark.com)](https://gluebenchmark.com/)"
  - "[SuperGLUE Benchmark (super.gluebenchmark.com)](https://super.gluebenchmark.com/)"
  - "[Wang et al. — GLUE: A Multi-Task Benchmark (2018)](https://arxiv.org/abs/1804.07461)"
  - "[Wang et al. — SuperGLUE (2019)](https://arxiv.org/abs/1905.00537)"
---

# GLUE / SuperGLUE

## Зачем нужен единый benchmark

До GLUE (2018) каждая NLP-модель сравнивалась на разных датасетах, часто cherry-picked авторами. Не было единого стандарта для оценки «понимания языка». GLUE решает эту проблему: **один benchmark, 9 задач, один финальный score** --- как ImageNet для computer vision, но для NLU.

GLUE (General Language Understanding Evaluation, Wang et al., 2018) стал **де-факто стандартом эпохи BERT** (2018--2022). Когда модели быстро «насытили» GLUE, появился **SuperGLUE** (2019) --- более сложный преемник.

## GLUE: 9 задач

### Breakdown по задачам

| Задача | Тип | Train | Что измеряет | Метрика |
|--------|-----|:-----:|--------------|---------|
| **CoLA** | Acceptability | 8.5K | Грамматическая корректность | Matthews Corr |
| **SST-2** | Sentiment | 67K | Тональность рецензий | Accuracy |
| **MRPC** | Paraphrase | 3.7K | Парафраз-детекция | F1 / Acc |
| **STS-B** | Similarity | 5.7K | Семантическая близость (1--5) | Spearman's $\rho$ |
| **QQP** | Paraphrase | 364K | Дубликаты вопросов (Quora) | F1 / Acc |
| **MNLI** | NLI | 393K | Textual entailment (3 класса) | Accuracy |
| **QNLI** | QA-as-NLI | 105K | Содержит ли предложение ответ | Accuracy |
| **RTE** | NLI | 2.5K | Textual entailment (binary) | Accuracy |
| **WNLI** | Coreference | 634 | Winograd-style coreference | Accuracy |

**Финальный GLUE score:** среднее по всем задачам. На практике WNLI часто исключается (датасет маленький и проблемный --- majority class baseline даёт 65.1%).

### Что на самом деле измеряют эти задачи

**Single-sentence tasks (CoLA, SST-2):** может ли модель понять одно предложение --- его грамматику и тональность.

**Similarity и paraphrase (MRPC, STS-B, QQP):** понимание семантической эквивалентности двух предложений.

**Natural Language Inference (MNLI, QNLI, RTE, WNLI):** самая важная группа. Задача: даны premise и hypothesis, определить отношение: entailment (следует), contradiction (противоречит), neutral (нейтрально). NLI считается «водородным тестом» на понимание языка --- требует логического рассуждения, знание мира, понимание лексической семантики.

### Формат входа для BERT-style моделей

```
Single sentence:  [CLS] This movie was great! [SEP]         -> SST-2
Sentence pair:    [CLS] Premise [SEP] Hypothesis [SEP]       -> MNLI
                  [CLS] Question [SEP] Candidate answer [SEP] -> QNLI
```

## Гонка на GLUE: ключевые результаты

| Модель | Год | GLUE Score | Прорыв |
|--------|:---:|:---------:|--------|
| Human baseline | --- | ~87 | --- |
| ELMo + BiLSTM | 2018 | ~70 | Contextual embeddings |
| OpenAI GPT | 2018 | 72.8 | Pre-train + fine-tune |
| **BERT-Large** | **2018** | **80.5** | Bidirectional + MLM |
| XLNet | 2019 | 88.4 | Permutation LM |
| **RoBERTa** | **2019** | **88.5** | Training recipe fix |
| ALBERT | 2019 | 89.4 | Parameter sharing |
| T5-11B | 2019 | ~90.3 | Text-to-text |
| DeBERTa-v3-large | 2021 | **92+** | Disentangled attention |

**Ключевые наблюдения:**

1. **BERT -> RoBERTa: +8 пунктов** без изменения архитектуры --- только рецепт обучения (см. [[02 Areas/ML & DL/Concepts/Architectures/RoBERTa|RoBERTa]])
2. **BERT обогнал human baseline** на части задач уже в 2019 году
3. **К 2021 модели превзошли человека** на GLUE в целом --- benchmark «насытился»

### Детальные результаты BERT

Из [[02 Areas/ML & DL/Papers/BERT]] (Table 1):

| Задача | BERT-Large | GPT (fine-tuned) | Прирост |
|--------|:---------:|:----------------:|:-------:|
| MNLI-m | 86.7% | 82.1% | +4.6% |
| QQP | 72.1 F1 | 70.3 F1 | +1.8% |
| RTE | 70.1% | 56.0% | **+14.1%** |
| SST-2 | 94.9% | 91.3% | +3.6% |

Самый большой прирост на **RTE** --- маленький датасет (2.5K), где bidirectional pre-training даёт огромное преимущество.

Ключевое ablation из BERT paper: **bidirectional vs left-to-right** -> -10.7 F1 на SQuAD. Это доказывает, что bidirectional self-attention критически важен для NLU задач.

## SuperGLUE: когда GLUE стал слишком лёгким

К середине 2019 модели начали превышать human baseline на GLUE. Wang et al. создали SuperGLUE --- более сложный benchmark с 8 задачами, требующими более глубокого рассуждения.

### 8 задач SuperGLUE

| Задача | Тип | Train | Сложность |
|--------|-----|:-----:|-----------|
| **BoolQ** | Yes/No QA | 9.4K | Passage comprehension |
| **CB** | NLI (CommitmentBank) | 250 | Мало данных! |
| **COPA** | Causal reasoning | 400 | Причинно-следственные связи |
| **MultiRC** | Multi-hop QA | 5.1K | Несколько правильных ответов |
| **ReCoRD** | Cloze (news) | 101K | Entity-based reading |
| **RTE** | NLI | 2.5K | Те же данные, что в GLUE |
| **WiC** | Word-in-context | 6K | Полисемия: "bank" в двух контекстах |
| **WSC** | Winograd schema | 554 | Coreference: "The trophy didn't fit in the suitcase because **it** was too big" |

**Human baseline: 89.8**

### Гонка на SuperGLUE

| Модель | Год | SuperGLUE | Vs Human |
|--------|:---:|:---------:|:--------:|
| BERT++ | 2019 | 69.0 | -20.8 |
| T5-11B | 2020 | 88.9 | -0.9 |
| **DeBERTa XXL** | **2021** | **89.9** | **+0.1** |

**DeBERTa (1.5B параметров) стала первой моделью, превзошедшей человека на SuperGLUE** (89.9 vs 89.8). Это веха: BERT-семейство достигло human-level NLU на академических benchmarks.

## Нестабильность на малых датасетах

Из [[02 Areas/ML & DL/Papers/Revisiting Few-sample BERT Fine-tuning]]: маленькие GLUE-датасеты (RTE, MRPC, CoLA, STS-B) демонстрируют **крайнюю нестабильность** при fine-tuning:

- **RTE (2.5K train):** 48% «degenerate» runs --- модель сходится к majority class baseline
- Причина: **bias в оптимизаторе Adam** на ранних шагах обучения
- Решение: re-initialization верхних слоёв BERT + bias correction -> +3.1% accuracy (69.5 -> 72.6)

Это важный урок: **маленькие benchmark-датасеты могут давать завышенную дисперсию результатов**, что искажает сравнение моделей.

## Ограничения GLUE / SuperGLUE и что пришло на смену

### Что GLUE/SuperGLUE **не** измеряют

| Не покрыто | Почему важно |
|-----------|-------------|
| **Generation quality** | GLUE --- только classification/regression |
| **Reasoning** | Нет math reasoning, logic chains |
| **Knowledge** | Нет проверки фактических знаний |
| **Long context** | Все задачи --- короткие тексты |
| **Multi-turn dialogue** | Нет conversational evaluation |
| **Safety / toxicity** | Нет оценки вредного контента |
| **Code** | Нет programming tasks |

### Benchmark saturation

К 2021--2022 году GLUE и SuperGLUE были «решены»: модели стабильно превышали human baseline. Benchmark перестал различать модели --- все показывали 90+.

### Наследники

| Benchmark | Год | Что измеряет | Актуальность |
|-----------|:---:|-------------|:------------:|
| **MMLU** | 2021 | Знания по 57 предметам | Высокая |
| **BIG-Bench** | 2022 | 204 задачи (reasoning, knowledge, etc.) | Средняя |
| **HumanEval** | 2021 | Coding (Python functions) | Высокая |
| **Arena ELO** | 2023 | Human preference в live chat | Высокая |
| **HELM** | 2022 | Holistic evaluation (accuracy + calibration + fairness) | Средняя |

## Почему GLUE/SuperGLUE всё ещё важны

1. **Исторический стандарт:** понять эволюцию NLP 2018--2022 невозможно без GLUE --- все ключевые статьи ([[02 Areas/ML & DL/Papers/BERT|BERT]], [[02 Areas/ML & DL/Papers/RoBERTa|RoBERTa]], [[02 Areas/ML & DL/Papers/DeBERTa|DeBERTa]], [[02 Areas/ML & DL/Papers/T5|T5]]) сравниваются именно по GLUE
2. **Baseline для encoder-only моделей:** BERT-style модели по-прежнему оцениваются на GLUE
3. **Методологический урок:** показали, как быстро benchmarks «насыщаются» и почему нужны постоянно обновляемые eval-сьюты
4. **Human-level milestone:** SuperGLUE --- первый NLU-benchmark, где машина превзошла человека

## Key papers

- [[02 Areas/ML & DL/Papers/BERT]] --- первая модель с GLUE 80+, детальные ablation
- [[02 Areas/ML & DL/Papers/RoBERTa]] --- GLUE 88.5 через оптимизацию рецепта обучения
- [[02 Areas/ML & DL/Papers/DeBERTa]] --- SuperGLUE 89.9 > human 89.8
- [[02 Areas/ML & DL/Papers/Revisiting Few-sample BERT Fine-tuning]] --- нестабильность на малых GLUE-датасетах

## Related concepts

- [[02 Areas/ML & DL/Concepts/Training/Fine-tuning|Fine-tuning]] --- основной метод для GLUE
- [[02 Areas/ML & DL/Concepts/Architectures/BERT|BERT]] --- модель, для которой GLUE стал главным benchmark
- [[02 Areas/ML & DL/Concepts/Architectures/RoBERTa|RoBERTa]] --- GLUE SOTA 2019
- [[02 Areas/ML & DL/Concepts/Architectures/DeBERTa|DeBERTa]] --- SuperGLUE SOTA, human-level
- [[02 Areas/ML & DL/Concepts/NLP/Text Classification|Text Classification]] --- GLUE содержит classification задачи
- [[02 Areas/ML & DL/Concepts/NLP/Transfer Learning|Transfer Learning]] --- GLUE измеряет transfer quality
