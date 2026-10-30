---
title: "DeBERTa: Decoding-enhanced BERT with Disentangled Attention"
url: https://arxiv.org/abs/2006.03654v6
authors: [Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen]
year: 2021
date_reviewed: 2026-10-22
type: source-note
status: legacy
category: paper
tags:
  - BERT
  - arch
Date: 2020-06-05
Organization: Microsoft
Parent item:
  - "[[BERT]]"
Status: Done
concepts:
  - "[[02 Areas/ML & DL/Concepts/Architectures/DeBERTa|DeBERTa]]"
  - "[[02 Areas/ML & DL/Concepts/Architectures/Encoder-only|Encoder-only]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Disentangled Attention|Disentangled Attention]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/Positional Encoding|Positional Encoding]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/GLUE|GLUE]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Masked Language Modeling|MLM]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/Transfer Learning|Transfer Learning]]"
raw: "[[02 Areas/ML & DL/raw/papers/deberta/paper.txt]]"
---

# DeBERTa: Decoding-enhanced BERT with Disentangled Attention

**Authors:** Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen (Microsoft)
**Published:** ICLR 2021 (arXiv:2006.03654v6, Oct 2021)
**URL:** https://arxiv.org/abs/2006.03654v6

Подробный разбор с примерами:
[[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03 BERT, RoBERTa и DeBERTa|BERT, RoBERTa и DeBERTa]].
Ниже рассматривается исходная DeBERTa по версии статьи v6;
[DeBERTa V3](https://arxiv.org/abs/2111.09543) меняет также задачу предобучения.

## TL;DR

В disentangled attention отдельно проецируются содержание и относительные позиции. Оценка пары содержит три слагаемых; внимание остаётся полным, не диагональным. Enhanced Mask Decoder использует абсолютные позиции при восстановлении MLM-токенов. LARGE обучена примерно на 78 ГБ текста, а отдельная 1.5B-конфигурация — на 160 ГБ. Результат 89.9 против human baseline 89.8 относится к SuperGLUE test у 1.5B + SiFT, а не к общей оценке языковых способностей.

## Problem

При сложении токенных и абсолютных позиционных эмбеддингов, как в BERT, проекции Q/K смешивают оба сигнала. DeBERTa задаёт отдельные проекции содержания и относительного положения и явно вычисляет их взаимодействия (§3.1).

Для EMD авторы приводят `a new [MASK] opened beside the new [MASK]`: оба пропуска следуют за `new`, но выполняют разные синтаксические роли. Это мотивация добавить абсолютные позиции при MLM-декодировании; она не доказывает, что весь относительный контекст у пропусков одинаков или что модель принципиально не может их различить.

## Method

### 1. Disentangled Attention (Section 3.1)

Каждый токен в позиции i представлен двумя векторами:

- `H_i` — content embedding
- `P_{i|j}` — relative position embedding от i до j

Attention score между i и j декомпозируется на **3 (из 4)** компонентов:

```
Ã_{i,j} = Q^c_i K^c_j^T         # (a) content-to-content
         + Q^c_i K^r_{δ(i,j)}^T  # (b) content-to-position
         + K^c_j Q^r_{δ(j,i)}^T  # (c) position-to-content
```
Четвёртый term (position-to-position) удалён как несущественный при relative encoding.

Масштабирующий фактор `1/√(3d)` вместо стандартного `1/√d` (три слагаемых, а не одно).

Таблица `P ∈ R^{2k×d}` общая для слоёв; в исходных опытах `k=512`. `O(kd)` относится к хранению относительных векторов вместо отдельного вектора для каждой пары. Матрица оценок внимания по-прежнему имеет размер `N×N`. Clipping расстояния выбирает крайнюю строку таблицы, но не запрещает дальний ключ.

### 2. Enhanced Mask Decoder (EMD) (Section 3.2)

В BERT абсолютные позиции складываются с токенными векторами на входе. DeBERTa использует их позже, в EMD. Авторы предполагают, что такое разделение облегчает изучение относительного положения; это их объяснение результата, не установленный универсальный закон.

EMD получает фиксированную контекстную память `H` и отдельно задаваемые запросы `I`. Результат первого attention-преобразования становится запросами второго при той же памяти. В опыте используются два прохода с общими весами, затем MLM-голова (§3.2, приложение A.8, рис. 2). В закреплённом официальном коде при `position_biased_input=False` память взята из предпоследнего выхода энкодера, первые запросы равны `H + absolute positions`, а последний блок вызывается дважды. EMD не сводится к bias словарных логитов и не генерирует отдельную последовательность.

В табл. 4 версии v6 удаление EMD снижает MNLI-m с 86.3 до 86.1, SQuAD 2.0 F1 с 82.5 до 81.3 и RACE accuracy с 71.7 до 70.3. Это BASE-абляция на Wikipedia/BookCorpus, не результаты LARGE.

### 3. SiFT: Scale-invariant Fine-Tuning (Section 4)

При fine-tuning возмущения применяются к нормализованным входным эмбеддингам; регуляризация сближает предсказания на исходном и возмущённом входах. Авторы сообщают об улучшении стабильности и используют SiFT для DeBERTa-1.5B на SuperGLUE (§4).

### Pre-training Details

- **Размер Large:** L=24, H=1024, A=16 — те же глубина, ширина и число голов, что у BERT-Large/RoBERTa-Large; attention и EMD отличаются. BPE-50K (как RoBERTa).
- **Данные:** Wikipedia (12GB) + BookCorpus (6GB) + OpenWebText (38GB) + STORIES (31GB) = **78GB** после дедупликации. (RoBERTa: 160GB)
- **Training:** batch=2K, 1M steps, 6 DGX-2 machines (96 V100), ~20 дней
- Предъявлений последовательностей: около 2 млрд против 4 млрд у сравниваемых RoBERTa/XLNet. Это не число уникальных текстов.

## Key Results

### GLUE Dev Set (Table 1)

| Модель | MNLI-m/mm | CoLA | QQP | SST-2 | STS-B | QNLI | RTE | MRPC | Avg |
|--------|-----------|------|-----|-------|-------|------|-----|------|-----|
| BERT_L | 86.6/- | 60.6 | 91.3 | 93.2 | 90.0 | 92.3 | 70.4 | 88.0 | 84.05 |
| RoBERTa_L | 90.2/90.2 | 68.0 | 92.2 | 96.4 | 92.4 | 93.9 | 86.6 | 90.9 | 88.82 |
| XLNet_L | 90.8/90.8 | 69.0 | 92.3 | 97.0 | 92.5 | 94.9 | 85.9 | 90.8 | 89.15 |
| ELECTRA_L | 90.9/- | 69.1 | 92.4 | 96.9 | 92.6 | 95.0 | 88.0 | 90.8 | 89.46 |
| DeBERTa_L | 91.1/91.1 | 70.5 | 92.3 | 96.8 | 92.8 | 95.3 | 88.3 | 91.9 | 90.00 |

Колонка Avg сохранена как в статье. У DeBERTa восемь напечатанных метрик суммируются в 719.0 и дают простое среднее 89.875, которое не совпадает с указанными 90.00. Причину расхождения таблица не поясняет. Этот сообщённый авторами Avg также нельзя сравнивать напрямую с тестовым GLUE 88.5 ансамбля RoBERTa.

### Extended NLU Benchmarks (Table 2)

| Модель | SQuAD v1.1 F1/EM | SQuAD v2.0 F1/EM | RACE | ReCoRD F1/EM | SWAG |
|--------|-----------------|-----------------|------|-------------|------|
| RoBERTa_L | 94.6/88.9 | 89.4/86.5 | 83.2 | 90.6/90.0 | 89.9 |
| XLNet_L | 95.1/89.7 | 90.6/87.9 | 85.4 | – | – |
| **DeBERTa_L** | **95.5/90.1** | **90.7/88.0** | **86.8** | **91.4/91.0** | **90.8** |

По табл. 1–2 улучшения над RoBERTa составляют 0.9 процентного пункта MNLI, 1.3 пункта SQuAD 2.0 F1 и 3.6 пункта RACE. В аннотации использована другая исходная цифра SQuAD RoBERTa — 88.4; для этого сравнения SQuAD взята табл. 2, где указано 89.4.

### Base Models (Table 3)

| Модель | MNLI-m/mm | SQuAD v1.1 F1/EM | SQuAD v2.0 F1/EM |
|--------|-----------|-----------------|-----------------|
| RoBERTa_Base | 87.6/- | 91.5/84.6 | 83.7/80.5 |
| XLNet_Base | 86.8/- | –/– | –/80.2 |
| **DeBERTa_Base** | **88.8/88.5** | **93.1/87.2** | **86.2/83.1** |

В трёх приведённых dev-задачах BASE превосходит сравниваемые RoBERTa/XLNet. На MNLI-m разница равна 1.2 и 2.0 процентного пункта соответственно. Это результат указанных рецептов, а не изолированная абляция архитектуры.

### SuperGLUE: 1.5B Model — First Human Parity

Отдельная конфигурация DeBERTa с 48 слоями и 1.5 млрд параметров обучена на 160 ГБ, имеет словарь 128K, общие проекции содержания/позиций и дополнительную свёртку (§5.3):

- Single model + SiFT: 89.9 (Human: 89.8) на SuperGLUE test
- **Ensemble: 90.3** (первое место на leaderboard на Jan 6, 2021)
- T5-11B в той же таблице получает 89.3. Это сравнение итоговых систем с разными рецептами, не доказательство приоритета архитектуры над числом параметров.

### Ablation: Disentangled Attention components (Table 4)

| Вариант | MNLI-m/mm | SQuAD v1.1 | SQuAD v2.0 | RACE |
|---------|-----------|-----------|-----------|------|
| DeBERTa_Base | 86.3/86.2 | 92.1/86.1 | 82.5/79.3 | 71.7 |
| -EMD | 86.1/86.1 | 91.8/85.8 | 81.3/78.0 | 70.3 |
| -C2P (no content-to-position) | 85.9/85.7 | 91.6/85.8 | 81.3/78.3 | 69.3 |
| -P2C (no position-to-content) | 86.0/85.8 | 91.7/85.7 | 80.8/77.6 | 69.6 |

Числа относятся к табл. 4 версии v6. В описании пунктов перед таблицей перепутаны буквенные ссылки (b)/(c); названия C2P/P2C здесь сопоставлены с формулой 4 и подписями строк таблицы.

### NLG: WikiText-103

В приложении A.4 используется причинная маска и отдельно обучается DeBERTa-MT с совместными MLM/ARLM-целями, 200 тыс. шагов. В табл. 10 Test PPL равен 21.6 у RoBERTa, 19.9 у DeBERTa и 19.5 у DeBERTa-MT. Это не результат свободной генерации обычного двунаправленного MLM-checkpoint без дополнительного обучения.

## My notes

- Сравнения LARGE и BASE нужно отделять от абляции на меньшем корпусе и от 1.5B + SiFT.
- Disentangled attention разделяет три вида оценки связи; оно не задаёт диагональную или локальную маску.
- EMD использует дополнительные attention-преобразования запросов к фиксированной памяти. Его устройство нельзя свести к словарному bias.
- Выводы о качестве, числе параметров и затратах требуют совпадающего протокола. SuperGLUE 89.9 не является общей оценкой превосходства над человеком или T5.
