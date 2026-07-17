---
title: "Seq2Seq"
aliases: [Seq2Seq, Sequence-to-Sequence, Encoder-Decoder]
type: concept
category: NLP
papers: []
courses:
  - "[[02 Areas/ML & DL/Courses/Stanford CS224N/Lecture 06 — LSTM and Machine Translation|CS224N L06]]"
  - "[[02 Areas/ML & DL/Courses/Stanford CS224N/Lecture 07 — Attention|CS224N L07]]"
sources:
  - "[Sutskever, Vinyals, Le — Sequence to Sequence Learning with Neural Networks (NeurIPS 2014)](https://arxiv.org/abs/1409.3215)"
  - "[Cho et al. — Learning Phrase Representations using RNN Encoder-Decoder (EMNLP 2014)](https://arxiv.org/abs/1406.1078)"
  - "[Bahdanau et al. — Neural Machine Translation by Jointly Learning to Align and Translate (ICLR 2015)](https://arxiv.org/abs/1409.0473)"
---

# Seq2Seq — Sequence-to-Sequence архитектура

## Зачем это нужно: задачи переменной длины

Многие NLP-задачи отображают **последовательность произвольной длины** в **другую последовательность произвольной длины**:
- **Machine Translation:** «Я люблю машинное обучение» → «I love machine learning»
- **Summarization:** статья → краткое изложение
- **Dialogue:** реплика пользователя → ответ системы
- **Speech recognition:** аудио → текст
- **Parsing:** предложение → дерево разбора (линеаризованное)

Классические RNN-архитектуры (до 2014) не подходили: они либо требовали фиксированной длины выхода, либо длина выхода должна была совпадать со входом (tagging).

**Seq2Seq** (Sutskever, Vinyals, Le, Google, NeurIPS 2014; параллельно Cho et al., 2014) — первая нейросетевая архитектура, нативно работающая в режиме **sequence → sequence** с разными длинами.

## Архитектура Encoder-Decoder

Модель состоит из **двух рекуррентных сетей**:

### Encoder

Читает входную последовательность $x_1, \ldots, x_{T_x}$ и сжимает её в **фиксированный контекстный вектор** $c$:

$$h_t = f(h_{t-1}, x_t), \quad c = h_{T_x}$$

где $f$ — LSTM/GRU-ячейка. Вектор $c$ — последнее скрытое состояние encoder'а — должен содержать «смысл всего входа».

### Decoder

Генерирует выходную последовательность $y_1, \ldots, y_{T_y}$ **авторегрессивно**, начиная с контекста $c$:

$$s_t = g(s_{t-1}, y_{t-1}, c), \quad P(y_t \mid y_{<t}, x) = \text{softmax}(W s_t)$$

Decoder — обычная [[02 Areas/ML & DL/Concepts/NLP/Language Model|языковая модель]], **условная** на $c$. На обучении используется teacher forcing (подаём правильный $y_{t-1}$), на инференсе — предсказанный предыдущий токен.

### Схема

```
Encoder:   x_1 → x_2 → x_3 → x_4 → [EOS]
            ↓     ↓     ↓     ↓      ↓
           h_1 → h_2 → h_3 → h_4 → h_5 = c
                                         ↓
Decoder:                            [BOS] → y_1 → y_2 → y_3 → [EOS]
                                      ↓     ↓     ↓     ↓
                                     s_1 → s_2 → s_3 → s_4
```

Два RNN соединены через последнее скрытое состояние.

## Обучение

Модель обучается end-to-end через максимизацию условной log-likelihood:

$$\mathcal{L} = -\sum_{(x,y) \in D} \log P(y \mid x; \theta) = -\sum \sum_{t=1}^{T_y} \log P(y_t \mid y_{<t}, x; \theta)$$

Градиент распространяется через весь граф: от decoder loss → через $c$ → в encoder. **Обе сети обучаются совместно.**

Оригинальная статья Sutskever использовала 4-слойные LSTM с 1000 hidden units, и — ключевая эмпирическая находка — **разворот входной последовательности**: «ABC → WXYZ» обучалось хуже, чем «CBA → WXYZ». Причина: при развороте начальные слова входа оказываются ближе к началу выхода по временным шагам, и градиенты лучше текут через LSTM.

## Декодирование на инференсе

Нужно найти $y^* = \arg\max_y P(y | x)$. Поиск точного максимума — экспоненциальный. Приближения:

### Greedy decoding

На каждом шаге брать самый вероятный токен: $y_t = \arg\max P(y_t | y_{<t}, x)$. Быстро, но плохо — локальная жадность не даёт глобальный оптимум.

### Beam search

Поддерживать top-$k$ гипотез ($k$ = beam width) на каждом шаге, разворачивать все, выбирать best-$k$ продолжений. Стандарт в MT, $k = 4 - 10$.

### Sampling

Сэмплирование из $P(y_t|\cdot)$ для генерации разнообразия. Используется в dialogue, creative generation. См. [[02 Areas/ML & DL/Concepts/Inference/Sampling|Sampling]].

## Information Bottleneck — главная проблема

В базовом Seq2Seq **весь смысл входного предложения должен уместиться в один вектор $c$** фиксированной длины (обычно 500-1000-мерный).

Для короткого предложения из 5 слов это работает. Для предложения из 50 слов — вектор становится «узким горлышком»:

- Длинные предложения переводятся **значительно хуже** (Cho et al. 2014 эмпирически показали деградацию качества с ростом длины)
- Модель «забывает» начало входа к моменту decoding'а конца
- Нет механизма **выборочного** доступа к частям входа

Это фундаментальная проблема архитектуры — не недостаток обучения.

## Attention как решение bottleneck

**Bahdanau et al. (ICLR 2015)** радикально решили проблему: **decoder должен иметь доступ ко всем скрытым состояниям encoder'а**, а не только к последнему.

Вместо одного $c$ — на каждом шаге decoder'а вычисляется **свой контекст** $c_t$ как взвешенная сумма encoder states:

$$c_t = \sum_{i=1}^{T_x} \alpha_{t,i} h_i, \quad \alpha_{t,i} = \frac{\exp(\text{score}(s_{t-1}, h_i))}{\sum_j \exp(\text{score}(s_{t-1}, h_j))}$$

Веса $\alpha_{t,i}$ — «внимание» decoder на позицию $i$ входа при генерации токена $t$. Функция score — обучаемая (additive в Bahdanau, dot-product в Luong).

**Ключевые эффекты:**
1. **Bottleneck снят** — вся информация входа доступна на каждом шаге decoding'а
2. **Soft alignment** — $\alpha_{t,i}$ интерпретируются как мягкое выравнивание между словами входа и выхода (аналог word alignment в SMT)
3. **Качество на длинных предложениях восстановлено**

См. [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]] для полного описания.

Attention — одно из самых влиятельных изобретений в NLP за десятилетие. Именно с него начинается путь к Transformer.

## Варианты и улучшения

### Copy mechanism / Pointer Networks

Для summarization и QA часто нужно **копировать слова из входа** (имена, числа, редкие термины). Copy mechanism (See et al., Pointer-Generator, 2017) смешивает softmax по словарю с распределением внимания, позволяя выводить токены напрямую из входа.

### Coverage mechanism

Для борьбы с **повторениями** (частая болезнь Seq2Seq): агрегировать attention-веса по всем decoder-шагам и штрафовать модель за повторное внимание на те же позиции.

### Bidirectional encoder

Стандарт с 2015: encoder — двунаправленная RNN, объединяющая forward и backward скрытые состояния. Даёт богаче representations для attention.

## Применения

Seq2Seq с attention (2015-2017) стал универсальной архитектурой для **любой** задачи «текст → текст»:

- **Machine Translation:** Google NMT (2016) — перевод Google Translate целиком перешёл на Seq2Seq+attention, заменив статистический phrase-based MT
- **Abstractive Summarization:** See et al. 2017, pointer-generator networks
- **Dialogue systems:** Vinyals & Le 2015 — первые end-to-end neural chatbots
- **Question Answering (generative):** ответы формулируются как генерация
- **Semantic Parsing:** предложение → формальное представление (SQL, logical form)
- **Speech recognition:** Listen, Attend and Spell (Chan et al. 2016)
- **Image Captioning:** CNN encoder + RNN decoder с attention (Show, Attend and Tell, 2015)

## Эволюция: от RNN-Seq2Seq к Transformer

**«Attention is all you need»** (Vaswani et al., NeurIPS 2017) — революция: если attention и так несёт основную работу, **нужны ли вообще RNN?**

Transformer сохраняет encoder-decoder парадигму Seq2Seq, но:
- Заменяет RNN на stacks of **self-attention** layers
- Encoder использует self-attention, decoder — self-attention + cross-attention к encoder'у
- Параллелизуется по времени (в отличие от последовательной RNN)
- Масштабируется до миллиардов параметров

См. [[02 Areas/ML & DL/Papers/Attention Is All You Need|Attention Is All You Need]] и [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]].

**Последующая эволюция:**
- **T5, BART** (2019-2020) — transformer seq2seq, предобученные на denoising → SOTA на MT, summarization
- **Decoder-only LLM** (GPT-family) — для многих задач оказалось, что **encoder не нужен**: условную генерацию $P(y|x)$ можно делать, просто подавая $x$ в качестве префикса decoder'у
- **Reasoning LM** (DeepSeek-R1, o1) — дальнейшее развитие decoder-only парадигмы

Seq2Seq как отдельная архитектура в 2024+ используется всё реже, но сама **концепция encoder-decoder** остаётся (T5, Whisper, vision-language models).

## Сравнение архитектур

| Свойство | Базовый Seq2Seq (2014) | Seq2Seq + Attention (2015) | Transformer (2017) |
|----------|------------------------|----------------------------|--------------------|
| Контекст на decoder | Один вектор $c$ | Взвешенная сумма всех $h_i$ | Multi-head cross-attention |
| Bottleneck | Есть | Нет | Нет |
| Параллелизация | Нет (RNN sequential) | Нет | Да (по времени) |
| Long-range deps | Слабо | Средне | Хорошо |
| Качество на длинных | Плохо | Хорошо | Отлично |
| Масштабирование | Ограничено | Ограничено | Линейное до триллионов |

## Related concepts

- [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]] — механизм, решивший bottleneck-проблему Seq2Seq
- [[02 Areas/ML & DL/Concepts/Architectures/Encoder-Decoder|Encoder-Decoder]] — архитектурная парадигма, частью которой является Seq2Seq
- [[02 Areas/ML & DL/Papers/Attention Is All You Need|Attention Is All You Need]] — статья, давшая Transformer-развитие Seq2Seq
- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] — современная реализация encoder-decoder
- [[02 Areas/ML & DL/Concepts/NLP/RNN/RNN|RNN]] / LSTM — архитектурная основа оригинального Seq2Seq
- [[02 Areas/ML & DL/Concepts/NLP/Cross-Attention|Cross-Attention]] — обобщение attention между двумя последовательностями
- [[02 Areas/ML & DL/Concepts/NLP/Language Model|Language Model]] — decoder Seq2Seq является условной LM
- [[02 Areas/ML & DL/Concepts/Evaluation/BLEU Score|BLEU]] — основная метрика для Seq2Seq в MT
