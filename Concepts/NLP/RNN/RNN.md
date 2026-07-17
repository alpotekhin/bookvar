---
title: "Recurrent Neural Networks (RNN)"
aliases: [RNN, Recurrent Neural Network, рекуррентная нейронная сеть, LSTM, GRU, Vanilla RNN]
type: concept
status: legacy
category: NLP
papers:
  - "[[02 Areas/ML & DL/Papers/Attention Is All You Need|Attention Is All You Need]]"
courses:
  - "[[02 Areas/ML & DL/Courses/Stanford CS224N/07 — Attention|CS224N Lecture 7]]"
sources:
  - "[Andrej Karpathy — The Unreasonable Effectiveness of RNNs (2015)](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)"
  - "[Christopher Olah — Understanding LSTM Networks (2015)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)"
  - "[d2l.ai — Recurrent Neural Networks](https://d2l.ai/chapter_recurrent-neural-networks/)"
  - "[Lena Voita — RNN & Seq2Seq](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html)"
---

# Recurrent Neural Networks (RNN)

## Зачем нужны рекуррентные сети

Обычные нейронные сети (MLP, CNN) работают с входами **фиксированного размера**: изображение 224x224, вектор из 768 чисел. Но язык --- это **последовательность переменной длины**: предложение может быть из 3 слов или из 50.

Рекуррентные нейронные сети (RNN) решают эту проблему: они обрабатывают последовательность **по одному элементу**, передавая скрытое состояние (hidden state) от шага к шагу. Это позволяет модели «помнить» контекст предыдущих элементов.

RNN доминировали в NLP с 2013 по 2017 год --- от language modeling и machine translation до speech recognition и text generation. В 2017 [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] заменил RNN в большинстве задач, но понимание RNN критически важно для понимания эволюции NLP и мотивации attention-механизмов.

## Vanilla RNN: базовая архитектура

### Формальное определение

На каждом временном шаге $t$ RNN получает вход $x_t$ и предыдущее скрытое состояние $h_{t-1}$, и вычисляет новое скрытое состояние:

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

$$y_t = W_{hy} h_t + b_y$$

где:
- $x_t \in \mathbb{R}^d$ --- входной вектор (эмбеддинг токена)
- $h_t \in \mathbb{R}^n$ --- скрытое состояние (hidden state), «память» сети
- $y_t$ --- выход на шаге $t$
- $W_{hh}, W_{xh}, W_{hy}$ --- **одни и те же** матрицы на каждом шаге (parameter sharing)

### Интуиция

Представь, что ты читаешь предложение слово за словом. После каждого слова ты обновляешь своё «понимание» предложения. Скрытое состояние $h_t$ --- это текущее «понимание», которое зависит от всех предыдущих слов.

### Режимы использования

| Режим | Вход -> Выход | Пример |
|-------|:------------:|--------|
| **One-to-many** | 1 -> $T$ | Image captioning: изображение -> описание |
| **Many-to-one** | $T$ -> 1 | Sentiment analysis: текст -> метка |
| **Many-to-many** | $T$ -> $T$ | POS-tagging: каждое слово -> тег |
| **Seq2Seq** | $T_1$ -> $T_2$ | Translation: предложение -> перевод |

## Проблема затухающих градиентов (Vanishing Gradients)

Главная беда vanilla RNN --- **gradient vanishing** при обучении на длинных последовательностях.

При обратном распространении через время (BPTT) градиент проходит через произведение якобианов:

$$\frac{\partial h_T}{\partial h_1} = \prod_{t=2}^{T} \frac{\partial h_t}{\partial h_{t-1}} = \prod_{t=2}^{T} W_{hh}^T \cdot \text{diag}(f'(h_{t-1}))$$

Если наибольшее сингулярное число $W_{hh}$ < 1, градиент **экспоненциально затухает** с ростом $T$. Если > 1 --- **экспоненциально взрывается**.

**Практические последствия:**
- Vanilla RNN не может выучить зависимости длиннее 10--20 шагов
- Предложение "The cat, which was sitting on the mat near the window, **was** sleeping" --- RNN не может связать «cat» и «was» через 10 слов

Решения:
- **Gradient clipping** --- обрезать норму градиента (помогает от взрыва, но не от затухания)
- **LSTM и GRU** --- архитектурное решение проблемы затухания

## LSTM: Long Short-Term Memory

LSTM (Hochreiter & Schmidhuber, 1997) --- главный прорыв в рекуррентных сетях. Ключевая идея: добавить **cell state** $c_t$ --- «магистраль», по которой информация может течь без помех, и три **гейта** (gates), которые контролируют поток информации.

### Формулы

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \quad \text{(forget gate)}$$
$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i) \quad \text{(input gate)}$$
$$\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c) \quad \text{(candidate)}$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \quad \text{(cell state update)}$$
$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o) \quad \text{(output gate)}$$
$$h_t = o_t \odot \tanh(c_t) \quad \text{(hidden state)}$$

### Интуиция по гейтам (Christopher Olah)

| Гейт | Что делает | Аналогия |
|------|-----------|----------|
| **Forget gate** ($f_t$) | Решает, что забыть из cell state | «Началось новое предложение --- забудь пол предыдущего субъекта» |
| **Input gate** ($i_t$) | Решает, что записать в cell state | «Новый субъект --- запомни его число» |
| **Output gate** ($o_t$) | Решает, что выдать наружу | «Сейчас нужен глагол --- покажи грамматическую информацию» |

### Почему LSTM решает vanishing gradient

Cell state $c_t$ обновляется через **сложение** (не умножение): $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$. Когда $f_t \approx 1$ и $i_t \approx 0$, информация проходит через сотни шагов почти без потерь --- как residual connection в ResNet.

## GRU: Gated Recurrent Unit

GRU (Cho et al., 2014) --- упрощённая версия LSTM с **двумя гейтами** вместо трёх:

$$z_t = \sigma(W_z [h_{t-1}, x_t]) \quad \text{(update gate)}$$
$$r_t = \sigma(W_r [h_{t-1}, x_t]) \quad \text{(reset gate)}$$
$$\tilde{h}_t = \tanh(W [\, r_t \odot h_{t-1},\, x_t\,])$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

### GRU vs LSTM

| Свойство | LSTM | GRU |
|----------|:----:|:---:|
| Гейтов | 3 (forget, input, output) | 2 (update, reset) |
| Отдельный cell state | Да ($c_t$) | Нет |
| Параметров | Больше (~4x hidden) | Меньше (~3x hidden) |
| Производительность | Чуть лучше на длинных зависимостях | Сравнимо или чуть хуже |
| Скорость обучения | Медленнее | Быстрее |

На практике: LSTM и GRU дают **сравнимое качество** на большинстве задач (Chung et al., 2014). GRU проще и быстрее, LSTM --- «безопаснее» для длинных последовательностей.

## Bidirectional RNN

Vanilla RNN / LSTM / GRU обрабатывают последовательность слева направо. Но для задач понимания (не генерации) важен **контекст с обеих сторон**.

Bidirectional RNN запускает **две RNN** --- forward ($\overrightarrow{h_t}$) и backward ($\overleftarrow{h_t}$) --- и конкатенирует их выходы:

$$h_t = [\overrightarrow{h_t}; \overleftarrow{h_t}]$$

Результат: каждая позиция видит и прошлое, и будущее. BiLSTM стал стандартом для:
- Named Entity Recognition (NER)
- Part-of-Speech Tagging (POS)
- Sentiment Analysis
- Reading Comprehension

BiLSTM + CRF (Lample et al., 2016) --- SOTA по NER до BERT.

## Seq2Seq: encoder-decoder на RNN

Seq2Seq (Sutskever et al., 2014) --- архитектура для задач, где длина входа и выхода различны (машинный перевод, summarization):

**Encoder:** RNN (обычно BiLSTM) читает входную последовательность, сжимает её в **один вектор фиксированной длины** --- последнее скрытое состояние $h_T$.

**Decoder:** другая RNN получает этот вектор как начальное состояние и генерирует выходную последовательность авторегрессивно (каждый шаг зависит от предыдущего выхода).

### Проблема бутылочного горлышка

Вся информация о входной последовательности сжимается в один вектор $h_T$ (обычно 512 или 1024 измерений). Для предложения из 50 слов это огромная потеря информации.

Эта проблема мотивировала создание [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]] (Bahdanau et al., 2014): вместо одного вектора декодер получает доступ ко **всем** скрытым состояниям энкодера и на каждом шаге решает, куда «смотреть».

## Почему Transformer заменил RNN

Таблица 1 из [[02 Areas/ML & DL/Papers/Attention Is All You Need|Attention Is All You Need]] --- главный аргумент:

| Свойство | RNN | Self-Attention | Преимущество |
|----------|:---:|:--------------:|:------------:|
| Сложность на слой | $O(n \cdot d^2)$ | $O(n^2 \cdot d)$ | RNN при $n < d$ |
| Последовательные операции | $O(n)$ | $O(1)$ | **Self-Attention** |
| Макс. длина пути | $O(n)$ | $O(1)$ | **Self-Attention** |
| Параллелизм | Нет | Полный | **Self-Attention** |

**Два решающих фактора:**

1. **$O(1)$ максимальная длина пути:** в RNN сигнал между позицией 1 и позицией 100 должен пройти 99 шагов, затухая на каждом. В self-attention --- одно прямое вычисление.

2. **Параллелизм:** $h_t$ в RNN зависит от $h_{t-1}$ --- нельзя вычислить шаг $t$, пока не закончен шаг $t-1$. Self-attention вычисляет все позиции параллельно, что критично для обучения на GPU/TPU кластерах.

## Хронология: от RNN до Transformer

| Год | Событие | Значение |
|-----|---------|----------|
| 1986 | RNN (Rumelhart, Hinton, Williams) | Backpropagation Through Time |
| **1997** | **LSTM** (Hochreiter & Schmidhuber) | Решение vanishing gradients |
| 2013 | Word2Vec (Mikolov et al.) | Плотные эмбеддинги для RNN |
| **2014** | **Seq2Seq** (Sutskever et al.) | Encoder-decoder для translation |
| 2014 | GRU (Cho et al.) | Упрощённый LSTM |
| 2014 | Attention (Bahdanau et al.) | Решение bottleneck в seq2seq |
| 2015 | BiLSTM + Attention | SOTA по NMT |
| 2016 | Google NMT (Wu et al.) | 8-layer stacked LSTM + attention |
| **2017** | **Transformer** (Vaswani et al.) | **Attention Is All You Need --- конец эпохи RNN** |
| 2018 | ELMo (Peters et al.) | Последний major success BiLSTM |

## RNN в современном мире

Хотя Transformer доминирует, RNN-идеи не умерли полностью:

- **[[02 Areas/ML & DL/Concepts/Architectures/RWKV|RWKV]]** (2023) --- hybrid: RNN-like inference (линейная сложность) + Transformer-like training (параллелизм)
- **[[02 Areas/ML & DL/Concepts/Architectures/Mamba|Mamba]]** (2023) --- State Space Model с селективным scan, по сути «модернизированная RNN» с $O(n)$ сложностью
- **xLSTM** (Hochreiter et al., 2024) --- возвращение LSTM с exponential gating и matrix memory

Эти архитектуры пытаются совместить лучшее от обоих миров: линейную сложность RNN по длине последовательности + качество Transformer на длинном контексте.

## Related concepts

- [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]] --- создан для решения bottleneck в seq2seq RNN
- [[02 Areas/ML & DL/Concepts/NLP/Self-Attention|Self-Attention]] --- заменил RNN в Transformer
- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] --- архитектура-преемник
- [[02 Areas/ML & DL/Concepts/Architectures/Mamba|Mamba]] --- SSM как современная альтернатива и RNN, и Transformer
- [[02 Areas/ML & DL/Concepts/Architectures/RWKV|RWKV]] --- RNN + Transformer hybrid

## Дополнительные ресурсы

- [Andrej Karpathy — The Unreasonable Effectiveness of RNNs](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) --- лучшее введение с примерами генерации
- [Christopher Olah — Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) --- визуальное объяснение гейтов LSTM
- [d2l.ai — Recurrent Neural Networks](https://d2l.ai/chapter_recurrent-neural-networks/) --- учебник с кодом
- [Lena Voita — Seq2Seq and Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html) --- контекст перехода от RNN к attention
