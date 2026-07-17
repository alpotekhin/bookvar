---
title: "RNN"
aliases: [RNN, Recurrent Neural Network, рекуррентная сеть]
type: concept
status: legacy
category: Architectures
papers: []
courses:
  - "[[02 Areas/ML & DL/Courses/Stanford CS224N/Lecture 05 — RNNs|CS224N L05]]"
  - "[[02 Areas/ML & DL/Courses/Stanford CS224N/Lecture 06 — Seq2Seq|CS224N L06]]"
sources:
  - "[Elman — Finding Structure in Time (1990)](https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1402_1)"
  - "[Karpathy — The Unreasonable Effectiveness of RNNs (2015)](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)"
  - "[Goodfellow et al. — Deep Learning, Chapter 10](https://www.deeplearningbook.org/contents/rnn.html)"
---

# RNN — Recurrent Neural Network

## Зачем это нужно: последовательности произвольной длины

Обычный MLP принимает вход фиксированной размерности. Но язык, речь, временные ряды — это последовательности **произвольной длины**, где порядок важен. RNN (Elman, 1990) предлагает решение: обрабатывать последовательность шаг за шагом, храня **скрытое состояние** $h_t$, которое суммирует всё увиденное ранее.

Ключевое свойство RNN — **разделение весов по времени** (weight sharing across time). Одна и та же матрица $W_{hh}$ применяется на каждом шаге, поэтому сеть обобщается на последовательности любой длины.

## Базовая формула

Скрытое состояние обновляется рекуррентно:

$$h_t = f(h_{t-1}, x_t) = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

Выход на шаге $t$:

$$y_t = W_{hy} h_t + b_y$$

где:
- $x_t \in \mathbb{R}^{d_x}$ — вход на шаге $t$ (например, эмбеддинг токена)
- $h_t \in \mathbb{R}^{d_h}$ — скрытое состояние (память сети)
- $W_{hh}, W_{xh}, W_{hy}$ — обучаемые матрицы, **общие для всех шагов**
- $f$ — нелинейность, обычно $\tanh$ (реже ReLU)

Начальное состояние $h_0$ обычно — нулевой вектор или обучаемый параметр.

### Визуализация развёртки

```
x_1    x_2    x_3    ...    x_T
 │      │      │             │
 ▼      ▼      ▼             ▼
h_0 ─► h_1 ─► h_2 ─► h_3 ... h_T
        │      │      │       │
        ▼      ▼      ▼       ▼
       y_1    y_2    y_3     y_T
```

Одна и та же ячейка RNN применяется $T$ раз. Это называется **unrolling** — развёртка по времени.

## Варианты использования

| Тип | Описание | Примеры задач |
|-----|----------|---------------|
| one-to-many | Один вход → последовательность | Image captioning |
| many-to-one | Последовательность → один выход | Sentiment classification |
| many-to-many (aligned) | Последовательность той же длины | POS-tagging, NER |
| many-to-many (seq2seq) | Последовательность → другая длина | Machine translation |

Для seq2seq задач используется архитектура **encoder-decoder**: один RNN кодирует вход в вектор, другой декодирует его в выходную последовательность.

## Backpropagation Through Time (BPTT)

Обучение RNN требует вычисления градиентов через всю развёрнутую сеть. Этот алгоритм называется **Backpropagation Through Time** (BPTT, Werbos 1990):

1. **Forward pass:** пройти последовательность, вычислить все $h_1, \ldots, h_T$ и loss $L = \sum_t L_t$
2. **Backward pass:** распространить градиенты назад по времени:

$$\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W_{hh}}$$

При этом градиент по $W_{hh}$ накапливается со всех временных шагов, а цепное правило проходит через все промежуточные $h_k$:

$$\frac{\partial L_t}{\partial h_k} = \frac{\partial L_t}{\partial h_t} \prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}}$$

### Truncated BPTT

Для длинных последовательностей (тысячи шагов) полный BPTT непомерно дорог по памяти — нужно хранить все $h_t$. **Truncated BPTT** разбивает последовательность на куски длины $k$ (обычно 32-128) и делает backward только внутри куска, сохраняя $h_t$ между кусками без градиентов. Стандартный приём в языковом моделировании.

## Vanishing и exploding gradients

Главная проблема vanilla RNN — **нестабильность градиентов** при длинных последовательностях. Произведение якобианов:

$$\prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}} = \prod_{j=k+1}^{t} W_{hh}^T \cdot \text{diag}(\tanh'(\cdot))$$

Если наибольшее сингулярное значение $W_{hh}$ меньше 1 — произведение **экспоненциально затухает** (vanishing gradients): сеть не может выучить зависимости длиннее ~10-20 шагов. Если больше 1 — **взрывается** (exploding gradients): обучение расходится.

### Решения

**Exploding gradients** — gradient clipping:

$$g \leftarrow g \cdot \frac{\text{threshold}}{\|g\|} \quad \text{если} \quad \|g\| > \text{threshold}$$

Простой и эффективный приём, введённый Pascanu et al. (2013).

**Vanishing gradients** — архитектурное решение: [[02 Areas/ML & DL/Concepts/Architectures/LSTM|LSTM]] (Hochreiter & Schmidhuber, 1997) и GRU (Cho et al., 2014) вводят **gating mechanism** и **cell state**, через который градиенты текут без затухания. Именно LSTM сделал RNN практически применимым для длинных последовательностей.

## Bidirectional RNN (BiRNN)

Обычный RNN видит только **левый контекст** — $h_t$ зависит от $x_1, \ldots, x_t$. Для задач, где важен весь контекст (NER, POS-tagging, классификация), используется **bidirectional RNN**: два RNN, идущих в противоположных направлениях:

$$\overrightarrow{h}_t = \text{RNN}_{forward}(x_t, \overrightarrow{h}_{t-1})$$
$$\overleftarrow{h}_t = \text{RNN}_{backward}(x_t, \overleftarrow{h}_{t+1})$$
$$h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t]$$

Конкатенация даёт вектор размерности $2d_h$, видящий и прошлое, и будущее. **BiLSTM + CRF** — стандартная архитектура для sequence labeling до эпохи BERT.

Важное ограничение: BiRNN нельзя использовать для language modeling или autoregressive generation — там будущее недоступно.

## Stacked (Deep) RNN

Несколько слоёв RNN друг над другом: выход $h_t$ нижнего слоя — вход $x_t$ верхнего:

$$h_t^{(l)} = \text{RNN}^{(l)}(h_t^{(l-1)}, h_{t-1}^{(l)})$$

Типично 2-4 слоя. Нижние слои выучивают локальные паттерны, верхние — более абстрактные. Dropout применяется только между слоями, не по времени (Zaremba et al., 2014), чтобы не ломать рекуррентные связи.

## Применение: «золотая эра» RNN (2014-2017)

До Transformer RNN (в виде LSTM/GRU) был SOTA почти во всех задачах NLP:

- **Machine Translation** — Seq2Seq с attention (Bahdanau 2014, Google NMT 2016)
- **Language Modeling** — 2-layer LSTM + dropout (Zaremba 2014)
- **Speech Recognition** — Deep BiLSTM + CTC
- **Text Generation** — Karpathy's char-RNN (2015)
- **Image Captioning** — CNN + LSTM decoder (Show and Tell, 2015)

## Почему Transformer заменил RNN

### Нет параллелизации по времени

RNN вычисляет $h_t$ **последовательно** — каждый шаг требует завершения предыдущего. Для последовательности длины $T$ — $T$ последовательных операций. На GPU это неэффективно: GPU оптимизирован под параллельные вычисления.

[[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] вычисляет [[02 Areas/ML & DL/Concepts/NLP/Self-Attention|self-attention]] для всех позиций **одновременно** — $O(1)$ последовательных шагов, $O(T^2)$ compute, но полностью параллельно по времени.

### Path length

В RNN информация от позиции 1 до позиции 1000 проходит 999 шагов, затухая. В Transformer любые две позиции связаны напрямую через attention — path length $O(1)$.

### Scaling laws

Transformer масштабируется с ростом compute (больше параметров, больше данных → лучше качество). RNN — нет: увеличение ширины/глубины даёт diminishing returns.

| Свойство | RNN | Transformer |
|----------|-----|-------------|
| Sequential ops | $O(T)$ | $O(1)$ |
| Path length | $O(T)$ | $O(1)$ |
| Compute per layer | $O(T \cdot d^2)$ | $O(T^2 \cdot d)$ |
| Memory | $O(d)$ на инференсе | $O(T \cdot d)$ (KV cache) |
| Параллелизация | Нет (по времени) | Полная |

## Наследие RNN

Несмотря на доминирование Transformer, рекуррентная парадигма возвращается:

- **Linear RNN / State Space Models** — [[02 Areas/ML & DL/Concepts/Architectures/Mamba|Mamba]] (2023), [[02 Areas/ML & DL/Concepts/Architectures/RWKV|RWKV]], [[02 Areas/ML & DL/Concepts/Architectures/RetNet|RetNet]]: возврат к $O(T)$ compute, но с hardware-aware параллелизацией
- **Gating mechanism** из LSTM вдохновил GLU/SwiGLU в современных LLM
- **Recurrent memory** для длинных контекстов — Transformer-XL, Memformer

RNN — важная часть истории NLP и базовая модель последовательностей, которую полезно понимать для работы с любыми sequence models.

## Related concepts

- [[02 Areas/ML & DL/Concepts/Architectures/LSTM|LSTM]] — решение vanishing gradients через cell state и gates
- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] — архитектура, заменившая RNN в NLP
- [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]] — впервые применён поверх RNN (Bahdanau 2014)
- [[02 Areas/ML & DL/Concepts/Training/Backpropagation|Backpropagation]] — BPTT как частный случай
- [[02 Areas/ML & DL/Concepts/Architectures/Mamba|Mamba]] — современное развитие рекуррентной парадигмы
