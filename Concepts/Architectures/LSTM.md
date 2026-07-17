---
title: "LSTM"
aliases: [LSTM, Long Short-Term Memory]
type: concept
category: Architectures
papers: []
courses:
  - "[[02 Areas/ML & DL/Courses/Stanford CS224N/Lecture 05 — RNNs|CS224N L05]]"
  - "[[02 Areas/ML & DL/Courses/Stanford CS224N/Lecture 06 — Seq2Seq|CS224N L06]]"
sources:
  - "[Hochreiter & Schmidhuber — Long Short-Term Memory (1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)"
  - "[Colah — Understanding LSTM Networks (2015)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)"
---

# LSTM — Long Short-Term Memory

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/lstm/lstm-chain.png]]
*Цепочка LSTM-ячеек: cell state (верхняя линия) проходит через всю последовательность с минимальными трансформациями, три гейта управляют потоком информации (источник: Colah, 2015)*

## Зачем это нужно: проблема затухающих градиентов

Vanilla RNN обрабатывает последовательности рекуррентно:

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$$

При backpropagation through time (BPTT) градиенты проходят через цепочку умножений на $W_{hh}$:

$$\frac{\partial L}{\partial h_t} = \frac{\partial L}{\partial h_T} \prod_{k=t+1}^{T} \frac{\partial h_k}{\partial h_{k-1}}$$

Если собственные значения $W_{hh}$ меньше 1 — градиенты **экспоненциально затухают** (vanishing gradients). Если больше 1 — **взрываются** (exploding gradients). На практике vanilla RNN не может выучить зависимости длиннее ~10-20 шагов.

**Exploding gradients** решаются gradient clipping: $g \leftarrow \frac{g}{\|g\|} \cdot \text{threshold}$ если $\|g\| > \text{threshold}$.

**Vanishing gradients** — фундаментальная проблема, которую решает LSTM (Hochreiter & Schmidhuber, 1997).

## Архитектура LSTM

Ключевая идея: добавить **cell state** $c_t$ — отдельный путь для информации, проходящий через всю последовательность с минимальными нелинейностями. Три **gate** (вентиля) контролируют, что записать, что забыть и что выдать.

### Forget gate — что забыть

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

Сигмоида выдаёт значения в $[0, 1]$ для каждого элемента cell state. Значение 0 = "полностью забыть", 1 = "полностью сохранить".

### Input gate — что записать

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

$i_t$ решает, какие позиции обновить. $\tilde{c}_t$ — кандидат на новые значения.

### Обновление cell state

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

Это **ключевая формула LSTM**. Cell state обновляется аддитивно — нет умножения на общую матрицу, градиенты текут по "шоссе" (highway) через всю последовательность. Forget gate решает, что удалить из старой памяти, input gate — что добавить.

### Output gate — что выдать

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(c_t)$$

Output gate фильтрует cell state, создавая hidden state $h_t$, который используется как выход LSTM и передаётся на следующий шаг.

### Визуализация потока данных

```
                    c_{t-1} ──────[×]────────[+]──────── c_t
                               ↑   forget     ↑  input
                               |               |
                    ┌──────────┤               ├──────────┐
                    │  f_t = σ(...)      i_t ⊙ c̃_t       │
                    │                                      │
     x_t ──┬───────┤                                      ├──── h_t
            │       │         o_t = σ(...)                 │
     h_{t-1}┘       └────────────┤                        │
                                 └───── h_t = o_t ⊙ tanh(c_t)
```

## Почему LSTM решает vanishing gradients

При backpropagation градиент по cell state:

$$\frac{\partial c_t}{\partial c_{t-1}} = f_t$$

Если forget gate $f_t \approx 1$ (что сеть легко выучивает), градиент проходит **без изменений**. Это прямой аналог residual connections, появившихся позже в ResNet (2015) и Transformer (2017).

**Важно:** LSTM не решает vanishing gradients полностью — при очень длинных последовательностях (>500 шагов) проблема возвращается. Но на практике LSTM работает для последовательностей в сотни шагов, что было прорывом по сравнению с vanilla RNN.

## GRU — упрощённый вариант

**GRU** (Gated Recurrent Unit, Cho et al., 2014) — вариант с **2 вентилями** вместо 3:

$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$$
$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$$
$$\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t])$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

**Отличия от LSTM:**
- Нет отдельного cell state — только hidden state
- Update gate $z_t$ совмещает функции forget и input gates
- Reset gate $r_t$ контролирует, сколько прошлого состояния использовать для кандидата
- Меньше параметров (~75% от LSTM)

| Свойство | LSTM | GRU |
|----------|------|-----|
| Gates | 3 (forget, input, output) | 2 (update, reset) |
| Состояния | $c_t$ (cell) + $h_t$ (hidden) | Только $h_t$ |
| Параметры (на ячейку) | $4 \times (d_h^2 + d_h \cdot d_x + d_h)$ | $3 \times (d_h^2 + d_h \cdot d_x + d_h)$ |
| Качество | Чуть лучше на длинных seq | Сопоставимо на коротких |
| Скорость | Медленнее | Быстрее |

На практике разница в качестве между LSTM и GRU минимальна. GRU предпочтительнее, когда compute ограничен.

## Bidirectional LSTM (BiLSTM)

Обычный LSTM обрабатывает последовательность **слева направо** — каждый $h_t$ видит только прошлый контекст. Для задач, где важен весь контекст (NER, classification), используется BiLSTM:

$$\overrightarrow{h}_t = \text{LSTM}_{forward}(x_t, \overrightarrow{h}_{t-1})$$
$$\overleftarrow{h}_t = \text{LSTM}_{backward}(x_t, \overleftarrow{h}_{t+1})$$
$$h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t]$$

Конкатенация двух направлений даёт вектор размерности $2d_h$, учитывающий и левый, и правый контекст.

**BiLSTM + CRF** — стандартная архитектура для sequence labeling (NER, POS-tagging) до BERT (2018).

## Stacked LSTM (Deep LSTM)

Несколько слоёв LSTM, где выход $h_t$ одного слоя — вход $x_t$ следующего:

$$h_t^{(l)} = \text{LSTM}^{(l)}(h_t^{(l-1)}, h_{t-1}^{(l)})$$

Типично 2-4 слоя. Нижние слои выучивают low-level features (синтаксис), верхние — high-level (семантика). Dropout между слоями для регуляризации.

## LSTM в NLP: золотая эра (2014-2018)

До появления Transformer (2017) и BERT (2018), LSTM был **SOTA архитектурой** почти для всех задач NLP:

| Задача | Архитектура | Год SOTA |
|--------|------------|----------|
| Machine Translation | Seq2Seq (2-layer LSTM + attention) | 2014-2017 |
| Language Modeling | 2-layer LSTM + dropout | 2016-2018 |
| NER | BiLSTM + CRF | 2015-2018 |
| Sentiment Analysis | BiLSTM + attention | 2015-2018 |
| Seq2Seq (summarization, QA) | Encoder-Decoder LSTM | 2015-2017 |
| Speech Recognition | Deep BiLSTM | 2014-2018 |

**Знаковые системы на LSTM:**
- **Google Neural Machine Translation** (GNMT, 2016) — 8-layer LSTM, перевёл Google Translate на нейросети
- **ELMo** (2018) — BiLSTM language model, предшественник BERT

## Почему Transformer заменил LSTM

### Параллелизация

LSTM вычисляет $h_t$ **последовательно** — $h_t$ зависит от $h_{t-1}$, который зависит от $h_{t-2}$, и т.д. Для последовательности длины $T$ нужно $T$ последовательных шагов. **Нельзя параллелизовать по времени.**

Transformer вычисляет attention для всех позиций **одновременно** — $O(1)$ sequential operations (при $O(T^2)$ compute). На GPU это даёт колоссальное ускорение.

### Дальнодействие

LSTM передаёт информацию через цепочку состояний: от позиции 1 к позиции 1000 информация проходит 999 шагов, затухая. В Transformer **любые две позиции связаны напрямую** через attention — path length $O(1)$.

### Масштабируемость

Transformers масштабируются с ростом compute (scaling laws), LSTM — нет. Увеличение LSTM (ширина, глубина) даёт diminishing returns.

| Свойство | LSTM | Transformer |
|----------|------|-------------|
| Sequential operations | $O(T)$ | $O(1)$ |
| Path length | $O(T)$ | $O(1)$ |
| Compute per layer | $O(T \cdot d^2)$ | $O(T^2 \cdot d)$ |
| Параллелизация | Нет (по времени) | Полная |
| Scaling laws | Слабые | Сильные |

## Наследие LSTM

Несмотря на доминирование Transformers, идеи LSTM живут:

- **Gating mechanism** — gates в LSTM вдохновили gating в Transformers (GLU, SwiGLU в LLaMA/Mistral)
- **Cell state highway** — прямой предшественник residual connections
- **[[02 Areas/ML & DL/Concepts/Architectures/Mamba|Mamba]]** (2023) — selective state space model, возврат к sequential processing с $O(T)$ compute, но с hardware-aware параллелизацией. Решает проблему $O(T^2)$ attention для длинных контекстов

LSTM остаётся хорошим выбором для задач с ограниченным compute, коротких последовательностей и edge deployment.

## Related concepts

- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] — архитектура, заменившая LSTM в NLP
- [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]] — механизм, впервые применённый с LSTM (Bahdanau, 2014), ставший основой Transformer
- [[02 Areas/ML & DL/Concepts/Architectures/Mamba|Mamba]] — современный "наследник" рекуррентного подхода через state space models
- [[02 Areas/ML & DL/Concepts/Architectures/BERT|BERT]] — Transformer-модель, заменившая BiLSTM в downstream NLP задачах
