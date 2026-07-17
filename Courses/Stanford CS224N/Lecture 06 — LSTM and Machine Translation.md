---
title: "CS224N — Lecture 6: LSTM RNNs and Neural Machine Translation"
course: "Stanford CS224N"
lecture: 6
type: source-note
status: legacy
raw: "[[02 Areas/ML & DL/raw/courses/Stanford CS224N/slides/cs224n-spr2024-lecture06-fancy-rnn]]"
concepts: ["[[02 Areas/ML & DL/Concepts/Architectures/LSTM|LSTM]]", "[[02 Areas/ML & DL/Concepts/Training/Vanishing Gradient|Vanishing Gradient]]", "[[02 Areas/ML & DL/Concepts/NLP/Machine Translation|Machine Translation]]", "[[02 Areas/ML & DL/Concepts/NLP/RNN/Bidirectional RNN|Bidirectional RNN]]", "[[02 Areas/ML & DL/Concepts/Training/Gradient Clipping|Gradient Clipping]]", "[[02 Areas/ML & DL/Concepts/NLP/Seq2Seq|Seq2Seq]]"]
---

# Lecture 6: LSTM RNNs and Neural Machine Translation

> *"Could we design an RNN with separate memory which is added to?"* -- Christopher Manning

Лектор: Christopher Manning.

## Recap: Language Models и RNN

**Language Model** -- система, предсказывающая следующее слово. **RNN** -- семейство нейросетей, обрабатывающих последовательности произвольной длины с общими весами на каждом шаге.

### Perplexity

Стандартная метрика для LM:

$$\text{Perplexity} = \prod_{t=1}^{T} \left(\frac{1}{P_{LM}(x^{(t+1)} \mid x^{(t)}, \ldots, x^{(1)})}\right)^{1/T} = \exp\left(\frac{1}{T}\sum_{t=1}^{T} -\log P_{LM}(x^{(t+1)} \mid \ldots)\right)$$

Равна экспоненте cross-entropy loss. **Меньше -- лучше**.

## [[02 Areas/ML & DL/Concepts/Training/Vanishing Gradient|Vanishing Gradient]]

### Интуиция

При backprop через $T$ шагов RNN, градиент функции потерь $J$ на шаге $T$ по hidden state на шаге $t$:

$$\frac{\partial J^{(T)}}{\partial h^{(t)}} = \frac{\partial J^{(T)}}{\partial h^{(T)}} \prod_{j=t+1}^{T} \frac{\partial h^{(j)}}{\partial h^{(j-1)}}$$

Каждый множитель $\frac{\partial h^{(j)}}{\partial h^{(j-1)}}$ зависит от $W_h$. Это **chain rule** по всем промежуточным шагам.

### Доказательство (линейный случай)

Для упрощения: если $f$ -- identity, то $h^{(t)} = W_h h^{(t-1)}$. Тогда:

$$\prod_{j=t+1}^{T} W_h = W_h^{T-t}$$

Через собственные разложения: $W_h^{T-t} = Q \Lambda^{T-t} Q^{-1}$. Если **собственные значения $|\lambda_i| < 1$**: $\Lambda^{T-t} \to 0$ при большом $T-t$ -- **vanishing gradient**. Если $|\lambda_i| > 1$: **exploding gradient**.

Для нелинейного случая: тот же эффект, с поправкой на $f'$.

### Практические последствия

**Vanishing gradient**: модель обучается только на **ближних** зависимостях (~7 токенов). Gradient signal от далёких позиций "тонет" в сигнале от ближних.

Пример: *"When she tried to print her tickets, she found that the printer was out of toner. She went to the stationery store to buy more toner... she finally printed her ________"*. RNN не может выучить зависимость между "tickets" (7й шаг) и target word в конце.

### [[02 Areas/ML & DL/Concepts/Training/Gradient Clipping|Gradient Clipping]]: решение exploding gradient

Если $\|g\| > \text{threshold}$:

$$g \leftarrow \frac{\text{threshold}}{\|g\|} \cdot g$$

Интуиция: шагаем в том же направлении, но меньший шаг. Exploding gradient -- **легко решаемая** проблема. Vanishing gradient -- фундаментальная.

## [[02 Areas/ML & DL/Concepts/Architectures/LSTM|LSTM]] (Hochreiter & Schmidhuber, 1997)

### Ключевая идея

Vanilla RNN: hidden state **перезаписывается** на каждом шаге. LSTM: отдельная **cell memory** $c_t$, обновляемая **аддитивно** (не перезаписывается).

### Компоненты LSTM

На шаге $t$: hidden state $h_t$ и cell state $c_t$ (оба вектора длины $n$):

**Forget gate** -- что забыть из прошлого cell state:

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$$

**Input gate** -- какие части нового контента записать:

$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$$

**New cell content** -- кандидат для записи:

$$\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)$$

**Cell state update** -- аддитивное обновление:

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

**Output gate** -- что из cell state передать в hidden state:

$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$$

**Hidden state**:

$$h_t = o_t \odot \tanh(c_t)$$

Все gates используют **sigmoid** (значения в $[0, 1]$): open (1), closed (0), или промежуточное.

### Почему LSTM решает vanishing gradient

- **Аддитивное обновление** $c_t$: если $f_t = 1$ и $i_t = 0$, информация в cell **сохраняется бесконечно**
- Это **gradient highway**: градиент проходит через $c_t$ напрямую, без умножения на $W_h$
- На практике: ~100 timesteps (вместо ~7 для vanilla RNN)

### GRU (Gated Recurrent Unit, Cho et al. 2014)

Упрощённая версия LSTM: объединяет forget и input gates в один **update gate**. Меньше параметров, сопоставимая производительность. На практике LSTM чаще используется из-за исторического преимущества.

### Историческая справка

- 1997: Hochreiter & Schmidhuber -- оригинальная публикация
- 2000: Gers et al. -- добавление forget gate (критическая часть modern LSTM!)
- ~2006: Alex Graves (студент Schmidhuber) -- LSTM + CTC для speech recognition
- 2013: Hinton привёл LSTM в Google -- начало массового использования

## Vanishing gradient -- не только проблема RNN

Это проблема **всех** глубоких архитектур (feedforward, convolutional). Решения:

| Метод | Идея |
|-------|------|
| **Residual connections (ResNet)** | $y = f(x) + x$ -- identity connection сохраняет информацию по умолчанию |
| **Dense connections (DenseNet)** | Каждый слой напрямую связан со всеми последующими |
| **Highway connections** | Динамический gate контролирует identity vs transformation (вдохновлено LSTM) |

Но RNN **особенно** уязвимы из-за **повторного умножения на одну и ту же матрицу** $W_h$ (Bengio et al., 1994).

## [[02 Areas/ML & DL/Concepts/NLP/RNN/Bidirectional RNN|Bidirectional RNN]]

### Мотивация

Однонаправленный RNN для "terribly" в *"the movie was terribly exciting!"* видит только **левый контекст** ("the movie was") -- "terribly" кажется негативным. Правый контекст ("exciting") меняет смысл на позитивный.

### Архитектура

Два отдельных RNN с **разными весами**:

$$\overrightarrow{h}_t = \text{RNN}_{fw}(\overrightarrow{h}_{t-1}, x_t)$$
$$\overleftarrow{h}_t = \text{RNN}_{bw}(\overleftarrow{h}_{t+1}, x_t)$$

Concatenated hidden state: $h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t]$ -- контекст **с обеих сторон**.

**Ограничение**: bidirectional RNN требует **доступа ко всей последовательности**. Применим для encoding, **не** для language modeling (генерации).

## Multi-layer RNNs

RNN уже "deep" по временнОй оси. Multi-layer (stacked) RNNs добавляют глубину по **вертикальной** оси:

- Нижние слои: low-level features (морфология, синтаксис)
- Верхние слои: high-level features (семантика)

Для RNN обычно оптимально **2-4 слоя**. Для [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] -- значительно больше (12-96).

## [[02 Areas/ML & DL/Concepts/NLP/Machine Translation|Neural Machine Translation]]

### Encoder-Decoder (Seq2Seq)

Архитектура Sutskever et al. 2014:

```
Encoder RNN:  Die Proteste waren am Wochenende eskaliert <EOS>
                ↓          ↓          ↓          ↓
              [h1]  →   [h2]  →   [h3]  →    [h_final]
                                                  ↓
Decoder RNN:                              <START> → The → protests → escalated → ...
```

**Encoder**: кодирует исходное предложение. Последний hidden state -- "meaning" входа.

**Decoder**: RNN-LM, инициализированный hidden state энкодера. Генерирует перевод по одному слову.

**Bottleneck**: всё предложение сжимается в **один вектор** $h_{final}$. Для длинных предложений это серьёзное ограничение -- мотивация для [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention]] (следующая лекция).

## Concepts covered

- [[02 Areas/ML & DL/Concepts/Architectures/LSTM|LSTM]] -- Long Short-Term Memory, аддитивное обновление cell state
- [[02 Areas/ML & DL/Concepts/Training/Vanishing Gradient|Vanishing Gradient]] -- экспоненциальное затухание градиента
- [[02 Areas/ML & DL/Concepts/Training/Gradient Clipping|Gradient Clipping]] -- решение exploding gradient через масштабирование нормы
- [[02 Areas/ML & DL/Concepts/NLP/RNN/Bidirectional RNN|Bidirectional RNN]] -- двунаправленная RNN для encoding
- [[02 Areas/ML & DL/Concepts/NLP/Seq2Seq|Seq2Seq]] -- encoder-decoder для machine translation
- [[02 Areas/ML & DL/Concepts/NLP/Machine Translation|Machine Translation]] -- от phrase-based SMT к neural MT
