---
title: "MIT 6.S191 — Sequence Modeling"
type: course-note
course: "MIT 6.S191"
---

# MIT 6.S191 — Deep Sequence Modeling

> Лекция 2. RNN, LSTM, внимание и Transformers — как моделировать последовательности.

**Курс:** [[MIT 6.S191/_index|MIT 6.S191]]
**Лекторы:** Alexander Amini, Ava Amini
**Связанные концепты:** [[RNN]], [[Attention Mechanism]], [[Self-Attention]], [[Positional Encoding]]

---

## Зачем моделировать последовательности

Огромная часть данных в мире — последовательности:
- **Текст** — последовательность слов/токенов
- **Аудио** — последовательность звуковых фреймов
- **Временные ряды** — цены, метрики, сенсоры
- **Видео** — последовательность кадров
- **Код** — последовательность токенов

Ключевое свойство: **порядок имеет значение**. "Собака укусила человека" ≠ "Человек укусил собаку".

---

## Recurrent Neural Networks (RNN)

### Идея

RNN обрабатывает последовательность **по одному элементу**, поддерживая **скрытое состояние** (hidden state), которое "запоминает" историю:

```
x₁ → [RNN Cell] → h₁
         ↓ h₁
x₂ → [RNN Cell] → h₂
         ↓ h₂
x₃ → [RNN Cell] → h₃
         ↓ h₃
...
```

### Формулы

```
h_t = σ(W_hh · h_{t-1} + W_xh · x_t + b_h)    # обновление состояния
y_t = W_hy · h_t + b_y                           # выход
```

Одни и те же веса W_hh, W_xh **переиспользуются** на каждом шаге — weight sharing. Это позволяет обрабатывать последовательности любой длины.

### Проблемы RNN

1. **Vanishing gradient** — при обратном распространении через много шагов градиенты затухают. Модель "забывает" далёкий контекст.

2. **Последовательная обработка** — нельзя параллелизировать: h_t зависит от h_{t-1}. Тренировка медленная.

3. **Ограниченная память** — вся информация сжимается в вектор h_t фиксированного размера.

---

## LSTM (Long Short-Term Memory)

LSTM решает проблему vanishing gradient через **gate-механизм**:

```
                    ┌──────────────── cell state C_t ─────────────────┐
                    │                                                  │
Forget gate: f_t = σ(W_f·[h_{t-1}, x_t] + b_f)     # что забыть
Input gate:  i_t = σ(W_i·[h_{t-1}, x_t] + b_i)     # что запомнить
Cell update: C̃_t = tanh(W_c·[h_{t-1}, x_t] + b_c)  # кандидат
Cell state:  C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t       # обновление
Output gate: o_t = σ(W_o·[h_{t-1}, x_t] + b_o)     # что выдать
Hidden:      h_t = o_t ⊙ tanh(C_t)
```

**Ключевая идея:** cell state C_t проходит через сеть почти без изменений (highway), gates контролируют поток информации. Градиенты текут свободнее.

### GRU (Gated Recurrent Unit)

Упрощённая версия LSTM с двумя gates вместо трёх:

```
z_t = σ(W_z·[h_{t-1}, x_t])         # update gate
r_t = σ(W_r·[h_{t-1}, x_t])         # reset gate
h̃_t = tanh(W·[r_t ⊙ h_{t-1}, x_t]) # candidate
h_t = (1-z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

Меньше параметров, сравнимое качество.

---

## Seq2Seq: Encoder-Decoder

Для задач, где вход и выход — последовательности разной длины (перевод, суммаризация):

```
Encoder:  x₁, x₂, ..., xₙ → h_n (context vector)
Decoder:  h_n → y₁, y₂, ..., yₘ
```

**Проблема:** вся информация о входной последовательности сжата в один вектор h_n. Для длинных предложений это bottleneck.

---

## Attention Mechanism

### Решение bottleneck

Вместо одного вектора, decoder получает доступ ко **всем** скрытым состояниям encoder:

```
score(h_dec, h_enc_j) = h_dec^T · h_enc_j         # dot-product
α_j = softmax(score_j)                             # attention weights
context = Σ_j α_j · h_enc_j                       # weighted sum
```

На каждом шаге decoder "смотрит" на разные части входа. Для перевода слова "chat" → "кот" attention будет высоким на исходном "chat".

### Виды attention

| Тип | Score function | Автор |
|-----|---------------|-------|
| Dot-product | h_d^T · h_e | — |
| Bilinear | h_d^T · W · h_e | Luong (2015) |
| Additive | v^T · tanh(W₁h_d + W₂h_e) | Bahdanau (2014) |

Подробнее: [[Attention Mechanism]]

---

## Self-Attention и Transformer

### От RNN к Transformer

Фундаментальная проблема RNN — **последовательная** обработка. Self-attention решает это: каждый элемент взаимодействует с каждым **параллельно**.

### Self-Attention

Для каждого токена создаём три вектора:

```
Q = x · W_Q    # Query: "что я ищу?"
K = x · W_K    # Key: "что я содержу?"
V = x · W_V    # Value: "какую информацию передать?"
```

Формула scaled dot-product attention:

```
Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V
```

**√d_k** — нормализация, предотвращающая слишком большие значения dot-product.

### Multi-Head Attention

Несколько "голов" внимания работают параллельно, каждая фокусируясь на разных аспектах:

```
head_i = Attention(Q·W_Q^i, K·W_K^i, V·W_V^i)
MultiHead = Concat(head_1, ..., head_h) · W_O
```

Одна голова может следить за синтаксисом, другая — за семантикой, третья — за позициями.

### Transformer Block

```
Input
  │
  ▼
[Multi-Head Self-Attention]
  │ + residual connection
  ▼
[Layer Norm]
  │
  ▼
[Feed-Forward Network]
  │ + residual connection
  ▼
[Layer Norm]
  │
  ▼
Output
```

**Residual connections** — позволяют градиентам обходить слои
**[[Layer Normalization]]** — стабилизирует тренировку

### [[Positional Encoding]]

Self-attention не знает о порядке (permutation invariant). Позиционная информация добавляется через:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

Современные модели используют RoPE (Rotary Position Embeddings) — более эффективный метод.

---

## Сравнение архитектур

| Свойство | RNN/LSTM | Transformer |
|----------|----------|-------------|
| Параллелизация | Нет | Да |
| Long-range dependencies | Слабые (vanishing grad) | Прямой доступ |
| Compute complexity | O(n) на шаг | O(n²) attention |
| Memory | O(1) скрытое состояние | O(n²) attention matrix |
| Тренировка | Медленная | Быстрая (параллельно) |
| Доминирование | До 2018 | 2018+ |

**Вывод:** Transformer полностью заменил RNN в NLP и LLM. RNN остаётся актуальным только для edge-cases (streaming, очень длинные последовательности с линейной сложностью — Mamba, RWKV).

---

## Практика: Lab 1 — Music Generation

В Lab 1 студенты строят **RNN для генерации музыки**:

1. Обучают character-level LSTM на ABC нотации
2. Модель учится предсказывать следующий символ
3. При генерации сэмплируют из распределения (temperature sampling)

Это даёт интуицию о том, как autoregressive generation работает — тот же принцип, что в GPT, только масштаб другой.

---

## Ключевые выводы

1. **RNN** обрабатывает последовательности рекуррентно, но страдает от vanishing gradient
2. **LSTM** решает проблему через gate-механизм и cell state
3. **Attention** устраняет bottleneck encoder-decoder, давая доступ ко всем позициям
4. **Transformer** = self-attention + FFN, полностью параллелен
5. Transformer **заменил RNN** как стандартную архитектуру для последовательностей

---

## Источники

- MIT 6.S191, Lecture 2 — https://introtodeeplearning.com/
- "Attention Is All You Need" (Vaswani et al., 2017)
- Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate" (2014)

---

**См. также:** [[RNN]], [[Attention Mechanism]], [[Self-Attention]], [[Positional Encoding]], [[MIT 6.S191 — Intro to DL]]
