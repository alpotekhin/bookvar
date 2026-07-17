---
title: "Voita NLP — Seq2Seq and Attention"
type: course-note
course: "Lena Voita NLP"
---

# Voita NLP — Seq2Seq and Attention

> От encoder-decoder до Transformer. Полный путь: как attention решил bottleneck и стал основой LLM.

**Курс:** [[Lena Voita NLP/_index|Lena Voita NLP Course]]
**Страница:** https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html
**Связанные концепты:** [[Attention Mechanism]], [[Self-Attention]], [[Positional Encoding]], [[Feed-Forward Network]], [[Tokenization]]

---

## Conditional Language Model

Seq2seq-задачи — это **условная генерация**: по входной последовательности x генерируем выходную y.

```
p(y | x) = p(y₁|x) · p(y₂|y₁, x) · p(y₃|y₁, y₂, x) · ...
```

Примеры: машинный перевод, суммаризация, question answering, dialogue.

В отличие от безусловной LM (p(y)), здесь каждый токен зависит от **всей** входной последовательности x.

---

## Encoder-Decoder Architecture

### Базовый LSTM Encoder-Decoder

**Encoder:** читает входную последовательность, формирует representation.

```
x₁ → [LSTM] → h₁
        ↓ h₁
x₂ → [LSTM] → h₂
        ↓ h₂
...
x_n → [LSTM] → h_n   ← финальное скрытое состояние = "контекст"
```

**Decoder:** получает финальное h_n как начальное состояние, генерирует выход:

```
h_n → [LSTM] → y₁
  y₁ → [LSTM] → y₂
    y₂ → [LSTM] → y₃
      ...
```

### Bottleneck Problem

Вся информация о входной последовательности **сжата** в один вектор h_n фиксированной размерности (256-1024). Для длинных предложений это катастрофа:

```
"The European Commission said it would propose measures to 
 protect the interests of consumers in the internal market"
                        ↓
              h_n ∈ R^512 ← ВСЯ информация здесь
                        ↓
              "Европейская комиссия заявила..."
```

Эмпирически: качество перевода **падает** с ростом длины предложения (Sutskever et al., 2014 показали, что reverse input помогает — ближний контекст важнее).

---

## Attention: решение bottleneck

### Ключевая идея (Bahdanau et al., 2014)

Вместо одного вектора, decoder на каждом шаге **обращается ко всем** скрытым состояниям encoder:

```
Encoder states: [h₁, h₂, h₃, ..., h_n]
                  ↑   ↑   ↑        ↑
Decoder step t:   α₁  α₂  α₃  ... α_n   (attention weights)
                  ↓   ↓   ↓        ↓
Context:        c_t = Σ αᵢ · hᵢ
```

**α_i** — "сколько внимания уделить i-й позиции входа на текущем шаге декодирования."

### Вычисление Attention

**Шаг 1: Attention scores**

```
score(s_t, h_j) = alignment_function(s_t, h_j)
```

где s_t — состояние decoder, h_j — состояние encoder на позиции j.

**Шаг 2: Attention weights (softmax)**

```
α_j = exp(score_j) / Σ_k exp(score_k)
```

**Шаг 3: Context vector**

```
c_t = Σ_j α_j · h_j
```

### Score Functions

Voita подробно разбирает три варианта:

| Тип | Формула | Кто |
|-----|---------|-----|
| Dot-product | s_t^T · h_j | Простейший |
| Bilinear | s_t^T · W · h_j | Luong (2015) |
| Additive (MLP) | v^T · tanh(W₁·s_t + W₂·h_j) | Bahdanau (2014) |

### Bahdanau vs Luong

**Bahdanau attention:**
- Bidirectional encoder (→ и ← LSTM)
- Additive score function (MLP)
- Attention **между** шагами decoder (перед предсказанием)

**Luong attention:**
- Unidirectional encoder
- Bilinear score function
- Attention **после** шага decoder RNN

Оба учат **soft alignment** — мягкое соответствие между позициями source и target. Attention weights визуализируются как alignment matrix.

---

## Transformer: Attention Is All You Need

### Мотивация

RNN + Attention всё ещё **последовательный** — нельзя параллелизировать encoder. Vaswani et al. (2017) предложили: убрать рекуррентность полностью, оставить только attention.

### Self-Attention

Каждый токен "общается" с каждым другим через три проекции:

```
Q = X · W_Q    (queries — "что я ищу")
K = X · W_K    (keys — "что я предлагаю")
V = X · W_V    (values — "какую информацию передать")
```

**Scaled Dot-Product Attention:**

```
Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V
```

**√d_k** (scaling) — критически важен. Без него при больших d_k dot-products растут, softmax "насыщается" (один элемент ≈ 1, остальные ≈ 0), и градиенты исчезают.

### Multi-Head Attention

Несколько "голов" параллельно, каждая с собственными W_Q, W_K, W_V:

```
head_i = Attention(X·W_Q^i, X·W_K^i, X·W_V^i)
MultiHead(X) = Concat(head_1, ..., head_h) · W_O
```

Зачем? Разные головы фокусируются на **разных типах отношений**:
- Позиционные: следят за соседними токенами
- Синтаксические: subject-verb, verb-object
- Семантические: coreference, semantic roles
- Редкие токены: фокусируются на необычных словах

Voita показала (в своём research), что большинство голов можно pruning-нуть без потери качества — модель избыточна.

### Masked Self-Attention (Decoder)

В decoder будущие токены **маскируются** — модель не должна "подглядывать":

```
Mask:
     t₁  t₂  t₃  t₄
t₁ [  1   0   0   0 ]
t₂ [  1   1   0   0 ]
t₃ [  1   1   1   0 ]
t₄ [  1   1   1   1 ]
```

Реализация: заменяем замаскированные позиции на -∞ перед softmax.

### Transformer Block — полная архитектура

**Encoder block:**

```
x → [Multi-Head Self-Attention] → + (residual) → [LayerNorm]
  → [Feed-Forward Network]      → + (residual) → [LayerNorm]
```

**Decoder block:**

```
y → [Masked Multi-Head Self-Attention] → + → [LayerNorm]
  → [Cross-Attention to Encoder]       → + → [LayerNorm]
  → [Feed-Forward Network]             → + → [LayerNorm]
```

### Компоненты

**[[Feed-Forward Network]]:**

```
FFN(x) = max(0, x·W₁ + b₁) · W₂ + b₂
```

Два линейных слоя с ReLU. Обычно inner dimension = 4x model dimension. Voita описывает FFN как **key-value memory**: первый слой (key) активирует "паттерны", второй (value) генерирует соответствующие распределения.

**Residual Connections:**

```
output = LayerNorm(x + Sublayer(x))
```

Позволяют градиентам обходить слои напрямую, критически важны для глубоких сетей.

**[[Layer Normalization]]:**

Нормализует каждое представление **независимо** (в отличие от BatchNorm, который нормализует по batch-у):

```
LN(x) = γ · (x - μ) / σ + β
```

**[[Positional Encoding]]:**

Sinusoidal positional encoding:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

Каждая позиция получает уникальный вектор. Свойство: PE(pos+k) — линейное преобразование PE(pos), что потенциально помогает модели обучить relative position attention.

---

## BPE: Byte Pair Encoding

Voita подробно разбирает BPE в контексте seq2seq:

### Зачем subword токенизация

Open-vocabulary проблема: в реальном мире бесконечно много слов (имена, неологизмы, составные слова в немецком). Word-level модели не справляются.

### BPE-Dropout

Стандартный BPE детерминирован — одно и то же слово всегда разбивается одинаково. BPE-dropout **случайно пропускает** некоторые merge-правила при тренировке:

```
"tokenization":
  Standard BPE: ["token", "ization"]
  BPE-dropout:  ["to", "ken", "iza", "tion"]  (один вариант)
                ["tok", "eniz", "ation"]       (другой вариант)
```

Это регуляризация: модель видит разные разбиения одних слов, становится робастнее. Embedding-и subword-ов становятся **более семантическими** (ближние соседи осмысленнее).

---

## Inference: Decoding Strategies

### Greedy Decoding

На каждом шаге выбираем токен с максимальной вероятностью:

```
y_t = argmax_w p(w | y_{<t}, x)
```

Быстро, но **субоптимально** — локально лучший выбор не гарантирует глобально лучшую последовательность.

### Beam Search

Храним B (beam size) лучших гипотез на каждом шаге:

```
Beam size = 3:

Step 1:  "The" (0.5)    "A" (0.3)    "My" (0.1)
Step 2:  "The cat" (0.3) "The dog" (0.1) "A cat" (0.2)
Step 3:  ...
```

На каждом шаге: для каждой из B гипотез рассматриваем все возможные продолжения, оставляем Top-B.

**Типичный beam size:** 4-10. Больше — marginal improvements. Слишком большой beam → generic/boring outputs.

### Length Normalization

Beam search предпочитает **короткие** последовательности (меньше множителей < 1 в произведении вероятностей). Решение:

```
score(y) = (1/|y|^α) · log p(y | x)
```

α = 0.6-1.0 компенсирует bias к коротким ответам.

---

## Analysis: что выучил Transformer

Voita приводит исследования внимания:

### Роли attention heads

- **Positional heads:** следят за соседними позициями (i-1, i+1)
- **Syntactic heads:** кодируют зависимости subject-verb, verb-object
- **Rare token heads:** фокусируются на редких словах

### Pruning attention heads

Большинство голов можно удалить без значительной потери качества. Модель имеет значительную избыточность — не все 96 голов (в 12-layer, 8-head модели) необходимы.

### Probing

Representations разных слоёв кодируют разную лингвистическую информацию:
- Ранние слои → POS tags
- Средние слои → syntax
- Поздние слои → semantics

---

## Ключевые выводы

1. **Encoder-decoder** с LSTM создаёт bottleneck — один вектор для всего input
2. **Attention** решает bottleneck, давая decoder доступ ко всем позициям
3. **Transformer** убирает рекуррентность, оставляя только attention — полная параллелизация
4. **Self-attention** = Q·K^T/√d · V + multi-head для разных типов отношений
5. **BPE** решает open-vocabulary, **BPE-dropout** добавляет регуляризацию
6. **Beam search** лучше greedy, но требует length normalization

---

## Источники

- Lena Voita, NLP Course: Seq2Seq and Attention — https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html
- Vaswani et al., "Attention Is All You Need" (2017)
- Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate" (2014)
- Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units" (BPE, 2016)

---

**См. также:** [[Attention Mechanism]], [[Self-Attention]], [[Positional Encoding]], [[Feed-Forward Network]], [[Voita NLP — Word Embeddings]], [[Voita NLP — Transfer Learning]]
