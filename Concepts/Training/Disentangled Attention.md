---
title: "Disentangled Attention"
aliases: [disentangled attention mechanism, разделённый механизм внимания]
type: concept
status: legacy
category: Training
papers:
  - "[[02 Areas/ML & DL/Papers/DeBERTa]]"
courses: []
sources:
  - "[He et al. -- DeBERTa: Decoding-enhanced BERT with Disentangled Attention (2020)](https://arxiv.org/abs/2006.03654)"
  - "[He et al. -- DeBERTaV3 (2023)](https://arxiv.org/abs/2111.09543)"
  - "[Su et al. -- RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE, 2021)](https://arxiv.org/abs/2104.09864)"
---

# Disentangled Attention

## Зачем это нужно: content и position -- разные вещи

В стандартном BERT каждый токен представлен **одним вектором**, в который сложены content embedding и absolute position embedding:

$$e_i = h_i + p_i$$

Когда вычисляется attention score $A_{ij} = e_i \cdot e_j^T$, содержание и позиция **смешиваются**:

$$A_{ij} = (h_i + p_i) \cdot (h_j + p_j)^T = \underbrace{h_i h_j^T}_{\text{content-content}} + \underbrace{h_i p_j^T}_{\text{content-pos}} + \underbrace{p_i h_j^T}_{\text{pos-content}} + \underbrace{p_i p_j^T}_{\text{pos-pos}}$$

Проблема: модель **не может независимо** решить, насколько важны:
- **Схожесть содержания** слов (semantic relatedness)
- **Расстояние** между словами (proximity)

Пример: в фразе "deep learning is powerful", слова "deep" и "learning" связаны и **семантически** (значение), и **позиционно** (соседи). Но "deep" и "powerful" связаны **семантически** (оба про силу/глубину), но **позиционно** далеки. Стандартный attention не может разделить эти два типа связи.

## Идея DeBERTa: два отдельных вектора

He et al. (2020) предложили **разделить** (disentangle) representation каждого токена на два **независимых вектора**:

- $H_i$ -- **content embedding** (семантика слова $i$)
- $P_{i|j}$ -- **relative position embedding** (позиция $i$ относительно $j$)

```
Стандартный BERT:
  Token i --> один вектор e_i = content + position

DeBERTa:
  Token i --> два вектора: H_i (content) и P_{i|j} (relative position)
              P зависит от разности позиций (i - j), не от абсолютной i
```

## Три компонента attention score

### Разложение

DeBERTa вычисляет attention score как сумму **трёх** (из четырёх возможных) взаимодействий:

$$A_{ij} = \underbrace{\frac{H_i W_q^c \cdot (H_j W_k^c)^T}{\sqrt{3d}}}_{\text{content-to-content}} + \underbrace{\frac{H_i W_q^c \cdot (P_{j|i} W_k^p)^T}{\sqrt{3d}}}_{\text{content-to-position}} + \underbrace{\frac{P_{i|j} W_q^p \cdot (H_j W_k^c)^T}{\sqrt{3d}}}_{\text{position-to-content}}$$

Четвёртый term (position-to-position) **исключён** -- эксперименты показали, что он не даёт значимого улучшения.

### Что каждый term захватывает

**Content-to-content** ($H_i \cdot H_j$): насколько **содержание** слова $i$ семантически связано с содержанием слова $j$. Это стандартный self-attention -- "о чём эти слова?".

```
"The cat sat on the mat"
  "cat" --> "sat": высокий c2c score (субъект-глагол)
  "cat" --> "mat": средний c2c score (рифма, но разная семантика)
```

**Content-to-position** ($H_i \cdot P_{j|i}$): слово $i$ обращает внимание на **позицию** слова $j$ относительно себя. "Мне как глаголу важно, чтобы мой субъект был **близко слева**."

```
"The cat sat on the mat"
  "sat" смотрит на позицию -1 (слово слева) -- там субъект
  "sat" смотрит на позицию +1 (слово справа) -- там предлог
```

**Position-to-content** ($P_{i|j} \cdot H_j$): **позиция** слова $i$ относительно $j$ обращает внимание на содержание $j$. "На расстоянии 2 слева от меня что-то важное -- что именно?"

```
"The cat sat on the mat"
  Для "sat": позиция -2 --> "The" (определитель, менее важен)
  Для "sat": позиция -1 --> "cat" (субъект, важен!)
```

### Масштабирование

Делитель $\sqrt{3d}$ (вместо стандартного $\sqrt{d}$) компенсирует сумму **трёх** матриц: дисперсия суммы = 3 * дисперсия одного term, поэтому нормализуем на $\sqrt{3}$.

## Relative vs Absolute Position Embeddings

### Почему relative positions лучше

DeBERTa использует **relative position embeddings**: $P_{i|j} = P_{i-j}$ зависит от **разности позиций**, а не от абсолютной позиции $i$.

```
Absolute positions (BERT, GPT-2):
  "deep learning" на позициях (5, 6) и (100, 101)
  --> разные position embeddings, хотя отношение то же

Relative positions (DeBERTa):
  "deep learning" всегда имеет P_{delta=1}
  --> одинаковое position embedding независимо от абсолютной позиции
```

**Преимущество 1**: **обобщение на длинные контексты**. Absolute positions на позиции 513 -- если модель обучалась на max 512 токенов, нет embedding для позиции 513. Relative positions всегда имеют embedding для $\delta = 1$ (соседний токен), независимо от абсолютной позиции.

**Преимущество 2**: **семантическая осмысленность**. Для многих языковых паттернов важно **расстояние** между словами, а не их абсолютная позиция в предложении. Субъект обычно **рядом** с глаголом, но может быть на любой абсолютной позиции.

### Реализация

DeBERTa хранит **до 512 relative position embeddings**: $P_k$ для $k \in [-256, 256]$. Позиции за пределами этого диапазона clamp-ятся к граничным значениям.

Каждый relative position embedding имеет ту же размерность $d$, что и content embedding. У каждого attention head свои проекционные матрицы $W_k^p$ и $W_q^p$ для position embeddings.

## Enhanced Mask Decoder (EMD)

### Проблема

Disentangled attention использует **только relative positions** -- абсолютная позиция маскированного токена нигде не учитывается. Но для MLM предсказания абсолютная позиция **важна**:

```
"A new [MASK] was opened in the city"
  [MASK] на позиции 3 (начало предложения) --> скорее "store", "restaurant"
  [MASK] на позиции 15 (конец документа) --> может быть другая вероятность
```

Абсолютная позиция влияет на распределение слов: начало предложения, конец абзаца, title vs body -- разные distributions.

### Решение: дополнительный Transformer слой

**Enhanced Mask Decoder** -- один дополнительный Transformer слой, который:
1. Берёт output DeBERTa (relative positions only)
2. Добавляет **absolute position embeddings** (как в BERT)
3. Пропускает через attention + FFN
4. Результат подаётся в prediction head

**Ключевой момент**: EMD применяется **только для маскированных позиций** при pre-training (MLM). Это минимизирует дополнительный compute.

```
DeBERTa Layers (relative positions only)
         |
         v
[Enhanced Mask Decoder]  <-- absolute positions добавлены здесь
         |
         v
[MLM Prediction Head: Linear + Softmax]
```

## Ключевые результаты

### DeBERTa Large vs RoBERTa Large

Тот же training compute, та же архитектура (кроме disentangled attention):

| Задача | RoBERTa Large | DeBERTa Large | Gain |
|--------|--------------|--------------|------|
| MNLI-m/mm | 90.2/90.2 | **91.1/90.9** | +0.9 |
| SQuAD v2.0 F1 | 88.4 | **90.7** | **+2.3** |
| RACE | 83.2 | **86.8** | **+3.6** |

Улучшения значительные, особенно на задачах, требующих **точного понимания позиционных отношений** (SQuAD: где в тексте ответ? RACE: логические связи в длинном тексте).

### DeBERTa XXL (1.5B): первая модель > human на SuperGLUE

| | Score |
|-|-------|
| Human baseline | 89.8 |
| **DeBERTa XXL** | **89.9** |
| T5 11B | 89.3 |

DeBERTa XXL (1.5B params) **первой превзошла human performance** на SuperGLUE -- при **7x меньше параметров** чем T5 11B.

## Связь с другими positional encoding методами

| Метод | Тип | Используется в | Disentangled? |
|-------|-----|---------------|--------------|
| Absolute (learned) | Absolute | BERT, GPT-2 | Нет (mixed) |
| Sinusoidal | Absolute | Original Transformer | Нет (mixed) |
| **Disentangled** | **Relative** | **DeBERTa** | **Да** |
| RoPE | Relative | LLaMA, Mistral | Частично |
| ALiBi | Relative | BLOOM | Нет (bias) |
| Relative bias | Relative | T5 | Нет (bias) |

### RoPE vs Disentangled Attention

RoPE (Su et al., 2021) -- другой подход к relative positions, доминирующий в decoder-only моделях:
- **RoPE**: вращает Q и K вектора на угол, зависящий от позиции. Relative position кодируется в **dot-product** Q и K через геометрические свойства вращения.
- **DeBERTa**: отдельные position embeddings с explicit content-position interactions.

RoPE проще в реализации и лучше масштабируется на длинные контексты (через NTK-aware scaling). DeBERTa's approach более expressive (три отдельных interaction term), но сложнее.

## DeBERTa-v3: современная production модель

DeBERTa-v3 (He et al., 2023) -- улучшенная версия:
- **ELECTRA-style pre-training** (Replaced Token Detection) вместо MLM
- **Gradient-disentangled embedding sharing** -- обмен embeddings между generator и discriminator
- Стала **стандартной production encoder** для NLU задач с высокими требованиями к точности

```
Когда использовать DeBERTa-v3?
- Entity extraction / NER в production
- Classification с high-stakes (медицина, финансы)
- QA / Reading Comprehension
- Любая NLU задача, где важна максимальная точность
```

## Key papers

- [[02 Areas/ML & DL/Papers/DeBERTa]] -- оригинал (He et al., 2020)

## Related concepts

- [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]] -- базовый механизм, который disentangled attention улучшает
- [[02 Areas/ML & DL/Concepts/NLP/Positional Encoding|Positional Encoding]] -- relative vs absolute positions
- [[02 Areas/ML & DL/Concepts/Architectures/DeBERTa|DeBERTa]] -- архитектура, использующая disentangled attention
- [[02 Areas/ML & DL/Concepts/Architectures/BERT|BERT]] -- baseline с absolute positions
- [[02 Areas/ML & DL/Concepts/Training/Masked Language Modeling|Masked Language Modeling (MLM)]] -- pre-training task, для которого нужен EMD

## Дополнительные ресурсы

- [He et al. -- DeBERTa (2020)](https://arxiv.org/abs/2006.03654) -- оригинальная статья
- [He et al. -- DeBERTaV3 (2023)](https://arxiv.org/abs/2111.09543) -- production-ready версия
- [Su et al. -- RoPE (2021)](https://arxiv.org/abs/2104.09864) -- альтернативный relative position encoding
- [HuggingFace -- DeBERTa-v3 models](https://huggingface.co/microsoft/deberta-v3-base) -- pretrained модели
