---
title: "Positional Encoding"
aliases: [positional embedding, position encoding, PE, RoPE, Rotary Positional Embedding, ALiBi, sinusoidal positional encoding]
type: concept
category: NLP
papers:
  - "[[02 Areas/ML & DL/Papers/Attention Is All You Need|Attention Is All You Need]]"
  - "[[02 Areas/ML & DL/Papers/Formal Algorithms for Transformers|Formal Algorithms for Transformers]]"
  - "[[02 Areas/ML & DL/Papers/DeBERTa|DeBERTa]]"
  - "[[02 Areas/ML & DL/Papers/LLaMA|LLaMA]]"
  - "[[02 Areas/ML & DL/Papers/Mistral 7B|Mistral 7B]]"
courses:
  - "[[02 Areas/ML & DL/Courses/Stanford CS224N/08 — Transformers|CS224N Lecture 8]]"
sources:
  - "[Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)"
  - "[Positional Embeddings in Transformers — Math Guide to RoPE & ALiBi](https://towardsdatascience.com/positional-embeddings-in-transformers-a-math-guide-to-rope-alibi/)"
  - "[EleutherAI — Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/)"
  - "[ICLR 2025 Blogpost — Positional Embeddings Evolution](https://iclr-blogposts.github.io/2025/blog/positional-embedding/)"
---

# Positional Encoding

## Зачем это нужно: attention не знает порядка

Self-attention — это операция над **множеством**, а не последовательностью. Если переставить токены местами, выход attention не изменится (permutation equivariance). Без позиционной информации модель не различает:

- *"собака укусила человека"*
- *"человек укусил собаку"*

Оба предложения содержат одинаковое множество токенов и дадут **идентичные** attention-веса. Позиционное кодирование решает эту проблему, инжектируя информацию о порядке.

**Формально** (из [[02 Areas/ML & DL/Papers/Formal Algorithms for Transformers|Formal Algorithms for Transformers]], Algorithm 2): для $t$-го токена финальный эмбеддинг = token embedding + position embedding:

$$e_t = W_e[:, x_t] + W_p[:, t]$$

где $W_p \in \mathbb{R}^{d_e \times T_{\max}}$ — позиционная матрица (фиксированная или обучаемая).

## 1. Sinusoidal Encoding (Vaswani et al., 2017)

Оригинальный Transformer использует **фиксированные** (не обучаемые) позиционные кодировки на основе синусов и косинусов разных частот.

### Формула

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

где $pos$ — позиция токена в последовательности, $i$ — индекс измерения (dimension), $d_{\text{model}}$ — размерность модели.

![[02 Areas/ML & DL/raw/papers/attention-is-all-you-need/images/transformer_positional_encoding_vectors.png]]
*Визуализация sinusoidal positional encoding: каждая строка — позиция, каждый столбец — dimension. Низкие dimensions (слева) меняются медленно, высокие (справа) — быстро (источник: Jay Alammar)*

### Интуиция

Представь двоичный счётчик: младший бит меняется каждый шаг, следующий — каждые два шага, и т.д. Sinusoidal encoding — это **непрерывная версия двоичного счётчика**:
- Первые dimensions (низкие $i$) — длинные волны, улавливают «грубую» позицию (начало / середина / конец)
- Последние dimensions (высокие $i$) — короткие волны, улавливают «тонкую» позицию (соседние токены)

### Почему именно sin/cos

Авторы выбрали эти функции потому, что для любого фиксированного смещения $k$:

$$PE_{pos+k} = f(PE_{pos})$$

то есть позицию $pos+k$ можно выразить как **линейное преобразование** позиции $pos$. Это позволяет модели легко обучиться «обращать внимание на позиции со смещением $k$» — например, «предыдущий токен» или «токен через 5 позиций».

### Характеристики

| Свойство | Значение |
|----------|----------|
| Обучаемые параметры | 0 |
| Тип | Абсолютная позиция |
| Экстраполяция | Теоретически — да (нет max position), практически — плохо |
| Используется в | Original Transformer (2017) |

## 2. Learned Absolute Embeddings (BERT, GPT-2)

Самый простой подход: позиция $t$ → обучаемый вектор $W_p[:, t]$, который **учится вместе со всей моделью** через обычный backprop.

### Как это работает

```
position_embedding = nn.Embedding(max_position, d_model)
# BERT: max_position = 512
# GPT-2: max_position = 1024 → 2048
```

Модель сама «решает», какой вектор назначить каждой позиции. Никаких формул — чистая оптимизация.

### Характеристики

| Свойство | Значение |
|----------|----------|
| Обучаемые параметры | $T_{\max} \times d_{\text{model}}$ |
| Тип | Абсолютная позиция |
| Экстраполяция | Нет — позиции за пределами $T_{\max}$ не определены |
| Используется в | BERT (512), GPT-2 (1024), GPT-3 (2048) |

**Практический результат**: learned embeddings слегка превосходят sinusoidal по качеству (Vaswani et al. отмечают "nearly identical results"), но не обобщаются на контексты длиннее обученного максимума.

## 3. Relative Positional Embeddings (DeBERTa, T5)

Идея: вместо абсолютных позиций кодировать **относительное расстояние** между парами токенов.

### DeBERTa: Disentangled Attention

Из [[02 Areas/ML & DL/Papers/DeBERTa|DeBERTa]]: разделяем content и position в отдельные embedding, и attention score складывается из трёх компонент:

$$A_{ij} = \underbrace{H_i H_j^T}_{\text{content-to-content}} + \underbrace{H_i P_{i|j}^T}_{\text{content-to-position}} + \underbrace{P_{i|j} H_j^T}_{\text{position-to-content}}$$

где $H_i$ — content embedding токена $i$, $P_{i|j}$ — relative position embedding для расстояния $i - j$.

- 512 relative position embeddings (от $-256$ до $+256$)
- При $|i-j| > 256$ используется крайнее значение (clipping)

### Характеристики

| Свойство | Значение |
|----------|----------|
| Обучаемые параметры | $(2k+1) \times d_{\text{model}}$ |
| Тип | Относительная позиция |
| Экстраполяция | Частичная (clipping для дальних позиций) |
| Используется в | DeBERTa, DeBERTa-v2 |

## 4. RoPE — Rotary Positional Embedding (LLaMA, Mistral, Qwen)

![[02 Areas/ML & DL/raw/papers/positional-encoding/images/rope-implementation.png]]
*Реализация RoPE: вращение пар измерений query и key векторов на угол, пропорциональный позиции (источник: Su et al., 2021)*

![[02 Areas/ML & DL/raw/papers/positional-encoding/images/rope-decay.png]]
*Long-term decay RoPE: внутреннее произведение убывает с ростом относительного расстояния между токенами (источник: Su et al., 2021)*

RoPE (Su et al., 2021) — самый широко используемый позиционный метод в современных LLM. Элегантно кодирует **относительную** позицию через **вращение** query и key векторов.

### Ключевая идея

Вместо того чтобы **прибавлять** позиционный вектор к эмбеддингу, RoPE **вращает** query и key векторы в двумерных подпространствах на угол, пропорциональный позиции.

Для пары измерений $(2i, 2i+1)$ вектора на позиции $m$:

$$R_{\theta,m} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}$$

где $\theta_i = 10000^{-2i/d}$ — частота для $i$-й пары dimensions (та же формула, что в sinusoidal encoding).

### Почему это работает

После вращения, dot-product между query на позиции $m$ и key на позиции $n$ зависит только от **разности** позиций:

$$\text{Re}\langle f(q, m), f(k, n) \rangle = g(q, k, m - n)$$

Угол между вращёнными векторами определяется разностью $m - n$, а не абсолютными позициями. Модель автоматически получает relative position encoding.

### Свойства

- **Сохраняет норму**: вращение не меняет длину вектора → стабильность обучения
- **Decaying dependency**: attention score естественно убывает с расстоянием (от LLaMA paper: это помогает моделировать locality)
- **Не добавляет параметров**: формула фиксирована, как sinusoidal
- **Экстраполяция**: значительно лучше, чем absolute, но всё ещё ограничена

### RoPE Scaling для длинных контекстов

Проблема: модель, обученная на 4096 позициях, плохо работает на 8192. Решения:
- **Position Interpolation** (Chen et al., 2023): масштабировать позиции, чтобы 8192 «влезло» в диапазон 0-4096. Простая идея, хорошие результаты.
- **YaRN** (Peng et al., 2023): более сложное масштабирование с учётом разных частот.
- **NTK-aware scaling**: модификация базовой частоты $\theta$.

Результат: LLaMA (4K) → LLaMA 2 (4K, extends to 32K) → LLaMA 3 (8K, extends to 128K через RoPE scaling).

### Характеристики

| Свойство | Значение |
|----------|----------|
| Обучаемые параметры | 0 |
| Тип | Относительная позиция (через вращение) |
| Экстраполяция | Хорошая (особенно с scaling) |
| Используется в | **LLaMA 1/2/3, Mistral, Qwen, Gemma, CodeLlama** |

## 5. ALiBi — Attention with Linear Biases (BLOOM)

ALiBi (Press et al., 2022) — радикально простой подход: **вообще не трогать эмбеддинги**, а добавить bias напрямую в attention scores.

### Формула

$$\text{score}(i, j) = q_i \cdot k_j - m_h \cdot |i - j|$$

где $m_h$ — slope, фиксированный для каждой головы: $m_h = 2^{-8h/H}$ для $h = 1, \ldots, H$ (геометрическая прогрессия).

### Интуиция

Каждая голова получает свой «штраф за расстояние»:
- Головы с маленьким $m_h$ (slope ≈ 0) → длинные зависимости, «видят далеко»
- Головы с большим $m_h$ → сильный recency bias, фокус на соседних токенах

Модель автоматически разделяет головы на «глобальные» и «локальные».

### Характеристики

| Свойство | Значение |
|----------|----------|
| Обучаемые параметры | 0 |
| Тип | Relative bias в attention |
| Экстраполяция | **Отличная** — лучшая среди всех методов |
| Используется в | BLOOM, MPT |

**Главное преимущество**: модель, обученная на 1024 токенах, работает на 2048+ без дообучения. ALiBi показывает лучшую экстраполяцию, чем RoPE без scaling.

**Недостаток**: на практике RoPE с scaling достигает сопоставимого или лучшего качества на длинных контекстах, и RoPE стал де-факто стандартом.

## Сравнительная таблица

| Метод | Тип | Параметры | Экстраполяция | Модели |
|-------|-----|-----------|---------------|--------|
| Sinusoidal | Absolute, fixed | 0 | Плохая | Original Transformer |
| Learned | Absolute, learned | $T \times d$ | Нет | BERT, GPT-2/3 |
| Relative (DeBERTa) | Relative, learned | $(2k+1) \times d$ | Частичная | DeBERTa |
| **RoPE** | Relative, fixed | 0 | **Хорошая** | **LLaMA, Mistral, Qwen** |
| ALiBi | Relative bias | 0 | Отличная | BLOOM, MPT |

## Хронология

| Год | Метод | Модель / Статья |
|-----|-------|----------------|
| 2017 | Sinusoidal PE | Vaswani et al. — Attention Is All You Need |
| 2018 | Learned absolute | BERT (512), GPT-2 (1024) |
| 2019 | Relative PE (T5) | Raffel et al. — T5 |
| 2020 | Disentangled relative | He et al. — DeBERTa |
| 2021 | **RoPE** | Su et al. — RoFormer |
| 2022 | ALiBi | Press et al. — Train Short, Test Long |
| 2023 | Position Interpolation | Chen et al. — Extending Context Window |
| 2023 | YaRN | Peng et al. — Yet another RoPE extension |
| 2024 | NTK-aware scaling | Community research for LLaMA 3 |

## Related concepts

- [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]] — без PE self-attention инвариантен к порядку
- [[02 Areas/ML & DL/Concepts/NLP/Self-Attention|Self-Attention]] — RoPE применяется к Q и K в self-attention
- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] — позиционное кодирование = обязательный компонент
- [[02 Areas/ML & DL/Concepts/Architectures/LLaMA|LLaMA]] — популяризировал RoPE
- [[02 Areas/ML & DL/Concepts/Architectures/Mistral 7B|Mistral 7B]] — RoPE + Sliding Window Attention

## Дополнительные ресурсы

- [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — визуализация sinusoidal PE
- [EleutherAI — Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/) — глубокий разбор математики RoPE
- [TDS — Positional Embeddings: Math Guide to RoPE & ALiBi](https://towardsdatascience.com/positional-embeddings-in-transformers-a-math-guide-to-rope-alibi/) — математическое сравнение
- [ICLR 2025 — Positional Embeddings Evolution](https://iclr-blogposts.github.io/2025/blog/positional-embedding/) — обзор от текста до vision
- [Michael Brenndoerfer — Position Encoding Comparison](https://mbrenndoerfer.com/writing/position-encoding-comparison-transformers) — интерактивное сравнение
