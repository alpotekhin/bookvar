---
title: "Self-Attention"
aliases: [intra-attention, self-attention mechanism, само-внимание]
type: concept
category: NLP
papers:
  - "[[02 Areas/ML & DL/Papers/Attention Is All You Need|Attention Is All You Need]]"
  - "[[02 Areas/ML & DL/Papers/Formal Algorithms for Transformers]]"
  - "[[02 Areas/ML & DL/Papers/BERT]]"
  - "[[02 Areas/ML & DL/Papers/DeBERTa]]"
  - "[[02 Areas/ML & DL/Papers/Flash Attention|Flash Attention]]"
courses:
  - "[[02 Areas/ML & DL/Courses/Stanford CS224N/08 — Transformers|CS224N Lecture 8]]"
sources:
  - "[Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)"
  - "[Lena Voita — Self-Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html)"
  - "[Peter Bloem — Transformers from Scratch](https://peterbloem.nl/blog/transformers)"
---

# Self-Attention

## Что такое self-attention и зачем он нужен

Self-attention (само-внимание, intra-attention) --- это частный случай [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|механизма внимания]], в котором **Query, Key и Value вычисляются из одной и той же последовательности**. Каждый токен обновляет своё представление, «опрашивая» все остальные токены этой же последовательности --- какие из них сейчас релевантны для его понимания.

До 2017 года attention использовался **между** двумя последовательностями: декодер «смотрел» на энкодер (Bahdanau, 2014; Luong, 2015). Vaswani et al. в статье "Attention Is All You Need" сделали ключевой шаг: attention **внутри** одной последовательности, причём как единственный механизм (без RNN/CNN). Именно self-attention --- главный строительный блок [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]].

## Формальное определение

Пусть дана входная последовательность $X \in \mathbb{R}^{n \times d}$ из $n$ токенов размерности $d$. Self-attention вычисляет:

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

где $W^Q, W^K \in \mathbb{R}^{d \times d_k}$, $W^V \in \mathbb{R}^{d \times d_v}$ --- обучаемые матрицы проекций.

Выход:

$$\text{SelfAttention}(X) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Из [[02 Areas/ML & DL/Papers/Formal Algorithms for Transformers|Formal Algorithms for Transformers]] (DeepMind):

```
Attention(X, Z=X | W, Mask):
    Q = W_q X       # queries из X
    K = W_k X       # keys из X (та же последовательность!)
    V = W_v X       # values из X
    S = K^T Q / sqrt(d_attn)
    V_tilde = V * softmax(mask(S))
```

Ключевое отличие от cross-attention: **$Z = X$** --- одна и та же последовательность используется и для запросов, и для ответов.

## Пошаговая интуиция: что происходит при вычислении

Представь предложение: *"The animal didn't cross the street because **it** was too tired."*

К чему относится «it»? Для человека очевидно --- к «animal». Self-attention позволяет модели установить эту связь автоматически.

**Шаг 1. Проекция.** Эмбеддинг каждого токена (например, 512-мерный) проецируется тремя матрицами в три вектора:
- **Query** ($q_i$) --- «что я ищу?» Токен «it» формирует запрос: «кто я такой? что стоит за мной?»
- **Key** ($k_i$) --- «чем я могу быть полезен?» Токен «animal» объявляет: «я --- существительное, субъект действия»
- **Value** ($v_i$) --- «что я содержу?» Информация, которую токен передаёт тем, кто на него обратит внимание

**Шаг 2. Скоринг.** Dot-product $q_i^T k_j$ для каждой пары позиций $(i, j)$. Чем больше score --- тем «релевантнее» позиция $j$ для позиции $i$.

**Шаг 3. Масштабирование.** Деление на $\sqrt{d_k}$. Без этого при больших $d_k$ dot-product растёт, softmax уходит в насыщение, градиенты исчезают. Если компоненты $q$ и $k$ --- независимые случайные величины с нулевым средним и единичной дисперсией, то $q \cdot k$ имеет дисперсию $d_k$.

**Шаг 4. Softmax.** Нормализация в вероятности: $\alpha_{ij} = \text{softmax}_j(s_{ij})$. Сумма весов по всем позициям = 1.

**Шаг 5. Взвешенная сумма Values.** Выход для позиции $i$: $\sum_j \alpha_{ij} v_j$ --- контекстно-обогащённое представление, содержащее информацию от всех релевантных позиций.

## Causal vs Bidirectional: два режима маскирования

Self-attention --- один и тот же механизм, но маска кардинально меняет поведение:

### Bidirectional (двунаправленный) self-attention

- **Mask $\equiv$ 1** --- нет маскировки
- Каждый токен на позиции $t$ видит **все** токены $1 \ldots T$
- Используется в encoder-моделях: [[02 Areas/ML & DL/Concepts/Architectures/BERT|BERT]], [[02 Areas/ML & DL/Concepts/Architectures/RoBERTa|RoBERTa]], [[02 Areas/ML & DL/Concepts/Architectures/DeBERTa|DeBERTa]]
- Задача: **понимание** (NLU) --- классификация, NER, извлечение информации

### Causal (однонаправленный) self-attention

- **Mask$[t_z, t_x] = \mathbb{1}[t_z \leq t_x]$** --- нижнетреугольная маска
- Будущие позиции маскируются значением $-\infty$ перед softmax (после softmax = 0)
- Токен на позиции $t$ видит только позиции $1 \ldots t$
- Используется в decoder-моделях: GPT, [[02 Areas/ML & DL/Concepts/Architectures/LLaMA|LLaMA]], Mistral
- Задача: **генерация** --- авторегрессивное предсказание следующего токена

### Prefix self-attention

- Часть последовательности (prefix) --- bidirectional, оставшаяся часть --- causal
- Используется в prefix LM (например, [[02 Areas/ML & DL/Concepts/Architectures/T5|T5]] в некоторых режимах), U-PaLM

### Сравнение

| Свойство | Bidirectional | Causal | Prefix |
|----------|:------------:|:------:|:------:|
| Видит будущее | Да | Нет | Частично |
| Авторегрессивная генерация | Нет | Да | Да |
| Типичные модели | BERT, RoBERTa | GPT, LLaMA | PaLM, T5 |
| Задача pre-training | MLM | CLM | MLM + CLM |

## Отличие от cross-attention

| Свойство | Self-Attention | Cross-Attention |
|----------|:-------------:|:---------------:|
| Q из | $X$ | $X$ (decoder) |
| K, V из | $X$ (та же последовательность) | $Z$ (encoder) |
| Назначение | Контекстуализация внутри одной последовательности | Обмен информацией между двумя последовательностями |
| Пример | Encoder layer, decoder masked self-attention | Encoder-decoder attention в Transformer |

## Сложность и ограничения

Матрица attention scores $QK^T$ имеет размер $n \times n$, что определяет сложность:

| Метрика | Self-Attention | RNN |
|---------|:-------------:|:---:|
| Сложность на слой | $O(n^2 \cdot d)$ | $O(n \cdot d^2)$ |
| Последовательные операции | $O(1)$ | $O(n)$ |
| Макс. длина пути | $O(1)$ | $O(n)$ |
| Параллелизм | Полный | Нет |

**Преимущество:** $O(1)$ максимальная длина пути --- любые два токена связаны напрямую, без промежуточных шагов. Это решает проблему **vanishing gradients на длинных зависимостях**, от которой страдали RNN.

**Цена:** $O(n^2)$ по длине последовательности. При $n = 4096$ матрица attention занимает $4096^2 \times 4 = 64$MB на одну голову (float32). При $n = 128{,}000$ это уже ~60GB. Отсюда мотивация для:
- [[02 Areas/ML & DL/Concepts/Inference/Flash Attention|Flash Attention]] --- IO-aware тайлинг, та же математика, но 2--4x быстрее
- Linear attention (Katharopoulos et al., 2020) --- аппроксимация kernel trick, $O(n \cdot d^2)$
- Sparse attention (BigBird, Longformer) --- attention только к подмножеству позиций
- [[02 Areas/ML & DL/Concepts/Architectures/Mamba|Mamba]] --- SSM, вообще без attention

## Disentangled self-attention (DeBERTa)

[[02 Areas/ML & DL/Papers/DeBERTa|DeBERTa]] предложила разделить content и position в self-attention. Вместо одного score между токенами вычисляются **четыре** компонента:

$$A_{ij} = \underbrace{H_i H_j^T}_{\text{content-to-content}} + \underbrace{H_i P_{i|j}^T}_{\text{content-to-position}} + \underbrace{P_{j|i} H_j^T}_{\text{position-to-content}} + \underbrace{P_{i|j} P_{j|i}^T}_{\text{(отбрасывается)}}$$

где $H$ --- content embeddings, $P$ --- relative position embeddings. Это позволяет модели лучше разделять «что» и «где» --- что привело к SOTA на SuperGLUE (89.9 > human 89.8).

## Multi-Head Self-Attention

На практике используется не одна, а несколько параллельных голов self-attention (см. [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism#Multi-Head Attention|Multi-Head Attention]]). Каждая голова работает в своём подпространстве:

$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \cdot W^O$$

В оригинальном Transformer: $h = 8$, $d_k = d_v = d/h = 64$. Разные головы специализируются: одна отслеживает синтаксис, другая --- кореференцию, третья --- позиционные паттерны.

## Историческое значение

| Год | Событие |
|-----|---------|
| 2016 | Self-attention внутри LSTM (Cheng et al.) --- первое использование intra-attention |
| **2017** | **Transformer** --- self-attention как **единственный** механизм (Vaswani et al.) |
| 2018 | BERT --- bidirectional self-attention + MLM = прорыв в NLU |
| 2018 | GPT --- causal self-attention + language modeling = прорыв в генерации |
| 2020 | DeBERTa --- disentangled self-attention, human-level SuperGLUE |
| 2022 | Flash Attention --- IO-efficient self-attention без изменения математики |
| 2023+ | SSM-альтернативы (Mamba, RWKV) --- попытки заменить квадратичный self-attention |

## Related concepts

- [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]] --- общий механизм, self-attention --- его частный случай
- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] --- архитектура, построенная на self-attention
- [[02 Areas/ML & DL/Concepts/NLP/Positional Encoding|Positional Encoding]] --- без него self-attention инвариантен к порядку
- [[02 Areas/ML & DL/Concepts/NLP/Layer Normalization|Layer Normalization]] --- стабилизация в каждом блоке
- [[02 Areas/ML & DL/Concepts/Inference/Flash Attention|Flash Attention]] --- эффективная реализация
- [[02 Areas/ML & DL/Concepts/Inference/KV-Cache|KV-Cache]] --- кэширование K/V при авторегрессивном инференсе
- [[02 Areas/ML & DL/Concepts/Training/Disentangled Attention|Disentangled Attention]] --- разделение content и position

## Дополнительные ресурсы

- [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) --- пошаговые визуализации self-attention
- [Peter Bloem — Transformers from Scratch](https://peterbloem.nl/blog/transformers) --- реализация с нуля на PyTorch
- [Lilian Weng — Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) --- обзор вариантов attention
