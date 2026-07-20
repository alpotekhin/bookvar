---
title: "CS224N — Lecture 7: Attention and Machine Translation Evaluation"
course: "Stanford CS224N"
lecture: 7
type: source-note
status: legacy
raw: "[[02 Areas/ML & DL/raw/courses/Stanford CS224N/slides/cs224n-spr2024-lecture07-final-project]]"
concepts: ["[[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention]]", "[[02 Areas/ML & DL/Concepts/NLP/BLEU Score|BLEU]]", "[[02 Areas/ML & DL/Concepts/NLP/Seq2Seq|Seq2Seq]]"]
---

# Lecture 7: Attention and Machine Translation Evaluation

> *"Attention provides a solution to the bottleneck problem."* -- Christopher Manning

Лектор: Christopher Manning.

## Оценка Machine Translation: BLEU

### BLEU (Bilingual Evaluation Understudy, Papineni et al. 2002)

Сравнивает машинный перевод с одним или несколькими reference-переводами:

$$\text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

где:
- $p_n$ -- **modified n-gram precision** (1-, 2-, 3-, 4-grams)
- $w_n = 1/N$ (обычно $N=4$)
- $BP$ -- **brevity penalty** для слишком коротких переводов: $BP = \min(1, e^{1 - r/c})$

### Проблемы BLEU

- Много **допустимых переводов** одного предложения -- хороший перевод может получить низкий BLEU из-за перефразирования
- Обычно только один reference -- результаты "in expectation"
- Не учитывает семантическое сходство (только n-gram overlap)

### Прогресс в Machine Translation

Neural MT радикально превзошёл phrase-based SMT с 2016 года:

| Год | Система | BLEU (En-De, newstest) |
|-----|---------|----------------------|
| 2013 | Phrase-based SMT | ~20 |
| 2015 | Early NMT (U. Montreal) | ~22 |
| 2017 | NMT + Attention | ~28 |
| 2019 | NMT (FAIR) | ~42 |

## Seq2Seq Bottleneck

### Проблема

В стандартной [[02 Areas/ML & DL/Concepts/NLP/Seq2Seq|Seq2Seq]] архитектуре (Sutskever et al. 2014) **одного вектора** (последний hidden state encoder-а) должно быть достаточно, чтобы передать **всю информацию** об исходном предложении. Для длинных предложений это -- **информационное горлышко** (bottleneck).

```
Encoder RNN:  il  a  m'  entarté    → [единственный вектор h_final]
                                           ↓
Decoder RNN:                         <START> → he → hit → me → ...
```

## [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]] (Bahdanau et al. 2015)

### Ключевая идея

На каждом шаге decoder-а использовать **прямое соединение** с encoder-ом, чтобы **фокусироваться** на конкретной части исходной последовательности.

### Пошаговый алгоритм

Encoder hidden states: $h_1, h_2, \ldots, h_N$. На шаге $t$ decoder имеет state $s_t$.

**Шаг 1: Attention scores** -- вычислить совместимость decoder state с каждым encoder state:

$$e_t^i = s_t^T h_i \quad \forall i = 1, \ldots, N$$

**Шаг 2: Attention distribution** -- softmax для получения вероятностей:

$$\alpha_t = \text{softmax}(e_t) \in \mathbb{R}^N$$

Сумма $= 1$. Каждый $\alpha_t^i$ показывает, насколько decoder "обращает внимание" на позицию $i$.

**Шаг 3: Attention output** -- взвешенная сумма encoder states:

$$a_t = \sum_{i=1}^{N} \alpha_t^i h_i$$

Attention output $a_t$ содержит в основном информацию от тех encoder states, которые получили **высокий вес**.

**Шаг 4: Конкатенация** -- объединяем attention output с decoder state:

$$[a_t; s_t] \to \hat{y}_t$$

Иногда $a_t$ передаётся также как вход decoder-а на следующем шаге (feed-through).

### Пять преимуществ Attention

1. **Решает bottleneck**: decoder смотрит напрямую на encoder states, а не через один вектор
2. **Gradient shortcut**: gradient проходит напрямую от decoder к encoder, минуя длинные цепочки
3. **Интерпретируемость**: attention distribution показывает alignment -- какое исходное слово соответствует какому выходному
4. **Soft alignment for free**: модель **самостоятельно** обучается alignment, без явной supervision
5. **Более человечный** процесс: переводчик тоже "смотрит назад" на исходное предложение

### Визуализация alignment

Attention weights формируют матрицу alignment:

```
           il    a    m'   entarté
he        [0.9  0.0  0.0   0.1]     ← фокус на "il"
hit       [0.0  0.1  0.0   0.9]     ← фокус на "entarté"
me        [0.0  0.0  0.9   0.1]     ← фокус на "m'"
with      [0.1  0.0  0.1   0.8]
a         [0.1  0.0  0.1   0.8]
pie       [0.0  0.0  0.0   1.0]     ← фокус на "entarté"
```

## Варианты Attention

### Общая формула

Для values $h_1, \ldots, h_N$ и query $s_t$:

1. Вычислить **attention scores** $e \in \mathbb{R}^N$
2. **Softmax**: $\alpha = \text{softmax}(e)$
3. **Weighted sum**: $a = \sum_i \alpha_i h_i$

Разница -- в способе вычисления scores $e$:

### Dot-product attention

$$e_t^i = s_t^T h_i$$

Предполагает $\dim(s_t) = \dim(h_i)$. Простейший и самый быстрый вариант.

### Multiplicative (Bilinear) attention (Luong et al. 2015)

$$e_t^i = s_t^T W h_i$$

Матрица $W \in \mathbb{R}^{d_s \times d_h}$ позволяет разные размерности $s_t$ и $h_i$. Более выразительный, но больше параметров.

### Additive attention (Bahdanau et al. 2015)

$$e_t^i = v^T \tanh(W_1 s_t + W_2 h_i)$$

Два отдельных линейных преобразования + нелинейность. Оригинальный вариант из первой attention-статьи. Может быть лучше при большой разнице размерностей.

### Scaled dot-product attention (Vaswani et al. 2017)

$$e_t^i = \frac{s_t^T h_i}{\sqrt{d_k}}$$

Масштабирование предотвращает насыщение softmax -- ключевой компонент [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]].

## Attention как общий механизм

Attention -- **не привязан** к seq2seq. Это общий принцип: **взвешенная агрегация информации**, где веса зависят от контекста. В следующей лекции (Transformers) -- **self-attention**: query, key, value из одной и той же последовательности.

Attention стал **одним из фундаментальных примитивов** глубокого обучения:
- Seq2seq NMT (Bahdanau 2015, Luong 2015)
- [[02 Areas/ML & DL/Concepts/NLP/Self-Attention|Self-Attention]] в Transformer (Vaswani 2017)
- Cross-attention в multimodal моделях
- Attention в Graph Neural Networks
- Attention в Reinforcement Learning (Decision Transformer)

## Concepts covered

- [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention]] -- прямое соединение decoder с encoder states, взвешенная агрегация
- [[02 Areas/ML & DL/Concepts/NLP/BLEU Score|BLEU]] -- n-gram precision + brevity penalty, стандартная метрика MT
- [[02 Areas/ML & DL/Concepts/NLP/Seq2Seq|Seq2Seq]] -- encoder-decoder архитектура и её bottleneck
