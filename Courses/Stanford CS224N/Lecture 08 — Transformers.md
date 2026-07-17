---
title: "CS224N — Lecture 8: Transformers"
course: "Stanford CS224N"
lecture: 8
type: source-note
status: legacy
raw: "[[02 Areas/ML & DL/raw/courses/Stanford CS224N/slides/cs224n-spr2024-lecture08-transformers]]"
concepts: ["[[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]]", "[[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]]", "[[02 Areas/ML & DL/Concepts/NLP/Self-Attention|Self-Attention]]", "[[02 Areas/ML & DL/Concepts/NLP/Multi-Head Attention|Multi-Head Attention]]", "[[02 Areas/ML & DL/Concepts/NLP/Positional Encoding|Positional Encoding]]", "[[02 Areas/ML & DL/Concepts/Training/Scaling Laws|Scaling Laws]]"]
---

# Lecture 8: Transformers

> *"By the end of this lecture, you will deeply understand the neural architecture that underpins virtually every state-of-the-art NLP model today!"* -- Anna Goldie

Лектор: Anna Goldie. Adapted from slides by Anna Goldie, John Hewitt.

## Влияние Transformers

### Революция в NLP и за его пределами

Transformer (Vaswani et al., 2017, "Attention Is All You Need") -- единая архитектура, стоящая за:

**NLP**: Machine Translation (WMT 2014: BLEU 28.4 EN-DE, 41.8 EN-FR -- SOTA при 100x меньшем compute), SuperGLUE (Vega v2, ST-MoE-32B, Turing NLR v5 -- все Transformer-based), LLM (GPT-4-Turbo, Claude 3 Opus, Gemini, Llama 3 -- все на LMSYS Arena Leaderboard)

**За пределами NLP**:
- **Protein Folding**: AlphaFold2 (Jumper et al., 2021, Nature)
- **Image Classification**: Vision Transformer, ViT (Dosovitskiy et al., 2020) -- outperforms ResNet при меньшем compute
- **ML for Systems**: GO-one (Zhou et al., 2020) -- Transformer-based compiler ускоряет другой Transformer

### Scaling Laws (Kaplan et al., 2020)

Loss уменьшается как power-law от compute, data, parameters:

$$L = (C_{\min} / 2.3 \cdot 10^8)^{-0.050}$$

Наблюдалось на **многих порядках величин** без замедления. Это мотивировало масштабирование: GPT-2 (1.5B) -> GPT-3 (175B) -> GPT-4 (~1.8T MoE).

## Мотивация: три desiderata для новой архитектуры

Vaswani et al. ставили три цели при проектировании Transformer:

### 1. Минимизировать computational complexity per layer

Из оригинальной статьи (Table 1):

| Layer Type | Complexity/Layer | Sequential Ops | Max Path Length |
|-----------|-----------------|----------------|-----------------|
| **Self-Attention** | $O(n^2 \cdot d)$ | $O(1)$ | **$O(1)$** |
| Recurrent | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k n)$ |

Когда $n \ll d$ (что типично: $n = 512$, $d = 512$), self-attention **дешевле** RNN per layer.

### 2. Минимизировать interaction distance (path length)

В [[02 Areas/ML & DL/Concepts/NLP/RNN/RNN|RNN]] информация между позицией 1 и позицией $n$ проходит через $O(n)$ шагов. Каждый шаг -- potential loss информации (vanishing gradients).

Пример из слайдов: *"The **chef** who ... **ate**"* -- между "chef" и "ate" может быть длинная придаточная конструкция. В BiLSTM информация о "chef" проходит O(sequence length) слоёв, прежде чем влияет на "ate".

В Self-Attention: **$O(1)$ path length**. Любые две позиции связаны **напрямую** через одну attention operation.

### 3. Максимизировать параллелизм

RNN: $h_t$ зависит от $h_{t-1}$ -> строго последовательно, $O(n)$ sequential operations. Self-Attention: все позиции обрабатываются **одновременно**, $O(1)$ sequential operations. Это позволило масштабировать обучение на кластерах GPU.

## Self-Attention: пошаговый разбор

### Создание Q, K, V

Каждый элемент последовательности $x_i$ проецируется в три вектора:

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

Интуиция (из слайдов + Transformer Explainer, Georgia Tech):
- **Query** -- "что я ищу?" (поисковый запрос)
- **Key** -- "что я содержу?" (заголовок страницы)
- **Value** -- "какую информацию я несу?" (содержимое страницы)

### Вычисление Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**Шаг за шагом** (по Jay Alammar):
1. **Score** = dot-product $Q_i \cdot K_j$ для всех пар позиций
2. **Scale** = делим на $\sqrt{d_k}$ (стабилизация градиентов -- без масштабирования softmax уходит в насыщение при больших $d_k$)
3. **Softmax** = нормализуем скоры в вероятности (сумма = 1 по каждой строке)
4. **Weighted sum** = умножаем каждый $V_j$ на его softmax-вес и суммируем

Результат: для каждой позиции -- **взвешенная комбинация** всех Value-векторов, где веса определяются совместимостью Query и Key.

### Почему /sqrt(d_k)?

Если компоненты $q$ и $k$ -- i.i.d. с нулевым средним и единичной дисперсией, то $q \cdot k$ имеет дисперсию $d_k$. При больших $d_k$ (напр. 64) значения dot-product растут, softmax даёт **почти one-hot** распределение, градиенты затухают. Деление на $\sqrt{d_k}$ возвращает дисперсию к 1.

## [[02 Areas/ML & DL/Concepts/NLP/Multi-Head Attention|Multi-Head Attention]]

Одна голова attention усредняет информацию и может упустить разные типы зависимостей. Решение -- $h$ параллельных голов с **разными** проекциями:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

В оригинальном Transformer: $h = 8$, $d_k = d_v = d_{\text{model}} / h = 512 / 8 = 64$. Итоговая стоимость примерно равна одной full-dimensional head.

**Зачем**: разные головы специализируются на разных задачах (анафора, синтаксис, позиционные паттерны -- визуализации из Appendix оригинальной статьи).

## Полная архитектура Transformer

### Encoder Block

```
Input Embedding + Positional Encoding
    |
[Multi-Head Self-Attention] --> Add & LayerNorm
    |
[Feed-Forward Network]      --> Add & LayerNorm
    |
Repeat 6x
```

### Decoder Block

```
Output Embedding + Positional Encoding
    |
[Masked Multi-Head Self-Attention] --> Add & LayerNorm
    |
[Multi-Head Cross-Attention]       --> Add & LayerNorm
    |  (Q from decoder, K/V from encoder)
[Feed-Forward Network]             --> Add & LayerNorm
    |
Repeat 6x
    |
Linear --> Softmax --> Output Probabilities
```

### Три вида attention

1. **Encoder self-attention**: bidirectional, каждый токен видит все
2. **Decoder masked self-attention**: causal mask ($-\infty$ перед softmax для будущих позиций) -- авторегрессивность
3. **Cross-attention**: Query из декодера, Key/Value из энкодера -- декодер "смотрит" на вход

### Residual Connections + Layer Norm

Каждый подслой обёрнут в:

$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

**Residual connections** -- gradient highway, позволяют обучать глубокие стеки (6+ слоёв). **[[02 Areas/ML & DL/Concepts/NLP/Layer Normalization|Layer Normalization]]** -- стабилизирует обучение, нормализует по feature dimension.

### Feed-Forward Network (FFN)

Два линейных слоя с ReLU:

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

$d_{ff} = 2048$ (4x от $d_{\text{model}} = 512$). FFN применяется **позиционно** (одинаково к каждому токену). Часто интерпретируется как "memory" -- хранилище фактических знаний.

### [[02 Areas/ML & DL/Concepts/NLP/Positional Encoding|Positional Encoding]]

Self-attention **инвариантен к порядку**: permutation equivariant. Без positional encoding "The dog bit the man" = "The man bit the dog".

Оригинальный Transformer: синусоидальные encoding:

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{\text{model}}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}})$$

Альтернатива: **learned positional embeddings** (GPT, BERT). Современные: **RoPE** (Rotary Position Embedding), **ALiBi**.

## Результаты

### Machine Translation (WMT 2014)

| Model | BLEU EN-DE | BLEU EN-FR | Training Cost (FLOPs) |
|-------|-----------|-----------|----------------------|
| GNMT + RL Ensemble | 26.30 | 41.16 | $1.8 \cdot 10^{20}$ |
| ConvS2S Ensemble | 26.36 | **41.29** | $1.2 \cdot 10^{21}$ |
| **Transformer (big)** | **28.4** | **41.8** | $2.3 \cdot 10^{19}$ |

Transformer: **лучший результат** при **в 10-100 раз меньшем compute**.

## Drawbacks и варианты

### Квадратичная сложность

$O(n^2)$ по длине последовательности -- attention matrix $n \times n$. При $n > 10000$ это проблема. Решения:
- [[02 Areas/ML & DL/Concepts/Inference/Flash Attention|Flash Attention]] -- IO-aware implementation, не меняет математику
- Sparse attention (BigBird, Longformer) -- attention к подмножеству позиций
- [[02 Areas/ML & DL/Concepts/Architectures/Mamba|Mamba]], [[02 Areas/ML & DL/Concepts/Architectures/RWKV|RWKV]] -- линейные альтернативы (SSM)

### Нет inductive bias для locality

RNN имеет bias к соседним токенам. Transformer обрабатывает все пары одинаково -- нужно **больше данных** для обучения.

## Concepts covered

- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] -- "Attention Is All You Need" (Vaswani et al., 2017)
- [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]] -- Q, K, V, scaled dot-product
- [[02 Areas/ML & DL/Concepts/NLP/Self-Attention|Self-Attention]] -- attention внутри одной последовательности
- [[02 Areas/ML & DL/Concepts/NLP/Multi-Head Attention|Multi-Head Attention]] -- параллельные головы
- [[02 Areas/ML & DL/Concepts/NLP/Positional Encoding|Positional Encoding]] -- синусоидальные или learned
- [[02 Areas/ML & DL/Concepts/Training/Scaling Laws|Scaling Laws]] -- power-law от compute/data/params
