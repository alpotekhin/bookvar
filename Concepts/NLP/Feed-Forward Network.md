---
title: "Feed-Forward Network"
aliases: [FFN, MLP sublayer, position-wise FFN, SwiGLU, GELU, GLU variants, feed-forward sublayer]
type: concept
category: NLP
papers:
  - "[[02 Areas/ML & DL/Papers/Attention Is All You Need|Attention Is All You Need]]"
  - "[[02 Areas/ML & DL/Papers/Formal Algorithms for Transformers|Formal Algorithms for Transformers]]"
  - "[[02 Areas/ML & DL/Papers/BERT|BERT]]"
  - "[[02 Areas/ML & DL/Papers/LLaMA|LLaMA]]"
  - "[[02 Areas/ML & DL/Papers/Mistral 7B|Mistral 7B]]"
courses: []
sources:
  - "[Shazeer 2020 — GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)"
  - "[Geva et al. 2021 — Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913)"
  - "[Dai et al. 2022 — Knowledge Neurons in Pretrained Transformers](https://arxiv.org/abs/2104.08696)"
  - "[Naoki Shibuya — SwiGLU Explained](https://naokishibuya.github.io/blog/2023-04-30-swiglu-2020/index.html)"
---

# Feed-Forward Network (FFN)

## Зачем это нужно: вычисления после маршрутизации

Каждый Transformer-блок содержит два подслоя: **Attention** и **FFN**. Если attention — это механизм маршрутизации («какая информация откуда нужна»), то FFN — это **вычислитель**, который обрабатывает собранную информацию. Причём FFN содержит **~2/3 всех параметров модели** — это основное «хранилище знаний» в Transformer.

## Стандартный FFN (Vaswani et al., 2017)

### Формула

$$\text{FFN}(x) = W_2 \cdot \sigma(W_1 x + b_1) + b_2$$

где:
- $x \in \mathbb{R}^{d_{\text{model}}}$ — вход (эмбеддинг одного токена)
- $W_1 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}$ — первая линейная проекция (expansion)
- $W_2 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}$ — вторая линейная проекция (contraction)
- $\sigma$ — нелинейная активация (ReLU в оригинале)
- $d_{ff} = 4 \times d_{\text{model}}$ — **expansion ratio 4x**

### Position-wise применение

FFN применяется **независимо к каждой позиции** — одинаковые параметры $W_1$, $W_2$ для всех токенов в последовательности. Это эквивалентно двум 1x1 свёрткам (kernel=1) с нелинейностью между ними.

**Зачем expansion 4x?** Первый слой расширяет $d_{\text{model}}$ в 4 раза, создавая overcomplete representation, в которой модель может выделить более тонкие паттерны. Второй слой сжимает обратно в $d_{\text{model}}$.

Для оригинального Transformer: $d_{\text{model}} = 512$, $d_{ff} = 2048$. Для BERT-base: $d_{\text{model}} = 768$, $d_{ff} = 3072$.

## Эволюция активаций: от ReLU к SwiGLU

### ReLU (Original Transformer, 2017)

$$\text{ReLU}(x) = \max(0, x)$$

Простая и быстрая, но имеет проблему «мёртвых нейронов» (dying ReLU): если нейрон получает отрицательный вход, градиент = 0, нейрон перестаёт обновляться. В overcomplete FFN (4x expansion) значительная часть нейронов может «умереть».

### GELU (BERT, GPT-2, 2018)

$$\text{GELU}(x) = x \cdot \Phi(x) \approx x \cdot \sigma(1.702x)$$

где $\Phi(x)$ — CDF стандартного нормального распределения. GELU — гладкая аппроксимация ReLU, которая позволяет маленьким отрицательным значениям «просачиваться» (в отличие от жёсткого обрезания в ReLU).

**Результат**: GELU > ReLU на большинстве NLP-бенчмарков. Стала стандартом для BERT-семейства моделей.

### Gated Linear Units (GLU) и их варианты

#### Идея GLU (Dauphin et al., 2016)

Вместо одной проекции + активации, используются **две** параллельные проекции, одна из которых служит **вентилем** (gate):

$$\text{GLU}(x) = \sigma(xW + b) \otimes (xV + c)$$

где $\sigma$ — sigmoid, $\otimes$ — поэлементное умножение. Sigmoid-ветка решает, **какую информацию пропустить**, по аналогии с gate-механизмами в LSTM.

#### Варианты GLU (Shazeer, 2020)

Noam Shazeer (2020) — *"GLU Variants Improve Transformer"* — систематически исследовал замену sigmoid на другие активации:

| Вариант | Формула | Активация |
|---------|---------|-----------|
| **GLU** | $\sigma(xW) \otimes xV$ | Sigmoid |
| **ReGLU** | $\text{ReLU}(xW) \otimes xV$ | ReLU |
| **GEGLU** | $\text{GELU}(xW) \otimes xV$ | GELU |
| **SwiGLU** | $\text{Swish}_1(xW) \otimes xV$ | Swish |
| Bilinear | $xW \otimes xV$ | Нет (линейная) |

#### Swish activation

$$\text{Swish}_\beta(x) = x \cdot \sigma(\beta x)$$

При $\beta = 1$: $\text{Swish}(x) = x \cdot \sigma(x)$. Гладкая, self-gated функция, обобщающая ReLU (при $\beta \to \infty$ → ReLU) и линейную функцию (при $\beta = 0$ → $x/2$).

## SwiGLU FFN (LLaMA, Mistral, PaLM)

### Полная формула

$$\text{FFN}_{\text{SwiGLU}}(x) = W_2 \cdot \left(\text{Swish}(xW_1) \otimes xV\right)$$

Из [[02 Areas/ML & DL/Papers/LLaMA|LLaMA]] §2.2 (Shazeer, 2020):

```python
def ffn_swiglu(x, W1, V, W2):
    return W2 @ (swish(x @ W1) * (x @ V))
```

### Три весовые матрицы вместо двух

Стандартный FFN имеет две матрицы: $W_1$ (up-projection) и $W_2$ (down-projection).

SwiGLU FFN имеет **три** матрицы: $W_1$ (gate projection), $V$ (up projection), $W_2$ (down projection). Это увеличивает количество параметров на 50%.

### Компенсация: hidden dimension 2/3

Чтобы сохранить примерно тот же FLOP count, скрытую размерность уменьшают:

$$d_{ff}^{\text{SwiGLU}} = \frac{2}{3} \times 4 \times d_{\text{model}} \approx 2.67 \times d_{\text{model}}$$

Конкретный пример для LLaMA-7B:
- $d_{\text{model}} = 4096$
- $d_{ff} = 4096 \times 4 \times \frac{2}{3} \approx 11008$
- Три матрицы: $W_1, V \in \mathbb{R}^{11008 \times 4096}$, $W_2 \in \mathbb{R}^{4096 \times 11008}$

### Результаты

Из экспериментов Shazeer (2020) на T5-base:

| Активация | Perplexity (pre-training) | GLUE | SQuAD |
|-----------|--------------------------|------|-------|
| ReLU (baseline) | baseline | baseline | baseline |
| GELU | −0.3 | +0.2 | +0.3 |
| **SwiGLU** | **−0.6** | **+0.5** | **+0.8** |
| GEGLU | −0.5 | +0.4 | +0.6 |

SwiGLU стабильно лучше. Автор честно пишет: *"We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence."*

## FFN как хранилище знаний

### «Key-Value Memories» (Geva et al., 2021)

Geva, Schuster et al. — *"Transformer Feed-Forward Layers Are Key-Value Memories"* — показали, что FFN можно интерпретировать как **ассоциативную память**:

- **$W_1$ (first layer) = keys**: каждая строка $W_1$ — «ключ», детектирующий определённый паттерн во входе
- **$W_2$ (second layer) = values**: соответствующая строка $W_2$ — «значение», содержащее ассоциацию

Процесс:
1. Вход $x$ сравнивается со всеми ключами через $W_1 x$ (dot-product)
2. Активация отбирает наиболее релевантные ключи
3. Соответствующие значения суммируются через $W_2$

### «Knowledge Neurons» (Dai et al., 2022)

Развитие идеи: определённые нейроны FFN хранят **конкретные факты**. Например:
- Некоторые нейроны активируются на «Italy» и предсказывают «Rome» (столица)
- Другие нейроны активируются на «Einstein» и предсказывают «physics»

**Распределение по слоям**: factual knowledge сосредоточены преимущественно в верхних слоях модели.

### Практические следствия

1. **Knowledge editing** (Meng et al., 2022 — ROME): можно редактировать конкретные факты, модифицируя определённые FFN-нейроны, не переобучая модель
2. **Attention vs FFN**: attention = маршрутизация/retrieval (какие токены важны), FFN = storage (что модель знает)
3. **RAG**: Retrieval-Augmented Generation частично заменяет функцию FFN как knowledge store, доставляя знания из внешних источников

## Количество параметров: почему FFN доминирует

FFN содержит примерно **2/3 всех параметров** модели:

### Стандартный FFN (2 матрицы)

$$\text{Params}_{\text{FFN}} = 2 \times d_{\text{model}} \times d_{ff} = 2 \times d \times 4d = 8d^2$$

### SwiGLU FFN (3 матрицы)

$$\text{Params}_{\text{FFN}} = 3 \times d_{\text{model}} \times d_{ff}^{\prime} \approx 3 \times d \times 2.67d = 8d^2$$

Примерно то же количество параметров за счёт уменьшения $d_{ff}$.

### Сравнение с Attention

Attention: $4 \times d^2$ (четыре проекции: Q, K, V, O).
FFN: $8d^2$.

Итого: FFN = 2x параметров Attention. В одном Transformer-блоке = $12d^2$ параметров.

### Конкретные числа

| Модель | $d_{\text{model}}$ | $d_{ff}$ | Params FFN/layer | Layers | Total FFN |
|--------|---------------------|----------|-----------------|--------|-----------|
| BERT-base | 768 | 3,072 | 4.7M | 12 | 56M |
| BERT-large | 1,024 | 4,096 | 8.4M | 24 | 201M |
| GPT-3 175B | 12,288 | 49,152 | ~1.2B | 96 | ~115B |
| LLaMA-7B | 4,096 | 11,008 | ~135M | 32 | ~4.3B |

## Связь FFN с Mixture of Experts (MoE)

В MoE-архитектурах (Mixtral, Switch Transformer) FFN-слой заменяется на **набор FFN-экспертов** с router, который выбирает top-$k$ экспертов для каждого токена.

Mixtral 8x7B:
- 8 экспертов (каждый = SwiGLU FFN)
- Router выбирает top-2 для каждого токена
- Всего 47B параметров, но только ~13B активны на каждый токен

Это подтверждает роль FFN как key-value memory: разные эксперты специализируются на разных типах знаний, и router «достаёт» нужную информацию для конкретного токена.

## Хронология

| Год | Milestone | FFN-вариант |
|-----|-----------|-------------|
| 2016 | Gated Linear Units | GLU (sigmoid gate) |
| 2017 | Original Transformer | ReLU FFN, $d_{ff} = 4d$ |
| 2018 | BERT | GELU FFN |
| 2019 | GPT-2 | GELU FFN |
| 2020 | GLU Variants (Shazeer) | SwiGLU, GEGLU, ReGLU |
| 2021 | Key-Value Memories (Geva) | FFN = ассоциативная память |
| 2022 | Knowledge Neurons (Dai) | Конкретные нейроны = факты |
| 2022 | ROME (Meng) | Editing facts через FFN |
| 2023 | **LLaMA** | SwiGLU + 2/3 × 4d = стандарт |
| 2023 | Mixtral | 8 SwiGLU-экспертов + router |

## Key papers

- [[02 Areas/ML & DL/Papers/Attention Is All You Need|Attention Is All You Need]] — оригинальный position-wise FFN с ReLU
- [[02 Areas/ML & DL/Papers/Formal Algorithms for Transformers|Formal Algorithms for Transformers]] — формализация FFN
- Shazeer 2020 — *GLU Variants Improve Transformer* — SwiGLU и другие GLU-варианты
- Geva et al. 2021 — *Transformer Feed-Forward Layers Are Key-Value Memories* — интерпретация FFN
- Dai et al. 2022 — *Knowledge Neurons in Pretrained Transformers*
- [[02 Areas/ML & DL/Papers/LLaMA|LLaMA]] — SwiGLU + 2/3 × 4d

## Related concepts

- [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]] — attention = routing, FFN = processing
- [[02 Areas/ML & DL/Concepts/NLP/Layer Normalization|Layer Normalization]] — нормализация перед/после FFN
- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] — FFN = основной подслой
- [[02 Areas/ML & DL/Concepts/Architectures/LLaMA|LLaMA]] — популяризировал SwiGLU FFN
- [[02 Areas/ML & DL/Concepts/Architectures/Mixtral|Mixtral]] — MoE = N x FFN + router

## Дополнительные ресурсы

- [Naoki Shibuya — SwiGLU Explained](https://naokishibuya.github.io/blog/2023-04-30-swiglu-2020/index.html) — подробный разбор GLU вариантов
- [labml.ai — Gated Linear Units and Variants](https://nn.labml.ai/transformers/glu_variants/simple.html) — реализация с кодом
- [Shazeer 2020 — GLU Variants Improve Transformer (paper)](https://arxiv.org/abs/2002.05202) — оригинальная статья
- [Geva et al. — FFN as Key-Value Memories](https://arxiv.org/abs/2012.14913) — интерпретация FFN как памяти
