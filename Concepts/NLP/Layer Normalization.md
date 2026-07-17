---
title: "Layer Normalization"
aliases: [LayerNorm, layer norm, RMSNorm, Pre-Norm, Post-Norm, batch normalization vs layer normalization]
type: concept
status: legacy
category: NLP
papers:
  - "[[02 Areas/ML & DL/Papers/Attention Is All You Need|Attention Is All You Need]]"
  - "[[02 Areas/ML & DL/Papers/Formal Algorithms for Transformers|Formal Algorithms for Transformers]]"
  - "[[02 Areas/ML & DL/Papers/BERT|BERT]]"
  - "[[02 Areas/ML & DL/Papers/LLaMA|LLaMA]]"
  - "[[02 Areas/ML & DL/Papers/Mistral 7B|Mistral 7B]]"
courses: []
sources:
  - "[Ba et al. 2016 — Layer Normalization (original paper)](https://arxiv.org/abs/1607.06450)"
  - "[Zhang & Sennrich 2019 — Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)"
  - "[Xiong et al. 2020 — On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)"
  - "[Pinecone — Batch and Layer Normalization](https://www.pinecone.io/learn/batch-layer-normalization/)"
  - "[Sebastian Raschka — RMSNorm vs LayerNorm](https://sebastianraschka.com/faq/docs/rmsnorm-vs-layernorm.html)"
---

# Layer Normalization

## Зачем это нужно: стабильность обучения глубоких сетей

Глубокие нейронные сети страдают от **internal covariate shift** — распределение активаций на входе каждого слоя меняется в процессе обучения, потому что обновляются параметры предыдущих слоёв. Это приводит к нестабильности: градиенты взрываются или затухают, модель расходится или учится крайне медленно.

Нормализация решает проблему: на каждом шаге приводит активации к стабильному распределению (нулевое среднее, единичная дисперсия), а затем масштабирует и сдвигает обучаемыми параметрами.

## Почему BatchNorm не работает для Transformer

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/layer-normalization/batch-normalization.png]]
*Batch Normalization: статистики вычисляются по batch dimension для каждого feature (источник: Pinecone)*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/layer-normalization/layer-normalization.png]]
*Layer Normalization: статистики вычисляются по feature dimension для каждого примера — не зависит от batch size (источник: Pinecone)*

**Batch Normalization** (Ioffe & Szegedy, 2015) нормализует по **batch dimension** — для каждого feature вычисляет среднее и дисперсию по всем примерам в mini-batch.

Это создаёт три критические проблемы для NLP:

### 1. Переменная длина последовательностей

В одном batch предложения имеют разную длину. Позиция 50 может присутствовать в 3 из 32 примеров — статистика на таких маленьких выборках нестабильна. Padding-токены вносят лишние нули, искажая нормализацию.

### 2. Зависимость от batch size

BatchNorm работает хорошо при batch_size = 32-256. При обучении больших LLM используется gradient accumulation с эффективным batch_size = 1-4 на GPU. На таких выборках оценки mean и variance шумные.

### 3. Разные режимы train/inference

BatchNorm использует running statistics при инференсе (не текущий batch). Для последовательностей это ведёт к мismatch: при обучении модель видела одно распределение, при инференсе с одним примером — другое.

**Layer Normalization** решает все три проблемы: нормализация по **feature dimension** каждого отдельного примера. Нет зависимости от batch, от длины соседних примеров, и train/inference работают одинаково.

## Формула LayerNorm

Для вектора активаций $\mathbf{x} \in \mathbb{R}^{d}$ (один токен на одном слое):

$$\mu = \frac{1}{d}\sum_{i=1}^{d} x_i$$

$$\sigma^2 = \frac{1}{d}\sum_{i=1}^{d} (x_i - \mu)^2$$

$$\text{LayerNorm}(\mathbf{x}) = \gamma \cdot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

где:
- $\mu$ — среднее по feature dimension
- $\sigma^2$ — дисперсия по feature dimension
- $\gamma, \beta \in \mathbb{R}^{d}$ — обучаемые параметры scale и shift
- $\epsilon$ — маленькая константа для численной стабильности (обычно $10^{-5}$ или $10^{-6}$)

Из [[02 Areas/ML & DL/Papers/Formal Algorithms for Transformers|Formal Algorithms for Transformers]] (Algorithm 6):

```
Input: e ∈ ℝ^(d_e)
Parameters: γ, β ∈ ℝ^(d_e)

m ← mean(e)           # среднее по feature dimension
v ← var(e)            # дисперсия
ê = (e - m) / √v · γ + β  # normalize + scale + shift
```

**Ключевое**: нормализация по feature dimension $d_e$ = 768 (BERT-base) или 4096 (LLaMA-7B). Каждый токен нормализуется **независимо** от других токенов и от других примеров в batch.

## BatchNorm vs LayerNorm: визуальное сравнение

Представь тензор активаций размером `[batch, seq_len, d_model]`:

| Метод | Нормализация по | Зависит от batch | Зависит от seq_len |
|-------|----------------|------------------|--------------------|
| **BatchNorm** | dim 0 (batch) для каждого feature | **Да** | **Да** |
| **LayerNorm** | dim 2 (features) для каждого токена | **Нет** | **Нет** |

BatchNorm: для feature $j$ считаем $\mu_j$ и $\sigma_j$ по всем примерам и позициям.
LayerNorm: для позиции $(b, t)$ считаем $\mu$ и $\sigma$ по всем features.

## Pre-Norm vs Post-Norm: где именно ставить LayerNorm

Расположение LayerNorm внутри Transformer-блока критически влияет на стабильность обучения. Существуют два подхода:

### Post-Norm (Original Transformer, BERT)

```
x → Sublayer(x) + x → LayerNorm(·)
```

$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

Нормализация **после** residual connection. Использовался в оригинальном Transformer (2017) и BERT (2018).

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/transformer_resideual_layer_norm_2.png]]
*Residual connection + LayerNorm (Post-Norm) внутри каждого блока (источник: Jay Alammar)*

### Pre-Norm (GPT-2, LLaMA, все современные LLM)

```
x → LayerNorm(x) → Sublayer(·) + x
```

$$\text{output} = x + \text{Sublayer}(\text{LayerNorm}(x))$$

Нормализация **перед** sublayer. Residual path остаётся «чистым» — градиенты протекают через identity connection без каких-либо трансформаций.

### Почему Pre-Norm стал стандартом

**Xiong et al. (2020)** — "On Layer Normalization in the Transformer Architecture" — показали теоретически и эмпирически:

1. **Gradient flow**: в Post-Norm при глубине $L$ слоёв градиенты проходят через $L$ операций LayerNorm. Каждая нормализация может уменьшать градиент, приводя к vanishing gradients. В Pre-Norm residual path свободен от нормализаций — градиент проходит как по «шоссе».

2. **Warmup не нужен**: Post-Norm требует careful learning rate warmup (обычно 4000-10000 шагов), иначе обучение расходится. Pre-Norm стабилен **без warmup**.

3. **Масштабирование**: при стекировании 96 слоёв (GPT-3, 175B параметров) Post-Norm становится нетренируемым без сложной инженерии. Pre-Norm масштабируется до произвольной глубины.

**Практический результат**: Pre-Norm немного проигрывает Post-Norm в финальном качестве (при условии успешного обучения обоих), но значительно стабильнее и проще в тренировке. Для больших моделей это решающее преимущество.

### Полная схема Pre-Norm Transformer блока

```
Input: x

# Self-Attention sublayer
h = x + MultiHeadAttention(LayerNorm(x))

# FFN sublayer  
output = h + FFN(LayerNorm(h))
```

Из [[02 Areas/ML & DL/Papers/Formal Algorithms for Transformers|Formal Algorithms for Transformers]]: BERT использует Post-Norm; GPT использует Pre-Norm.

## RMSNorm: упрощение для масштабирования

### Мотивация

Zhang & Sennrich (2019) заметили: **центрирование (вычитание среднего) в LayerNorm не обязательно**. Основная польза нормализации — масштабирование, а не сдвиг. Можно убрать mean и bias, оставив только re-scaling.

### Формула

$$\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2}$$

$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} \cdot \gamma$$

Или эквивалентно:

$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma$$

### Чем отличается от LayerNorm

| Аспект | LayerNorm | RMSNorm |
|--------|-----------|---------|
| Вычитание среднего | Да ($x - \mu$) | **Нет** |
| Bias $\beta$ | Да | **Нет** |
| Scale $\gamma$ | Да | Да |
| Нормализующий фактор | $\sqrt{\sigma^2 + \epsilon}$ | $\sqrt{\text{mean}(x^2) + \epsilon}$ |
| Параметры | $2d$ ($\gamma$ + $\beta$) | $d$ (только $\gamma$) |

**Математический insight**: когда среднее $\mu = 0$, RMSNorm и LayerNorm дают идентичный результат. На практике $\mu$ часто близко к нулю, поэтому разница в качестве минимальна.

### Почему RMSNorm быстрее

1. **Одна редукция вместо двух**: LayerNorm вычисляет $\mu$ (одна редукция), затем $\sigma^2$ (вторая редукция). RMSNorm — одна редукция ($\text{mean}(x^2)$).
2. **Меньше параметров**: нет $\beta$, экономия памяти.
3. **Ускорение ~7-10%** на GPU (по данным Zhang & Sennrich). При 100B+ параметрах и триллионах токенов это экономит дни/недели обучения.

### Принятие в индустрии

Из [[02 Areas/ML & DL/Papers/LLaMA|LLaMA]] §2.2: RMSNorm + Pre-Norm — одно из трёх ключевых архитектурных решений LLaMA (наряду с SwiGLU и RoPE).

| Модель | Нормализация | Расположение |
|--------|-------------|-------------|
| Original Transformer | LayerNorm | Post-Norm |
| BERT | LayerNorm | Post-Norm |
| GPT-2 | LayerNorm | Pre-Norm |
| GPT-3 | LayerNorm | Pre-Norm |
| **LLaMA 1/2/3** | **RMSNorm** | **Pre-Norm** |
| **Mistral 7B** | **RMSNorm** | **Pre-Norm** |
| **Gemma** | **RMSNorm** | **Pre-Norm** |
| **Qwen** | **RMSNorm** | **Pre-Norm** |

RMSNorm + Pre-Norm = стандарт де-факто для всех открытых LLM с 2023 года.

## Где именно нормализация в Transformer

В каждом Transformer-блоке нормализация применяется дважды:

1. **Перед/после Self-Attention** (в зависимости от Pre/Post):
   - Нормализует входные эмбеддинги перед вычислением Q, K, V
   - Стабилизирует attention scores

2. **Перед/после FFN**:
   - Нормализует выход attention перед feed-forward
   - Без этого FFN (с expansion ratio 4x) может усиливать нестабильности

3. **Финальная нормализация** (Pre-Norm only):
   - Дополнительный LayerNorm/RMSNorm после последнего блока, перед output projection
   - GPT-2, LLaMA добавляют этот слой для стабилизации выхода стека

## Хронология нормализации в Transformer

| Год | Метод + Расположение | Модель | Значение |
|-----|---------------------|--------|----------|
| 2015 | BatchNorm | ResNet | Стандарт для CNN |
| 2016 | LayerNorm | Ba et al. | Нормализация без batch dependency |
| 2017 | LayerNorm + Post-Norm | Original Transformer | Первое применение в Transformer |
| 2018 | LayerNorm + Post-Norm | BERT | Массовое принятие |
| 2019 | LayerNorm + **Pre-Norm** | GPT-2 | Переход к Pre-Norm |
| 2019 | **RMSNorm** | Zhang & Sennrich | Упрощение LayerNorm |
| 2020 | Pre-Norm теоретический анализ | Xiong et al. | Доказательство стабильности |
| 2023 | **RMSNorm + Pre-Norm** | **LLaMA** | Индустриальный стандарт |
| 2025 | HybridNorm, Peri-LN | Research | Попытки объединить лучшее |

## Key papers

- Ba, Kiros, Hinton 2016 — *Layer Normalization* — оригинальная статья
- Zhang & Sennrich 2019 — *Root Mean Square Layer Normalization* — RMSNorm
- Xiong et al. 2020 — *On Layer Normalization in the Transformer Architecture* — доказательство преимущества Pre-Norm
- [[02 Areas/ML & DL/Papers/Formal Algorithms for Transformers|Formal Algorithms for Transformers]] — Algorithm 6, pre/post-norm сравнение
- [[02 Areas/ML & DL/Papers/LLaMA|LLaMA]] — RMSNorm Pre-Norm как архитектурное решение

## Related concepts

- [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]] — нормализация стабилизирует attention scores
- [[02 Areas/ML & DL/Concepts/NLP/Feed-Forward Network|Feed-Forward Network]] — нормализация перед/после FFN
- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] — нормализация = неотъемлемая часть каждого блока
- [[02 Areas/ML & DL/Concepts/Architectures/LLaMA|LLaMA]] — популяризировал RMSNorm + Pre-Norm
- [[02 Areas/ML & DL/Concepts/NLP/Positional Encoding|Positional Encoding]] — тоже влияет на стабильность обучения

## Дополнительные ресурсы

- [Sebastian Raschka — RMSNorm vs LayerNorm](https://sebastianraschka.com/faq/docs/rmsnorm-vs-layernorm.html) — краткое сравнение с кодом
- [Pinecone — Batch and Layer Normalization](https://www.pinecone.io/learn/batch-layer-normalization/) — визуальное объяснение
- [Michael Brenndoerfer — Pre-Norm vs Post-Norm](https://mbrenndoerfer.com/writing/pre-norm-vs-post-norm) — интерактивное сравнение
- [d2l.ai — The Transformer Architecture](http://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html) — LayerNorm в контексте полной архитектуры
