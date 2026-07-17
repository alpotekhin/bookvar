---
title: "RetNet"
aliases: [RetNet, Retentive Network, Retention Mechanism, Multi-Scale Retention]
type: concept
status: legacy
category: Architectures
papers:
  - "[[02 Areas/ML & DL/Papers/RetNet|RetNet]]"
courses: []
sources:
  - "[Microsoft Research — RetNet (GitHub)](https://aka.ms/retnet)"
  - "[Shantanu Chandra — Retentive Networks Explained (Medium)](https://medium.com/ai-fusion-labs/retentive-networks-retnet-explained-the-much-awaited-transformers-killer-is-here-6c17e3e8add8)"
  - "[Marvik — Exploring RetNet: The Evolution of Transformers](https://blog.marvik.ai/2024/07/16/exploring-retnet-the-evolution-of-transformers/)"
---

# RetNet

## Зачем это нужно: «Impossible Triangle»

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/retnet/impossible-triangle.png]]
*«Impossible Triangle»: до RetNet ни одна архитектура не могла одновременно обеспечить training parallelism, low-cost inference и good performance (источник: оригинальная статья)*

Transformer решает две из трёх задач:
1. **Training parallelism** — полная параллелизация через self-attention
2. **Good performance** — лучшее качество на language modeling
3. **Low-cost inference** — **НЕТ**. $O(n)$ KV-cache, растущая latency и memory

Три направления исследований пытались решить «impossible triangle»:

| Подход | Training | Inference | Quality |
|--------|----------|-----------|---------|
| Linear Attention | Параллельно | $O(1)$ | Хуже Transformer |
| Recurrent (RNN) | **Не** параллельно | $O(1)$ | Хуже Transformer |
| S4/Hyena | Параллельно | $O(1)$ | Хуже Transformer |
| **RetNet** | **Параллельно** | **$O(1)$** | **Сопоставимо** |

RetNet (Sun et al., Microsoft Research, 2023) предлагает **retention mechanism** — замену self-attention, которая поддерживает **три парадигмы вычислений** и претендует на решение impossible triangle.

## Retention Mechanism: от рекуррентности к параллелизму

### Исходная формулировка: рекуррентная

Начнём с простого отображения $v(n) \mapsto o(n)$ через state $s_n$:

$$s_n = \mathbf{A} s_{n-1} + K_n^T v_n$$
$$o_n = Q_n s_n = \sum_{m=1}^{n} Q_n \mathbf{A}^{n-m} K_m^T v_m$$

где $Q_n, K_n$ — content-aware проекции: $Q = XW_Q$, $K = XW_K$.

### Диагонализация A

Матрица $\mathbf{A}$ диагонализуется как $\Lambda(\gamma e^{i\theta})\Lambda^{-1}$, где $\gamma$ — scalar decay rate, $\theta$ — positional angle. Поглощая $\Lambda$ в $W_Q$ и $W_K$:

$$o_n = \sum_{m=1}^{n} \gamma^{n-m} (Q_n e^{in\theta})(K_m e^{im\theta})^\dagger v_m$$

Это **xPos** — relative position embedding из Transformer литературы. RetNet математически **выводит** positional encoding из рекуррентной формулировки, а не добавляет его эвристически.

## Три представления Retention

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/retnet/dual-form.png]]
*Двойственная форма RetNet: (a) Parallel — матричное вычисление для обучения, (b) Recurrent — пошаговое вычисление для инференса (источник: оригинальная статья)*

### 1. Parallel (обучение)

$$Q = (XW_Q) \odot \Theta, \quad K = (XW_K) \odot \bar{\Theta}, \quad V = XW_V$$

$$D_{nm} = \begin{cases} \gamma^{n-m}, & n \geq m \\ 0, & n < m \end{cases}$$

$$\text{Retention}(X) = (QK^T \odot D) V$$

Матрица $D$ объединяет **causal masking** (нижнетреугольная) и **exponential decay** (затухание по расстоянию) в одну операцию. Сравни с attention:

| | Attention | Retention |
|---|-----------|-----------|
| Формула | $\text{softmax}(QK^T / \sqrt{d_k}) V$ | $(QK^T \odot D) V$ |
| Softmax | Да | **Нет** |
| Positional encoding | Отдельный (RoPE/xPos) | **Встроен** в $D$ |
| Causal mask | Отдельный ($-\infty$) | **Встроен** в $D$ |

Убрав softmax, RetNet делает вычисление **линейным** — это ключ к рекуррентной формулировке.

### 2. Recurrent (инференс)

$$S_n = \gamma S_{n-1} + K_n^T V_n$$
$$\text{Retention}(X_n) = Q_n S_n$$

$O(1)$ на шаг: одно матричное обновление state $S_n$ и одно умножение для выхода. **Не нужен KV-cache** — вся история в state матрице $S_n$.

### 3. Chunkwise Recurrent (длинные последовательности)

Гибрид: входная последовательность разбивается на chunk'и размера $B$.

$$\text{Inner-Chunk:} \quad (Q_{[i]} K_{[i]}^T \odot D) V_{[i]}$$
$$\text{Cross-Chunk:} \quad (Q_{[i]} R_{i-1}) \odot \xi$$

Внутри chunk — **parallel** (эффективно на GPU). Между chunk'ами — **recurrent** (экономия памяти). Линейная memory complexity по длине последовательности.

```python
# Pseudocode: Chunkwise Retention
def ChunkwiseRetention(q, k, v, past_kv, decay_mask, chunk_decay, inner_decay):
    # Inner-chunk: parallel
    retention = q @ k.transpose(-1, -2) * decay_mask
    inner_retention = retention @ v
    # Cross-chunk: recurrent
    cross_retention = (q @ past_kv) * inner_decay
    output = group_norm(inner_retention + cross_retention)
    # Update state
    current_kv = chunk_decay * past_kv + k.transpose(-1, -2) @ v
    return output, current_kv
```

## Multi-Scale Retention (MSR)

$$\gamma = 1 - 2^{-5-\text{arange}(0,h)} \in \mathbb{R}^h$$

$h$ голов с **разными** decay rates $\gamma_i$. Каждая голова «помнит» на разную глубину:
- Голова с $\gamma \approx 1$ — длинная память (медленное затухание)
- Голова с $\gamma \approx 0.97$ — короткая память (быстрое затухание)

Это аналог multi-head attention, но вместо разных «точек зрения» — разные **временные масштабы**.

### Swish Gate и GroupNorm

$$Y = \text{GroupNorm}_h(\text{Concat}(\text{head}_1, \ldots, \text{head}_h))$$
$$\text{MSR}(X) = (\text{swish}(XW_G) \odot Y) W_O$$

GroupNorm (а не LayerNorm) — потому что разные $\gamma$ создают **разную дисперсию** для каждой головы. Swish gate добавляет нелинейность.

Retention layer **без softmax** может иметь числовые проблемы. Три нормализации:
1. $QK^T / \sqrt{d}$
2. Замена $D$ на $\tilde{D}_{nm} = D_{nm} / \sqrt{\sum_i D_{ni}}$
3. Нормализация retention scores: $\tilde{R}_{nm} = R_{nm} / \max(|\sum_i R_{ni}|, 1)$

Scale-invariance GroupNorm делает эти нормализации **бесплатными** — они не меняют forward/backward.

## Полная архитектура

$L$ одинаковых блоков:

$$Y^l = \text{MSR}(\text{LN}(X^l)) + X^l$$
$$X^{l+1} = \text{FFN}(\text{LN}(Y^l)) + Y^l$$

FFN: $\text{FFN}(X) = \text{gelu}(XW_1)W_2$. Intermediate dimension = $2d$ (вместо $4d$ у Transformer), потому что retention layer уже содержит 8d^2 параметров (vs. 4d^2 у attention).

## Бенчмарки

### Language Modeling Scaling

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/retnet/scaling-curve.png]]
*Perplexity scaling: RetNet начинает превосходить Transformer при размере >2B параметров (источник: оригинальная статья)*

Три размера (1.3B, 2.7B, 6.7B), обучение на 100B токенов. RetNet **сопоставим** с Transformer, причём разрыв в пользу RetNet **растёт с размером модели**.

### Zero-Shot / Few-Shot (6.7B)

| Task | Transformer | RetNet |
|------|------------|--------|
| HellaSwag | 55.9 | **60.7** |
| BoolQ | 62.0 | 62.2 |
| COPA | 69.0 | **77.0** |
| PIQA | 74.6 | 75.4 |
| Winograd | 69.5 | **77.2** |
| Average (0-shot) | 66.07 | **69.51** |

RetNet 6.7B **превосходит** Transformer 6.7B на всех downstream задачах. Среднее: +3.4% zero-shot, +3.3% few-shot.

### Сравнение с другими моделями (200M)

| Метод | In-Domain PPL↓ |
|-------|---------------|
| RWKV | 30.92 |
| H3 | 29.97 |
| Hyena | 32.08 |
| Linear Transformer | 37.41 |
| Transformer | 26.74 |
| **RetNet** | **26.05** |

RetNet **превосходит** Transformer на 200M и все альтернативные архитектуры.

## Inference Cost: главное преимущество

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/retnet/inference-cost-comparison.png]]
*Inference cost RetNet vs. Transformer (6.7B, 8K контекст): 8.4x быстрее decoding, 70% экономии памяти, 15.6x меньше latency (источник: оригинальная статья)*

### Memory

Память Transformer линейно растёт с длиной (KV-cache). Память RetNet **постоянна** — ~3% overhead сверх model weights (state матрицы). При 8K входе: RetNet = 15GB vs. Transformer = 45GB.

### Throughput

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/retnet/inference-memory-throughput.png]]
*GPU memory и throughput при inference: RetNet имеет постоянную стоимость, Transformer деградирует с длиной (источник: оригинальная статья)*

Throughput Transformer **падает** с длиной (KV-cache растёт → больше memory reads). RetNet — **константный** throughput. При 8K: RetNet 8.4x быстрее.

### Latency

Latency Transformer **растёт** с batch size (больше KV-cache → больше memory bandwidth). RetNet — **нечувствителен** к batch size. Это позволяет использовать большие batch'и без деградации latency, что критически важно для production deployment.

| Метрика (6.7B, 8K) | Transformer | RetNet | Выигрыш |
|---------------------|------------|--------|---------|
| GPU Memory | 45.0 GB | 15.6 GB | **2.9x** |
| Throughput | ~30 wps | ~250 wps | **8.4x** |
| Latency (batch=8) | ~300 ms | ~20 ms | **15.6x** |

## Training Cost

| Модель | Memory (GB) | Throughput (wps) |
|--------|------------|-----------------|
| Transformer | 74.8 (1.3B) | 10,832 |
| Transformer + FlashAttn | 38.8 | 63,965 |
| **RetNet** | **34.5** | **73,345** |

RetNet при обучении **быстрее даже FlashAttention** на ванильном PyTorch (без специальных kernels). 25-50% экономии памяти, 7x ускорение vs. стандартный Transformer.

## RetNet vs. другие архитектуры

| Архитектура | Training Parallel | Inference $O(1)$ | Long-Seq Memory | Performance |
|-------------|------------------|------------------|-----------------|-------------|
| Transformer | Да | Нет ($O(n)$) | $O(n^2)$ | Лучшее |
| Linear Transformer | Да | Да | $O(n)$ | Слабое |
| RNN | Нет | Да | $O(n)$ | Слабое |
| [[02 Areas/ML & DL/Concepts/Architectures/RWKV|RWKV]] | Частично | Да | $O(n)$ | Хорошее |
| H3/S4 | Да | Да | $O(n \log n)$ | Хорошее |
| Hyena | Да | Нет ($O(n)$) | $O(n \log n)$ | Хорошее |
| **RetNet** | **Да** | **Да** | **$O(n)$** | **Отличное** |

RetNet — **единственная** архитектура с галочками во всех четырёх столбцах (на момент публикации).

## Связь с другими методами

**Transformer / xPos** — parallel retention аналогично attention, но без softmax. Positional encoding (xPos) **математически выводится** из retention, а не добавляется.

**S4** — если $Q_n$ и $K_n$ content-unaware (фиксированные), retention деградирует до S4.

**Linear Attention** — использует kernel $\phi(q_i)\phi(k_j)$ для замены softmax. RetNet не аппроксимирует softmax, а **выводит** retention с нуля из рекуррентной формулировки.

**AFT/[[02 Areas/ML & DL/Concepts/Architectures/RWKV|RWKV]]** — RWKV использует element-wise operations и channel-wise decay. RetNet сохраняет **full-dimensional states** ($d \times d$ матрицы), что даёт бо́льшую ёмкость.

## Related concepts

- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] — архитектура, которую RetNet стремится заменить
- [[02 Areas/ML & DL/Concepts/Architectures/Mamba|Mamba]] — SSM-based альтернатива, тоже с $O(1)$ инференсом
- [[02 Areas/ML & DL/Concepts/Architectures/RWKV|RWKV]] — RNN-based альтернатива с element-wise операциями
- [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]] — retention как замена attention
- [[02 Areas/ML & DL/Concepts/NLP/Positional Encoding|Positional Encoding]] — xPos выводится из retention mechanism

## Дополнительные ресурсы

- [Microsoft Research — RetNet](https://aka.ms/retnet) — официальная страница
- [Shantanu Chandra — RetNet Explained](https://medium.com/ai-fusion-labs/retentive-networks-retnet-explained-the-much-awaited-transformers-killer-is-here-6c17e3e8add8) — подробный разбор с визуализациями
- [Marvik — Exploring RetNet](https://blog.marvik.ai/2024/07/16/exploring-retnet-the-evolution-of-transformers/) — сравнение с Transformer
- [arXiv — Retentive Network (2307.08621)](https://arxiv.org/abs/2307.08621) — оригинальная статья
