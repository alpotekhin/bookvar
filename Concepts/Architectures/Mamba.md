---
title: "Mamba"
aliases: [Mamba, Selective SSM, S6, Selective State Space Model]
type: concept
category: Architectures
papers:
  - "[[02 Areas/ML & DL/Papers/Mamba|Mamba]]"
courses: []
sources:
  - "[Maarten Grootendorst — A Visual Guide to Mamba and State Space Models](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state)"
  - "[Towards Data Science — Here Comes Mamba: The Selective State Space Model](https://towardsdatascience.com/here-comes-mamba-the-selective-state-space-model-435e5d17a451/)"
  - "[IBM — What Is A Mamba Model?](https://www.ibm.com/think/topics/mamba-model)"
  - "[Goomba Lab — State Space Duality (Mamba-2)](https://goombalab.github.io/blog/2024/mamba2-part1-model/)"
---

# Mamba

## Зачем это нужно: проблема Transformer на длинных последовательностях

Transformer доминирует в deep learning, но у него фундаментальная проблема: **квадратичная сложность** $O(n^2)$ по длине последовательности. Матрица attention размером $n \times n$ растёт квадратично — при 100K токенов это $10^{10}$ элементов. Кроме того, авторегрессивный инференс требует **KV-cache**, который линейно растёт с длиной контекста — это bottleneck и по памяти, и по latency.

Многочисленные попытки создать subquadratic альтернативы (linear attention, Hyena, H3, RWKV) не смогли сравниться с Transformer по качеству на **дискретных данных** (текст, код). Gu & Dao (2023) определили корневую причину: все эти модели — **Linear Time-Invariant (LTI)**, то есть их параметры одинаковы для каждого токена. Это не позволяет им выполнять **content-based reasoning** — фильтрацию по содержанию входа.

## Фундамент: Structured State Space Models (S4)

Mamba строится на фундаменте S4 (Gu et al., 2022). SSM описывается непрерывной системой:

$$h'(t) = \mathbf{A}h(t) + \mathbf{B}x(t), \quad y(t) = \mathbf{C}h(t)$$

где $h(t) \in \mathbb{R}^N$ — скрытое состояние, $\mathbf{A} \in \mathbb{R}^{N \times N}$ — матрица динамики, $\mathbf{B}, \mathbf{C}$ — проекции входа и выхода.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/mamba/ssm-three-views.png]]
*Три представления SSM: непрерывное, рекуррентное, свёрточное (источник: Maarten Grootendorst)*

### Три представления одной модели

**Дискретизация.** Непрерывная система дискретизируется через шаг $\Delta$ (zero-order hold):

$$\bar{\mathbf{A}} = \exp(\Delta \mathbf{A}), \quad \bar{\mathbf{B}} = (\Delta \mathbf{A})^{-1}(\exp(\Delta \mathbf{A}) - \mathbf{I}) \cdot \Delta \mathbf{B}$$

**Рекуррентное** — токен-за-токеном: $h_t = \bar{\mathbf{A}} h_{t-1} + \bar{\mathbf{B}} x_t$, $y_t = \mathbf{C} h_t$. Эффективно при инференсе — $O(1)$ на шаг.

**Свёрточное** — фиксированный kernel $\mathbf{K} = (\mathbf{C}\bar{\mathbf{B}}, \mathbf{C}\bar{\mathbf{A}}\bar{\mathbf{B}}, \ldots)$, применяемый ко всей последовательности: $y = x * \mathbf{K}$. Эффективно при обучении — полная параллелизация.

Ключевое свойство S4: **параметры $(\Delta, \mathbf{A}, \mathbf{B}, \mathbf{C})$ фиксированы** для всех позиций. Это даёт LTI, что и позволяет переключаться между рекуррентным и свёрточным режимами. Но это же свойство ломает модель на задачах, требующих content-awareness.

## Почему LTI ломается: интуиция через синтетические задачи

Авторы выделяют два тестовых случая:

1. **Selective Copying** — классическая задача копирования, но с *рандомным* расположением токенов для запоминания. LTI-модели решают vanilla Copying (фиксированные позиции), но ломаются, когда нужно *по содержанию* решить, что запомнить.

2. **Induction Heads** — если модель видела биграм «Harry Potter», то при следующем появлении «Harry» должна предсказать «Potter». Требуется ассоциативная память на основе содержания.

Проблема LTI: матрицы $\mathbf{B}$ и $\mathbf{C}$ **не зависят от входа**. Модель не может решить «этот токен важен, запомню» или «это шум, проигнорирую» — она одинаково обрабатывает всё.

## Selection Mechanism (S6): ключевая инновация

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/mamba/selective-ssm.png]]
*Слева: S4 с фиксированными параметрами. Справа: S6 (Mamba) — параметры B, C, Delta зависят от входа (источник: Maarten Grootendorst)*

Решение элегантно простое — **сделать параметры функциями входа**:

$$\mathbf{B}_t = \text{Linear}_N(x_t), \quad \mathbf{C}_t = \text{Linear}_N(x_t), \quad \Delta_t = \text{softplus}(\text{Parameter} + \text{Linear}_1(x_t))$$

Теперь $\mathbf{B}$, $\mathbf{C}$, $\Delta$ имеют **length dimension** — разные для каждой позиции. Модель стала **time-varying**.

### Что делает каждый параметр

**$\Delta$ (step size)** — главный рычаг. Управляет балансом «запомнить текущий вход vs. сохранить предыдущее состояние»:
- Большой $\Delta$ → фокус на текущем токене, reset состояния (запоминание)
- Малый $\Delta$ → токен игнорируется, состояние сохраняется (забывание шума)

**Связь с гейтами RNN (Теорема 1).** При $N = 1$, $\mathbf{A} = -1$, $\mathbf{B} = 1$:

$$g_t = \sigma(\text{Linear}(x_t)), \quad h_t = (1 - g_t) h_{t-1} + g_t x_t$$

Selective SSM — это **принципиальное обобщение** гейтированных RNN (LSTM/GRU). Дискретизация SSM — фундамент, а эвристические гейты — частный случай.

**$\mathbf{B}_t$** — контролирует, какую информацию из входа $x_t$ записать в состояние $h_t$ (content-based gating).

**$\mathbf{C}_t$** — контролирует, какую информацию из состояния $h_t$ прочитать в выход $y_t$ (context-based gating).

### Цена селективности

Time-varying параметры **ломают эквивалентность со свёрткой** (3). Kernel больше нельзя предвычислить — он разный для каждого входа. Остаётся только рекуррентный путь вычисления. Казалось бы, это конец эффективности. Но...

## Hardware-Aware Parallel Scan: как сохранить скорость

Три классические техники, адаптированные для GPU:

### 1. Kernel Fusion
Наивная реализация требует материализации промежуточного состояния $h$ размером $(B, L, D, N)$ в GPU HBM. Вместо этого:
- Параметры $(\Delta, \mathbf{A}, \mathbf{B}, \mathbf{C})$ загружаются из HBM в **SRAM** (быстрая on-chip память)
- Дискретизация и рекуррентное вычисление выполняются **в SRAM**
- Только финальный выход $(B, L, D)$ записывается обратно в HBM

Аналогия с [[02 Areas/ML & DL/Concepts/Inference/Flash Attention|Flash Attention]]: минимизация IO между уровнями memory hierarchy.

### 2. Parallel Scan (Blelloch, 1990)
Рекуррентное вычисление $h_t = \bar{\mathbf{A}}_t h_{t-1} + \bar{\mathbf{B}}_t x_t$ кажется последовательным. Но операция ассоциативна — можно вычислять части параллельно и объединять. Work-efficient parallel scan даёт $O(\log L)$ последовательных шагов вместо $O(L)$.

### 3. Recomputation
Промежуточные состояния **не сохраняются** для backward pass. Вместо этого пересчитываются заново при backpropagation (когда входы снова загружены в SRAM). Экономия памяти за счёт дополнительных FLOPs — тот же трейдофф, что в FlashAttention.

Результат: fused selective scan layer имеет **те же требования к памяти**, что и оптимизированный Transformer с FlashAttention.

## Архитектура Mamba: один гомогенный блок

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/mamba/mamba-block.png]]
*Mamba block: объединение H3 block и Gated MLP в единую структуру (источник: Maarten Grootendorst)*

Вместо чередования Attention + MLP (как в Transformer) или SSM + MLP (как в H3), Mamba использует **один повторяющийся блок**:

1. **Input projection** — линейная проекция с expansion factor $E = 2$
2. **Два branch:**
   - Основной: Conv1D → SiLU → **Selective SSM**
   - Gate: SiLU activation
3. **Element-wise multiply** (gating)
4. **Output projection**

Два стека блока Mamba = $12D^2$ параметров, что совпадает с Transformer (MHA + MLP). Блоки складываются с RMSNorm + residual connections.

### Отсутствующие компоненты
- **Нет attention** — вместо него Selective SSM
- **Нет отдельного MLP** — встроен в блок через gating
- **Нет positional encoding** — рекуррентная природа SSM неявно кодирует позицию через $\Delta$

## Бенчмарки: первая attention-free модель уровня Transformer

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/mamba/mamba-vs-transformer.png]]
*Scaling laws: Mamba — первая attention-free модель, сопоставимая с Transformer++ (LLaMA recipe) по perplexity (источник: Maarten Grootendorst)*

### Scaling Laws (125M — 1.3B)
Mamba — **первая** attention-free модель, совпадающая с Transformer++ (LLaMA recipe: RoPE, SwiGLU, RMSNorm) по perplexity. При увеличении длины последовательности разрыв с Transformer только **растёт в пользу Mamba** (линейный vs. квадратичный скейлинг).

### Zero-Shot Downstream Evaluations

| Модель | Params | Pile ppl↓ | LAMBADA acc↑ | HellaSwag↑ | PIQA↑ | Average↑ |
|--------|--------|-----------|-------------|-----------|------|---------|
| Pythia-1.4B | 1.4B | 7.51 | 61.7 | 52.1 | 71.0 | 55.2 |
| RWKV-1.5B | 1.5B | 7.70 | 56.4 | 52.5 | 72.4 | 54.3 |
| **Mamba-1.4B** | **1.4B** | **6.80** | **64.9** | **59.1** | **74.2** | **59.7** |
| Pythia-2.8B | 2.8B | 6.73 | 64.7 | 59.3 | 74.0 | 59.1 |
| **Mamba-2.8B** | **2.8B** | **6.22** | **69.2** | **66.1** | **75.2** | **63.3** |
| Pythia-6.9B | 6.9B | 6.51 | 67.1 | 64.0 | 75.2 | 61.7 |

Mamba-1.4B **превосходит Pythia-2.8B** (модель в 2x больше). Mamba-2.8B сопоставим с **Pythia-6.9B**.

### Synthetic Tasks
- **Selective Copying**: S6 (Mamba) — 99.8% accuracy vs. S4 — 56.4%
- **Induction Heads**: экстраполяция до **1M токенов** (4000x длиннее обучения). Ни одна другая модель не экстраполирует дальше 2x.

### Мультимодальность
- **DNA**: до 1M контекст, лучше HyenaDNA и Transformer
- **Audio**: снижение FID на speech generation более чем в 2 раза

## Инференс: 5x throughput при линейном скейлинге

| Свойство | Transformer | Mamba |
|----------|-------------|-------|
| Сложность обучения | $O(n^2 d)$ | $O(n d)$ |
| Сложность инференса/шаг | $O(n)$ (KV-cache) | $O(1)$ |
| Память при инференсе | $O(n)$ (растёт с контекстом) | $O(1)$ (фиксированный state) |
| Throughput | Базовый | **5x выше** |
| Генерация | Замедляется с длиной | Постоянная скорость |

При инференсе Mamba разворачивается как RNN: $O(1)$ на токен, **не нужен KV-cache**. Состояние фиксированного размера $DN$ — достаточно для любой длины контекста.

## Почему Mamba сломала LTI: глубокий взгляд

Суть инновации — переход от **compression** к **selection**. Как формулируют авторы:

> Фундаментальная проблема sequence modeling — сжатие контекста в малое состояние. Attention эффективен, но неэффективен, потому что *не сжимает вообще* (KV-cache = весь контекст). Рекуррентные модели эффективны, но их качество ограничено тем, *как хорошо* они сжимают.

Mamba решает дилемму: **selectivity** позволяет сжимать *умно* — запоминая релевантное и забывая шум. Три механизма:

1. **Variable spacing** — фильтрация шумовых токенов («um», fillers)
2. **Filtering context** — reset состояния для удаления нерелевантной истории
3. **Boundary resetting** — при $\Delta_t \to \infty$ состояние сбрасывается (аналог attention mask для границ документов)

## Наследие и развитие

Mamba открыла путь для SSM-based архитектур:
- **Mamba-2** (Gu & Dao, 2024) — State Space Duality, связь SSM с structured attention, ещё быстрее
- **Jamba** (AI21, 2024) — гибрид Mamba + Transformer + MoE
- **Zamba** — Mamba для vision
- Активное применение в DNA modeling, audio, time series

## Related concepts

- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] — архитектура, которую Mamba стремится заменить
- [[02 Areas/ML & DL/Concepts/Architectures/RWKV|RWKV]] — другой linear-time подход (RNN с linear attention)
- [[02 Areas/ML & DL/Concepts/Architectures/RetNet|RetNet]] — retention mechanism, третий кандидат на замену attention
- [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]] — $O(n^2)$ механизм, который SSM заменяет
- [[02 Areas/ML & DL/Concepts/Inference/Flash Attention|Flash Attention]] — IO-aware оптимизация, вдохновившая hardware-aware scan
- [[02 Areas/ML & DL/Concepts/Inference/KV-Cache|KV-Cache]] — проблема Transformer, которую Mamba устраняет

## Дополнительные ресурсы

- [Maarten Grootendorst — A Visual Guide to Mamba and State Space Models](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state) — лучшие визуализации SSM и selection mechanism
- [Towards Data Science — Here Comes Mamba](https://towardsdatascience.com/here-comes-mamba-the-selective-state-space-model-435e5d17a451/) — подробный разбор с кодом
- [Goomba Lab — State Space Duality (Mamba-2)](https://goombalab.github.io/blog/2024/mamba2-part1-model/) — теоретическая связь SSM и attention
- [GitHub — state-spaces/mamba](https://github.com/state-spaces/mamba) — официальная реализация
