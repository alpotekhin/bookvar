---
title: "KV-Cache"
aliases: [KV-Cache, KV Cache, Key-Value Cache, KV кэш]
type: concept
category: Inference
papers:
  - "[[02 Areas/ML & DL/Papers/Mistral 7B|Mistral 7B]]"
  - "[[02 Areas/ML & DL/Papers/Flash Attention|Flash Attention]]"
sources:
  - "[NVIDIA — Mastering LLM Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)"
  - "[Lienhart — LLM Inference: KV Caching, a deeper look](https://medium.com/@plienhar/llm-inference-series-4-kv-caching-a-deeper-look-4ba9a77746c8)"
  - "[vLLM — PagedAttention](https://arxiv.org/abs/2309.06180)"
  - "[Introl — KV Cache Optimization](https://introl.com/blog/kv-cache-optimization-memory-efficiency-production-llms-guide)"
courses: []
---

# KV-Cache (Key-Value Cache)

## Зачем это нужно: проблема повторных вычислений

В decoder-only Transformer (GPT, LLaMA, Mistral) генерация текста — **авторегрессивный** процесс: каждый новый токен зависит от всех предыдущих. На шаге $t$ модель вычисляет attention нового токена ко **всей** предыдущей последовательности.

Без кэширования: чтобы сгенерировать токен $t+1$, нужно пересчитать Key и Value для **всех** токенов $1, 2, \ldots, t$. Генерация $N$ токенов требует $O(N^2)$ вычислений attention — квадратичная сложность.

**KV-Cache** решает эту проблему: ключи ($\mathbf{K}$) и значения ($\mathbf{V}$) предыдущих токенов **сохраняются в GPU memory** и переиспользуются. На каждом шаге вычисляются K, V только для **одного** нового токена и добавляются к кэшу.

$$\mathbf{K}_t = [\mathbf{K}_{t-1}; \mathbf{k}_t], \quad \mathbf{V}_t = [\mathbf{V}_{t-1}; \mathbf{v}_t]$$

где $\mathbf{k}_t = \mathbf{W}^K \mathbf{x}_t$ и $\mathbf{v}_t = \mathbf{W}^V \mathbf{x}_t$ — K, V нового токена.

Результат: генерация $N$ токенов — $O(N)$ вычислений attention вместо $O(N^2)$.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/kv-cache/kv-cache-mechanism.png]]
*KV-Cache: ключи и значения предыдущих токенов сохраняются и переиспользуются при генерации каждого нового токена (источник: Omri Mallis)*

## Как это работает: две фазы inference

### Фаза 1: Prefill (prompt processing)

Входной prompt обрабатывается **целиком параллельно** (как при training). Вычисляются K, V для всех токенов prompt и сохраняются в KV-cache:

$$\mathbf{K}_{\text{cache}} = \mathbf{X}_{\text{prompt}} \mathbf{W}^K, \quad \mathbf{V}_{\text{cache}} = \mathbf{X}_{\text{prompt}} \mathbf{W}^V$$

Prefill — **compute-bound**: доминирует матричное умножение. [[02 Areas/ML & DL/Concepts/Inference/Flash Attention|FlashAttention]] эффективно ускоряет эту фазу.

### Фаза 2: Decode (autoregressive generation)

Генерация токенов один за другим. На каждом шаге:

1. Вычисляем $\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t$ для нового токена
2. Добавляем $\mathbf{k}_t, \mathbf{v}_t$ в KV-cache
3. Вычисляем attention: $\text{softmax}(\mathbf{q}_t \cdot \mathbf{K}_{\text{cache}}^\top / \sqrt{d_k}) \cdot \mathbf{V}_{\text{cache}}$

Decode — **memory-bound**: на каждом шаге один query vector делает attention к длинному кэшу. Матричное умножение маленькое ($1 \times d \cdot d \times T$), зато нужно загрузить весь KV-cache из HBM.

```
Prefill (параллельно):
  Prompt: [How] [do] [you] [make] [pasta] [?]
           ↓     ↓    ↓     ↓      ↓     ↓
  KV-cache:  k₁v₁  k₂v₂  k₃v₃  k₄v₄  k₅v₅  k₆v₆

Decode (последовательно):
  Шаг 1: q₇ attend to [k₁..k₆] → "First"    → cache: [k₁..k₆, k₇]
  Шаг 2: q₈ attend to [k₁..k₇] → ","         → cache: [k₁..k₇, k₈]
  Шаг 3: q₉ attend to [k₁..k₈] → "boil"      → cache: [k₁..k₈, k₉]
  ...
```

## Проблема: взрывной рост памяти

KV-cache — **главный memory bottleneck** при inference длинных контекстов.

### Формула размера KV-cache

Для одного запроса:

$$\text{KV-cache size} = 2 \times L \times T \times d_{\text{model}} \times \text{sizeof(dtype)}$$

где:
- $2$ — Key и Value
- $L$ — число слоёв (layers)
- $T$ — длина последовательности (prompt + generated)
- $d_{\text{model}}$ — размерность модели

### Конкретные числа

| Модель | Layers | d_model | Context | Dtype | KV-cache на запрос |
|--------|--------|---------|---------|-------|-------------------|
| LLaMA-2 7B | 32 | 4096 | 4K | FP16 | **1 GB** |
| LLaMA-2 70B | 80 | 8192 | 4K | FP16 | **10 GB** |
| LLaMA-2 70B | 80 | 8192 | 128K | FP16 | **320 GB** |

Для LLaMA-2 70B с контекстом 128K: KV-cache **превышает размер самой модели** (140 GB в FP16). При batch size 32 — это **10 TB** KV-cache. Очевидно, нужны оптимизации.

### Проблема фрагментации

Классические inference системы выделяют **непрерывный блок** памяти под KV-cache для максимального контекста. Но реальная длина генерации неизвестна заранее:
- **Internal fragmentation** — выделено 2048 токенов, использовано 500 → 75% waste
- **External fragmentation** — после завершения запросов остаются «дырки» разного размера
- Суммарно: **60-80% GPU memory wasted** в традиционных системах

## Оптимизация 1: GQA и MQA — уменьшение числа KV-голов

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/kv-cache/gqa-comparison.png]]
*Multi-Head Attention (MHA) vs Multi-Query Attention (MQA) vs Grouped-Query Attention (GQA): GQA — компромисс между полной гибкостью MHA и эффективностью MQA (источник: Omri Mallis)*

### Multi-Query Attention (MQA)

Shazeer (2019): **все** query-головы разделяют **одну** пару KV-голов. Вместо $h$ наборов K, V — только один.

$$\text{Экономия KV-cache}: \frac{1}{h} \quad \text{(для } h = 32 \text{ голов — в 32 раза)}$$

Минус: может терять качество из-за чрезмерного sharing.

### Grouped-Query Attention (GQA)

Ainslie et al. (2023): компромисс между MHA и MQA. Query-головы делятся на $g$ групп, каждая группа разделяет одну KV-пару.

$$\text{Экономия KV-cache}: \frac{g}{h}$$

**Примеры в production:**

| Модель | Query heads | KV heads | Тип | KV-cache сокращение |
|--------|-----------|----------|-----|-------------------|
| LLaMA-2 70B | 64 | 8 | GQA | 8x |
| Mistral 7B | 32 | 8 | GQA | 4x |
| Gemma 2B | 8 | 1 | MQA | 8x |
| Falcon 180B | 232 | 8 | GQA | 29x |

GQA стал **де-факто стандартом** в modern LLMs: значительная экономия памяти с минимальной потерей качества (within 1% на большинстве бенчмарков по сравнению с MHA).

## Оптимизация 2: Rolling Buffer Cache (Sliding Window)

[[02 Areas/ML & DL/Concepts/Architectures/Mistral 7B|Mistral 7B]] использует **Sliding Window Attention (SWA)** с фиксированным размером KV-cache = $W$ (window size = 4096).

Принцип: каждый токен attend'ит только к $W$ ближайшим предыдущим токенам. KV-cache — **кольцевой буфер**: позиция $i$ хранится в ячейке $i \mod W$.

```
Window W = 4:

Шаг 5:  cache[0]=k₅  cache[1]=k₂  cache[2]=k₃  cache[3]=k₄   (k₁ перезаписан)
Шаг 6:  cache[0]=k₅  cache[1]=k₆  cache[2]=k₃  cache[3]=k₄   (k₂ перезаписан)
Шаг 7:  cache[0]=k₅  cache[1]=k₆  cache[2]=k₇  cache[3]=k₄   (k₃ перезаписан)
```

**Экономия:** для последовательности 32K при $W = 4096$: **8x уменьшение** KV-cache memory.

**Почему это работает:** хотя каждый слой видит только $W$ токенов, при $L$ слоях информация может проходить через $L \times W$ позиций. Для Mistral 7B: $32 \times 4096 = 131072$ токенов теоретического «рецептивного поля».

## Оптимизация 3: PagedAttention (vLLM)

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/kv-cache/paged-attention.png]]
*PagedAttention: KV-cache разбивается на страницы фиксированного размера, хранящиеся неконтигуально в GPU memory — аналогия с виртуальной памятью ОС (источник: Omri Mallis)*

PagedAttention (Kwon et al., 2023) — прорыв в management KV-cache, вдохновлённый **виртуальной памятью ОС**.

### Проблема

Традиционные системы выделяют **непрерывный** блок GPU memory под KV-cache. Это приводит к фрагментации: 60-80% памяти теряется.

### Решение

PagedAttention разбивает KV-cache на **страницы** (blocks) фиксированного размера (например, 16 токенов). Страницы хранятся **нонконтигуально** (не обязательно рядом в физической памяти). **Block table** маппирует логические позиции в физические адреса.

```
Запрос: "The capital of France is Paris"

Логические позиции:  [0..15] → Page 3  (физический адрес)
                     [16..31] → Page 7
                     [32..35] → Page 1  (частично заполнена)

Block Table:
  Logical Block 0 → Physical Block 3
  Logical Block 1 → Physical Block 7
  Logical Block 2 → Physical Block 1
```

### Ключевые преимущества

1. **Near-zero waste** — память выделяется по мере необходимости, без предварительного резервирования. <4% waste вместо 60-80%.
2. **Memory sharing** — для parallel sampling (beam search, best-of-N) KV-cache промпта физически **расшаривается** между ветвями. Copy-on-write при модификации.
3. **2-4x throughput improvement** — больше запросов помещается в GPU memory.

PagedAttention — основа **vLLM** (UC Berkeley), одного из самых популярных inference engines.

## Оптимизация 4: Quantized KV-Cache

Хранение K, V в низкой точности:

| Precision | Размер на элемент | Сокращение | Влияние на качество |
|-----------|------------------|------------|-------------------|
| FP16 | 2 bytes | baseline | — |
| INT8 | 1 byte | 2x | <0.5% degradation |
| INT4 | 0.5 bytes | 4x | 1-3% degradation |
| FP8 (E4M3) | 1 byte | 2x | <0.3% degradation |

INT8 KV-cache quantization стал стандартом в production — экономит 2x памяти с минимальной потерей качества. Поддерживается в vLLM, TGI, TensorRT-LLM.

## Оптимизация 5: KV-Cache Compression

Более агрессивные подходы:

### Token Dropping / Eviction

Не все токены одинаково важны для attention. H2O (Heavy Hitter Oracle, Zhang et al., 2023) отслеживает, какие токены получают высокие attention weights, и **удаляет** непопулярные из кэша.

### StreamingLLM (Xiao et al., 2023)

Наблюдение: initial tokens (первые 1-4 токена) получают непропорционально высокие attention weights — «attention sinks». StreamingLLM хранит:
- **Initial tokens** (4 штуки) — как attention sinks
- **Recent window** (последние $W$ токенов) — для локального контекста

Это позволяет стабильную генерацию на **бесконечно** длинных текстах с фиксированным KV-cache.

## Две фазы, два bottleneck'а

| Фаза | Compute profile | Bottleneck | Оптимизация |
|------|----------------|-----------|-------------|
| **Prefill** | Compute-bound | FLOPs (matmul) | FlashAttention, tensor parallelism |
| **Decode** | Memory-bound | HBM bandwidth (загрузка KV-cache) | GQA, quantization, PagedAttention |

При decode на каждом шаге загружается **весь** KV-cache из HBM для одного attention-вычисления. Для LLaMA-2 70B с контекстом 4K: загрузка ~10 GB KV-cache для одного токена. При HBM bandwidth 2 TB/s — это 5ms только на чтение кэша.

## Зависимость скорости inference от KV-cache

```
Throughput (tokens/s) vs Context Length для LLaMA-2 70B (A100 80GB):

Context     KV-cache    Batch   Throughput
256         0.2 GB      64      ~3200 tok/s
1024        0.8 GB      32      ~1600 tok/s
4096        3.2 GB      8       ~400 tok/s
16384       12.8 GB     2       ~100 tok/s
32768       25.6 GB     1       ~50 tok/s

→ При увеличении контекста 128x, throughput падает ~64x
```

KV-cache определяет **максимальный batch size**, что напрямую влияет на throughput и стоимость inference.

## Хронология

| Год | Milestone |
|-----|-----------|
| 2019 | MQA (Shazeer) — один KV-head |
| 2022 | FlashAttention — IO-aware attention |
| 2023 | GQA (Ainslie et al.) — компромисс MHA/MQA |
| 2023 | Mistral 7B — Rolling Buffer Cache (SWA) |
| 2023 | PagedAttention / vLLM — виртуальные страницы |
| 2023 | H2O — KV-cache eviction по attention scores |
| 2023 | StreamingLLM — infinite context с fixed KV |
| 2024 | INT8/FP8 KV quantization — стандарт в production |
| 2024 | KV-cache offloading (CPU/NVMe) для длинных контекстов |

## Related concepts

- [[02 Areas/ML & DL/Concepts/Inference/Flash Attention|Flash Attention]] — оптимизирует prefill фазу; ортогонален к KV-Cache оптимизациям
- [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]] — KV-Cache кэширует K, V проекции из attention
- [[02 Areas/ML & DL/Concepts/Inference/Speculative Decoding|Speculative Decoding]] — генерация нескольких токенов за шаг; зависит от KV-Cache
- [[02 Areas/ML & DL/Concepts/Architectures/Mistral 7B|Mistral 7B]] — Rolling Buffer Cache, GQA
- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] — архитектура, генерирующая KV-Cache

## Дополнительные ресурсы

- [NVIDIA — Mastering LLM Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) — обзор техник
- [vLLM Paper — PagedAttention](https://arxiv.org/abs/2309.06180) — оригинальная статья PagedAttention
- [Lienhart — KV Caching: a deeper look](https://medium.com/@plienhar/llm-inference-series-4-kv-caching-a-deeper-look-4ba9a77746c8) — подробный разбор
- [Introl — KV Cache Optimization Guide](https://introl.com/blog/kv-cache-optimization-memory-efficiency-production-llms-guide) — production-ориентированный гайд
