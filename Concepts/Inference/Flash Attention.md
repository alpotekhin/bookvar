---
title: "Flash Attention"
aliases: [FlashAttention, FlashAttention-2, FlashAttention-3, IO-aware Attention]
type: concept
status: legacy
category: Inference
papers:
  - "[[02 Areas/ML & DL/Papers/Flash Attention|Flash Attention (Dao et al., 2022)]]"
  - "[[02 Areas/ML & DL/Papers/Flash Attention 2|Flash Attention 2 (Dao, 2023)]]"
sources:
  - "[Dao et al. — FlashAttention (2022)](https://arxiv.org/abs/2205.14135)"
  - "[Gordic — ELI5: Flash Attention](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad)"
  - "[DigitalOcean — Designing Hardware-Aware Algorithms: FlashAttention](https://www.digitalocean.com/community/tutorials/flashattention)"
  - "[Galileo — How FlashAttention Eliminates Memory Bottlenecks](https://galileo.ai/blog/stanford-flashattention-algorithm)"
courses: []
---

# Flash Attention

## Зачем это нужно: attention — это memory bottleneck

Стандартный attention вычисляет три матричных операции:

$$\mathbf{S} = \mathbf{Q}\mathbf{K}^\top \in \mathbb{R}^{N \times N}, \quad \mathbf{P} = \text{softmax}(\mathbf{S}) \in \mathbb{R}^{N \times N}, \quad \mathbf{O} = \mathbf{P}\mathbf{V} \in \mathbb{R}^{N \times d}$$

Проблема: матрицы $\mathbf{S}$ и $\mathbf{P}$ имеют размер $N \times N$. Для GPT-2 ($N = 1024, d = 64$) это $1024 \times 1024 \approx 1M$ элементов, но для контекста 16K — это $256M$ элементов. Они **материализуются в GPU HBM** (видеопамять), и каждая операция (softmax, mask, dropout) заново читает и пишет эту огромную матрицу.

Почему это медленно? Потому что attention — **memory-bound** операция: время определяется не количеством FLOPs, а количеством обращений к памяти.

## Иерархия памяти GPU: ключевая интуиция

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/flash-attention/fig1-tiling.jpg]]
*Figure 1 из FlashAttention (Dao et al., 2022): слева — иерархия памяти GPU; в центре — tiling-алгоритм; справа — 7.6x ускорение на GPT-2.*

Иерархия памяти A100 GPU:

| Уровень | Объём | Пропускная способность | Аналогия |
|---------|-------|----------------------|----------|
| **SRAM** (on-chip) | ~20 MB (192KB x 108 SM) | **19 TB/s** | Рабочий стол |
| **HBM** (видеопамять) | 40-80 GB | 1.5-2.0 TB/s | Книжная полка |
| **DRAM** (CPU RAM) | >1 TB | 12.8 GB/s | Склад в другом здании |

SRAM в **~10x быстрее** HBM, но в **~1000x меньше**. Стандартный attention записывает и считывает $N \times N$ матрицу в HBM на каждом шаге — это bottleneck. FlashAttention перемещает вычисления в SRAM, работая с маленькими блоками данных.

**Compute-bound vs memory-bound** — критическое различие:
- **Compute-bound**: время определяется FLOPs (например, большие матричные умножения). GPU загружен на 100%.
- **Memory-bound**: время определяется чтением/записью в HBM. GPU простаивает, ожидая данных. **Attention — memory-bound.**

## Стандартная реализация: почему она медленная

```
Алгоритм 0: Standard Attention
1. Загрузить Q, K из HBM →  вычислить S = QKᵀ →  записать S в HBM     ← O(N²) HBM writes
2. Загрузить S из HBM →     вычислить P = softmax(S) →  записать P     ← O(N²) HBM reads/writes
3. Загрузить P, V из HBM →  вычислить O = PV →  записать O в HBM       ← O(N²) HBM reads
                                                                Итого: Θ(Nd + N²) HBM accesses
```

Три раза материализуем $N \times N$ матрицу в HBM. При $N = 4096$, каждая матрица в FP16 — это $4096^2 \times 2 = 32$ MB. Умножаем на количество голов и слоёв — gigabytes обращений к памяти.

## Алгоритм FlashAttention: tiling + online softmax

Главная идея: **никогда не материализовать $N \times N$ матрицу**. Вместо этого — разбить Q, K, V на блоки, загрузить их в SRAM и вычислить attention блочно.

### Шаг за шагом (Algorithm 1 из статьи)

**Подготовка:**
1. Вычислить размеры блоков: $B_c = \lceil M / 4d \rceil$, $B_r = \min(\lceil M / 4d \rceil, d)$, где $M$ — размер SRAM.
2. Разбить Q на $T_r = \lceil N / B_r \rceil$ блоков, K и V на $T_c = \lceil N / B_c \rceil$ блоков.
3. Инициализировать O = 0, $\ell$ = 0, $m = -\infty$ в HBM.

**Outer loop** (по блокам K, V):
```
for j = 1 to T_c:
    Загрузить K_j, V_j из HBM в SRAM                    ← один раз на блок
```

**Inner loop** (по блокам Q):
```
    for i = 1 to T_r:
        Загрузить Q_i, O_i, ℓ_i, m_i из HBM в SRAM
        
        В SRAM вычислить:
        S_ij = Q_i · K_j^T                              ← блочный скор B_r × B_c
        m̃_ij = rowmax(S_ij)                             ← поблочный максимум
        P̃_ij = exp(S_ij - m̃_ij)                        ← поблочный softmax (числитель)
        ℓ̃_ij = rowsum(P̃_ij)                            ← поблочная сумма
        
        Обновить статистики:
        m_i^new = max(m_i, m̃_ij)                        ← глобальный max
        ℓ_i^new = e^(m_i - m_i^new) · ℓ_i + e^(m̃_ij - m_i^new) · ℓ̃_ij
        
        Обновить выход:
        O_i ← diag(ℓ_i^new)^(-1) · (diag(ℓ_i) · e^(m_i - m_i^new) · O_i + e^(m̃_ij - m_i^new) · P̃_ij · V_j)
        
        Записать O_i, ℓ_i, m_i обратно в HBM
```

**Результат:** $\mathbf{O} = \text{softmax}(\mathbf{Q}\mathbf{K}^\top)\mathbf{V}$ — **exact** attention, математически идентичный стандартной реализации.

### Online Softmax Trick: ключевая математика

Проблема: softmax требует знать **все** значения для нормализации ($\sum_j e^{x_j}$), но у нас есть только текущий блок. Как вычислить корректный softmax блочно?

Трюк: храним две running-статистики для каждой строки:
- $m$ — текущий максимум (для численной стабильности)
- $\ell$ — текущая сумма экспонент

При обработке нового блока с максимумом $\tilde{m}$ и суммой $\tilde{\ell}$:

$$m^{\text{new}} = \max(m, \tilde{m})$$
$$\ell^{\text{new}} = e^{m - m^{\text{new}}} \cdot \ell + e^{\tilde{m} - m^{\text{new}}} \cdot \tilde{\ell}$$

Это **mathematically exact** — при последовательной обработке всех блоков получаем точно тот же результат, что и стандартный softmax по всей строке. Идея идёт из алгоритмического приёма «algebraic aggregation» (Milakov & Gimelshein, 2018).

### Kernel Fusion

Все операции — matmul (QK^T), softmax, masking, dropout, matmul (PV) — выполняются в **одном CUDA kernel**. Данные загружаются из HBM один раз, все вычисления происходят в SRAM, результат записывается обратно один раз.

Сравним: стандартный PyTorch attention запускает 5+ отдельных CUDA kernels, каждый из которых читает из HBM и пишет в HBM.

### Recomputation вместо хранения

Для backward pass стандартная реализация хранит $\mathbf{S}$ и $\mathbf{P}$ ($O(N^2)$ памяти). FlashAttention хранит только:
- Выход $\mathbf{O}$ ($N \times d$)
- Статистики softmax $m$ и $\ell$ ($N$ каждый)

В backward pass — **пересчитывает** $\mathbf{S}$ и $\mathbf{P}$ из блоков Q, K, V. Это добавляет FLOPs, но **ускоряет** backward, потому что избегает O(N^2) HBM reads.

Парадокс: **больше FLOPs, но быстрее**. Именно это значит «IO-aware» — оптимизировать по обращениям к памяти, а не по FLOPs.

## IO-сложность: доказательство оптимальности

**Стандартный attention:**

$$\Theta(Nd + N^2) \text{ обращений к HBM}$$

**FlashAttention:**

$$\Theta\left(\frac{N^2 d^2}{M}\right) \text{ обращений к HBM}$$

где $M$ — размер SRAM. Для типичных значений ($d = 64$, $M = 100\text{KB}$) FlashAttention требует **до 9x меньше** обращений к HBM.

**Нижняя граница** (Theorem 5 в статье): не существует exact attention алгоритма, который бы асимптотически улучшил количество HBM-обращений FlashAttention для всех размеров SRAM. То есть FlashAttention — **оптимален**.

## FlashAttention-2: удвоение скорости

FlashAttention-2 (Dao, 2023) — 2x ускорение через три оптимизации:

### 1. Уменьшение non-matmul FLOPs

На A100 non-matmul операции (softmax statistics, масштабирование) выполняются в **16x медленнее**, чем matmul (используют CUDA cores, а не Tensor cores). FA-2 переносит rescaling из inner loop во внешний, снижая количество non-matmul операций.

### 2. Параллелизация по sequence length

FA-1: outer loop по K/V, inner loop по Q. Каждый thread block обрабатывает один блок Q.
FA-2: **swap loops** — outer loop по Q, inner loop по K/V. Это позволяет параллелизировать по блокам Q, занимая больше SM (streaming multiprocessors) одновременно. Особенно важно при длинных последовательностях.

### 3. Устранение split-K между warps

FA-1: разные warps внутри thread block работают над разными частями K/V и синхронизируются через shared memory. FA-2: каждый warp обрабатывает полный блок K/V, устраняя синхронизацию.

**Результат:** FA-2 достигает **50-73% от теоретического максимума** TFLOPs/s (до 230 TFLOPs/s forward на A100). End-to-end GPT training: 225 TFLOPs/s, **72% Model FLOPs Utilization (MFU)**.

### Поддержка GQA/MQA

FA-2 нативно поддерживает [[02 Areas/ML & DL/Concepts/Inference/KV-Cache|Grouped-Query Attention и Multi-Query Attention]] без дублирования K/V head'ов в памяти. Правильное индексирование вместо expand+copy.

### Causal masking: пропуск 50% вычислений

При causal (autoregressive) attention ~50% блоков полностью замаскированы (будущие позиции). FA-2 **пропускает** эти блоки целиком, давая 1.7-1.8x ускорение для decoder-моделей.

## Бенчмарки

| Модель | Стандартный | FlashAttention | Ускорение |
|--------|-----------|----------------|-----------|
| GPT-2 (seq 1K) attention | baseline | FlashAttention | **7.6x** |
| GPT-2 (seq 1K) end-to-end | HuggingFace | FlashAttention | **3x** |
| BERT-large (seq 512) | MLPerf 1.1 record | FlashAttention | **15%** |
| Long-range arena (seq 1K-4K) | baseline | FlashAttention | **2.4x** |

**Качество:** FlashAttention **не меняет математику** attention — результат bit-for-bit идентичен (с точностью до floating point). Но позволяет тренировать с более длинным контекстом:
- GPT-2 с контекстом 4K (вместо 1K) — **0.7 лучше perplexity**
- Path-X (seq 16K): первый Transformer, который превышает chance level — **61.4%** accuracy
- Path-256 (seq 64K): **63.1%** accuracy — невозможно без FlashAttention

## Что FlashAttention НЕ делает

Важно понимать:
- **Не меняет сложность по FLOPs**: $O(N^2 d)$ — те же вычисления, что стандартный attention
- **Не является approximate attention**: в отличие от Linformer, Performer, BigBird — FlashAttention вычисляет **exact** attention
- **Не снижает O(N^2) по памяти для KV-cache**: FlashAttention оптимизирует training и prefill, но при inference [[02 Areas/ML & DL/Concepts/Inference/KV-Cache|KV-Cache]] всё равно растёт линейно

FlashAttention изменил perspective: вместо «давайте придумаем approximate attention с O(N)» — «давайте сделаем exact attention быстрым за счёт hardware-aware engineering».

## Хронология

| Год | Milestone | Статья |
|-----|-----------|--------|
| 2018 | Online softmax trick | Milakov & Gimelshein |
| 2020 | Memory-efficient attention (O(1) extra memory) | Rabe & Staats |
| **2022** | **FlashAttention: IO-aware tiling** | **Dao et al.** |
| 2023 | FlashAttention-2: 2x speedup | Dao |
| 2024 | FlashAttention-3: FP8, asynchronous tiling | Dao et al. |
| 2024 | Интеграция в PyTorch 2.0+ (torch.nn.functional.scaled_dot_product_attention) | PyTorch |

## Где используется

FlashAttention стал **де-факто стандартом**:
- **PyTorch 2.0+** — `torch.nn.functional.scaled_dot_product_attention` выбирает FlashAttention автоматически
- **HuggingFace Transformers** — `model.to(torch_dtype=torch.float16, attn_implementation="flash_attention_2")`
- **vLLM, TGI** — inference engines используют FA для prefill
- **xFormers** — Meta's library с FA
- **Triton** — OpenAI's compiler с FA reference implementation

## Related concepts

- [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]] — математика, которую FlashAttention ускоряет
- [[02 Areas/ML & DL/Concepts/Inference/KV-Cache|KV-Cache]] — кэширование K/V при inference, ортогонально к FlashAttention
- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] — полная архитектура
- [[02 Areas/ML & DL/Concepts/Architectures/Mamba|Mamba]] — альтернативный подход: заменить attention на O(N) SSM

## Дополнительные ресурсы

- [Dao et al. — FlashAttention (2022)](https://arxiv.org/abs/2205.14135) — оригинальная статья
- [Gordic — ELI5: Flash Attention](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad) — доступное объяснение с визуализациями
- [FlashAttention on a Napkin (2024)](https://arxiv.org/abs/2412.03317) — диаграммный подход к пониманию IO-awareness
- [GitHub: Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) — официальная реализация
