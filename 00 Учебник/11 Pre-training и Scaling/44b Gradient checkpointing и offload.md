---
title: Gradient checkpointing и offload
type: textbook-chapter
status: canonical
last_updated: 2026-07-24
---

# 44b. Gradient checkpointing и offload

Backward требует промежуточные значения forward. Если сохранять активации всех $L$ блоков, память грубо растёт как $O(LBSD)$ для batch $B$, sequence $S$ и hidden size $D$. Gradient checkpointing оставляет только границы сегментов и повторяет внутренний forward во время backward.

## Что нужно знать и чему научимся

Нужны устройство Transformer-блока, automatic differentiation и различие HBM, host DRAM и NVMe. После главы можно составить ledger сохранённых tensors, выбрать selective policy по bytes/FLOPs, построить prefetch timeline и доказать, что оптимизация не изменила gradient.

## Что именно занимает память

Пиковый ledger следует снимать по времени:

| Состояние | Масштаб | Время жизни |
|---|---:|---|
| параметры | $P$ | весь шаг или внутри FSDP unit |
| gradients | $P$ | от backward слоя до update |
| Adam moments | $2P$ FP32 | весь шаг |
| активации | $\sim LBSD$ с коэффициентами операторов | forward → backward слоя |
| temporaries | зависят от kernel | один operator |

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/training-systems-courses/44-cs336-dp-memory.png]]

*Источник: Percy Liang, Tatsu Hashimoto и команда Stanford, CS336, лекция 8, слайд 17: [PDF](https://github.com/stanford-cs336/lectures/blob/main/lecture_08.pdf). Рисунок отделяет постоянные состояния обучения от активаций, которыми управляет checkpointing.*

Пример: 48 блоков сохраняют по 256 MiB активаций — 12 GiB. Разбиение на восемь сегментов сохраняет около восьми границ, 2 GiB, но повторно вычисляет внутренние блоки. Это верхнеуровневая оценка: FlashAttention, dropout masks и fused kernels меняют ledger.

## Полное и выборочное сохранение

Равномерное checkpointing минимизирует число живых границ примерно при сегментах порядка $\sqrt L$, но одинаково обращается с дешёвой нормировкой и дорогой attention. Selective checkpointing сохраняет выход дорогих или недетерминированных операторов, а дешёвые elementwise-операции пересчитывает. Решение принимают по паре «saved bytes / recompute FLOPs», а не по имени слоя.

Повторный forward должен воспроизвести RNG. Иначе dropout mask меняется и gradient уже не соответствует исходному forward. In-place mutation и stateful layers тоже нарушают контракт.

Naive attention материализует score/probability tensors порядка $O(BHS^2)$, тогда как FlashAttention выполняет tiled online softmax и для backward сохраняет главным образом входы, выход и статистики нормировки порядка $O(BHSD)$, повторяя tile-local промежуточные значения. Поэтому $O(LBSD)$ — не замена полному ledger: attention backend нужно указать отдельно.

| Operator | Сохранить | Пересчитать | Причина |
|---|---|---|---|
| LayerNorm | input/statistics | affine output | statistics нужны backward |
| QKV projection | block input | Q/K/V | освобождаются три больших tensors |
| FlashAttention | Q/K/V, output, log-sum-exp по kernel contract | tile probabilities | полный $S^2$ tensor не сохраняется |
| MLP | input | first GEMM + activation | много saved bytes, но дорогой recompute |
| dropout | RNG state/counter | mask | воспроизводимость без хранения mask |

```text
for segment in transformer_segments:
    boundary = checkpoint(segment_forward, boundary,
                          preserve_rng_state=true)
loss = head(boundary)
backward(loss)  # runtime повторяет forward сегмента перед его backward
```

## Offload — другая граница

CPU offload переносит параметры, gradients, optimizer state или сохранённые активации через PCIe/NVLink-C2C; disk/NVMe offload добавляет ещё более медленный уровень. Для тензора $X$ время не меньше

$$T_{\mathrm{move}}\ge \frac{|X|}{BW_{\mathrm{link}}}+\alpha.$$

Перенос 16 GiB, то есть $17.18$ GB, по эффективным 24 GB/s PCIe занимает не менее $17.18/24=0.716$ s в одну сторону. Если независимое вычисление следующего участка длится 0.4 s, минимум 0.316 s останется открытым. Offload экономит HBM, но не байты и не время.

Async prefetch запускает H2D следующего состояния до использования, а eviction — после последнего использования текущего. Двойной буфер требует HBM для current + next. Слишком дальний prefetch снова повышает peak memory; слишком поздний создаёт stall.

Для NVMe нужен staging через pinned host memory, крупные последовательные операции и контроль queue depth. Тысячи мелких файлов превращают задачу bandwidth в metadata bottleneck.

### Worked configuration

Пусть 48 блоков дают 12 GiB saved activations, а бюджет после weights и buffers — 4 GiB. Восемь checkpoint-сегментов оставляют около 2 GiB boundaries. Два 512-MiB pinned buffers поддерживают double buffering; prefetch сегмента $j+1$ запускается при вычислении $j$, eviction $j-1$ — только после его backward. Peak равен `boundaries + current + next + temporaries`, а не одним boundaries. Если trace показывает три prefetched сегмента, policy нарушает рассчитанный бюджет.

## Инженерный выбор

Сначала измеряют пик по категориям. Если доминируют активации — checkpointing; optimizer state — ZeRO-1/offload; параметры — FSDP/ZeRO-3; временный tensor одного operator — изменить kernel/TP. Комбинация допустима, но её проверяют timeline: checkpointing увеличивает compute window и иногда помогает скрыть prefetch, а иногда конкурирует с ним за memory bandwidth.

Проверка состоит из двух частей. Численная: logits, loss и gradients с сохранением RNG совпадают с baseline в выбранном допуске. Системная: peak allocated/reserved HBM, bytes D2H/H2D, recompute FLOPs и exposed transfer time сняты с одного и того же шага.

## Источники

- EDLS, [week 4](https://github.com/mryab/efficient-dl-systems), activation checkpointing and offload.
- Chen et al., [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174), 2016.
- PyTorch, [activation checkpointing documentation](https://pytorch.org/docs/stable/checkpoint.html).
- PyTorch, [Selective Activation Checkpointing, sections “Selective Activation Checkpoint” and “Memory Budget API”](https://pytorch.org/blog/activation-checkpointing-techniques/).
- Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135), §3.2–3.3 and Algorithm 1.

← [[44a Processes, collectives и DDP]] · Далее: [[44c Tensor и sequence parallelism]]
