---
title: Gradient checkpointing и offload
type: textbook-chapter
status: canonical
last_updated: 2026-07-24
---

# 44b. Gradient checkpointing и offload

Backward требует промежуточные значения forward. Если сохранять активации всех $L$ блоков, память грубо растёт как $O(LBSD)$ для batch $B$, sequence $S$ и hidden size $D$. Gradient checkpointing оставляет только границы сегментов и повторяет внутренний forward во время backward.

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

## Offload — другая граница

CPU offload переносит параметры, gradients, optimizer state или сохранённые активации через PCIe/NVLink-C2C; disk/NVMe offload добавляет ещё более медленный уровень. Для тензора $X$ время не меньше

$$T_{\mathrm{move}}\ge \frac{|X|}{BW_{\mathrm{link}}}+\alpha.$$

Перенос 16 GiB по эффективным 24 GB/s PCIe занимает не менее 0.67 s в одну сторону. Если независимое вычисление следующего участка длится 0.4 s, минимум 0.27 s останется открытым. Offload экономит HBM, но не байты и не время.

Async prefetch запускает H2D следующего состояния до использования, а eviction — после последнего использования текущего. Двойной буфер требует HBM для current + next. Слишком дальний prefetch снова повышает peak memory; слишком поздний создаёт stall.

Для NVMe нужен staging через pinned host memory, крупные последовательные операции и контроль queue depth. Тысячи мелких файлов превращают задачу bandwidth в metadata bottleneck.

## Инженерный выбор

Сначала измеряют пик по категориям. Если доминируют активации — checkpointing; optimizer state — ZeRO-1/offload; параметры — FSDP/ZeRO-3; временный tensor одного operator — изменить kernel/TP. Комбинация допустима, но её проверяют timeline: checkpointing увеличивает compute window и иногда помогает скрыть prefetch, а иногда конкурирует с ним за memory bandwidth.

Проверка состоит из двух частей. Численная: logits, loss и gradients с сохранением RNG совпадают с baseline в выбранном допуске. Системная: peak allocated/reserved HBM, bytes D2H/H2D, recompute FLOPs и exposed transfer time сняты с одного и того же шага.

## Источники

- EDLS, [week 4](https://github.com/mryab/efficient-dl-systems), activation checkpointing and offload.
- Chen et al., [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174), 2016.
- PyTorch, [activation checkpointing documentation](https://pytorch.org/docs/stable/checkpoint.html).

← [[44a Processes, collectives и DDP]] · Далее: [[44c Tensor и sequence parallelism]]
