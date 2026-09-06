---
title: ZeRO, FSDP2, DeviceMesh и DTensor
type: textbook-chapter
status: canonical
last_updated: 2026-09-06
source_unit_id:
  - lecture-08-zero-sharding
  - lecture-08-zero-memory-accounting
  - assignment-02-task-optimizer-state-sharding
  - assignment-02-task-optimizer-state-sharding-accounting
  - assignment-02-task-fsdp
  - assignment-02-task-fsdp-accounting
  - assignment-02-task-fsdp-calcs
  - assignment-02-task-fsdp-tp-calcs
---

<a id="cs336-distributed-fsdp"></a>

# 44e. ZeRO, FSDP2, DeviceMesh и DTensor

AllGather/ReduceScatter из [[44a Processes, collectives и DDP]] и layouts из [[44c Tensor и sequence parallelism]] позволяют проследить parameter shard от покоя до materialization, gradient reduction, optimizer update и checkpoint. Это, в свою очередь, объясняет выбор unit boundary и prefetch без скрытого пика памяти.

DDP реплицирует параметры $P$, gradients $G$ и optimizer state $O$. ZeRO последовательно делит их по data-parallel rank:

| Режим | Реплицировано | Разделено | постоянная память на rank |
|---|---|---|---:|
| DDP | $P,G,O$ | — | $P+G+O$ |
| ZeRO-1 | $P,G$ | $O$ | $P+G+O/N$ |
| ZeRO-2 | $P$ | $G,O$ | $P+(G+O)/N$ |
| ZeRO-3 / full shard | — | $P,G,O$ | $(P+G+O)/N$ |

Чтобы таблица превратилась в расчёт, нужно назвать precision policy. В
иллюстративной схеме Stanford CS336 один параметр имеет BF16 weight (2 B), BF16
gradient (2 B) и два FP32 Adam moments (8 B): всего 12 B/parameter без master
FP32 weight, временных buffers, allocator fragmentation и активаций. При иной
policy — например, FP32 master weight или FP32 gradients — строку следует
пересчитать, а не переносить число 12.

| Режим при этой 12-B policy | Weight | Gradient | Adam state | Всего на rank |
|---|---:|---:|---:|---:|
| DDP | 2 | 2 | 8 | 12 B/param |
| ZeRO-1 | 2 | 2 | $8/N$ | $4+8/N$ B/param |
| ZeRO-2 | 2 | $2/N$ | $8/N$ | $2+10/N$ B/param |
| ZeRO-3 | $2/N$ | $2/N$ | $8/N$ | $12/N$ B/param |

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/l08-p17.png]]

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/l08-p22.png]]

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/l08-p25.png]]

*Первый кадр задаёт допущения для расчёта байтов на параметр; следующие
прослеживают разделение optimizer state, градиентов и параметров. Источник:
Stanford CS336
Spring 2026, Lecture 8, PDF pp. 17, 22 and 25, pinned commit
[`8b59b507`](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_08.pdf).
Это учебный ledger при указанной policy, не capacity oracle для реального job.
Для p. 17 использован body crop `[0, 205, 2200, 1215]`, для p. 25 —
`[0, 205, 2200, 1140]` из 220-dpi renders; в обоих случаях удалён только
заголовок, а схема и подписи сохранены полностью.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/distributed/zero-memory-partitioning.svg]]

*Источник: Harvard Edge ML Systems Book, [Distributed Training, figure `fig-zero-memory-partitioning`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/distributed_training/distributed_training.qmd), CC BY-NC-SA 4.0.*

## Время жизни полного параметра

В full shard перед forward FSDP-unit выполняет AllGather своих parameter shards. Полные параметры живут ровно до завершения вычисления unit и затем освобождаются. В backward они снова собираются; после вычисления gradient ReduceScatter оставляет каждому rank только его gradient shard.

```text
rest:       parameter shard
pre-fwd:    all-gather -> full parameter
post-fwd:   free/reshard -> parameter shard
pre-bwd:    all-gather -> full parameter
post-bwd:   reduce-scatter(full gradient) -> gradient shard; free full parameter
update:     local optimizer state updates local parameter shard
```

Для полного параметрического payload $M$ на rank ring all-gather передаёт $(N-1)M/N$, reduce-scatter — столько же. Если параметры собираются дважды (forward и backward), сетевой объём шага около

$$V_{\mathrm{rank}}\approx 3\frac{N-1}{N}M,$$

без учёта metadata: два AllGather и один ReduceScatter. Это не «бесплатная память», а обмен replication во времени.

Для ZeRO-1 при идеализированном ring и крупном сообщении итоговый объём
communication может совпасть с DDP: reduce-scatter gradient и all-gather
обновлённых parameter shards заменяют all-reduce. Однако startup latency,
доступ optimizer к параметрам, topology, packing, временные buffers и
fragmentation никуда не исчезают. Поэтому «ZeRO-1 бесплатен» — полезная
интуиция о leading-order bytes, но не инженерная гарантия.

### Peak-memory synchronization

Пусть два соседних FSDP units имеют 6 и 8 GiB полных параметров, local shards по 0.75 и 1 GiB. Aggressive forward prefetch второго до освобождения первого создаёт минимум $6+8=14$ GiB собранных параметров; последовательный lifetime — максимум 8 GiB. Prefetch скрывает сеть только пока дополнительный пик помещается.

Forward prefetch запускает AllGather следующего unit во время compute текущего. Backward prefetch может собирать предыдущий по reverse order. Rate limiter ограничивает число незавершённых all-gathers; boundaries FSDP unit определяют компромисс: крупный unit — крупный пик, мелкий — больше latency и hooks.

## FSDP2, DeviceMesh и DTensor

FSDP2 выражает sharding per-parameter через DTensor и регистрирует pre/post-forward и backward hooks, которые reshard/unshard состояние в момент использования. Это делает composability с TP явной, но hooks обязаны запускаться в одинаковом порядке на всех rank.

`DeviceMesh` именует оси, например `(dp=16,tp=8)`, вместо ручной арифметики rank. `DTensor` связывает global shape с placements:

- `Replicate()` — полный tensor на каждом rank оси;
- `Shard(dim)` — части global dimension;
- `Partial(reduce_op)` — локальные частичные результаты, требующие редукции.

Пример: weight $[8192,32768]$ BF16 на mesh `dp=4,tp=8` может иметь placements `[Shard(0) по dp, Shard(1) по tp]`; local shape $[2048,4096]$, то есть $1/32$ global storage. Но matmul layout propagation может потребовать redistribute — фактический collective должен быть виден в профиле.

Hybrid sharding делит параметры внутри группы из $S$ rank и реплицирует shard-группы $R$ раз, $N=SR$. Это уменьшает дорогой межузловой AllGather, сохраняя часть экономии: постоянное state порядка $(P+G+O)/S$, а не `/N`.

```text
for unit in forward_order:
    all_gather(unit.parameter_shards)
    prefetch(next_unit, max_in_flight=1)
    output = unit.forward(input)
    reshard(unit.parameters)
for unit in reverse_order:
    all_gather(unit.parameter_shards)
    input_grad, full_grad = unit.backward(output_grad)
    grad_shard = reduce_scatter(full_grad)
    reshard(unit.parameters)
optimizer.step(local_parameter_shard, grad_shard, local_state)
```

Реальный FSDP2 выполняет переходы hooks. Если control flow различается между ranks, один может войти в AllGather следующего unit, пока другой ожидает ReduceScatter текущего: порядок hooks — часть протокола.

## Контракт корректности custom FSDP

Официальный тест Assignment 2 проверяет не только final loss. Он сравнивает
custom wrapper с unwrapped model при одинаковой инициализации, собирает полный
state через `fsdp_gather_full_params`, а отдельный параметризованный тест
проверяет gradient synchronization для разных `compute_dtype`. После backward
`fsdp_on_after_backward` обязан завершить ожидающие collectives до optimizer
step. Следовательно, приемлемая реализация должна доказать четыре инварианта:

1. у каждого parameter shard есть один владелец, а сборка восстанавливает
   точную исходную форму;
2. forward/backward собирает полный параметр только на оговорённое время;
3. gradient shard имеет ту же нормировку, что dense reference;
4. mixed-precision communication не меняет master-state ownership и остаётся в
   оговорённом численном допуске.

Только после этих сравнений имеет смысл измерять peak memory, communication
time и overlap. Уменьшившийся allocator peak при неверном gradient — не
оптимизация.

## Distributed Checkpoint

Checkpoint сохраняет global logical DTensor независимо от текущего placement. Каждый rank пишет shards, planner фиксирует global shapes, dtype, keys и mapping. При restore на другой mesh shards перераспределяются. Обязательны optimizer/scheduler/scaler, RNG и data position; проверка — следующий шаг, а не успешный `load`.

Для 70B-модели условный state 1.12 TB. При aggregate storage bandwidth 200 GB/s идеальный lower bound 5.6 s; если каждый из 64 rank создаёт тысячи файлов, metadata и contention увеличат время. Поэтому distributed save использует крупные shards и staging.

## Практика и первоисточники

- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week05_fsdp/lecture.pdf|EDLS Week 5 — лекция]];
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week05_fsdp/seminar.pdf|EDLS Week 5 — seminar slides]];
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week05_fsdp/homework/README|EDLS Week 5 — практическое задание]];
- [[02 Areas/ML & DL/06 Практика/12 Собрать и проверить FSDP|практика FSDP и переносимого checkpoint]].

Пиковую память определяют моменты, когда AllGather материализует полные параметры,
и то, как долго они остаются в памяти до освобождения. Поэтому границы
FSDP-модулей выбирают по времени жизни временно собранных параметров, а не только
по удобству программного интерфейса.

- EDLS, pinned commit `e632aa89…`, [`week05_fsdp/lecture.pdf`, PDF pp. 13–31 FSDP units/lifetimes/overlap, pp. 54–70 sharding levels/ZeRO/hybrid/FSDP2](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week05_fsdp/lecture.pdf), and [`week05_fsdp/seminar.pdf`, PDF pp. 5–10 DeviceMesh/DTensor, pp. 11–20 FSDP2/hooks/memory, pp. 35–36 PyTorch DCP](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week05_fsdp/seminar.pdf).
- Rajbhandari et al., [ZeRO](https://arxiv.org/abs/1910.02054), 2019.
- PyTorch, [FSDP2 tutorial, “How FSDP2 works” and “2D parallelism”](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html), [fully_shard API](https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html), и [Distributed Checkpoint, `get_state_dict`/`set_state_dict`](https://pytorch.org/docs/stable/distributed.checkpoint.html).
- Harvard Edge ML Systems Book, commit `45ecc8d…`, [Distributed Training, `sec-distributed-training-systems-systems-zero-redundancy-optimizer-zero-20bd` and `sec-distributed-training-systems-systems-fully-sharded-data-parallel-fsdp-79a3`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/distributed_training/distributed_training.qmd).
- Stanford CS336 Spring 2026, [Lecture 8](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_08.pdf), PDF pp. 14–30.
- Stanford CS336 Spring 2026, [Assignment 2](https://github.com/stanford-cs336/assignment2-systems/tree/ca8bc81a59b70516f7ebb2da4808daade877c736), optimizer-state sharding and FSDP tasks plus official tests.

← [[44d Pipeline parallelism]] · Далее: [[44f Expert и hybrid parallelism]]
