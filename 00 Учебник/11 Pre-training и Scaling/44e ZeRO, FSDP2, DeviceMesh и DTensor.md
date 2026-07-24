---
title: ZeRO, FSDP2, DeviceMesh и DTensor
type: textbook-chapter
status: canonical
last_updated: 2026-07-24
---

# 44e. ZeRO, FSDP2, DeviceMesh и DTensor

## Полный маршрут EDLS

- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week05_fsdp/lecture.pdf|EDLS Week 5 — полная лекция]];
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week05_fsdp/seminar.pdf|EDLS Week 5 — seminar slides]];
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week05_fsdp/homework/README|EDLS Week 5 — homework]];
- [[02 Areas/ML & DL/06 Практика/12 Собрать и проверить FSDP|практика FSDP и переносимого checkpoint]].

Здесь особенно важно не ограничиваться API: lecture и seminar показывают
lifetime полных параметров во время all-gather и reduce-scatter. Именно эти
временные materializations объясняют peak memory и выбор wrap policy.

DDP реплицирует параметры $P$, gradients $G$ и optimizer state $O$. ZeRO последовательно делит их по data-parallel rank:

## Что нужно знать и чему научимся

Нужны AllGather/ReduceScatter из 44a и layouts из 44c. Цель — проследить parameter shard от покоя до materialization, gradient reduction, optimizer update и checkpoint, а затем выбрать unit boundary и prefetch без скрытого peak.

| Режим | Реплицировано | Разделено | постоянная память на rank |
|---|---|---|---:|
| DDP | $P,G,O$ | — | $P+G+O$ |
| ZeRO-1 | $P,G$ | $O$ | $P+G+O/N$ |
| ZeRO-2 | $P$ | $G,O$ | $P+(G+O)/N$ |
| ZeRO-3 / full shard | — | $P,G,O$ | $(P+G+O)/N$ |

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/distributed/zero-memory-partitioning.svg]]

*Источник: Harvard Edge ML Systems Book, [Distributed Training, figure `fig-zero-memory-partitioning`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/distributed_training/distributed_training.qmd), CC BY-NC-SA 4.0.*

## Время жизни полного параметра

В full shard перед forward FSDP-unit выполняет AllGather своих parameter shards. Полные параметры живут ровно до завершения вычисления unit и затем освобождаются. В backward они снова собираются; после вычисления gradient ReduceScatter оставляет каждому rank только его gradient shard.

Для полного параметрического payload $M$ на rank ring all-gather передаёт $(N-1)M/N$, reduce-scatter — столько же. Если параметры собираются дважды (forward и backward), сетевой объём шага около

$$V_{\mathrm{rank}}\approx 3\frac{N-1}{N}M,$$

без учёта metadata: два AllGather и один ReduceScatter. Это не «бесплатная память», а обмен replication во времени.

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

## Distributed Checkpoint

Checkpoint сохраняет global logical DTensor независимо от текущего placement. Каждый rank пишет shards, planner фиксирует global shapes, dtype, keys и mapping. При restore на другой mesh shards перераспределяются. Обязательны optimizer/scheduler/scaler, RNG и data position; проверка — следующий шаг, а не успешный `load`.

Для 70B-модели условный state 1.12 TB. При aggregate storage bandwidth 200 GB/s идеальный lower bound 5.6 s; если каждый из 64 rank создаёт тысячи файлов, metadata и contention увеличат время. Поэтому distributed save использует крупные shards и staging.

## Источники

- EDLS, pinned commit `e632aa89…`, [`week05_fsdp/lecture.pdf`, PDF pp. 13–31 FSDP units/lifetimes/overlap, pp. 54–70 sharding levels/ZeRO/hybrid/FSDP2](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week05_fsdp/lecture.pdf), and [`week05_fsdp/seminar.pdf`, PDF pp. 5–10 DeviceMesh/DTensor, pp. 11–20 FSDP2/hooks/memory, pp. 35–36 PyTorch DCP](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week05_fsdp/seminar.pdf).
- Rajbhandari et al., [ZeRO](https://arxiv.org/abs/1910.02054), 2019.
- PyTorch, [FSDP2 tutorial, “How FSDP2 works” and “2D parallelism”](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html), [fully_shard API](https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html), и [Distributed Checkpoint, `get_state_dict`/`set_state_dict`](https://pytorch.org/docs/stable/distributed.checkpoint.html).
- Harvard Edge ML Systems Book, commit `45ecc8d…`, [Distributed Training, `sec-distributed-training-systems-systems-zero-redundancy-optimizer-zero-20bd` and `sec-distributed-training-systems-systems-fully-sharded-data-parallel-fsdp-79a3`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/distributed_training/distributed_training.qmd).

← [[44d Pipeline parallelism]] · Далее: [[44f Expert и hybrid parallelism]]
