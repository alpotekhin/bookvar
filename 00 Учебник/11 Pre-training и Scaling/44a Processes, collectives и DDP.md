---
title: Processes, collectives и DDP
type: textbook-chapter
status: canonical
last_updated: 2026-07-24
---

# 44a. Processes, collectives и DDP

Один процесс обслуживает одно устройство. Его `rank` — номер в группе, `world_size=N` — число участников; группы позволяют выполнять разные collectives на разных осях параллелизма. Point-to-point `send/recv` задают обмен явно, collective выражает общий шаблон и позволяет библиотеке выбрать алгоритм.

## Полный маршрут по исходным материалам

- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week03_data_parallel/lecture.pdf|EDLS Week 3 — полная лекция]];
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week03_data_parallel/practice.ipynb|EDLS Week 3 — исходный practice notebook]];
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week03_data_parallel/homework/README|EDLS Week 3 — homework]];
- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol2/collective_communication|Harvard CS249r — Collective Communication]];
- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol2/distributed_training|Harvard CS249r — Distributed Training]].

Лекцию EDLS следует проходить вместе с notebook: схемы ring, gossip и gradient
compression становятся проверяемыми только после измерения message size,
latency и effective bandwidth. Harvard дополняет эксперимент систематическим
разбором топологий и collective algorithms.

## Что нужно знать и чему научимся

Нужны только tensors и synchronous data parallelism. После главы можно восстановить shape/state переход любого collective, оценить latency/bandwidth lower bound, написать DDP-step и отличить полезное overlap от нарисованного profiler-ом суммарного времени.

## От сообщений к коллективным операциям

Broadcast копирует тензор одного rank всем; reduce собирает результат у root; all-reduce возвращает редукцию каждому; all-gather собирает все shards каждому; reduce-scatter одновременно редуцирует и оставляет rank только его shard; all-to-all пересылает каждому rank отдельную часть. Последняя операция особенно важна для MoE.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/distributed/collective-primitives-overview.svg]]

*Источник: Harvard Edge ML Systems Book, [Collective Communication, figure `fig-collective-primitives-overview`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/collective_communication/collective_communication.qmd), CC BY-NC-SA 4.0.*

Для сообщения $M$ байт простая модель

$$T\approx \alpha n_{\mathrm{rounds}}+\beta V,\qquad \beta=1/BW_{\mathrm{effective}}.$$

В ring all-reduce тензор делят на $N$ блоков. Reduce-scatter занимает $N-1$ шагов, затем all-gather ещё $N-1$. Каждый rank передаёт

$$V_{\mathrm{ring}}=2\frac{N-1}{N}M,$$

а время примерно $2(N-1)\alpha+2\frac{N-1}{N}M\beta$. При $N=8$, $M=1$ GiB это $1.75$ GiB на rank. На эффективных 25 GB/s bandwidth-член равен примерно 70 ms; при $\alpha=5\,\mu s$ startup добавляет лишь 70 μs. Для 4 KiB всё наоборот: latency доминирует.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/distributed/ring-allreduce.svg]]

*Источник: Harvard Edge ML Systems Book, [Collective Communication, figure `fig-ring-allreduce`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/collective_communication/collective_communication.qmd), CC BY-NC-SA 4.0.*

Butterfly/recursive doubling соединяет rank с партнёром, отличающимся одним битом номера, и завершает редукцию за $\log_2N$ раундов. Она привлекательна для малых сообщений, где сокращение раундов важнее полного использования кольцевой bandwidth. Иерархический алгоритм сначала редуцирует внутри NVLink-узла, затем между узлами и снова распространяет локально.

## DDP как replicated state machine

DDP копирует параметры и optimizer state, раздаёт разные микропакеты и во время backward объединяет градиенты. Параметры группируют в buckets: как только bucket готов, его all-reduce перекрывается с вычислением более ранних слоёв. Но bucket, готовый слишком поздно, остаётся на critical path.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/distributed/data-parallel-flow.svg]]

*Источник: Harvard Edge ML Systems Book, [Distributed Training, figure `fig-data-parallel-flow`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/distributed_training/distributed_training.qmd), CC BY-NC-SA 4.0.*

Если каждый rank вычисляет средний градиент по $b$ примерам, то после sum all-reduce нужно делить на $N$ (или доверить это реализации), чтобы получить градиент global batch $Nb$. Нельзя одновременно суммировать loss и ещё раз делить gradient: это меняет learning rate.

```text
initialize_process_group(rank, world_size)
model = replicate_same_parameters()
for local_batch in distributed_sampler():
    loss = model(local_batch) / gradient_accumulation_steps
    backward(loss)                 # готовые buckets запускают async all-reduce
    if accumulation_boundary:
        wait_all_buckets()
        optimizer.step()
        optimizer.zero_grad()
```

Gradient переходит из состояния `local partial` в `globally averaged replica`, не меняя shape. При accumulation collective либо подавляют до последнего microbatch, либо его стоимость умножается на число microbatches.

### Полезность масштабирования

Пусть backward занимает 180 ms, а 1-GiB ring — 70 ms. Если 55 ms скрыты вычислением, шаг платит 15 ms. Добавление GPU, уменьшившее compute до 100 ms, может открыть уже 40 ms communication: speedup становится сублинейным. Измерять нужно exposed collective time, а не сумму длительностей NCCL kernels.

## Когда точный all-reduce слишком дорог

Gossip усредняет параметры или градиенты только с соседями. Один раунд дешевле глобального collective, но консенсус достигается постепенно и зависит от спектральных свойств графа.

Сжатие обязано учитывать потерянный остаток. При error feedback:

$$u_t=g_t+e_t,\quad q_t=C(u_t),\quad e_{t+1}=u_t-q_t.$$

Без $e_t$ систематически отброшенные малые компоненты исчезают навсегда. PowerSGD аппроксимирует матричный gradient $G\in\mathbb R^{m\times n}$ как $PQ^\top$ ранга $r$. Вместо $mn$ чисел передаются $r(m+n)$; для $4096^2$ и $r=8$ коэффициент сжатия $16{,}777{,}216/65{,}536=256$. Цена — две малые collective, ортогонализация и возможная ошибка оптимизации; embedding и bias обычно требуют другого пути.

## Проверка

На двух rank с одинаковым seed сравнивают один DDP-step с single-process global batch: loss до update, усреднённые gradients и параметры после update. Затем искусственно задерживают один rank: ожидаемый результат не меняется, а step time показывает straggler amplification.

## Источники

- EDLS, pinned commit `e632aa89…`, [`week03_data_parallel/lecture.pdf`, PDF pp. 24–35 “All-Reduce data parallel”/“Faster allreduce”/Ring, pp. 39–51 Gossip, pp. 52–62 gradient compression/Error Feedback/PowerSGD](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week03_data_parallel/lecture.pdf).
- Harvard Edge ML Systems Book, [Collective Communication](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/collective_communication/collective_communication.qmd), sections `sec-collective-communication-primitives`, `sec-collective-communication-allreduce`.
- Harvard Edge ML Systems Book, commit `45ecc8d…`, [Distributed Training, `sec-distributed-training-systems-systems-data-parallelism-0c8f`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/distributed_training/distributed_training.qmd).
- Vogels et al., [PowerSGD](https://arxiv.org/abs/1905.13727), 2019.

← [[44 Distributed training и mixed precision]] · Далее: [[44b Gradient checkpointing и offload]]
