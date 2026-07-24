---
title: Expert и hybrid parallelism
type: textbook-chapter
status: canonical
last_updated: 2026-07-24
---

# 44f. Expert и hybrid parallelism

Архитектуру router, auxiliary loss и capacity разбирает [[02 Mixture of Experts — routing, capacity и serving]]. Здесь вопрос системный: как доставить выбранные токены владельцам экспертов и вернуть outputs в исходный порядок.

## Dispatch, compute, combine

После top-$k$ routing каждый source rank группирует токены по destination expert. Первый all-to-all делает dispatch, локальные experts выполняют MLP, второй возвращает outputs. Metadata обязана сохранить `(source rank, token index, expert slot, gate weight)`.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/distributed/moe-all-to-all-routing.svg]]

*Источник: Harvard Edge ML Systems Book, [Distributed Training, figure `fig-moe-all-to-all-routing`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/distributed_training/distributed_training.qmd), CC BY-NC-SA 4.0.*

При $T$ токенах, hidden $D$, BF16 и top-$k$, logical activation payload одного направления порядка $2kTD$ байт до padding/metadata. $T=8192,D=4096,k=2$ даёт 128 MiB для dispatch и ещё 128 MiB для combine в глобальном tensor view.

## Дисбаланс — это время, а не только loss

Если средняя нагрузка $Tk/E$, а самый загруженный expert получает $L_{\max}$, compute imbalance

$$\rho=\frac{L_{\max}}{Tk/E}.$$

При $T=8192,k=2,E=64$ среднее 256 токенов. Expert с 512 задаёт $\rho=2$: синхронный слой ждёт его, даже если сеть идеальна. Capacity factor ограничивает allocation, но overflow ведёт к drop или reroute — это уже изменение вычисления и требует quality check.

GroupedGEMM объединяет experts с разными числами строк в один запуск: список матриц разделяет weights, а токеновые segments различаются. Это уменьшает launch overhead и padding относительно отдельных GEMM; слишком малые expert batches всё равно плохо используют tensor cores.

## TP или EP

TP делит **каждый** expert и требует collectives внутри его MLP; EP хранит целых experts и платит all-to-all токенов. TP помогает, если один expert не помещается или его GEMM достаточно велика; EP — если experts помещаются, а суммарные weights нет. Обычно EP пересекает узлы, TP остаётся внутри NVLink node. Некоторые Megatron-конфигурации требуют sequence parallelism при совместном TP+EP: оси нельзя выбирать независимо от layout invariants реализации.

## Гибридный и 3D-parallelism

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/distributed/3d-parallelism-cube.svg]]

*Источник: Harvard Edge ML Systems Book, [Distributed Training, figure `fig-3d-parallelism-cube`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/distributed_training/distributed_training.qmd), CC BY-NC-SA 4.0.*

Для 1024 GPU пример mesh: `DP=16 × PP=8 × TP=8`. В MoE часть DP-оси может стать `EP=8 × EDP=2`; произведение физических осей всё равно 1024, а process groups различны для dense и expert layers. Порядок размещения следует частоте обменов: TP — внутри узла, EP — внутри полной-bandwidth rail group, PP — между соседними topology domains, DP — по оставшейся оси.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/distributed/parallelism-decision-tree.svg]]

*Источник: Harvard Edge ML Systems Book, [Distributed Training, figure `fig-parallelism-decision-tree`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/distributed_training/distributed_training.qmd), CC BY-NC-SA 4.0.*

Проверка MoE системы сравнивает token-to-expert assignments, counts per expert, reconstructed output и weight-gradients с single-rank implementation. В профиле нужны all-to-all bytes, skew $\rho$, доля padding/dropped tokens и GroupedGEMM utilization.

## Источники

- EDLS, [week 6](https://github.com/mryab/efficient-dl-systems), expert and hybrid parallelism.
- Harvard Edge ML Systems Book, [Distributed Training](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/distributed_training/distributed_training.qmd), sections `sec-distributed-training-model-parallelism` and `sec-distributed-training-hybrid`.
- NVIDIA Megatron Core, [parallelism strategies](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html).

← [[44e ZeRO, FSDP2, DeviceMesh и DTensor]] · Далее: [[44g Network, storage и distributed checkpoints]]
