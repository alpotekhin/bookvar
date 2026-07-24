---
title: Expert и hybrid parallelism
type: textbook-chapter
status: canonical
last_updated: 2026-07-24
---

# 44f. Expert и hybrid parallelism

Архитектуру router, auxiliary loss и capacity разбирает [[02 Mixture of Experts — routing, capacity и serving]]. Здесь вопрос системный: как доставить выбранные токены владельцам экспертов и вернуть outputs в исходный порядок.

## Что нужно знать и чему научимся

Нужны all-to-all из 44a, TP/SP из 44c и process meshes из 44e. После главы можно проследить token state через dispatch/GroupedGEMM/combine, рассчитать payload и imbalance, а затем разместить EP вместе с TP, PP и DP без нарушения divisibility constraints.

## Dispatch, compute, combine

После top-$k$ routing каждый source rank группирует токены по destination expert. Первый all-to-all делает dispatch, локальные experts выполняют MLP, второй возвращает outputs. Metadata обязана сохранить `(source rank, token index, expert slot, gate weight)`.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/distributed/moe-all-to-all-routing.svg]]

*Источник: Harvard Edge ML Systems Book, [Distributed Training, figure `fig-moe-all-to-all-routing`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/distributed_training/distributed_training.qmd), CC BY-NC-SA 4.0.*

При $T$ токенах, hidden $D$, BF16 и top-$k$, logical activation payload одного направления порядка $2kTD$ байт до padding/metadata. $T=8192,D=4096,k=2$ даёт 128 MiB для dispatch и ещё 128 MiB для combine в глобальном tensor view.

```text
routes = topk(router(hidden), k)
packed, inverse_map = pack_by_destination(hidden, routes)
received = all_to_all(packed, ep_group)
expert_outputs = grouped_gemm(received, local_expert_weights)
returned = all_to_all(expert_outputs, ep_group)
output = unpack_and_weight(returned, inverse_map, routes.gates)
```

State transition имеет вид `[token,D] → [destination,slot,D] → [expert,row,D] → [token,k,D] → [token,D]`. Ошибка inverse map может оставить правильные counts и конечный shape, но перемешать outputs, поэтому проверяют token IDs.

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

### Worked configuration: 1024 GPU

Dense baseline `DP=16 × PP=8 × TP=8` преобразуем для MoE в четырёхмерный mesh

```text
(edp=2, ep=8, pp=8, tp=8), world_size = 2*8*8*8 = 1024
rank(edp,ep,pp,tp) = (((edp*8 + ep)*8 + pp)*8 + tp)
```

Например, rank с координатой `(1,3,5,6)` имеет номер 750. Его группы задаются фиксацией трёх координат:

- TP: `(1,3,5,0..7)`, восемь shards одной матрицы;
- PP: `(1,3,0..7,6)`, восемь последовательных стадий;
- EP: `(1,0..7,5,6)`, восемь владельцев разных experts;
- EDP: `(0..1,3,5,6)`, две replicas одного expert shard;
- dense-DP: `(0..1,0..7,5,6)`, 16 replicas dense TP-shard на данной PP-stage.

Такое перечисление важно: «группа из восьми» без координат легко случайно смешивает PP-stage или TP-shard и коллективная операция формально завершается над несовместимыми tensors.

Пусть PP-stage содержит dense-параметры $P_d$ и MoE-параметры $P_e$ для 64 experts. Dense часть на rank занимает $P_d/TP=P_d/8$ элементов и реплицируется по `EP×EDP=16`; её gradients редуцируются в dense-DP group. Каждый EP-rank владеет восемью experts, а TP делит матрицы каждого из них, поэтому expert weights на rank — $P_e/(EP\cdot TP)=P_e/64$ элементов. Они реплицированы только по EDP=2, и expert-gradient AllReduce идёт по двухранговой EDP group. Optimizer state следует тем же ownership rules.

### Forward и backward в этой конфигурации

Одна EDP-replica подаёт в EP-group $T_{\mathrm{group}}=8192$ tokens, то есть в среднем 1024 source tokens на EP-rank. При $D=4096$, top-2 и BF16 полный packed view содержит 128 MiB. Равномерный source rank упаковывает 16 MiB assignments; примерно $7/8$ назначения удалённые, поэтому сеть несёт около 14 MiB/rank на dispatch без metadata/padding. После all-to-all rank получает в среднем 2048 expert-token rows для восьми локальных experts, TP-group выполняет sharded GroupedGEMM, а combine all-to-all платит сопоставимые 14 MiB/rank и восстанавливает порядок токенов.

Backward повторяет коммуникационный граф в обратном направлении: gradient outputs упаковываются по сохранённому inverse map, all-to-all доставляет их владельцам experts, TP collectives формируют input- и weight-gradients, затем обратный all-to-all возвращает input-gradients source ranks. Dense gradients AllReduce выполняется по 16-rank dense-DP group, expert gradients — только по EDP=2; смешать эти группы означало бы усреднить разные expert shards.

Цена конфигурации видна из ledger: EP уменьшает expert weight/state в восемь раз, TP — ещё в восемь, но дважды за слой пересылает token activations и платит TP collectives внутри experts. EDP=2 удваивает expert state во всём world, зато даёт две независимые data replicas. При skew $\rho=2$ самый загруженный rank может получить около 4096 rows вместо среднего 2048, поэтому communication buffers и GroupedGEMM time должны иметь запас по измеренному percentile, а не только по среднему.

Проверка начинается с координат: вывести membership всех групп и доказать, что каждый world rank входит ровно в ожидаемые TP/PP/EP/EDP группы. Затем сравнить с single-rank MoE token IDs до/после inverse map, outputs, input-gradients и gradients каждого expert. Профиль должен дать около 14 MiB/rank off-rank dispatch и combine при равномерном routing, а counters — подтвердить средние 256 assignments/expert и измеренный skew. Отдельный тест переставляет `ep` координаты: после корректной загрузки shards результат не меняется.

Порядок физического размещения следует частоте обменов: TP — внутри NVLink node, EP — внутри полной-bandwidth rail group, PP — между соседними topology domains, EDP — по оставшейся оси.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/distributed/parallelism-decision-tree.svg]]

*Источник: Harvard Edge ML Systems Book, [Distributed Training, figure `fig-parallelism-decision-tree`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/distributed_training/distributed_training.qmd), CC BY-NC-SA 4.0.*

Проверка MoE системы сравнивает token-to-expert assignments, counts per expert, reconstructed output и weight-gradients с single-rank implementation. В профиле нужны all-to-all bytes, skew $\rho$, доля padding/dropped tokens и GroupedGEMM utilization.

## Источники

- EDLS, pinned commit `e632aa89…`, [`week06_dl_arithmetic/lecture.pdf`, slides/PDF pp. 117–120 MoE/GroupedGEMM, pp. 127–136 TP versus EP and communication arithmetic, pp. 137–144 PP/1F1B/ZeroBubble/DualPipeV](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week06_dl_arithmetic/lecture.pdf); earlier architectural context: [`week04_large_models/lecture.pdf`, PDF pp. 72–76 Expert Parallelism/Switch Transformer](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week04_large_models/lecture.pdf).
- Harvard Edge ML Systems Book, commit `45ecc8d…`, [Distributed Training, `sec-distributed-training-systems-systems-expert-parallelism-b896` and `sec-distributed-training-systems-systems-hybrid-parallelism-5674`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/distributed_training/distributed_training.qmd).
- NVIDIA Megatron Core, [Parallelism strategies, “Expert Parallelism (EP)” compatibility note](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html).

← [[44e ZeRO, FSDP2, DeviceMesh и DTensor]] · Далее: [[44g Network, storage и distributed checkpoints]]
