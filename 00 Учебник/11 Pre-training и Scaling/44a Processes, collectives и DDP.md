---
title: Процессы, коллективные операции и DDP
type: textbook-chapter
status: canonical
last_updated: 2026-10-27
last_verified: 2026-10-27
source_language: mixed
source_unit_id:
  - lecture-07-figure-step-5-rendering-1
  - lecture-07-section-19-part-1-building-blocks-of-distributed-communication-computation
  - lecture-07-collective-communication-mechanisms
  - lecture-07-collective-operation-code-trace
  - lecture-07-section-130-nvidia-collective-communication-library-nccl
  - lecture-07-distributed-benchmark-experiment
  - lecture-07-parallelism-strategy-comparison
  - lecture-08-networking-collectives
  - assignment-02-distributed-benchmark-controls
  - assignment-02-distributed-rank-topology
  - assignment-02-task-distributed-communication-single-node
  - assignment-02-task-naive-ddp
  - assignment-02-task-naive-ddp-benchmarking
  - assignment-02-task-minimal-ddp-flat-benchmarking
  - assignment-02-task-ddp-overlap-individual-parameters
  - assignment-02-task-ddp-overlap-individual-parameters-benchmarking
  - assignment-02-task-alternate-ring-all-reduce
  - assignment-02-task-data-parallel-calcs
primary_sources:
  - https://github.com/pytorch/tutorials/blob/c8ccbbf7fcb8cb71adda1b957bd6bab50e29c2e6/intermediate_source/dist_tuto.rst
  - https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/distributed/distributed_c10d.py
  - https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_07.py
  - https://github.com/stanford-cs336/assignment2-systems/tree/ca8bc81a59b70516f7ebb2da4808daade877c736
  - https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html
  - https://docs.pytorch.org/docs/2.14/distributed.html
  - https://arxiv.org/abs/2006.15704v1
  - https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/notes/cuda.md
  - https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/distributed.md
  - https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/notes/ddp.md
  - https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/_dynamo/backends/distributed.py
  - https://huggingface.co/spaces/nanotron/ultrascale-playbook
  - https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/performance_engineering/performance_engineering.qmd
---

<a id="cs336-distributed-ddp"></a>
<a id="44a-processes-collectives-и-ddp"></a>

# 44a. Процессы, коллективные операции и DDP

Пусть один шаг обучения использует 256 последовательностей, а в памяти GPU
помещается только 32. Можно последовательно обработать восемь частей батча,
накапливая градиенты. Если есть восемь GPU, те же части можно обработать
одновременно: каждая GPU хранит копию модели и вычисляет градиент на своих
данных. Затем нужно объединить восемь градиентов и выполнить одно согласованное
обновление параметров. Это и есть параллелизм по данным — *data parallelism*.

Размер локального батча уменьшается,
но параметры модели, их градиенты и состояния обычного оптимизатора по-прежнему
хранятся целиком на каждой GPU. DDP помогает быстрее обрабатывать данные;
сам по себе он не позволяет разместить модель, которая не помещается на одном
устройстве. Распределение самих параметров и состояний между устройствами —
отдельная задача. Способы такого разделения сопоставлены в
[[44 Distributed training и mixed precision|предыдущей обзорной главе]].

## Процесс, устройство и группа процессов

В обычной конфигурации PyTorch DDP на каждой GPU работает отдельный процесс.
У него собственная память Python, экземпляр модели, оптимизатор и загрузчик
данных. Процессы исполняют одну программу, но получают разные части данных.
Чтобы отправить сообщение нужному участнику, каждому процессу назначают
номер — *rank*. Это номер процесса, а не универсальный идентификатор GPU.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/a2-rank-topology.png]]

*Нумерация восьми процессов на двух машинах. На второй машине локальные номера
снова начинаются с нуля, а глобальные продолжаются с четырёх. Источник:
Percy Liang, Tatsu Hashimoto и команда Stanford CS336, Assignment 2, Figure 3,
PDF p. 32; исходная схема — Lightning Fabric.
[Задание в зафиксированной версии](https://github.com/stanford-cs336/assignment2-systems/blob/ca8bc81a59b70516f7ebb2da4808daade877c736/cs336_assignment2_systems.pdf#page=32).*

`global_rank` обозначает номер во всём запуске, `local_rank` — номер процесса
на его машине, а `node_rank` — номер машины. При показанной на рисунке
последовательной нумерации процесс 6 находится на узле 1 и использует локальное
устройство 2. Поэтому передать ему `cuda:6` было бы ошибкой: на этой машине
доступны только устройства с локальными номерами от 0 до 3. При иной настройке
видимости GPU соответствие локального номера и устройства задают явно.

Коллективная операция выполняется не обязательно всеми процессами запуска,
а участниками выбранной группы — `ProcessGroup`. После `init_process_group`
существует группа по умолчанию, включающая весь запуск. Например,
`dist.get_world_size()` вернёт 8, а `dist.get_world_size(group)` для группы
`[0, 2, 4, 6]` — 4. В последней группе глобальный процесс 6 имеет внутригрупповой
номер 3. Группы позволяют, например, синхронизировать градиенты между копиями
модели отдельно от обмена внутри каждой копии, разделённой по тензорам.

При обычном вызове `dist.new_group([0, 2, 4, 6])` создание группы выполняют
все процессы исходного запуска, включая тех, кто в новую группу не войдёт.
Затем в её коллективных операциях участвуют только её члены. Если групп
несколько, процессы создают их в одном порядке. Это правило относится
к настройкам по умолчанию, без `use_local_synchronization=True`, у которого
свои требования. Нельзя переносить условие «только выбранные rank»
с обмена на создание группы без проверки API.

Для запуска одной копии процесса на каждой из четырёх GPU одной машины
можно использовать `torchrun --standalone --nproc-per-node=4 train.py`.
Начало `train.py` выглядит так:

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

# build_model() — конструктор модели из обычного обучения на одной GPU.
model = build_model().to(device)
ddp = DDP(model, device_ids=[local_rank])
```

`torchrun` запускает процессы и передаёт им параметры соединения через
переменные окружения. `init_process_group` устанавливает связь, а NCCL
выполняет обмен тензорами между GPU. В конце программы каждый процесс вызывает
`dist.destroy_process_group()`. Для проверки логики на CPU используют Gloo
и создают `DDP(model)` без `device_ids`. Эти варианты не заменяют друг друга
при измерении скорости: проверка на CPU ничего не говорит о пропускной
способности GPU-сети.
[Документация PyTorch: torchrun](https://docs.pytorch.org/docs/2.14/elastic/run.html),
[группы процессов](https://docs.pytorch.org/docs/2.14/distributed.html#groups).

## От сообщений к коллективным операциям

Пара `send`/`recv` задаёт один обмен: какой процесс отправляет тензор и какой
его получает. При обучении постоянно повторяются более общие задачи:
скопировать параметры всем участникам, сложить градиенты или собрать части
тензора. Для них существуют коллективные операции. Программа задаёт результат,
который должен получить каждый участник, а библиотека выбирает порядок
пересылок с учётом размера сообщения и соединений между устройствами.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/distributed/collective-primitives-overview.svg]]

*Сравнение входов и выходов коллективных операций. Редукция складывает
значения, а gather объединяет части тензора.
Источник: Vijay Janapa Reddi и авторы Harvard Edge ML Systems Book,
[Collective Communication, `fig-collective-primitives-overview`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/collective_communication/collective_communication.qmd),
CC BY-NC-SA 4.0. В части ReduceScatter буквы обозначают одинаковые позиции
в разных входных тензорах, а не обязательно одинаковые значения: `sum(A)`
суммирует все вклады в часть A. Центральные прямоугольники обозначают
операции, не отдельный центральный сервер.*

Рассмотрим четыре процесса из [лекции 7 CS336](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_07.py).
До обмена у них находятся тензоры:

| Процесс | Локальный тензор |
|---|---|
| 0 | `[0, 1, 2, 3]` |
| 1 | `[1, 2, 3, 4]` |
| 2 | `[2, 3, 4, 5]` |
| 3 | `[3, 4, 5, 6]` |

All-reduce складывает соответствующие элементы и возвращает полный результат
`[6,10,14,18]` каждому процессу. Форма тензора не меняется. Операция `SUM` ничего не знает
о градиентах, размере батча или усреднении: если нужно среднее, сумму надо
разделить на число участников. В готовом DDP это деление уже предусмотрено;
добавлять его вручную второй раз не следует.

```python
# Один и тот же фрагмент исполняют четыре процесса созданной группы.
values = torch.arange(4, dtype=torch.float32, device=device) + rank
dist.all_reduce(values, op=dist.ReduceOp.SUM)
# На каждом процессе values == tensor([6., 10., 14., 18.]).
```

`reduce(SUM)` выполняет то же сложение, но итог нужен только выбранному
получателю. `broadcast` решает обратную по направлению задачу: копирует один
готовый тензор от выбранного процесса остальным. Так можно распространить
начальные параметры модели.

`reduce_scatter(SUM)` складывает тензоры, но распределяет результат по частям:
в примере процессам достанутся соответственно `[6]`, `[10]`, `[14]`, `[18]`.
Если затем вызвать `all_gather`, каждый получит все четыре части и соберёт
`[6, 10, 14, 18]`. Поэтому all-reduce можно реализовать как reduce-scatter,
за которым следует all-gather. Когда полный градиент на каждой GPU не нужен,
можно остановиться после первой половины — эта возможность используется
при распределённом хранении градиентов и состояний оптимизатора.

Наконец, `all_to_all` позволяет каждому процессу отправить разные части
разным получателям. В MoE это нужно для распределения токенов по устройствам
с соответствующими экспертами. В отличие от all-reduce, содержимое тензоров
здесь не складывается.

<!-- BEGIN REVIEW: collective destinations 2026-10-27 -->

<a id="collective-destinations-gather-scatter-all-to-all"></a>

### Who receives the result: scatter, gather and all-gather

Suppose rank 0 holds four tensors, $t_0,t_1,t_2,t_3$, and wants rank $r$
to receive $t_r$. A scatter distributes those four tensors. Broadcasting
$t_0$ would instead copy the same tensor to every rank. The original PyTorch
diagram labels both the tensor identities and their destinations:

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/collective-destinations-2026-10-27/scatter.png|PyTorch scatter: rank 0 distributes t0, t1, t2 and t3 to the corresponding four ranks]]

*Séb Arnold and PyTorch Tutorials contributors,
[Writing Distributed Applications with PyTorch, “Scatter”](https://docs.pytorch.org/tutorials/intermediate/dist_tuto.html#collective-communication),
[complete original PNG](https://github.com/pytorch/tutorials/blob/c8ccbbf7fcb8cb71adda1b957bd6bab50e29c2e6/_static/img/distributed/scatter.png),
[BSD-3-Clause copyright and conditions](/licenses/pytorch-tutorials-c8ccbbf-BSD-3-Clause.txt).
Unmodified figure. Its upper and lower rows are the before/after states of
the same four processes: rank 0 also receives its own part. Arrows describe
the result, not a prescribed physical network topology.*

For gather, every rank starts with a tensor $t_r$, and rank 0 receives the
ordered list $[t_0,t_1,t_2,t_3]$. No addition takes place. With one-element
tensors containing 0, 1, 2 and 3, the receiver can concatenate them into
`[0,1,2,3]`. A sum-reduction of the same inputs would produce the one-element
tensor `[6]` instead.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/collective-destinations-2026-10-27/gather.png|PyTorch gather: four ranks send distinct tensors to an ordered list on rank 0]]

*Same authors and tutorial,
[complete original “Gather” PNG](https://github.com/pytorch/tutorials/blob/c8ccbbf7fcb8cb71adda1b957bd6bab50e29c2e6/_static/img/distributed/gather.png),
[BSD-3-Clause](/licenses/pytorch-tutorials-c8ccbbf-BSD-3-Clause.txt).
The output belongs to the selected destination. Input tensor identity is
retained; gather does not reduce its elements.*

The tutorial gives the API meanings in its original English:

<!-- BEGIN SOURCE: PyTorch tutorial destination rules -->

- `dist.gather(tensor, gather_list, dst, group)`: Copies `tensor`
  from all processes in `dst`.
- `dist.all_gather(tensor_list, tensor, group)`: Copies `tensor`
  from all processes to `tensor_list`, on all processes.

<!-- END SOURCE: PyTorch tutorial destination rules -->

*Original wording: Séb Arnold and PyTorch Tutorials contributors,
[“Collective Communication”, source revision `c8ccbbf`](https://github.com/pytorch/tutorials/blob/c8ccbbf7fcb8cb71adda1b957bd6bab50e29c2e6/intermediate_source/dist_tuto.rst#L248-L257),
[license](/licenses/pytorch-tutorials-c8ccbbf-BSD-3-Clause.txt).
Only literal-code markup and list indentation are converted to Markdown.*

In the first rule, `dst` names the destination process. In the second, every
process allocates an output list and receives the same ordered contents.
The `all_` prefix changes who receives the assembled result. It does not
change concatenation into summation.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/collective-destinations-2026-10-27/all_gather.png|PyTorch all-gather: every rank receives t0, t1, t2 and t3 in rank order]]

*Same authors and tutorial,
[complete original “All-Gather” PNG](https://github.com/pytorch/tutorials/blob/c8ccbbf7fcb8cb71adda1b957bd6bab50e29c2e6/_static/img/distributed/all_gather.png),
[BSD-3-Clause](/licenses/pytorch-tutorials-c8ccbbf-BSD-3-Clause.txt).
Each lower box has the complete ordered list. The crossed lines show logical
delivery to every recipient; a library can implement it using a ring, a
tree or another schedule rather than separate direct transfers.*

For four inputs of $n$ elements each, a gather destination stores $4n$
output elements while each other rank retains only its local input.
All-gather stores $4n$ output elements on every rank, or $16n$ across the
four output buffers. This counts application-visible output storage, not
temporary buffers or network traffic. Reduce and all-reduce keep an $n$-element
reduced result at the destination and at every rank respectively; a non-root
buffer after `reduce` is not a defined copy of the reduction result.

| Operation | Result in this four-rank example | Recipients |
| --- | --- | --- |
| Scatter | Rank $r$ receives $t_r$ | Each rank gets its own part |
| Broadcast | Every rank receives the same $t_0$ | All four ranks |
| Gather | Ordered list $[t_0,t_1,t_2,t_3]$ | Selected destination only |
| All-gather | The same complete ordered list | All four ranks |
| Reduce, SUM | $t_0+t_1+t_2+t_3$ | Selected destination only |
| All-reduce, SUM | The same elementwise sum | All four ranks |

### All-to-all sends a different part to each destination

All-gather starts with one local tensor per source and makes its contents
available everywhere. In all-to-all, source rank $r$ supplies a separate
piece $s_{r,c}$ for each destination $c$. If source 0 supplies `[0,1,2,3]`,
source 1 `[10,11,12,13]`, source 2 `[20,21,22,23]` and source 3
`[30,31,32,33]`, destination 2 receives `[2,12,22,32]`. It receives the
third piece from each source, in source-rank order, not all sixteen values
and not their sum. Interpreting source rows and destination columns gives
a transpose of this small *ownership table*, not necessarily a transpose
of every tensor's numerical dimensions.

The following fragment runs on four processes in an already initialized
default CPU/Gloo group. Each tensor has one-dimensional integer storage;
all pieces have equal size. It uses fresh buffers for each collective so
that the results do not overwrite inputs needed by a later example.

```python
import torch
import torch.distributed as dist

assert dist.get_world_size() == 4 and dist.get_backend() == "gloo"
rank = dist.get_rank()

parts = [torch.tensor([r]) for r in range(4)] if rank == 0 else None
local = torch.empty(1, dtype=torch.int64)
dist.scatter(local, scatter_list=parts, src=0)
assert local.tolist() == [rank]

root_list = [torch.empty_like(local) for _ in range(4)] if rank == 0 else None
dist.gather(local, gather_list=root_list, dst=0)
if rank == 0:
    assert [part.item() for part in root_list] == [0, 1, 2, 3]
else:
    assert root_list is None

everywhere = [torch.empty_like(local) for _ in range(4)]
dist.all_gather(everywhere, local)
assert [part.item() for part in everywhere] == [0, 1, 2, 3]

same = torch.tensor([42 if rank == 0 else -1])
dist.broadcast(same, src=0)
assert same.tolist() == [42]

reduced = local.clone()
dist.reduce(reduced, dst=0, op=dist.ReduceOp.SUM)
if rank == 0:
    assert reduced.tolist() == [6]  # no claim about non-root contents

sent = torch.arange(4, dtype=torch.int64) + 10 * rank
received = torch.empty_like(sent)
dist.all_to_all_single(received, sent)
assert received.tolist() == [rank, 10 + rank, 20 + rank, 30 + rank]
```

Every rank calls `scatter` and `gather`; only the root supplies their full
input/output list. The lists are ordered by group rank, not by arrival time.
The example uses the world group, so group ranks and global ranks coincide.
`gather` fills the provided list; its ordinary synchronous return value is
not the list. For scatter, every process preallocates its receiving tensor;
for gather, only the root preallocates the output list. The selected PyTorch
API requires same-sized tensors for `gather` and
`scatter`; the equal-sized example also avoids variable-split requirements
in `all_to_all_single`. Backend support must be checked for the chosen
operation and device; a Gloo result does not establish NCCL performance.
[PyTorch v2.14.1 API source](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/distributed/distributed_c10d.py).

### Uneven all-to-all: send counts become receive counts

Expert routing can produce different token counts for each destination.
A variable-size implementation needs compatible send/receive counts and
output storage. It can exchange those counts and specify matching split
sizes. The PyTorch API's uneven-split example makes the bookkeeping visible:

| Source rank | To rank 0 | To rank 1 | To rank 2 | To rank 3 | Total sent |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 2 | 2 | 1 | 1 | 6 |
| 1 | 3 | 2 | 2 | 2 | 9 |
| 2 | 2 | 1 | 1 | 1 | 5 |
| 3 | 2 | 2 | 2 | 1 | 7 |
| Total received | 9 | 7 | 6 | 5 | 27 |

Each source stores its outgoing pieces consecutively, in destination order.
Rank 0's input is `[0,1,2,3,4,5]`, so its one-element piece for rank 2 is
`[4]`: the first two destinations consume two elements each. Rank 1's
input is `[10,11,12,13,14,15,16,17,18]`; its piece for rank 2 starts after
three plus two elements and is `[15,16]`. Ranks 2 and 3 contribute `[23]`
and `[34,35]`. Destination 2 concatenates these pieces in source order,
giving `[4,15,16,23,34,35]`.

Consequently, source $r$'s `input_split_sizes[c]` must agree with destination
$c$'s `output_split_sizes[r]`. The receive-count lists are columns of the
table, not copies of each receiver's own send-count row. Rank 2 sends five
elements but receives six; allocating `empty_like(input)` would give the
wrong output size. The same four-process CPU/Gloo group can execute this
adaptation of the original numerical example:

```python
# Continues the initialized four-rank CPU/Gloo example above.
send_counts = [[2, 2, 1, 1], [3, 2, 2, 2],
               [2, 1, 1, 1], [2, 2, 2, 1]]
input_splits = send_counts[rank]
output_splits = [row[rank] for row in send_counts]
payload = torch.arange(sum(input_splits), dtype=torch.int64) + 10 * rank
output = torch.empty(sum(output_splits), dtype=torch.int64)
dist.all_to_all_single(
    output, payload,
    output_split_sizes=output_splits,
    input_split_sizes=input_splits,
)
expected_outputs = [
    [0, 1, 10, 11, 12, 20, 21, 30, 31],
    [2, 3, 13, 14, 22, 32, 33],
    [4, 15, 16, 23, 34, 35],
    [5, 17, 18, 24, 36],
]
assert output.tolist() == expected_outputs[rank]
```

*PyTorch contributors,
[`all_to_all_single`, complete “Another example with uneven split”](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/distributed/distributed_c10d.py#L6045-L6070),
[license](/licenses/pytorch-v2.14.1-LICENSE.txt).
The original input values, split counts and all four outputs are retained;
the fragment constructs the arrays and supplies named arguments. The shared
count table is given explicitly for teaching rather than obtained through a
preceding network exchange. This API is marked experimental in that version.*

Here the counts measure scalar elements. If an example instead sends token
rows of a two-dimensional `[tokens, hidden]` tensor, the API splits dimension
0 in rows; a flattened representation must account for the hidden width.
Within one GPU, selecting and grouping those tokens still uses indexed
[[02 Areas/ML & DL/00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению#indexed-gather-scatter-and-gradients|gather/scatter operations]].
Inter-process collectives move the grouped buffers; the local index tensor
and the destination process rank describe different address spaces.

<!-- END REVIEW: collective destinations 2026-10-27 -->

Коллективные операции требуют согласованного участия процессов: совпадающих
типов операций, совместимых размеров и типов тензоров, одинакового порядка
вызовов. Если один процесс вошёл в all-reduce градиента слоя A, а другой —
слоя B, библиотека не восстановит правильный смысл по именам переменных.
Операция может завершиться с ошибкой, зависнуть или сложить не те значения.
Дополнительные группы также создают в согласованном порядке.

Согласованность не требует, чтобы процессы вызывали операцию в одну
микросекунду. Более быстрый участник может ждать медленного. Но очередной
вызов должен обозначать одну и ту же операцию над одним и тем же логическим
объектом. Если формы и типы совпадают, ошибочный обмен градиента веса
с градиентом другого веса может выглядеть для библиотеки совершенно
допустимым. Поэтому проверка форм не заменяет проверку порядка.

Даже вывод общей метрики подчиняется этому правилу. Вызвать all-reduce
значения потерь должны все участники, а ограничить условием `rank == 0`
можно только последующую запись результата. Если поместить под это условие
сам all-reduce, остальные процессы не войдут в обмен. DDP автоматически
синхронизирует градиенты, но не превращает локальное скалярное значение
потерь в глобальную метрику.

<a id="ddp-как-replicated-state-machine"></a>

## Почему копии модели остаются одинаковыми

Каждый процесс начинает шаг с одними и теми же параметрами $\theta_t$,
но видит собственную часть батча. Значения функции потерь и локальные градиенты
поэтому обычно различаются. После усреднения градиентов все процессы получают
один вектор $\bar g_t$. Если они применяют одинаковый оптимизатор с одинаковым
состоянием, то вычисляют одинаковые новые параметры.

Далее подразумевается обычный DDP с фиксированной группой из $N$ процессов,
без пользовательского обработчика обмена. Такой обработчик может изменить
операцию объединения и правила деления. Примеры запуска используют группу
по умолчанию; при отдельной группе DDP и счётчик токенов должны использовать
одних и тех же участников и её размер, а не размер всего запуска.

Для SGD это видно непосредственно:

$$
\bar g_t = \frac{1}{N}\sum_{r=0}^{N-1}g_{r,t},
\qquad
\theta_{t+1}=\theta_t-\eta\bar g_t.
$$

Для Adam рассуждение такое же, но совпадать должны ещё накопленные первый
и второй моменты. Если один процесс восстановил их из контрольной точки,
а другой начал с нулей, одинаковые градиенты уже не гарантируют одинаковых
обновлений. DDP синхронизирует начальное состояние модели при обычной
инициализации, но не восстанавливает оптимизатор, расписание скорости обучения
или состояние масштабирования потерь. Их согласованная загрузка — обязанность
программы обучения. Параметры не пересылаются после каждого шага оптимизатора:
их согласованность поддерживается общим градиентом.
[PyTorch DDP: API и оговорки](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html).

У модели бывают и буферы, не обновляемые оптимизатором, например накопленные
статистики BatchNorm. Их распространение — отдельный механизм DDP, а не
усреднение градиентов. В PyTorch 2.14 синхронизацией буферов перед прямым
проходом управляет `forward_sync_buffers`; прежнее имя `broadcast_buffers`
устарело. Копирование буферов не превращает локальный BatchNorm в
нормализацию по глобальному пакету: для согласования статистик в самом
вычислении нужен соответствующий алгоритм, например SyncBatchNorm.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/distributed/data-parallel-flow.svg]]

*На каждой GPU остаётся полная модель; разделяются данные, а перед обновлением
объединяются градиенты. На рисунке четыре реплики по 32 примера,
глобальный пакет равен 128. Источник: Vijay Janapa Reddi и авторы Harvard Edge ML
Systems Book, [Distributed Training, `fig-data-parallel-flow`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/distributed_training/distributed_training.qmd),
CC BY-NC-SA 4.0. Доля времени обмена 10–40% в нижней подписи — иллюстративная
оценка источника, а не гарантированный диапазон для DDP; в этой главе
она не используется в расчётах.*

### Среднее по процессам и среднее по данным

Пусть в каждом из $N$ процессов ровно $b$ примеров. Локальная функция потерь
усредняет индивидуальные потери $\ell_i$:

$$
\begin{gathered}
L_r=\frac{1}{b}\sum_{i\in B_r}\ell_i,\\
\frac{1}{N}\sum_r\nabla L_r
=\frac{1}{Nb}\sum_r\sum_{i\in B_r}\nabla\ell_i.
\end{gathered}
$$

Справа — градиент средней потери по всему батчу из $Nb$ примеров. Именно
поэтому сочетание локального `reduction="mean"` и обычного DDP даёт нужный
результат при равных локальных батчах. Если вместо средних локально вычислять
суммы, DDP разделит их только на $N$, но не на $b$.

При неравных батчах среднее локальных средних неверно. Допустим, первый
процесс обработал два примера со средним градиентом 1, а второй — шесть
примеров со средним градиентом 3. DDP, усреднив два числа, получит
$(1+3)/2=2$. Средний градиент восьми примеров равен
$(2\cdot1+6\cdot3)/8=2{,}5$. Ошибка возникает потому, что двум маленьким
и шести большим слагаемым назначены разные веса.

Для языковой модели считать нужно не строки батча, а учитываемые целевые
токены. Padding и позиции, исключённые маской функции потерь, в знаменатель
не входят. Пусть на процессе $r$ таких токенов $m_r$, а во всей группе
$M=\sum_r m_r$. Чтобы получить среднюю потерю по всем токенам, перед обратным
проходом задают

$$
\widetilde L_r=\frac{N}{M}\sum_{i=1}^{m_r}\ell_{r,i}.
$$

Тогда DDP выполнит своё усреднение:

$$
\frac{1}{N}\sum_r\nabla\widetilde L_r
=\frac{1}{M}\sum_r\sum_i\nabla\ell_{r,i}.
$$

В примере выше локальные суммы градиентов равны 2 и 18. Умножение на $N/M=1/4$
даёт $0{,}5$ и $4{,}5$; после DDP получается $(0{,}5+4{,}5)/2=2{,}5$.
При взвешенных потерях знаменатель тоже должен соответствовать определению
целевой функции: число токенов не заменяет сумму их весов.

## Как разделить данные: DistributedSampler

DDP не делит входной батч и не выбирает обучающие примеры. Если каждый процесс
запустит обычный загрузчик по всему датасету в одном порядке, все GPU будут
вычислять одно и то же. `DistributedSampler` выдаёт каждому процессу свою
последовательность индексов. Например, без перемешивания датасет из восьми
примеров на двух процессах делится на `[0, 2, 4, 6]` и `[1, 3, 5, 7]`.

```python
from torch.utils.data import DataLoader, DistributedSampler

sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    seed=42,             # Одинаковый seed на всех процессах.
    drop_last=True,
)
loader = DataLoader(dataset, batch_size=32, sampler=sampler,
                    shuffle=False, drop_last=True)

for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # До создания итератора loader.
    for batch in loader:
        train_step(batch)
```

Общий seed нужен, чтобы процессы построили одну перестановку, а затем взяли
разные её части. `set_epoch` меняет эту перестановку от эпохи к эпохе. Без него
перемешивание будет повторяться в одном и том же порядке.

Два параметра `drop_last` решают разные задачи. У sampler он отбрасывает хвост
общего датасета, не делящийся на число процессов. Если `drop_last=False`,
sampler, наоборот, добавляет повторные индексы, чтобы всем досталось одинаковое
количество примеров. Так, при `shuffle=False` для десяти примеров и трёх
процессов он дополнит список индексами 0 и 1. При перемешивании повторяются
первые элементы общей перестановки, а не обязательно эти два индекса.
У `DataLoader` параметр удаляет неполный **локальный
батч** после разделения. Повторы допустимы не во всех задачах: при точном
оценивании метрик их надо исключить из подсчёта или использовать разбиение
без дополнения. Это поведение видно в [исходном коде DistributedSampler](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/utils/data/distributed.py#L107-L139).

Неравное число токенов в одном шаге и неравное число шагов в эпохе — разные
проблемы. Формула выше исправляет веса градиентов, но не помогает, если один
процесс закончил цикл, пока другой ещё ждёт его в all-reduce. Для такого
сценария у DDP есть `join()`. При дополнительных коллективных операциях,
которые `join()` не учитывает, например при собственном all-reduce счётчика
токенов, нельзя просто обернуть цикл в `join()` и считать проблему решённой.
Простой вариант — согласовать число шагов. Если в цикле остаются
дополнительные коллективные операции, документация требует включить
`throw_on_early_termination=True`, чтобы все участники завершили цикл
с исключением при первом исчерпавшем данные процессе.
[PyTorch: обработка неравных входов](https://docs.pytorch.org/tutorials/advanced/generic_join.html).

### Что делает join с закончившим работу процессом

Пусть процесс 0 получил один пакет, а процесс 1 — два. После первого
обновления процессу 0 больше нечего вычислять. Внутри `ddp.join()` он
продолжает участвовать в служебных обменах, имитирующих необходимые DDP
операции второго процесса. Он не получает новые обучающие примеры и не
исполняет пользовательский цикл оптимизатора. Поэтому второй процесс может
закончить свой пакет, не ожидая отсутствующего обратного прохода первого.

Параметр `divide_by_initial_world_size=True` сохраняет исходное число участников
в делителе градиента;
при `False` учитываются только процессы, у которых ещё есть данные.
Для двух процессов и единственного оставшегося локального градиента 3
получится соответственно 1,5 или 3. Первый вариант сохраняет прежний вес
вклада этой реплики, второй эквивалентен усреднению в уменьшившейся группе.
Ни один вариант сам по себе не решает задачу нормировки по переменному
числу целевых токенов.

После выхода всех участников `join()` распространяет модель последнего
закончившего процесса, чтобы веса снова совпадали. Но он не синхронизирует
состояния пользовательского оптимизатора, расписания и scaler. Если один
Adam выполнил один шаг, а другой два, его моменты и счётчик шагов различаются,
даже когда финальные веса одинаковы. Прежде чем продолжать обучение
снова всеми репликами, необходимо согласованно восстановить эти состояния
или использовать алгоритм, специально поддерживающий такой режим.

Наконец, `join()` не является способом пережить падение процесса:
закончивший участник должен оставаться живым и выполнять предусмотренные
обмены. Отказ процесса и неравномерный объём данных — разные ситуации.
Семантика делителя и финального распространения модели описана в
[PyTorch 2.14, DDP.join](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.join).

## Накопление градиентов и no_sync

Накопление градиентов разбивает один шаг оптимизатора на несколько прямых
и обратных проходов. Параметры между ними не меняются, а очередной `backward()`
прибавляет градиент к уже накопленному. При $K$ одинаковых микробатчах на
каждом из $N$ процессов эффективный размер батча равен $N K b$.

Синхронизировать градиенты после каждого микробатча необязательно: сложение
локальных градиентов можно выполнить до обмена. Контекст `ddp.no_sync()`
отключает синхронизацию **градиентов**; первый обычный прямой и обратный проход после него
синхронизирует накопленное. В контекст нужно включить **и прямой проход,
и backward**, а не только вызов `backward()`.

Это не общий запрет связи между процессами. Вызванный программой
`all_reduce` по-прежнему выполнится, как и собственные обмены слоёв.
Нельзя использовать `no_sync()` как разрешение одному процессу пропустить
часть общей последовательности коллективных операций.

```python
from contextlib import nullcontext

# window — непустое окно микробатчей одного шага оптимизатора.
# В этом примере во всех микробатчах и процессах одинаковое
# число учитываемых элементов, а criterion возвращает среднее.
optimizer.zero_grad(set_to_none=True)
for index, (inputs, targets) in enumerate(window):
    final_microbatch = index == len(window) - 1
    context = nullcontext() if final_microbatch else ddp.no_sync()
    with context:
        predictions = ddp(inputs)
        loss = criterion(predictions, targets) / len(window)
        loss.backward()
optimizer.step()
```

`zero_grad` вызывается один раз перед окном, `step` — один раз после него.
Деление на `len(window)` учитывает также последнее, укороченное окно эпохи.
Однако оно корректно только при указанном условии одинаковых размеров
микробатчей. Для последовательностей разной длины нужна нормировка по общему
числу учитываемых токенов **за всё окно и на всех процессах**:

```python
# Каждый labels содержит уже выровненные целевые токены.
# -100 отмечает позиции, которые не участвуют в функции потерь.
group = ddp.process_group
group_size = dist.get_world_size(group)
device = next(ddp.parameters()).device
local_count = sum(int((labels != -100).sum()) for _, labels in window)
global_count = torch.tensor(local_count, device=device, dtype=torch.int64)
dist.all_reduce(global_count, op=dist.ReduceOp.SUM, group=group)
total_tokens = global_count.item()
if total_tokens == 0:
    raise ValueError("В окне нет учитываемых целевых токенов")

optimizer.zero_grad(set_to_none=True)
for index, (inputs, labels) in enumerate(window):
    context = nullcontext() if index == len(window) - 1 else ddp.no_sync()
    with context:
        logits = ddp(inputs)
        loss_sum = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        (loss_sum * group_size / total_tokens).backward()
optimizer.step()
```

Второго деления на число микробатчей здесь нет: знаменатель `total_tokens`
уже включает всё окно. На всех процессах должно быть одинаковое число
прямых и обратных проходов в окне. Код показывает нормировку и синхронизацию;
перенос входов на нужное устройство и формирование сдвинутых целей языковой модели должны
быть выполнены загрузчиком или предыдущим этапом программы. Если включены
AMP и ограничение нормы градиента, снятие масштаба и ограничение выполняют
после накопления и синхронизации, перед обновлением параметров.
[PyTorch: no_sync](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.no_sync),
[накопление градиентов с AMP](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html#gradient-accumulation).

Ноль целей на одном процессе ещё не означает пустого глобального пакета.
Если все его метки равны `-100`, `cross_entropy(..., reduction="sum")`
даёт нулевую сумму с нулевым градиентом. Процесс всё равно выполняет прямой
и обратный проходы и участвует в редукции вкладов остальных реплик.
Локальный `continue` нарушил бы порядок обменов. Если же счётчик равен нулю
**во всей группе**, каждый участник получает один и тот же результат
all-reduce и показанный код согласованно прекращает шаг до `backward()`.

Нулевые цели не разрешают подавать некорректные входы: все используемые
индексы и формы должны оставаться допустимыми, а промежуточные вычисления —
конечными. Умножение уже возникшего NaN на ноль его не устраняет.

<a id="от-корректного-ddp-к-перекрытию-communication"></a>

## Как совместить обмен градиентами с обратным проходом

В последовательной реализации сначала вычисляются все градиенты, затем
каждый тензор передаётся в all-reduce. Если у модели тысячи небольших
тензоров, программа тысячи раз платит задержку запуска операции. В задании 2
CS336 предлагается сначала объединить градиенты в один непрерывный буфер
и передать его одним вызовом. Это уменьшает число запусков, но весь обмен
по-прежнему начинается только после окончания обратного прохода.

Пусть два параметра имеют формы `(2,)` и `(2,2)`. На первом процессе
их градиенты равны `[1,3]` и `[[5,7],[9,11]]`, на втором —
`[3,5]` и `[[7,9],[11,13]]`. Можно передать два тензора по отдельности,
а можно сначала получить непрерывные векторы
`[1,3,5,7,9,11]` и `[3,5,7,9,11,13]`.
После суммирования и деления на два общий вектор равен
`[2,4,6,8,10,12]`. Первые два элемента возвращаются в градиент
первого параметра, следующие четыре — в матрицу `[[6,8],[10,12]]`.
Объём чисел не изменился; вместо двух операций запущена одна.

Ниже показан этот этап ручного DP после `loss.backward()`
и перед `optimizer.step()`. Модель не обёрнута в штатный DDP:
повторять за ним редукцию не нужно. Во всех процессах одинаковый
непустой список параметров, все градиенты существуют, плотные,
одного типа и на одном устройстве. Локальные средние используют
одинаковое число учитываемых элементов.

```python
grads = [p.grad for p in model.parameters()]
flat = torch.cat([g.reshape(-1) for g in grads])
dist.all_reduce(flat, op=dist.ReduceOp.SUM)
flat.div_(dist.get_world_size())

offset = 0
for grad in grads:
    count = grad.numel()
    grad.copy_(flat[offset:offset + count].view_as(grad))
    offset += count
```

Исходные формы и порядок нужны для обратного размещения. Смешанные
типы, разреженные и отсутствующие градиенты требуют отдельной обработки;
тихо исключить разный набор параметров на разных процессах нельзя.
`torch.cat` создаёт дополнительный буфер, а копирование обратно тоже
занимает время. Более короткая последовательность вызовов не доказывает
ускорения без измерения. Такая последовательность соответствует
[CS336 Assignment 2, §5.3.1](https://github.com/stanford-cs336/assignment2-systems/blob/ca8bc81a59b70516f7ebb2da4808daade877c736/cs336_assignment2_systems.pdf#page=34).

Следующее улучшение использует порядок вычисления производных. У обычной
цепочки слоёв градиенты выходных слоёв становятся готовы раньше градиентов
входных. Передачу уже готовых градиентов можно начать, пока GPU продолжает
вычислять остальные. Для этого DDP объединяет несколько тензоров градиентов
в группу — *bucket*. Как только все её градиенты готовы и наступила очередь
этой группы, запускается асинхронный all-reduce.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ddp-review-2026-09/pytorch-ddp-fig4-gradient-reduction.png]]

*Две копии сети из двух линейных слоёв. Стрелки autograd идут от функции
потерь к параметрам; пунктир показывает копирование градиентов в буферы DDP;
красные линии — обмен между соответствующими буферами. Источник: Shen Li
и соавторы, [PyTorch Distributed: Experiences on Accelerating Data Parallel
Training](https://arxiv.org/pdf/2006.15704v1#page=6), Figure 4, 2020.
Рисунок вырезан из страницы без изменения содержимого;
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

На рисунке `bucket1` содержит градиенты второго, ближайшего к функции потерь
слоя: $g_{w_2}$ и $g_{b_2}$. Их можно отправить раньше, чем готовы
$g_{w_1}$ и $g_{b_1}$ в `bucket2`. Вызов `backward()` не требуется вручную
разбивать на части: DDP устанавливает обработчики autograd, которые узнают
о готовности градиентов. К моменту возврата из обычного `backward()` DDP
обеспечивает зависимости, необходимые для использования синхронизированных
градиентов оптимизатором. В собственной реализации из задания ту же
обязанность должен выполнить написанный студентом код.

При асинхронных вызовах `dist.all_reduce(..., async_op=True)` возвращается
объект `Work`. Его `wait()` обеспечивает необходимое ожидание результата
для последующих операций, но на CUDA это не следует путать с завершением
всей работы GPU с точки зрения CPU-таймера. Для измерения времени используют
CUDA events или явную синхронизацию устройства. Вызов `async_op=False` тоже
не делает обычный CPU-таймер автоматически корректным.
[CS336 Assignment 2, §5.3](https://github.com/stanford-cs336/assignment2-systems/blob/ca8bc81a59b70516f7ebb2da4808daade877c736/cs336_assignment2_systems.pdf#page=34).

Размер группы градиентов задаёт компромисс. Очень маленькие группы позволяют
рано начать обмен, но создают много коротких операций. Очень большие
эффективнее используют канал, но поздно становятся готовы. В крайнем случае
один буфер на всю модель полностью устраняет возможность такого перекрытия.
Поэтому `bucket_cap_mb` выбирают по измеренному времени шага и его временной
диаграмме, а не только по скорости отдельного all-reduce.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/distributed-overview-2026-10-23/hf-ddp-sequential.svg]]

*Nouamane Tazi и соавторы, Hugging Face,
[Ultra-Scale Playbook, исходный SVG](https://huggingface.co/spaces/nanotron/ultrascale-playbook/blob/c93890dd4b22356ee18ed9b863e6f2d0909729bc/assets/images/dp_overlap1.svg),
[Apache-2.0](/licenses/ultrascale-Apache-2.0.txt). Сначала завершён весь
backward, затем идёт общая редукция `ALL`. Числа 0–2 обозначают слои,
две строки — вычисления и передачу, а не разные процессы.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/distributed-overview-2026-10-23/hf-ddp-overlap.svg]]

*Те же авторы, [исходный SVG с перекрытием](https://huggingface.co/spaces/nanotron/ultrascale-playbook/blob/c93890dd4b22356ee18ed9b863e6f2d0909729bc/assets/images/dp_overlap2.svg),
[Apache-2.0](/licenses/ultrascale-Apache-2.0.txt). Передача частей
градиента слоя 2 начинается до окончания backward слоёв 1 и 0.
В конце остаётся хвост обмена. Разбиение на много буферов даёт
возможность перекрытия, но не гарантирует полного скрытия сети.*

Для конкретного расписания предположим, что четыре градиента готовы
в 2, 4, 6 и 8 ms, каждый имеет размер 1 MB. Один поток обменов
последовательно обрабатывает сообщения за $1\ \mathrm{ms}+S/W$,
где $W=1$ MB/ms. Вычисления от обменов не замедляются.
После всего backward четыре отдельные передачи займут ещё 8 ms:
конец приходится на 16 ms. Один объединённый буфер размером 4 MB
передаётся с 8 до 13 ms. Два буфера по 2 MB готовы в 4 и 8 ms:
первый передаётся с 4 до 7 ms, второй — с 8 до 11 ms.

В этом примере объединение сократило затраты на запуски с 4 до 1 ms,
но не скрыло передачу. Для двух буферов суммарная задержка запусков равна 2 ms;
обмен закончился раньше благодаря готовности первого до конца backward.
Это заданная учебная модель, не измерение NCCL. Реальный выбор меняют
порядок готовности, конкуренция за GPU и сеть, а также уже выполняемые
обмены.

### Порядок операций и неиспользованные параметры

Процессы могут вычислять градиенты с разной скоростью и даже в разном порядке.
Поэтому правило «сразу отправить любой готовый буфер» само по себе неверно.
Все процессы должны вызвать all-reduce для буферов в одной последовательности.
DDP соблюдает этот порядок, даже если более поздний буфер локально готов раньше.
Это видно и в [реализации `Reducer::mark_bucket_ready`](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/csrc/distributed/c10d/reducer.cpp#L1051-L1074).

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ddp-review-2026-09/pytorch-ddp-fig3-order-and-unused-gradients.png]]

*Слева показан неверный обмен между разными группами градиентов, справа —
группа, которая не станет полностью готовой из-за отсутствующего $g_3$.
Источник: Shen Li и соавторы, [PyTorch Distributed](https://arxiv.org/pdf/2006.15704v1#page=5),
Figure 3, 2020. Сохранены подписи и легенда оригинала;
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

В левой части первый процесс раньше закончил $g_1$ и $g_2$, а второй —
$g_3$ и $g_4$. Если оба отправят первый готовый буфер, соответствующие
вызовы all-reduce сложат разные параметры. Правая часть показывает другую
проблему: параметр может не участвовать в текущем вычислительном графе,
и его обработчик готовности вообще не сработает. Тогда ожидание всех
обработчиков никогда не закончится.

Если в модели есть условные ветви с неиспользованными параметрами, режим
`find_unused_parameters=True` позволяет DDP найти такие параметры по графу.
Проверка требует дополнительной работы, поэтому включать её без необходимости
не стоит. Состав и порядок регистрации параметров должны быть одинаковыми
во всех процессах; добавлять параметры после создания DDP нельзя.
Рисунки описывают механизм DDP из статьи 2020 года. Точные правила построения
и перестройки буферов менялись, поэтому параметры текущего API проверяют
по [документации PyTorch](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html),
а не принимают старую схему за полный список возможностей реализации.

<!-- BEGIN REVIEW: DDP overlap 2026-10-26 -->

<a id="ddp-overlap-readiness-streams-compilation"></a>

### Buckets, readiness on every rank, and the exposed tail

The earlier schedule used a single list of gradient-ready times. In a
distributed step, each rank has its own list. A locally ready bucket cannot
complete its reduction without the other ranks' contributions, and a later
bucket cannot bypass the collective order established for the group.
These constraints determine how much communication remains after backward.

<!-- BEGIN SOURCE: Ultra-Scale gradient bucketing -->

GPU operations are usually more efficient when performed on large tensors rather
than having many operations running on smaller tensors. This is also true for
communication operations. Thus, we can advantageously group gradients into
buckets and launch a single all-reduce for all the gradients within the same
bucket instead of performing independent all-reduce for each gradient. It will
generally look like the following:

<!-- END SOURCE: Ultra-Scale gradient bucketing -->

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ddp-overlap-review-2026-10/gradient-buckets.svg|Ultra-Scale Playbook: gradient buckets overlap with backward and leave a communication tail]]

*Nouamane Tazi and the Hugging Face Ultra-Scale Playbook contributors,
[“Second optimization: Bucketing gradients”](https://huggingface.co/spaces/nanotron/ultrascale-playbook),
[pinned original dp_overlap3.svg](https://huggingface.co/spaces/nanotron/ultrascale-playbook/blob/c93890dd4b22356ee18ed9b863e6f2d0909729bc/assets/images/dp_overlap3.svg),
[Apache-2.0](/licenses/ultrascale-Apache-2.0.txt).
Unmodified schematic; the numbered regions identify work, not ranks.*

Purple communication regions begin while orange backward regions remain.
The last purple region extends beyond backward; the next forward starts
after it. The drawing does not give measured durations or prescribed
bucket sizes.

Consider two ranks and four buckets, always reduced in index order 0,1,2,3.
Their local readiness times, in illustrative milliseconds, are:

| Bucket | Ready on ranks 0 / 1 | Collective interval |
|---|---|---|
| 0 | 2 / 3 | 3–5 |
| 1 | 4 / 9 | 9–11 |
| 2 | 6 / 7 | 11–13 |
| 3 | 8 / 10 | 13–15 |

This model has one ordered communication resource. A collective begins only
after every rank is ready for that bucket and the previous collective has
finished; each takes 2 ms. Computation and communication do not slow each
other down. With local readiness $t_{p,i}$ and duration $c_i$, the recurrence is

$$
\begin{gathered}
R_i=\max_p t_{p,i},\\
s_i=\max(f_{i-1},R_i),\\
f_i=s_i+c_i,\qquad f_{-1}=0.
\end{gathered}
$$

Bucket 0 waits for rank 1 until time 3 and finishes at 5. Bucket 1 then
waits until 9. Although bucket 2 is ready on both ranks by 7, it must follow
bucket 1, so it starts at 11. The final reduction finishes at 15, while
all local backward work is done by 10: the exposed tail is 5 ms.

The four collectives contain only 8 ms of communication, less than the
10 ms backward duration. That inequality has not hidden the tail. Comparing
total communication with total compute omits release times and dependencies.
Likewise, summing all NCCL kernel durations across devices does not recover
this critical path.

```python
def ordered_bucket_schedule(ready_by_rank, durations):
    count = len(durations)
    if not ready_by_rank or not count:
        raise ValueError("Need at least one rank and bucket")
    if any(len(row) != count for row in ready_by_rank):
        raise ValueError("Every rank must describe the same buckets")
    if any(t < 0 for row in ready_by_rank for t in row) or any(c < 0 for c in durations):
        raise ValueError("Times must be nonnegative")
    intervals = []
    finish = 0
    for index, duration in enumerate(durations):
        ready = max(row[index] for row in ready_by_rank)
        start = max(finish, ready)
        finish = start + duration
        intervals.append((start, finish))
    backward_done = max(max(row) for row in ready_by_rank)
    tail = max(0, finish - backward_done)
    return intervals, tail

intervals, tail = ordered_bucket_schedule(
    [[2, 4, 6, 8], [3, 9, 7, 10]], [2, 2, 2, 2]
)
assert intervals == [(3, 5), (9, 11), (11, 13), (13, 15)]
assert tail == 5
assert sum([2, 2, 2, 2]) < 10  # still not sufficient for full overlap
```

The function is a CPU scheduling model, not an NCCL simulator. Real collectives
may begin local or partial work before the last rank arrives, use several
channels, and overlap with other traffic. The deliberately serialized model
isolates readiness and fixed ordering; it does not predict a cluster's timing.
The ordering requirement is documented in the
[PyTorch DDP design note](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/notes/ddp.md#internal-design).

### Stream ordering and allocation lifetime

A gradient producer must finish before a collective reads its buffer, and
a consumer must wait for the reduced values. Keeping the allocation valid
is another requirement. Returning from a Python call does not establish all
of those dependencies.

<!-- BEGIN SOURCE: PyTorch side-stream allocation lifetime -->

A [CUDA stream](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams) is a linear sequence of execution that belongs to a specific
device.  You normally do not need to create one explicitly: by default, each
device uses its own "default" stream.

Operations inside each stream are serialized in the order they are created,
but operations from different streams can execute concurrently in any
relative order, unless explicit synchronization functions (such as
`torch.cuda.synchronize` or `torch.cuda.Stream.wait_stream`) are
used.  For example, the following code is incorrect:

```python
cuda = torch.device('cuda')
s = torch.cuda.Stream()  # Create a new stream.
A = torch.empty((100, 100), device=cuda).normal_(0.0, 1.0)
with torch.cuda.stream(s):
    # sum() may start execution before normal_() finishes!
    B = torch.sum(A)
```

When the "current stream" is the default stream, PyTorch automatically performs
necessary synchronization when data is moved around, as explained above.
However, when using non-default streams, it is the user's responsibility to
ensure proper synchronization.  The fixed version of this example is:

```python
cuda = torch.device('cuda')
s = torch.cuda.Stream()  # Create a new stream.
A = torch.empty((100, 100), device=cuda).normal_(0.0, 1.0)
s.wait_stream(torch.cuda.default_stream(cuda))  # NEW!
with torch.cuda.stream(s):
    B = torch.sum(A)
A.record_stream(s)  # NEW!
```

There are two new additions.  The `torch.cuda.Stream.wait_stream` call
ensures that the `normal_()` execution has finished before we start running
`sum(A)` on a side stream.  The `torch.Tensor.record_stream` (see for
more details) ensures that we do not deallocate A before `sum(A)` has
completed.  You can also manually wait on the stream at some later point in
time with `torch.cuda.default_stream(cuda).wait_stream(s)` (note that it
is pointless to wait immediately, since that will prevent the stream execution
from running in parallel with other work on the default stream.)  See the
documentation for `torch.Tensor.record_stream` on more details on when
to use one or another.

Note that this synchronization is necessary even when there is no
read dependency, e.g., as seen in this example:

```python
cuda = torch.device('cuda')
s = torch.cuda.Stream()  # Create a new stream.
A = torch.empty((100, 100), device=cuda)
s.wait_stream(torch.cuda.default_stream(cuda))  # STILL REQUIRED!
with torch.cuda.stream(s):
    A.normal_(0.0, 1.0)
    A.record_stream(s)
```

Despite the computation on `s` not reading the contents of `A` and no
other uses of `A`, it is still necessary to synchronize, because `A`
may correspond to memory reallocated by the CUDA caching allocator, with
pending operations from the old (deallocated) memory.

<!-- END SOURCE: PyTorch side-stream allocation lifetime -->

*PyTorch contributors, original English from
[CUDA semantics, “CUDA streams”, v2.14.1](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/notes/cuda.md#cuda-streams),
[full copyright and license notices](/licenses/pytorch-v2.14.1-LICENSE.txt).
Only documentation-role markup is changed to inline code. These CUDA snippets
assume `import torch`; they are not CPU examples or GPU runs performed here.*

`wait_stream` orders future work on its receiving stream after work already
submitted to the other stream. It does not make the CPU wait for the entire
device. `record_stream` informs the allocator about use of the tensor's
storage on another stream; it does not wait for the data producer. It can
prevent premature reuse of storage even after the Python tensor is released.
It does not itself fix a missing producer-to-consumer dependency.

The [record_stream API](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.record_stream.html)
also permits manually joining the non-creation stream back to the creation
stream before releasing the tensor. Placement affects both overlap and how
long storage must remain live. The operation ordering and the storage lifetime
have to be correct whichever method is chosen.

For an asynchronous NCCL collective, the returned `Work.wait()` establishes
the dependency on the currently active CUDA stream in the documented path;
it is not a whole-device CPU timing barrier. Here is a two-rank example
using an already initialized default NCCL group, with each process's current
CUDA device set as in the setup above. It adapts the official
[distributed CUDA-semantics example](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/distributed.md#synchronous-and-asynchronous-collective-operations),
and explicitly orders the final CPU read as well.

```python
assert dist.get_world_size() == 2
rank = dist.get_rank()
device = torch.device("cuda", torch.cuda.current_device())
caller = torch.cuda.current_stream(device)
values = torch.tensor([float(rank)], device=device)
work = dist.all_reduce(values, op=dist.ReduceOp.SUM, async_op=True)

# Independent work can be submitted without reading or modifying values.
independent = torch.arange(16, device=device).square()
consumer = torch.cuda.Stream(device=device)
with torch.cuda.stream(consumer):
    work.wait()
    result = values + 100
values.record_stream(consumer)

# The CPU copy is issued from caller, so it must wait for consumer's result.
caller.wait_stream(consumer)
result.record_stream(caller)
torch.testing.assert_close(result.cpu(), torch.tensor([101.0]))
```

The source collective sums 0 and 1; the consumer adds 100 only after the sum
is available. Waiting on the consumer stream leaves the independent caller
work eligible to overlap with communication. The later wait protects the
read of the result from the caller stream. This is CUDA/NCCL example code,
not an executed test on this CPU-only machine. Correct dependencies permit
overlap; they do not establish that the hardware actually ran work concurrently.

### Compiling backward can change when hooks become ready

The compilation boundary also changes readiness. A faster backward region
can still delay communication if it exposes all gradients to the autograd
engine only after the entire region finishes. The PyTorch design note
describes the Dynamo/AOTAutograd path and its DDPOptimizer intervention:

<!-- BEGIN SOURCE: PyTorch DDPOptimizer overlap -->

DDP's performance advantage comes from overlapping allreduce collectives with computations during backwards.
AotAutograd prevents this overlap when used with TorchDynamo for compiling a whole forward and whole backward graph,
because allreduce ops are launched by autograd hooks _after_ the whole optimized backwards computation finishes.

TorchDynamo's DDPOptimizer helps by breaking the forward graph at the logical boundaries of DDP's allreduce buckets
during backwards.  Note: the goal is to break the graph during backwards, and the simplest implementation is to
break the forward graphs and then call AotAutograd and compilation on each section.  This allows DDP's allreduce hooks
to fire in-between sections of backwards, and schedule communications to overlap with compute.

<!-- END SOURCE: PyTorch DDPOptimizer overlap -->

*PyTorch contributors, original English from
[DDP design note, “TorchDynamo DDPOptimizer”](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/notes/ddp.md#torchdynamo-ddpoptimizer).
The general design note explicitly contains historical implementation details;
the graph-splitting mechanism is also present in the selected
[v2.14.1 DDPOptimizer source](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/_dynamo/backends/distributed.py#L389-L452).
This describes that path, not every compiler, reducer or compiled-autograd configuration.*

Use the same scheduling function with two buckets, each taking 3 ms to reduce.
An eager backward produces them at times 4 and 8: reductions run at 4–7
and 8–11. Suppose compilation shortens backward from 8 to 6 ms, but both
hooks become ready only at 6. Reductions then run at 6–9 and 9–12.
The local computation got faster while this part of the distributed step
got slower, from 11 to 12 ms. If two compiled regions expose readiness at
3 and 6 instead, reductions run at 3–6 and 6–9.

```python
# Uses ordered_bucket_schedule from the readiness example above.
eager_two_buckets, _ = ordered_bucket_schedule([[4, 8], [4, 8]], [3, 3])
whole_backward, _ = ordered_bucket_schedule([[6, 6], [6, 6]], [3, 3])
split_backward, _ = ordered_bucket_schedule([[3, 6], [3, 6]], [3, 3])
assert eager_two_buckets == [(4, 7), (8, 11)]
assert whole_backward == [(6, 9), (9, 12)]
assert split_backward == [(3, 6), (6, 9)]
```

The times are illustrative inputs to the CPU model, not measurements of
Inductor or predictions of the splits chosen for a real model. They isolate
the effect of gradient availability while keeping the two reductions' costs
unchanged. A backward region being opaque to the autograd engine also does
not mean that it consists of one device kernel.

The design note recommends wrapping the model with DDP before compilation,
so the compiler can see its bucket configuration. Using this chapter's local
device numbering, the setup lines become:

```python
ddp = DDP(model, device_ids=[local_rank])
ddp = torch.compile(ddp)
```

These replace the setup's DDP construction; do not wrap an existing DDP
instance again. Inspect actual regions and buckets with the documented
`TORCH_LOGS=ddp_graphs` diagnostics before interpreting a speed difference.
The selected implementation counts parameter sizes in reverse graph order;
it can extend a bucket to obtain a usable graph output and bypasses splitting
when there is only one bucket. It is not a guarantee of one compiled region
per layer or perfectly aligned boundaries in every model. The
[[02 Areas/ML & DL/00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению#torch-compile-graphs-guards-recompilation|Dynamo explanation in S8]]
introduces FX graphs, guards and ordinary graph breaks; here the deliberate
partition serves the distributed schedule.

### Resource contention is outside the readiness-only model

The scheduling function treats the supplied readiness times and collective
durations as unchanged during overlap. A real GPU may share SM execution
capacity and memory bandwidth between compute kernels and kernel-based
communication. Those durations can change when both run concurrently.
Some communication paths instead offload data movement: the documented
[copy-engine collective path](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/distributed.md#copy-engine-collectives)
uses symmetric-memory tensors and a zero-CTA policy. It has its own setup
requirements and is not the default meaning of any `async_op=True` call.

For an already independent pair of operations, `max(compute, communication)`
is the ideal concurrent duration only under the no-interference assumptions.
For gradient synchronization there are also the release times demonstrated
above. Neither a byte-count lower bound from link bandwidth nor a promise
of a fixed number of NCCL SMs determines a measured step's exposed tail.

Compare the compute region alone, the collective alone, and their concurrent
execution, then inspect the complete step's dependency path. Include
warmup, rank placement, bucket configuration and the timing boundary in the
record. A CPU Gloo test can establish numerical correctness of a reduction;
it cannot show CUDA overlap, NCCL resource use or GPU speedup.

<!-- END REVIEW: DDP overlap 2026-10-26 -->

## Сколько стоит all-reduce

Даже правильно организованный обмен занимает время. Для оценки полезно
разделить задержку запуска и собственно передачу данных:

$$
T\approx\alpha n_{\mathrm{rounds}}+\frac{V}{B}.
$$

Здесь $\alpha$ — задержка одного раунда, $V$ — объём данных, передаваемый
одним участником, а $B$ — эффективная пропускная способность соответствующего
канала. Это приближённая модель: она не учитывает конкуренцию потоков,
вычисления при редукции и все особенности топологии.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/distributed/ring-allreduce.svg]]

*Кольцевой all-reduce состоит из накопления суммы частей и распространения
готовых частей. Источник: Vijay Janapa Reddi и авторы Harvard Edge ML Systems
Book, [Collective Communication, `fig-ring-allreduce`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/collective_communication/collective_communication.qmd),
CC BY-NC-SA 4.0.*

В кольцевом алгоритме тензор размером $S$ байт делится на $N$ частей.
В первой фазе, reduce-scatter, каждый процесс $N-1$ раз отправляет соседу
часть размером $S/N$ и добавляет к принимаемой части свой вклад.
После этой фазы у каждого есть одна полностью просуммированная часть.
Во второй фазе, all-gather, за следующие $N-1$ пересылок каждый получает
остальные части. Объём **отправленных** данных на один процесс равен

$$
V_{\mathrm{ring}}=2\frac{N-1}{N}S,
$$

$$
\begin{aligned}
T_{\mathrm{ring}}\approx{}&2(N-1)\alpha\\
&+2\frac{N-1}{N}\frac{S}{B}.
\end{aligned}
$$

Полученных данных столько же. В этой формуле не нужно повторно складывать
отправленный и принятый объёмы: предполагается одновременная работа обоих
направлений. Для восьми процессов и тензора 1 GiB каждый отправляет 1,75 GiB.
При $B=25$ GB/s передача занимает около 75,2 мс, а при
$\alpha=5\,\mu\mathrm{s}$ задержки раундов добавляют 70 мкс.
Для тензора 4 KiB, напротив, передача занимает лишь около 0,29 мкс,
и почти всё расчётное время приходится на задержки запуска.

### Один фрагмент на четырёх процессах

Вернёмся к четырём тензорам `[0,1,2,3]`, `[1,2,3,4]`,
`[2,3,4,5]` и `[3,4,5,6]`. Разобьём каждый на четыре части
по одному числу. Направим кольцо $0\to1\to2\to3\to0$.
Для первого фрагмента можно выбрать путь, который заканчивается
на процессе 0:

| Раунд reduce-scatter | Передача первого фрагмента |
|---|---|
| 1 | Процесс 1 отправляет 1 процессу 2; тот добавляет свои 2 и получает 3. |
| 2 | Процесс 2 отправляет частичную сумму 3 процессу 3; тот добавляет 3 и получает 6. |
| 3 | Процесс 3 отправляет 6 процессу 0; тот добавляет 0, итог равен 6. |

В те же раунды по другим участкам кольца движутся остальные фрагменты.
После третьего раунда процессы 0, 1, 2 и 3 владеют соответственно
готовыми частями 6, 10, 14 и 18. Во время all-gather их уже
не суммируют: передают и записывают на правильные позиции.
Процесс 0 за следующие три раунда получает 18, затем 14, затем 10
и собирает `[6,10,14,18]`. Каждый участник отправил шесть сообщений
размером $S/4$, то есть $1{,}5S$ байтов.

В [задаче `alternate_ring_all_reduce`, CS336 Assignment 2, §8.1](https://github.com/stanford-cs336/assignment2-systems/blob/ca8bc81a59b70516f7ebb2da4808daade877c736/cs336_assignment2_systems.pdf#page=41)
предложен другой алгоритм: каждый участник передаёт по кольцу целые
исходные тензоры и добавляет каждый полученный тензор к своей сумме.
Для четырёх процессов хватит трёх раундов, но каждый передаст уже
$3S$ байтов. В общем случае время без задержек равно $(N-1)S/B$;
для кольца с разбиением — $2(N-1)S/(NB)$. Отношение равно $N/2$.
При $N=4$ целые тензоры требуют вдвое большего исходящего объёма,
при $N=8$ — вчетверо. Число раундов меньше, но для больших сообщений
это не компенсирует объём. При коротких сообщениях сравнивают также
задержки: $(N-1)\alpha$ против $2(N-1)\alpha$.

Кольцо не является единственным способом вычислить all-reduce. Например,
рекурсивное удвоение при $N$, равном степени двойки, объединяет пары процессов
за $\log_2 N$ раундов. Простая версия пересылает полный тензор в каждом раунде:
задержек меньше, но объём передачи больше. Поэтому для небольших сообщений
выгоден один алгоритм, для крупных — другой. NCCL выбирает алгоритм
и протокол с учётом оборудования и размера сообщения.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/l08-p6.png]]

*Разные пути обмена между GPU: прямые соединения и NVSwitch внутри машины,
PCIe и сетевые адаптеры HCA для других маршрутов. Источник: Tatsu Hashimoto,
[Stanford CS336, Lecture 8](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_08.pdf#page=6),
PDF p. 6. Поколения интерфейсов и числа на рисунке относятся к приведённому
примеру, а не к любому современному кластеру.*

Внутри машины GPU могут обмениваться через NVLink/NVSwitch; между машинами
путь обычно включает сетевые адаптеры. Поэтому одинаковый all-reduce на
восьми GPU одного узла и на восьми разных узлах может иметь совершенно разную
стоимость. Иерархические алгоритмы учитывают эту разницу и организуют обмен
внутри узлов отдельно от межузлового. RDMA — способ доступа к удалённой памяти,
а не альтернатива Ethernet: RoCE реализует RDMA поверх Ethernet.

### Где возникает предел параллелизма по данным

В [CS336 Assignment 2, §8.2](https://github.com/stanford-cs336/assignment2-systems/blob/ca8bc81a59b70516f7ebb2da4808daade877c736/cs336_assignment2_systems.pdf#page=41)
стоимость обмена сопоставляется с backward одного FFN. Возьмём
вход из $b$ строк ширины $d$ и внутреннюю ширину $h$:

$$
\begin{gathered}
U_1=XW_1,\qquad U_2=XW_2,\\
H=\operatorname{SiLU}(U_1)\odot U_2,\qquad Y=HW_3.
\end{gathered}
$$

Матрицы $W_1,W_2$ имеют форму $d\times h$, $W_3$ — $h\times d$.
У каждого из $N$ процессов находится $b/N$ строк, а все три
матрицы остаются полными. Обозначим скорость устройства через $f$
FLOP/s, исходящую полосу через $w$ байт/s. В отличие от буквы $B$
в оценке кольца, строчная $b$ здесь означает число строк, не полосу.

Backward каждого линейного преобразования вычисляет два произведения:
градиент по входу и градиент по весам. В трёх матрицах получается
шесть произведений по $2bdh/N$ FLOPs. Производные SiLU и умножения
покоординатны; в этой оценке их стоимость опускается:

$$
\begin{gathered}
F_{\mathrm{backward}}=\frac{12bdh}{N},\\
t_{\mathrm{compute}}=\frac{12bdh}{Nf}.
\end{gathered}
$$

При двухбайтных градиентах их полный размер $S=6dh$ байтов.
Кольцевой all-reduce передаёт $2(N-1)S/N$ байтов на участника,
поэтому без задержек

$$
t_{\mathrm{comm}}=\frac{12dh(N-1)}{Nw}.
$$

Потребуем, чтобы сетевой работы было не больше вычислительной.
После сокращения $12dh/N$ получаем

$$
\frac{N-1}{w}\le\frac{b}{f},\qquad
N\le1+\frac{bw}{f}.
$$

При увеличении числа процессов вычисления делятся между ними,
а размер полного градиента не уменьшается. Поэтому фиксированный
пакет нельзя ускорять неограниченно. Ширины матриц сократились
в этом отношении потому, что в выбранной модели и FLOPs, и байты
градиента пропорциональны $dh$; в реальной модели есть и другие
операции, задержки и ограничения размеров матричных умножений.

Пусть $b=10752$, $d=h=7168$, $f=1{,}792\cdot10^{14}$ FLOP/s,
$w=10^{11}$ байт/s. Тогда $bw/f=6$, а предел оценки — $N\le7$.
Для семи процессов и вычисления, и обмен занимают примерно
5,285 ms. Для восьми вычисления занимают 4,624 ms, обмен —
5,395 ms: уменьшение локального пакета не ускорило передачу
полного градиента. Числа получены из заданных скоростей,
а не измерены на конкретном ускорителе.

Условие сравнивает объёмы работы и не доказывает, что весь обмен
удастся скрыть. Последний градиент становится готов только в конце
backward; начатая после этого передача оставляет хвост. При FP32
для редукции вместо FP16/BF16 время передачи в этой модели
удваивается, и предел становится $N\le1+bw/(2f)$.

### Полезность масштабирования

Пусть обратный проход без обмена занимает 180 мс, а обмен — 70 мс.
Если 55 мс передачи проходят одновременно с вычислением, общая длительность
этой части шага равна $180+70-55=195$ мс. Теперь добавим GPU и сократим
вычисление до 100 мс. Если скрыть удаётся только 30 мс передачи,
получится $100+70-30=140$ мс. Вычисления ускорились в 1,8 раза, а этот участок
шага — только в $195/140\approx1{,}39$ раза. Это условный пример временного
расписания, не результат измерения на конкретном кластере.

Сумма длительностей всех ядер NCCL на всех GPU не равна замедлению шага:
одни ядра работают параллельно, другие перекрываются с вычислением.
Нужна временная диаграмма конкретного шага — какая передача ещё продолжается,
когда градиенты уже вычислены и оптимизатор мог бы начать обновление.

## Когда точный all-reduce слишком дорог

Уменьшить стоимость связи можно не только настройкой буферов. Но если вместо
точного объединения всех градиентов передавать приближение или обмениваться
лишь с частью процессов, меняется уже алгоритм обучения. Ускорение здесь
нельзя оценивать без проверки качества модели.

При *gossip*-обмене каждый участник усредняет значения только с соседями
в выбранном графе. После одного раунда глобального среднего обычно ещё нет:
информация о дальних участниках должна пройти через промежуточные вершины.
Чем хуже связность графа, тем медленнее достигается согласие. Поэтому gossip
не является просто более быстрым точным all-reduce.

Например, пусть четыре процесса образуют кольцо и хранят числа
$[0,0,0,4]$. На одном раунде каждый заменяет своё число средним
своего и двух соседних значений из предыдущего раунда. Получается
$[4/3,0,4/3,4/3]$, хотя глобальное среднее равно 1.
После второго раунда значения равны $[8/9,8/9,8/9,4/3]$:
процессы сблизились, но ещё не согласованы.

В таком симметричном обмене среднее всех чисел сохраняется.
Если повторять только усреднение, значения сходятся к нему.
Во время обучения между обменами добавляются разные локальные
градиенты, поэтому остаётся учитывать расхождение реплик.
Для направленных или несимметричных графов простое среднее соседей
не обязано сохранять глобальное среднее: нужны дополнительные
условия на веса или корректирующий алгоритм. Пример относится
к симметричному кольцу, не ко всем вариантам gossip из
[EDLS, Week 3, p39–51](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week03_data_parallel/lecture.pdf#page=39).

При сжатии градиентов часть информации теряется. Например, если каждый раз
округлять маленькие значения до нуля, их накопленный вклад может исчезнуть
полностью. Метод *error feedback* сохраняет неотправленный остаток и добавляет
его к градиенту следующего шага:

$$
\begin{gathered}
u_t=g_t+e_t,\qquad q_t=C(u_t),\\
e_{t+1}=u_t-q_t.
\end{gathered}
$$

Здесь $C$ — операция сжатия, $q_t$ — переданное приближение, $e_t$ — остаток.
Например, при постоянном градиенте $0{,}4$ и округлении до целого без остатка
передавался бы ноль на каждом шаге. С остатком первые три переданных значения
будут 0, 1 и 0, а остатки — $0{,}4$, $-0{,}2$ и $0{,}2$:
сумма переданного и текущий остаток сохраняют суммарный вклад $1{,}2$.
Это объясняет назначение остатка, но само по себе не доказывает сходимость
любого компрессора.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ddp-2026-10-23/edls-error-feedback.png]]

*Авторы курса Efficient DL Systems (ВШЭ и ШАД),
[Week 3, полный слайд 58](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week03_data_parallel/lecture.pdf#page=58),
[MIT](/licenses/edls-e632aa89-MIT.txt). Слева показан первый шаг
с нулевым начальным остатком; неотправленная часть поступает
на следующий шаг. На правой панели подпись компрессора `C(g)`
сокращённая: его входом уже служит $g+e$, то есть $u_t$
в обозначениях выше. Слайд сохранён без перерисовки и перевода.*

Остаток — дополнительное состояние обучения, обычно полноразмерное.
Поэтому сокращение передаваемых байтов не обязательно сокращает
память градиентов. Если остаток сбросить при восстановлении,
следующее обновление уже не продолжит прежнюю последовательность
сжатых обновлений. Для воспроизведения нужно сохранять и это состояние.

PowerSGD использует низкоранговое приближение матрицы градиента
$G\in\mathbb R^{m\times n}$ в виде $PQ^\top$. При ранге $r$ вместо $mn$
элементов передаются факторы суммарного размера $r(m+n)$. Для матрицы
$4096\times4096$ и $r=8$ это 65 536 вместо 16 777 216 чисел, то есть
в 256 раз меньше. Однако время шага не уменьшится в 256 раз: остаются
вычисление факторов, ортогонализация, два обмена и обработка тензоров,
которые невыгодно сжимать. Векторы и маленькие матрицы часто передают точно.
Сжатие градиента embedding-матрицы не запрещено её названием; решение
зависит от формы, представления градиента и реализации метода.
[Vogels et al., PowerSGD, 2019](https://arxiv.org/abs/1905.13727),
[реализация PyTorch PowerSGD](https://docs.pytorch.org/docs/2.14/ddp_comm_hooks.html#powersgd-communication-hook).

### Как согласовать низкоранговое приближение

Независимо разложить локальные градиенты и усреднить их факторы
недостаточно: произведение средних факторов не равно среднему
произведений. В [PowerSGD, §3, Algorithm 1](https://arxiv.org/html/1905.13727v3#S3)
процессы согласуют факторы последовательно. Пусть $M_i$ — локальная
матрица после добавления остатка, а $Q_0$ — одна и та же начальная
матрица ширины $r$ у всех участников.

Сначала каждый вычисляет $P_i=M_iQ_0$. All-reduce со средним даёт
общую $P=\overline M Q_0$, где $\overline M$ — средняя матрица.
Её столбцы ортонормируют, получая $\widehat P$. Теперь все процессы
используют именно эту общую основу для второго произведения
$Q_i=M_i^\top\widehat P$. После второго усреднения получают
$Q=\overline M^\top\widehat P$ и общий результат
$\widehat M=\widehat P Q^\top$.

Например, возьмём два процесса, ранг один и

$$
\begin{gathered}
M_0=\begin{bmatrix}3&0\\0&1\end{bmatrix},\qquad
M_1=\begin{bmatrix}1&0\\0&3\end{bmatrix},\\
Q_0=\begin{bmatrix}1\\0\end{bmatrix}.
\end{gathered}
$$

Первое умножение даёт столбцы $[3,0]^\top$ и $[1,0]^\top$.
Их среднее $[2,0]^\top$ после нормировки становится
$\widehat P=[1,0]^\top$. Второе умножение снова даёт
$[3,0]^\top$ и $[1,0]^\top$, поэтому $Q=[2,0]^\top$.
Итог:

$$
\widehat M=\begin{bmatrix}2&0\\0&0\end{bmatrix},\qquad
\overline M=\begin{bmatrix}2&0\\0&2\end{bmatrix}.
$$

Все участники получили одинаковое обновление, но потеряли вторую
ненулевую координату. Согласованность реплик не означает точности
сжатия. В этом маленьком примере $r(m+n)=mn=4$, поэтому экономии
объёма вообще нет; он показывает порядок операций. Для больших
матриц, узких факторов и достаточно хорошего приближения тот же
порядок уменьшает передаваемые данные.

Начальную основу можно переносить с предыдущего шага — это warm start.
Он улучшает приближение без полной SVD, но не гарантирует наилучшее
приближение ранга $r$ после одной итерации для произвольной матрицы.
Реализация также учитывает стоимость двух редукций, вырожденных
факторов и несжимаемых тензоров. В PyTorch метод подключается как
DDP communication hook; совместимость сохранённого состояния,
нормировки и момента включения сжатия проверяют по
[документации hook](https://docs.pytorch.org/docs/2.14/ddp_comm_hooks.html#powersgd-communication-hook).

## Если один процесс остановился или пошёл по другой ветви

Ошибка распределённой программы нередко видна не там, где возникла.
Например, процесс 1 получил неверный индекс токена и завершился во время
прямого прохода. Процесс 0 дойдёт до редукции градиентов и будет ждать
участника, которого уже нет. Его сообщение о тайм-ауте описывает следствие;
причину нужно искать в более раннем журнале процесса 1.

Поэтому журнал должен содержать rank, номер шага и микропакета, а также
сохранённое исходное исключение. У `init_process_group` задают конечный
тайм-аут, соответствующий ожидаемой работе. Он ограничивает ожидание,
но не исправляет ошибку и не гарантирует безопасное продолжение того же
шага. После нарушения работы группы обычный вариант — завершить все
её процессы и восстановить согласованное состояние из контрольной точки.
Не следует перехватывать исключение только на одной реплике и молча
продолжать её цикл.

Когда все процессы ещё исправны, решение пропустить шаг можно согласовать
до входа в DDP: например, свести флаги невалидного пакета одной коллективной
операцией, которую вызовет каждый участник. Аналогично в примере выше
все участники узнают о пустой глобальной маске и останавливают шаг.
Такой протокол должен быть предусмотрен заранее; после произвольного сбоя
нельзя полагаться на то, что остальные процессы всё ещё готовы выполнить
этот дополнительный обмен.

Для диагностики PyTorch предоставляет `TORCH_DISTRIBUTED_DEBUG=DETAIL`:
проверки помогают обнаружить несовпадающие формы и последовательности
операций, но не знают смысла двух одинаково устроенных тензоров. Ещё один
инструмент — `monitored_barrier` с конечным тайм-аутом. Он показывает,
какие участники не достигли контрольной точки, и поддерживается Gloo;
в NCCL-запуске для него нужна отдельно созданная Gloo-группа.
Обычный `barrier` тоже требует всех участников: добавлять его в обработчик
ошибки, чтобы «дождаться остальных», может лишь создать новое ожидание.
[PyTorch: диагностика распределённых запусков](https://docs.pytorch.org/docs/2.14/distributed.html#debugging-torch-distributed-applications).

При нормальном завершении каждый процесс освобождает свои группы через
`destroy_process_group`. Блок `finally` помогает выполнить локальную
очистку при исключении Python, но не является распределённым восстановлением
и не сработает при принудительном завершении процесса. Протокол перезапуска
и состав контрольной точки разбираются в
[[44h Fault tolerance и fleet orchestration|главе об отказоустойчивости]].

## Проверка

Корректность и скорость проверяют отдельно. Сначала берут модель без DDP
и её копии на двух процессах. Обычная модель получает полный батч,
а процессы — его части. Нужно сравнить не только итоговую функцию потерь,
но и каждый градиент, а затем параметры после одного шага. Для Adam полезно
сравнить и состояния оптимизатора после нескольких шагов: одного совпавшего
обновления недостаточно для проверки восстановления из контрольной точки.

У такой сверки должны быть контролируемые условия: одинаковые начальные
параметры, данные и определение функции потерь. Dropout, случайные
преобразования входов и слои, зависящие от состава локального батча,
могут дать отличия и без ошибки DDP. Для первого теста их исключают.
Сравнение проводят с численным допуском, поскольку другой порядок сложения
чисел с плавающей точкой может изменить младшие разряды.

Отдельно проверяют равные и неравные локальные батчи, переменное число токенов,
накопление градиентов и последнее неполное окно. Если выполняется Assignment 2
CS336, необходимо сохранить обе архитектуры из параметризованных официальных
тестов, а не ограничиться одним удобным MLP. Искусственная задержка одного
процесса не должна менять результат шага, но увеличит время ожидания остальных.

При измерении обмена фиксируют число процессов, backend, размещение по узлам,
размер и тип тензора. Перед серией выполняют прогрев и согласуют её начало
через `barrier`; при измерении CPU-таймером дожидаются завершения CUDA-операций.
Время лучше собирать со всех процессов: слишком раннее завершение измерения
на одном участнике может скрыть ожидание на другом. Одинаковые размеры входных
тензоров обязательны для рассматриваемого all-reduce.

Наконец, указывать просто «пропускная способность» недостаточно.
В `nccl-tests` используются две нормировки:

$$
\begin{gathered}
B_{\mathrm{algorithm}}=S/t,\\
B_{\mathrm{bus}}=\frac{2(N-1)}{N}\frac{S}{t}
\quad\text{(all-reduce)}.
\end{gathered}
$$

Первая описывает скорость обработки тензора исходного размера, вторая
учитывает объём пересылок коллективной операции. Числа сравнимы только
при одинаковом определении. Формулы и оговорки приведены в
[NVIDIA nccl-tests: Performance](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md).

Проверки этой главы запускаются командой
`python publishing/tools/check_ddp_examples.py` из корня репозитория.
Скрипт использует небольшие модели и отдельные процессы CPU/Gloo: сравнивает
потери, градиенты, обновления и состояния оптимизатора с однопроцессным
эталоном. Он не измеряет NCCL, пропускную способность GPU или эффективность
перекрытия обмена с CUDA-вычислениями.

## Практика и первоисточники

Для последовательного разбора процесса обучения полезны три источника.
В лекции 7 CS336 коллективные операции показаны на маленьких тензорах,
а параллелизм по данным — на MLP с явным усреднением градиентов. Assignment 2
превращает этот пример в реализацию с объединением градиентов и перекрытием
обмена. Статья авторов PyTorch объясняет, какие условия должна соблюдать
такая реализация и почему одного all-reduce в конце `backward` недостаточно
для хорошей скорости.

- Percy Liang, Stanford CS336 Spring 2026: [Lecture 7 — Parallelism](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_07.py), функции `collective_operations`, `data_parallelism_main`, `benchmarking`.
- Tatsu Hashimoto, Stanford CS336 Spring 2026: [Lecture 8 — Parallelism Basics](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_08.pdf), pp. 6–8 и 14–17: соединения, коллективные операции, ограничения DDP.
- Stanford CS336: [Assignment 2 — Systems](https://github.com/stanford-cs336/assignment2-systems/tree/ca8bc81a59b70516f7ebb2da4808daade877c736), §5.1–5.3 и задачи по кольцевому all-reduce. Зафиксирована версия курса 2026 года.
- Shen Li et al., [PyTorch Distributed: Experiences on Accelerating Data Parallel Training](https://arxiv.org/abs/2006.15704), 2020, §2.2–3.2. Это описание ранней реализации DDP; актуальные параметры API следует проверять отдельно.
- PyTorch 2.14: [DDP](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html), [DistributedSampler](https://docs.pytorch.org/docs/2.14/data.html#torch.utils.data.distributed.DistributedSampler), [DDP design note](https://docs.pytorch.org/docs/2.14/notes/ddp.html). Проверено 13 октября 2026 года; design note содержит исторические детали, поэтому порядок редукций, `join`, `no_sync` и sampler дополнительно сверены с исходным кодом версии 2.14.0.

Более широкий разбор алгоритмов обмена и его сжатия есть в EDLS и Harvard:

- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week03_data_parallel/lecture.pdf|EDLS Week 3 — лекция]] и [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week03_data_parallel/practice.ipynb|исходный notebook]]. В [оригинальном PDF](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week03_data_parallel/lecture.pdf) pp. 24–35 посвящены all-reduce и кольцу, pp. 39–51 — gossip, pp. 52–62 — сжатию градиентов.
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week03_data_parallel/homework/README|EDLS Week 3 — практическое задание]].
- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol2/collective_communication|Harvard ML Systems — Collective Communication]] и [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol2/distributed_training|Distributed Training]]: алгоритмы обмена и организация распределённого обучения.
- Thijs Vogels, Sai Praneeth Karimireddy, Martin Jaggi, [PowerSGD](https://arxiv.org/abs/1905.13727), 2019: низкоранговое сжатие и экспериментальная проверка его стоимости.

← [[44 Distributed training и mixed precision]] · Далее: [[44b Gradient checkpointing и offload]]
