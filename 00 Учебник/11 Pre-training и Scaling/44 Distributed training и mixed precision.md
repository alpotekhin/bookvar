---
title: Распределённое обучение и смешанная точность
type: textbook-chapter
status: canonical
last_updated: 2026-07-31
---

# 44. Распределённое обучение и смешанная точность

Пусть мы хотим обучать семимиллиардную языковую модель с AdamW. Одни только
параметры в BF16 занимают около 14 GB, но на шаге обучения к ним добавляются
градиенты, два момента оптимизатора, иногда FP32-копия весов, активации и
временные буферы. В результате модель, которая без труда помещается на одном
80-гигабайтном GPU для inference, может потребовать значительно больше 120 GB
на устройство при обучении.

Распределённое обучение начинается не с выбора между DDP и FSDP, а с
восстановления полного пути одного training step. Нужно знать, где в каждый
момент находятся параметры, активации и градиенты, кто владеет каждой их
частью и какая операция должна привести все копии модели к одному состоянию.
Только после этого оси параллелизма перестают выглядеть как список сокращений.

## Один шаг обучения до распределения

Рассмотрим decoder-only Transformer. На вход подаётся batch токенов

$$
X\in\mathbb{N}^{B\times S},
$$

где $B$ — число последовательностей, а $S$ — их длина. После embedding
получаем активации $H_0\in\mathbb{R}^{B\times S\times d}$. Каждый из $L$
блоков сохраняет достаточно промежуточных значений, чтобы в backward вычислить
градиенты по входу и параметрам. Последний линейный слой производит logits
$Z\in\mathbb{R}^{B\times S\times V}$, а cross-entropy сравнивает их со
следующими токенами.

Один шаг состоит из пяти частей:

1. loader формирует batch и attention mask;
2. forward вычисляет logits и loss;
3. backward проходит по графу в обратном порядке и накапливает gradients;
4. optimizer обновляет параметры и свои состояния;
5. scheduler и счётчики переходят к следующему шагу.

Если используются $A$ microbatches с gradient accumulation, effective batch
равен $B_{global}=B_{micro}AN_{DP}$. Оптимизатор вызывается один раз после
$A$ backward-проходов. Loss либо делят на $A$ заранее, либо эквивалентно
нормируют накопленный градиент. Эта мелочь принципиальна: изменение числа GPU
не должно незаметно менять масштаб обновления.

## Из чего складывается память

Для $P$ параметров AdamW типичный mixed-precision ledger выглядит так:

| Состояние | Типичный dtype | Байтов на параметр |
|---|---:|---:|
| рабочие веса | BF16/FP16 | 2 |
| master weights | FP32 | 4 |
| градиенты | BF16 или FP32 | 2 или 4 |
| первый момент Adam | FP32 | 4 |
| второй момент Adam | FP32 | 4 |

Итого получается 16–18 B на параметр, то есть 112–126 GB для модели 7B ещё
до учёта активаций. Реальный пик выше из-за communication buckets, собранных
шардов и allocator fragmentation. Память активаций зависит уже не только от
$P$, но и от $B$, $S$, $d$ и $L$; особенно быстро она растёт при увеличении
длины контекста.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/training-systems-courses/44-cs336-dp-memory.png]]

*Источник: Stanford CS336, лекция 8, слайд 17: [оригинальный PDF](https://github.com/stanford-cs336/lectures/blob/main/lecture_08.pdf). Слайд отделяет параметры, градиенты, состояния оптимизатора и активации — именно эти категории нужно считать независимо.*

Для 7B-модели на восьми GPU обычный data parallelism не решает проблему:
каждый GPU всё ещё хранит собственные 112–126 GB model state. Если же
параметры, градиенты и optimizer state разделены между восемью устройствами,
идеальная нижняя оценка составляет 14–15,75 GB на rank. Это ещё не итог:
перед вычислением слоя могут временно собираться его полные параметры, а рядом
должны поместиться активации и сетевые буферы.

## Data parallelism: разные примеры, одна модель

Самый простой способ ускорить обучение модели, которая уже помещается на одном
GPU, — запустить по процессу на каждом устройстве. Все ranks начинают с
одинаковых параметров, но получают разные части batch:

$$
X_r\in\mathbb{N}^{B_{local}\times S},\qquad
B_{global}=N_{DP}B_{local}.
$$

Каждый rank самостоятельно выполняет forward и backward. Получившиеся
градиенты различаются, поэтому перед optimizer step DDP выполняет all-reduce:

$$
g=\frac{1}{N_{DP}}\sum_{r=0}^{N_{DP}-1}g_r.
$$

После редукции на каждом rank находится одинаковый $g$, и одинаковый
optimizer state производит одинаковое обновление параметров. Если data order,
randomness или scale градиента расходятся, реплики перестают быть
математически эквивалентны однопроцессному запуску.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/distributed/data-parallel-flow.svg]]

*Источник: Harvard Edge ML Systems Book, [Distributed Training, figure `fig-data-parallel-flow`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/distributed_training/distributed_training.qmd), CC BY-NC-SA 4.0.*

DDP объединяет параметры в buckets и запускает редукцию bucket сразу после
того, как backward вычислил все входящие в него gradients. Поэтому часть сети
можно скрыть вычислением предыдущих слоёв. Время шага определяется не суммой
длительностей NCCL kernels, а той частью communication, которая осталась на
critical path. При масштабировании compute на rank уменьшается, окно для
overlap сужается, и вчера скрытый all-reduce становится заметным.

## Collectives — язык распределённого шага

Оси параллелизма удобно описывать через несколько коллективных операций:

| Операция | Что было на rank | Что стало | Где встречается |
|---|---|---|---|
| all-reduce | локальный tensor одинаковой формы | редуцированный tensor на каждом rank | DDP, TP |
| all-gather | shard | полный tensor на каждом rank | FSDP, sequence parallelism |
| reduce-scatter | полный partial result | редуцированный shard | FSDP, TP |
| all-to-all | части для разных получателей | переставленные части | expert/context parallelism |
| broadcast | tensor только у root | его копия у всех | initialization, metadata |

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/distributed/collective-primitives-overview.svg]]

*Источник: Harvard Edge ML Systems Book, [Collective Communication, figure `fig-collective-primitives-overview`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/collective_communication/collective_communication.qmd), CC BY-NC-SA 4.0.*

Размер передаваемого tensor — только половина оценки. В простейшей модели
время collective записывают как

$$
T\approx n_{rounds}\alpha+\frac{Q}{B_{effective}},
$$

где $\alpha$ — latency одного раунда, $Q$ — объём переданных байтов, а
$B_{effective}$ — достижимая, не паспортная пропускная способность. Маленькие
сообщения ограничены latency; большие — bandwidth. Топология тоже важна:
обмен внутри NVLink-домена и тот же обмен через межузловую сеть имеют разную
цену.

## Когда реплика модели не помещается: ZeRO и FSDP

Data parallelism делит данные, но реплицирует всё состояние модели. ZeRO
последовательно устраняет эту избыточность:

| Режим | Реплицировано | Разделено между $N_{DP}$ ranks |
|---|---|---|
| DDP | параметры, gradients, optimizer state | — |
| ZeRO-1 | параметры, gradients | optimizer state |
| ZeRO-2 | параметры | gradients, optimizer state |
| ZeRO-3 / full-shard FSDP | — | параметры, gradients, optimizer state |

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/distributed/zero-memory-partitioning.svg]]

*Источник: Harvard Edge ML Systems Book, [Distributed Training, figure `fig-zero-memory-partitioning`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/distributed_training/distributed_training.qmd), CC BY-NC-SA 4.0. На рисунке показано, как последовательное шардирование трёх категорий состояния снижает постоянную память rank.*

При full-shard FSDP rank в покое хранит только свою часть параметров. Перед
forward очередного FSDP unit выполняется all-gather, и на короткое время каждый
rank получает его полные веса. После вычисления они освобождаются. В backward
веса этого unit собираются снова, а полный gradient превращается через
reduce-scatter в локальный shard. Затем каждый rank обновляет только
принадлежащую ему часть параметров и optimizer state.

Если BF16-параметры unit занимают $M$ байтов, ring all-gather и
reduce-scatter передают на rank примерно $(N-1)M/N$ каждый. Два all-gather
(forward и backward) и один reduce-scatter дают порядок

$$
V_{rank}\approx 3\frac{N-1}{N}M
$$

за шаг. FSDP обменивает постоянную репликацию на временную materialization и
communication. Слишком крупный unit создаёт высокий memory peak; слишком
мелкий — множество коротких collectives. Prefetch помогает только тогда,
когда следующий all-gather перекрывается с compute и оба полных unit
одновременно помещаются в память.

## Tensor parallelism: делим одну матрицу

Если даже один слой или его временные тензоры слишком велики, параметры слоя
раскладывают между устройствами. Пусть линейный слой вычисляет

$$
Y=XW,\qquad X\in\mathbb{R}^{M\times d_{in}},
\quad W\in\mathbb{R}^{d_{in}\times d_{out}}.
$$

При column parallelism матрицу $W$ режут по $d_{out}$:
$W=[W_1,\ldots,W_T]$. Каждый rank получает полный $X$ и вычисляет
$Y_i=XW_i$ формы $[M,d_{out}/T]$. Если следующая операция умеет работать с
таким layout, shards не нужно немедленно собирать. При row parallelism режется
$d_{in}$: rank вычисляет частичный результат $Y_i=X_iW_i$, после чего partial
results суммируются all-reduce или reduce-scatter.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/distributed/tensor-parallel-split.svg]]

*Источник: Harvard Edge ML Systems Book, [Distributed Training, figure `fig-tensor-parallel-split`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/distributed_training/distributed_training.qmd), CC BY-NC-SA 4.0.*

В Transformer согласованные column/row splits позволяют не собирать
промежуточную активацию после каждой матрицы. Но collective всё равно
повторяется в каждом блоке, поэтому TP обычно ограничивают быстрым
внутриузловым interconnect. В отличие от FSDP, который перемещает веса около
границ unit, TP постоянно перемещает активации или partial outputs.

## Pipeline parallelism: делим слои

Pipeline parallelism назначает последовательные группы слоёв разным stages.
Stage 0 принимает $H_0$, вычисляет свои блоки и посылает activation boundary
следующему stage; backward возвращает gradient этой границы в обратную
сторону. Параметры одного stage не нужно собирать на остальных устройствах.

Если выполнить сначала весь forward одного batch, а потом backward, большая
часть stages будет простаивать. Поэтому batch делят на microbatches и запускают
их конвейером. Для $p$ stages и $m$ microbatches идеализированная доля bubble
для расписания GPipe равна

$$
\mathrm{bubble\ fraction}\approx\frac{p-1}{m+p-1}.
$$

Чем больше $m$, тем лучше загрузка stages, но тем больше одновременно живых
активаций. Расписание 1F1B после разогрева чередует forward и backward и
сокращает этот пик. На практике stages должны быть сбалансированы не по числу
слоёв, а по измеренному времени и памяти; самый медленный stage задаёт темп
всего pipeline.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/distributed/pipeline-parallelism.svg]]

*Источник: Harvard Edge ML Systems Book, [Distributed Training, figure `fig-pipeline-parallelism`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/distributed_training/distributed_training.qmd), CC BY-NC-SA 4.0.*

## Expert parallelism: параметры идут к токенам не всегда

В Mixture-of-Experts каждый токен выбирает небольшое число экспертов. При
expert parallelism разные ranks хранят разные expert MLP. Router формирует для
каждого token representation адрес назначения; all-to-all отправляет токены
владельцам экспертов, а второй all-to-all возвращает результаты в исходный
порядок.

EP сокращает память expert weights на rank, но время слоя определяется самым
загруженным экспертом. Поэтому capacity, auxiliary balancing loss и
topology-aware placement влияют не только на качество маршрутизации, но и на
системную эффективность. Dense-параметры и experts могут иметь разные
data-parallel groups: редуцировать gradient эксперта по ranks, которые хранят
других экспертов, математически неверно.

## Как оси складываются в одну конфигурацию

На кластере из 64 GPU можно выбрать, например,

$$
N=DP\times PP\times TP\times EP=2\times4\times4\times2.
$$

Это не означает четыре независимых переключателя. У каждого rank есть
координата в многомерной сетке, а collective выполняется только внутри группы
соответствующей оси. TP-группы стараются размещать внутри узла; PP соединяет
соседние stages; EP учитывает пропускную способность all-to-all; DP связывает
реплики одной и той же части модели. FSDP может заменить обычную DP-репликацию
внутри этой же оси.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/training-systems-courses/44-cs336-parallelism-table.png]]

*Источник: Stanford CS336, лекция 8, слайд 55: [оригинальный PDF](https://github.com/stanford-cs336/lectures/blob/main/lecture_08.pdf). Таблица сопоставляет объект разбиения, экономию памяти и характер communication для основных осей.*

Sequence parallelism делит по ranks те активации, которые не обязаны быть
полностью реплицированы при TP. Context parallelism распределяет позиции
длинной последовательности и организует обмен ключами и значениями или
частичными attention results. Эти техники экономят activation memory, но не
отменяют причинно-следственную маску и требуют отдельной проверки layout.

## Смешанная точность как часть алгоритма

Формат чисел определяет одновременно память, объём communication, доступные
kernels и численную устойчивость. BF16 и FP16 занимают по два байта, но устроены
по-разному: BF16 сохраняет восьмибитную экспоненту FP32 и широкий dynamic
range, тогда как FP16 имеет больше бит мантиссы, но гораздо быстрее достигает
overflow и underflow.

В типичном mixed-precision step матричные умножения получают BF16/FP16 inputs,
а accumulation выполняется в FP32. Softmax, normalization statistics и
некоторые reductions также оставляют в более широком формате. Adam moments и
часто master weights хранятся в FP32. Поэтому переход на BF16 не уменьшает весь
model-state ledger ровно вдвое.

Для FP16 малые gradients могут округлиться в ноль. Loss scaling использует

$$
\widetilde L=sL,\qquad \nabla\widetilde L=s\nabla L.
$$

Backward выполняется для масштабированного loss, затем gradients делят на
$s$. Только после этого допустим gradient clipping. Dynamic scaler пропускает
optimizer step при `inf`/`nan`, уменьшает scale и постепенно повышает его после
стабильных шагов. BF16 благодаря широкому диапазону обычно не требует loss
scaling, хотя это не делает все операции автоматически устойчивыми.

```python
optimizer.zero_grad(set_to_none=True)
for microbatch in microbatches:
    with autocast(dtype=training_dtype):
        loss = model(microbatch) / len(microbatches)
    scaler.scale(loss).backward()

scaler.unscale_(optimizer)       # сначала вернуть настоящий масштаб
clip_grad_norm_(model.parameters(), max_norm)
scaler.step(optimizer)           # пропустить update при overflow
scaler.update()
```

В распределённом запуске ranks должны согласованно решить, выполнять ли step:
overflow на одном устройстве нельзя проигнорировать на остальных. Dtype
collective тоже задаётся явно. Например, BF16 all-reduce передаёт вдвое меньше
байтов, чем FP32, но может изменить ошибку суммирования; для чувствительных
редукций разумно сравнить обе кривые обучения.

FP8 добавляет scale для weights, activations и gradients, часто с отдельной
историей `amax`. Его преимущество нельзя выводить только из числа бит:
необходимы поддерживаемые hardware kernels, подходящие shapes и контроль
saturation/zero rate. Проверка всегда включает loss curve и downstream
качество, а не только отсутствие `NaN`.

## Сквозной расчёт: 7B на восьми GPU

Возьмём модель с $P=7\cdot10^9$, BF16 weights и gradients, FP32 master weights
и двумя FP32 Adam moments. Model state занимает

$$
7\cdot10^9(2+2+4+4+4)=112\,\mathrm{GB}.
$$

Обычный DDP требует 112 GB на каждом rank и не помещается на 80-GB GPU даже
до активаций. Full-shard FSDP снижает постоянную долю до

$$
112/8=14\,\mathrm{GB/rank}.
$$

Предположим, параметры одного FSDP unit занимают 2 GB в BF16. Во время его
вычисления полный unit добавляет примерно 2 GB, а агрессивный prefetch
следующего — ещё 2 GB. Если активации без checkpointing занимают 48 GB,
получаем уже около $14+2+2+48=66$ GB, не считая buffers и fragmentation.
Activation checkpointing, сокращающий сохранённые активации втрое ценой
повторного forward, опускает эту часть примерно до 16 GB и создаёт рабочий
запас.

За один unit FSDP передаст на rank приблизительно

$$
3\cdot\frac{7}{8}\cdot2=5.25\,\mathrm{GB}.
$$

При эффективных 200 GB/s это не менее 26 ms без учёта latency. Если compute
unit занимает 40 ms, большую часть обмена потенциально можно скрыть. Если
mixed precision и fused kernels сокращают compute до 20 ms, сеть внезапно
выходит на critical path. Именно поэтому ускорение отдельных kernels иногда
ухудшает scaling efficiency: оно уменьшает окно overlap.

Если один Transformer block всё ещё не помещается или его matmul недостаточно
быстр, можно добавить TP=2 внутри каждой пары NVLink-связанных GPU и оставить
FSDP по четырём таким парам. Это уменьшит локальные веса и крупные активации
матриц, но добавит collectives на каждом слое. Решение принимают по профилю, а
не по максимальному числу одновременно включённых техник.

## Как выбрать первый следующий шаг

| Наблюдаемое ограничение | Что пробуют сначала | Основная цена |
|---|---|---|
| модель помещается, шаг слишком медленный | DDP | gradient all-reduce |
| доминируют сохранённые активации | activation checkpointing | дополнительный forward compute |
| optimizer/gradient state не помещается | ZeRO-1/2 или FSDP | collectives и более сложный checkpoint |
| один слой или GEMM слишком велик | TP | обмены в каждом Transformer block |
| модель слишком глубока для одного устройства | PP | bubble, boundary activations, schedule |
| MoE expert weights не помещаются | EP | all-to-all и load imbalance |
| длинный контекст переполняет память | sequence/context parallelism | перераспределение активаций/KV |

Сначала измеряют peak memory по категориям и timeline шага. Затем добавляют одну
ось и проверяют три свойства: численную эквивалентность с доверенным запуском,
фактическое снижение нужной категории памяти и изменение critical path.
Конфигурация, которая «запустилась», ещё не обязательно корректна или быстрее.

## Что проверять при масштабировании

Надёжный переход начинается с однопроцессного baseline: сохраняют IDs примеров,
logits, loss, нормы и несколько элементов gradients, peak memory и step time.
После каждой новой оси сравнивают результат с этим запуском на одинаковом
global batch.

Для производительности фиксируют:

- useful tokens/s и время шага, а не только kernel throughput;
- MFU с явно указанной формулой model FLOPs;
- peak и reserved memory по категориям;
- exposed collective time и объём байтов каждой группы;
- pipeline bubble и дисбаланс stages/experts;
- время записи и восстановления checkpoint.

Strong scaling сохраняет размер задачи и делит её между большим числом GPU;
weak scaling увеличивает global batch вместе с числом устройств. Смешивать эти
режимы в одной таблице нельзя. Для checkpoint недостаточно сохранить shards
весов: нужны optimizer, scheduler, scaler, RNG каждого rank, позиция data
loader и metadata глобальных shapes. После восстановления выполняют следующий
batch и сравнивают его с непрерывным control run.

## Детальные главы

- [[44a Processes, collectives и DDP|Процессы, collectives и DDP]] — алгоритмы коллективных операций и overlap.
- [[44b Gradient checkpointing и offload|Gradient checkpointing и offload]] — время жизни активаций и перенос состояний.
- [[44c Tensor и sequence parallelism|Tensor и sequence parallelism]] — layouts матриц, attention и длинный context.
- [[44d Pipeline parallelism|Pipeline parallelism]] — schedules, microbatches и bubble.
- [[44e ZeRO, FSDP2, DeviceMesh и DTensor|ZeRO, FSDP2, DeviceMesh и DTensor]] — lifecycle shards и API PyTorch.
- [[44f Expert и hybrid parallelism|Expert и hybrid parallelism]] — MoE routing и многомерные meshes.
- [[44g Network, storage и distributed checkpoints|Сеть, хранилище и distributed checkpoints]] — физический путь данных.
- [[44h Fault tolerance и fleet orchestration|Отказоустойчивость и управление кластером]] — сбои, recovery и планирование.

## Практика и первоисточники

- Stanford CS336, [Lecture 8: Parallelism](https://github.com/stanford-cs336/lectures/blob/main/lecture_08.pdf) — memory accounting, data/tensor/pipeline parallelism и композиция осей.
- Efficient Deep Learning Systems, [week 2: Fast Pipelines and Mixed Precision](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week02_fast_pipelines/lecture.pdf), [week 3: Data Parallel Training](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week03_data_parallel/lecture.pdf), [week 4: Model Parallelism](https://github.com/mryab/efficient-dl-systems/tree/e632aa89ca9e6638d52e1b686095e7442faffbb0/week04_model_parallel) и [week 5: FSDP](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week05_fsdp/lecture.pdf).
- Harvard Edge ML Systems Book, [Distributed Training](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/distributed_training/distributed_training.qmd) и [Collective Communication](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/collective_communication/collective_communication.qmd), CC BY-NC-SA 4.0.
- Micikevicius et al., [Mixed Precision Training](https://arxiv.org/abs/1710.03740), 2018.
- Rajbhandari et al., [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054), 2020.
- Narayanan et al., [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473), 2021.
- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/tinytorch/08_training|TinyTorch 08 — Training]] — однопроцессный исполняемый baseline, с которым удобно сравнивать распределённый шаг.

← [[43 Scaling laws]] · Далее: [[44a Processes, collectives и DDP]]
