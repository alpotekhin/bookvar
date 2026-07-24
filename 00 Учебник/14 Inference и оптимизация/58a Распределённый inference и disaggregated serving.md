---
title: "Параллелизм и коллективные операции в LLM inference"
type: textbook-chapter
status: canonical
last_updated: 2026-07-23
last_verified: 2026-07-23
primary_sources:
  - https://jax-ml.github.io/scaling-book/sharding/
  - https://jax-ml.github.io/scaling-book/inference/
  - https://arxiv.org/abs/1909.08053
  - https://arxiv.org/abs/2104.04473
  - https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html
  - https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html
---

# Параллелизм и коллективные операции в LLM inference

Распределённый inference нужен по двум разным причинам. Иногда один экземпляр
модели не помещается в память одного ускорителя или не укладывается в требуемую
задержку. Тогда вычисление одного запроса приходится разделять между несколькими
GPU. В другом случае модель прекрасно помещается на одном устройстве, но один
экземпляр не успевает обслуживать весь входящий поток. Тогда модель реплицируют и
распределяют между копиями независимые запросы. Оба решения используют несколько
GPU, однако перемещают разные данные, ограничиваются разными каналами связи и
по-разному влияют на задержку одного пользователя.

Эта глава строится вокруг простого вопроса: **какой тензор разделён, где он
находится до операции и где должен оказаться после неё?** Аббревиатуры DP, TP, PP,
SP, CP и EP полезны только после ответа на этот вопрос. Если следить за формой и
расположением тензоров, становится понятно, зачем системе нужны `all-reduce`,
`all-gather`, `reduce-scatter`, `all-to-all` и point-to-point передача, почему TP
обычно держат внутри NVLink-домена, а также почему одна и та же конфигурация может
хорошо работать для prefill и плохо — для decode.

## От одного Transformer-блока к группе GPU

Пусть активации имеют форму $X\in\mathbb{R}^{B\times D}$, где $B$ здесь означает
число одновременно обрабатываемых токенов, а $D$ — ширину модели. Линейный слой
вычисляет $Y=XW$. На одном GPU и $X$, и $W$, и $Y$ локальны. Если матрица $W$ не
помещается или умножение нужно ускорить, её можно разрезать по столбцам:

$$
W=[W_1\;W_2\;\ldots\;W_p],\qquad
Y=[XW_1\;XW_2\;\ldots\;XW_p].
$$

Каждый из $p$ GPU получает полный $X$, хранит только свой $W_i$ и производит
свой фрагмент выхода. Пока следующая операция умеет работать с таким разбиением,
ничего собирать не требуется. Если же следующему слою нужен полный $Y$, shards
придётся объединить. Альтернативное разбиение по строкам даёт каждому GPU
частичную сумму; тогда требуется редукция. Именно из этих двух вариантов
distributed matmul выводится большая часть коммуникации в Transformer.

[How to Scale Your Model](https://jax-ml.github.io/scaling-book/sharding/)
предлагает полезную дисциплину: указывать не только глобальную форму массива, но
и ось device mesh, по которой разделено каждое измерение. Это снимает
двусмысленность слова «распределённый». Тензор может быть полностью
реплицирован, разделён по batch, hidden, sequence, heads или experts; два тензора
одинаковой формы могут иметь совершенно разное физическое размещение.

## Коллективные операции: четыре преобразования размещения

Коллективная операция выполняется согласованно всеми ranks одной process group.
Это не «сетевой вызов после вычисления», а часть самого распределённого
алгоритма: пока нужный collective не завершён, следующий слой часто не имеет
корректных входов.

### AllGather: из shards в полную копию

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/distributed-serving-2026/nccl-allgather.png]]

*Официальная схема NCCL: каждый rank передаёт свой фрагмент, после чего все ranks
получают конкатенацию фрагментов в порядке rank. Источник: NVIDIA,
[Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html),
BSD-licensed NCCL documentation.*

Если rank $i$ хранит $x_i$, то после AllGather каждый получает
$[x_0,x_1,\ldots,x_{p-1}]$. Объём локального результата увеличивается в $p$ раз.
AllGather нужен, когда последующая операция не может продолжать вычисление над
разделённым представлением. Например, sequence parallelism может хранить
активации LayerNorm по частям, а перед tensor-parallel linear восстановить
нужное размещение.

### ReduceScatter: просуммировать и оставить результат разделённым

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/distributed-serving-2026/nccl-reducescatter.png]]

*ReduceScatter сначала редуцирует соответствующие элементы входов, затем оставляет
на каждом rank только его часть результата. Источник: официальная документация
NCCL.*

Пусть каждый rank вычислил partial output $y_i$ одной и той же глобальной формы.
Полный результат равен $y=\sum_i y_i$. Если следующая операция допускает
разделённый $y$, нет смысла размножать всю сумму на каждом устройстве:
ReduceScatter сразу оставляет $1/p$ результата. Это уменьшает локальную память и
часто лучше сочетается с последующим AllGather, чем отдельный AllReduce.

### AllReduce: полная сумма на каждом rank

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/distributed-serving-2026/nccl-allreduce.png]]

*AllReduce возвращает результат редукции всем ranks. Его можно мыслить как
ReduceScatter, за которым следует AllGather. Источник: официальная документация
NCCL.*

AllReduce возникает в классическом Megatron TP после row-parallel projection:
GPU вычисляют частичные суммы по разрезанному contracting dimension, а следующий
residual path ожидает одинаковую полную активацию на всех ranks. Простая модель
стоимости ring AllReduce для сообщения размером $M$ байт на $p$ ranks содержит
два обхода кольца:

$$
T_{AR}\approx 2(p-1)\alpha+2\frac{p-1}{p}\frac{M}{\beta},
$$

где $\alpha$ — задержка одного шага, а $\beta$ — эффективная пропускная
способность. Формула грубая: NCCL выбирает алгоритм с учётом topology и размера
сообщения. Но она показывает два разных режима. Для маленьких decode activations
заметна latency; для больших сообщений prefill важнее bandwidth.

### AllToAll: персональная перестановка между всеми ranks

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/distributed-serving-2026/nccl-alltoall.png]]

*При AllToAll каждый rank отправляет отдельный fragment каждому назначению и
получает отдельный fragment от каждого источника. Источник: официальная
документация NCCL.*

AllToAll — не редукция. Это глобальная перестановка размещения. В MoE router
назначает каждый токен одному или нескольким экспертам; tokens, находящиеся на
разных ranks, нужно доставить владельцам выбранных экспертов. После локального
FFN выполняется обратная перестановка. Поэтому EP нагружает bisection bandwidth
и чувствителен к неравномерности числа токенов гораздо сильнее, чем dense TP.

Наконец, pipeline parallelism использует преимущественно point-to-point
send/receive между соседними stages. Здесь сообщение проходит не ко всем ranks,
а по определённому ребру pipeline; цена проявляется в задержке передачи и
простоях несбалансированных стадий.

## Data parallelism: независимые реплики сервиса

В training data parallelism означает одинаковые веса, разные mini-batches и
синхронизацию gradients. В обычном inference градиентов нет. Поэтому полезнее
говорить о **request-level replication**: каждая replica содержит полный
исполняемый экземпляр модели, а router направляет ей целые запросы.

DP повышает суммарную request capacity почти линейно, пока не упирается во
внешний bottleneck. Он не уменьшает весовую память одного экземпляра и не
ускоряет отдельный forward pass. Зато replicas не требуют синхронного collective
на каждом слое, независимо формируют continuous batches и служат естественной
границей отказа.

Репликация всё же не полностью stateless. У каждого worker собственный KV cache
и собственная история prefix-cache blocks. Маршрутизация только по текущей
очереди хорошо балансирует вычисления, но может уничтожить prefix locality;
маршрутизация только по cache hit создаёт hot replicas. В MoE слово DP также
может обозначать более сложные process groups — например, attention weights
реплицированы, а experts распределены. Поэтому конфигурация `DP=8` без описания
model placement недостаточна.

## Tensor parallelism: матрица разделена внутри слоя

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/inference/tensor-parallel-routing.svg]]

*Harvard ML Systems, Vol. II, tensor-parallel routing; [pinned original](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/inference/images/svg/tensor-parallel-routing.svg), CC BY-NC-SA 4.0.*

Megatron-LM выбрал сопряжённую пару column-parallel и row-parallel linear.
Первую матрицу MLP делят по output dimension. GeLU или SwiGLU можно вычислить
локально над каждым fragment, не синхронизируя нелинейность. Вторую матрицу делят
по input dimension; каждый rank получает partial sum, после чего результаты
редуцируются. Аналогично Q, K и V можно делить по heads, а output projection —
по contracting dimension.

Это разбиение привлекательно тем, что между двумя большими GEMM коммуникации нет,
а синхронизация происходит в нескольких заранее известных местах блока. Но
каждый Transformer layer повторяет эти collectives. Если TP пересекает медленную
межузловую сеть, задержка накапливается десятки раз за один token step.

Для prefill число токенов $B$ велико: GEMM достаточно крупный, и коммуникацию
иногда удаётся перекрыть вычислениями. Для decode каждый запрос даёт один token,
а batch ограничен KV memory и SLO. Локальный GEMM становится короче, тогда как
collective latency никуда не исчезает. С другой стороны, TP распределяет веса по
нескольким HBM channels и способен уменьшить нижнюю границу времени чтения
параметров. Поэтому decode может выигрывать от TP даже в memory-bound режиме,
пока межсоединение быстрее сэкономленного HBM traffic.

Практический выбор степени TP начинается не с максимума GPU, а с трёх проверок:

1. помещаются ли weights, runtime buffers и целевой KV budget;
2. уменьшается ли measured step latency при переходе TP1 → TP2 → TP4 → TP8;
3. остаются ли TP ranks внутри быстрого NVLink/NVSwitch domain.

После этого оценивают tokens/s на GPU: TP8 может дать меньшую latency, но худшую
стоимость токена, чем две независимые TP4 replicas.

## Pipeline parallelism: слои разделены по глубине

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/inference/pipeline-parallel-routing.svg]]

*Harvard ML Systems, Vol. II, pipeline-parallel routing; [pinned original](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/inference/images/svg/pipeline-parallel-routing.svg), CC BY-NC-SA 4.0.*

PP назначает последовательные слои разным stages. Активация проходит stage 0,
передаётся stage 1 и так далее. В отличие от TP, большинство слоёв не вызывает
collective между всеми GPU; связь нужна на границах stages. Это позволяет
разместить модель, которая не помещается в один узел, и использовать более
медленный межузловой канал для относительно компактных activations.

Но один запрос не становится параллельным по глубине: его следующий stage ждёт
предыдущий. Throughput возникает, когда pipeline одновременно содержит разные
microbatches. До заполнения и при опустошении часть stages простаивает — это
pipeline bubble. В online inference величина bubble зависит не только от числа
stages, но и от переменной длины prompt, завершения decode sequences и
неодинаковой стоимости слоёв.

Равное число layers на stage не гарантирует баланс. Embedding и output head
могут быть дорогими, MoE layers зависят от token routing, а разные attention
варианты по-разному масштабируются с контекстом. Поэтому Megatron Core допускает
custom pipeline layouts и virtual stages. Для serving план нужно проверять на
реальном распределении input/output lengths, а не только на одинаковых
synthetic sequences.

PP часто выбирают, когда модель не помещается при разумном intra-node TP.
Комбинация, например TP8×PP4, означает: каждый stage исполняется группой из восьми
GPU, а четыре такие группы хранят разные quarters модели. Request проходит четыре
stages, а внутри каждого слоя выполняются TP collectives.

## Sequence parallelism и context parallelism — не синонимы

Терминология библиотек различается, но в Megatron эти методы решают разные
задачи. **Sequence parallelism (SP)** дополняет TP: операции вроде LayerNorm и
dropout, которым не нужен обмен между токенами, выполняются над разделённой
sequence dimension. Вместо полного AllReduce используются сопряжённые
ReduceScatter/AllGather, и replicated activation memory уменьшается.

**Context parallelism (CP)** разделяет по sequence dimension весь network input и
все activations, включая attention. Линейные слои могут работать локально, но
каждый query должен увидеть K/V всего разрешённого causal prefix. Поэтому ranks
обмениваются KV blocks или накапливают partial attention results.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/distributed-serving-2026/megatron-context-parallel-overview.png]]

*Transformer layer при TP2×CP2. Операции около attention относятся к CP, около
linear blocks — к TP; AG и RS обозначают AllGather и ReduceScatter. Источник:
NVIDIA, [Megatron Core Context Parallelism](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html),
Figure 1.*

Для длинного prefill CP уменьшает activation memory на rank и позволяет
распараллелить очень длинный context. Для decode новый query имеет sequence
length 1, однако исторический KV cache остаётся длинным. Здесь полезно различать
query distribution и KV sharding: можно разделить KV по heads, batch или context,
но partial attention outputs всё равно нужно объединить на каждом шаге. CP не
является бесплатным способом ускорить decode и особенно чувствителен к latency
между ranks.

## Expert parallelism: параметры распределены, токены движутся

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/inference/expert-parallel-routing.svg]]

*Harvard ML Systems, Vol. II, expert-parallel routing; [pinned original](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/inference/images/svg/expert-parallel-routing.svg), CC BY-NC-SA 4.0.*

Dense MLP применяет одни weights ко всем токенам. В MoE имеется $E$ экспертов, а
router выбирает для токена обычно $k\ll E$. EP распределяет experts между ranks,
чтобы не реплицировать все параметры. На каждом MoE layer выполняется цепочка:

1. router вычисляет expert indices и weights;
2. tokens группируются по назначениям;
3. AllToAll отправляет tokens ranks-владельцам;
4. локальные grouped GEMM считают experts;
5. обратный AllToAll возвращает outputs;
6. результаты top-k experts смешиваются в исходном token order.

Compute определяется числом активированных параметров, memory capacity — всеми
размещёнными experts, а коммуникация — количеством routed tokens. Это три разные
величины. Средний баланс не гарантирует низкую tail latency: один переполненный
expert задерживает collective для всей группы. Capacity factor, token dropping,
padding, expert replication и Expert Parallel Load Balancing меняют баланс между
качеством, памятью и задержкой и должны быть явно зафиксированы в benchmark.

EP также взаимодействует с TP. Если каждый expert сам разделён tensor-parallel,
система сначала доставляет token нужной expert group, затем выполняет collectives
внутри неё. Современные конфигурации DeepSeek-подобных MoE могут сочетать EP,
expert TP, attention DP и PP. Произведение чисел GPU ещё не описывает реальную
сеть process groups; нужен rank map и перечень collectives на layer.

## Как топология превращает алгоритм в производительность

У GPU есть несколько уровней движения данных: HBM внутри устройства, PCIe,
NVLink/NVSwitch внутри сервера и InfiniBand/RoCE между серверами. Их bandwidth и
latency различаются на порядки. Кроме того, важны NUMA placement, число NIC,
oversubscription leaf/spine fabric и то, через какой link реально проходит NCCL.

Частые синхронные collectives следует помещать в самый быстрый domain. Отсюда
обычное, но не абсолютное правило:

- TP — внутри NVLink/NVSwitch node;
- EP — внутри high-bisection-bandwidth domain;
- PP — допускает границы узлов, если activation transfer невелик;
- DP replicas — могут находиться в разных fault/network domains.

Правило меняется с workload. Длинный prefill имеет крупные GEMM и способен
скрывать часть communication. Decode step короток и сильнее реагирует на каждую
добавленную синхронизацию. Большое сообщение ограничено bandwidth; маленькое —
latency и software launch overhead. Поэтому название interconnect не заменяет
измерение NCCL collectives именно для размеров сообщений модели.

Полезный диагностический порядок таков: сначала измерить single-GPU kernel/HBM
roofline, затем microbenchmark collectives по нужным process groups, после этого
профилировать layer timeline и только затем end-to-end serving. Низкий GPU
utilization сам по себе не доказывает нехватку запросов: GPU может ждать
collective, remote expert или следующий pipeline stage.

## Как выбирать композицию DP×TP×PP×CP×EP

Начинать следует с минимальной группы, в которую помещается модель вместе с
реальным KV budget. Для dense model обычно сначала используют intra-node TP;
если этого недостаточно, добавляют PP. После выбора model-parallel replica
оставшиеся GPU превращают в независимые service replicas. CP добавляют только
при ограничении по длинному контексту. EP задаётся архитектурой MoE, но степень и
placement экспертов всё равно остаются инженерным выбором.

Для каждого кандидата нужно записать:

- какие weights реплицированы и какие sharded;
- placement activations и KV cache;
- collective sequence одного dense и одного MoE layer;
- границы node/NVLink/NIC;
- максимальный batch/KV tokens;
- measured TTFT, ITL, throughput и tokens/s/GPU.

Например, запись «32 GPU» почти бесполезна. Запись «4 replicas × (TP8), TP ranks
внутри четырёх NVSwitch nodes, request router cache-aware» уже позволяет понять
путь данных. Для MoE может потребоваться более подробная форма: attention DP4,
EP32, expert TP1, redundant shared experts и отдельная communication group для
token dispatch.

## Граница этой главы

Parallelism отвечает на вопрос, как один model instance или множество replicas
используют GPU и interconnect. Он не определяет, где должны исполняться prefill и
decode. Обе фазы можно выполнять в одной distributed replica либо разнести по
разным пулам с разными TP/PP/DP layouts. Во втором случае появляется новый вид
движения данных — передача KV cache между экземплярами, а routing и autoscaling
становятся связанными задачами. Это предмет следующей главы.

## Небольшой пример чтения конфигурации

Пусть 70B dense-модель обслуживается на шестнадцати GPU как две replicas TP8.
Внутри каждой replica восемь ranks хранят по одной восьмой tensor-parallel
weights; после row-parallel projections они участвуют в AllReduce. Router
распределяет целые requests между двумя replicas, поэтому между ними нет
collective на каждом layer. У каждой replica отдельный continuous batch и
отдельный KV cache. Если один NVSwitch node содержит восемь GPU, такая схема
оставляет частые TP collectives внутри node и использует межузловую сеть только
для frontend traffic и служебной координации.

Альтернатива TP4×DP4 использует четыре replicas. Она может дать больший aggregate
throughput и лучшую failure isolation, но на каждый rank приходится вдвое больше
weights; остаётся меньше HBM для KV и длинного контекста. TP16×DP1, напротив,
распределяет weights шире и может снизить single-request decode latency, однако
collectives пересекут границу узла, а единственная replica станет общей точкой
очереди и отказа. Ни одна запись не лучше сама по себе: TP8×DP2, TP4×DP4 и
TP16×DP1 занимают те же шестнадцать GPU, но дают разные latency, capacity, KV
budget и network traffic. Именно поэтому benchmark должен публиковать layout, а
не только модель и число ускорителей.

## Источники и дальнейшее чтение

- Austin et al., [How to Scale Your Model: Sharded Matrices](https://jax-ml.github.io/scaling-book/sharding/) — вывод collectives из размещения distributed tensors.
- Austin et al., [All About Transformer Inference](https://jax-ml.github.io/scaling-book/inference/) — sharding prefill, decode и KV cache с roofline-анализом.
- Shoeybi et al., [Megatron-LM](https://arxiv.org/abs/1909.08053) — исходная схема tensor parallel Transformer.
- Narayanan et al., [Efficient Large-Scale Language Model Training Using Megatron-LM](https://arxiv.org/abs/2104.04473) — композиция tensor, pipeline и data parallelism; механика partitioning применима и к inference.
- NVIDIA, [Megatron Core Parallelism Strategies](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html) — современная терминология DP, TP, PP, CP и EP.
- NVIDIA, [NCCL Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) — точные определения коллективных операций.

**Предыдущая глава:** [[58 Спекулятивное декодирование|Спекулятивное декодирование]]

**Следующая глава:** [[58a2 Раздельное обслуживание prefill и decode|Раздельное обслуживание prefill и decode]]
