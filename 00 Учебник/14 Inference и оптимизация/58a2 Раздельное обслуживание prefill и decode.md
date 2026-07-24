---
title: "Раздельное обслуживание prefill и decode"
type: textbook-chapter
status: canonical
last_updated: 2026-07-23
last_verified: 2026-07-23
primary_sources:
  - https://jax-ml.github.io/scaling-book/inference/
  - https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf
  - https://arxiv.org/abs/2311.18677
  - https://arxiv.org/abs/2407.00079
  - https://docs.nvidia.com/dynamo/design-docs/disaggregated-serving
  - https://sgl-project.github.io/advanced_features/pd_disaggregation.html
---

# Раздельное обслуживание prefill и decode

Во время генерации один запрос дважды предъявляет к системе почти противоположные
требования. Сначала модель получает весь prompt и строит для него KV cache. Затем
она много раз исполняет короткий forward pass, каждый раз добавляя ровно один
новый токен. Первая стадия обычно располагает большим параллелизмом по токенам и
хорошо использует tensor cores; вторая повторно читает веса и всё более длинный
KV cache ради одного шага. Если обе стадии исполняются на одном GPU pool, ими
проще управлять и не нужно переносить состояние. Но они конкурируют за scheduler
slots, HBM и выбранную конфигурацию parallelism.

**Prefill/decode disaggregation** разрывает эту связь: prefill workers принимают
prompt и создают KV, decode workers получают это состояние и продолжают
авторегрессию. Такое разделение не является безусловным ускорением. Оно заменяет
локальную интерференцию распределённым протоколом с передачей большого состояния,
согласованием двух очередей и отдельным контуром capacity planning. Чтобы понять,
когда обмен выгоден, сначала разберём обе стадии количественно.

## Один запрос: от очереди до последнего токена

Упрощённый жизненный цикл online request выглядит так:

1. frontend проверяет запрос, применяет chat template и tokenization;
2. router выбирает replica и учитывает возможность prefix-cache hit;
3. scheduler помещает prompt в waiting queue;
4. prefill вычисляет скрытые состояния, K/V каждого слоя и logits последней позиции;
5. sampler выбирает первый output token — заканчивается TTFT;
6. запрос занимает slot и KV blocks в decode batch;
7. каждый decode step читает weights и KV, создаёт следующий token и дописывает KV;
8. при EOS, stop condition, cancellation или limit blocks освобождаются.

Пользователь воспринимает по меньшей мере две задержки. **Time to first token**
включает frontend, очередь и prefill:

$$
TTFT=T_{frontend}+T_{queue,P}+T_{prefill}+T_{first\ sample}.
$$

**Inter-token latency** или **time per output token** характеризует паузы после
первого токена. Для ответа длиной $O$ end-to-end latency приближённо равна

$$
E2E=TTFT+\sum_{i=2}^{O}ITL_i.
$$

Среднее ITL скрывает stalls: пользователь заметит длинную паузу, даже если
соседние токены пришли быстро. Поэтому serving оценивают percentiles TTFT и ITL,
а не только aggregate tokens/s.

## Почему prefill чаще compute-bound

Во время prefill все $S$ prompt tokens известны. Линейные слои получают матрицу
активаций с большой token dimension и переиспользуют один раз загруженные weights
для множества строк. Arithmetic intensity растёт с числом токенов. Для достаточно
длинного prompt крупные GEMM достигают compute roofline; attention также имеет
много работы над каждым загруженным блоком.

Это не означает, что любой prefill compute-bound. Короткие prompts, маленькие
batch, kernel launch overhead, quantization и неудачный TP могут оставить GPU
memory- или communication-bound. Однако системный контраст остаётся: у prefill
есть независимые token positions и возможность построить крупную матричную
операцию. У decode такой возможности внутри одного запроса нет.

Стоимость prefill зависит прежде всего от input sequence length. Attention до
применения оптимизированного tiled kernel имеет квадратичную работу по $S$;
линейные слои — линейную по числу токенов, но доминируют по параметрам. Длинный
prompt способен занять GPU на десятки или сотни обычных decode steps. Именно
поэтому scheduler, вставляющий prefill в текущий decode batch, может вызвать
заметную паузу в уже стримящемся ответе.

## Почему decode чаще memory-bandwidth-bound

После prefill следующий token неизвестен до завершения текущего шага. Один
request даёт лишь одну новую позицию, поэтому увеличить token batch можно только
объединив много параллельных запросов. На каждом шаге система читает weights всей
модели; attention дополнительно читает индивидуальный KV cache каждого request.

Размер KV для одного request с $S$ сохранёнными токенами равен

$$
M_{KV}=2L S n_{kv}d_h b,
$$

где $L$ — число слоёв, $n_{kv}$ — число KV heads, $d_h$ — размер head, $b$ —
байт на элемент, а множитель 2 соответствует K и V. MQA, GQA, MLA и KV
quantization уменьшают эту величину и тем самым влияют не только на вместимость,
но и на decode latency и стоимость будущей передачи между pools.

Для небольшого decode batch нижняя граница step time определяется чтением
parameters и KV:

$$
T_{step}\gtrsim
\frac{M_{weights}+\sum_{r\in batch}M_{KV,r}}
{\text{aggregate HBM bandwidth}}.
$$

При росте batch weights переиспользуются лучше, но каждый новый request приносит
свой KV. Поэтому throughput растёт с diminishing returns, а HBM capacity
ограничивает maximum concurrency. Большой batch выгоден оператору, но увеличивает
время одного step и ITL. Serving engine постоянно выбирает точку на
latency-throughput Pareto frontier.

## Интерференция в общем GPU pool

Современный unified engine использует iteration-level scheduling: завершив один
forward pass, он может удалить закончившиеся sequences, принять новые и собрать
следующий batch. Это намного эффективнее статического batching. Тем не менее
prefill и decode остаются разными по продолжительности и resource profile.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/distributed-serving-2026/distserve-fig2-prefill-decode-interference.png]]

*DistServe, Figure 2: время batch при добавлении одного prefill к decode requests
для 13B-модели. Decode замедляется в ожидании длинного prefill, а сам prefill
также удлиняется из-за совместного выполнения. Источник: Zhong et al.,
[OSDI 2024](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf), crop
оригинального рисунка без содержательных изменений.*

Есть три обычные стратегии, но ни одна не устраняет фундаментальный конфликт.
Приоритет prefill сокращает очередь новых requests и TTFT, но создаёт ITL spikes
у активных ответов. Приоритет decode делает streaming ровнее, но prompts ждут.
Chunked prefill разбивает длинный prompt на fragments и чередует их с decode. Это
ограничивает длительность одной блокировки, однако требует подобрать chunk size:
слишком малый chunk недогружает GPU и повторно читает предыдущий KV, слишком
крупный снова создаёт длинную паузу.

Есть и второй конфликт: unified replica должна использовать совместимые model
placement и parallelism для обеих фаз. Prefill может предпочитать меньший TP и
крупные compute-efficient GEMM, а decode — более широкий TP для снижения HBM
weight-read latency. Decode хранит много долгоживущих KV caches; prefill нужно
лишь временное состояние текущих prompts. Один layout вынужден быть
компромиссным.

## Что именно меняет disaggregation

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/inference/disaggregated-serving.svg]]

*Harvard ML Systems, Vol. II, `inference.qmd`: независимые prefill/decode pools с KV handoff; [pinned original](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/inference/images/svg/disaggregated-serving.svg), CC BY-NC-SA 4.0.*

В disaggregated architecture имеются как минимум два независимо масштабируемых
типа model instances. Prefill instance получает tokens, вычисляет KV cache и
первый token. После этого decode instance должен получить совместимое KV state,
зарезервировать memory blocks и включить request в continuous batch. Weights
обычно присутствуют в обоих pools; экономия возникает не из-за одной копии
модели, а из-за специализации и независимого provisioning.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/distributed-serving-2026/distserve-fig6-runtime-architecture.png]]

*DistServe, Figure 6: controller сначала выбирает prefill instance, затем decode
instance; KV cache становится явным передаваемым состоянием. Источник: Zhong et
al., [DistServe](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf),
crop оригинального рисунка.*

Разделение даёт четыре потенциальных преимущества:

1. prefill больше не вставляет длинную работу между decode steps;
2. pools независимо масштабируются под TTFT и ITL SLO;
3. для фаз можно выбрать разные TP/PP/DP layouts или hardware;
4. prefill workers не обязаны удерживать KV всех активных generations.

Взамен появляются четыре обязательные цены:

1. weights модели дублируются в двух pools;
2. KV cache входит в network critical path;
3. router должен согласовать выбор prefill и decode destinations;
4. отказ и backpressure распространяются между стадиями.

Поэтому сравнение unified и disaggregated deployments должно учитывать все GPU,
network и SLO attainment. Пиковое tokens/s только decode pool создаёт ложный
вывод, потому что prefill capacity и KV transfer оплачены, но исключены из
знаменателя.

## KV transfer как отдельная стадия запроса

Передаваемый объём почти совпадает с KV memory, созданной для prompt. Нижняя
граница времени передачи:

$$
T_{transfer}\ge
\frac{M_{KV}}{BW_{effective}}+T_{coordination}.
$$

Номинальные 400 Gbit/s NIC не означают 50 GB/s полезного KV traffic. На
эффективность влияют PCIe path, GPUDirect RDMA, число NIC, NUMA placement,
registration, message fragmentation, concurrent flows и fallback на TCP. Для
TP/PP cache состоит из shards: не один большой tensor перемещается между двумя
процессами, а согласованный набор layer/head shards между соответствующими ranks.

Существует несколько вариантов протокола.

**Push**: prefill отправляет KV сразу после вычисления. Это просто, но burst из
prefill completions способен переполнить decode memory. **Pull**: decode сначала
резервирует capacity и забирает KV, когда готов; DistServe использует prefill GPU
memory как временный buffer. Pull облегчает backpressure, но требует хранить KV
на source до подтверждения.

**Layer-wise streaming** начинает передачу K/V ранних слоёв, пока prefill ещё
вычисляет поздние. Mooncake описывает overlap computation и transfer. Выигрыш
зависит от того, успевает ли сеть закончить к моменту завершения последнего слоя;
иначе transfer tail всё равно увеличит TTFT. **Remote exposure** или shared cache
может позволить decode читать blocks из внешнего memory tier, но decode attention
обычно слишком чувствителен к bandwidth, чтобы постоянно работать с remote KV;
чаще remote tier используется для bootstrap, reuse или offload.

Нельзя передавать KV только потому, что совпадает model name. Должны совпадать
revision, dtype/quantization, layer layout, number of KV heads, block size,
positional convention, parallel rank mapping и backend serialization. Изменение
TP degree между pools требует явного resharding. Некоторые системы поддерживают
разные layouts, но цена преобразования должна входить в measurement.

## Routing: вычислительная нагрузка встречается с locality

В stateless web service least-loaded routing часто достаточно. В LLM serving
destination хранит дорогое состояние. Router решает две задачи: где дешевле
выполнить prefill с учётом cached prefix и где есть место для будущего decode KV.
Эти решения могут конфликтовать.

Если всегда выбирать maximum prefix overlap, популярный system prompt создаёт
hotspot. Если всегда выбирать shortest queue, одинаковые prefixes размазываются
по replicas и recompute увеличивает TTFT. Практический score сочетает cache
overlap, queued tokens, estimated service time, available KV blocks и topology.
При PD-disaggregation нужно выбрать пару или последовательность destinations:
prefill worker, decode worker и transfer path.

Reservation decode capacity до запуска prefill уменьшает риск закончить дорогой
prefill и обнаружить, что KV некуда поместить. Но ранняя reservation удерживает
memory, пока prefill работает, и может снизить utilization. Позднее назначение
лучше балансирует актуальную нагрузку, но увеличивает вероятность ожидания или
повторной маршрутизации. Здесь нет универсальной политики: выбор зависит от
распределения input/output lengths и burstiness.

KV-aware routing требует актуального индекса blocks. Event delivery может быть
точным, durable или приблизительным. Устаревшая запись о cache hit приводит к
miss и recompute; устаревшая запись о свободной памяти — к admission failure.
Control-plane consistency обычно слабее транзакционной базы данных, поэтому
protocol должен терпеть расхождения и подтверждать решение на worker.

## Две связанные очереди и backpressure

После разделения prefill становится producer, decode — consumer долгоживущего
состояния. Если prefill завершает requests быстрее, чем decode освобождает KV,
растёт промежуточный backlog. Свободные tensor cores prefill не означают, что
можно продолжать admission: система ограничена downstream memory и generation
rate.

Backpressure может опираться на несколько сигналов:

- queued input tokens и predicted TTFT в prefill pool;
- active/queued KV tokens и predicted ITL в decode pool;
- bytes ожидающего transfer и measured network throughput;
- reserved, received и evictable KV blocks;
- долю rejected, cancelled и recomputed requests.

Обычная ошибка — ограничивать очередь числом requests. Prompt в 128 tokens и
prompt в 128K создают несопоставимую работу; decode request с output budget 32 и
4096 tokens удерживает memory разное время. Полезнее планировать в token units и
учитывать joint distribution input/output lengths.

Mooncake показывает ещё один эффект: решение об admission на основании текущей
decode load запаздывает на длительность prefill. Если принять много requests при
свободном decode pool, все они придут downstream позже одной волной. Затем
система начнёт отвергать новые requests, prefill опустеет, а после drain цикл
повторится. Предсказание будущей нагрузки и coordinated admission уменьшают эти
противофазные колебания.

## Independent scaling — не независимое управление

Раздельные pools позволяют менять число prefill и decode replicas по отдельности,
но scaling decisions связаны workload. Prefill capacity приблизительно зависит
от arrival rate и input token distribution; decode capacity — от output rate,
active contexts и KV memory. Изменение доли длинных prompts требует больше P,
изменение длины reasoning outputs — больше D.

Обычный CPU/GPU utilization — слабый autoscaling signal. Compute-bound prefill
может иметь высокий utilization и нормальный TTFT; memory-bound decode способен
иметь умеренный compute utilization и нарушать ITL. Более подходящие сигналы:

| Pool | Основные сигналы |
|---|---|
| Prefill | queued input tokens, ISL, TTFT percentiles, prefix-hit rate |
| Decode | active KV tokens, context length, ITL, free KV blocks |
| Transfer | pending bytes, transfer latency, RDMA throughput/fallbacks |
| Frontend | arrival rate, cancellations, deadline/SLO class |

[NVIDIA Dynamo Planner](https://docs.nvidia.com/dynamo/components/planner/planner-guide)
разделяет долгосрочное capacity planning и быструю реакцию на load. Performance
model оценивает, сколько P/D replicas нужно для target TTFT/ITL, а short-interval
loop реагирует на queue и KV utilization. Полезная operational практика — сначала
запустить planner в advisory mode и сравнить его recommendations с replay trace,
не позволяя сразу менять production replicas.

Scale-up не мгновенен: нужно назначить GPU, загрузить weights, инициализировать
NCCL, прогреть kernels/CUDA graphs и зарегистрировать worker. Если cold start
дольше burst, reactive autoscaler опоздает. Нужны minimum warm capacity,
prediction или queue admission. Scale-down также stateful: decode worker должен
drain active sequences или мигрировать KV; prefill worker нельзя удалить во время
неподтверждённого transfer.

## DistServe, Splitwise и Mooncake: три исследовательских акцента

**Splitwise** начинает с hardware asymmetry. Prompt computation требует сильных
compute accelerators, token generation недоиспользует compute и больше зависит от
memory. Авторы рассматривают homogeneous и heterogeneous pools, стоимость и power
budget. Это хороший источник для вопроса «зачем специализировать hardware», но не
полная современная реализация cache-aware routing.

**DistServe** формулирует цель как per-GPU goodput при одновременных TTFT и TPOT
SLO. Он показывает measured interference, независимо подбирает parallelism для
фаз и учитывает network bandwidth при placement. Runtime использует controller,
prefill/decode instances и pull-based KV transmission. Главный урок: максимум
throughput без latency constraints может выбрать систему, непригодную для
interactive service.

**Mooncake** ставит KV cache в центр architecture. Помимо P/D pools, система
использует CPU DRAM и SSD как disaggregated cache, prefix-aware scheduling,
layer-wise transfer, hot-block replication и overload admission. Это особенно
важно для long-context и multi-turn workloads: recompute уже существующего
prefix может стоить дороже его поиска и перемещения.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/distributed-serving-2026/mooncake-figure1-architecture.png]]

*Mooncake, Figure 1: Conductor совместно управляет cache-aware prefill,
балансировкой KV blocks и decode; CPU/DRAM/SSD образуют распределённый cache между
GPU pools. Источник: Qin et al., [Mooncake](https://arxiv.org/abs/2407.00079),
оригинальный рисунок из открытого technical report.*

Эти работы не следует подавать как три взаимоисключающих продукта. Они выделяют
разные уровни одной системы: phase specialization, SLO-aware placement и
KV-centric state management.

## Как идеи реализованы в современных стеках

vLLM и SGLang остаются inference engines: они исполняют модель, управляют KV и
scheduler внутри worker. Оба имеют механизмы PD disaggregation и KV connectors,
но точные flags и backends быстро меняются. Их документацию следует использовать
для воспроизводимой лабораторной работы конкретной версии, а не превращать flags
в фундаментальные определения.

Dynamo расположен уровнем выше и координирует frontend, KV-aware router, P/D
workers, transfer и planner. В описанном протоколе prefill возвращает metadata о
состоянии, router добавляет её в decode request, а NIXL выбирает подходящий
transport, включая NVLink или RDMA/UCX. Это удобный production case study:
engine-specific metadata различается, но request lifecycle остаётся тем же.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/distributed-serving-2026/dynamo-architecture.png]]

*Официальная архитектура NVIDIA Dynamo: request path, KV-aware routing,
disaggregated workers и memory/transfer layer показаны как части одного runtime.
Источник: [ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo/blob/main/docs/assets/img/architecture.png),
Apache-2.0.*

При разборе реализации полезно искать ответы на семь вопросов:

1. кто резервирует decode KV memory и когда;
2. кто владеет source KV до acknowledgement;
3. как выбирается P:D pair и учитывается prefix locality;
4. каким transport и между какими ranks идут blocks;
5. поддерживаются ли разные TP/PP layouts;
6. как request восстанавливается при отказе P, D или transfer;
7. какие метрики управляют admission и autoscaling.

Название «disaggregated» без этих ответов описывает только deployment diagram.

## Когда разделение выгодно

PD-disaggregation особенно привлекателен при устойчивой высокой нагрузке,
длинных или неоднородных prompts, строгом ITL, крупных decode batches и
возможности быстро передавать KV. Он позволяет изолировать long-prefill spikes,
отдельно подбирать hardware/layout и использовать свободную prefill memory для
prefix cache.

Unified serving часто лучше для небольшой установки, низкой request rate,
коротких prompts, малых моделей или медленной межузловой сети. Он хранит одну
копию weights на replica, не имеет KV handoff и проще восстанавливается после
отказа. Chunked prefill может дать достаточную latency isolation без отдельного
pool.

Решение принимают по controlled experiment. Для одинаковых model revision,
quality settings и workload trace сравнивают:

- p50/p95/p99 TTFT и ITL;
- SLO attainment и goodput на **все** выделенные GPU;
- HBM/KV utilization обоих pools;
- network bytes и transfer percentiles;
- prefix-cache hit и recompute rate;
- поведение при burst, long prompts и worker failure;
- cold-start и resize time.

Если disaggregation улучшает среднее, но ухудшает tail из-за transfer queue, задача
не решена. Если он повышает raw throughput, но требует вдвое больше GPU и снижает
per-GPU goodput, это также не выигрыш. Ценность разделения — предсказуемое
соблюдение пользовательских SLO при экономически приемлемой capacity.

## Источники и дальнейшее чтение

- Austin et al., [All About Transformer Inference](https://jax-ml.github.io/scaling-book/inference/) — связное объяснение prefill, generation, continuous batching и disaggregated serving.
- Zhong et al., [DistServe](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf), OSDI 2024 — interference, goodput, placement и runtime KV transfer.
- Patel et al., [Splitwise](https://arxiv.org/abs/2311.18677), ISCA 2024 — phase-specific hardware, provisioning, power и cost.
- Qin et al., [Mooncake](https://arxiv.org/abs/2407.00079) — KV-centric scheduling, distributed cache, layer-wise transfer и overload handling.
- NVIDIA, [Dynamo Disaggregated Serving](https://docs.nvidia.com/dynamo/design-docs/disaggregated-serving) и [Architecture Flow](https://docs.nvidia.com/dynamo/latest/design-docs/architecture-flow) — современный orchestration protocol и NIXL transfer.
- SGLang, [PD Disaggregation](https://sgl-project.github.io/advanced_features/pd_disaggregation.html) — unified-scheduling interference и текущая engine implementation.

**Предыдущая глава:** [[58a Распределённый inference и disaggregated serving|Параллелизм и коллективные операции]]  
**Следующая глава:** [[58b Benchmarking, SLO и эксплуатация inference|Benchmarking, SLO и эксплуатация inference]]
