---
title: "Queueing и capacity planning"
type: textbook-chapter
status: canonical
last_updated: 2026-07-24
primary_sources:
  - https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/model_serving/model_serving.qmd
  - https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/inference/inference.qmd
---

# Queueing и capacity planning

## Материалы для перехода от kernel к service

- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol2/inference|Harvard CS249r — Inference]];
- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol1/benchmarking|Harvard CS249r — Benchmarking]];
- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol1/model_serving|Harvard CS249r — Model Serving]];
- [[02 Areas/ML & DL/06 Практика/17 Развернуть наблюдаемый ML сервис|практика наблюдаемого сервиса и load test]].

Эта глава завершает inference-маршрут: измеренный service time превращается в
capacity только после добавления arrival process, queue discipline и tail SLO.
Практика требует показать как рабочую точку, так и режим насыщения.

Пиковое число токенов в секунду не отвечает на вопрос, сколько пользователей
выдержит сервис. Запрос может провести больше времени в очереди, чем на GPU, а
при приближении к насыщению небольшая вариация длины превращается в длинный хвост
задержки. Определения TTFT, TPOT и goodput канонически заданы в
[[58b Benchmarking, SLO и эксплуатация inference|главе о benchmarking]];
здесь они становятся ограничениями модели очередей и плана мощности.

## Из времени запроса получается бюджет SLO

Для запроса с $O$ выходными токенами удобно писать

$$T_{e2e}=T_{queue}+T_{prefill}+(O-1)T_{pot}+T_{post}.$$

TTFT включает очередь, prefill и выдачу первого токена; TPOT описывает
последующие интервалы. Поэтому оптимизация prefill может улучшить TTFT и почти не
изменить длинный decode, а рост decode batch — повысить throughput ценой TPOT.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/inference/ttft-tpot-timeline.svg]]

*Harvard ML Systems, Vol. II, `inference.qmd`, схема TTFT/TPOT; оригинальный SVG
из pinned commit
[`45ecc8d`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/inference/images/svg/ttft-tpot-timeline.svg),
CC BY-NC-SA 4.0.*

## Little's Law: среднее число запросов в системе

Для устойчивого интервала закон Литтла связывает arrival rate $\lambda$,
среднее время в системе $W$ и среднее число запросов $N$:

$$N=\lambda W.$$

Если сервис получает 8 RPS, а средний end-to-end latency равен 2.5 s, внутри
находится в среднем 20 запросов. Это не означает batch 20: часть запросов ждёт,
часть проходит prefill, остальные находятся на разных decode-шагах. Но ledger
памяти должен выдерживать их KV-состояние или admission должен ограничивать
число in-flight.

Для 20 запросов, средней текущей длины 4k и 128 KiB KV на токен средний KV ledger
равен $20\cdot4096\cdot128$ KiB = 10 GiB. P95 длины и burst делают необходимый
резерв больше. Закон Литтла также служит проверкой telemetry: если наблюдаемые
RPS и latency не согласуются с in-flight, определения окон или состояний
различаются.

## Почему возникает utilization knee

При малой нагрузке свободный worker начинает запрос сразу. По мере роста
utilization $\rho$ вариативность service time создаёт очередь: короткий запрос
оказывается за длинным, batch ждёт заполнения, prefill конкурирует с decode.
Даже простая M/M/1-модель даёт $W=1/(\mu-\lambda)$; это не точная модель LLM,
но она правильно показывает расходимость при $\lambda\rightarrow\mu$.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/inference/queuing-hockey-stick.svg]]

*Harvard ML Systems, Vol. II, `inference.qmd`, `queuing-hockey-stick.svg`:
задержка резко растёт у границы насыщения; [точный pinned locator](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/inference/images/svg/queuing-hockey-stick.svg),
CC BY-NC-SA 4.0.*

Планировать постоянную работу на 100% нельзя. Capacity выбирают до knee, где
p95/p99 ещё выполняют SLO при заданной смеси длин. Headroom покрывает burst,
отказ replica, компакцию cache и шум соседних процессов.

## Static и continuous batching

Static batching ждёт формирования пакета и удерживает завершившиеся строки до
окончания самой длинной. Он прост, но создаёт queue delay и padding waste.
Continuous batching пересобирает пакет на границе итерации: завершённый запрос
уходит, новый занимает освободившееся место. Состояния `waiting`, `running`,
`preempted` и `finished`, token budget и block table подробно разобраны в
[[55b Scheduling — continuous batching, chunked prefill и prefix caching|главе о scheduling]].

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/inference/batching-strategies.svg]]

*Harvard ML Systems, Vol. II, сравнение batching strategies; [оригинал в pinned
commit](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/inference/images/svg/batching-strategies.svg),
CC BY-NC-SA 4.0.*

Больший batch амортизирует чтение весов, но удлиняет итерацию. Поэтому capacity
curve измеряют как `offered RPS -> achieved RPS, TTFT p95, TPOT p95, error
rate`, а не выбирают максимальный batch из microbenchmark.

## Worked capacity plan

Пусть одна replica на фиксированном trace устойчиво даёт 6 RPS при TTFT p95
0.9 s и TPOT p95 45 ms. Следующая точка 7 RPS даёт TTFT p95 2.4 s и нарушает
SLO 1.5 s: безопасная capacity равна 6, а не 7 RPS. Для пика 22 RPS простой
нижний расчёт даёт $\lceil22/6\rceil=4$ replicas.

Но SLO должен пережить отказ одной replica: при четырёх replicas после отказа
остаётся 18 RPS, недостаточно. Нужно
$R=\lceil22/6\rceil+1=5$ replicas либо управляемая деградация. Если replica
занимает 2 GPU по 3.20 USD за GPU-час, базовая цена равна
$5\cdot2\cdot3.20=32$ USD/час. При 22 RPS это около
$32/(22\cdot3600)=0.000404$ USD на запрос до учёта frontend, idle periods и
storage. Сравнивать варианты следует по cost per SLO-compliant request, а не по
цене GPU.

## Token-weighted demand

RPS скрывает работу. При среднем input 2k и output 256 один запрос добавляет
2k input tokens к prefill и 256 последовательных decode-итераций. Для смеси из
70% чата `(512,512)` и 30% суммаризации `(8192,128)` при 10 RPS получаем
`28 160 input tok/s` и `3 968 output tok/s`. Prefill pool и decode pool следует
планировать по разным потокам; в unified pool trace должен сохранять их
совместную интерференцию.

При P/D disaggregation добавляется ограничение KV-transfer:
$B_{KV}=\lambda M_{KV,prompt}$. Если средний передаваемый кеш 256 MiB и
$\lambda=10$ RPS, полезный поток уже 2.5 GiB/s; p95 burst, protocol overhead и
другой traffic требуют запаса. Подробный протокол handoff дан в
[[58a2 Раздельное обслуживание prefill и decode|главе о disaggregated serving]].

## Tail latency, admission и autoscaling

Среднее не защищает пользователя. План мощности фиксирует percentiles и окно:
например, `TTFT p95 < 1.5 s`, `TPOT p99 < 80 ms`, error `<0.1%` за пять минут.
Admission controller отклоняет запрос до дорогого prefill, если deadline и
доступный KV budget несовместимы. Очереди разделяют по интерактивному и batch
traffic, чтобы длинные фоновые задачи не создавали head-of-line blocking.

Autoscaling по GPU utilization запаздывает и не видит KV capacity. Полезнее
совместно наблюдать queued input tokens, predicted decode tokens, active KV
tokens, TTFT/TPOT forecast и cache hit. Scale-up должен учитывать cold start
весов; scale-down — drain активного decode, иначе экономия создаёт потерю
состояния и повторный prefill.

## Capacity experiment

Для каждой конфигурации увеличивают open-loop offered load ступенями, пока
achieved throughput перестаёт расти или нарушается первый SLO. На каждой ступени
достаточно долго собирают p50/p95/p99, queue wait, prefill, TPOT, KV occupancy,
preemption, errors и стоимость. Затем повторяют с burst, отказом replica и
реальным распределением длин. Последняя SLO-compliant ступень, уменьшенная на
операционный headroom, и есть заявляемая capacity.

## Источники

- Harvard Edge ML Systems Book, [Model Serving](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/model_serving/model_serving.qmd), разделы о latency budget, queueing и Little's Law.
- Harvard Edge ML Systems Book, [Inference](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/inference/inference.qmd), разделы batching, queueing, KV capacity и routing.
- Yu et al., [Orca](https://www.usenix.org/system/files/osdi22-yu.pdf) — iteration-level scheduling.
- Zhong et al., [DistServe](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf) — goodput под TTFT/TPOT SLO.

**Предыдущая глава:** [[58b Benchmarking, SLO и эксплуатация inference|Benchmarking, SLO и эксплуатация inference]]

**Следующая глава:** [[02 Areas/ML & DL/00 Учебник/19 Deployment, Reliability и MLOps/01 ML workflow|Жизненный цикл ML-системы]]
