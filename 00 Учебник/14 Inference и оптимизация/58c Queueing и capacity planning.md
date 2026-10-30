---
title: "Queueing и capacity planning"
type: textbook-chapter
status: canonical
last_updated: 2026-10-20
last_verified: 2026-10-20
primary_sources:
  - https://ocw.mit.edu/courses/6-263j-data-communication-networks-fall-2002/resources/lectures8_9/
  - https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/model_serving/model_serving.qmd
  - https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/inference/inference.qmd
  - https://sre.google/sre-book/addressing-cascading-failures/
  - https://sre.google/sre-book/handling-overload/
  - https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/
---

# Queueing и capacity planning

Сервис получает 30 запросов в секунду, а завершает 24. За минуту накапливается
360 запросов. Если новые реплики поднимут скорость обслуживания ровно до 30,
очередь перестанет расти, но накопленные 360 запросов останутся.
Чтобы очередь уменьшалась, нужна скорость выше 30 запросов/с.

Такой расчёт требует различать входной поток, выполненную работу и состояние
системы. В LLM-сервисе добавляются переменные длины ответов, совместное
обслуживание запросов в batch и растущий KV-cache. По одному числу токенов/с
нельзя определить ни число реплик, ни допустимое ожидание.

В [[58b Benchmarking, SLO и эксплуатация inference|предыдущей главе]]
разобраны измерения TTFT, TPOT и совместного соблюдения SLO. Здесь эти измерения
используются для расчёта очереди, резерва памяти и числа независимых реплик.
Числа в учебных примерах условные; они не описывают измеренный кластер.

## Из времени запроса получается бюджет SLO

Пусть первый токен должен прийти не позднее чем через 1.5 s. В конкретном
запросе входная обработка и доставка занимают 0.1 s, prefill 0.6 s,
выбор и доставка первого токена ещё 0.1 s. На ожидание остаётся
$1.5-0.1-0.6-0.1=0.7$ s. Очередь на несколько секунд уже несовместима
с этим требованием, даже если после запуска GPU работает быстро.

При последовательном прохождении стадий время до первого токена можно записать как

$$
T_{\mathrm{TTFT}}
= T_{\mathrm{front}} + W_q + T_{\mathrm{prefill}}
+ T_{\mathrm{first}}.
$$

$W_q$ обозначает только ожидание начала обслуживания, $T_{\mathrm{first}}$
включает выбор первого токена и оставшуюся доставку после prefill.
Если у запроса $O>1$ выходных токенов, к TTFT добавляется сумма $O-1$
последующих интервалов. Первый выход уже выбран по результату prefill.
Завершение HTTP-потока и постобработка могут произойти позже последнего токена.

Это учёт времени одного запроса. Складывать p95 отдельных стадий как p95
всего запроса нельзя: медленные стадии могут встречаться в разных запросах.
Для SLO проверяют совместные записи с одинаковыми границами измерения.

В continuous batching ожидание возникает и после начала работы:
длинный prefill может задержать decode, а вытесненный запрос ждёт возобновления.
Поэтому одной входной очередью полный путь не описывается. Простые модели
ниже сначала относятся к отдельному последовательно работающему серверу;
их отличие от LLM-планировщика будет оговорено явно.

<a id="littles-law-среднее-число-запросов-в-системе"></a>

## Закон Литтла: сколько запросов находится в системе

Рассмотрим четыре запроса к одному исполнителю. Он обслуживает их в порядке
поступления, без прерывания. Для начала обслуживания нужен и пришедший запрос,
и свободный исполнитель.

| Запрос | Поступление, s | Обслуживание, s | Начало, s | Завершение, s | Ожидание, s | Всего в системе, s |
|---|---:|---:|---:|---:|---:|---:|
| A | 0 | 3 | 0 | 3 | 0 | 3 |
| B | 1 | 1 | 3 | 4 | 2 | 3 |
| C | 2 | 2 | 4 | 6 | 2 | 4 |
| D | 5 | 1 | 6 | 7 | 1 | 2 |

B приходит во время A и ждёт до третьей секунды. C ждёт завершения A и B.
D приходит на пятой секунде, когда C осталось работать одну секунду.

Сумма времени в системе равна $3+3+4+2=12$ запросо-секундам.
Если вместо отдельных запросов наблюдать число незавершённых запросов $N(t)$,
площадь под этой ступенчатой функцией тоже равна 12: каждая секунда
присутствия каждого запроса посчитана один раз. За интервал $H=7$ s
среднее число запросов равно $12/7$, частота завершений $4/7$ запросов/с,
а среднее время одного запроса равно $12/4=3$ s. Следовательно,

$$
\overline N = \lambda\,\mathbb E[T].
$$

В этом конечном примере система пуста до первого поступления и после последнего
завершения, поэтому площади совпадают точно. В работающем непрерывно сервисе
используют согласованные долгосрочные средние, с учётом запросов на границах
окна. В устойчивом режиме частоты поступлений и выходов совпадают. При отказах,
отменах и таймаутах нужно определить, какое событие считается выходом:
это не обязательно успешное завершение ответа.

```python
import math
import numpy as np

def fcfs_schedule(arrivals, service):
    arrivals = np.asarray(arrivals, dtype=float)
    service = np.asarray(service, dtype=float)
    assert arrivals.ndim == service.ndim == 1
    assert len(arrivals) == len(service) > 0
    assert np.all(np.isfinite(arrivals)) and np.all(np.isfinite(service))
    assert np.all(arrivals >= 0) and np.all(service > 0)
    assert np.all(np.diff(arrivals) >= 0)
    starts, finishes = [], []
    free = 0.
    for arrival, duration in zip(arrivals, service):
        start = max(arrival, free)
        free = start + duration
        starts.append(start)
        finishes.append(free)
    return np.asarray(starts), np.asarray(finishes)

arrivals = np.array([0., 1., 2., 5.])
starts, finishes = fcfs_schedule(arrivals, [3., 1., 2., 1.])
waits = starts - arrivals
times = finishes - arrivals
assert np.allclose(starts, [0., 3., 4., 6.])
assert np.allclose(waits, [0., 2., 2., 1.])
assert np.allclose(times, [3., 3., 4., 2.])
horizon = finishes[-1]
mean_population = times.sum() / horizon
assert np.isclose(mean_population, (len(times) / horizon) * times.mean())
assert np.isclose(waits.sum() / horizon, 5 / 7)
```

Тот же подсчёт применим отдельно к очереди:
$\overline N_q=\lambda\mathbb E[W_q]$. В примере среднее число ожидающих
равно $5/7$, а среднее число обслуживаемых равно 1: исполнитель занят
все семь секунд. Их сумма даёт $12/7$.

Закон Литтла не требует пуассоновских поступлений или экспоненциальных длительностей.
Он связывает средние величины, но сам по себе не предсказывает задержку.
При 8 запросах/с и среднем времени 2.5 s в системе находятся в среднем
20 запросов. Из этого не следует ни batch 20, ни требование держать KV
всех двадцати на GPU. Часть может ждать в CPU-очереди, часть проходить prefill,
часть уже генерировать ответ.

Такое разделение среднего числа запросов, задержки и GPU batch проводится
в [Harvard ML Systems, Little's Law](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/model_serving/model_serving.qmd#L2152).
Для проверки метрик нужны одинаковая популяция запросов, единицы и границы
системы. Произведение RPS на p99 latency не является законом Литтла.

<a id="почему-возникает-utilization-knee"></a>

## Почему разброс времени обслуживания увеличивает ожидание

У вновь пришедшего запроса могут быть два препятствия: незавершённая работа
текущего запроса и запросы, которые уже стоят перед ним. На рисунке из лекции
MIT $R_i$ обозначает остаток текущего обслуживания, $N_i=3$ число ожидающих,
а $X_j$ длительности их обслуживания. Сам запрос $i$ ещё не начал работу.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-queueing-review-2026-10/mit-mg1-waiting-timeline.png]]

*Eytan Modiano, MIT 6.263J, [M/G/1 Queues, слайд 5](https://ocw.mit.edu/courses/6-263j-data-communication-networks-fall-2002/65d9ab519ec4851af812f4b89ffaeedc_Lectures8_9.pdf#page=5),
CC BY-NC-SA 4.0. Фрагмент исходной временной схемы; обозначения и геометрия сохранены.
В слайдах $W_i$ — ожидание, которому здесь соответствует $W_{q,i}$.*

Ожидание равно остатку текущей работы плюс сумме работ перед запросом.
Длинный запрос задерживает больше последующих поступлений, чем короткий.
Кроме того, случайное поступление чаще попадает внутрь длинного обслуживания:
оно занимает большую долю времени.

Для количественного расчёта в лекции используется очередь M/G/1.
Первая буква M означает пуассоновские поступления с частотой $\lambda$.
G допускает общее распределение времени обслуживания $S$.
Один исполнитель обслуживает запросы по порядку, без прерывания;
длительности независимы и одинаково распределены, не зависят от поступлений.
Пусть $s=\mathbb E[S]$, второй момент конечен, а $\rho=\lambda s<1$.
$\rho$ здесь доля времени занятости этого исполнителя, не показание GPU
utilization из системного монитора.

На следующем рисунке $R(t)$ убывает от полной длительности запроса до нуля.
Для обслуживания длиной $S$ под графиком получается треугольник площади
$S^2/2$. Поэтому средний остаток, посчитанный по всему времени, включая
простои, равен $\lambda\mathbb E[S^2]/2$.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-queueing-review-2026-10/mit-mg1-residual-time.png]]

*Modiano, [M/G/1 Queues, слайд 6](https://ocw.mit.edu/courses/6-263j-data-communication-networks-fall-2002/65d9ab519ec4851af812f4b89ffaeedc_Lectures8_9.pdf#page=6),
CC BY-NC-SA 4.0. Исходный график остаточного времени. Площадь длинного
треугольника растёт квадратично с длительностью обслуживания.*

Пуассоновские поступления видят те же средние состояния, что и наблюдение
в случайный момент времени. Это свойство называется PASTA.
Среднее число ожидающих по закону Литтла равно $\lambda\mathbb E[W_q]$.
Каждому из них нужно в среднем $s$ секунд обслуживания. Значит,

$$
\mathbb E[W_q]
= \frac{\lambda\mathbb E[S^2]}{2}
+ \lambda\mathbb E[W_q]s.
$$

Перенос второго слагаемого влево даёт формулу Поллачека — Хинчина:

$$
\begin{aligned}
\mathbb E[W_q]
&= \frac{\lambda\mathbb E[S^2]}{2(1-\rho)} \\
&= \frac{\rho}{1-\rho}\,\frac{1+c_s^2}{2}\,s, \\
c_s^2&=\frac{\operatorname{Var}(S)}{s^2}.
\end{aligned}
$$

Полное время в системе равно $\mathbb E[T]=s+\mathbb E[W_q]$.
Второй момент учитывает и среднюю длительность, и её разброс.

Пусть $\lambda=8$ запросов/с и $s=0.1$ s. Загрузка во всех случаях равна 80%.

| Длительности обслуживания | $\mathbb E[S^2]$, s² | Среднее ожидание, s | Среднее время в системе, s |
|---|---:|---:|---:|
| Всегда 0.1 s, M/D/1 | 0.01 | 0.2 | 0.3 |
| Экспоненциальные со средним 0.1 s, M/M/1 | 0.02 | 0.4 | 0.5 |
| 90% по 0.01 s, 10% по 0.91 s | 0.0829 | 1.658 | 1.758 |

В последней строке среднее всё ещё равно
$0.9\cdot0.01+0.1\cdot0.91=0.1$ s. Но 10% длинных запросов занимают
91% времени обслуживания. Их вклад во второй момент
$0.1\cdot0.91^2$ намного больше вклада коротких.
Одинаковая средняя скорость обслуживания не обеспечивает одинаковую очередь.

```python
def mg1_means(rate, mean_s, second_s2):
    assert rate >= 0 and mean_s > 0
    assert second_s2 >= mean_s ** 2 - 1e-14
    rho = rate * mean_s
    assert rho < 1
    wait = rate * second_s2 / (2 * (1 - rho))
    return rho, wait, mean_s + wait

assert np.allclose(mg1_means(8., .1, .01), [.8, .2, .3])
assert np.allclose(mg1_means(8., .1, .02), [.8, .4, .5])
mix_mean = .9 * .01 + .1 * .91
mix_second = .9 * .01 ** 2 + .1 * .91 ** 2
assert np.isclose(mix_second, .0829)
assert np.allclose(mg1_means(8., mix_mean, mix_second), [.8, 1.658, 1.758])
```

Для M/M/1 $\mathbb E[S^2]=2s^2$. При $\mu=1/s$ получается
$\mathbb E[T]=1/(\mu-\lambda)$. Если $\mu=10$ запросов/с, рост поступлений
с 5 до 9 запросов/с увеличивает среднее время с 0.2 до 1 s.
При 9.5 запросах/с оно уже равно 2 s. В этой модели нет универсальной
границы безопасности 70% или 85%: при $\rho<1$ среднее конечно, но может
давно нарушать требование к задержке.

M/M/1 даёт вдвое большее **ожидание в очереди**, чем M/D/1 при тех же
$\lambda$ и $s$. Полное время в системе отличается не вдвое, поскольку
обе модели добавляют одно и то же обслуживание. Для смеси из третьей строки
M/M/1, наоборот, недооценивает ожидание. Поэтому считать её всегда
консервативной моделью LLM-нагрузки нельзя.

Из формулы среднего не следуют p95 и p99. Для хвоста нужны распределение
при явно выбранной модели либо измеренная трасса. Коррелированные всплески,
зависимые агентные вызовы и обслуживание общей GPU-пачкой также не удовлетворяют
всем предположениям M/G/1. Применять к ним полученные секунды как точный
прогноз нельзя; механизм зависимости от разброса остаётся полезным ориентиром.

<!-- serving-operations-completion: fanout start -->
## Fan-out, stragglers and hedged requests

A retrieval service asks 100 independent shards for candidates and waits
for every shard before merging them. Each shard finishes within one second
with probability 0.99. The complete request finishes within that boundary
only when all 100 shard requests do. Under independence, its probability
is $0.99^{100}\approx0.366$. About 63.4% of complete requests therefore
exceed one second, even though only 1% of individual shard requests do.

This is the fan-out example from Dean and Barroso's
[The Tail at Scale](https://research.google/pubs/the-tail-at-scale/),
discussed in Harvard's *Model Serving*. It concerns a request waiting for
all its branches. It does not describe a load balancer choosing one replica,
nor does it treat TP ranks cooperating on one forward as independent requests.

For branch completion times $T_1,\ldots,T_n$, waiting for all gives
$T_{\mathrm{all}}=\max_i T_i$. If their common cumulative distribution is
$F(t)$ and the times are independent,

$$
\Pr(T_{\mathrm{all}}>t)=1-F(t)^n.
$$

| Branches needed | Chance that one branch exceeds 1 s | Chance that the request exceeds 1 s, under independence |
|---:|---:|---:|
| 1 | 1% | 1% |
| 10 | 1% | ≈9.56% |
| 100 | 1% | ≈63.40% |
| 2000 | 0.01% | ≈18.13% |

The last row is the paper's second scale example. The calculation says
nothing about why a branch is slow. Shared congestion, synchronized pauses
or a common storage dependency can make the times dependent. For perfectly
correlated branches with the same 1% slow-event probability, the chance of
at least one slow branch is 1%, not 63.4%. Marginal branch percentiles alone
do not determine the joint distribution.

```python
import math

def independent_fanout_slow(branch_slow_probability, branches):
    p = branch_slow_probability
    if not 0 <= p <= 1 or type(branches) is not int or branches < 1:
        raise ValueError('a probability and a positive branch count are required')
    return 1.0 if p == 1 else -math.expm1(branches * math.log1p(-p))

assert math.isclose(independent_fanout_slow(.01, 100), .6339676587267705)
assert math.isclose(independent_fanout_slow(.0001, 2000), .18127743473446827)
assert independent_fanout_slow(0, 100) == 0
assert independent_fanout_slow(1, 100) == 1
```

Returning a partial result changes the contract. In a sharded search,
stopping after 95 of 100 shards can miss the highest-ranked document in
one of the five missing shards. A service may explicitly permit that
trade-off, but it must measure answer quality as well as completion time.
Dropping the slowest branches is not a free acceleration of the original
all-branches operation.

### A second attempt while the first is still running

A retry usually follows a failed attempt. A hedge starts another attempt
before the first has finished, then uses the first acceptable response.
Harvard states the cancellation constraint in the original language:

<!-- source-excerpt: harvard-hedge-cancellation -->
> Ordinary launched inference kernels are not cheaply interrupted
> mid-execution. When a hedged request completes, the duplicate must be
> cancelled, but if inference has already begun on the GPU, cancellation
> approaches include checking a cancellation flag before launching
> inference, accepting wasted compute for the in-flight kernel, or using
> request prioritization to deprioritize the duplicate. Since hedging
> typically applies only to a small tail of requests, the overhead from
> occasional wasted compute can remain acceptable when the policy is tuned
> carefully.
<!-- /source-excerpt -->

*Harvard ML Systems, [Tail-tolerant techniques](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/model_serving/model_serving.qmd#sec-model-serving-tailtolerant-techniques-066e),
CC BY-NC-SA 4.0. Complete original paragraph; only line wrapping changes.*

Suppose the primary request would finish at 800 ms. At 100 ms, the client
sends a hedge to another replica, where it takes 80 ms. The user receives
the result at 180 ms. If the primary would instead have finished at 150 ms,
it wins; the hedge has already consumed 50 ms before cancellation reaches
it. If the primary finishes at 60 ms, no hedge is sent.

| Primary completion | Hedge starts | Hedge completion if not cancelled | User completion | Additional request sent |
|---:|---:|---:|---:|---|
| 800 ms | 100 ms | 180 ms | 180 ms | yes |
| 150 ms | 100 ms | 180 ms | 150 ms | yes |
| 60 ms | not started | — | 60 ms | no |

For a hedge delay $d$, primary duration $X$ and independent backup duration
$Y$, the completion time in this simplified model is
$\min(X,d+Y)$ when $X>d$, and $X$ otherwise. For $t\geq d$,

$$
\Pr(T_{\mathrm{hedged}}>t)
=\Pr(X>t)\Pr(Y>t-d).
$$

Independence and unchanged duration distributions are assumptions here.
The second replica may share the same bottleneck; extra requests can also
change both replicas' queues. A simulation using independent fixed service
times cannot establish a production latency benefit under the added load.

```python
def hedge_completion(primary_ms, backup_ms, delay_ms):
    values = [primary_ms, backup_ms, delay_ms]
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('finite nonnegative durations are required')
    sent = primary_ms > delay_ms  # equality resolves to the primary here
    return (min(primary_ms, delay_ms + backup_ms) if sent else primary_ms, sent)

assert hedge_completion(800, 80, 100) == (180, True)
assert hedge_completion(150, 80, 100) == (150, True)
assert hedge_completion(60, 80, 100) == (60, False)
assert hedge_completion(100, 80, 100) == (100, False)
```

Sending a hedge after the baseline p95 can produce roughly 5% more
attempts only if the observed distribution and request class remain
appropriate. It does not imply 5% more GPU-seconds. The late requests may
be longer than average, the losing attempt may continue running, and the
added work may increase queueing for unrelated requests. Keep both attempt
counts and consumed resources; bound speculative duplicates separately
from the ordinary retry budget below.

For streaming LLM output, select a winner before exposing incompatible
output streams. Once the user has received a prefix, switching to a
different sampled continuation can corrupt the response. A hedge that
finishes prefill first can still have worse decode latency. Tool execution
and other side effects require an idempotency or coordination policy;
discarding a duplicate answer does not undo an action already performed.

Tied requests coordinate earlier: copies are queued together, and the
server that starts work tells the others to remove their queued copies.
Cancellation races can still allow more than one execution. Neither
hedging nor tied requests guarantees an immediately free replica. Under
fleet-wide overload, additional copies compete for the capacity needed
to recover; admission and load shedding remain necessary.

*The Harvard-derived discussion and Bookvar calculations use CC BY-NC-SA 4.0.
The tables are worked traces, not latency measurements or copied paper figures.*

<!-- serving-operations-completion: fanout end -->

<a id="static-и-continuous-batching"></a>

## Ожидание формирования batch и ожидание свободного GPU

При формировании пакета есть задержка даже у свободного исполнителя.
Пусть запросы приходят каждые 0.1 s, а запуск начинается только после
накопления четырёх. Поступления происходят в 0, 0.1, 0.2 и 0.3 s.
К моменту запуска они прождали 0.3, 0.2, 0.1 и 0 s.
Первый ждёт 0.3 s, среднее ожидание равно 0.15 s.

Если пуассоновский поток разбивается на полные группы из $B$ запросов
без таймера, среднее ожидание внутри группы равно $(B-1)/(2\lambda)$,
а для первого запроса $(B-1)/\lambda$.
Это разные величины. При 500 запросах/с и $B=32$ получаются 31 и 62 ms.
К ним добавляется ожидание занятого исполнителя и время самого вычисления.
Формула не описывает распределение задержки и не учитывает неполный batch,
который запускается по таймеру.

```python
def batch_formation(arrivals, batch_size):
    arrivals = np.asarray(arrivals, dtype=float)
    assert batch_size >= 1 and len(arrivals) > 0
    assert len(arrivals) % batch_size == 0
    assert np.all(np.diff(arrivals) >= 0)
    groups = arrivals.reshape(-1, batch_size)
    launches = groups[:, -1:]
    return (launches - groups).ravel()

assert np.allclose(batch_formation([0., .1, .2, .3], 4), [.3, .2, .1, 0.])
assert np.isclose(batch_formation(np.arange(32) / 500, 32).mean(), .031)
assert np.isclose(batch_formation(np.arange(32) / 500, 32)[0], .062)
```

Эта модель соответствует полным группам независимых задач.
Continuous batching допускает новые запросы на границах итераций,
но время итерации зависит от числа активных последовательностей, длин контекста,
prefill-работы и kernels. Если укрупнение пакета удлинило шаг с 40 до 70 ms,
больше запросов могут продвигаться одновременно, а каждый из них получать
токены реже.

Ограничение на число последовательностей, бюджет токенов итерации
и число KV-блоков ограничивают разные ресурсы. Удвоение одного из этих
лимитов не означает удвоения ёмкости сервиса. Механизм continuous batching,
chunked prefill и prefix reuse подробно разобран в
[[55b Scheduling — continuous batching, chunked prefill и prefix caching|главе о scheduling]].
При расчёте ёмкости используются измерения всего планировщика, а не
длительность одного изолированного forward.

## Несколько реплик и распределение очереди

У двух реплик может быть достаточная суммарная мощность и неудачное
распределение работы. Пусть первая занята до десятой секунды, вторая свободна.
Два новых запроса требуют по одной секунде. Если оба закреплены за первой
репликой, они начнутся в 10 и 11 s. Если их может забрать свободная реплика
из общей очереди, начала будут в 0 и 1 s.

```python
def route_jobs(arrivals, service, initial_free, assignments=None):
    arrivals = np.asarray(arrivals, dtype=float)
    service = np.asarray(service, dtype=float)
    free = list(map(float, initial_free))
    assert len(arrivals) == len(service) > 0 and len(free) > 0
    assert np.all(arrivals >= 0) and np.all(service > 0)
    assert np.all(np.diff(arrivals) >= 0) and min(free) >= 0
    if assignments is not None:
        assert len(assignments) == len(arrivals)
    starts, replicas = [], []
    for i, (arrival, duration) in enumerate(zip(arrivals, service)):
        r = (min(range(len(free)), key=lambda j: (free[j], j))
             if assignments is None else assignments[i])
        assert 0 <= r < len(free)
        start = max(arrival, free[r])
        starts.append(start)
        replicas.append(r)
        free[r] = start + duration
    return np.asarray(starts), np.asarray(replicas)

private, _ = route_jobs([0., 0.], [1., 1.], [10., 0.], [0, 0])
pooled, _ = route_jobs([0., 0.], [1., 1.], [10., 0.])
assert np.allclose(private, [10., 11.])
assert np.allclose(pooled, [0., 1.])
```

Этот пример предполагает известное время освобождения и не делит один запрос
между исполнителями. В реальном LLM-сервисе число открытых соединений плохо
оценивает такую работу: один длинный ответ может занимать больше времени
и памяти, чем несколько коротких. Маршрутизатору нужны сведения о pending
prefill, активном decode, KV и локальности префикса.

Закрепление за репликой может экономить повторный prefill. Тогда сравнивают
выигрыш от кэша со стоимостью ожидания: короткая очередь на холодной реплике
иногда лучше длинной на реплике с нужным префиксом. Такой выбор уже разобран
в [[58a2 Раздельное обслуживание prefill и decode|главе о раздельном serving]].
Общая очередь уменьшает перекос лишь для работы, которую действительно
можно переназначить; активный decode с резидентным KV так свободно не переносится.

Несколько GPU внутри одной TP-реплики совместно выполняют один запрос.
Они не являются несколькими независимыми серверами из модели M/M/c.
Реплика на TP2, потерявшая один из двух необходимых GPU, обычно теряется
как единица обслуживания. При подсчёте резерва учитывают размещение
и область отказа, а не только общее количество устройств.

## Число запросов и объём KV-cache

Для пула KV ёмкостью 16 GiB и расхода 128 KiB на позицию помещается
131 072 позиции. Предположим, что это бюджет одного конкретного пула
после выделения весов, рабочих буферов и остальных обязательных резервов.

Двадцать активных запросов с контекстом по 4096 позиций занимают
$20\cdot4096=81\,920$ позиций, или 10 GiB.
Без учёта их дальнейшего роста свободны 49 152 позиции, или 6 GiB.
Но если каждому из этих запросов обещано ещё до 1024 позиций, из свободного
бюджета уже нужно зарезервировать 20 480. Для новых запросов остаётся 28 672.

Пусть новый запрос имеет вход 8192 и верхнюю границу ответа 1024 токена.
Консервативная заявка составляет 9216 позиций. Здесь намеренно резервируется
весь вход плюс максимальный выход, хотя последний выданный токен может
не попасть в KV до завершения. При размере блока 16 позиций заявка занимает
576 блоков. Вместо пяти новых запросов без учёта старых обещаний можно
принять только три.

```python
def kv_blocks(lengths, block_tokens=16):
    assert block_tokens > 0
    assert all(length >= 0 and int(length) == length for length in lengths)
    return sum((int(length) + block_tokens - 1) // block_tokens for length in lengths)

def new_request_slots(pool_bytes, bytes_per_position, current, remaining,
                      new_prompt, new_output_max, block_tokens=16):
    assert pool_bytes >= 0 and bytes_per_position > 0
    assert len(current) == len(remaining) and len(current) > 0
    assert all(r >= 0 for r in remaining)
    total_blocks = pool_bytes // (bytes_per_position * block_tokens)
    promised_blocks = kv_blocks([c + r for c, r in zip(current, remaining)], block_tokens)
    request_blocks = kv_blocks([new_prompt + new_output_max], block_tokens)
    assert request_blocks > 0
    free_blocks = max(0, total_blocks - promised_blocks)
    return free_blocks // request_blocks

pool_bytes = 16 * 2 ** 30
bytes_per_position = 128 * 2 ** 10
current = [4096] * 20
assert sum(current) * bytes_per_position == 10 * 2 ** 30
assert new_request_slots(pool_bytes, bytes_per_position, current, [0] * 20, 8192, 1024) == 5
assert new_request_slots(pool_bytes, bytes_per_position, current, [1024] * 20, 8192, 1024) == 3
```

Резервирование не означает, что все будущие блоки уже физически выделены.
Оно запрещает пообещать одну и ту же память нескольким будущим продолжениям.
При учёте верхних границ система сохраняет запас ценой меньшего допуска;
при прогнозе выходных длин допускает больше запросов, но должна иметь
политику для ошибок прогноза: ожидание, preemption, offload либо отказ.

В примере нет общего физического префикса между запросами, поэтому блоки
можно суммировать. При prefix sharing учитывают реальные уникальные блоки
и ссылки на них. Кэш завершённых запросов тоже занимает память, пока
его не вытеснили. На TP/PP-топологии проверяют бюджет каждого соответствующего
GPU или stage; суммарные свободные GiB не исправляют переполнение одного rank.

Закон Литтла не заменяет этот подсчёт. Даже для средних во времени число
резидентных запросов и их длины могут быть зависимы. Усреднять нужно
фактическую сумму резидентных позиций, а не произведение двух несогласованных
средних. Формат KV, MQA/GQA, блоки и физическое разделение разобраны в
[[55 KV-cache, пакетирование и PagedAttention|главе о KV-cache]].

<a id="token-weighted-demand"></a>

## Состав нагрузки и несколько ресурсных ограничений

Пусть 70% запросов имеют 512 входных и 512 выходных токенов,
а остальные 30% — 8192 входных и 128 выходных. При 10 запросах/с
средние потоки равны

$$
\begin{aligned}
r_{\mathrm{input}}&=10(0.7\cdot512+0.3\cdot8192) \\
&=28\,160\quad\text{токенов/с},
\end{aligned}
$$

$$
\begin{aligned}
r_{\mathrm{output}}&=10(0.7\cdot512+0.3\cdot128) \\
&=3968\quad\text{токенов/с}.
\end{aligned}
$$

В обычном autoregressive decode после prefill остаётся в среднем
395.8 шага на запрос: первый из выходных токенов уже выбран.
Получается 3958 последовательностных decode-шагов/с.
Планировщик может объединять их в GPU-итерации; число таких итераций
не равно 3958.

Если использовать только усреднённый запрос, теряется связь входа с выходом,
состоянием кэша и моментом поступления. Работа attention длинного decode
зависит от длины уже накопленного контекста. Поэтому токенные потоки нужны
для первичной оценки, но эксперимент должен сохранять полную трассу.
Контрпример с одинаковыми средними длинами приведён в предыдущей главе.

При P/D-разделении добавляется передача KV. Если на запрос передаётся
в среднем 256 MiB, 10 запросов/с создают 2.5 GiB/s полезного потока.
Если сеть способна устойчиво передавать 5 GiB/s такого трафика, её
средняя граница составит 20 запросов/с, ещё до учёта всплесков, копирований
и конкурирующего трафика.

```python
def workload_flows(rate, weights, inputs, outputs, kv_bytes):
    weights = np.asarray(weights, dtype=float)
    inputs = np.asarray(inputs, dtype=float)
    outputs = np.asarray(outputs, dtype=float)
    assert rate >= 0 and kv_bytes >= 0
    assert len(weights) == len(inputs) == len(outputs) > 0
    assert np.all(weights >= 0) and np.isclose(weights.sum(), 1.)
    assert np.all(inputs >= 0) and np.all(outputs >= 1)
    return {
        'input_tokens_s': rate * (weights @ inputs),
        'output_tokens_s': rate * (weights @ outputs),
        'decode_sequence_steps_s': rate * (weights @ (outputs - 1)),
        'kv_bytes_s': rate * kv_bytes,
    }

flows = workload_flows(10., [.7, .3], [512, 8192], [512, 128], 256 * 2 ** 20)
assert np.isclose(flows['input_tokens_s'], 28160.)
assert np.isclose(flows['output_tokens_s'], 3968.)
assert np.isclose(flows['decode_sequence_steps_s'], 3958.)
assert np.isclose(flows['kv_bytes_s'] / 2 ** 30, 2.5)
```

Изолированные тесты могут дать, например, границы 12 запросов/с по prefill,
15 по decode и 20 по сети для фиксированной смеси. Устойчивый поток должен
быть ниже каждой границы. Но минимум 12 ещё не является гарантированной
ёмкостью: p95/p99, локальные очереди, KV и совместная интерференция могут
ограничить поток раньше. Для общего пула P и D независимо измеренные
пределы особенно нельзя считать одновременно достижимыми.

Критика планирования только по QPS и переход к фактическим ресурсам
подробно сформулированы в [Google SRE, The Pitfalls of “Queries per Second”](https://sre.google/sre-book/handling-overload/#the-pitfalls-of-queries-per-second).
Для LLM нужно различать как минимум вычислительную работу,
занятую память и передачу состояния; ни один из этих ресурсов не выводится
только из числа HTTP-запросов.

<a id="worked-capacity-plan"></a>

## Число реплик, запас и область отказа

Пусть на фиксированной трассе одна реплика устойчиво выполняет 6 запросов/с
с нужной долей совместного выполнения TTFT и TPOT. Точка 7 запросов/с
нарушает требование к TTFT. Число 6 здесь последняя проверенная подходящая
точка; оно не доказывает, что между 6 и 7 нет лучшей.

Для пика 22 запросов/с без дополнительного запаса нужно
$\lceil22/6\rceil=4$ реплики. Их суммарная проверенная ёмкость равна
24 запросам/с. После потери одной реплики остаётся 18, чего недостаточно.
Пять реплик переживают такой отказ с оставшейся ёмкостью 24.

Но после отказа нагрузка 22 использует $22/24\approx91.7\%$ этого предела.
Если требуется работать не выше 80% проверенной ёмкости даже после отказа,
понадобятся шесть реплик:

$$
R \ge f+\left\lceil\frac{\lambda_{\mathrm{peak}}}{u C_{\mathrm{SLO}}}\right\rceil.
$$

$C_{\mathrm{SLO}}$ — проверенная частота для одной реплики на данной нагрузке,
$u$ — дополнительная доля использования этого предела, $f$ число одновременно
потерянных реплик. $u=0.8$ не означает 80% показаний GPU utilization.
Это выбранный операционный запас поверх измеренного ограничения по SLO.

```python
def replica_count(peak_rate, slo_capacity, use_fraction=1., lost_replicas=0):
    assert peak_rate > 0 and slo_capacity > 0
    assert 0 < use_fraction <= 1
    assert lost_replicas >= 0 and int(lost_replicas) == lost_replicas
    return int(lost_replicas) + math.ceil(peak_rate / (use_fraction * slo_capacity))

assert replica_count(22., 6.) == 4
assert replica_count(22., 6., lost_replicas=1) == 5
assert replica_count(22., 6., .8, lost_replicas=1) == 6
assert replica_count(22., 6., .8, lost_replicas=2) == 7
```

Формула предполагает одинаковые независимые реплики, достаточный балансировщик
и линейное сложение их проверенных ёмкостей. Она не учитывает общую сетевую
границу и ухудшение cache locality после отказа. Кластерную точку подтверждают
нагрузочным опытом, включая сам переход после потери реплик.

Если на узле находятся две реплики TP2, потеря узла убирает сразу две.
В предыдущем примере с $u=0.8$ нужны уже семь реплик, а не шесть.
Если каждая реплика пересекает несколько узлов, последствия определяются
фактическим размещением. Для резервирования задают сценарий отказа:
GPU, процесс, узел, стойка или зона доступности.

### Как отказ одного кластера перегружает другой

В примере Google SRE два frontend отправляют суммарно 1000 запросов/с
в кластер A и 200 в кластер B. После отказа B кластер A получает 1200.
Если он не выдерживает эту частоту, часть его серверов тоже может выйти
из обслуживания. Тогда оставшимся приходится ещё больше работы.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-queueing-review-2026-10/sre-normal-load.webp]]

*Mike Ulrich, Google SRE, [Addressing Cascading Failures, Figure 22-2](https://sre.google/sre-book/addressing-cascading-failures/#server-overload).
CC BY-NC-ND 4.0. Полный исходник без изменений. Стрелки показывают
распределение 1200 запросов/с между A и B.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-queueing-review-2026-10/sre-failover-load.webp]]

*Ulrich, [Figure 22-3](https://sre.google/sre-book/addressing-cascading-failures/#server-overload),
CC BY-NC-ND 4.0. Исходник без изменений. После потери B весь поток
направлен в A; перенаправление не создаёт дополнительной мощности.*

Для LLM к такому росту потока может добавиться потеря KV и повторная обработка
длинных входов. Проверка отказа поэтому начинается не с утверждения
«остальных GPU хватает», а с трассы перенаправленных запросов, состояния кэша
и оставшихся лимитов. Сначала измеряют, сколько полезной работы сохраняется
в переходе, затем сколько остаётся после стабилизации.

### Цена запроса, прошедшего SLO

При условной цене 3.20 USD за GPU-час и двух GPU на реплику пять реплик стоят
$5\cdot2\cdot3.20=32$ USD/ч только по GPU.
При 22 запросах/с за час приходят 79 200 запросов.
Если 97% успешно выполнили оба ограничения, полезный знаменатель равен
76 824 запросам. Стоимость на такой запрос составляет
$32/76\,824\approx0.000417$ USD, а не $32/79\,200$.

Шесть реплик с дополнительным запасом стоят 38.40 USD/ч при той же условной
ставке. Резервные и простаивающие GPU тоже входят в счёт. К нему добавляются
остальные компоненты сервиса. Названия «тёплая» и «горячая» реплика сами
по себе не означают скидку на уже выделенное устройство.

## Всплеск нагрузки и время запуска реплик

Вернёмся к потоку из начала главы. Четыре реплики обеспечивают суммарно
24 запроса/с. Поток вырастает с 18 до 30 запросов/с.
Обнаружение, решение, получение GPU, загрузка весов и подготовка runtime
занимают вместе 60 s. Всё это время дополнительная мощность ещё недоступна.

Для оценки накопления работы используем непрерывную модель: запросы имеют
одинаковую стоимость, а поступления и обработка идут с постоянными скоростями.
В ней размер очереди $B$ за интервал $\Delta t$ меняется как

$$
B_{\mathrm{next}}
= \max\!\left(0,\ B+(\lambda-C)\Delta t\right).
$$

От 0 до 60 s очередь вырастет до $(30-24)\cdot60=360$ запросов.
Если затем добавлены две готовые реплики, ёмкость станет 36 запросов/с.
При продолжающемся входе 30 запросов/с для старой очереди остаётся
6 запросов/с; её опустошение займёт ещё 60 s.

| Момент | Доступная скорость | Вход | Очередь | Оценка B/C при неизменной текущей скорости |
|---|---:|---:|---:|---:|
| До всплеска | 24/с | 18/с | 0 | 0 |
| Перед запуском новых реплик, 60 s | 24/с | 30/с | 360 | 15 s |
| Сразу после запуска | 36/с | 30/с | 360 | 10 s |
| После опустошения, 120 s | 36/с | 30/с | 0 | 0 |

Оценка B/C предполагает, что текущая скорость сохранится до начала обслуживания.
При её скорой смене реальное ожидание считают по всему будущему расписанию:
значение 15 s перед добавлением реплик не является таким прогнозом.

Мощность добавлена, но ожидающие запросы не исчезли. Если ограничение
ожидания равно 0.6 s, очередь в 360 запросов уже не укладывается в него.
При скорости 24/с очередь из примерно 14 запросов соответствует этому
бюджету в такой модели. Для разнородных LLM-запросов оценивают работу
перед запросом и остаток deadline, а не переносят это число как лимит production.

```python
def fluid_step(backlog, offered_rate, capacity, seconds):
    assert min(backlog, offered_rate, capacity) >= 0 and seconds > 0
    available = backlog + offered_rate * seconds
    served = min(available, capacity * seconds)
    return available - served, served

backlog, served = fluid_step(0., 30., 24., 60.)
assert np.isclose(backlog, 360.) and np.isclose(served, 1440.)
assert np.isclose(backlog / 24., 15.)
assert np.isclose(backlog / 36., 10.)
backlog, served = fluid_step(backlog, 30., 36., 60.)
assert np.isclose(backlog, 0.) and np.isclose(served, 2160.)
assert np.isclose(fluid_step(0., 30., 36., 60.)[0], 0.)
```

Эта модель не воспроизводит continuous batching и не вычисляет p99.
Она показывает баланс накопленной работы и запас скорости для восстановления.
Новая реплика должна не только загрузить веса, но и пройти проверку готовности
по фактическому пути вычисления. Холодный prefix cache может дополнительно
изменить её первоначальную ёмкость.

Предсказуемый суточный рост позволяет подготовить реплики заранее.
Непредсказуемые всплески можно встретить уже готовым резервом или ограничением
допуска. Реактивное масштабирование устраняет устойчивый дефицит позднее,
но не возвращает уже нарушенные deadlines.

Разделы [Harvard Inference: Cold start, Reactive scaling и Warm pool management](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/inference/inference.qmd)
разделяют получение устройств, загрузку и прогрев. Конкретные секунды
из примеров источника не являются гарантией запуска для другого кластера:
задержку до готовой ёмкости измеряют на выбранных весах, storage и runtime.

<a id="tail-latency-admission-и-autoscaling"></a>

## Admission: какие запросы ещё можно принять

При допуске запроса проверяют и память, и возможность закончить нужную стадию
в срок. Большой запас GiB не исправляет слишком длинную очередь;
пустая очередь не исправляет недостаток блоков для продолжения.

Пусть TTFT deadline равен 1.5 s, уже прошло 0.2 s.
Оценка ожидания составляет 0.5 s, prefill 0.6 s, выбор и доставка
первого токена 0.1 s. Итого $0.2+0.5+0.6+0.1=1.4$ s:
по этим оценкам запрос ещё помещается в бюджет.
При ожидании 0.8 s оценка вырастет до 1.7 s.

```python
def fits_first_token_deadline(deadline, elapsed, wait, prefill, first):
    assert deadline > 0 and min(elapsed, wait, prefill, first) >= 0
    return elapsed + wait + prefill + first <= deadline

assert fits_first_token_deadline(1.5, .2, .5, .6, .1)
assert not fits_first_token_deadline(1.5, .2, .8, .6, .1)
```

Это оценка, а не доказательство будущей задержки. Для допуска нужны проверка
ошибок прогноза, запас и наблюдение за фактическим результатом.
Если запрос уже просрочен, запуск дорогого prefill обычно лишь забирает
ресурс у запросов, которые ещё могут завершиться вовремя.
Дедлайн передают по стадиям вместе с идентификатором запроса.
После отмены нужно прекратить вычисления и освободить KV безопасным
для незавершённых операций способом; закрытие соединения само этого не доказывает.

Ограниченная очередь позволяет вернуть явный отказ вместо неограниченного
ожидания. Отказы учитываются в пользовательском SLI, а не исчезают
из знаменателя. Для фоновых задач может быть допустима отдельная длинная очередь;
для интерактивного ответа тот же порядок непригоден. Разделение очередей
и квот защищает классы нагрузки, но уменьшение общей мощности каждой группы
тоже проверяют: изоляция имеет стоимость.

В [Google SRE, Queue Management и Latency and Deadlines](https://sre.google/sre-book/addressing-cascading-failures/#queue-management)
разобрана связь очереди с просроченной работой. Число запросов или потоков
из CPU-примера не переносится на GPU как готовая настройка.

### Метрика масштабирования должна иметь определённый знаменатель

Для очереди из 1200 ожидающих входных токенов и цели 300 токенов на готовую
реплику первичная рекомендация равна $\lceil1200/300\rceil=4$ реплики.
Если сейчас их три, средняя очередь равна 400 токенам на реплику.
Та же рекомендация получается из $3\cdot400/300=4$.
Повторное умножение общей очереди 1200 на число реплик дало бы лишний множитель.

Базовое отношение текущей и целевой метрики описано в
[официальной документации Kubernetes HPA, Algorithm details](https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/#algorithm-details).
В реальном контроллере итог меняют готовность, пропуски метрик, допуски
и ограничения скорости изменения числа реплик. Формула отношения метрик
сама по себе не предсказывает время запуска GPU или будущую задержку.

Для LLM одних ожидающих входных токенов недостаточно: активные продолжения,
KV, длины и cache reuse меняют доступную ёмкость. Одновременно наблюдают
возраст старейших запросов, фактическое выполнение SLO и прогресс готовности.
При уменьшении числа реплик сначала прекращают новые назначения, затем дают
активным запросам завершиться либо выполняют предусмотренный перенос.
Наличие нового процесса ещё не означает, что старый decode можно уничтожить.

## Повторные запросы тоже входят в нагрузку

При независимой вероятности неудачи $p=0.2$ и не более чем трёх попытках
на исходный запрос среднее число попыток равно $1+p+p^2=1.24$.
Вероятность успеха в этой условной модели равна $1-p^3=0.992$.
Поток из 100 исходных запросов/с создаёт в среднем 124 попытки/с.
При неудаче каждой попытки он создаст все 300.

```python
def retry_load(failure_probability, max_attempts):
    assert 0 <= failure_probability <= 1 and max_attempts >= 1
    expected_attempts = sum(failure_probability ** i for i in range(max_attempts))
    success_probability = 1 - failure_probability ** max_attempts
    return expected_attempts, success_probability

assert np.allclose(retry_load(.2, 3), [1.24, .992])
assert np.allclose(retry_load(1., 3), [3., 0.])
assert np.allclose(retry_load(0., 3), [1., 1.])
```

В перегруженной системе неудачи не независимы: дополнительные попытки
увеличивают поток и могут увеличить вероятность следующего отказа.
Поэтому формулу с постоянным $p$ нельзя использовать как обещание восстановления.
Ограничивают и число попыток одного запроса, и суммарный бюджет повторов.
Случайная задержка между повторениями уменьшает синхронные волны,
но не создаёт недостающую вычислительную мощность.

Если повторения выполняются на нескольких уровнях, число обращений
перемножается. Три попытки на каждом из трёх последовательных уровней
могут породить 27 обращений к нижнему сервису. В этом числе уже включена
первая попытка: три повтора означали бы четыре попытки и максимум 64 обращения.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-queueing-review-2026-10/sre-retry-stack.webp]]

*Alejandro Forero Cuervo, Google SRE, [Handling Overload, Figure 21-2](https://sre.google/sre-book/handling-overload/#deciding-to-retry),
CC BY-NC-ND 4.0. Полная исходная схема без изменений.
Если перегружен DB Frontend, повторять обращение может Backend B;
повторы выше по цепочке способны повторить уже выполненную работу.*

Для потокового LLM-ответа после выдачи части текста повторный запуск
не является прозрачным продолжением. Нужны явные правила повторного запроса,
идемпотентности инструментов и учёта частичного результата.
В статистике отделяют исходные пользовательские операции от попыток.
Тогда частота попыток объясняет загрузку, а успех исходных операций
описывает результат для пользователя.

<a id="capacity-experiment"></a>

## Как подтвердить расчёт измерением

Число реплик сначала оценивают по измеренной ёмкости одной реплики,
составу нагрузки, памяти, сети и выбранным сценариям отказа.
Проверка должна охватывать не только нормальный режим, но и накопление
работы до появления новой мощности.

Для каждой конфигурации нужен фиксированный набор длин и зависимостей,
воспроизводимое расписание поступлений, режим кэша и критерии успеха.
В open-loop опыте повышают входную частоту, сохраняют фактические поступления,
завершения, ошибки и изменение очереди. Ступень, на которой очередь всё ещё
растёт, не считается установившейся только потому, что сервер продолжает
выдавать токены. После прекращения подачи учитывают время и исходы оставшихся
запросов, не теряя их из отчёта.

Затем повторяют выбранную рабочую точку с коротким всплеском, холодным добавлением
реплик и конкретной потерей мощности. Отдельно проверяют возвращение
к нормальной нагрузке: исчезновение всплеска не гарантирует, что backlog
уже обработан или что повторы остановились.
Политики допуска, отмены и завершения активных запросов должны исполняться,
а не только присутствовать в конфигурации.

Результат расчёта связывают с ревизией модели и runtime, топологией,
трассой и состоянием кэша. Вместе сохраняют проверенную частоту,
совместное выполнение SLO, размеры очередей, резидентный и обещанный KV,
число готовых реплик и стоимость полезного результата.
При смене модели, длины контекста или планировщика эти границы проверяют заново.

## Практика и первоисточники

- Eytan Modiano, MIT 6.263J, [Lectures 8–9: M/G/1 Queues](https://ocw.mit.edu/courses/6-263j-data-communication-networks-fall-2002/resources/lectures8_9/).
  Слайды 2–6 содержат условия, примеры M/M/1 и M/D/1 и вывод через остаточную работу.
- Mor Harchol-Balter, [Performance Modeling and Design of Computer Systems](https://www.cs.cmu.edu/~harchol/PerformanceModeling/book.html).
  Для дальнейшего чтения: главы 13–15 о серверных пулах и глава 23 об M/G/1.
- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol1/model_serving|Harvard ML Systems — полный исходный раздел Model Serving]].
  Закон Литтла, формирование batch, очередь и планирование мощности.
- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol2/inference|Harvard ML Systems — Inference at Scale]].
  Состояние LLM-запросов, KV, cold start и масштабирование.
- Mike Ulrich, Google SRE, [Addressing Cascading Failures](https://sre.google/sre-book/addressing-cascading-failures/).
  Server Overload, Queue Management, Retries, Latency and Deadlines.
- Alejandro Forero Cuervo, Google SRE, [Handling Overload](https://sre.google/sre-book/handling-overload/).
  Ресурсная стоимость, бюджеты повторов и зависимые сервисы.
- Kubernetes, [Horizontal Pod Autoscaling: Algorithm details](https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/#algorithm-details).
  Определения текущей и целевой метрики; проверено 20 октября 2026 года.
- [[02 Areas/ML & DL/06 Практика/17 Развернуть наблюдаемый ML сервис|Практика: наблюдаемый ML-сервис и нагрузочный тест]].

**Предыдущая глава:** [[58b Benchmarking, SLO и эксплуатация inference|Benchmarking, SLO и эксплуатация inference]]

К модулям: [[02 Areas/ML & DL/00 Учебник/_index#Углублённые модули|Выбрать углублённый модуль]]
