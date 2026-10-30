---
title: "Benchmarking, SLO и эксплуатация LLM inference"
type: textbook-chapter
status: canonical
last_updated: 2026-10-20
last_verified: 2026-10-20
source_unit_id:
  - lecture-10-section-4-understanding-the-inference-workload
  - lecture-10-dynamic-workload-failure-mode
primary_sources:
  - https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_10.py
  - https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/benchmarking/benchmarking.qmd
  - https://github.com/vllm-project/vllm/blob/b8e9001bf9568c801985889deff34d3b9fe6d0b0/vllm/benchmarks/serve.py
  - https://docs.nvidia.com/aiperf/v0.13.0/reference/ai-perf-metrics-reference
  - https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf
  - https://sre.google/workbook/alerting-on-slos/
---

# Benchmarking, SLO и эксплуатация LLM inference

За десять секунд теста сервер завершил пять из шести запросов. Один ответ
пришёл поздно, два других генерировались с большими паузами; только два
уложились во все заданные ограничения. В отчёте можно показать 0.5 успешного
завершения в секунду или 0.2 завершения с приемлемой задержкой в секунду.
Это разные характеристики одного запуска.

Числа в этом примере учебные. Чтобы получить такие показатели из настоящего
теста, нужны записи отдельных запросов: когда их предполагалось отправить,
когда отправили, что и когда получили, чем завершился поток. По этим записям
восстанавливаются метрики, ошибки и очереди. Итогового числа токенов в секунду
для этого недостаточно.

## Что входит в измерение

В [Harvard ML Systems, Benchmarking Granularity](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/benchmarking/benchmarking.qmd#sec-benchmarking-benchmarking-granularity-3855)
различаются отдельная операция, полная модель и сквозной путь запроса.
Измерение attention-ядра помогает сравнить реализации на одной форме тензоров.
Измерение полного forward включает остальные слои и обмены. Клиентский тест
добавляет сеть, обработку входа, очередь и доставку ответа. Ускорение на одном
уровне нужно заново проверить на следующем.

Например, последовательный путь тратит 80 ms вне GPU и 20 ms в оптимизируемом
вычислении. Ускорение этого вычисления вдвое уменьшает итог со 100 до 90 ms,
то есть даёт примерно 1.11×. Даже мгновенное вычисление оставило бы 80 ms.
Это применение закона Амдала при неизменных остальных стадиях. Под нагрузкой
очередь тоже может измениться, поэтому сквозной выигрыш измеряют отдельно.

Тестовая обвязка, или *benchmark harness*, подготавливает запросы, задаёт
расписание, обращается к системе и собирает результаты. Часть этой работы
может находиться за границей таймера. На следующем рисунке показаны роли
генератора нагрузки MLPerf.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-benchmark-review-2026-10/mlperf-loadgen.svg]]

*MLCommons, [LoadGen: Integration Example and Flow](https://github.com/mlcommons/inference/blob/3fbc329939999c13d0a7b5e67fb2092287e06047/loadgen/README.md#integration-example-and-flow),
[исходный SVG](https://github.com/mlcommons/inference/blob/3fbc329939999c13d0a7b5e67fb2092287e06047/loadgen/loadgen_integration_diagram.svg),
Apache-2.0. Схема перенесена без изменений. Подготовка данных, генерация
запросов, система под тестом и сбор результатов имеют разные обязанности.*

LoadGen не интерпретирует входные данные и сам не проверяет качество модели.
Его пользователь реализует загрузку данных и обращение к системе под тестом.
В MLPerf часть подготовки QSL допускается вне измеряемого пути. Поэтому такой
результат нельзя автоматически называть задержкой от ввода текста пользователем
до ответа. В собственном тесте отдельно указывают, включены ли токенизация,
chat template, извлечение документов, сетевой маршрут и постобработка.

## Временная шкала одного запроса

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/vllm-latency_diagram.png]]

*Aleksa Gordić, [Inside vLLM](https://www.aleksagordic.com/blog/vllm),
[исходное изображение](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2025-vllm-anatomy/latency_diagram.png).
Стрелки отмечают отправку запроса и получение токенов пользователем.
TTFT заканчивается на первом токене, ITL относится к соседним токенам.*

Пусть запрос отправлен в момент $t_0$, а выходные токены получены в моменты
$t_1,\ldots,t_O$. Тогда время до первого токена и интервалы между токенами равны

$$
\begin{aligned}
TTFT&=t_1-t_0,\\
ITL_i&=t_i-t_{i-1}\quad(i\ge2).
\end{aligned}
$$

Клиентский TTFT включает всё между выбранными метками: передачу запроса,
очередь сервера, prefill, выбор первого токена и его доставку. Первый токен
выбирается по результату prefill; дополнительный полный decode-шаг для него
не требуется. Время до последнего токена и средний интервал после первого:

$$
\begin{aligned}
E2E&=t_O-t_0,\\
TPOT&=\frac{t_O-t_1}{O-1}\quad(O>1),\\
E2E&=TTFT+(O-1)TPOT.
\end{aligned}
$$

Для запроса с выходными метками 200, 260, 320 и 380 ms получаем TTFT 200 ms,
три интервала по 60 ms и E2E 380 ms. Для меток 120, 140, 440 и 460 ms средний
TPOT равен примерно 113.3 ms, но внутри была пауза 300 ms. Последовательность
интервалов сохраняет сведения, потерянные при усреднении.

```python
import numpy as np

def request_metrics(sent_ms, token_times_ms):
    times = np.asarray(token_times_ms, dtype=float)
    assert len(times) > 0 and times[0] >= sent_ms
    assert np.all(np.diff(times) >= 0)
    intervals = np.diff(times)
    return {
        'ttft': times[0] - sent_ms,
        'e2e': times[-1] - sent_ms,
        'itl': intervals.tolist(),
        'tpot': float(intervals.mean()) if len(intervals) else None,
    }

a = request_metrics(0., [200., 260., 320., 380.])
d = request_metrics(0., [120., 140., 440., 460.])
assert a['ttft'] == 200. and a['tpot'] == 60.
assert d['itl'] == [20., 300., 20.]
assert np.isclose(d['tpot'], 340. / 3)
assert request_metrics(0., [250.])['tpot'] is None
```

Для одного выходного токена нет интервала после первого: TPOT помечается
как N/A, а E2E совпадает с TTFT. Закрытие HTTP-потока, запись результата
в хранилище и другая последующая работа могут закончиться позже. Их время
добавляют в отдельно определённую расширенную задержку.

### Событие потока и токен не всегда совпадают

В SSE-потоке бывают метаданные роли, keepalive, статистика usage, фрагменты
reasoning и полезный текст. Одно содержательное событие может включать несколько
токенов. Сетевой пакет, SSE-событие и токен модели являются разными единицами.

Предположим, шесть токенов получены тремя пачками по два в моменты 200, 300
и 500 ms. Между пачками наблюдаются паузы 100 и 200 ms. Времён выдачи отдельных
токенов внутри пачки у клиента нет. Выражение $(500-200)/(6-1)=60$ ms можно
получить, но оно не восстанавливает шесть отдельных моментов генерации.
Если нормировать только на токены после первой пачки, получится
$(500-200)/(6-2)=75$ ms. В отчёте нужно назвать принятую нормировку.

Названия метрик также различаются между инструментами. В просмотренном
[vLLM, commit b8e9001](https://github.com/vllm-project/vllm/blob/b8e9001bf9568c801985889deff34d3b9fe6d0b0/vllm/benchmarks/lib/endpoint_request_func.py)
`itl` заполняется интервалами содержательных событий соответствующего backend,
а `tpot` вычисляется по числу выходных токенов. В
[AIPerf v0.13.0](https://docs.nvidia.com/aiperf/v0.13.0/reference/ai-perf-metrics-reference#inter-token-latency-itl)
имя ITL относится к среднему, похожему на наш TPOT; распределение межсобытийных
пауз называется ICL. Для пачки первого ответа предусмотрена отдельная поправка.
Сопоставлять два столбца «ITL p99» без этих определений нельзя.

Для reasoning-модели дополнительно различают первый токен вообще и первый
токен финального ответа. Пользователь может не видеть reasoning-поток.
Непотоковый endpoint позволяет измерить полный ответ, но не восстановить
настоящий клиентский TTFT из одной конечной временной метки.

Оборвавшееся соединение может оставить частичный ответ. Его сохраняют для
диагностики, но не объявляют полным успешным результатом. Условие `success`
зависит от backend инструмента; его сверяют с ожидаемым завершением протокола
и причиной остановки генерации.

## Throughput и request rate

В [Stanford CS336, Lecture 10, throughput_and_latency](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_10.py#L338)
сравниваются задержка одного запроса и суммарный выход пакета. Если восемь
запросов за шаг по 50 ms получают по одному токену, суммарный выход равен
$8/0.05=160$ токенов/с. Если пакет из 32 выполняется за 80 ms, получится
400 токенов/с. При этом отдельный запрос получает новый токен раз в 80 ms
вместо 50 ms. Эти числа иллюстративные, не измерения Llama или конкретного GPU.

Для целого теста различают частоту завершений, число выходных токенов в секунду
и входной поток токенов. Входной throughput обычно считает логические токены
запросов, в том числе повторно использованные префиксы. Он не равен количеству
заново вычисленных позиций. Если движок обслуживает один системный префикс
из кэша, высокий логический input throughput сам по себе не показывает скорость
его повторного вычисления.

Пусть все пять успешных запросов начального примера имеют по 1024 входных
токена, а длины выходов равны 4, 4, 4, 4 и 1. За десять секунд это 0.5 успешного
завершения/с, 1.7 выходного токена/с и 512 логических входных токенов/с.
Сумма входа и выхода составит 513.7 токена/с. Ни одна из величин не равна
другой. Здесь для сопоставления с метриками завершённых запросов не учтена
возможная работа, потраченная на неуспешный запрос.

## Goodput: считать только пригодный результат

Зададим учебные ограничения: TTFT не более 300 ms, средний TPOT не более 80 ms,
а ответ должен успешно завершиться. Для однотокенного ответа проверяем TTFT,
не применяя отсутствующий TPOT.

| Запрос | TTFT, ms | Интервалы после первого, ms | TPOT, ms | Результат проверки |
|---|---:|---|---:|---|
| A | 200 | 60, 60, 60 | 60 | оба ограничения выполнены |
| B | 500 | 20, 20, 20 | 20 | поздний первый токен |
| C | 100 | 100, 100, 100 | 100 | медленная генерация |
| D | 120 | 20, 300, 20 | 113.3 | медленная генерация |
| E | 250 | нет | N/A | TTFT выполнен |
| F | нет полного результата | нет | N/A | timeout через 10 s |

Введём для запроса $r$ индикатор $g_r$: он равен единице, только если запрос
успешен и прошёл все применимые ограничения. Goodput за выбранный интервал $T$
равен

$$
G=\frac{\sum_r g_r}{T}.
$$

В примере прошли A и E: $G=2/10=0.2$ запроса/с. Успешно завершилось пять:
throughput равен $5/10=0.5$ запроса/с. Доля пригодных ответов среди успешных
составляет $2/5$, а среди всех шести отправленных запросов $2/6$.
Для продуктового SLO нельзя незаметно убрать timeout из знаменателя.

```python
records = [
    {'id': 'A', 'tokens': [200., 260., 320., 380.]},
    {'id': 'B', 'tokens': [500., 520., 540., 560.]},
    {'id': 'C', 'tokens': [100., 200., 300., 400.]},
    {'id': 'D', 'tokens': [120., 140., 440., 460.]},
    {'id': 'E', 'tokens': [250.]},
    {'id': 'F', 'tokens': None},
]

def summarize(records, duration_s, ttft_limit, tpot_limit):
    assert len(records) > 0 and duration_s > 0
    successful, good, output_tokens = 0, 0, 0
    for record in records:
        if record['tokens'] is None:
            continue
        metrics = request_metrics(0., record['tokens'])
        successful += 1
        output_tokens += len(record['tokens'])
        good += (metrics['ttft'] <= ttft_limit and
                 (metrics['tpot'] is None or metrics['tpot'] <= tpot_limit))
    return {
        'completed_rps': successful / duration_s,
        'goodput_rps': good / duration_s,
        'output_tps': output_tokens / duration_s,
        'attainment_all': good / len(records),
    }

summary = summarize(records, 10., 300., 80.)
assert summary == {
    'completed_rps': .5, 'goodput_rps': .2,
    'output_tps': 1.7, 'attainment_all': 2 / 6,
}
```

В [calculate_metrics vLLM](https://github.com/vllm-project/vllm/blob/b8e9001bf9568c801985889deff34d3b9fe6d0b0/vllm/benchmarks/serve.py#L587)
для goodput выбранные границы применяются совместно к успешным запросам.
Однотокенный ответ имеет нулевой TPOT для этой проверки, но не входит
в обычное распределение TPOT. В нашей записи он остаётся N/A; принятое правило
даёт тот же результат проверки, не превращая отсутствие измерения в нулевую
задержку.

В статье DistServe используется также другой показатель: максимальная
входная частота, при которой достигается заданная доля соблюдения SLO,
с нормировкой на выделенные GPU. Это граница нагрузочного эксперимента,
а не число прошедших запросов в секунду в произвольной точке. Поэтому
`request_goodput` конкретного запуска и максимальную допустимую частоту
DistServe нельзя без пояснения подставлять друг вместо друга.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-benchmark-review-2026-10/distserve-figure1-original.png]]

*Yinmin Zhong et al., [DistServe, Figure 1, PDF p.2](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf#page=2).
Оригинальные панели и подпись из PDF. Нагрузка авторов: 13B-модель,
A100 80GB, вход 512 и выход 64 токена. Пунктирные границы относятся к P90
TTFT 0.4 s и TPOT 0.04 s. Это историческое измерение, не результат Bookvar.*

Синяя кривая общего исполнения пересекает ограничение TPOT примерно при
1.6 запроса/с, раньше ограничения TTFT. Оранжевая и зелёная кривые относятся
к отдельным фазам; из них нельзя получить производительность полной
раздельной системы простым выбором большей частоты. Для неё дополнительно
считают P- и D-ресурсы и передачу KV, как в
[[58a2 Раздельное обслуживание prefill и decode|предыдущей главе]].

Два отдельных attainment по 90% также не гарантируют 90% совместного
выполнения. Если из ста запросов десять нарушили только TTFT, а ещё десять
только TPOT, по каждой метрике прошло 90%, но обе выполнили 80%.
Совместный индикатор вычисляется по исходным запросам.

## Почему нужны percentiles

Percentile задаёт положение в распределении задержек. Для отсортированных
$n$ наблюдений метод *nearest rank* берёт элемент с номером
$\lceil pn\rceil$, где $0<p\le1$. Другие методы интерполируют соседние
элементы. Для задержек 10, 10, 10 и 1000 ms ближайший ранговый p95 равен
1000 ms, а линейно интерполированный p95 NumPy равен 851.5 ms.

Это не две разные работы сервера. Отличается способ оценивания по четырём
наблюдениям. В просмотренном vLLM используется `np.percentile`; метод и версию
библиотеки полезно сохранять вместе с данными.

Для оценки редких нарушений можно рассмотреть отдельную биномиальную модель.
При вероятности нарушения $q$ вероятность не увидеть ни одного в $n$
независимых испытаниях равна $(1-q)^n$. Приравняв её к $\alpha=0.05$,
получим одностороннюю 95%-границу $q_{upper}=1-\alpha^{1/n}$.

```python
from math import ceil

def nearest_rank(values, p):
    assert len(values) > 0 and 0 < p <= 1
    return sorted(values)[ceil(p * len(values)) - 1]

sample = [10., 10., 10., 1000.]
assert nearest_rank(sample, .95) == 1000.
assert np.isclose(np.percentile(sample, 95, method='linear'), 851.5)

# Один короткий ответ содержит один длинный интервал; другой — 99 коротких.
pooled_intervals = [500.] + [10.] * 99
assert np.percentile(pooled_intervals, 95) == 10.
assert np.mean([500., np.mean([10.] * 99)]) == 255.

# Односторонняя 95%-граница вероятности нарушения при нуле наблюдённых нарушений.
def zero_failure_upper_bound(n, alpha=.05):
    assert n > 0 and 0 < alpha < 1
    return 1 - alpha ** (1 / n)

assert .029 < zero_failure_upper_bound(100) < .030
assert .00029 < zero_failure_upper_bound(10_000) < .00030
```

При объединении всех ITL длинный ответ получает больший вес. В примере
99 коротких интервалов по 10 ms скрывают один интервал 500 ms: общий p95
равен 10 ms. Если сначала усреднить каждый из двух запросов, а затем запросы,
среднее будет 255 ms. Это разные популяции, а не альтернативные способы
показать одно среднее.

Из ста независимых запросов без единого нарушения нельзя уверенно заключить,
что вероятность нарушения меньше 1%. Односторонняя биномиальная граница выше
даёт примерно 2.95%. Для десяти тысяч наблюдений она около 0.03%.
Этот расчёт предполагает независимые испытания с одной вероятностью нарушения.
Всплески и общие простои делают запросы зависимыми; тысячи соседних токенов
не являются тысячами независимых повторений всего эксперимента.

### Окна и объединение реплик

Для каждой квантили указывают окно, число наблюдений и класс запросов.
Суточный p99 может скрыть короткий инцидент; общий p99 по всем длинам может
скрыть систематически медленные длинные входы. Отдельно проверяют модели,
endpoint, tenant, длины и причины завершения.

Усреднять p99 двух реплик нельзя. Если одна обработала тысячу запросов,
а другая десять, их квантили имеют разный вес и не содержат полного
распределения. Пусть все тысяча задержек первой равны 10 ms, а десять
задержек второй 1000 ms. Среднее двух p99 равно 505 ms, но p99 объединённых
1010 наблюдений по nearest rank равен 10 ms. Для общей квантили объединяют
исходные наблюдения либо
счётчики согласованных histogram buckets. Гистограмма даёт ограниченную
границами бакетов точность: если все задержки от 500 до 1000 ms попали
в один бакет, точное положение p99 внутри него уже не восстановить.

Неуспешный запрос нельзя записывать как latency=0. Если таймер остановился
по timeout 5 s, известны ошибка и факт отсутствия полного ответа до дедлайна,
но неизвестно, когда сервер закончил бы работу без отмены. Квантили только
успешных ответов публикуют вместе с долей ошибок и timeout, а не как
характеристику всех пользователей.

## Open-loop и closed-loop нагрузка

В closed-loop тесте каждый из фиксированного числа клиентов отправляет
следующий запрос после предыдущего ответа, иногда с паузой пользователя.
Если $C=20$ клиентов, среднее время ответа $W=2$ s, паузы нет, то в устойчивом
режиме частота будет около $C/W=10$ запросов/с. При росте $W$ до 4 s тот же
тест создаст около 5 запросов/с. Так моделируется фиксированное число
последовательных пользователей.

В open-loop тесте есть внешнее расписание поступлений, не зависящее от ответов.
Для однородного пуассоновского потока интервалы независимы и имеют
экспоненциальное распределение со средним $1/\lambda$. Смена распределения
интервалов меняет всплески при той же средней частоте. Ни один такой закон
не гарантирует соответствия реальному трафику; для этого нужна трасса
с временными метками.

### Где появляется клиентская очередь

Пусть четыре запроса назначены на 0, 0.1, 0.2 и 0.3 s, но клиент разрешает
только один незавершённый запрос. Каждый запрос занимает одну секунду,
первый токен приходит через 0.2 s после отправки. Реальные отправки
произойдут в 0, 1, 2 и 3 s.

```python
import heapq

def dispatch_with_limit(arrivals, service_s, concurrency):
    assert concurrency >= 1 and service_s >= 0
    assert all(arrival >= 0 for arrival in arrivals)
    assert all(b >= a for a, b in zip(arrivals, arrivals[1:]))
    free = [0.] * concurrency
    sends, finishes = [], []
    for arrival in arrivals:
        sent = max(arrival, heapq.heappop(free))
        finished = sent + service_s
        heapq.heappush(free, finished)
        sends.append(sent)
        finishes.append(finished)
    return np.asarray(sends), np.asarray(finishes)

arrivals = np.array([0., .1, .2, .3])
sends, finishes = dispatch_with_limit(arrivals, 1., 1)
assert np.allclose(sends, [0., 1., 2., 3.])
first_tokens = sends + .2
assert np.allclose(first_tokens - sends, [.2] * 4)
assert np.allclose(first_tokens - arrivals, [.2, 1.1, 2., 2.9])
```

Если начать таймер только при реальной отправке, все четыре TTFT будут
по 0.2 s. Относительно назначенных поступлений они равны 0.2, 1.1, 2.0
и 2.9 s. Очередь находится в генераторе, а сервер может выглядеть свободным.
Пример моделирует именно клиентский лимит, не очередь continuous batching
внутри GPU.

В [vLLM b8e9001](https://github.com/vllm-project/vllm/blob/b8e9001bf9568c801985889deff34d3b9fe6d0b0/vllm/benchmarks/serve.py#L984)
ожидание семафора сохраняется отдельно как `client_queue_time`; времена
запроса начинаются внутри backend после допуска. При сочетании
`--request-rate` и `--max-concurrency` заданная и фактическая частоты могут
разойтись. Нужно проверить обе и отставание от расписания.

В этой ревизии `get_request` сначала генерирует gamma-интервалы, а для
постоянной частоты нормирует их суммарную длительность к `N/request_rate`.
Поэтому конечное расписание не является буквально набором независимых
экспоненциальных интервалов, даже когда `burstiness=1`. Для воспроизводимости
сохраняют фактические времена, а не только имя распределения.

Если тест вообще не создаёт запросы, которые должны были прийти во время
замедления, их ожидание исчезает из статистики. Это называют
*coordinated omission*. Для модели фиксированных пользователей closed-loop
может быть верен; подменять им независимый поток поступлений нельзя.
Состояние генератора, CPU, соединений и сети проверяют наряду с сервером.

<a id="workload-важнее-названия-gpu"></a>

## Состав запросов и состояние кэша

Одинаковых средних длин входа и ответа недостаточно. Рассмотрим два набора
из двух запросов:

| Набор | Первый запрос: вход, выход | Второй запрос: вход, выход |
|---|---|---|
| A | 128, 512 | 8192, 32 |
| B | 128, 32 | 8192, 512 |

В обоих случаях суммарный вход равен 8320, выход 544 токенам. Но в B длинный
ответ создаётся после длинного входа: каждый его decode-шаг обращается
к большему контексту.

Для одного слоя плотного causal attention посчитаем число key-позиций,
доступных обычным decode-шагам. Первый выход уже выбран на prefill.
На шаге $j=1,\ldots,O-1$ внимание охватывает $S+j$ позиций, включая текущую.
Поэтому сумма равна $(O-1)S+O(O-1)/2$.

```python
def decode_key_positions(prompt_tokens, output_tokens):
    assert prompt_tokens >= 0 and output_tokens >= 1
    return (output_tokens - 1) * prompt_tokens + output_tokens * (output_tokens - 1) // 2

workload_a = [(128, 512), (8192, 32)]
workload_b = [(128, 32), (8192, 512)]
assert sum(s for s, o in workload_a) == sum(s for s, o in workload_b) == 8320
assert sum(o for s, o in workload_a) == sum(o for s, o in workload_b) == 544
a_positions = sum(decode_key_positions(s, o) for s, o in workload_a)
b_positions = sum(decode_key_positions(s, o) for s, o in workload_b)
assert b_positions > 9 * a_positions
```

В B эта составляющая работы больше чем в девять раз. Это не предсказание
девятикратного замедления всей модели: веса, MLP, prefill, ядра и параллелизм
в расчёт не вошли. Пример показывает, какую информацию теряет перестановка
выходных длин между запросами.

Трасса должна сохранять пары длин, порядок поступлений, общие префиксы,
ветвления диалога, sampling и причины завершения. Для агента следующий вызов
может зависеть от результата инструмента и содержать предыдущую историю.
Независимо перемешанные HTTP-запросы не воспроизводят такую зависимость.

Пример такого теста описан в
[MLPerf Agentic Inference, июль 2026](https://mlcommons.org/2026/07/agentic-inference-for-mlperf-inference/).
Внутри одной траектории клиент ждёт полного ответа, прежде чем послать
следующий ход. При этом будущие входы восстанавливаются из записанной трассы,
а не из текущего ответа модели. Так сохраняют зависимый порядок запросов
и воспроизводимую историю. Это позволяет сравнить serving, но отличается
от живого агента, который выбирает следующие действия по собственным ответам.
Успешность решения задач требует отдельной проверки.

Тёплый prefix cache и пустой cache задают разные опыты. Повторение одного
и того же набора может ускорить второй запуск за счёт накопленного KV.
Для проверки холодного кэша используют поддерживаемый сброс или независимые
префиксы с теми же длинами, подтверждая отсутствие попаданий. Для тёплого
режима заранее задают прогрев и измеряют фактическую долю повторного
использования. Частоту попаданий по запросам отличают от доли повторно
использованных токенов: один очень длинный префикс способен доминировать
во второй величине.

Для синтетического теста фиксируют длины после токенизации, не число символов.
Сохраняют chat template, BOS/EOS, truncation и реальные выходные длины.
`ignore_eos` отключает остановку по EOS. Остальные ограничения длины
и остановки проверяют отдельно: один этот параметр не гарантирует ровно
заданное число выходных токенов. Такой режим полезен для изоляции формы
вычислений, но меняет освобождение KV по сравнению с естественными остановками.

## Воспроизводимый протокол

Перед запуском сохраняют конфигурацию, по которой можно восстановить систему
и нагрузку. В неё входят ревизии весов и токенизатора, шаблон диалога, точность
и квантование, движок и его commit, версии CUDA и драйвера, kernels, число
и тип GPU, TP/PP/DP/EP, топология, power limit и ограничения контекста.
Отдельно сохраняют настройки планировщика, бюджет KV, prefix caching,
chunked prefill и speculative decoding.

Для нагрузки нужны ревизия набора, seed, порядок запросов, sampling, stop rules,
частота, concurrency, timeout и политика повторов. В сырых результатах
должны различаться назначенное поступление, фактическая отправка, первый
и последующие результаты, конец потока и статус. Для сопоставления клиентских
и серверных записей нужен request ID. Разность меток разных машин не считают
односторонней сетевой задержкой без согласованного времени и оценки его ошибки.

Прогрев и измерение разделяют. Холодный старт включает получение GPU, загрузку
весов, создание групп обменов, компиляцию и подготовку графов. Рабочий прогрев
может также менять allocator и частоты устройства. Стабильная задержка прогретых
ядер не доказывает, что холодная реплика быстро примет реальную нагрузку.
Оба режима измеряют отдельно.

<a id="воспроизводимое-измерение-qwen3-4b"></a>

### Измерение отдельных фаз: пример Efficient DL Systems

В [семинаре week 8](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week08_inference_software/seminar.ipynb)
используются Qwen3-4B, Transformers 4.53.0 и FP16. Полный исходный notebook
сохранён в [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week08_inference_software/seminar.ipynb|библиотеке курса]].
Он позволяет проследить forward с кэшем и вызов `model.generate`.
Последний выполняет и обработку входа, и продолжение ответа.

Если prefill занял 120 ms, а полный `generate` для десяти выходных токенов
300 ms, то средний полный выход равен $10/0.3\approx33.3$ токена/с.
После первого токена осталось девять шагов и 180 ms: их средний интервал
20 ms, или 50 токенов/с на запрос. Деление всех десяти токенов на оставшиеся
180 ms дало бы неправильные 55.6 токена/с.

```python
def phase_accounting(prefill_s, total_s, output_tokens):
    assert 0 <= prefill_s <= total_s and output_tokens > 1
    remaining = output_tokens - 1
    decode_s = total_s - prefill_s
    assert decode_s > 0
    return {
        'full_output_tps': output_tokens / total_s,
        'decode_tpot_s': decode_s / remaining,
        'decode_output_tps': remaining / decode_s,
    }

phases = phase_accounting(.12, .30, 10)
assert np.isclose(phases['full_output_tps'], 100 / 3)
assert np.isclose(phases['decode_tpot_s'], .02)
assert np.isclose(phases['decode_output_tps'], 50.)
```

Такая разность времён имеет смысл, если граница первого токена и остальные
накладные расходы действительно согласованы. Для строгого разделения
измеряют prefill и последующие шаги одного прохода, а не вычитают среднее
prefill другого запуска с иной формой входа.

CUDA-вычисления асинхронны относительно CPU. Таймер вокруг Python-вызова
без ожидания GPU способен измерить только постановку работы. Синхронизацию
или CUDA events выбирают под нужную границу измерения. Это подробно
разобрано в [[02 Areas/ML & DL/00 Учебник/10 ML Systems/07 Profiling ML-нагрузки|главе о profiling]].

В исходном семинаре prefill-прогрев идёт с `use_cache=False`, измерение
с `use_cache=True`, а перед попытками очищается allocator cache.
Счётчик входа использует размер padded-тензора. Поэтому при адаптации
отдельно считают полезные и padded-позиции, передают корректную attention mask
и проверяют, что прогрев соответствует измеряемому пути. Число запрошенных
выходных токенов также сверяют с фактически полученными. Сохранённых
измерений GPU в notebook нет; приведённые здесь 120/300 ms являются
учебными исходными данными.

### Производительность и качество образуют один контракт

Смена точности или pruning может изменить ответы. Тогда проверяют качество
на выбранной задаче и производительность той же ревизии модели.
Для sampling фиксируют правила выбора, ограничения и остановки; для
speculative decoding проверяют условия сохранения нужного распределения,
а не требуют совпадения каждой случайной траектории.

A/B-сравнение использует одинаковые запросы и расписание. Повторения
чередуют, например A–B–B–A, чтобы постепенный прогрев или изменение внешней
нагрузки не совпали только с одним вариантом. Для оценки разброса сохраняют
несколько независимых запусков, распределения и исходные записи, не только
лучшее число. Запросы внутри одной перегруженной очереди связаны; обычный
bootstrap отдельных токенов не измеряет неопределённость всего опыта.

Методология качества и статистических сравнений разобрана в
[[02 Areas/ML & DL/00 Учебник/18 Evaluation и методология/59 Оценивание моделей и контаминация|главе об оценивании моделей]].
Здесь в сравниваемый объект входят модель, runtime, планировщик, kernels,
аппаратура и нагрузка.

<!-- serving-operations-completion: energy start -->
## Energy per result: measurement boundary and time window

A server can draw less power and still spend more energy on each answer.
Suppose configuration A consumes 600 W while completing 20 acceptable
responses per second. Configuration B consumes 400 W but completes only
10 acceptable responses per second. Their energy costs are 30 and 40 J per
acceptable response respectively. The lower-power configuration loses this
comparison because it takes longer to produce the same amount of useful work.
These are teaching inputs, not measurements of a GPU or serving engine.

Power is a rate, measured in watts: one watt is one joule per second.
Energy accumulates over time. For a measurement interval of duration $T$,

$$
\begin{gathered}
E=\int_0^T P(t)\,dt,\\
\overline P=E/T,\\
e_{\mathrm{good}}=E/N_{\mathrm{good}}.
\end{gathered}
$$

Here $N_{\mathrm{good}}$ counts results that meet the declared acceptance
conditions in the same interval. If $G=N_{\mathrm{good}}/T$ is goodput, then
$e_{\mathrm{good}}=\overline P/G$. Dividing watts by requests per second
gives joules per request. Dividing watts by the number of requests, without
the duration, does not.

### What the power meter includes

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/serving-operations-2026-10-29/power-measurement-boundaries.svg]]

*Vijay Janapa Reddi and contributors, Harvard ML Systems,
[Power Measurement Boundaries](https://github.com/harvard-edge/cs249r_book/blob/2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd/book/quarto/contents/vol1/benchmarking/benchmarking.qmd#sec-benchmarking-power-measurement-boundaries-982c),
[complete original SVG](https://mlsysbook.ai/vol1/benchmarking/benchmarking_files/mediabag/68403b70d9a611957fc377cc27f4711283a8df2f.svg),
CC BY-NC-SA 4.0. The original course diagram follows Tschand et al.,
[MLPerf Power](https://arxiv.org/abs/2410.12032v2). No cropping or redrawing.*

The diagram distinguishes a chip, an inference system-on-chip with external
DRAM, an inference node and a training installation. Its solid boundary and
dashed exclusions carry the meaning even without colour. In the node
example, accelerators, host memory, NIC, local storage and active cooling
are inside the boundary; remote storage is outside. In the training
example, compute nodes and measured or estimated network switches count,
while the storage node and data-centre cooling do not. These are the
illustrated accounting boundaries, not a claim that every benchmark uses
the same exclusions.

A GPU power counter measures a different object from an external meter
at the node's power inlet. Adding GPU telemetry to node-level AC power
would usually count the GPU twice. Conversely, reporting only device
power omits the host and the rest of that node. A comparison must name
the measurement point, included devices and any estimated components.
The [MLPerf Power paper, §III](https://arxiv.org/html/2410.12032v2#S3)
specifies these boundaries separately for the different deployment classes.

### Integrate the trace, then divide by results

Consider a 60-second interval with three constant-power phases. The same
server remains allocated throughout; 900 completed responses meet the
declared latency and quality conditions.

| Interval | Power | Duration | Energy |
|---|---:|---:|---:|
| 0–10 s | 300 W | 10 s | 3000 J |
| 10–40 s | 600 W | 30 s | 18,000 J |
| 40–60 s | 300 W | 20 s | 6000 J |
| Whole interval | 450 W average | 60 s | 27,000 J |

The energy per accepted result is $27{,}000/900=30$ J. The simple average
of the three listed power levels is 400 W, which is wrong for this trace:
the middle level lasts longer than the first. The time weights must stay
with the readings.

```python
from math import isclose, isfinite

def interval_energy(boundaries_s, interval_watts, good_results):
    if len(boundaries_s) != len(interval_watts) + 1 or not interval_watts:
        raise ValueError('one power value per time interval is required')
    if not all(isfinite(x) for x in [*boundaries_s, *interval_watts]):
        raise ValueError('times and power must be finite')
    durations = [b - a for a, b in zip(boundaries_s, boundaries_s[1:])]
    if any(dt <= 0 for dt in durations) or any(p < 0 for p in interval_watts):
        raise ValueError('positive durations and nonnegative power are required')
    if type(good_results) is not int or good_results < 0:
        raise ValueError('good_results must be a nonnegative integer')
    joules = sum(dt * p for dt, p in zip(durations, interval_watts))
    seconds = sum(durations)
    return joules, joules / seconds, (joules / good_results if good_results else None)

assert interval_energy([0, 10, 40, 60], [300, 600, 300], 900) == (27000, 450, 30)
assert interval_energy([0, 60], [400], 600) == (24000, 400, 40)
assert interval_energy([0, 60], [400], 0) == (24000, 400, None)
assert isclose(27000 / 3_600_000, .0075)  # joules to kWh
```

This integration is exact for the piecewise-constant teaching trace.
Point samples of an unknown physical signal require a declared interpolation
rule and enough sampling bandwidth; the code cannot recover a power spike
between samples. Cumulative energy counters can provide interval energy
directly, but their update cadence, reset/wrap behaviour and device boundary
still need to be checked. With several devices, align their measurement
windows before adding their energies.

If the idle baseline is 250 W, subtracting $250\cdot60=15{,}000$ J gives
12,000 J of energy above that baseline. This is an *incremental* estimate,
not the 27,000 J actually consumed within the stated boundary. It depends
on a valid baseline under comparable conditions. Report total and
baseline-subtracted energy under different names. Shared background work
also prevents attribution of the entire measured trace to one tenant
without a separate allocation rule.

### Sustained operation and the workload matter

Harvard's discussion of standardized measurement connects batching,
idle time and thermal behaviour in one paragraph:

<!-- source-excerpt: harvard-power-batch-idle-thermal -->
> Batch size creates a nonlinear relationship with power consumption that
> single-point measurements cannot characterize. Larger batches improve
> compute efficiency (better amortization of memory loads) but increase
> memory pressure and peak power requirements, meaning the most efficient
> batch size for throughput may differ from the most efficient batch size
> for energy. Measurement across multiple batch sizes is essential for a
> complete efficiency profile. System idle states deserve equal attention,
> particularly for intermittent edge workloads: a wake-word detection
> TinyML system that actively processes audio for only a small fraction
> of operating time may be dominated by idle power consumption rather than
> inference energy. Finally, sustained ML workloads can cause temperature
> increases that trigger thermal throttling and alter power consumption
> patterns—an effect particularly acute in edge devices, where thermal
> constraints limit sustained performance and make extended benchmarking
> runs essential for realistic characterization.
<!-- /source-excerpt -->

*Harvard ML Systems, [Standardized power measurement](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/benchmarking/benchmarking.qmd#sec-benchmarking-standardized-power-measurement-7fae),
CC BY-NC-SA 4.0. Complete original paragraph; only line wrapping changes.*

Those trends do not supply a universal batch-size curve. Measure the
chosen engine, shapes, cache regime and arrival process at each operating
point. Keep SLO and quality gates unchanged. Counting rejected requests
as energy-efficient responses would reward a server for discarding work;
counting only successful requests while omitting the energy spent on
failures would remove part of its bill.

For a finite workload, define whether energy accounting includes startup,
warm-up and the final drain. For a steady-state window, report which work
crosses its boundaries, as with throughput above. A short GPU kernel
benchmark cannot establish the energy per user answer for a service that
also tokenizes text, performs retrieval and waits on remote tools.

PUE applies only when converting an IT-energy boundary into a specified
facility-energy estimate. Applying it to a meter that already includes
the same cooling overhead counts that overhead again. A device TDP or
configured power limit is not a measurement of mean power. Keep the
instrument trace, timestamps, accepted-result count and boundary description
with the report so that joules per result can be recomputed.

*The Harvard-derived explanation and Bookvar teaching trace in this section
use CC BY-NC-SA 4.0. No power measurement was executed here.*

<!-- serving-operations-completion: energy end -->

## Поиск рабочей ёмкости

Нагрузку увеличивают ступенями. Для каждой сохраняют предложенную
и фактическую частоты, завершения, совместное соблюдение SLO, задержки,
ошибки и изменение очереди. Следующая условная таблица относится к одному
фиксированному составу запросов и 60-секундным ступеням.

| Предложено, запросов/с | Завершено/с | TTFT p95, ms | TPOT p95, ms | Прошли оба SLO | Очередь: начало → конец |
|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 250 | 30 | 100% | 0 → 0 |
| 4 | 4 | 500 | 40 | 99% | 0 → 0 |
| 6 | 6 | 900 | 60 | 97% | 0 → 0 |
| 8 | 6.2 | 3500 | 100 | 70% | 0 → 108 |

Пусть требуется не менее 95% совместного выполнения TTFT ≤1000 ms
и TPOT ≤70 ms. Из проверенных точек подходит 6 запросов/с. Точка 8 даёт
больше завершений, но не выполняет задержки и накапливает примерно
$(8-6.2)\cdot60=108$ запросов. Ожидать от неё стационарной очереди бессмысленно:
в таком режиме она продолжает расти.

Это пока результат сетки измерений, не доказанный глобальный максимум.
Можно уточнить границу между 6 и 8, повторить точки и проверить всплески.
Если запросы отклоняются или отменяются, в баланс очереди добавляют и эти
выходы: поток завершений один не объясняет её изменение.

Два способа учёта времени нельзя смешивать. В тесте конечного набора
отправляют заданную когорту и ждут её завершения; длительность включает
период отправки и последующее опустошение. В установившемся временном окне
считают реальные события внутри окна; часть запросов пришла раньше или
завершится позже. Если из одной схемы взять число запросов, а из другой
длительность, throughput и goodput перестанут описывать один опыт.

Максимальную рабочую частоту проверяют также при смене длин, отказе реплики
и холодном добавлении новой. Выделение реплик и операционный запас разобраны
в [[58c Queueing и capacity planning|следующей главе]].

## SLI, SLO и алерты в эксплуатации

SLI задаёт наблюдаемое свойство и способ подсчёта. Для потокового API это может
быть доля запросов, которые успешно завершились и выполнили границы TTFT
и пауз. SLO задаёт целевую долю и окно. HTTP 200, отправленный до генерации,
сам по себе не означает успешного ответа.

Для SLO 99.9% допустимая доля плохих событий равна 0.1%. При 10 000
учитываемых запросов бюджет составляет десять плохих событий. Если их оказалось
150, текущая доля равна 1.5%, то есть в 15 раз выше допустимой.
Это отношение называют burn rate.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-benchmark-review-2026-10/sre-burn-budget.webp]]

*Steven Thurgood et al., Google, [The Site Reliability Workbook, Figure 5-4](https://sre.google/workbook/alerting-on-slos/#4-alert-on-burn-rate),
CC BY-NC-ND 4.0. Исходный график перенесён без изменений. Для 30-дневного
бюджета постоянные burn rate 1, 2 и 10 исчерпывают его за 30, 15 и 3 дня
при принятой авторами модели постоянного потока.*

```python
def burn_rate(bad, eligible, slo=.999):
    assert eligible > 0 and 0 <= bad <= eligible and 0 < slo < 1
    return (bad / eligible) / (1 - slo)

burn = burn_rate(150, 10_000)
assert np.isclose(burn, 15.)
assert np.isclose(30 / burn, 2.)  # Прогноз при постоянной будущей доле и потоке.
assert np.isclose(14.4 * 1 / (30 * 24), .02)

def two_window_page(burn_1h, burn_5m, threshold=14.4):
    return burn_1h > threshold and burn_5m > threshold

assert two_window_page(20., 20.)
assert not two_window_page(20., 0.)
```

Длинное окно показывает накопленный расход, короткое проверяет, продолжается
ли ухудшение. В примере Google условие для срочного уведомления требует
превышения burn rate 14.4 одновременно за час и за пять минут. Час при
такой скорости соответствует 2% 30-дневного бюджета при постоянной
интенсивности запросов. Эти параметры являются примером политики,
не обязательными границами для каждого LLM-сервиса.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-benchmark-review-2026-10/sre-alert-windows.webp]]

*Thurgood et al., [The Site Reliability Workbook, Figure 5-6](https://sre.google/workbook/alerting-on-slos/#6-multiwindow-multi-burn-rate-alerts),
CC BY-NC-ND 4.0. Исходник без изменений. После окончания всплеска короткая
зелёная оценка снижается раньше длинной оранжевой; уведомление требует
превышения порога в обоих окнах.*

При неодинаковом трафике бюджет считают по событиям: суммируют плохие
и учитываемые запросы, а не усредняют проценты минут. Одна ошибка из десяти
ночных запросов и сто ошибок из ста тысяч дневных имеют разный вес.
При отсутствии запросов отношение не определено; отсутствие данных
не записывают как измеренные 0% ошибок.

Правила включения запросов в SLI задают заранее. Невалидный клиентский ввод
может быть исключён из выбранной популяции, но timeout принятого валидного
запроса нельзя исключить только потому, что полного ответа нет.
Отмены пользователя и повторные попытки также получают явные правила учёта.

Клиентские и серверные метки связывают request ID. Для поиска причины
сопоставляют очередь, prefill, паузы генерации, заполнение KV, preemption,
передачу KV и ошибки обменов. Загрузка GPU помогает объяснять задержку,
но не заменяет проверку пользовательского SLO.

## Деградация и перегрузка

Если поступления устойчиво превышают обслуживание, ограничение очереди
переводит часть ожидания в явные отказы. Эти отказы остаются в отчёте.
Например, вариант, который быстро обслужил десять запросов и отклонил
девяносто, не выполнил SLO для ста пользователей только потому, что latency
десяти успешных ответов мала.

Повторные попытки добавляют нагрузку. Если исходный запрос имеет до двух
повторов, при массовом отказе один пользователь может породить три попытки.
Сохраняют и идентификатор исходного запроса, и номера попыток: пользовательский
успех и серверная частота попыток относятся к разным уровням.

Уменьшение максимального выхода, переход на меньшую модель или сокращение
контекста меняют предоставленный результат. Такие режимы объявляют отдельными
вариантами с собственной проверкой качества. Отказ GPU, потерю D после
prefill, перезапуск и drain проверяют отдельными сценариями; стабильный
успешный запуск не доказывает восстановление после отказа.

<a id="минимальный-честный-отчёт"></a>

## Что сохранять вместе с результатом

Воспроизводимый результат состоит из конфигурации, исходной нагрузки,
журнала отдельных запросов и агрегатов с определениями. Для каждой цифры
должно быть понятно, какие запросы вошли в числитель, какой интервал
в знаменатель и какие ошибки исключены из распределения задержек.

В отчёте приводят границы измеряемого пути, версии и топологию, режим кэша
и прогрева, назначенную и фактическую нагрузки, число попыток и ошибок,
популяции квантилей, совместное соблюдение SLO, расход GPU-времени
и разброс повторений. Графики нагрузки и очереди позволяют увидеть, в каком
режиме получено итоговое число. Обезличенные сырые записи сохраняют так,
чтобы агрегаты можно было пересчитать без нового GPU-прогона.

## Практика и первоисточники

- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol1/benchmarking|Harvard ML Systems — полный исходный раздел Benchmarking]]. Для этой главы: Micro/Macro/End-to-end benchmarks, Benchmark harness, System specifications, Run rules и Inference metrics.
- [Stanford CS336, Lecture 10](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_10.py). Различие задержки запроса и суммарного throughput; ссылка на исходный код лекции.
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week08_inference_software/seminar.ipynb|Efficient DL Systems, week 8]]. KV-cache, forward и полный generate; исходный язык сохранён.
- [[02 Areas/ML & DL/06 Практика/17 Развернуть наблюдаемый ML сервис|Практика: наблюдаемый ML-сервис]].
- MLCommons, [LoadGen README, commit 3fbc329](https://github.com/mlcommons/inference/blob/3fbc329939999c13d0a7b5e67fb2092287e06047/loadgen/README.md). Границы генератора, QSL и системы под тестом.
- Zhong et al., [DistServe, OSDI 2024](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf). §1 и Figure 1: максимальная нагрузка при ограничениях TTFT/TPOT.
- vLLM, [calculate_metrics и benchmark, commit b8e9001](https://github.com/vllm-project/vllm/blob/b8e9001bf9568c801985889deff34d3b9fe6d0b0/vllm/benchmarks/serve.py), [обработчик потока](https://github.com/vllm-project/vllm/blob/b8e9001bf9568c801985889deff34d3b9fe6d0b0/vllm/benchmarks/lib/endpoint_request_func.py), [CLI](https://docs.vllm.ai/en/stable/cli/bench/serve/). Версионные определения метрик и расписания; snapshot 20 октября 2026 года.
- NVIDIA, [AIPerf v0.13.0: Metrics Reference](https://docs.nvidia.com/aiperf/v0.13.0/reference/ai-perf-metrics-reference). TTFT, ITL, ICL и обработка пачек.
- Google, [Implementing SLOs](https://sre.google/workbook/implementing-slos/) и [Alerting on SLOs](https://sre.google/workbook/alerting-on-slos/). Популяции, окна и расход бюджета.
- MLCommons, [Agentic Inference for MLPerf Inference](https://mlcommons.org/2026/07/agentic-inference-for-mlperf-inference/). Дополнительный материал о зависимых многошаговых нагрузках.

**Предыдущая глава:** [[58a2 Раздельное обслуживание prefill и decode|Раздельное обслуживание prefill и decode]]

**Следующая глава:** [[58c Queueing и capacity planning|Queueing и capacity planning]]
