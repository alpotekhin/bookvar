---
title: Собрать минимальный inference engine
type: practice
status: canonical
last_updated: 2026-09-15
---

# Собрать минимальный inference engine

Начните с [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week08_inference_software/seminar.ipynb|EDLS Week 8 seminar]] и выполните [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week08_inference_software/homework/homework_week8.ipynb|оригинальный homework notebook]].

Минимальный engine должен отдельно представлять request state, prefill, decode,
KV-cache и scheduler. Сначала добейтесь совпадения greedy generation с
эталонной реализацией при batch=1. Затем добавьте несколько запросов разной
длины и continuous batching.

Измеряйте TTFT, общую задержку запроса, число выходных токенов в секунду и
пиковую память KV. Межтокенная задержка (ITL) — каждый отдельный интервал
$t_i-t_{i-1}$ между поступлениями соседних токенов. TPOT для запроса с $N>1$
выходными токенами — средний интервал $(t_N-t_1)/(N-1)$; при $N=1$ он не
определён. Хвост распределения ITL может содержать длинные паузы, скрытые
в среднем TPOT, поэтому покажите оба распределения и единицу усреднения.
При объединении нескольких токенов в один сетевой chunk время его получения
не позволяет восстановить отдельные серверные интервалы: измеряйте их на
сервере или явно называйте клиентскую метрику межпакетной.
На timeline должны различаться prefill и decode. Проведите sweep
concurrency и покажите точку насыщения, где throughput перестаёт расти, а
queueing и tail latency увеличиваются.

Сдайте код, correctness tests, workload description и графики. Одно число
tokens/s без распределения длин, arrival process и SLO не считается результатом.
