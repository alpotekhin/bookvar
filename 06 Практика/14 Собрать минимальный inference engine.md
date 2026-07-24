---
title: Собрать минимальный inference engine
type: practice
status: canonical
last_updated: 2026-07-24
---

# Собрать минимальный inference engine

Начните с [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week08_inference_software/seminar.ipynb|EDLS Week 8 seminar]] и выполните [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week08_inference_software/homework/homework_week8.ipynb|оригинальный homework notebook]].

Минимальный engine должен отдельно представлять request state, prefill, decode,
KV-cache и scheduler. Сначала добейтесь совпадения greedy generation с
эталонной реализацией при batch=1. Затем добавьте несколько запросов разной
длины и continuous batching.

Измеряйте TTFT, inter-token latency/TPOT, end-to-end latency, output tokens/s и
peak KV memory. На timeline должны различаться prefill и decode. Проведите sweep
concurrency и покажите точку насыщения, где throughput перестаёт расти, а
queueing и tail latency увеличиваются.

Сдайте код, correctness tests, workload description и графики. Одно число
tokens/s без распределения длин, arrival process и SLO не считается результатом.
