---
title: Реализовать ring all-reduce
type: practice
status: canonical
last_updated: 2026-07-24
---

# Реализовать ring all-reduce

Работа следует [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week03_data_parallel/practice.ipynb|EDLS Week 3 practice notebook]] и [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week03_data_parallel/homework/README|исходному homework]]. Теория collectives целиком доступна в [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol2/collective_communication|Harvard CS249r — Collective Communication]].

Реализуйте reduce-scatter и all-gather для $p$ процессов, разбив tensor на $p$
chunks. После каждого шага проверяйте, какой chunk находится у каждого rank:
отладочная таблица важнее сразу работающего финального результата.

Сравните собственную реализацию с `torch.distributed.all_reduce`:

- correctness для разных размеров, dtype и числа ranks;
- latency для маленьких сообщений;
- effective bandwidth для больших;
- объём переданных данных на rank;
- поведение при неравномерном размере chunks.

Затем запустите один DDP training step и найдите на timeline gradient buckets и
overlap backward с communication. Итог — код, тесты, диаграмма шагов ring и
график времени от размера сообщения.
