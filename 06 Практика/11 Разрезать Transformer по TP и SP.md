---
title: Разрезать Transformer по tensor и sequence parallelism
type: practice
status: canonical
last_updated: 2026-07-24
---

# Разрезать Transformer по TP и SP

Исходная работа: [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week04_large_models/practice_part2.ipynb|EDLS Week 4, practice part 2]]. Перед кодом прочитайте [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol2/distributed_training|Harvard CS249r — Distributed Training]].

Сначала вручную разрежьте пару linear layers column-parallel/row-parallel.
Подпишите shape каждого локального tensor и точку collective. Проверьте, что
склеенный output совпадает с dense baseline. Затем проведите через тот же учёт
attention и MLP Transformer block.

Добавьте sequence parallelism для тех активаций, которые не обязаны быть
реплицированы. Для TP=1,2,4 сравните:

- локальный parameter/activation memory;
- число и bytes collectives;
- step latency и scaling efficiency;
- максимальный допустимый batch/sequence;
- numerical error относительно baseline.

Сдать нужно не только работающий код, но и таблицу tensor shapes до/после каждой
операции. Именно она показывает, что разбиение понято, а не случайно собрано из
готовых wrappers.
