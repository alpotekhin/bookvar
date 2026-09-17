---
title: Разрезать Transformer по tensor и sequence parallelism
type: practice
status: canonical
last_updated: 2026-09-15
---

# Разрезать Transformer по TP и SP

Исходная работа: [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week04_large_models/practice_part2.ipynb|EDLS Week 4, practice part 2]]. Перед кодом прочитайте [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol2/distributed_training|Harvard CS249r — Distributed Training]].

Сначала распределите два последовательных линейных слоя по схемам
column-parallel и row-parallel. Подпишите форму каждого локального тензора
и места коллективных обменов. Проверьте, что собранный результат совпадает
с результатом неразделённой модели. Затем разберите так же внимание и MLP
в блоке Transformer.

Добавьте **Megatron sequence parallelism**: разбиение по позициям для
активаций таких операций, как LayerNorm и dropout, которые при обычном TP
реплицировались на всех участниках. На границах соответствующих участков
используются all-gather и reduce-scatter; восстановите эти границы по
[Korthikanti et al., 2022](https://arxiv.org/abs/2205.05198).
Не подменяйте это context parallelism: разнесение Q/K/V длинной
последовательности с распределённым вычислением самого внимания — другое
разбиение. Для TP=1,2,4 сравните:

- локальный parameter/activation memory;
- число и bytes collectives;
- step latency и scaling efficiency;
- максимальный допустимый batch/sequence;
- numerical error относительно baseline.

Приложите к коду таблицу форм тензоров до и после каждой операции. По ней должно
быть видно, какую часть данных хранит каждый процесс и зачем на каждой границе
нужен обмен.
