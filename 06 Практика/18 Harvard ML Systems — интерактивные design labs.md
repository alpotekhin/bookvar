---
title: Harvard ML Systems — интерактивные design labs
type: practice
status: canonical
last_updated: 2026-08-06
---

# Harvard ML Systems: интерактивные design labs

Harvard ML Systems сопровождает главы лабораториями на Marimo. В них читатель
сначала фиксирует прогноз, затем меняет параметры системы и сопоставляет
наблюдение с уравнением или ограничением. Bookvar сохраняет все 34 исходные
лаборатории на английском языке; эта страница задаёт короткий маршрут и единый
формат результатов, не переводя и не сокращая оригинальные задания.

## Как работать с лабораторией

1. Откройте локальную страницу и pinned source, указанный в её начале.
2. До изменения controls запишите прогноз и физическую причину.
3. Выполните sweep одного параметра, сохраняя остальные неизменными.
4. Сохраните график/таблицу и отметьте область, где первоначальная модель
   перестаёт объяснять результат.
5. Запишите design decision: какой вариант вы выбрали бы при заданном SLO и
   почему.

Локальные страницы содержат complete original source, commit и лицензию. Для
исполнения используйте исходный `.py` по pinned GitHub-ссылке; Markdown-версия
нужна для чтения, поиска и долговременной атрибуции.

## Маршрут по вычислениям и производительности

- [[05 Источники/Courses/Harvard ML Systems/labs/vol1/lab_05_nn_compute|Neural-network computation]] — формы, операции и стоимость;
- [[05 Источники/Courses/Harvard ML Systems/labs/vol1/lab_07_ml_frameworks|ML frameworks]] — dispatch и execution strategy;
- [[05 Источники/Courses/Harvard ML Systems/labs/vol1/lab_11_hw_accel|Hardware acceleration]] — соответствие workload и accelerator;
- [[05 Источники/Courses/Harvard ML Systems/labs/vol1/lab_12_perf_bench|Performance benchmarking]] — распределения измерений;
- [[05 Источники/Courses/Harvard ML Systems/labs/vol2/lab_09_perf_engineering|Performance engineering]] — bottleneck и выбор оптимизации.

Сопоставляйте результаты с [[02 Areas/ML & DL/00 Учебник/10 ML Systems/03 Измерение производительности и roofline]] и [[02 Areas/ML & DL/00 Учебник/10 ML Systems/07 Profiling ML-нагрузки]].

## Маршрут по распределённым системам

- [[05 Источники/Courses/Harvard ML Systems/labs/vol2/lab_02_compute_infra|Compute infrastructure]];
- [[05 Источники/Courses/Harvard ML Systems/labs/vol2/lab_03_communication|Network communication]];
- [[05 Источники/Courses/Harvard ML Systems/labs/vol2/lab_04_data_storage|Data storage]];
- [[05 Источники/Courses/Harvard ML Systems/labs/vol2/lab_05_dist_train|Distributed training]];
- [[05 Источники/Courses/Harvard ML Systems/labs/vol2/lab_06_collective_communication|Collective communication]];
- [[05 Источники/Courses/Harvard ML Systems/labs/vol2/lab_07_fault_tolerance|Fault tolerance]];
- [[05 Источники/Courses/Harvard ML Systems/labs/vol2/lab_08_fleet_orch|Fleet orchestration]].

Для каждого эксперимента сдайте не только лучший результат, но и границу
режимов: размер сообщения, utilization, число ranks или failure rate, после
которых выбранная стратегия перестаёт быть выгодной.

## Маршрут по inference и эксплуатации

- [[05 Источники/Courses/Harvard ML Systems/labs/vol1/lab_10_model_compress|Model compression]];
- [[05 Источники/Courses/Harvard ML Systems/labs/vol1/lab_13_model_serving|Model serving]];
- [[05 Источники/Courses/Harvard ML Systems/labs/vol1/lab_14_ml_ops|ML operations]];
- [[05 Источники/Courses/Harvard ML Systems/labs/vol2/lab_10_inference|LLM inference]];
- [[05 Источники/Courses/Harvard ML Systems/labs/vol2/lab_12_ops_scale|Operations at scale]].

Итоговый отчёт должен связывать control с TTFT/TPOT, throughput, memory,
queueing или failure recovery. Сравнение без workload distribution и SLO не
считается системным выводом.

## Маршрут по безопасности и ответственности

- [[05 Источники/Courses/Harvard ML Systems/labs/vol2/lab_13_security_privacy|Security and privacy]];
- [[05 Источники/Courses/Harvard ML Systems/labs/vol2/lab_14_robust_ai|Robust AI]];
- [[05 Источники/Courses/Harvard ML Systems/labs/vol2/lab_15_sustainable_ai|Sustainable AI]];
- [[05 Источники/Courses/Harvard ML Systems/labs/vol2/lab_16_responsible_ai|Responsible AI]];
- [[05 Источники/Courses/Harvard ML Systems/labs/vol2/lab_17_fleet_synthesis|Fleet synthesis]].

Здесь design ledger должен содержать проверяемый риск, наблюдаемую метрику,
порог реакции и владельца решения. Общая фраза «учесть fairness/security» не
заменяет эксплуатационный контракт.

## Формат сдачи

Для каждой выбранной лаборатории сохраните исходный прогноз, controls и версии,
полученный artifact, применимое уравнение, объяснение расхождения и design
decision. Минимальный сквозной проект включает по одной лаборатории из трёх
разных маршрутов и заключение о том, как оптимизация одного слоя изменила
ограничения следующего.

Полный каталог: [[05 Источники/Courses/Harvard ML Systems/Labs and slides]].
