---
title: Harvard ML Systems — интерактивные design labs
type: practice
status: canonical
last_updated: 2026-10-29
last_verified: 2026-10-29
---

# Harvard ML Systems: интерактивные design labs

Harvard ML Systems сопровождает главы лабораториями на Marimo. В них читатель
сначала фиксирует прогноз, затем меняет параметры системы и сопоставляет
наблюдение с уравнением или ограничением. Bookvar сохраняет 34 исходные
лаборатории на английском языке из закреплённой версии
`45ecc8d82fcae70c149cdce550d3b3d3411df913`; эта страница задаёт короткий маршрут и единый
формат результатов, не переводя и не сокращая оригинальные задания.

## Как работать с лабораторией

1. Откройте локальную страницу и pinned source, указанный в её начале.
2. До изменения controls запишите прогноз и физическую причину.
3. Выполните sweep одного параметра, сохраняя остальные неизменными.
4. Сохраните график/таблицу и отметьте область, где первоначальная модель
   перестаёт объяснять результат.
5. Запишите design decision: какой вариант вы выбрали бы при заданном SLO и
   почему.

В локальных страницах сохранены полные исходные `.py`, commit и лицензия.
При проверке 15 сентября 2026 года все 34 файла совпали с этой ревизией
upstream по контрольным суммам; код внутри Markdown также совпал, кроме
конечных пробелов и переводов строк, удаляемых импортёром. Проверен синтаксис
Python, **не** исполнение всех интерактивных ячеек. Это каталог исходников,
а не обещание, что 34 приложения уже работают внутри сайта Bookvar.

Для исполнения используйте исходный `.py` по закреплённой GitHub-ссылке и
окружение соответствующей версии курса. Отдельный файл может зависеть от общих
модулей репозитория: сохраняйте структуру проекта, а не только одну ячейку.
Markdown-версия служит для чтения и поиска; графики и элементы управления
нужно открывать в самом приложении Marimo.

<!-- BEGIN REVIEW: original Marimo runtime 2026-10-29 -->
## Запустить оригинальное приложение

В [архиве исполняемой практики](/examples/runtime-labs-2026-10-29.zip) есть
launcher, который подготавливает согласованное окружение курса. После распаковки
выполните из корня архива:

```bash
python3 Code/run_harvard_lab.py --lab vol2/lab_10_inference.py --port 4338
```

Нужны Python, `uv` и сеть для первого получения зависимостей. Страница откроется
по адресу `http://127.0.0.1:4338`. Приложение запускается отдельным процессом;
статическая страница GitHub Pages не исполняет его Python-код. Остановить
локальный сервер можно `Ctrl+C` в терминале.

Подготовка сохраняет 34 исходных `.py`, общие модули, лицензию и две небольшие
библиотеки курса из того же commit. Всего это 261 файл и 4.42 MB исходных
материалов; Python-зависимости дополнительно сохраняются в кеше `uv`.
Репозиторий, наборы данных и модельные веса целиком не скачиваются.
Для каждого исходного файла проверяется Git blob ID и записывается SHA256.
Пользовательские правки в каталоге runtime не перезаписываются: несовпадение
хеша останавливает подготовку.

Одних опубликованных wheel-файлов этой ревизии недостаточно. Они отстают от
части исходников: четыре лаборатории сначала падали на отсутствующих
`track_display_label`, `action_box`, `calc_alpha_beta_crossover` или поле
`bandwidth_per_direction`. Launcher сохраняет и подключает общие Python-модули
того же закреплённого commit. После этого все 34 приложения выполнили свои
начальные ячейки без исключений. Это проверка запуска, а не отметка о решении
всех упражнений.

Lab V2-10 дополнительно проверена через интерфейс: выбраны прогнозы и
checkpoint-ответы частей A–D, открыт Synthesis и разблокирован экспорт отчёта.
При изменении demand с 100 на 200 событий/с вычисляемый суточный расход меняется
с 77 760 до 155 520 USD, а crossover — с 25.7 на 12.9 дня. Меняются и исходные
Plotly-графики. Эти числа вычисляет учебная модель Harvard; они не описывают
нагрузочный тест нашего сервиса. JSON из кнопки Download JSON snapshot сохранён
в архиве как `Results/runtime-2026-10-29/marimo-lab10-ui.json`; список
`incomplete_fields` в нём пуст.

Чтобы открыть другую лабораторию, замените аргумент, например на
`vol1/lab_11_hw_accel.py`, `vol2/lab_06_collective_communication.py` или
`vol2/lab_12_ops_scale.py`. Названия и содержание исходных приложений остаются
английскими. Настройки и design ledger записывает приложение курса; для
изолированного учебного эксперимента используйте отдельный пользовательский
каталог или окружение.
<!-- END REVIEW: original Marimo runtime 2026-10-29 -->

## Маршрут по вычислениям и производительности

Подробный разбор [[02 Areas/ML & DL/06 Практика/18a Сравнить архитектуры по ресурсным ограничениям|Lab 06: выбор архитектуры]]
сопоставляет допустимость, качество и запас до ограничений, объясняет исходные
графики и отделяет сценарные коэффициенты от измерений оборудования.

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
