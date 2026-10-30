---
title: Harvard ML Systems — complete transfer matrix
type: source-note
status: editorial
last_updated: 2026-10-29
---

# Harvard ML Systems — complete transfer matrix

<!-- BEGIN CURRENT RECONCILIATION 2026-10-29 -->
## Текущее состояние — 29 октября 2026

Этот раздел уточняет результат последнего прохода. Записи ниже него и 173 строки
исходной матрицы сохранены как история: `audited` по-прежнему означает просмотр
и сохранение источника, а не полный перенос главы. В частности, старые пометки
«Marimo runtime не запускался» относятся к указанной в них дате.

### Пять системных тем

| Тема и исходная строка | Что сопоставлено и дополнено | Проверяемый отчёт и граница результата |
|---|---|---|
| Model Compression — `H-V1-10` | Прочитан весь опубликованный материал закреплённого QMD, строки 1–8139; карта содержит 80 разделов и 16 непрерывных диапазонов. В 57b добавлены iterative pruning/reset, NAS/DARTS, early exits, compound scaling, folding BatchNorm и binary dot product. Три оригинальных SVG дополняют уже принятые SVD, distillation и tensor factorization. | `docs/audit/2026-10-29-systems-completion.md` и `docs/audit/2026-10-29-compression-source-section-map.json`. Содержательное сопоставление этой версии главы завершено с замечаниями к источнику; шесть новых программ проверены на CPU. Это не запуск NAS, проверка качества сжатой обученной модели или полная приёмка всех lab/deck/native-figure объектов курса. |
| Hardware Acceleration — `H-V1-11` | Прочитан опубликованный материал строк 1–5935. Для 113 разделов записаны 62 решения covered-existing, 40 added и 11 inapplicable; есть решения по 14 фигурам. В S2 раскрыты systolic/dataflow, matrix/vector/scalar paths, im2col, layout, placement/lifetimes, Unified Memory, SoC и package/node/fleet. Сохранены исходный Harvard SVG и две полные анимации Google TPU. | `docs/audit/2026-10-29-hardware-source-comparison.md` и `.json`, а также systems-completion выше. Hidden generators/TikZ не объявлены прочитанным учебным текстом; covered-existing включает дополнения текущего прохода. Примеры CPU не являются измерением GPU/TPU, миграции страниц или энергии. |
| Benchmarking — `H-V1-12` | Полностью прочитаны строки 1–2214 импортированной главы, 88 разделов опубликованного объяснения. В 58b добавлены границы измерения энергии, интеграл мощности, idle baseline и стоимость успешного результата; сохранён полный оригинальный SVG. Остальные механизмы сопоставлены с существующими измерительными главами. | `docs/audit/2026-10-29-serving-full-source-reconciliation.md`. Содержательное сопоставление завершено с замечаниями; исполнен пример учёта энергии. Это не физическое измерение мощности и не запуск MLPerf. |
| Model Serving — `H-V1-13` | Полностью прочитаны строки 1–2171, 97 разделов опубликованного объяснения. В 58c добавлены fan-out, корреляция, partial results и hedging; в serving engines — NUMA, thread pools, mmap/residency/transfer и загрузка через meta/assign. | Тот же полный отчёт и `docs/audit/2026-10-29-serving-source-section-map.json`. Содержательное сопоставление завершено; численные примеры, временные таблицы и CPU storage/output проверены. Условные задержки не выдаются за измеренное поведение production-сервиса. |
| MLOps — `H-V1-14` | Полностью прочитаны строки 1–2230, 101 раздел опубликованного объяснения. Добавлены ML Test Score, стоимость переобучения и устаревания, точное условие оптимума, A/B sample size и время наблюдения. Сохранены оригинальные рисунки инфраструктуры и политик переобучения. | Тот же полный отчёт и секционная карта. Содержательное сопоставление завершено с замечаниями; проверены исходные фрагменты и численные примеры. Восемь выбранных проверок не названы полным перечнем 28 тестов ML Test Score; production-изменений не было. |

Карта последних трёх глав содержит 289 заголовков: 286 разделов объяснения
и три приложения с ответами, исключённые из переноса. Записаны 59 тематических
решений; `outline-only` больше нет. Она заменяет первоначальную неполную
опись, а не отменяет её исторический отчёт. Всего закрыто шесть выявленных
пробелов в четырёх существующих главах; остальные разделы распределены между
уже раскрытыми механизмами, отдельными тематическими главами, историческим
контекстом источника и аргументированными исключениями. Неподтверждённые
проценты ускорения, кейсы и ошибочные обобщения не перенесены как факты.

### Практика: исполнено и подготовлено

- Все 34 исходных Marimo-приложения исполнили начальное состояние с зависимостями
  того же commit `45ecc8d82fcae70c149cdce550d3b3d3411df913`. Для V2 Lab 10
  дополнительно пройдены интерактивные этапы A–D и сохранён настоящий отчёт.
  У остальных 33 приложений не заявляется прохождение всех UI-этапов.
- Практика 09 выполняет настоящее межпроцессное Gloo ring-общение: 24 конфигурации,
  72 результата rank, а также двухпроцессный DDP с проверкой шага относительно
  общего batch. Это локальная корректность, не пропускная способность сети GPU.
- Практика 14 обучает тестовый WordLevel tokenizer, сохраняет и загружает tiny
  random Qwen3 checkpoint; cached/uncached вычисления проверены на CPU и MPS.
  Это не pretrained Qwen и не проверка языкового качества.
- Практика 17 собрана и запущена в Docker: HTTP/gRPC, readiness, валидация,
  HTTP overload, настоящий Prometheus scrape и 620 поступлений open-loop
  нагрузки. Модель — маленький RGB-классификатор с явно заданной задержкой,
  не VGG/LLM. gRPC overload реализован, но этот исход в эксперименте не проверен.
- CUDA/NCCL ветви подготовлены с fail-fast проверками. На этом хосте CUDA нет;
  GPU throughput, physical peak memory, multi-node overlap и GPU SLO-capacity
  остаются неизмеренными. Короткая локальная нагрузка не закрывает эти пункты.

Код, сохранённые результаты, ограничения и состав ZIP:
`docs/audit/2026-10-29-practice-completion.md`. Предыдущие 54 CPU-запуска
и HTTP/NDJSON loopback не пересчитаны как новые результаты.

### EN и CS336 — отдельные результаты

Завершено содержательное сопоставление всех 29 существующих файлов EN:
28 глав и оглавление имеют статус current по закреплённым каноническим SHA.
Сохранены 183 размещения исходных рисунков; исполнены 176 CPU-блоков.
Все 29 адресов проверены на широком и узком экране. Это не перевод всех
149 канонических страниц; отсутствующие EN-главы остаются явными fallback.
Поимённый состав, история прежних partial и проверки:
`docs/audit/2026-10-29-english-reconciliation.md` и `.json`.

CS336 не увеличивает число принятых Harvard units. Его точечные результаты:
V4 CSA/HCA встроены в семейство DeepSeek; пять изображений ARC/AIR/GCG —
в главы Evaluation/Responsible systems с пояснениями. Четыре старых отложения
Lecture 02 разобраны: CPU/GPU и deep-network механизмы уже объяснены,
два повторных показа deep-network — дубликаты. Точные source units, происхождение
семи добавленных рисунков и пределы проверки:
`docs/audit/2026-10-29-cs336-residuals.md`.

Остаются отдельными требованиями: перевод остальных глав, аппаратные измерения,
UI-прохождение остальных Marimo-сценариев и тематическая приёмка других
Harvard/HSE units, для которых пока есть только импорт или ссылка.
Этот проход не объявляет весь Harvard, HSE, EDLS или CS336 полностью перенесённым.
<!-- END CURRENT RECONCILIATION 2026-10-29 -->

## История сверок и исходная матрица

27 октября 2026, секционная сверка: H-V1-06 получил 37 явных решений по
всему закреплённому исходнику, с актуальными SHA 17 тематических страниц
и ссылками на их содержательную приёмку. Добавлен недостающий расчёт камеры:
частичная память, условные 75 ms, активная энергия и отдельный полный
двухрежимный бюджет с потреблением покоя. Разобраны KWS и ограничения
исходных выводов о точности/ёмкости. Карта:
`docs/audit/2026-10-27-nn-section-map.{md,json}`. Текущий статус строки —
thematic review complete with findings; это не завершение курса, Marimo UI,
GPU/устройственных измерений или отложенной EN-паритетности.

27 октября 2026, Winograd: в ходе секционной сверки найдено отсутствующее
объяснение алгоритма из Hardware Optimization Patterns. В CNN добавлены
вывод и численные 1D/2D примеры, channel-GEMM, valid tiling, storage и
FP16-контрпример. Сохранены исходные английский текст/матрицы wincnn и
полный рисунок Dolz et al. из векторного PDF с CC BY 4.0. 9→4 — сравнение
variable products на выход, не измеренное ускорение. Приёмка:
`docs/audit/2026-10-27-winograd.md`. H-V1-06 остаётся integration-partial:
сведение остальных уже прочитанных разделов в единую карту ещё не завершено.

27 октября 2026, native figures: отдельно просмотрены все 24 изображения
Network Architectures и их подписи. В CNN перенесён полный исходный SVG
развёртки свёртки: исправлено пояснение строк/столбцов, разобраны оба канала,
все четыре выхода и связь с `F.unfold`. Selection SVG остаётся в практике
18a. Остальные объекты сохранены в архиве курса с индивидуальными причинами:
дублирование, недостающие условия или конкретные ошибки. Отчёт и все хеши:
`docs/audit/2026-10-27-nn-native-figures.{md,json}`. Это закрывает отдельную
проверку native directory, не весь план Harvard/HSE/EDLS и не общий аудит прав.

27 октября 2026, тензоры и GPU: полностью прочитаны и сопоставлены
747 строк основ тензоров и 910 строк GPU/CUDA/памяти. Формы, broadcasting,
редукции, индексация, tiling, coalescing, банки/XOR, fusion и зависимости
streams уже раскрыты на примерах. GPU-глава не изменена; в первой главе
перенесена одна переполнявшаяся при 360 px формула QKᵀ, без изменения
чисел или рассуждения. Примеры NumPy и CPU-модель GPU прошли проверки.
NVIDIA spot-check подтвердил секторную модель и указанные ограничения
TMEM; это не новое измерение CUDA. Затем целиком прочитаны roofline (1105
строк) и kernels (1699 строк до дополнения): их разборы трафика, tiling,
strides, fusion, online reduction и градиентов сохранены. В S8 добавлен
пошаговый разбор дерева редукции: четыре кадра оригинальной анимации Flogr,
промежуточные суммы, work/depth, padding, барьеры и исходный CUDA helper.
CPU-пример проверяет каждый уровень для 1025 массивов; различены 31 полезное
сложение и 160 lane-level сложений полной warp-схемы. Затем добавлен
пошаговый indexed gather/scatter: исходный SVG Matthias Fey, тексты PyTorch
на английском, повторные индексы, scatter-add backward, отличие Gather от
GatherElements и локальной индексации от сетевого обмена. Четыре напечатанных
CPU-примера, 18 rank/axis/shape случаев, оба gradcheck и исходные ONNX reference
helpers проверены отдельно от GPU-производительности. Полностью сопоставлена
текущая 44a (1411 строк до дополнения): существующие DDP/кольцо/overlap
сохранены, недостающие rooted/all-rank получатели и неравные all-to-all splits
разобраны с тремя оригинальными рисунками PyTorch и четырьмя CPU/Gloo
процессами. Различены сумма и список, порядок rank и порядок прихода,
строки send-counts и столбцы receive-counts. Native figure disposition
остаётся открытым; успешный обмен не считается измерением GPU-сети.
В адаптере исправлен
стык DOI/картинки: теперь все 24 изображения Network Architectures рендерятся,
а хеш оригинала не изменён. Проверены ещё такие вставки — всего 17 в десяти
исходных страницах, в обоих языковых маршрутах. Это не приёмка их содержания:
у восстановленного Transformer обнаружена неверная разводка входов
cross-attention, поэтому его не переносят как корректную схему в учебник.

27 октября 2026, Shared Building Blocks: полностью сопоставлены три
текущие главы — основы инициализации/нормализации (873 строки),
pre-norm/RMSNorm/SwiGLU (712 строк до правки) и dense FFN (845 строк).
Содержательных пробелов относительно выбранного раздела Harvard не найдено:
выводы производных, контрпримеры, backward обеих ветвей и TP уже раскрыты.
Изменены только переносы четырёх формул, переполнявшихся при 360 px,
и даты двух страниц. Прежнее изложение сохраняется по восстановленному SHA.
Не перенесены универсальные гарантии Jacobian conditioning/стабильности,
проценты ускорения без условий и фиксированный минимальный batch size.
Остаются systems primitives и native figure disposition; H-V1-06/H-F1-06
не объявлены завершёнными.

26 октября 2026, Neural Architectures: отозванная связь с LLaMA заменена
частичным тематическим размещением в практике 18a. Полностью прочитаны
Lab 06 (1721 строк), deck (1896 строк, notes/backup), helper (413 строк)
и четыре variant-сценария. Разобраны пять ограничений, исходный budget chart,
различия quality-score/headroom/bias, дискретный sweep и пределы сценарных
коэффициентов. Перенесены полный `_scaled` и исходный английский список
шагов из Decision framework. Две версии источников разделены: chapter
2bd97c5, lab/deck 45ecc8d. Бюджетный график отрендерен из неизменённого
кода, selection SVG сохранён из архива; старый URL последнего уже 404.
Не перенесены как факты неподтверждённые проценты утилизации и противоречивые
расчёты слайдов. Затем прочитаны введение и разделы MLP/CNN до строки 1990;
длинные координаты zebra просмотрены в исходном SVG. Текущие главы MLP/CNN
прочитаны целиком для сравнения. Основные механизмы уже раскрыты; недостающая
поворотная эквивариантность встроена в CNN через оригинальные английские
фрагменты и три изображения escnn, включая две GIF. Затем последовательное
чтение доведено до строки 3690: RNN, attention, Transformer и sparse RecSys.
Полная текущая RNN сопоставлена с источником; добавлены предварительные
проекции, зависимости нескольких слоёв и суммирование градиентов весов,
исходный английский фрагмент и полный SVG D2L. Сравнение остальных трёх
разделов продолжено полным чтением self-attention и полного Transformer:
основные механизмы уже раскрыты, новый конечный Q/K-пример поясняет
неоднозначность весов attention. Затем полная глава ранжирования сопоставлена
с RecSys: отсутствующий DLRM встроен после DeepFM. Сохранены исходный метод
взаимодействий, два английских фрагмента и три полные схемы TorchRec;
добавлены пошаговый пример и проверка градиентов/размещения таблиц.
Systems-сопоставление ещё не завершено. Источник прочитан последовательно до 4559, включая Shared
Building Blocks и Computational Primitives; спорные универсальные выводы
о Jacobians, нормализации, im2col и memory access не перенесены.
Затем целиком прочитаны Architecture Selection, Fallacies/Pitfalls и
Summary до строки 5256: последовательное чтение исходной главы завершено.
Остаются тематическое сопоставление с системными главами и
проверка native figure directory; это не закрытие
H-V1-06/H-F1-06. Приёмка:
`docs/audit/2026-10-26-architecture-selection.md`.
Дополнение CNN: `docs/audit/2026-10-26-cnn-equivariance.md`.
Дополнение RNN: `docs/audit/2026-10-26-rnn-execution.md`.
Сопоставление attention/Transformer: `docs/audit/2026-10-26-attention-transformer.md`.

26 октября 2026, H-L2-12 / H-SL2-12: Lab 12 и полный TeX со speaker notes
прочитаны. В практику 17 перенесён связный расчёт четырёх ограничений выпуска:
availability/quality/count, выборка и exposure canary, две фазы инцидента,
конъюнкция policy guardrails. Сохранены три исходных английских фрагмента
и полная canary-функция; исполнены восемь чистых функций оригинала без
Marimo/ledger. Пример показывает отсутствующую deadline-проверку, задержку
меток и условность runbook/policy коэффициентов. В MLOps добавлены исходные
fleet-alert расчёты с разграничением expectation, independence и notifications.
Две спорные схемы отклонены с причинами; переиспользованы полные исходные
Prometheus/SRE иллюстрации. Это тематическая проверка, не закрытие всего
H-V2-12 и не запуск интерактивной лаборатории. Отчёт:
`docs/audit/2026-10-26-ops-scale.md`.

26 октября 2026, после FinOps: визуальный пробел MIG закрыт двумя целыми
авторскими векторными рисунками MISO (CC BY 4.0), не перерисовкой и не
маленькими NVIDIA PNG. В MLOps сопоставлены GI/CI, выбранные профили A100,
исчерпание memory slices, версия R510 и зависимость результата от job mix.
Источники 2022 года сверены с текущими NVIDIA Concepts/MPS: динамический
thread cap не выдан за выделенные SM; STP не выдан за p99 или requests/s.
Переиспользование MISO дополняет объяснение resource management, но не
закрывает оставшиеся H-V2-12 chapter/lab/slides. Приёмка:
`docs/audit/2026-10-26-mig-visuals.md`.

26 октября 2026: FinOps/TCO из H-V2-12 встроены в MLOps тематически:
ledger без двойного учёта, распределение общего/idle бюджета, стоимость
завершённого результата, энергия и повторы обучения, качество и окупаемость.
Два английских фрагмента сохранены дословно, исходные сценарии разобраны
на числах, TCO-график воспроизведён из неизменённой Python-ячейки Harvard.
Границы экспорта расширены для подписей, обрезавшихся обычным сохранением
canvas. Условные цены, доли и ожидаемые эффекты не выданы за замеры;
формула стоимости по request latency не перенесена без исправления модели
учёта GPU-времени. Это дополнение не закрывает полный chapter/lab/slides;
MIG-иллюстрация остаётся открытым пунктом. Проверки и приёмка:
`docs/audit/2026-10-26-finops.md`.

26 октября 2026: добавлен ограниченный раздел resource scheduling / tenant
isolation в MLOps. Исходные английские объяснения обучения/serving из H-V2-12
связаны с полной документацией Kueue и Kubernetes и выбранными NVIDIA
sharing-разделами. Сохранены восемь фрагментов и три оригинальных рисунка;
квоты, физическое размещение и runtime-границы не выданы за одно свойство.
Материал H-V1-14 о порогах утилизации/иерархии памяти прочитан, но его
универсальные диагностические выводы не перенесены. Эти частичные связи
не закрывают полный перенос главы, lab или slides. Качественная схема
MIG остаётся отдельным визуальным замечанием, без собственной перерисовки.

26 октября 2026: MLOps расширена двумя тематическими разделами из H-V1-14
и H-V2-12: исторические признаки и миграция их определений. Перенесены
исходные английские объяснения, SQL и полная таблица lineage; добавлены
разбор поздней доступности, TTL, стоимости хранения и backfill по разделам.
Временная схема Harvard сохранена целиком; архитектурная схема отобрана
из Feast v0.66.0. Исходная Harvard architecture SVG отклонена из-за обещания
автоматического исключения skew. Это проверка выбранных разделов, не приёмка
всех глав, лабораторий или слайдов. Resource management и isolation прочитаны
для сравнения, их недостающее изложение остаётся отдельным пунктом.

26 октября 2026: в принятую главу ML workflow добавлены два ограниченных
англоязычных раздела из H-V1-03: бюджет полезных экспериментов и несколько
частот обратной связи. Сохранены исходный расчёт IterationTax и полный абзац
Multi-scale feedback, перенесён неизменённый SVG временных масштабов.
Числа accuracy помечены как допущения; контрпример не позволяет принять
быструю итерацию за гарантию выигрыша модели. Проверены 600 дополнительных
арифметических случаев, в том числе зрелость когорты и доступность метки.
Принятые 635 строк восстанавливаются тестом. Это расширение выбранных разделов,
не полный перенос всей главы, Marimo lab или deck.

26 октября 2026: полный inference lab H-L2-10 (2219 строк) и deck H-SL2-10
(1940 строк, включая notes и backup) прочитаны. В практику 14 добавлены
исходные английские объяснения и формулы, расчёт стоимости, GQA KV с запасом,
исполняемый scheduler, проверка ограничений и backlog при cold start.
Рисунки Harvard с неподтверждёнными процентами и несогласованными шкалами
не перенесены: повторно использованы два полных оригинальных PNG Hugging Face.
Сценарные формулы lab не выданы за benchmark; арифметическая ошибка
100/(8×5)=0,25, вывод точного p99 из P2C и обязательный переход reasoning
в compute-bound режим не перенесены как факты. Marimo UI и GPU-движок
не исполнялись, поэтому полная интеграция inference не объявляется закрытой.

26 октября 2026: полный код H-L2-09 (1602 строки) и H-SL2-09 (1796 строк,
37 явных frames, notes и backup) прочитаны; оригиналы сверены с readable layer.
В [[06 Практика/13 Оптимизировать один Transformer step#performance-evidence-profiling-regression-capacity|практике 13]]
раскрыты четыре этапа lab: профиль, проверка эффекта, serving-мощность,
отбор конфигураций и отчёт. Перенесены исходные английские инструкции,
добавлены полный пример Welch CI, расчёт мощности теста и четыре исполняемых
CPU-примера. Исходный CI-порог lab не выдан за MDE с заданной мощностью,
сценарные p99/quality/capacity — за реальные замеры. Оригинальный Marimo UI
не воспроизводился. Девять архивных SVG просмотрены, новых копий не добавлено;
для профиля повторно использована точная иллюстрация CS336 с атрибуцией.
Пробел MBU дополнен в главе 55a: расчёт через время шага и throughput,
различие модельных байтов и HBM-счётчиков, MoE и перекос между GPU.
Перенесены полное английское Appendix A JAX Scaling Book и два исходных
PNG с проверенной MIT license. Это не измерение MBU на GPU и не приёмка
всего H-V2-09. Полный перенос Performance Engineering остаётся открытым.

26 октября 2026: Measurement at Scale сопоставлен с полными S3 и S7.
В [[02 Areas/ML & DL/00 Учебник/10 ML Systems/03 Измерение производительности и roofline#fleet-mfu-scaling-and-preserved-progress|S3 добавлен раздел измерений кластера]]:
MFU/HFU, strong/weak scaling, взвешивание работы и мощности, сохранённый
прогресс после сбоев. Перенесены исходные английские объяснения Ultra-Scale,
NVIDIA и Google Goodput и два полных PNG NVIDIA. Сценарий Harvard 8/128 GPU
пересчитан с явным 6PD и помечен как предположение, а не замер.
Снижение MFU не приписано автоматически сети; универсальный «hero run tax»
не перенесён. Стадийные числа 70B Optimization Playbook не имеют артефакта
запуска и не приняты как эксперимент. Его profile/change/reprofile уже
раскрыт в S7; H-V2-09 остаётся частичным, lab/slides не закрыты этой правкой.

26 октября 2026: overlap из Performance Engineering сопоставлен с текущей
[[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/44a Processes, collectives и DDP#ddp-overlap-readiness-streams-compilation|главой DDP]].
Существующее объяснение сохранено; добавлены готовность buckets на разных
rank, зависимость от порядка, полный исходный раздел PyTorch о streams,
allocation lifetime и взаимодействие Dynamo/AOTAutograd с DDPOptimizer.
Перенесена ещё одна исходная HF SVG, а не синтетическая кривая overlap
из Harvard. Суммарная длительность backward не выдана за доказательство
полного скрытия сети. Это дополнительный тематический destination H-V2-09;
вся исходная глава, lab и slides по-прежнему не приняты целиком.

26 октября 2026: для Performance Engineering уточнено размещение.
В главу S8 о GPU kernels и Triton добавлены исходные английские объяснения
PyTorch: CUDA Graphs (capture/replay, буферы, память, трассы и окупаемость)
и Dynamo (FX, guards, cache reuse, graph breaks, dynamic shapes), с четырьмя
исходными рисунками и исполняемыми CPU-примерами. Harvard используется как карта темы, а не источник
непроверенных универсальных чисел ускорения. В §Compilation pipeline также
найдена ошибка порядка зависимостей: между GELU и LayerNorm стоит второй
Linear, поэтому описанное там прямое объединение GELU–LayerNorm не следует
из приведённого графа. Этот фрагмент не перенесён.
H-V2-09 имеет только частичную связь; лаборатория, весь набор слайдов и
каталог рисунков остаются на отдельной проверке. Их прежний единственный
destination FlashAttention заменён кандидатом S8, без повышения готовности.

22 октября 2026: после переработки LLaMA строки H-V1-06, H-L1-06,
H-SL1-06 и H-F1-06 возвращены в `re-review`. В главе теперь подробно
используются первичные статьи, код LLaMA и CS336; прежнее размещение
Harvard Neural Architectures больше не подтверждено текущим текстом.
Указанный destination остаётся кандидатом. Полные исходники сохранены;
их знания и рисунки требуют отдельного тематического сопоставления.
Формальная ссылка на курс ради проверки покрытия не добавлялась.

20 октября 2026: для главы Edge и federated deployment сверены выбранные
разделы Architectural trade-offs, Client scheduling и Federated privacy из
commit `45ecc8d82fcae70c149cdce550d3b3d3411df913`. Из Harvard перенесены
`edge_intelligence_straggler_cutoff_strip.svg` и `secure-agg.svg`;
остальные новые рисунки взяты из TinyTL, FedAvg и LiteRT. Происхождение
зафиксировано в `05 Источники/Originals/edge-2026-10-20/references.json`.
Assets в H-V2-11, H-SL2-11 и H-F2-11 остаются историческими кандидатами,
не перечнем принятых иллюстраций. Лаборатория H-L2-11 и полный набор слайдов
этой правкой не интегрированы; присутствие исходников не закрывает перенос.

20 октября 2026: глава MLOps переработана; приёмка описана в
`docs/audit/2026-10-20-mlops.md`. В ней использованы выбранные
разделы Harvard из commit `45ecc8d82fcae70c149cdce550d3b3d3411df913`,
а не полный перенос исторического snapshot ниже. Из новых Harvard-рисунков
подтверждён только `ops_scale_cross_model_blast.svg`; остальные иллюстрации
главы взяты из Google Cloud и SRE. Поля Assets в строках H-V1-14, H-V2-12,
H-SL1-14, H-SL2-12, H-F1-14 и H-F2-12 остаются перечнем исходных кандидатов,
не списком встроенных рисунков. Лаборатории и слайды сохранены как оригиналы;
их наличие и ссылка не закрывают перенос всех содержащихся там знаний.

14 октября 2026: четыре строки Performance Engineering (H-V2-09, H-L2-09,
H-SL2-09, H-F2-09) возвращены в состояние `re-review`. В новой редакции
FlashAttention источник Harvard и его рисунки больше не используются.
Destination в этих строках остаётся прежним кандидатом, а не подтверждённым
местом интеграции. Полные исходные главы, слайды и лаборатории сохранены;
их тематическое размещение требуется пересмотреть. Наличие файлов не считается
переносом их содержания в главу 56.

27 октября 2026, TensorFlow: в «Жизненный цикл ML-системы» перенесены
полные исходные объяснения SavedModel, пять кодовых ячеек и оригинальный
граф. Исполнены загрузка в процессе без исходных классов, изменение и
дообучение переменных, различия способов вызова сигнатуры и CPU-конвертация
в Lite. Все 2360 строк текущей MLOps прочитаны для сопоставления выбора
runtime, совместимости и эксплуатационных ограничений. Этот пункт карты
Frameworks закрыт; JAX runtime и три native-кандидата остаются открытыми.
EN — проверенный fallback, не перевод. Приёмка:
`docs/audit/2026-10-27-tensorflow-export.md`. H-V1-07 остаётся partial.

28 октября 2026, 3D placement: оригинальный векторный рисунок Harvard
и полная английская подпись встроены в главу о распределённом обучении.
Показаны 32 логические координаты, группы TP/DP/PP и пакет 30 вместо
ошибочных 480 последовательностей. Это размещение, не топология сети
или расписание конвейера. Приёмка: `docs/audit/2026-10-28-frameworks-placement.md`.
Из прежних трёх кандидатов остаются два неиспользованных FL/transfer PNG;
H-V1-07 остаётся partial, полный курс не объявлен перенесённым.

27 октября 2026, JAX runtime: в существующую главу о ядрах и компиляции
встроены оригинальные объяснения и примеры JAX на английском: трассировка,
явное состояние и refs, специализации кэша, ветвления, циклы и передача
буфера. Исходная схема `psum` сопровождает полный пример коллективной суммы
на четырёх логических CPU-устройствах. Сравнение с `vmap` явно задаёт
однодевайсную mesh; это не тест GPU или сети. Приёмка:
`docs/audit/2026-10-27-jax-runtime.md`. Три native-кандидата и полный H-V1-07
остаются открытыми; английская страница пока использует явный fallback.

27 октября 2026, экспорт: в главу «Жизненный цикл ML-системы» добавлены
исходные объяснения Harvard, PyTorch и ONNX Runtime, полная оригинальная
схема execution providers и исполняемый пример PyTorch → ONNX → CPU.
Разобраны границы shape-контракта и отсутствие custom kernel. Числовые
проверки и осмотр desktop/mobile/увеличения пройдены. EN остаётся явно
помеченным fallback, а не готовым переводом. Это дополнение к
H-V1-07, а не завершение всей главы Frameworks или курса.

27 октября 2026, Frameworks: прочитан текст закреплённой главы, включая
Python и псевдокод; пропущенный фильтром TikZ ещё требует визуального разбора.
Полный исходный пример регистрации перенесён в
[[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/01 Нейрон и MLP|MLP]],
где добавлены регистрация/градиент/оптимизатор, буферы, ModuleList и загрузка
значений в уже созданную модель. Утверждения исходника о metaclass,
фильтрации frozen parameters и восстановлении архитектуры не приняты.
Прежняя destination Profiling — кандидат для системных частей, не место
для всей темы; остальная секционная сверка H-V1-07 остаётся открытой.
Ревизия и границы приёмки: `docs/audit/2026-10-27-module-state.md`.
Перечень Assets в H-F1-07 остаётся перечнем кандидатов.
Следующим проходом целиком прочитана S8; сохранены уже подробные Dynamo,
fusion, CUDA Graphs, overlap и amortization. В неё добавлен полный исходный
Harvard-пример неверной трассировки ветки и исходные PyTorch-классы
`M_old`/`M_new` с разбором `torch.cond` и export. Примеры исполняются на CPU;
это не проверка GPU-ускорения. Приёмка:
`docs/audit/2026-10-27-tracing-control-flow.md`. Остальная сверка H-V1-07,
AD/runtime/training-step и native figures остаются открытыми.

Позднейшие проходы 27 октября: native figures осмотрены индивидуально,
три кандидата оставлены открытыми. AD и training-step получили отдельную
содержательную и визуальную приёмку; ссылки приведены в строке H-V1-07.
Полный исходный JAX-пример теперь дополнен числовым разбором grad/vmap/jit,
форм градиентов, функционального SGD и ошибочного broadcasting в backpropagation.
Приёмка: `docs/audit/2026-10-27-jax-transforms.md`. Это проверка CPU-вычислений,
не производительности или multi-device выполнения. Полная секционная карта,
сверка platform/deployment/selection и оставшиеся findings ещё не закрыты.

Полная карта 37 разделов Frameworks с текущими SHA 13 destinations сохранена
в `docs/audit/2026-10-27-frameworks-section-map.{md,json}`. Она отдельно
помечает уже разобранные механизмы, ошибки/неподтверждённые оценки источника
и открытые сравнения. Наличие карты не закрывает hooks, JAX runtime,
platform/deployment/selection и три held figure candidates.

В MLP добавлен полный исходный module-hooks пример и числовой разбор
градиентов выхода, входа и весов, условий вызова hook и удаления handles.
Ревизия и результаты проверки: `docs/audit/2026-10-27-module-hooks.md`.
Это не закрывает оставшиеся platform/deployment/selection и JAX runtime.

В Edge добавлены полный исходный micro-runtime passage и разбор tensor arena:
model buffer против SRAM, lifetimes и повторное использование памяти,
подготовка против инференса, регистрация/поддержка операций. Прежняя глава
с локальным обучением и FL сохранена. Ревизия и границы проверки:
`docs/audit/2026-10-27-micro-arena.md`. Это дополнение не является запуском
прошивки или полным переносом platform/export-разделов.

Pinned source: [harvard-edge/cs249r_book](https://github.com/harvard-edge/cs249r_book/tree/2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd), published snapshot commit `2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd`. Source language is English. `book/`, `labs/`, and `slides/` are CC BY-NC-SA 4.0; MLSys·im is Apache-2.0. TinyTorch software is MIT and its embedded educational prose is CC BY-NC-SA 4.0; all twenty progressive modules are preserved in the local source layer. Vol II material is marked preview.

`audited` в последнем столбце означает только то, что исходный объект просмотрен,
классифицирован и сохранён в полном импорте. Это **не** утверждение, что его
содержание уже встроено в указанную каноническую главу. Фактическое состояние
интеграции ведётся отдельно в
[[05 Источники/Source maps/ML systems — Bookvar gap matrix|ML systems — Bookvar gap matrix]]
и проверяется по существованию destination, прямой ссылке на полный источник и
наличию перенесённых объяснений или упражнений.

| ID | Source path | Material | Adds to Bookvar | Decision | Destination | Assets | Status |
|---|---|---|---|---|---|---|---|
| H-V1-01 | `book/quarto/contents/vol1/introduction/introduction.qmd` | DAM, Iron Law, energy hierarchy | yes | integrate | `00 Учебник/10 ML Systems/01 Модель как часть системы.md` | _system-map.svg; introduction_energy_hierarchy.svg; introduction_iron_law_bars.svg | audited |
| H-V1-02 | `book/quarto/contents/vol1/ml_systems/ml_systems.qmd` | workload characteristics and systems constraints | yes | integrate | `00 Учебник/10 ML Systems/01 Модель как часть системы.md` | _edge_ml_iot.svg; ml_systems_dam_locator.svg; ml_systems_deployment_span.svg | audited |
| H-V1-03 | `book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd` | ML lifecycle, iteration budget and feedback timescales | yes | integrate | `00 Учебник/19 Deployment, Reliability и MLOps/01 ML workflow.md` | complete feedback-timescales SVG verified at 45ecc8d; other historical figures remain candidates | partial-reviewed-2026-10-26; two source-backed extensions added to accepted chapter; full source/lab/deck transfer remains open |
| H-V1-04 | `book/quarto/contents/vol1/data_engineering/data_engineering.qmd` | data quality, debt, storage, and pipelines | yes | integrate | `00 Учебник/11 Pre-training и Scaling/41 Сбор, очистка и смеси данных.md` | _kws_multilingual.svg; _kws_pipeline.svg; data_engineering_active_learning_budget.svg | audited |
| H-V1-05 | `book/quarto/contents/vol1/nn_computation/nn_computation.qmd` | matrix computation, activations, and memory; small-MLP accounting example | yes | cross-link | `00 Учебник/10 ML Systems/04 Арифметика Transformer и MoE.md` | original figures remain in the full source chapter, not embedded in S4 | cross-linked 2026-10-13; see evidence below |
| H-V1-06 | `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd` | inductive bias and architecture bottlenecks | yes | integrate | `06 Практика/18a Сравнить архитектуры по ресурсным ограничениям.md` | archived native selection SVG and original lab plot; secondary CNN uses escnn PNG/GIFs, RNN uses complete D2L SVG, attention/Transformer retain Alammar/D2L originals; RecSys uses three complete original TorchRec diagrams; local reduction uses the complete original Flogr GIF; indexed movement uses the complete original Matthias Fey SVG; collective destinations use three complete original PyTorch PNGs; normalization/gating and tensors/GPU retain existing primary-source figures; 24 native objects individually reviewed; one im2row SVG added to CNN with corrected caption; selection SVG retained in practice; original Dolz Winograd workflow retained with CC BY 4.0; camera case uses two explicit budget tables | source-chapter-thematic-review-complete-with-findings-2026-10-27: 37 explicit section decisions cover the pinned 5256 source lines and bind 17 current destinations to revision-specific explanation reviews; map docs/audit/2026-10-27-nn-section-map.{md,json}; includes worked Winograd, complete wildlife memory/deadline/energy reconstruction, keyword-spotting boundary and all earlier mechanism additions; all 24 native objects have individual decisions; source contradictions retained as findings, not facts; excludes full course, Marimo/hardware execution and deferred EN parity |
| H-V1-07 | `book/quarto/contents/vol1/frameworks/frameworks.qmd` | graphs, autodiff, dispatch, and compilation | yes | integrate | `00 Учебник/10 ML Systems/07 Profiling ML-нагрузки.md` | full original tensor-layout SVG in tensors; original Baydin graph and both full AD trace tables in backpropagation; original CS231n two-layer diagram accompanies the training-step walkthrough; all 21 published objects and 46 native paths inspected, original 3D SVG/caption integrated with worked 32-GPU groups; two unreferenced rasters retained as references only | Thematic review complete with source findings, 28 Oct 2026. All 37 sections map to 14 reviewed chapters. Detailed decisions and evidence: `docs/audit/2026-10-27-frameworks-section-map.md`. Excludes labs/decks, whole-course completion and English parity. |
| H-V1-08 | `book/quarto/contents/vol1/training/training.qmd` | training memory, mixed precision, and data parallelism | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44a Processes, collectives и DDP.md` | original TensorFlow cache/prefetch figures; original GaLoRE algorithms, subspaces and six-bar memory comparison; existing precision and FlashAttention originals retained | Thematic review complete with source findings, 28 Oct 2026. All 32 sections map to 19 reviewed chapters; energy/CO2e maps to the existing [[00 Учебник/18 Evaluation и методология/59a Responsible systems#Экологическая стоимость как системная метрика|Responsible systems explanation]]. Decisions: `docs/audit/2026-10-28-training-reconciliation.md`. Excludes labs/decks, whole-course completion and English parity. |
| H-V1-09 | `book/quarto/contents/vol1/data_selection/data_selection.qmd` | active learning, data selection, and curricula | yes | integrate | `00 Учебник/11 Pre-training и Scaling/41b Отбор обучающих примеров — EL2N, GraNd и forgetting.md` | [EL2N, GraNd и forgetting](../../../textbook/pretraining/data-selection-scores/); [Active learning: querying labels](../../../textbook/pretraining/active-learning/); [Geometric selection](../../../textbook/pretraining/geometric-coresets/); [Semi-supervised learning and FixMatch](../../../textbook/pretraining/semi-supervised-fixmatch/); [SimCLR: two-view pretraining and evaluation](../../../textbook/pretraining/simclr-representations/); [MoCo: queue and momentum encoder](../../../textbook/pretraining/moco-queue-momentum/); [MAE: masked patch reconstruction](../../../textbook/pretraining/mae-masked-autoencoders/); synthetic generation and curricula remain in existing chapter 53. Source lines 1–4741 read; the source dispositions and scoped acceptance evidence follow. New semi-supervised explanation and assigned amortization arithmetic are checked against primary FixMatch/code and graph examples; see `docs/audit/2026-10-28-semi-supervised.md` for acceptance evidence. SimCLR is reconciled against its primary paper and the complete UvA notebook; the test-split checkpoint-selection caveat is explicit. MoCo is reconciled against its primary papers and pinned source implementation, including executed forward/loss/gradient/queue examples and original Figure 1; see `docs/audit/2026-10-28-moco.md`. The MAE reader traces the original patch layout, masking, decoder restoration, masked loss and full-step methods and retains complete saved notebook figures; its local execution scope is explicit. [Selection engineering and evaluation](../../../textbook/pretraining/selection-engineering-evaluation/) now develops the source cost, I/O, distributed-selection, echoing and measurement mechanisms with full worked examples and five unchanged original Harvard web-edition figures. Their current NC-ND license is kept distinct from the archived QMD NC-SA license. [Data augmentation](../../../textbook/neural-networks/data-augmentation/) now develops target semantics, original D2L crops, Mixup/CutMix, AutoAugment/RandAugment, text/speech constraints and TTA. It preserves original source excerpts and three original figures. Back-translation is checked against Sennrich rather than copied as a roundtrip definition; the KWS count distinguishes 500 parents and 50 speakers from 10,000 derived records. See `docs/audit/2026-10-28-data-augmentation.md` for checks and source/version boundaries. Chapter 53 already covers LLM synthetic generation. [Synthetic images and sim-to-real](../../../textbook/pretraining/synthetic-images-sim-to-real/) now develops renderer annotations, camera conventions, domain randomization, covariate-shift weights, DANN/GRL with executed gradient checks, marginal-alignment counterexamples, FID/MMD limits, real-mixture validation and privacy boundaries. It retains seven original BlenderProc/DANN figures and three original code excerpts; the DANN lambda convention is explicit. Decision Framework now retains the original English prerequisite table, original Figure 12 and a worked quality-constrained method choice. Distillation maps to existing chapters 08 and 57b without repeating their accepted explanations. All 14 pinned native files have byte-verified individual visual dispositions in `05 Источники/Originals/data-selection-reconciliation-2026-10-28/native-figure-decisions.json`; covers and unlabelled margin motifs are not counted as new mechanism figures. [GAN and DCGAN](../../../textbook/neural-networks/gan-dcgan/) now develops adversarial objectives, saturation, the idealized discriminator, Gaussian distribution fitting, full source update functions, convolutional shapes, BatchNorm state and unconditional-label limits. It retains eight complete D2L PyTorch bodies, four original SVGs and a full reused author animation; the 20-epoch Gaussian run and image-shaped backward step are executed separately from source plot outputs. See `docs/audit/2026-10-28-gan-dcgan.md`. All 14 generated-QMD figure blocks now have individual dispositions: six keep, four link-only and four omit; 12 original web objects are archived, while two decisions are explicitly source-code-only, not claimed as rendered reviews. A 31-range map covers all 4741 pinned source lines and binds 19 current reader hashes. [Confident Learning and label review](../../../textbook/pretraining/data-selection-scores/#confident-learning) now develops out-of-fold probabilities, class thresholds, raw and calibrated joints, candidate ranking and review, with two original paper figures and three executed examples; see `docs/audit/2026-10-29-confident-learning.md`. [Image duplicate selection](../../../textbook/pretraining/dedup-pii-corpus-quality/#image-deduplication) now develops byte/pixel matching, dHash/pHash collisions and SemDeDup selection with three executable examples and an unchanged original figure. [Cold embedding IDs](../../../textbook/recommender-systems/ranking-neural-models/#cold-embedding-ids) now develops retry deduplication, vocabulary cutoff, OOV and optimizer-state remapping, managed collision ownership and tail evaluation with four executable examples and the original TorchRec diagram. All three recorded reader gaps are closed; see `docs/audit/2026-10-29-selection-gaps.md`. This does not change the explicit link-only/omit decisions or claim acceptance of other course units. See `docs/audit/2026-10-28-data-selection-section-map.json` and `docs/audit/2026-10-28-data-selection-reconciliation.md`. | re-review |
| H-V1-10 | `book/quarto/contents/vol1/model_compression/model_compression.qmd` | pruning, quantization, low-rank methods, and distillation | yes | integrate | `00 Учебник/14 Inference и оптимизация/57 Квантизация языковых моделей.md` | _activation_histogram.svg; _quantization_error.svg; kernel_weights.svg. Partial thematic follow-through: CP, Tucker, factorized convolutions and TT now have detailed explanations, original TensorLy excerpts, five executed examples and the unchanged Harvard tensor-decomposition SVG in chapter 57b. See `docs/audit/2026-10-29-tensor-factorization.md`. Whole-unit section-by-section acceptance is still pending; `audited` below remains the original source-import status. | audited |
| H-V1-11 | `book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd` | GPU architecture, memory hierarchy, and roofline | yes | integrate | `00 Учебник/10 ML Systems/02 GPU, CUDA и иерархия памяти.md` | hw_acceleration_dam_locator.svg; hw_acceleration_energy_ladder.svg; hw_acceleration_roofline_elbow.svg | audited |
| H-V1-12 | `book/quarto/contents/vol1/benchmarking/benchmarking.qmd` | benchmark design, latency distributions, and MLPerf | yes | integrate | `00 Учебник/10 ML Systems/03 Измерение производительности и roofline.md` | benchmarking_confidence_detectability.svg; benchmarking_power_ladder.svg; benchmarking_tail_latency_gap.svg | audited |
| H-V1-13 | `book/quarto/contents/vol1/model_serving/model_serving.qmd` | batching, queueing, TTFT, and TPOT | yes | integrate | `00 Учебник/14 Inference и оптимизация/58c Queueing и capacity planning.md` | model_serving_blast_radius.svg; model_serving_latency_budget_bar.svg; vol1_model_serving_margin_001.svg | audited |
| H-V1-14 | `book/quarto/contents/vol1/ml_ops/ml_ops.qmd` | monitoring, drift, CI/CD, and staged rollout | yes | integrate | `00 Учебник/19 Deployment, Reliability и MLOps/02 MLOps.md` | correction-cascade-chain.svg; ml_ops_drift_threshold_knee.svg; vol1_ml_ops_margin_001.svg | partial-reviewed-2026-10-26 |
| H-V1-15 | `book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd` | fairness, explainability, sustainability, and governance | yes | integrate | `00 Учебник/18 Evaluation и методология/59a Responsible systems.md` | _fairness_threshold.svg; governance_layers.svg; interpretability_spectrum.svg | audited |
| H-V1-16 | `book/quarto/contents/vol1/conclusion/conclusion.qmd` | systems invariants and course synthesis | yes | cross-link | `00 Учебник/10 ML Systems/01 Модель как часть системы.md` | vol1_conclusion_fleet_mtbf_ladder.svg; vol1_conclusion_margin_001.svg; vol1_conclusion_margin_002.svg | audited |
| H-V2-01 | `book/quarto/contents/vol2/introduction/introduction.qmd` | node-to-fleet systems framing | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44h Fault tolerance и fleet orchestration.md` | _ai-triad.svg; _fleet-stack.svg; _roadmap.svg | preview-audited |
| H-V2-02 | `book/quarto/contents/vol2/compute_infrastructure/compute_infrastructure.qmd` | accelerators, HBM, infrastructure stack, and TCO | yes | integrate | `00 Учебник/10 ML Systems/02 GPU, CUDA и иерархия памяти.md` | accelerator-spectrum.svg; bandwidth-hierarchy.svg; compute_infrastructure_cxl_bandwidth_gap.svg | preview-audited |
| H-V2-03 | `book/quarto/contents/vol2/network_fabrics/network_fabrics.qmd` | RDMA, topology, congestion, and bandwidth models | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44g Network, storage и distributed checkpoints.md` | bandwidth-hierarchy.svg; fat-tree-detail.svg; five-level-model.svg | preview-audited |
| H-V2-04 | `book/quarto/contents/vol2/data_storage/data_storage.qmd` | storage hierarchy, GDS, loading, and checkpoint paths | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44g Network, storage и distributed checkpoints.md` | bandwidth-cliffs.svg; checkpoint-staging.svg; complete-data-path.svg | preview-audited |
| H-V2-05 | `book/quarto/contents/vol2/distributed_training/distributed_training.qmd` | DP, TP, PP, ZeRO, and hybrid parallelism | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44c Tensor и sequence parallelism.md` | 3d-parallelism-cube.svg; 3d-parallelism.svg; _zero-partitioning.svg | preview-audited |
| H-V2-06 | `book/quarto/contents/vol2/collective_communication/collective_communication.qmd` | collective primitives, ring and tree algorithms, and overlap | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44a Processes, collectives и DDP.md` | collective-primitives-overview.svg; collective_communication_alpha_beta_dominance.svg; collective_communication_comm_dominance.svg | preview-audited |
| H-V2-07 | `book/quarto/contents/vol2/fault_tolerance/fault_tolerance.qmd` | failure taxonomy, checkpoint intervals, and elastic recovery | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44h Fault tolerance и fleet orchestration.md` | _cicd-pipeline.svg; _error-masking.svg; _hot-spare.svg | preview-audited |
| H-V2-08 | `book/quarto/contents/vol2/fleet_orchestration/fleet_orchestration.qmd` | placement, gang scheduling, Slurm, Kubernetes, and autoscaling | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44h Fault tolerance и fleet orchestration.md` | capacity-reservation.svg; elastic-training-state-machine.svg; fleet_orchestration_bw_hierarchy.svg | preview-audited |
| H-V2-09 | `book/quarto/contents/vol2/performance_engineering/performance_engineering.qmd` | fusion, compilation, CUDA Graphs, memory walls, fleet measurement | yes | cross-link | `00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению.md` | four PyTorch compiler/graph figures; HF bucketing SVG in DDP; two NVIDIA scaling PNGs in S3 (additional destinations documented above); Harvard figure candidates not accepted by these additions | partial-compiler-overlap-graphs-and-measurement-2026-10-26; illustrative 70B case not accepted as benchmark, full lab/deck open |
| H-V2-10 | `book/quarto/contents/vol2/inference/inference.qmd` | KV cache, PagedAttention, routing, and queueing | yes | integrate | `00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention.md` | batch-size-tradeoffs.svg; batching-strategies.svg; continuous-batching.svg | preview-audited |
| H-V2-11 | `book/quarto/contents/vol2/edge_intelligence/edge_intelligence.qmd` | edge deployment and federated learning | yes | cross-link | `00 Учебник/19 Deployment, Reliability и MLOps/03 Edge и federated deployment.md` | _edge-paradigms.svg; _federated-averaging-cycle.svg; _learning-paradigms.svg | preview-audited |
| H-V2-12 | `book/quarto/contents/vol2/ops_scale/ops_scale.qmd` | platform operations, registries, observability, and rollout | yes | integrate | `00 Учебник/19 Deployment, Reliability и MLOps/02 MLOps.md` | _ml-cicd.svg; _mlops-maturity.svg; _model-dependency-graph.svg | partial-reviewed-2026-10-26 |
| H-V2-13 | `book/quarto/contents/vol2/security_privacy/security_privacy.qmd` | poisoning, extraction, privacy attacks, and defenses | yes | integrate | `01 Справочник/Security и Robustness/01 Security и privacy.md` | _enclave.svg; _laser-bitflip.svg; _membership-inference.svg | preview-audited |
| H-V2-14 | `book/quarto/contents/vol2/robust_ai/robust_ai.qmd` | distribution shift, adversarial robustness, and silent corruption | yes | integrate | `01 Справочник/Security и Robustness/02 Robustness.md` | _adversarial-matrix.svg; _autoencoder.svg; _boundary-shift.svg | preview-audited |
| H-V2-15 | `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd` | energy, carbon, water, and cooling | yes | integrate | `00 Учебник/18 Evaluation и методология/59a Responsible systems.md` | _ai-lca.svg; _dc-projections.svg; _energy-roofline.svg | preview-audited |
| H-V2-16 | `book/quarto/contents/vol2/responsible_ai/responsible_ai.qmd` | fairness, interpretability, governance, and feedback loops | yes | integrate | `00 Учебник/18 Evaluation и методология/59a Responsible systems.md` | _fairness-impossibility.svg; _responsible-architecture.svg; bias-loop.svg | preview-audited |
| H-V2-17 | `book/quarto/contents/vol2/conclusion/conclusion.qmd` | fleet-scale synthesis | yes | cross-link | `00 Учебник/11 Pre-training и Scaling/44h Fault tolerance и fleet orchestration.md` | conclusion_tail_latency_rise.svg; fleet-stack-conclusion.svg; vol2_conclusion_margin_001.svg | preview-audited |
| H-L1-00 | `labs/vol1/lab_00_introduction.py` | Marimo investigation for lab_00_introduction | yes | cross-link | `00 Учебник/10 ML Systems/01 Модель как часть системы.md` | none | audited |
| H-L1-01 | `labs/vol1/lab_01_ml_intro.py` | Marimo investigation for lab_01_ml_intro | yes | cross-link | `00 Учебник/10 ML Systems/01 Модель как часть системы.md` | none | audited |
| H-L1-02 | `labs/vol1/lab_02_ml_systems.py` | Marimo investigation for lab_02_ml_systems | yes | cross-link | `00 Учебник/10 ML Systems/01 Модель как часть системы.md` | none | audited |
| H-L1-03 | `labs/vol1/lab_03_ml_workflow.py` | Marimo investigation for lab_03_ml_workflow | yes | cross-link | `00 Учебник/19 Deployment, Reliability и MLOps/01 ML workflow.md` | none | audited |
| H-L1-04 | `labs/vol1/lab_04_data_engr.py` | Marimo investigation for lab_04_data_engr | yes | cross-link | `00 Учебник/11 Pre-training и Scaling/41 Сбор, очистка и смеси данных.md` | none | audited |
| H-L1-05 | `labs/vol1/lab_05_nn_compute.py` | Marimo model for tensor shapes, batch, precision, memory, and resource limits | yes | cross-link | `00 Учебник/10 ML Systems/04 Арифметика Transformer и MoE.md` | none | cross-linked 2026-10-13; runtime not verified |
| H-L1-06 | `labs/vol1/lab_06_nn_arch.py` | Marimo investigation for lab_06_nn_arch | yes | integrate | `06 Практика/18a Сравнить архитектуры по ресурсным ограничениям.md` | original normalized-budget chart, generated from unchanged plotting statements | content-reviewed A–D: 12 candidates and 300 sweeps; Marimo/UI/ledger and actual hardware validation unrun |
| H-L1-07 | `labs/vol1/lab_07_ml_frameworks.py` | Marimo investigation for lab_07_ml_frameworks | yes | cross-link | `00 Учебник/10 ML Systems/07 Profiling ML-нагрузки.md` | none | audited |
| H-L1-08 | `labs/vol1/lab_08_model_train.py` | Marimo investigation for lab_08_model_train | yes | cross-link | `00 Учебник/11 Pre-training и Scaling/44a Processes, collectives и DDP.md` | none | audited |
| H-L1-09 | `labs/vol1/lab_09_data_selection.py` | Marimo investigation for lab_09_data_selection | yes | cross-link | `00 Учебник/11 Pre-training и Scaling/53 Синтетические данные и учебные программы.md` | none | audited |
| H-L1-10 | `labs/vol1/lab_10_model_compress.py` | Marimo investigation for lab_10_model_compress | yes | cross-link | `00 Учебник/14 Inference и оптимизация/57 Квантизация языковых моделей.md` | none | audited |
| H-L1-11 | `labs/vol1/lab_11_hw_accel.py` | Marimo investigation for lab_11_hw_accel | yes | cross-link | `00 Учебник/10 ML Systems/02 GPU, CUDA и иерархия памяти.md` | none | audited |
| H-L1-12 | `labs/vol1/lab_12_perf_bench.py` | Marimo investigation for lab_12_perf_bench | yes | cross-link | `00 Учебник/10 ML Systems/03 Измерение производительности и roofline.md` | none | audited |
| H-L1-13 | `labs/vol1/lab_13_model_serving.py` | Marimo investigation for lab_13_model_serving | yes | cross-link | `00 Учебник/14 Inference и оптимизация/58c Queueing и capacity planning.md` | none | audited |
| H-L1-14 | `labs/vol1/lab_14_ml_ops.py` | Marimo investigation for lab_14_ml_ops | yes | cross-link | `00 Учебник/19 Deployment, Reliability и MLOps/02 MLOps.md` | none | audited |
| H-L1-15 | `labs/vol1/lab_15_responsible_engr.py` | Marimo investigation for lab_15_responsible_engr | yes | cross-link | `00 Учебник/18 Evaluation и методология/59a Responsible systems.md` | none | audited |
| H-L1-16 | `labs/vol1/lab_16_ml_conclusion.py` | Marimo investigation for lab_16_ml_conclusion | yes | cross-link | `00 Учебник/10 ML Systems/01 Модель как часть системы.md` | none | audited |
| H-L2-01 | `labs/vol2/lab_01_introduction.py` | Marimo investigation for lab_01_introduction | yes | cross-link | `00 Учебник/11 Pre-training и Scaling/44h Fault tolerance и fleet orchestration.md` | none | preview-audited |
| H-L2-02 | `labs/vol2/lab_02_compute_infra.py` | Marimo investigation for lab_02_compute_infra | yes | cross-link | `00 Учебник/10 ML Systems/02 GPU, CUDA и иерархия памяти.md` | none | preview-audited |
| H-L2-03 | `labs/vol2/lab_03_communication.py` | Marimo investigation for lab_03_communication | yes | cross-link | `00 Учебник/11 Pre-training и Scaling/44g Network, storage и distributed checkpoints.md` | none | preview-audited |
| H-L2-04 | `labs/vol2/lab_04_data_storage.py` | Marimo investigation for lab_04_data_storage | yes | cross-link | `00 Учебник/11 Pre-training и Scaling/44g Network, storage и distributed checkpoints.md` | none | preview-audited |
| H-L2-05 | `labs/vol2/lab_05_dist_train.py` | Marimo investigation for lab_05_dist_train | yes | cross-link | `00 Учебник/11 Pre-training и Scaling/44c Tensor и sequence parallelism.md` | none | preview-audited |
| H-L2-06 | `labs/vol2/lab_06_collective_communication.py` | Marimo investigation for lab_06_collective_communication | yes | cross-link | `00 Учебник/11 Pre-training и Scaling/44a Processes, collectives и DDP.md` | none | preview-audited |
| H-L2-07 | `labs/vol2/lab_07_fault_tolerance.py` | Marimo investigation for lab_07_fault_tolerance | yes | cross-link | `00 Учебник/11 Pre-training и Scaling/44h Fault tolerance и fleet orchestration.md` | none | preview-audited |
| H-L2-08 | `labs/vol2/lab_08_fleet_orch.py` | Marimo investigation for lab_08_fleet_orch | yes | cross-link | `00 Учебник/11 Pre-training и Scaling/44h Fault tolerance и fleet orchestration.md` | none | preview-audited |
| H-L2-09 | `labs/vol2/lab_09_perf_engineering.py` | bottleneck, regression evidence, capacity and trade-off investigation | yes | integrate-adapted | `06 Практика/13 Оптимизировать один Transformer step.md` | existing original CS336 trace reused, source heuristic graphs not copied | partial-2026-10-26; complete source read, four decisions adapted and eight pure numeric helpers checked; original Marimo UI/report export not reproduced |
| H-L2-10 | `labs/vol2/lab_10_inference.py` | cost crossover, state/KV capacity, batching and feasibility investigation | yes | integrate-adapted | `06 Практика/14 Собрать минимальный inference engine.md` | two complete licensed original HF batching PNGs reused | partial-reviewed-2026-10-26; full source read, four lab stages and cold-start extension explained; five actual CPU fences and source proxies checked, original Marimo runtime and model inference not reproduced |
| H-L2-11 | `labs/vol2/lab_11_edge_intelligence.py` | Marimo investigation for lab_11_edge_intelligence | yes | cross-link | `00 Учебник/19 Deployment, Reliability и MLOps/03 Edge и federated deployment.md` | none | preview-audited |
| H-L2-12 | `labs/vol2/lab_12_ops_scale.py` | Marimo investigation for lab_12_ops_scale | yes | integrate | `06 Практика/17 Развернуть наблюдаемый ML сервис.md` | unchanged SRE per-version canary graph; full source figures retained in original | partial-reviewed: A–D mechanism walkthrough and pure-function execution, linked from MLOps; original Marimo runtime/gates unrun |
| H-L2-13 | `labs/vol2/lab_13_security_privacy.py` | Marimo investigation for lab_13_security_privacy | yes | cross-link | `01 Справочник/Security и Robustness/01 Security и privacy.md` | none | preview-audited |
| H-L2-14 | `labs/vol2/lab_14_robust_ai.py` | Marimo investigation for lab_14_robust_ai | yes | cross-link | `01 Справочник/Security и Robustness/02 Robustness.md` | none | preview-audited |
| H-L2-15 | `labs/vol2/lab_15_sustainable_ai.py` | Marimo investigation for lab_15_sustainable_ai | yes | cross-link | `00 Учебник/18 Evaluation и методология/59a Responsible systems.md` | none | preview-audited |
| H-L2-16 | `labs/vol2/lab_16_responsible_ai.py` | Marimo investigation for lab_16_responsible_ai | yes | cross-link | `00 Учебник/18 Evaluation и методология/59a Responsible systems.md` | none | preview-audited |
| H-L2-17 | `labs/vol2/lab_17_fleet_synthesis.py` | Marimo investigation for lab_17_fleet_synthesis | yes | cross-link | `00 Учебник/11 Pre-training и Scaling/44h Fault tolerance и fleet orchestration.md` | none | preview-audited |
| H-TT-01 | `tinytorch/src/01_tensor/01_tensor.py` | TinyTorch progressive module 01_tensor | yes | cross-link | `00 Учебник/00 Математические и ML-основания/01 Векторы, матрицы и тензоры.md` | none | imported-source |
| H-TT-02 | `tinytorch/src/02_activations/02_activations.py` | TinyTorch progressive module 02_activations | yes | cross-link | `00 Учебник/01 Основы нейронных сетей/02 Функции активации.md` | none | imported-source |
| H-TT-03 | `tinytorch/src/03_layers/03_layers.py` | TinyTorch progressive module 03_layers | yes | cross-link | `00 Учебник/01 Основы нейронных сетей/01 Нейрон и MLP.md` | none | imported-source |
| H-TT-04 | `tinytorch/src/04_losses/04_losses.py` | TinyTorch progressive module 04_losses | yes | cross-link | `00 Учебник/00 Математические и ML-основания/04 Функции потерь.md` | none | imported-source |
| H-TT-05 | `tinytorch/src/05_dataloader/05_dataloader.py` | TinyTorch progressive module 05_dataloader | yes | cross-link | `00 Учебник/10 ML Systems/06 Data pipeline, padding и packing.md` | none | imported-source |
| H-TT-06 | `tinytorch/src/06_autograd/06_autograd.py` | TinyTorch progressive module 06_autograd | yes | cross-link | `00 Учебник/01 Основы нейронных сетей/01 Нейрон, градиент и backpropagation.md` | none | imported-source |
| H-TT-07 | `tinytorch/src/07_optimizers/07_optimizers.py` | TinyTorch progressive module 07_optimizers | yes | cross-link | `00 Учебник/01 Основы нейронных сетей/02 Оптимизация и стабильность обучения.md` | none | imported-source |
| H-TT-08 | `tinytorch/src/08_training/08_training.py` | TinyTorch progressive module 08_training | yes | cross-link | `00 Учебник/11 Pre-training и Scaling/44 Distributed training и mixed precision.md` | none | imported-source |
| H-TT-09 | `tinytorch/src/09_convolutions/09_convolutions.py` | TinyTorch progressive module 09_convolutions | yes | cross-link | `00 Учебник/01 Основы нейронных сетей/07 CNN — от свёртки до ResNet.md` | none | imported-source |
| H-TT-10 | `tinytorch/src/10_tokenization/10_tokenization.py` | TinyTorch progressive module 10_tokenization | yes | cross-link | `00 Учебник/02 Представление текста и токенизация/02 BPE, WordPiece и Unigram.md` | none | imported-source |
| H-TT-11 | `tinytorch/src/11_embeddings/11_embeddings.py` | TinyTorch progressive module 11_embeddings | yes | cross-link | `00 Учебник/02 Представление текста и токенизация/01 От слов к embeddings.md` | none | imported-source |
| H-TT-12 | `tinytorch/src/12_attention/12_attention.py` | TinyTorch progressive module 12_attention | yes | cross-link | `00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V.md` | none | imported-source |
| H-TT-13 | `tinytorch/src/13_transformers/13_transformers.py` | TinyTorch progressive module 13_transformers | yes | cross-link | `00 Учебник/05 Attention и Transformer/03 Полный Transformer.md` | none | imported-source |
| H-TT-14 | `tinytorch/src/14_profiling/14_profiling.py` | TinyTorch progressive module 14_profiling | yes | cross-link | `00 Учебник/10 ML Systems/07 Profiling ML-нагрузки.md` | none | imported-source |
| H-TT-15 | `tinytorch/src/15_quantization/15_quantization.py` | TinyTorch progressive module 15_quantization | yes | cross-link | `00 Учебник/14 Inference и оптимизация/57 Квантизация языковых моделей.md` | none | imported-source |
| H-TT-16 | `tinytorch/src/16_compression/16_compression.py` | TinyTorch progressive module 16_compression | yes | cross-link | `00 Учебник/14 Inference и оптимизация/57 Квантизация языковых моделей.md` | none | imported-source |
| H-TT-17 | `tinytorch/src/17_acceleration/17_acceleration.py` | TinyTorch progressive module 17_acceleration | yes | cross-link | `00 Учебник/10 ML Systems/02 GPU, CUDA и иерархия памяти.md` | none | imported-source |
| H-TT-18 | `tinytorch/src/18_memoization/18_memoization.py` | TinyTorch progressive module 18_memoization | yes | cross-link | `00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention.md` | none | imported-source |
| H-TT-19 | `tinytorch/src/19_benchmarking/19_benchmarking.py` | TinyTorch progressive module 19_benchmarking | yes | cross-link | `00 Учебник/10 ML Systems/03 Измерение производительности и roofline.md` | none | imported-source |
| H-TT-20 | `tinytorch/src/20_capstone/20_capstone.py` | TinyTorch progressive module 20_capstone | yes | cross-link | `06 Практика/13 Оптимизировать один Transformer step.md` | none | imported-source |
| H-SIM-00 | `mlsysim/docs/tutorial/exercises.qmd` | MLSys·im exercise exercises | yes | integrate | `00 Учебник/14 Inference и оптимизация/58c Queueing и capacity planning.md` | none | audited |
| H-SIM-01 | `mlsysim/docs/tutorials/00_hello_roofline.qmd` | MLSys·im exercise 00_hello_roofline | yes | integrate | `00 Учебник/10 ML Systems/03 Измерение производительности и roofline.md` | none | audited |
| H-SIM-02 | `mlsysim/docs/tutorials/01_memory_wall.qmd` | MLSys·im exercise 01_memory_wall | yes | integrate | `00 Учебник/10 ML Systems/03 Измерение производительности и roofline.md` | none | audited |
| H-SIM-03 | `mlsysim/docs/tutorials/01_pipeline_callbacks.qmd` | MLSys·im exercise 01_pipeline_callbacks | yes | integrate | `00 Учебник/10 ML Systems/03 Измерение производительности и roofline.md` | none | audited |
| H-SIM-04 | `mlsysim/docs/tutorials/02_differential_explainer.qmd` | MLSys·im exercise 02_differential_explainer | yes | integrate | `00 Учебник/10 ML Systems/03 Измерение производительности и roofline.md` | none | audited |
| H-SIM-05 | `mlsysim/docs/tutorials/02_two_phases.qmd` | MLSys·im exercise 02_two_phases | yes | integrate | `00 Учебник/10 ML Systems/03 Измерение производительности и roofline.md` | none | audited |
| H-SIM-06 | `mlsysim/docs/tutorials/03_kv_cache.qmd` | MLSys·im exercise 03_kv_cache | yes | integrate | `00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention.md` | none | audited |
| H-SIM-07 | `mlsysim/docs/tutorials/04_starving_the_gpu.qmd` | MLSys·im exercise 04_starving_the_gpu | yes | integrate | `00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention.md` | none | audited |
| H-SIM-08 | `mlsysim/docs/tutorials/05_quantization.qmd` | MLSys·im exercise 05_quantization | yes | integrate | `00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention.md` | none | audited |
| H-SIM-09 | `mlsysim/docs/tutorials/06_scaling_1000_gpus.qmd` | MLSys·im exercise 06_scaling_1000_gpus | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44f Expert и hybrid parallelism.md` | none | audited |
| H-SIM-10 | `mlsysim/docs/tutorials/07_geography.qmd` | MLSys·im exercise 07_geography | yes | integrate | `00 Учебник/14 Inference и оптимизация/58c Queueing и capacity planning.md` | none | audited |
| H-SIM-11 | `mlsysim/docs/tutorials/08_nine_million_dollar.qmd` | MLSys·im exercise 08_nine_million_dollar | yes | integrate | `00 Учебник/14 Inference и оптимизация/58c Queueing и capacity planning.md` | none | audited |
| H-SIM-12 | `mlsysim/docs/tutorials/09_sensitivity.qmd` | MLSys·im exercise 09_sensitivity | yes | integrate | `00 Учебник/14 Inference и оптимизация/58c Queueing и capacity planning.md` | none | audited |
| H-SIM-13 | `mlsysim/docs/tutorials/10_gpu_vs_wafer.qmd` | MLSys·im exercise 10_gpu_vs_wafer | yes | integrate | `00 Учебник/14 Inference и оптимизация/58c Queueing и capacity planning.md` | none | audited |
| H-SIM-14 | `mlsysim/docs/tutorials/11_training_memory_capacity_moe.qmd` | MLSys·im exercise 11_training_memory_capacity_moe | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44f Expert и hybrid parallelism.md` | none | audited |
| H-SIM-15 | `mlsysim/docs/tutorials/12_design_space_exploration.qmd` | MLSys·im exercise 12_design_space_exploration | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44f Expert и hybrid parallelism.md` | none | audited |
| H-SIM-16 | `mlsysim/docs/tutorials/12_full_stack_audit.qmd` | MLSys·im exercise 12_full_stack_audit | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44f Expert и hybrid parallelism.md` | none | audited |
| H-SIM-17 | `mlsysim/docs/tutorials/distributed.qmd` | MLSys·im exercise distributed | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44f Expert и hybrid parallelism.md` | none | audited |
| H-SL1-00 | `slides/vol1/00_course_overview/00_course_overview.tex` | Beamer deck 00_course_overview | yes | integrate | `00 Учебник/10 ML Systems/01 Модель как часть системы.md` | course-roadmap.svg; deployment-spectrum.svg; five-percent-overview.svg | audited |
| H-SL1-01 | `slides/vol1/01_introduction/01_introduction.tex` | Beamer deck 01_introduction | yes | integrate | `00 Учебник/10 ML Systems/01 Модель как часть системы.md` | ai-eras-timeline.svg; amdahls-pipeline.svg; arduino-nicla.svg | audited |
| H-SL1-02 | `slides/vol1/02_ml_systems/02_ml_systems.tex` | Beamer deck 02_ml_systems | yes | integrate | `00 Учебник/10 ML Systems/01 Модель как часть системы.md` | ch02-amdahl-camera-pipeline.svg; ch02-decision-framework.svg; ch02-hybrid-voice-assistant.svg | audited |
| H-SL1-03 | `slides/vol1/03_ml_workflow/03_ml_workflow.tex` | Beamer deck 03_ml_workflow | yes | integrate | `00 Учебник/19 Deployment, Reliability и MLOps/01 ML workflow.md` | ch03-constraint-propagation.svg; ch03-day150-timeline.svg; ch03-dual-pipeline.svg | audited |
| H-SL1-04 | `slides/vol1/04_data_engineering/04_data_engineering.tex` | Beamer deck 04_data_engineering | yes | integrate | `00 Учебник/11 Pre-training и Scaling/41 Сбор, очистка и смеси данных.md` | ch04-data-debt-categories.svg; ch04-dataset-compiler.svg; ch04-drift-types.svg | audited |
| H-SL1-05 | `slides/vol1/05_nn_computation/05_nn_computation.tex` | Forward and Backward Pass; Your Turn: Training Memory Budget | yes | cross-link | `00 Учебник/10 ML Systems/04 Арифметика Transformer и MoE.md` | figures preserved with original slide source; no Harvard slide figure embedded in S4 | cross-linked 2026-10-13; not fully integrated |
| H-SL1-06 | `slides/vol1/06_nn_architectures/06_nn_architectures.tex` | Beamer deck 06_nn_architectures | yes | cross-link | `06 Практика/18a Сравнить архитектуры по ресурсным ограничениям.md` | complete original slide archive retained; no new deck figure substituted for corrected calculations | reviewed with findings: inconsistent arithmetic/universal utilization claims not adopted; source linked for comparison, not blanket integration |
| H-SL1-07 | `slides/vol1/07_frameworks/07_frameworks.tex` | Beamer deck 07_frameworks | yes | integrate | `00 Учебник/10 ML Systems/07 Profiling ML-нагрузки.md` | ch07-autograd-tape.svg; ch07-compilation-continuum.svg; ch07-eager-vs-static-graph.svg | audited |
| H-SL1-08 | `slides/vol1/08_training/08_training.tex` | Beamer deck 08_training | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44a Processes, collectives и DDP.md` | ch08-data-parallelism.svg; ch08-gradient-checkpointing.svg; ch08-iron-law-training.svg | audited |
| H-SL1-09 | `slides/vol1/09_data_selection/09_data_selection.tex` | Beamer deck 09_data_selection | yes | integrate | `00 Учебник/11 Pre-training и Scaling/53 Синтетические данные и учебные программы.md` | ch09-active-learning-loop.svg; ch09-coreset-economics.svg; ch09-curriculum-learning.svg | audited |
| H-SL1-10 | `slides/vol1/10_model_compression/10_model_compression.tex` | Beamer deck 10_model_compression | yes | integrate | `00 Учебник/14 Inference и оптимизация/57 Квантизация языковых моделей.md` | compression-deployment-spectrum.svg; compression-tradeoff-regions.svg; knowledge-distillation-pipeline.svg | audited |
| H-SL1-11 | `slides/vol1/11_hw_acceleration/11_hw_acceleration.tex` | Beamer deck 11_hw_acceleration | yes | integrate | `00 Учебник/10 ML Systems/02 GPU, CUDA и иерархия памяти.md` | accelerator-spectrum.svg; compute-bw-divergence.svg; cpu-vs-gpu-die.svg | audited |
| H-SL1-12 | `slides/vol1/12_benchmarking/12_benchmarking.tex` | Beamer deck 12_benchmarking | yes | integrate | `00 Учебник/10 ML Systems/03 Измерение производительности и roofline.md` | benchmark-anatomy-pipeline.svg; benchmark-evolution-timeline.svg; benchmark-gaming-antipatterns.svg | audited |
| H-SL1-13 | `slides/vol1/13_model_serving/13_model_serving.tex` | Beamer deck 13_model_serving | yes | integrate | `00 Учебник/14 Inference и оптимизация/58c Queueing и capacity planning.md` | batching-comparison.svg; continuous-vs-static-batching.svg; inference-pipeline.svg | audited |
| H-SL1-14 | `slides/vol1/14_ml_ops/14_ml_ops.tex` | Beamer deck 14_ml_ops | yes | integrate | `00 Учебник/19 Deployment, Reliability и MLOps/02 MLOps.md` | canary-deployment.svg; cicd-pipeline-ml.svg; degradation-equation.svg | audited |
| H-SL1-15 | `slides/vol1/15_responsible_engr/15_responsible_engr.tex` | Beamer deck 15_responsible_engr | yes | integrate | `00 Учебник/18 Evaluation и методология/59a Responsible systems.md` | carbon-footprint-tco.svg; confusion-matrix-groups.svg; explainability-spectrum.svg | audited |
| H-SL1-16 | `slides/vol1/16_conclusion/16_conclusion.tex` | Beamer deck 16_conclusion | yes | integrate | `00 Учебник/10 ML Systems/01 Модель как часть системы.md` | conservation-of-complexity.svg; course-journey-timeline.svg; iron-law-retrospective.svg | audited |
| H-SL2-00 | `slides/vol2/00_course_overview/00_course_overview.tex` | Beamer deck 00_course_overview | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44h Fault tolerance и fleet orchestration.md` | c3-taxonomy.svg; course-progression.svg; course-roadmap.svg | preview-audited |
| H-SL2-01 | `slides/vol2/01_introduction/01_introduction.tex` | Beamer deck 01_introduction | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44h Fault tolerance и fleet orchestration.md` | c3-taxonomy-triangle.svg; dam-to-c3.svg; fleet-law-visual.svg | preview-audited |
| H-SL2-02 | `slides/vol2/02_compute_infrastructure/02_compute_infrastructure.tex` | Beamer deck 02_compute_infrastructure | yes | integrate | `00 Учебник/10 ML Systems/02 GPU, CUDA и иерархия памяти.md` | accelerator-spectrum.svg; bandwidth-hierarchy.svg; gpu-evolution.svg | preview-audited |
| H-SL2-03 | `slides/vol2/03_network_fabrics/03_network_fabrics.tex` | Beamer deck 03_network_fabrics | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44g Network, storage и distributed checkpoints.md` | alpha-beta-model.svg; bandwidth-hierarchy.svg; congestion-control.svg | preview-audited |
| H-SL2-04 | `slides/vol2/04_data_storage/04_data_storage.tex` | Beamer deck 04_data_storage | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44g Network, storage и distributed checkpoints.md` | checkpoint-staging.svg; gds-path.svg; pipelined-loading.svg | preview-audited |
| H-SL2-05 | `slides/vol2/05_distributed_training/05_distributed_training.tex` | Beamer deck 05_distributed_training | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44c Tensor и sequence parallelism.md` | 3d-parallelism-cube.svg; data-parallel-flow.svg; hybrid-3d.svg | preview-audited |
| H-SL2-06 | `slides/vol2/06_collective_communication/06_collective_communication.tex` | Beamer deck 06_collective_communication | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44a Processes, collectives и DDP.md` | alpha-beta-model.svg; collective-primitives.svg; compute-comm-timeline.svg | preview-audited |
| H-SL2-07 | `slides/vol2/07_fault_tolerance/07_fault_tolerance.tex` | Beamer deck 07_fault_tolerance | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44h Fault tolerance и fleet orchestration.md` | bathtub-curve.svg; checkpoint-timeline.svg; circuit-breaker-states.svg | preview-audited |
| H-SL2-08 | `slides/vol2/08_fleet_orchestration/08_fleet_orchestration.tex` | Beamer deck 08_fleet_orchestration | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44h Fault tolerance и fleet orchestration.md` | cost-optimization.svg; elastic-training.svg; fragmentation-problem.svg | preview-audited |
| H-SL2-09 | `slides/vol2/09_performance_engineering/09_performance_engineering.tex` | Beamer deck 09_performance_engineering | yes | cross-link | `00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению.md` | nine archival SVG candidates inspected; MBU destination in chapter 55a uses two complete original JAX PNGs | partial-reviewed-2026-10-26; full deck/notes read, practice and MBU destinations added; unsupported universal claims not copied, full thematic transfer remains open |
| H-SL2-10 | `slides/vol2/10_inference/10_inference.tex` | Beamer deck 10_inference | yes | integrate-adapted | `06 Практика/14 Собрать минимальный inference engine.md` | source HF alternatives; native Harvard percentage/timeline figures reviewed but not accepted as quantitative evidence | partial-reviewed-2026-10-26; full deck/notes read; cost, GQA exercise and cold-start mechanism added; unsupported utilization, exact-P99 and reasoning-bottleneck claims not copied, full thematic integration remains open |
| H-SL2-11 | `slides/vol2/11_edge_intelligence/11_edge_intelligence.tex` | Beamer deck 11_edge_intelligence | yes | integrate | `00 Учебник/19 Deployment, Reliability и MLOps/03 Edge и federated deployment.md` | adaptation-strategies.svg; centralized-vs-decentralized.svg; cloud-vs-edge-learning.svg | preview-audited |
| H-SL2-12 | `slides/vol2/12_ops_scale/12_ops_scale.tex` | Beamer deck 12_ops_scale | yes | integrate | `00 Учебник/19 Deployment, Reliability и MLOps/02 MLOps.md` | fleet-alert arithmetic + unchanged Prometheus original; staged-rollout/monitoring-pyramid not used, reasons recorded | partial-reviewed: full source read, selected mechanisms integrated, worked cases also in practice 17; other deck topics retain prior chapter mappings, not a blanket completion |
| H-SL2-13 | `slides/vol2/13_security_privacy/13_security_privacy.tex` | Beamer deck 13_security_privacy | yes | integrate | `01 Справочник/Security и Robustness/01 Security и privacy.md` | attack-surface-layers.svg; data-poisoning.svg; defense-stack.svg | preview-audited |
| H-SL2-14 | `slides/vol2/14_robust_ai/14_robust_ai.tex` | Beamer deck 14_robust_ai | yes | integrate | `01 Справочник/Security и Robustness/02 Robustness.md` | adversarial-attack-fgsm.svg; data-poisoning.svg; defense-layers.svg | preview-audited |
| H-SL2-15 | `slides/vol2/15_sustainable_ai/15_sustainable_ai.tex` | Beamer deck 15_sustainable_ai | yes | integrate | `00 Учебник/18 Evaluation и методология/59a Responsible systems.md` | carbon-geography.svg; carbon-lifecycle.svg; energy-wall.svg | preview-audited |
| H-SL2-16 | `slides/vol2/16_responsible_ai/16_responsible_ai.tex` | Beamer deck 16_responsible_ai | yes | integrate | `00 Учебник/18 Evaluation и методология/59a Responsible systems.md` | explainability-methods.svg; fairness-metrics-comparison.svg; fairness-tax-visual.svg | preview-audited |
| H-SL2-17 | `slides/vol2/17_conclusion/17_conclusion.tex` | Beamer deck 17_conclusion | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44h Fault tolerance и fleet orchestration.md` | competency-framework.svg; compound-ai-systems.svg; fermi-brain-vs-cluster.svg | preview-audited |
| H-F1-01 | `book/quarto/contents/vol1/introduction/images/` | Native figure directory for DAM, Iron Law, energy hierarchy | yes | integrate | `00 Учебник/10 ML Systems/01 Модель как часть системы.md` | _system-map.svg; introduction_energy_hierarchy.svg; introduction_iron_law_bars.svg | audited |
| H-F1-02 | `book/quarto/contents/vol1/ml_systems/images/` | Native figure directory for workload characteristics and systems constraints | yes | integrate | `00 Учебник/10 ML Systems/01 Модель как часть системы.md` | _edge_ml_iot.svg; ml_systems_dam_locator.svg; ml_systems_deployment_span.svg | audited |
| H-F1-03 | `book/quarto/contents/vol1/ml_workflow/images/` | Native figure directory for ML lifecycle and constraint propagation | yes | integrate | `00 Учебник/19 Deployment, Reliability и MLOps/01 ML workflow.md` | ml_workflow_constraint_cost_escalation.svg; ml_workflow_feedback_timescales.svg; vol1_ml_workflow_margin_001.svg | audited |
| H-F1-04 | `book/quarto/contents/vol1/data_engineering/images/` | Native figure directory for data quality, debt, storage, and pipelines | yes | integrate | `00 Учебник/11 Pre-training и Scaling/41 Сбор, очистка и смеси данных.md` | _kws_multilingual.svg; _kws_pipeline.svg; data_engineering_active_learning_budget.svg | audited |
| H-F1-05 | `book/quarto/contents/vol1/nn_computation/images/` | Original figure collection retained with Neural Computation | available in source | source-only | `05 Источники/Courses/Harvard ML Systems/vol1/nn_computation.md` | source inventory, not S4 embeds: _activity_classification.svg; _bio_nn2ai_nn.svg; ai_dl_progress.svg | preserved in source; no claim of full figure integration |
| H-F1-06 | `book/quarto/contents/vol1/nn_architectures/images/` | Native figures for architecture mechanisms and bottlenecks | yes | integrate | `00 Учебник/01 Основы нейронных сетей/07 CNN — от свёртки до ResNet.md` | complete original im2row SVG with corrected explanatory caption; complete archived selection SVG retained in practice 18a; all 24 native objects retained in source archive | native-disposition-complete-with-findings-2026-10-27: exact 24-object decisions in docs/audit/2026-10-27-nn-native-figures.json; no blanket promotion of wrong wiring, unqualified memory/intensity claims or duplicate illustrations; not whole-course completion |
| H-F1-07 | `book/quarto/contents/vol1/frameworks/images/` | Native figure directory plus published graph/layout diagrams | yes | integrate | `00 Учебник/00 Математические и ML-основания/01 Векторы, матрицы и тензоры.md` | 46 pinned native paths / 43 distinct native byte objects; 21 published objects; union 56 distinct objects; full generated tensor-memory-layout SVG integrated without redrawing | Individual figure review complete, 28 Oct 2026. Tensor layout and 3D placement integrated; FL/transfer rasters reference-only, third-party permission unverified. All 56 object decisions: `docs/audit/2026-10-27-frameworks-native-figures.json`. |
| H-F1-08 | `book/quarto/contents/vol1/training/images/` | Native figure directory for training memory, mixed precision, and data parallelism | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44a Processes, collectives и DDP.md` | 33 pinned native paths / 32 objects; 27 separately archived rendered objects; 50-object union individually reviewed | Individual-object review complete with source findings, 28 Oct 2026. Primary GaLoRE six-bar SVG selected over the simplified redraw/JPEG; decorative, misleading and unverified third-party media retained as references. Decisions: `docs/audit/2026-10-28-training-figures.json`. |
| H-F1-09 | `book/quarto/contents/vol1/data_selection/images/` | Native figure directory for active learning, data selection, and curricula | yes | integrate | `00 Учебник/11 Pre-training и Scaling/53 Синтетические данные и учебные программы.md` | Pinned directory listing and all original bytes independently checked: six SVG, six PNG, two WebP files. Five complete original web-edition figures are retained in selection engineering with source bytes/page snapshots and the actual NC-ND notice; these do not substitute for a complete native directory review. The tiny echo-threshold margin SVG was inspected and omitted from the reader because its axes are unlabeled. The source marginal-domain-gap sketch is not adopted as evidence for decision boundaries: unconditional densities alone cannot establish the target label rule. Seven primary BlenderProc/DANN originals now explain scene labels and adaptation in the sim-to-real reader; they are not counted as completion of the archived native-directory inventory. All 14 native files are individually visually reviewed with exact pinned Git blob and SHA-256 checks (6,684,684 bytes total). The cover variants and six margin SVGs are omitted from new reader placement for recorded explanatory reasons; the historical/extrapolation PNG is link-only; the existing labelled-cover source-page embed is preserved. This closes the native-directory disposition only. The GAN gap now has a full thematic reader with four unchanged D2L SVGs and one reused 25-frame transposed-convolution animation; these are separately sourced additions, not replacements for unreviewed Harvard objects. The separate generated-QMD inventory is now complete as a disposition record: six keep, four link-only, four omit across all 14 blocks. Twelve original web objects are archived; two absent current-web objects have code/semantic decisions, not claimed rendered inspection. The optimization stack is retained in 41i and pretraining amortization in 41e, both unchanged originals. This closes figure disposition, not the three remaining reader mechanisms listed in `docs/audit/2026-10-28-data-selection-section-map.json`; see `docs/audit/2026-10-28-data-selection-reconciliation.md`. | re-review |
| H-F1-10 | `book/quarto/contents/vol1/model_compression/images/` | Native figure directory for pruning, quantization, low-rank methods, and distillation | yes | integrate | `00 Учебник/14 Inference и оптимизация/57 Квантизация языковых моделей.md` | _activation_histogram.svg; _quantization_error.svg; kernel_weights.svg | audited |
| H-F1-11 | `book/quarto/contents/vol1/hw_acceleration/images/` | Native figure directory for GPU architecture, memory hierarchy, and roofline | yes | integrate | `00 Учебник/10 ML Systems/02 GPU, CUDA и иерархия памяти.md` | hw_acceleration_dam_locator.svg; hw_acceleration_energy_ladder.svg; hw_acceleration_roofline_elbow.svg | audited |
| H-F1-12 | `book/quarto/contents/vol1/benchmarking/images/` | Native figure directory for benchmark design, latency distributions, and MLPerf | yes | integrate | `00 Учебник/10 ML Systems/03 Измерение производительности и roofline.md` | benchmarking_confidence_detectability.svg; benchmarking_power_ladder.svg; benchmarking_tail_latency_gap.svg | audited |
| H-F1-13 | `book/quarto/contents/vol1/model_serving/images/` | Native figure directory for batching, queueing, TTFT, and TPOT | yes | integrate | `00 Учебник/14 Inference и оптимизация/58c Queueing и capacity planning.md` | model_serving_blast_radius.svg; model_serving_latency_budget_bar.svg; vol1_model_serving_margin_001.svg | audited |
| H-F1-14 | `book/quarto/contents/vol1/ml_ops/images/` | Native figure directory for monitoring, drift, CI/CD, and staged rollout | yes | integrate | `00 Учебник/19 Deployment, Reliability и MLOps/02 MLOps.md` | correction-cascade-chain.svg; ml_ops_drift_threshold_knee.svg; vol1_ml_ops_margin_001.svg | audited |
| H-F1-15 | `book/quarto/contents/vol1/responsible_engr/images/` | Native figure directory for fairness, explainability, sustainability, and governance | yes | integrate | `00 Учебник/18 Evaluation и методология/59a Responsible systems.md` | _fairness_threshold.svg; governance_layers.svg; interpretability_spectrum.svg | audited |
| H-F1-16 | `book/quarto/contents/vol1/conclusion/images/` | Native figure directory for systems invariants and course synthesis | yes | integrate | `00 Учебник/10 ML Systems/01 Модель как часть системы.md` | vol1_conclusion_fleet_mtbf_ladder.svg; vol1_conclusion_margin_001.svg; vol1_conclusion_margin_002.svg | audited |
| H-F2-01 | `book/quarto/contents/vol2/introduction/images/` | Native figure directory for node-to-fleet systems framing | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44h Fault tolerance и fleet orchestration.md` | _ai-triad.svg; _fleet-stack.svg; _roadmap.svg | preview-audited |
| H-F2-02 | `book/quarto/contents/vol2/compute_infrastructure/images/` | Native figure directory for accelerators, HBM, infrastructure stack, and TCO | yes | integrate | `00 Учебник/10 ML Systems/02 GPU, CUDA и иерархия памяти.md` | accelerator-spectrum.svg; bandwidth-hierarchy.svg; compute_infrastructure_cxl_bandwidth_gap.svg | preview-audited |
| H-F2-03 | `book/quarto/contents/vol2/network_fabrics/images/` | Native figure directory for RDMA, topology, congestion, and bandwidth models | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44g Network, storage и distributed checkpoints.md` | bandwidth-hierarchy.svg; fat-tree-detail.svg; five-level-model.svg | preview-audited |
| H-F2-04 | `book/quarto/contents/vol2/data_storage/images/` | Native figure directory for storage hierarchy, GDS, loading, and checkpoint paths | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44g Network, storage и distributed checkpoints.md` | bandwidth-cliffs.svg; checkpoint-staging.svg; complete-data-path.svg | preview-audited |
| H-F2-05 | `book/quarto/contents/vol2/distributed_training/images/` | Native figure directory for DP, TP, PP, ZeRO, and hybrid parallelism | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44c Tensor и sequence parallelism.md` | 3d-parallelism-cube.svg; 3d-parallelism.svg; _zero-partitioning.svg | preview-audited |
| H-F2-06 | `book/quarto/contents/vol2/collective_communication/images/` | Native figure directory for collective primitives, ring and tree algorithms, and overlap | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44a Processes, collectives и DDP.md` | collective-primitives-overview.svg; collective_communication_alpha_beta_dominance.svg; collective_communication_comm_dominance.svg | preview-audited |
| H-F2-07 | `book/quarto/contents/vol2/fault_tolerance/images/` | Native figure directory for failure taxonomy, checkpoint intervals, and elastic recovery | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44h Fault tolerance и fleet orchestration.md` | _cicd-pipeline.svg; _error-masking.svg; _hot-spare.svg | preview-audited |
| H-F2-08 | `book/quarto/contents/vol2/fleet_orchestration/images/` | Native figure directory for placement, gang scheduling, Slurm, Kubernetes, and autoscaling | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44h Fault tolerance и fleet orchestration.md` | capacity-reservation.svg; elastic-training-state-machine.svg; fleet_orchestration_bw_hierarchy.svg | preview-audited |
| H-F2-09 | `book/quarto/contents/vol2/performance_engineering/images/` | Native figure directory for fusion, compilation, CUDA Graphs, and memory walls | yes | re-review | `00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению.md` | block-quantization.svg; compilation-pipeline.svg; diagnostic-flow.svg | candidate-remapped-2026-10-26; not copied into canonical S8 |
| H-F2-10 | `book/quarto/contents/vol2/inference/images/` | Native figure directory for KV cache, PagedAttention, routing, and queueing | yes | integrate | `00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention.md` | batch-size-tradeoffs.svg; batching-strategies.svg; continuous-batching.svg | preview-audited |
| H-F2-11 | `book/quarto/contents/vol2/edge_intelligence/images/` | Native figure directory for edge deployment and federated learning | yes | integrate | `00 Учебник/19 Deployment, Reliability и MLOps/03 Edge и federated deployment.md` | _edge-paradigms.svg; _federated-averaging-cycle.svg; _learning-paradigms.svg | preview-audited |
| H-F2-12 | `book/quarto/contents/vol2/ops_scale/images/` | Native figure directory for platform operations, registries, observability, and rollout | yes | integrate | `00 Учебник/19 Deployment, Reliability и MLOps/02 MLOps.md` | time-travel.svg; _ml-cicd.svg; _mlops-maturity.svg; _model-dependency-graph.svg | partial-reviewed-2026-10-26 |
| H-F2-13 | `book/quarto/contents/vol2/security_privacy/images/` | Native figure directory for poisoning, extraction, privacy attacks, and defenses | yes | integrate | `01 Справочник/Security и Robustness/01 Security и privacy.md` | _enclave.svg; _laser-bitflip.svg; _membership-inference.svg | preview-audited |
| H-F2-14 | `book/quarto/contents/vol2/robust_ai/images/` | Native figure directory for distribution shift, adversarial robustness, and silent corruption | yes | integrate | `01 Справочник/Security и Robustness/02 Robustness.md` | _adversarial-matrix.svg; _autoencoder.svg; _boundary-shift.svg | preview-audited |
| H-F2-15 | `book/quarto/contents/vol2/sustainable_ai/images/` | Native figure directory for energy, carbon, water, and cooling | yes | integrate | `00 Учебник/18 Evaluation и методология/59a Responsible systems.md` | _ai-lca.svg; _dc-projections.svg; _energy-roofline.svg | preview-audited |
| H-F2-16 | `book/quarto/contents/vol2/responsible_ai/images/` | Native figure directory for fairness, interpretability, governance, and feedback loops | yes | integrate | `00 Учебник/18 Evaluation и методология/59a Responsible systems.md` | _fairness-impossibility.svg; _responsible-architecture.svg; bias-loop.svg | preview-audited |
| H-F2-17 | `book/quarto/contents/vol2/conclusion/images/` | Native figure directory for fleet-scale synthesis | yes | integrate | `00 Учебник/11 Pre-training и Scaling/44h Fault tolerance и fleet orchestration.md` | conclusion_tail_latency_rise.svg; fleet-stack-conclusion.svg; vol2_conclusion_margin_001.svg | preview-audited |

## S4 cross-link evidence — 2026-10-13

For `H-V1-05`, S4 links to the full **Neural Computation** chapter and directly
to *Model size and computational complexity*. Its small MLP example is
recomputed in the training-state section: 109,386 parameters, 437,544 bytes
for FP32 weights, and 1,750,176 bytes for weights, gradients, and two FP32
Adam moments. The scope of the factor-four comparison is stated explicitly;
it is not a universal training-to-inference memory ratio.

For `H-L1-05`, the source/practice list links to the complete lab and gives a
specific `cloud_fleet` route through tensor shape, batch, and precision.
The text distinguishes model estimates from hardware measurements and
links to the existing Marimo execution instructions. Reading its code does
not establish that the interactive application was executed.

For `H-SL1-05`, S4 links to the preserved slide source and names two useful
sections. This is a cross-link, not integration of the entire deck or its
figures. `H-F1-05` is source-only: S4 currently illustrates memory accounting
and distributed execution with selected JAX, Hugging Face, NeMo, and EDLS
figures. The existence of those illustrations does not prove transfer of
Harvard's figure directory.

The book chapter is pinned to `2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd`;
the imported lab and slides are pinned to
`45ecc8d82fcae70c149cdce550d3b3d3411df913`, as their frontmatter and
`labs-slides-manifest.json` record. They are not one identical source snapshot.

Literal rows: 173.
