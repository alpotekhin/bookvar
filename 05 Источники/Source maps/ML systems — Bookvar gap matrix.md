---
title: ML systems — Bookvar gap matrix
type: source-note
status: editorial
last_updated: 2026-10-29
---

# ML systems — Bookvar gap matrix

<!-- BEGIN CURRENT RECONCILIATION 2026-10-29 -->
## Актуальные результаты и остатки — 29 октября 2026

Состояния в исторической таблице ниже не переписаны задним числом. Текущий
результат определяется этой сверкой и отчётами по конкретной редакции,
а не прежним словом `integrated`.

| Область | Что теперь проверено | Что этим не закрыто |
|---|---|---|
| Model Compression | Полное содержательное сопоставление 80 разделов закреплённого Harvard QMD; дополнены pruning/reset, NAS, early exits, compound scaling, BN folding и binary arithmetic. Отчёт: `docs/audit/2026-10-29-systems-completion.md`; карта: `docs/audit/2026-10-29-compression-source-section-map.json`. | Измеренная точность сжатой обученной модели, GPU-ускорение, запуск NAS; полная приёмка всех lab/deck объектов. |
| Hardware Acceleration | 113 секционных решений и 14 решений по фигурам; CPU-примеры, исходные TPU-анимации и systolic SVG. Отчёт: `docs/audit/2026-10-29-hardware-source-comparison.md` и `.json`. | Реальное выполнение CUDA/TPU, page-migration/energy замеры, автоматически предполагаемая поддержка всех операций на NPU. |
| Benchmarking, Model Serving, MLOps | Все три опубликованных объяснения прочитаны полностью: 286 разделов, 59 тематических решений, ни одной outline-only записи. Шесть выявленных пробелов закрыты: энергия, fan-out/hedging, ML Test Score, retraining economics, A/B sample/time и NUMA/thread/mmap. Сохранены три оригинальных рисунка; CPU-примеры и прежний текст проверены. Отчёт: `docs/audit/2026-10-29-serving-full-source-reconciliation.md`; карта: `docs/audit/2026-10-29-serving-source-section-map.json`. | Три приложения с ответами исключены из переноса; исторические кейсы и списки suites остаются контекстом источника, неподтверждённые обобщения не приняты. Production/MLPerf/физические power measurements и NUMA/GPU benchmark не выполнялись. |
| Marimo | Все 34 начальных состояния исполнились с согласованными зависимостями; V2 Lab 10 пройден в UI до A–D и сохранённого отчёта. | Полное прохождение остальных 33 UI-сценариев; учебные вычисления не являются hardware benchmark. |
| Ring/DDP | 24 конфигурации, 72 результата rank через настоящие локальные Gloo-процессы; DDP-градиент и SGD совпадают с global-batch reference. | CUDA/NCCL, межузловая сеть, bandwidth/overlap измерения. |
| Минимальный inference engine | WordLevel tokenizer, tiny random Qwen3 checkpoint, offline reload и cached/uncached сравнение на CPU/MPS; прежние 54 CPU-запуска и HTTP/NDJSON сохранены. | Pretrained-model качество, physical peak memory и GPU SLO-capacity. |
| Наблюдаемый сервис | Реальный Docker build, HTTP/gRPC, readiness, ошибочные входы, HTTP overload, Prometheus scrape и 620 open-loop поступлений. | gRPC overload-исход не проверен; RGB fixture и заданные 20 ms не подменяют VGG/LLM-производительность, production-нагрузку или полноценный tracing. |
| English | Все 29 существующих EN-файлов содержательно сопоставлены с полными каноническими текстами и имеют status current по точному SHA. Сохранены 183 размещения исходных рисунков, исполнены 176 CPU-блоков; все страницы проверены при 360 и 1280 px. | Это 28 глав и оглавление, не перевод всех 149 страниц. Отсутствующие EN-главы остаются явно обозначенными canonical-language fallback. |

Доказательства выполнения практики: `docs/audit/2026-10-29-practice-completion.md`
и `06 Практика/Results/runtime-2026-10-29/`. В частности, старые остатки
«text/tokenizer/checkpoint», «container build», «actual scrape/gRPC/load» и
«Marimo runtime» теперь разделены на выполненные локальные проверки и
непроверенные аппаратные/интерактивные условия, а не оставлены одним пунктом.
CUDA/NCCL команды завершаются до запуска и создания результатов, если
возможности хоста не подходят.

Точный состав EN и история закрытых partial: `docs/audit/2026-10-29-english-reconciliation.md` и `.json`.
Точечная сверка CS336: `docs/audit/2026-10-29-cs336-residuals.md` — V4 CSA/HCA,
пять ARC/AIR/GCG иллюстраций, два уже раскрытых механизма Lecture 02 и два
повторных изображения. Это не приёмка всего CS336 или дополнительных Harvard
units. Проверка двух старых CS231n-слайдов теперь подтверждена совпадением
пикселей с официальным PDF: `docs/audit/2026-10-29-cs231n-source-pixels.json`;
она не создаёт новую лицензию на изображения.

Остальные units Harvard/HSE/EDLS без отдельной содержательной приёмки
сохраняют прежние ограничения. Новые темы этим обновлением карты не добавлены.
<!-- END CURRENT RECONCILIATION 2026-10-29 -->

## Историческая карта размещения

Эта таблица фиксирует размещение материалов. Историческая метка `integrated`
означает только структурную связь: страница существует, ссылается на локальный
оригинал и включена в карту назначения. Она не подтверждает полноту переноса
знаний или текущую приёмку объяснения. Для этого нужны постраничное чтение,
проверка примеров и рисунков и отчёт по конкретной редакции в `docs/audit/`.

| Bookvar concept | Canonical destination | Best source IDs | State |
|---|---|---|---|
| System framing | `00 Учебник/10 ML Systems/01 Модель как часть системы.md` | H-V1-01, H-V1-02, H-SL1-01 | integrated |
| GPU and CUDA | `00 Учебник/10 ML Systems/02 GPU, CUDA и иерархия памяти.md`; `06 Практика/06 Измерить CUDA без самообмана.md` | H-V1-11, H-V2-02, E-01-L, E-01-S | integrated |
| Measurement and roofline | `00 Учебник/10 ML Systems/03 Измерение производительности и roofline.md`; chapter 55a | H-V1-12, H-SIM-01, H-SIM-02, E-01-S | integrated |
| Transformer and MoE arithmetic | `00 Учебник/10 ML Systems/04 Арифметика Transformer и MoE.md` | H-V1-05, E-06-L | integrated |
| Numerical formats | `00 Учебник/10 ML Systems/05 Численные форматы и mixed precision.md` | H-V1-08, E-02-L, E-02-A1 | integrated |
| Input pipeline | `00 Учебник/10 ML Systems/06 Data pipeline, padding и packing.md`; `06 Практика/08 Ускорить data pipeline.md` | H-V1-04, E-02-S, E-02-H1, E-02-H2 | integrated |
| Profiling | `00 Учебник/10 ML Systems/07 Profiling ML-нагрузки.md` | H-V1-07, E-02-H3, E-06-S | integrated |
| Data parallelism and collectives | chapter 44a; practice 09 | H-V2-06, H-V2-09 overlap comparison, E-03-L, E-03-S, E-03-H; versioned PyTorch docs | partial; accepted DDP explanation preserved and readiness/stream-lifetime/DDPOptimizer material added 2026-10-26; full course and laboratory transfer remains open |
| Checkpointing and offload | chapter 44b; practice 10 | E-04-L, E-04-S1 | integrated |
| Tensor, sequence, and pipeline parallelism | chapters 44c–44d; practice 11 | H-V2-05, H-SL2-05, E-04-L, E-04-S2 | integrated |
| ZeRO and FSDP | chapter 44e; practice 12 | E-05-L, E-05-S, E-05-H | integrated |
| Expert and hybrid parallelism | chapter 44f and the MoE architecture chapter | H-V2-05, H-SIM-14, E-06-L | integrated |
| Networks, storage, and checkpoints | chapter 44g | H-V2-03, H-V2-04, H-F2-03, H-F2-04 | integrated |
| Fault tolerance and orchestration | chapter 44h | H-V2-07, H-V2-08, H-SL2-07, H-SL2-08 | integrated |
| KV cache, scheduling, and mini-engine | chapters 55–55c; practices 14 and 14a | H-V2-10, H-L2-10, H-SL2-10, E-08-L, E-08-S, E-08-H | partial; neural core/policies and 54 finite CPU runs integrated; bounded token-ID HTTP/NDJSON lifecycle tested over actual loopback TCP 2026-10-26; original Marimo/text-tokenizer/checkpoint contract, deployment metrics, physical peak memory and GPU/SLO-capacity remain open |
| Kernels, compilation and CUDA Graphs | S8 GPU kernels and Triton; DDP chapter 44a; S3 measurement; inference physics 55a; serving chapter 55c; practice 13 | H-V2-09, H-L2-09, H-SL2-09, E-06-S, E-06-A; versioned PyTorch CUDA/compiler documentation; JAX Scaling Book | partial; compiler/graphs/overlap/fleet measurement integrated; complete lab/deck read, four-stage evidence practice and MBU explanation added 2026-10-26; original Marimo runtime not reproduced, staged 70B numbers not accepted as benchmark, full course transfer remains open |
| Quantization and KV compression | chapters 57–57a; practice 15 | H-V1-10, H-SIM-06, E-09-L, E-09-S | integrated |
| General model compression | chapter 57b; TinyTorch 16 | H-V1-10, H-L1-10, H-SL1-10 | integrated |
| Speculative decoding | chapter 58; practice 16 | E-09-L, E-09-H | integrated |
| Queueing, SLO, and capacity | chapters 58b–58c | H-V1-13, H-V2-10, H-SIM-10, E-08-S | integrated |
| Workflow, deployment, and MLOps | deployment chapters 01–03; practice 17 | H-V1-03, H-V1-14, H-V2-11, H-V2-12, H-L2-11, H-L2-12, H-SL2-12, E-07 | partial; EDLS Week 7 walkthrough, workflow iteration/feedback, feature-history/lineage/backfill, resource-policy/tenant-isolation, FinOps/TCO and MIG explanation/visuals added 2026-10-26; Lab 12 A–D arithmetic integrated into practice 17, eight pure source functions executed without Marimo/ledger, missing canary deadline and synthetic coefficients explained; fleet false-crossing expectation distinguished from independent-test probability and notifications; unchanged Prometheus/SRE figures reused, contradictory rollout/pyramid candidates rejected; container builds, actual scrapes/gRPC/load experiment, original lab UI/gates and full Harvard chapter transfer remain open |
| Security, robustness, responsibility, and sustainability | chapter 59a; two Security and Robustness reference pages | H-V1-15, H-V2-13, H-V2-14, H-V2-15, H-V2-16 | integrated |

Автоматическая проверка выполняется командой
`pnpm --dir publishing check:source-coverage`. Она не оценивает качество прозы,
и проверяет существование destination и его ссылки на источник. Она не
устанавливает, что все объяснения исходного курса встроены в учебник.

26 октября 2026: в Edge встроен полный английский Math Peek о памяти
из H-L2-11, с явными допущениями FP16/FP32, расчётом 120/152/192 MB
и оговоркой о неверной нижней границе 5× в оригинале. Исправлено статическое
извлечение текста всех 34 Marimo labs; их исходный код не изменён.
Это не означает выполнения лабораторий, переноса HTML-интерфейсов и графиков
или тематической интеграции всех 35 Beamer decks. Подробности:
`docs/audit/2026-10-26-harvard-lab-extraction.md`.

В архивном слое этих 35 decks восстановлены формулы, таблицы, 269 исходных
SVG и 35 отдельных слайдов с уравнениями (`mlsysfocus`); заметки преподавателя
отделены от текста слайдов. Это исправление источников. Статусы тематического
переноса выше не повышены: код слайдов не исполнялся, а конкретные утверждения
и иллюстрации по-прежнему требуют сопоставления с учебными главами.
