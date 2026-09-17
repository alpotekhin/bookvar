---
title: Практика
type: practice
status: active
last_updated: 2026-09-15
---

# Практика

Практические работы превращают формулу или архитектурную схему в наблюдаемое
поведение. Для каждой работы указан конкретный результат: реализация,
измерение, трассировка чужого кода или набор проверяемых утверждений.

Здесь есть короткие запускаемые примеры и самостоятельные проекты. Работы
20–24 опираются на задания CS336, а 25–27 предлагают сравнительные эксперименты
по агентам; их полная реализация существенно больше одного небольшого упражнения.
Где подготовлен минимальный пример, он выделен в начале работы. Его успешный
запуск не означает выполнения всех требований проекта. Универсальной оценки
в часах нет: время зависит от подготовки, оборудования и выбранного объёма.

| Практика | Предпосылки | Результат |
|---|---|---|
| [[02 Areas/ML & DL/06 Практика/01 Собрать micrograd и bigram LM|Собрать micrograd и биграммную модель]] | Python, производная | Собственный скалярный autograd и минимальная языковая модель |
| [[02 Areas/ML & DL/06 Практика/02 Реализовать causal self-attention|Причинное self-attention]] | Матрицы, softmax | Интерактивная проверка маски и реализация на PyTorch |
| [[02 Areas/ML & DL/06 Практика/03 Прочитать современный decoder в LitGPT|Прочитать декодер LitGPT]] | Transformer block | Трасса тензоров от embedding до logits |
| [[02 Areas/ML & DL/06 Практика/04 Проследить RLVR pipeline в Open-R1|Проследить RLVR в Open-R1]] | SFT, policy gradient | Карта данных, генерации, verifier и обновления policy |
| [[02 Areas/ML & DL/06 Практика/05 Измерить KV-cache и quantization|Измерить KV-cache и квантование]] | Inference basics | Таблица памяти, скорости и ошибки квантования |
| [[02 Areas/ML & DL/06 Практика/06 Измерить CUDA без самообмана|Измерить CUDA без самообмана]] | Python, PyTorch, CUDA | Воспроизводимый benchmark и profiler trace |
| [[02 Areas/ML & DL/06 Практика/06a Проверить mixed precision и loss scaling|Проверить mixed precision и loss scaling]] | PyTorch, CUDA, численные форматы | Trace, memory/accuracy table и воспроизводимый underflow case |
| [[02 Areas/ML & DL/06 Практика/07 Построить roofline и найти bottleneck|Построить roofline]] | FLOP, bytes, CUDA timing | Roofline целевого GPU и проверенный диагноз bottleneck |
| [[02 Areas/ML & DL/06 Практика/08 Ускорить data pipeline|Ускорить data pipeline]] | DataLoader, tokenizer | Timeline и сравнение padding/bucketing/packing |
| [[02 Areas/ML & DL/06 Практика/09 Реализовать ring all-reduce|Реализовать ring all-reduce]] | Distributed PyTorch | Collective, тесты и bandwidth curve |
| [[02 Areas/ML & DL/06 Практика/10 Измерить checkpointing и offload|Checkpointing и offload]] | Transformer training | Pareto-график memory–time |
| [[02 Areas/ML & DL/06 Практика/11 Разрезать Transformer по TP и SP|Tensor и sequence parallelism]] | Collectives, tensor shapes | Sharded block и ledger коммуникаций |
| [[02 Areas/ML & DL/06 Практика/12 Собрать и проверить FSDP|Собрать FSDP]] | DDP, optimizer states | FSDP run, memory snapshots и переносимый checkpoint |
| [[02 Areas/ML & DL/06 Практика/13 Оптимизировать один Transformer step|Оптимизировать Transformer step]] | Profiling, AMP, attention | Проверенная серия оптимизаций и итоговая ведомость |
| [[02 Areas/ML & DL/06 Практика/14 Собрать минимальный inference engine|Минимальный inference engine]] | KV-cache, batching | Scheduler, prefill/decode и SLO-графики |
| [[02 Areas/ML & DL/06 Практика/15 Реализовать W8A8 и SmoothQuant|W8A8 и SmoothQuant]] | Quantization basics | Калибровка, accuracy и kernel-level benchmark |
| [[02 Areas/ML & DL/06 Практика/16 Проверить speculative decoding и rollback KV|Speculative decoding]] | Sampling, KV-cache | Корректный acceptance и rollback cache |
| [[02 Areas/ML & DL/06 Практика/17 Развернуть наблюдаемый ML сервис|Наблюдаемый ML-сервис]] | HTTP, Docker, metrics | Контейнер, dashboard и load-test |
| [[02 Areas/ML & DL/06 Практика/18 Harvard ML Systems — интерактивные design labs|Harvard ML Systems design labs]] | Вычислительные основы и системное мышление | Prediction, sweep, artifact и design ledger |
| [[02 Areas/ML & DL/06 Практика/19 Найти автомобильный номер в видеопотоке|Найти автомобильный номер в видеопотоке]] | Detection, tracking, OCR | Track-level ALPR pipeline с evidence и end-to-end метриками |
| [[02 Areas/ML & DL/06 Практика/20 Собрать языковую модель с нуля|Собрать языковую модель с нуля]] | BPE, Transformer, оптимизация | Проверяемая модель от tokenizer до генерации |
| [[02 Areas/ML & DL/06 Практика/21 Профилировать и ускорить Transformer kernel|Профилировать и ускорить Transformer kernel]] | PyTorch profiler, Triton | Проверка корректности, профиль выполнения и сравнение скорости |
| [[02 Areas/ML & DL/06 Практика/22 Провести scaling-law campaign|Провести scaling-law campaign]] | IsoFLOP, fitting, uncertainty | Данные запусков, fit и честный прогноз |
| [[02 Areas/ML & DL/06 Практика/23 Собрать воспроизводимый pretraining corpus|Собрать pretraining corpus]] | WARC/WET, filtering, MinHash | Версионированный корпус и fixed-budget сравнение |
| [[02 Areas/ML & DL/06 Практика/24 Post-training и RLVR для математического reasoning|Post-training и RLVR для reasoning]] | SFT, DPO, GRPO, verifier | Сравнимые policy runs и evidence bundle |
| [[02 Areas/ML & DL/06 Практика/25 Воспроизводимо оценить и red-team компьютерного агента|Оценить и red-team компьютерного агента]] | Agent loop, grader, sandbox | Воспроизводимый OSWorld-style запуск, trace и отчёт об угрозах |
| [[02 Areas/ML & DL/06 Практика/26 Поиск доказательства в Lean с verifier|Найти доказательство в Lean с verifier]] | Формальная логика, Lean basics | Проверяемая proof-search траектория и разбор ошибок |
| [[02 Areas/ML & DL/06 Практика/27 Проверяемый научный поиск на символьной регрессии|Проверяемый научный поиск]] | Регрессия, search, held-out data | Символьная гипотеза, независимая проверка и журнал поиска |

Первая интерактивная лаборатория уже доступна на сайте. Работы 06–17 используют
полностью импортированные оригинальные лекции, notebooks и homework EDLS: ссылки
внутри каждой работы ведут не на краткий пересказ, а на исходный учебный
материал. Работа 18 связывает 34 оригинальные Harvard Marimo labs с каноническими
главами и единым форматом design ledger. Работы 20–24 продолжают Stanford CS336:
от byte-level BPE и Transformer block через Triton и scaling campaign к
воспроизводимому корпусу и RLVR. Работы 25–27 переносят тот же принцип
в агентные системы: среда и verifier становятся частью эксперимента, а итогом
служит не рассказ модели об успехе, а воспроизводимый trace и проверяемое
состояние. По мере развития сайта измерительные
стенды можно переносить в интерактивные демо, не меняя постановку эксперимента.
