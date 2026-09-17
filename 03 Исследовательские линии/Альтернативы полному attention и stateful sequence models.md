---
title: "Альтернативы полному attention и stateful sequence models"
type: research-line
status: active
started: 2021
last_updated: 2026-09-01
last_verified: 2026-09-01
key_concepts: [SSM, Mamba, RWKV, linear attention, Gated DeltaNet, Test-Time Training, hybrid architecture]
key_models: [Mamba, Jamba2, Falcon-H1, Nemotron 3, Qwen3-Next, Kimi Linear]
primary_sources:
  - https://arxiv.org/abs/2312.00752
  - https://arxiv.org/abs/2405.21060
  - https://arxiv.org/abs/2412.06464
  - https://arxiv.org/abs/2407.04620
  - https://arxiv.org/abs/2406.07887
---

# Альтернативы полному attention и stateful sequence models

## Текущий тезис

Transformer не был вытеснен одной новой архитектурой. Наблюдаемый сдвиг другой:
полное внимание перестаёт быть обязательным в каждом слое. Несколько открытых
линеек используют recurrent SSM или gated linear-attention state в большинстве
слоёв и сохраняют редкий full, global или sparse attention для точного доступа к
прошлым позициям.

Это уже не единичные paper-прототипы: есть веса, kernels и runtime classes.
Однако основная масса зрелого production-serving по-прежнему оптимизирована под
Transformer-подобные модели. Наличие класса архитектуры в движке нельзя считать
доказательством одинаковой зрелости quantization, prefix caching, speculative
decode, LoRA, P/D disaggregation и CUDA graphs.

## Карта механизмов

| Механизм | Что хранится | Как читается прошлое | Основная цена |
|---|---|---|---|
| full attention | отдельные K/V позиций | новый query адресует архив | растущий KV и дорогой prefill |
| Mamba/SSM | векторное или структурированное состояние | через обученную динамику | необратимое сжатие |
| RWKV | матричное recurrent state | time-mix/read gate | отдельная экосистема kernels |
| linear attention | матрица key–value associations | умножением query на state | интерференция ассоциаций |
| Gated DeltaNet | matrix state с correction и decay | query к исправляемой памяти | сложные parallel/recurrent paths |
| TTT-layer | параметры маленькой модели | forward внутреннего learner | update на каждом токене |
| TTT-E2E | request-local изменяемые веса | обычный SWA Transformer после online updates | mutable serving state |
| hybrid | state + частичный KV | дешёвое сжатие и редкое точное чтение | два cache-контракта |

Устойчивая механика разобрана в
[[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/03 Mamba, RWKV, RetNet и гибридные архитектуры|учебной главе]].
Эта страница отслеживает свежие семейства и границы production evidence.

## Современные линии

### Чистые recurrent и SSM

**Mamba-1/2** имеет открытые веса, официальный CUDA/Triton-код и классы в
распространённых библиотеках. Это делает архитектуру воспроизводимой, но её
feature path уже Llama-подобного Transformer. **Mamba-3** имеет статью и код в
официальном repository; результаты статьи ограничены масштабом до 1.5B, поэтому
его следует называть research frontier, а не production frontier.

**Falcon Mamba** — полезный attention-free baseline с открытыми весами.
**RWKV-7** развивает dynamic matrix state и имеет сильную специализированную
экосистему, но mainstream feature parity документирована слабее.

Карточки: [[02 Areas/ML & DL/02 Атлас моделей/Семейства/Mamba|Mamba]] ·
[[02 Areas/ML & DL/02 Атлас моделей/Семейства/RWKV|RWKV]] ·
[[02 Areas/ML & DL/02 Атлас моделей/Семейства/Falcon|Falcon]].

### Scheduled hybrids

**Jamba/Jamba2** чередует Mamba и attention и независимо добавляет MoE. Jamba2
поставляется с весами и официальными командами запуска, что сильнее чистого
paper checkpoint. Это всё ещё vendor/runtime evidence, а не универсальный
workload benchmark.

**Nemotron-H/3** использует много Mamba-2 слоёв, редкие GQA/global-attention
слои и в части моделей MoE. NVIDIA публикует широкий собственный путь обучения
и запуска. Его зрелость внутри NVIDIA stack не следует автоматически переносить
на другой hardware и engine.

**Qwen3-Next** и последующие гибридные Qwen используют ритм Gated DeltaNet и
attention; **Kimi Linear** сочетает Kimi Delta Attention и редкие MLA layers.
Эти линии показывают, что индустрия исследует не только Mamba, а более широкий
класс recurrent matrix memory.

Карточки: [[02 Areas/ML & DL/02 Атлас моделей/Семейства/Jamba|Jamba]] ·
[[02 Areas/ML & DL/02 Атлас моделей/Семейства/Nemotron|Nemotron]] ·
[[02 Areas/ML & DL/02 Атлас моделей/Семейства/Qwen|Qwen]] ·
[[02 Areas/ML & DL/02 Атлас моделей/Семейства/Kimi|Kimi]].

### Parallel hybrids

**Falcon-H1** размещает attention и Mamba-2 параллельными ветвями одного mixer.
Это не «несколько Mamba-heads». Обе ветви видят одну глубину representation, а
затем объединяются. Паттерн уменьшает риск потерять деталь до следующего редкого
attention-layer, но усложняет kernels, sharding и баланс загрузки ветвей.

### Transformer отвечает сжатием архива

MLA, GQA, sparse attention, sliding windows и compressed attention не являются
SSM. Они сохраняют адресуемое чтение, уменьшая число KV heads, плотность связей
или размер представления. DeepSeek V2/V3/R1 относится к этой линии: MLA+MoE
изменяет стоимость Transformer, а не заменяет attention recurrent state.

Риск для full attention поэтому двусторонний: его могут заменить stateful
mixers или сделать редким/сжатым внутри Transformer-подобной архитектуры.

## Production evidence ladder

Архитектурный статус следует записывать как цепочку доказательств.

1. **Paper:** формула и эксперименты опубликованы.
2. **Weights:** checkpoint доступен и лицензия понятна.
3. **Reference code:** можно воспроизвести forward и recurrent state.
4. **Fast kernels:** есть отдельные train, prefill и decode paths.
5. **Runtime class:** модель загружается mainstream engine.
6. **Feature row:** проверены quantization, parallelism, LoRA, batching, prefix
   reuse, speculative decode, P/D и CUDA graphs.
7. **Workload evidence:** опубликованы engine revision, hardware, input/output
   distributions, concurrency и SLO percentiles.
8. **Deployment evidence:** система наблюдается в эксплуатации и имеет rollback.

Vendor-managed API может быть реальным продуктом, но не раскрывать уровни 6–8.
И наоборот, открытый kernel не доказывает устойчивый online service.

## Срез зрелости на 1 сентября 2026

Эта таблица намеренно не является рейтингом качества. Она фиксирует уровень
публичного инженерного свидетельства и должна перепроверяться перед каждым
использованием.

| Линия | Публичное свидетельство | Осторожная формулировка |
|---|---|---|
| Mamba-1/2, Falcon Mamba | papers, weights, kernels, runtime classes | runtime-ready; feature parity проверять |
| Mamba-3 | paper и официальный research code | research frontier, не production class |
| Jamba/Jamba2 | weights, vendor deployment, несколько engine paths | vendor-deployable; workload evidence ограничено |
| Falcon-H1 | weights, native paths в нескольких стеках | runtime-ready, не deployment-proven |
| Nemotron-H/3 | weights и наиболее полный NVIDIA path | strong vendor-stack evidence |
| Qwen3-Next | weights и mainstream classes | версия и feature combination критичны |
| Kimi Linear | paper, weights, kernel/vLLM implementation | молодой serving path |
| RWKV-7 | paper, weights, специализированные runtime | отдельная экосистема |
| TTT / TTT-E2E | papers, research repos, часть checkpoints | исследовательская система |
| DeepSeek V3/R1 | зрелый MLA Transformer path | не пример отказа от attention |

Текущие первичные указатели: [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models/),
[SGLang model support tracker](https://github.com/sgl-project/sglang/issues/18458),
[TensorRT-LLM supported models](https://nvidia.github.io/TensorRT-LLM/models/supported-models.html).
Строка в списке поддерживаемых моделей не должна сокращаться до слова
«production» без функционального smoke и workload benchmark.

## Stateful serving как отдельная исследовательская проблема

### Пакетирование

У каждого запроса своё mutable state. Планировщик должен менять соответствие
между строкой batch и state slot после каждого завершения, preemption и
возобновления. Ошибка индексации смешивает истории пользователей даже при
правильном forward одного запроса.

### Prefix reuse и branching

Для общего префикса нужен snapshot recurrent state в точке ветвления. У TTT-E2E
snapshot включает request-local weights. Политика cache теперь выбирает не
только страницы K/V, но и гранулярность checkpoints изменяемого состояния.

### Speculative decode

Rejection требует rollback. Append-only KV можно обрезать, а recurrent state
нужно откатить, скопировать либо воспроизвести. Глубокое speculative tree
умножает число потенциальных ветвей состояния.

### Prefill/decode transfer

Чистая SSM передаёт финальное состояние prefill, hybrid — state плюс частичный
KV, TTT-E2E — изменяемые параметры. Разный TP/PP layout требует явного
resharding. Малый объём state не отменяет coordination и correctness contract.

### Precision

Квантизация весов и состояния — разные решения. Ошибка состояния повторно
используется на каждом шаге и может накапливаться. Нужны parity tests для полного
prefill, chunked prefill, token-by-token decode, reset и переносов между workers.

## Как проверять качество памяти

Минимальный набор не должен ограничиваться Needle-in-a-Haystack.

| Проверка | Какая операция памяти нужна |
|---|---|
| copying | дословно воспроизвести фрагмент |
| phonebook | выбрать одну из многих пар после delayed query |
| MQAR | выполнить несколько key–value запросов |
| state tracking | обновлять текущее значение сущности |
| multi-hop retrieval | связать несколько удалённых свидетельств |
| long documents | работать с шумом и неравномерной плотностью фактов |

Perplexity оценивает среднее предсказание токена и может скрыть отказ на одной
редкой детали. Formal context length измеряет допустимый вход, но не lossless
memory. Controlled architecture comparison должен удерживать dataset,
tokenizer, parameter count и training compute.

## Где Transformer действительно рискует

Риск выше, когда:

- output очень длинный и чтение KV доминирует;
- поток непрерывен и естественно описывается динамическим состоянием;
- память устройства ограничена;
- workload допускает сжатие прошлого;
- редкий attention закрывает exact-recall tail;
- sparse или compressed attention сохраняет качество при меньшей цене.

Позиции Transformer сильнее, когда нужны произвольный exact recall, сложное
ветвление, зрелая quantization/serving экосистема и предсказуемая отладка cache.
Самый вероятный сценарий — не исчезновение attention, а специализация: дешёвое
состояние почти везде и дорогой адресуемый доступ там, где он окупается.

## Открытые вопросы

- Как измерять информационную ёмкость learned state на естественном языке?
- Какая частота attention оптимальна для разных retrieval workloads?
- Можно ли стандартизовать paged recurrent state и branch-safe snapshots?
- Когда TTT-updates дают память, а когда вызывают дрейф и забывание?
- Как квантизовать state без накопления систематической ошибки?
- Какие гибриды сохраняют преимущество после честного SLO benchmark на зрелом
  Transformer engine?

## Источники и связанные страницы

- [[02 Areas/ML & DL/05 Источники/Papers/TTT layers|TTT layers]].
- [[02 Areas/ML & DL/05 Источники/Papers/End-to-End Test-Time Training|TTT-E2E]].
- [[02 Areas/ML & DL/05 Источники/Papers/Gated Delta Networks|Gated Delta Networks]].
- [[02 Areas/ML & DL/05 Источники/Papers/Recall in efficient sequence models|Zoology, Repeat After Me и Mamba-based LMs]].
- [[02 Areas/ML & DL/03 Исследовательские линии/Эффективный attention|Эффективный attention]].
- [[02 Areas/ML & DL/03 Исследовательские линии/Длинный контекст|Длинный контекст]].
