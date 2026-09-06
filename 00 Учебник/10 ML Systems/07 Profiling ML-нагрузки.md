---
title: Profiling ML-нагрузки
type: textbook-chapter
status: draft
last_verified: 2026-09-06
source_language: mixed
source_unit_id:
  - lecture-06-benchmark-profile-experiment
  - lecture-06-section-169-add-dim-2048
  - lecture-06-section-244-naive-gelu
  - lecture-06-gelu-profiler-comparison
  - assignment-02-nsight-systems-trace
  - assignment-02-task-nsys-profile
  - assignment-02-task-memory-profiling
---

<a id="cs336-systems-profiling"></a>

# Profiling ML-нагрузки

Benchmarking говорит, что программа медленная; profiling показывает, где
исчезает время и память. Начинают с самого дешёвого уровня и углубляются только
после локализации bottleneck.

## Полный диагностический маршрут

Название оператора само по себе не объясняет задержку. Сначала находят пустоты и
зависимости на timeline, затем переходят к operator- и kernel-level counters.

## Иерархия инструментов

| Вопрос | Инструмент |
|---|---|
| CPU или GPU простаивает? | system metrics, DCGM, timeline |
| Какая Python-функция занята? | `py-spy`, `cProfile`, Scalene |
| Какие PyTorch operators дороги? | PyTorch Profiler |
| Почему растёт CUDA memory? | PyTorch Memory Snapshot |
| Где gaps, copies и synchronization? | Nsight Systems |
| Почему конкретный kernel медленный? | Nsight Compute |

`nvidia-smi utilization=100%` означает лишь, что GPU исполнял хоть что-то в
sampling interval: dummy wait kernel тоже может дать 100%. Это не MFU.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/performance/diagnostic-flow.svg]]

*Оригинальная иллюстрация Harvard CS249r, Vol. II, Performance Engineering,
§ “Iron Law Diagnostic Flowchart”, locator
`sec-performance-engineering-iron-law`; [исходный SVG](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/performance_engineering/images/svg/diagnostic-flow.svg),
CC BY-NC-SA 4.0; файл не изменён. Дерево фиксирует порядок исключения I/O, CPU
и communication stalls до перехода к kernel-level диагнозу.*

## CPU: py-spy

Sampling profiler периодически снимает stacks и почти не меняет программу:

```bash
py-spy top --pid PID
py-spy record -o profile.svg --pid PID
```

Возможность attach к уже запущенному процессу особенно полезна для stalls.
Sampling может пропустить очень короткие функции; instrumenting profiler точнее
по calls, но сильнее perturb execution.

## PyTorch Profiler

Профилируют ограниченное окно после warmup:

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for batch in loader:
        step(batch)
        prof.step()
```

Смотрят CPU self time, CUDA time, shapes, memory и trace. `record_shapes` и
stacks имеют overhead, поэтому profiler-run не используют как финальный
benchmark.

## Memory Snapshot

Snapshot отвечает «кто аллоцировал live/reserved blocks?»:

```python
torch.cuda.memory._record_memory_history()
run_steps()
torch.cuda.memory._dump_snapshot("snapshot.pickle")
```

Это PyTorch-private API на момент курса; имя и viewer зависят от версии.
Snapshot видит allocator PyTorch, но может не видеть всю память CUDA libraries.
Различают allocated live tensors, reserved caching allocator и non-PyTorch
allocations. Снимок до/после помогает найти утечку ссылок, fragmentation и
временный peak.

## Nsight Systems

Systems trace связывает CPU threads, CUDA API, kernels, streams, memcpy и
collectives. Типовые паттерны:

- большие плотные kernels без gaps — оптимизировать kernels/precision;
- множество крошечных kernels и CPU launch gaps — fusion/compile/CUDA Graphs;
- частые D2H и synchronization — убрать `.item()` и зависимости;
- GPU ждёт DataLoader — workers, decode, prefetch, pinned transfer;
- communication не перекрыта — изменить schedule/stream dependencies.

Nsight Compute нужен после Systems, когда выбран конкретный kernel: occupancy,
memory throughput, Tensor Core instructions, stalls и roofline position.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/a2-nsight-trace.png]]

*Оригинальный trace из [Stanford CS336 Assignment 2, Figure 1, p. 5](https://github.com/stanford-cs336/assignment2-systems/blob/ca8bc81a59b70516f7ebb2da4808daade877c736/cs336_assignment2_systems.pdf),
commit `ca8bc81`. Верхние дорожки показывают GPU activity, ниже расположены
CUDA kernels и вложенные NVTX/PyTorch ranges. Вертикальное сопоставление дорожек
связывает участок Python/модуля с kernels, которые он действительно запустил.*

### NVTX как связь между моделью и timeline

Profiler не знает, что для автора программы означает «один Transformer block»
или «attention forward». Эти границы добавляют NVTX ranges, а Nsight Systems
сохраняет их вместе с CUDA API и device timeline:

```python
torch.cuda.nvtx.range_push("attention.forward")
try:
    y = attention(q, k, v)
finally:
    torch.cuda.nvtx.range_pop()
```

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi \
  --output=artifacts/transformer-step \
  python benchmark.py
```

Capture начинают после warmup: иначе timeline занят initialization и compile.
NVTX range должен охватывать ровно тот Python interval, который требуется
объяснить, но его длительность на CPU нельзя выдавать за device time. CUDA
launch асинхронен; причинную связь устанавливают по вложенности range, CUDA API
и зависимостям streams.

### Три GeLU как учебный профиль

Lecture 6 выполняет один и тот же tanh-approximate GeLU тремя способами:

1. **naive PyTorch expression** раскладывается на умножения, сложения, `tanh`
   и несколько промежуточных tensors;
2. **builtin `torch.nn.functional.gelu`** вызывает библиотечный путь;
3. **`torch.compile(naive_gelu)`** получает граф целиком и может породить
   fused kernel.

Перед сравнением курс проверяет equality с исходной функцией, затем отдельно
измеряет и профилирует варианты. В trace нужно искать не только меньшее число
kernels, но и исчезнувшие HBM round trips, launch gaps, graph breaks и новый
register/occupancy режим. Компиляция иногда проигрывает на первой итерации или
новой shape из-за compile cost; builtin может уже быть оптимальным. Поэтому
финальный вывод даёт повторный benchmark без profiler overhead.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/gelu-profile-naive.png]]

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/gelu-profile-builtin.png]]

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/gelu-profile-compiled.png]]

*Три таблицы построены непосредственно по выводу профилировщика из Stanford
CS336 Lecture 6, шаги трассы 245, 248 и 251, для одного и того же входа
`dim=16384`. В eager-варианте видно девять запусков, тогда как библиотечный
GeLU и скомпилированный граф исполняют вычисление одним kernel. Конкретные
времена относятся к машине курса; переносимый вывод здесь — структура запуска,
а не отношение микросекунд. [Закреплённая версия исходного кода](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_06.py#L265-L302).*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/performance/profiling-hierarchy.svg]]

*Источник: Harvard CS249r, Vol. II preview, Performance Engineering; [исходный
SVG](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/performance_engineering/images/svg/profiling-hierarchy.svg),
CC BY-NC-SA 4.0. Сопоставьте строку с наблюдаемым симптомом: общая задержка ведёт
к системному профилю, а уже локализованный медленный kernel — к аппаратным
счётчикам.*

## End-to-end кейс

Baseline после warmup: step median/p95 420/470 ms. `py-spy` показывает ожидание
`next(loader)`. PyTorch trace раскладывает median: 110 ms loader wait, 18 ms
H2D, 275 ms CUDA kernels, 17 ms launch gaps. Nsight Systems подтверждает:
pageable H2D не перекрывается с compute.

После `pin_memory`, persistent workers и bounded prefetch loader wait почти
целиком перекрыт, critical path — 304 ms. Затем trace показывает сотни мелких
elementwise kernels; `torch.compile` даёт 286 ms. Непрофилированная повторная
серия даёт median/p95 288/315 ms при тех же tokens и loss. Так разделяются
причина, эффект изменения и perturbation самого profiler.

## Воспроизводимый цикл

1. Зафиксировать workload, hardware/software versions и baseline distribution.
2. Найти доминирующий interval на end-to-end timeline.
3. Сформулировать одну гипотезу в терминах compute/data/latency.
4. Изменить одну вещь.
5. Повторить benchmark без profiler overhead.
6. Проверить correctness и memory peak.

Вместе с trace сохраняют git commit, command/config, input shapes и token count,
GPU/driver/CUDA/cuDNN/NCCL, PyTorch/compiler versions, clocks/power mode,
warmup/active schedule, rank/host и before-after benchmark. PyTorch export
делают через `tensorboard_trace_handler`; Nsight сохраняют в `.nsys-rep`,
memory snapshot — в versioned `.pickle`. Trace может содержать stack paths,
shapes и NVTX labels с пользовательскими данными — перед публикацией его
очищают. Сравнение артефактов разных версий без manifest ненадёжно: fusion,
operator names и private snapshot API меняются.

Framework layer также платит dispatch tax: eager graph удобно отлаживать, но
Python и operator dispatch заметны при мелких операциях. Compilation/fusion
помогают только если trace действительно показывает этот режим.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/performance/optimization-decision-tree.svg]]

*Оригинальная иллюстрация Harvard CS249r, Vol. II, Performance Engineering,
§ “Optimization decision tree”, locator
`sec-performance-engineering-optimization-decision-tree`; [исходный
SVG](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/performance_engineering/images/svg/optimization-decision-tree.svg),
CC BY-NC-SA 4.0; файл не изменён. Она превращает установленный bottleneck в
выбор класса вмешательства и удерживает profiling перед optimization.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/foundation/frameworks_dispatch_tax_divergence.svg]]

*Источник: Harvard CS249r, Frameworks; [исходный SVG](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/frameworks/images/svg/frameworks_dispatch_tax_divergence.svg),
CC BY-NC-SA 4.0. Сравните расхождение кривых при уменьшении размера операции:
если полезная работа сокращается быстрее стоимости dispatch, bottleneck лежит
выше уровня GPU-kernel.*

## От Python-функции до kernel: где находится framework

Вычислительный граф из главы об обратном распространении описывает зависимости
между операциями. Для исполнения этого описания framework решает ещё три
задачи: когда запускать операцию, какие промежуточные значения сохранить для
градиента и как отобразить высокоуровневую операцию на конкретное устройство.
Поэтому PyTorch, JAX или TensorFlow — не просто библиотеки тензоров. Между
записью `y = layer(x)` и инструкциями GPU находится стек преобразований.

```text
Python/module API
    ↓ operator dispatch, shapes, dtype, device
eager trace or captured graph
    ↓ automatic differentiation / backward graph
graph transformations and decomposition
    ↓ fusion, layout, memory planning, scheduling
backend IR and generated kernels
    ↓ runtime launch, streams, allocator, collectives
CPU / GPU / accelerator
```

Eager execution немедленно передаёт каждый оператор runtime. Оно удобно для
отладки и динамического control flow, но платит Python и dispatch overhead на
каждом шаге. Graph execution сначала фиксирует зависимости, после чего может
увидеть несколько операторов сразу. Цена — compilation latency, guards на
формы и типы, возможные graph breaks и необходимость перекомпиляции при новом
варианте входа.

### Capture не равен optimization

Получить graph — ещё не значит ускорить программу. Compiler должен доказать,
что преобразование сохраняет семантику, выбрать decomposition сложного
оператора, согласовать layout, распланировать промежуточную память и создать
kernel для целевого backend. Динамическая форма, изменение Python state,
data-dependent branch или неподдержанный custom operator способны разорвать
граф и вернуть часть исполнения в eager.

Для `torch.compile` полезно отдельно измерять:

- cold compile time;
- число уникальных graphs и recompilations;
- долю шага внутри compiled regions;
- steady-state latency при тех же shapes;
- корректность outputs и gradients;
- peak memory, потому что fusion меняет lifetime тензоров.

Компиляция оправдана, если deployment выполнит достаточно шагов, чтобы
амортизировать cold cost, а captured workload остаётся стабильным.

## Fusion как устранение движения данных

Пусть две поэлементные операции читают и записывают activation размера $N$.
При раздельном исполнении промежуточный тензор записывается в HBM и снова
читается. Fused kernel держит его в регистрах или shared memory и делает один
launch. Арифметика почти не меняется; исчезают лишние bytes и dispatch.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/performance/operator-fusion.svg]]

*Слева отдельные kernels материализуют промежуточные результаты в глобальной
памяти; справа fused kernel сохраняет их ближе к вычислительным блокам.
Источник: Harvard CS249r, Vol. II, [Performance Engineering — Operator
Fusion](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/performance_engineering/images/svg/operator-fusion.svg),
CC BY-NC-SA 4.0, исходный SVG не изменён.*

Fusion не всегда улучшает результат. Слишком крупный kernel может увеличить
register pressure, снизить occupancy, усложнить scheduling или повторно
вычислять дорогие значения. Поэтому решение принимается по timeline и kernel
counters, а не по числу объединённых operators.

## Interoperability: graph как контракт с границами

ONNX и другие переносимые IR помогают передать структуру модели между
framework и inference runtime. Они фиксируют граф операторов, shapes/types и
веса, но не переносят автоматически Python preprocessing, tokenizer, sampling,
custom kernels и всю семантику динамической модели. Экспорт считается
проверенным после сравнения outputs на representative inputs, включая крайние
длины и динамические shapes.

Полезно различать три артефакта:

1. checkpoint хранит параметры для конкретной реализации;
2. exported graph описывает вычисление в заданном operator set;
3. compiled engine содержит план под конкретные hardware, shapes и precision.

Смена артефакта меняет область воспроизводимости. Engine нельзя считать
универсальным продолжением checkpoint, если его tactic selection и calibration
зависят от GPU или профиля форм.

## Диагноз по уровню стека

| Наблюдение | Вероятный уровень | Следующее доказательство |
|---|---|---|
| длинные CPU gaps между короткими kernels | Python/dispatch | CPU stack + CUDA API timeline |
| graph breaks и частые recompilations | capture/guards | compiler logs и набор входных shapes |
| много HBM traffic между elementwise ops | graph/kernel | fusion candidate + memory trace |
| один большой медленный GEMM | kernel/hardware | Nsight Compute roofline и Tensor Core use |
| правильный eager, неверный engine | export/backend | layer-wise output comparison |

Эта таблица связывает framework с profiling: оптимизация должна происходить на
том уровне, на котором найдено ограничение. Переписывать CUDA kernel бессмысленно,
если GPU ждёт Python; включать compiler бессмысленно, если один GEMM уже занимает
весь critical path и работает у аппаратного потолка.

## Практика и первоисточники

- [[05 Источники/Courses/Harvard ML Systems/tinytorch/14_profiling|TinyTorch 14 — Profiling]]: исполняемый `Profiler` для подсчёта параметров и FLOP, измерения памяти и распределения latency.
- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol2/performance_engineering|Harvard CS249r — Performance Engineering]]: диагностическое дерево, roofline и анализ bottleneck.
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week02_fast_pipelines/seminar/practice.ipynb|EDLS Week 2 — практика профилирования]].
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week06_dl_arithmetic/seminar/practice.ipynb|EDLS Week 6 — практика арифметики и профилирования]].

- [[05 Источники/Courses/Harvard ML Systems/vol1/frameworks|Harvard ML Systems, Vol. I — ML Frameworks]] — полная локальная глава: computational graphs, automatic differentiation, eager/operator dispatch, compilation, interoperability и границы framework abstraction.
- [EDLS week 2 lecture](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week02_fast_pipelines/lecture.pdf) — “Profiling: what and why”, “How to profile Python/GPU/PyTorch code?”, “PyTorch Profiler + trace viewer”, “Nsight Systems/Nsight Compute”, “Profiling: typical patterns”; title locators used because incremental slides repeat in the PDF.
- [EDLS week 2 profiler practice](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week02_fast_pipelines/seminar/practice.ipynb)
- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [PyTorch Memory Snapshot](https://pytorch.org/docs/stable/torch_cuda_memory.html)
- [Nsight Systems](https://docs.nvidia.com/nsight-systems/)
- [Harvard CS249r, Frameworks](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/frameworks/frameworks.qmd)
- [Stanford CS336 Lecture 6, pinned `8b59b507`, steps 93–296](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_06.py) — benchmark/profile loop and naive/builtin/compiled GeLU.
- [Stanford CS336 Assignment 2, pinned `ca8bc81`, pp. 3–14](https://github.com/stanford-cs336/assignment2-systems/blob/ca8bc81a59b70516f7ebb2da4808daade877c736/cs336_assignment2_systems.pdf) — Nsight/NVTX and memory-profiling evidence contract.

← [[02 Areas/ML & DL/00 Учебник/10 ML Systems/06 Data pipeline, padding и packing|Data pipeline, padding и packing]] ·
[[02 Areas/ML & DL/00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению|GPU kernels и Triton]] →
