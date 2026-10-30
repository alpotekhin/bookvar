# S7: редакционная переработка и проверка, 2026-10-08

## Область работы

Переработана только `00 Учебник/10 ML Systems/07 Profiling ML-нагрузки.md`;
добавлены три оригинальных PNG и два полных файла лицензий в
`00 Учебник/Assets/Figures/curated/profiling-review-2026-10/`.
Shared registries, CSS, navigation, английские страницы не изменялись этим
исполнителем. Commit/push/deploy не выполнялись.

Глава теперь развивается от задержки и критического пути к профилю операторов,
CPU/worker-процессам, Nsight Systems, GeLU/компиляции, памяти и Nsight Compute.
Сохранены семь `source_unit_id`, явный `cs336-systems-profiling` и прежние
якоря переименованных разделов. Связи S6 → S7 → S8 есть во вступлении и
навигации. Сохранены полезные темы dispatch, fusion, graph breaks, amortization,
экспорта модели и воспроизводимости. Все числа сквозного примера названы
расчётными; исключён рассказ о будто бы выполненной серии GPU-измерений.

## Прочитанные оригиналы

- [Max Ryabinin, EDLS week 2, pinned lecture PDF](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week02_fast_pipelines/lecture.pdf): слайды 4–8, 33–44; страницы 37, 39, 42 дополнительно отрендерены и осмотрены. Скачанный оригинал совпал побайтно с существующим `/tmp/bookvar-edls-week02.pdf`.
- [EDLS week 2 practice](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week02_fast_pipelines/seminar/practice.ipynb): раздел 2 целиком, CPU/CUDA aggregates, shapes, memory, Nsight; прочитан сохранённый исходный текст с outputs.
- [Stanford CS336 Lecture 6](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_06.py): benchmarking/profiling, определения GeLU и `run_operation1`; в локальном `var/traces/lecture_06.json` непосредственно проверены outputs шагов 245, 248, 251. Они действительно содержат 9 / 1 / 1 запусков; `dim=16384` означает матрицу 16384 × 16384, не вектор.
- [CS336 Assignment 2](https://github.com/stanford-cs336/assignment2-systems/blob/ca8bc81a59b70516f7ebb2da4808daade877c736/cs336_assignment2_systems.pdf): pp. 3–6 и 9–14, включая NVTX, tracebacks/overhead, memory lifecycle и saved tensors. Существующий Figure 1 crop осмотрен лично.
- [PyTorch Profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html): self/total, CPU/CUDA, grouping, schedule, callback/export; [исходный tutorial](https://github.com/pytorch/tutorials/blob/c80a43460c838fa28efa55e5680f3e97781e288e/recipes_source/recipes/profiler_recipe.py).
- [Understanding CUDA Memory Usage, PyTorch 2.8](https://docs.pytorch.org/docs/2.8/torch_cuda_memory.html): обе визуализации, границы наблюдения аллокатора, ограничение history, disable recording. Актуальная страница stable также проверена; графика закреплена за v2.8.0.
- [CUDA memory management](https://docs.pytorch.org/docs/2.14/notes/cuda.html#memory-management), [Working with Graph Breaks](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.graph_breaks_index.html), [Dealing with Recompilations](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.recompilation.html).
- [Nsight Systems focused profiling](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#focused-profiling) и [Nsight Compute Replay](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#replay): capture-range contract, NVTX, replay и влияние на измерение.
- [py-spy FAQ](https://github.com/benfred/py-spy#frequently-asked-questions): `--idle`, `--subprocesses`, attach permissions; [ONNX Concepts](https://onnx.ai/onnx/intro/concepts.html): граф, типы, operator sets и границы переносимости.

Старая ссылка PyTorch `/stable/compile/programming_model.common_graph_breaks.html`
переадресовывала на отсутствующий путь. В главе оставлена проверенная ссылка
на новую структуру документации 2.14.

## Иллюстрации и происхождение

Все три новые картинки получены из исходных репозиториев, не из слайда-посредника.
Копии побайтно идентичны источникам. Изменения: **нет**; crop, перерисовка,
добавление надписей и генерация изображений не выполнялись.

### `pytorch-profiler-trace.png`

- Автор: PyTorch contributors; автор сопровождающего tutorial — Shivam Raikundalia.
- [Страница](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html), [исходный PNG](https://github.com/pytorch/tutorials/blob/c80a43460c838fa28efa55e5680f3e97781e288e/_static/img/trace_img.png).
- Commit: `c80a43460c838fa28efa55e5680f3e97781e288e`.
- License: BSD-3-Clause; [закреплённый LICENSE](https://github.com/pytorch/tutorials/blob/c80a43460c838fa28efa55e5680f3e97781e288e/LICENSE).
- Размер: 3016 × 498. SHA-256: `ee1600300ce66949edbdd66ecb17aa9d8e0473f86991414d33ffe81b19681d02`.
- Полная лицензия сохранена рядом как `pytorch-tutorials-BSD-3-Clause.txt`, SHA-256 `5703fcbe2d5dfcc48d486d26e4b9d2e48255c24602e1bd0270b60b8978116f86`.
- Личный осмотр: дорожки CPU и stream 7 различимы, ось времени сохранена. На странице нужен доступ к полноразмерному изображению из-за широкого формата.

### `pytorch-active-memory-timeline.png`

- Автор: PyTorch contributors.
- [Страница](https://docs.pytorch.org/docs/2.8/torch_cuda_memory.html#active-memory-timeline), [исходный PNG](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/docs/source/_static/img/torch_cuda_memory/active_memory_timeline.png).
- Commit: `ba56102387ef21a3b04b357e5b183d48f0afefc7` (v2.8.0).
- License: BSD-style PyTorch LICENSE; [закреплённый файл](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/LICENSE).
- Размер: 1018 × 1056. SHA-256: `163d6d1ae4ae67e4cd98a03bdbefbfd37399f478ff0008eadc91af664aff5e5d`.
- Личный осмотр: сохранены размерная ось, полосы выделений и стек выбранного блока; исходный screenshot слегка обрезает верхний UI, но не учебную область. В подписи не приписаны неподтверждённые границы forward/backward.

### `pytorch-allocator-state-history.png`

- Автор: PyTorch contributors.
- [Страница](https://docs.pytorch.org/docs/2.8/torch_cuda_memory.html#allocator-state-history), [исходный PNG](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/docs/source/_static/img/torch_cuda_memory/allocator_state_history.png).
- Commit и license: те же, что у Active Memory Timeline.
- Размер: 1036 × 1068. SHA-256: `f14edfe988f4973c94a8b1f3d338a386d8c99aa19e0bb6285ef439eaa2238fdb`.
- Личный осмотр: видны список событий, выбранное событие, сегменты/блоки и стек. Screenshot не выдаётся за собственный OOM-эксперимент.
- Для обеих картинок полный PyTorch LICENSE сохранён рядом в `pytorch-LICENSE.txt`, SHA-256 `a8a2c6b67cf8a597ffaa5c82380a6bd1b82629c0417e00673d8013d662eb500a`.

Повторно использован существующий `stanford-cs336-2026/systems/a2-nsight-trace.png`
(MIT, Figure 1, p. 5; оригинальный crop, SHA в действующем registry).
Ранее созданные `gelu-profile-*.png` — оформленные заново таблицы, а не
оригинальные screenshots; из S7 их embeds удалены, сами shared assets сохранены.
Обобщённые Harvard SVG заменены в главе предметными оригинальными screenshots;
исходные файлы и ссылки на подробные главы Harvard сохранены.
EDLS pp. 39–43 содержат заимствованные изображения Paul Bridger: MIT курса
не считалась доказательством права на отдельное повторное использование этих
сторонних изображений. Они не добавлены.

## Выполненные проверки

Исполняемый проверочный скрипт на момент проверки: `/tmp/s7-profiling-validation.py`.
Главный редактор переносит его в `publishing/tools/check_profiling_examples.py`.
Команда фактического запуска (cwd `/tmp`):

```bash
uv run --with torch --python /Users/aleksan.potekhin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 python /tmp/s7-profiling-validation.py
```

Результат: exit 0, PyTorch 2.14.1, macOS arm64, `torch.cuda.is_available() == False`.

- AST всех 5 Python-блоков: passed.
- Все wikilinks и пути изображений главы разрешаются локально: passed.
- Самостоятельный profiler-example выполнен без изменения текста блока: сохранён один JSON, 767 событий, по 3 диапазона `forward_and_loss`, `backward`, `optimizer_step`; все параметры модели конечны.
- Trace: `/var/folders/6c/j5sghwf51gx41g_055zlcs_m0000gq/T/s7-profiling-cpu-ybisnz3o/artifacts/profiler/step-5.json`.
- Naive/builtin GeLU: совпадение outputs и gradients для `(17,)`, `(7, 13)`, `(0,)`, FP32/FP64, `rtol=1e-5`, `atol=1e-6`.
- `torch.compile` wrapper создан; исполнение optimizing backend этим тестом не проверялось.
- Очистка memory-history после обычного выхода и исключения: passed на mock; это проверка Python control flow, не CUDA API.
- Арифметика расчётного примера: 420 → 310 → 293, passed.
- `git diff --check` по главе: passed после финальных текстовых правок.

## Граница готовности

Подготовлена и проверена русская глава и происхождение картинок. CUDA-профиль,
NVTX/Nsight и реальный Memory Snapshot локально не выполнялись: NVIDIA GPU
здесь недоступен. Утверждений о новом GPU-ускорении нет.

Главному редактору переданы регистрационные данные трёх PNG. Он отдельно
выполняет личную проверку, registry/license publication, общие build/link/output
checks и визуальную проверку страницы. Локальная проверка текста/PNG не
выдаётся за проверку опубликованного сайта. Английская версия этой задачей
не изменялась; parity остаётся отдельной обязанностью общего пакета.
