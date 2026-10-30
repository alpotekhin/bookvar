---
title: Profiling ML-нагрузки
type: textbook-chapter
status: draft
last_updated: 2026-10-27
last_verified: 2026-10-27
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

Две программы обучают одну модель на одинаковых данных, но одна обрабатывает
вдвое меньше токенов в секунду. Причиной может быть медленный декодер,
копирование данных, сотни коротких запусков CUDA, неудачное матричное умножение
или ожидание другого GPU. По средней длительности шага эти случаи неотличимы.
Чтобы выбрать изменение, нужно восстановить последовательность событий:
что исполнялось, что в этот момент ожидало и какое ожидание задержало результат.

Измерение производительности (*benchmarking*) устанавливает, насколько быстро
работает программа при заданных условиях. Профилирование (*profiling*) связывает
это время с участками программы, вычислительными ядрами и перемещением данных.
Оно объясняет измерение, но не заменяет его: сбор событий сам расходует ресурсы
и может менять взаимное перекрытие операций.

В [[02 Areas/ML & DL/00 Учебник/10 ML Systems/06 Data pipeline, padding и packing|предыдущей главе]]
конвейер должен был вовремя готовить полезные токены. Теперь нужно проверить,
получилось ли это в действительности. К
[[02 Areas/ML & DL/00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению|написанию GPU-ядер]]
имеет смысл переходить с конкретной задачей: например, устранить несколько
проходов по памяти в операции, заметной в полном шаге, а не просто заменить
любой оператор PyTorch собственным кодом.

<a id="полный-диагностический-маршрут"></a>

## Время работы и зависимости

### Длительность работы и длительность ожидания

Вызов `model(x)` на CPU обычно не ждёт окончания всех вычислений на GPU.
Процессор подготавливает аргументы и ставит работу в очередь, а ускоритель
исполняет её позднее. Поэтому короткий Python-вызов может запускать длинное
ядро, а длинный вызов `.item()` — почти ничего не вычислять, но ждать результата
предшествующих операций. Если переписать только `.item()`, не изменив потребность
CPU в результате, ожидание просто переместится в другое место.

Временная шкала (*timeline*) помещает события CPU и GPU на общую ось времени.
Прямоугольник показывает интервал события, дорожка — поток исполнения или
другой источник событий. Вложенные прямоугольники обычно обозначают вызовы
внутри вызовов. Между дорожками важна не только близость по времени, но и
причинная связь: какой CPU-вызов поставил ядро в очередь, какой stream его
исполнил и какой результат требуется следующей операции.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/profiling-review-2026-10/pytorch-profiler-trace.png]]

*Оригинальный снимок из PyTorch Tutorials, Shivam Raikundalia и участники
проекта, [«PyTorch Profiler», раздел об экспорте trace](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html).
Сверху — вложенные вызовы операторов на CPU, ниже — ядра в CUDA stream 7.
CPU успевает продвинуться по программе, пока GPU ещё исполняет поставленную
работу. [Исходный PNG](https://github.com/pytorch/tutorials/blob/c80a43460c838fa28efa55e5680f3e97781e288e/_static/img/trace_img.png)
перенесён без изменений, BSD-3-Clause. Шкала времени относится к примеру
документации, а не к измерению в этой книге.*

Полное время определяется критическим путём — цепочкой зависимостей,
завершения которой необходимо дождаться для получения результата. Допустим,
подготовка следующего пакета занимает 40 мс и полностью перекрывается с
70 мс вычислений текущего пакета. Сумма занятого времени двух ресурсов
равна 110 мс, но длительность данного перекрытого интервала — 70 мс.
Ускорение подготовки с 40 до 20 мс само по себе этот интервал не сократит.
Если же подготовка занимает 90 мс, у GPU возникает 20 мс открытого ожидания.
Это расчётный пример различия между суммой работ и задержкой результата,
а не модель всех накладных расходов конвейера.

Поэтому таблица «самых дорогих операторов» — указатель для дальнейшего
исследования. Оператор может иметь большое суммарное время, но исполняться
параллельно с другим ограничивающим участком. И наоборот, несколько небольших
ожиданий между операциями могут определять значительную часть задержки шага.

<a id="диагноз-по-уровню-стека"></a>

### Иерархия инструментов

Начинают с вопроса, который требуется разрешить, и выбирают достаточный
уровень наблюдения. Чем глубже инструмент вмешивается в исполнение, тем уже
должно быть окно сбора: профилировать всю эпоху аппаратными счётчиками обычно
не нужно.

| Инструмент | Что можно выяснить и чего не видно |
|---|---|
| Метрики системы, `nvidia-smi`, DCGM | Есть ли простой, как меняются скорость и память. Усреднение не показывает зависимости отдельных операций. |
| `py-spy`, `cProfile`, Scalene | Где Python вычисляет, а где ждёт. Python-стек не раскрывает внутреннее исполнение CUDA. |
| PyTorch Profiler | Какие операторы, формы и места вызова дают основную работу. Агрегаты скрывают порядок и перекрытие событий. |
| PyTorch Memory Snapshot | Какие тензоры живы в момент пика и где они созданы. По умолчанию видны выделения CUDA-аллокатора PyTorch, не вся память процесса. |
| Nsight Systems | Последовательность CPU, GPU и передач данных. Локализует ожидание, но не объясняет каждую аппаратную причину. |
| Nsight Compute | Аппаратные ограничения выбранного ядра. Подробный сбор меняет исполнение; это не финальный замер полного шага. |

Значение `GPU-Util=100%` в `nvidia-smi` не означает достижения пиковой
производительности. Этот показатель отражает долю интервала наблюдения,
в течение которой исполнялось хотя бы одно ядро. В нём одинаково «заняты»
эффективное матричное умножение и малоэффективное ожидание внутри ядра.
MFU отвечает на другой вопрос — какую долю условной пиковой вычислительной
производительности составляет полезная работа модели. Выбор счётчика и
методика подсчёта FLOP рассмотрены в
[[02 Areas/ML & DL/00 Учебник/10 ML Systems/03 Измерение производительности и roofline|главе об измерениях и roofline]].
Такое различие особенно подчёркивает Max Ryabinin в
[Efficient DL Systems, week 2, слайды 4–8 и 33–44](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week02_fast_pipelines/lecture.pdf).

## Подготовка наблюдаемого эксперимента

Профиль имеет смысл только вместе с описанием нагрузки. Прямой проход,
полный шаг обучения и генерация по одному токену могут запускать разные
ядра и по-разному использовать память. Нужно заранее зафиксировать формы
тензоров, типы данных, режим обучения, число полезных токенов, используемый
алгоритм attention и то, входит ли загрузка данных в измеряемый интервал.
Профиль заранее размещённого на GPU случайного тензора не подтверждает
эффективность чтения реального датасета.

Прогрев отделяют от исследуемой устойчивой работы. Первый вызов может
инициализировать CUDA, загрузить библиотеку, подобрать алгоритм или
скомпилировать граф. Первое обновление Adam обычно создаёт состояния
оптимизатора. Все эти расходы реальны, но отвечают на вопрос о холодном
старте. Если интересует установившаяся скорость, прогрев должен охватить
именно тот режим и те формы, которые затем будут измеряться. Постоянно
меняющиеся длины могут вызывать новые компиляции уже после первоначального
прогрева — это свойство нагрузки, которое нельзя спрятать выбором удобного окна.

Затем выбирают несколько представительных шагов. Для первого прохода обычно
достаточно событий CPU и CUDA и собственных меток крупных стадий. Формы,
стеки и историю памяти добавляют, когда возник соответствующий вопрос:
подробная диагностика увеличивает объём trace и накладные расходы.

<a id="pytorch-profiler"></a>

### PyTorch Profiler: окно сбора и метки стадий

Ниже — самостоятельный пример небольшого шага обучения. Он создаёт данные
на устройстве заранее, поэтому намеренно **не измеряет DataLoader и H2D**.
На машине без CUDA он собирает только CPU-профиль; такой запуск проверяет
инструментирование, но ничего не говорит о скорости GPU.

```python
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.nn.Sequential(
    torch.nn.Linear(256, 512),
    torch.nn.GELU(approximate="tanh"),
    torch.nn.Linear(512, 256),
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randn(32, 256, device=device)
target = torch.randn_like(x)


def train_step():
    optimizer.zero_grad(set_to_none=True)
    with record_function("forward_and_loss"):
        prediction = model(x)
        loss = torch.nn.functional.mse_loss(prediction, target)
    with record_function("backward"):
        loss.backward()
    with record_function("optimizer_step"):
        optimizer.step()


for _ in range(5):
    train_step()
if device.type == "cuda":
    torch.cuda.synchronize()

output_dir = Path("artifacts/profiler")
output_dir.mkdir(parents=True, exist_ok=True)
activities = [ProfilerActivity.CPU]
if device.type == "cuda":
    activities.append(ProfilerActivity.CUDA)


def save_trace(prof):
    prof.export_chrome_trace(str(output_dir / f"step-{prof.step_num}.json"))
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=12))


with profile(
    activities=activities,
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=save_trace,
    record_shapes=False,
    profile_memory=False,
    with_stack=False,
) as prof:
    for _ in range(5):
        train_step()
        prof.step()
```

Здесь есть два разных прогрева. Пять шагов перед контекстом приводят модель
и оптимизатор в рабочее состояние. Параметр `warmup=1` внутри расписания
подготавливает сборщик событий, после чего три шага попадают в сохранённый
профиль; `wait=1` пропускает первый шаг. `prof.step()` сообщает профилировщику
границу очередного шага, а `on_trace_ready` сохраняет завершённое окно.
Произвольный вызов `prof.step()` не является синхронизацией GPU.

Номера шагов внутри контекста начинаются с нуля. Для этого расписания:

| Шаг цикла | Действие профилировщика |
|---|---|
| 0 | `wait`: сбор ещё выключен |
| 1 | `warmup`: события собираются, но затем отбрасываются |
| 2, 3, 4 | `active`: события входят в итоговый trace |

После пятого вызова `prof.step()` завершено одно окно; callback сохраняет
`step-5.json`. Это три исследуемых шага, а не пять. Если нужен следующий
цикл сбора, меняют `repeat` и число итераций; один callback не сохраняет
автоматически всю предшествующую историю обучения.

JSON открывают в [Perfetto](https://ui.perfetto.dev/) или совместимом
просмотрщике trace. Сохранение каждого окна через callback удобно и при
нескольких циклах расписания: одному файлу соответствует одно окно.
Другой вариант экспорта — `torch.profiler.tensorboard_trace_handler`.
Используемые поля расписания и значения агрегатов описаны в
[PyTorch Profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html).

В реальном конвейере меткой `record_function("input.next_batch")` оборачивают
именно `next(iterator)`, а отдельной меткой — копирование на устройство.
Если метка поставлена только внутри тела `for batch in loader`, получение
`batch` уже произошло до входа в тело цикла. Ожидание данных окажется за
пределами этой метки, и профиль будет неполным именно там, где его хотели
исследовать.

## Как читать таблицу операторов

У строки профилировщика есть имя операции, число вызовов и несколько видов
времени. `Self CPU` исключает время дочерних вызовов; `CPU total` включает их.
Например, `aten::conv2d` может вызвать `aten::convolution`, а тот —
библиотечную реализацию. Складывать `CPU total` этих строк нельзя: значительная
часть интервала будет посчитана несколько раз.

Рассмотрим один CPU-поток без параллельных дочерних вызовов. Диапазон
`forward_and_loss` длится 12 мс. Внутри него находятся неперекрывающиеся
`aten::mm` длительностью 8 мс и `aten::gelu` длительностью 2 мс;
остальные 2 мс принадлежат самому диапазону. Если у этих двух операторов
нет дочерних событий, таблица выглядит так:

| Событие | Self CPU, мс | CPU total, мс |
|---|---:|---:|
| `forward_and_loss` | 2 | 12 |
| `aten::mm` | 8 | 8 |
| `aten::gelu` | 2 | 2 |

Сумма `CPU total` равна 22 мс, хотя весь диапазон занял 12 мс:
восемь и две миллисекунды уже входят в строку родителя. Сумма
`Self CPU` здесь равна 12 мс. Для нескольких потоков даже сумма
невложенных событий может превысить длительность окна, потому что
потоки работают одновременно. Это расчётная таблица; реальные
поля и группировка показаны в
[PyTorch Profiler, Using profiler to analyze execution time](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#using-profiler-to-analyze-execution-time).

Для CUDA отдельно учитываются события устройства. CPU-время вызова включает
работу на стороне хоста и возможные ожидания, а CUDA-время относится к
исполнению на ускорителе. Большое `CPU total` при малом собственном CPU-времени
не доказывает, что Python выполнил много вычислений. Аналогично сумма времени
ядер из нескольких streams не обязана совпадать с длительностью полного шага:
ядра могут перекрываться. Названия колонок и ключи сортировки зависят от версии
PyTorch; для CUDA в используемых курсом версиях встречаются
`self_cuda_time_total` и `cuda_time_total`.

Пусть два независимых ядра разных streams исполнялись в интервалах
3–8 и 6–10 мс. Их суммарное время равно 9 мс, но объединённый
интервал занятого GPU — 3–10 мс, то есть 7 мс. Если измеряемое окно
началось в нуле и закончилось в 10 мс, его длительность — ещё одна
величина, 10 мс. По сумме 9 мс нельзя восстановить ни 3 мс начального
простоя, ни пересечение ядер длительностью 2 мс: нужен trace.

Следующий вопрос — сколько раз операция вызвана и с какими формами.
Один вызов на 10 мс и тысяча вызовов по 10 мкс имеют одинаковую сумму,
но предлагают разные направления оптимизации. Во втором случае существенны
запуски и диспетчеризация. Если включить `record_shapes=True`, вызов
`prof.key_averages(group_by_input_shape=True)` разделит одинаковые имена
по входным формам. Это полезно, когда несколько матричных умножений одного
типа исполняются совершенно по-разному из-за размеров.

Именно такой переход делает
[Stanford CS336 Lecture 6, раздел `profiling`](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_06.py):
сначала сложение матриц размера $2048\times2048$, затем матричное умножение
того же размера и умножение $128\times128$. В таблицах появляется не только
другая длительность, но и другое имя низкоуровневого ядра. Части имени могут
указывать библиотеку, архитектуру GPU, тип чисел и размеры плитки. Это
подсказки о выбранной реализации, а не устойчивый программный интерфейс.

После сортировки выбирают несколько кандидатов и возвращаются на временную
шкалу. Требуется выяснить, встречаются ли они в каждом шаге, совпадают ли
с замедленными шагами и находятся ли на критическом пути. Короткое окно
из трёх обычных шагов не объясняет редкую паузу раз в несколько сотен шагов.
Для такой паузы сначала находят её период или условие, затем захватывают
нужный интервал вместе с предшествующими событиями.

<a id="cpu-py-spy"></a>

## CPU: вычисление, ожидание и процессы DataLoader

Если GPU часто остаётся без работы, нужно понять, кто задерживает следующую
порцию команд. `py-spy` периодически считывает стеки Python-процесса, не требуя
оборачивать каждую функцию таймером. Для уже запущенной программы полезны:

```bash
py-spy top --pid PID
py-spy record --idle --subprocesses -o profile.svg --pid PID
```

`PID` заменяют идентификатором своего процесса; возможность подключения
зависит от прав операционной системы. `--subprocesses` позволяет включить
дочерние Python-процессы, что важно для многопроцессного DataLoader.
`--idle` включает стеки ожидающих потоков: по умолчанию `py-spy` старается
исключать их из выборки. Без этого флага отсутствие функции ожидания в отчёте
не доказывает отсутствия ожидания в программе. Ограничения подключения,
нативных стеков и определения простаивающих потоков перечислены в
[документации py-spy](https://github.com/benfred/py-spy#frequently-asked-questions).

Широкий участок декодирования изображения в профиле рабочего процесса
поддерживает гипотезу о дорогом преобразовании. Долгое ожидание очереди
в главном процессе говорит лишь о том, что пакет не готов: причина может
находиться в хранилище, рабочем процессе или последовательном участке
формирования пакета. Эти наблюдения нужно сопоставлять. Увеличение числа
workers не поможет, если все они упираются в пропускную способность одного
диска, и способно ухудшить ситуацию при нехватке памяти.

Выборочный профилировщик может пропустить очень короткую функцию; зато он
удобен для длительной программы и приблизительного распределения времени.
`cProfile` учитывает вызовы функций с другой детализацией и другой ценой
инструментирования. Ни один Python-профиль сам по себе не показывает,
насколько эффективно CUDA-ядро использует память или вычислительные блоки.
CPU-профиль объясняет подготовку работы, а GPU-профиль — её исполнение.

<a id="nsight-systems"></a>

## Nsight Systems: восстановить причинную цепочку

Когда нужно сопоставить CPU-потоки, CUDA API, ядра, streams и передачи данных,
используют Nsight Systems. Важно различать дорожку **CUDA API** — работу
процессора с CUDA runtime — и дорожку **CUDA kernels**, где показано исполнение
на устройстве. Длительный `cudaStreamSynchronize` на CPU и длинное ядро на GPU
могут быть двумя наблюдениями одного и того же ожидания, а не двумя
последовательными расходами времени.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/a2-nsight-trace.png]]

*Оригинальная временная шкала из Stanford CS336, авторы курса,
[Assignment 2, Figure 1, p. 5](https://github.com/stanford-cs336/assignment2-systems/blob/ca8bc81a59b70516f7ebb2da4808daade877c736/cs336_assignment2_systems.pdf#page=5).
Верхние дорожки показывают аппаратные метрики; ниже видны ядра CUDA,
NVTX-диапазоны и границы модулей PyTorch. Использован существующий crop
исходной страницы, без перерисовки; лицензия MIT. Это профиль примера курса,
не новый эксперимент Bookvar.*

На рисунке сначала полезно найти `BasicTransformerLM.layers.11` и
`BasicTransformerLM.layers.12` в нижней части. Их границы позволяют сопоставить
повторяющиеся участки модели с группами ядер выше. Затем можно сравнить
`GPU Active`, `SMs Active`, `SM Warp Occupancy` и `DRAM Bandwidth`: почти
непрерывная активность GPU не делает все эти показатели максимальными.
Наконец, узкие разрывы в дорожке ядер нужно увеличивать и связывать
с событиями хоста. По одному этому обзорному снимку нельзя доказать ни
конкретную причину разрыва, ни вычислительный или memory-bound режим ядра.

### NVTX как связь между моделью и timeline

Профилировщик не знает, что автор считает «одним блоком Transformer».
NVTX-метки задают смысловые границы в программе. Например, этот фрагмент
помечает вызов уже определённой функции `attention`:

```python
torch.cuda.nvtx.range_push("attention.forward")
try:
    y = attention(q, k, v)
finally:
    torch.cuda.nvtx.range_pop()
```

Длительность диапазона на CPU не равна времени вычислений на GPU. CUDA-вызов
может вернуться до их окончания. В Nsight Systems связь устанавливают через
вызовы запуска, соответствующие ядра и зависимости streams; NVTX помогает
найти эти события и отнести их к нужной части модели.

Для ограниченного окна можно использовать CUDA Profiler API. В следующем
фрагменте `run_steps` — уже определённая функция, исполняющая выбранные шаги
нагрузки. Её первый вызов служит прогревом, второй — исследуемым окном:

```python
run_steps()  # Включая нужные формы и первоначальную компиляцию.
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
try:
    run_steps()
finally:
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
```

Скрипт запускают, предварительно создав каталог `artifacts`:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi \
  --output=artifacts/transformer-step \
  python benchmark.py
```

`--capture-range=cudaProfilerApi` требует вызовов `cudaProfilerStart` и
`cudaProfilerStop`. Одни NVTX-метки при такой настройке не начинают запись.
Результат — файл `.nsys-rep`, который открывают в Nsight Systems.
Оба способа ограничения окна — через CUDA Profiler API и через NVTX —
описаны в [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#focused-profiling).
Синхронизации выше задают границы окна, но не вставляются между каждым ядром:
иначе само измерение уничтожит исследуемое перекрытие.

### Типовые рисунки на временной шкале

Если перед началом каждого пакета GPU простаивает, а на CPU растягивается
получение данных, проверяют конвейер из S6: чтение, декодирование, очередь
workers и формирование пакета. Если данные готовы, но перед вычислениями
остаётся длинное H2D-копирование, исследуют передачу. `pin_memory=True`
и `non_blocking=True` создают предпосылки для асинхронного пути, но сами
по себе не доказывают перекрытия копирования с вычислениями. Копирование
и зависимое от него ядро всё равно должны исполняться в правильном порядке;
выигрыш требует работы, которую можно выполнять одновременно.

Другой рисунок — цепочка коротких ядер с промежутками, пока CPU готовит
следующий запуск. Здесь первыми кандидатами становятся уменьшение числа
операторов, объединение вычислений, компиляция и, при подходящем стабильном
режиме, CUDA Graphs. Увеличение пакета тоже может повысить отношение полезной
работы к цене запуска, но меняет задержку, расход памяти и иногда сам режим
обучения. Это уже изменение нагрузки, а не бесплатная настройка.

Чередование вычисления, D2H-копирования и блокирующего ожидания заставляет
искать обращения к данным GPU из Python: `.item()`, перевод на CPU, печать
тензоров, ветвление по значению. Не всякую синхронизацию можно убрать —
результат действительно может быть нужен хосту. Но часто метрику можно
накапливать на GPU и читать реже, сохранив нужную семантику логирования.

Если ядра идут плотно, исчезновение промежутков уже не является главным
резервом. Следующий вопрос — насколько полезна и эффективна исполняемая
работа: нет ли лишнего padding, повторного чтения больших тензоров,
неудачного алгоритма или неподходящего численного формата. Для нескольких
GPU добавляется ожидание коллективных операций. Долгая коммуникация
может объясняться не только сетью, но и тем, что один rank поздно подготовил
данные. Для такого вывода одного локального trace недостаточно: сопоставляют
согласованные окна всех участвующих процессов.

## Три GeLU как учебный профиль

В [Stanford CS336 Lecture 6](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_06.py)
сравниваются три реализации приближения GeLU через `tanh`: цепочка операций
PyTorch, встроенная функция и скомпилированная цепочка. Для встроенного
варианта существенно указать `approximate="tanh"`: значение по умолчанию
использует другую формулу.

```python
def naive_gelu(x):
    return 0.5 * x * (
        1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x))
    )


def builtin_gelu(x):
    return torch.nn.functional.gelu(x, approximate="tanh")


compiled_gelu = torch.compile(naive_gelu)
```

В сохранённом профиле курса на входе $16384\times16384$ исходная композиция
запускает девять ядер. Встроенная функция и скомпилированный граф в том
эксперименте обходятся одним ядром каждый. Это наблюдение конкретного
исполнения, а не гарантия для любой версии PyTorch и любого устройства.
Одна формула может превращаться либо в цепочку отдельных проходов
по данным, либо в одну объединённую операцию.

В первом случае промежуточные тензоры материализуются в глобальной памяти,
а следующие операции снова их читают. После объединения промежуточные
значения могут оставаться внутри ядра. Для маленького входа особенно важна
цена девяти запусков вместо одного; для большого — ещё и перемещение
промежуточных данных. По одной таблице времени нельзя раздельно измерить
вклад этих двух причин. Число запусков видно в trace, а фактический трафик
памяти уточняется аппаратными счётчиками. Подробный разбор объединения
и реализации на Triton продолжается в S8.

Проверка такого изменения состоит не только в сравнении скоростей. На
представительных входах сравнивают выходы и, для обучения, градиенты;
фиксируют допустимые абсолютную и относительную ошибки. Затем отдельно
измеряют первоначальную компиляцию и установившийся режим. Нельзя включить
первый вызов `compiled_gelu` в одну серию, исключить инициализацию встроенной
функции из другой и приписать всю разницу качеству ядер.

<a id="от-python-функции-до-kernel-где-находится-framework"></a>

## Что происходит между Python-вызовом и запуском ядра

Вызов слоя проходит несколько уровней. Диспетчер выбирает реализацию по
операции, устройству и типам аргументов. Автоматическое дифференцирование
сохраняет необходимые значения и строит вычисления обратного прохода.
Если применяется компиляция, граф преобразуется: сложные операции могут
раскладываться на более простые, а соседние — объединяться. После этого
исполняющая среда управляет запусками, streams и памятью. Поэтому ускорение
может происходить без изменения математической архитектуры модели.

<!-- BEGIN REVIEW: training step 2026-10-27 -->
### A complete training step: from tensors to profiler events

For a two-layer MLP, the forward pass computes a hidden activation and then
class scores. The backward pass needs matrix products too, but their operands
and shapes differ from those in the forward pass. A profiler records those
individual operations. To identify one, follow the derivative it computes.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/training-step-2026-10-27/cs231n-two-layer.jpeg]]

*Andrej Karpathy, Stanford CS231n,
[Neural Networks Part 1, Layer-wise organization](https://cs231n.github.io/neural-networks-1/#layers).
Complete original image, unchanged; [MIT License](/licenses/cs231n-MIT.txt).
Three inputs feed four hidden units, which feed two output scores. The input
coordinates do not count as a trainable layer. The worked model below has
the same two weight transformations, with widths 784–256–10; bias terms are
included in the code but not drawn in this source diagram.*

The following is the complete original `Training Step Anatomy` fragment from
Vijay Janapa Reddi and Harvard ML Systems contributors,
[ML Frameworks](https://github.com/harvard-edge/cs249r_book/blob/2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd/book/quarto/contents/vol1/frameworks/frameworks.qmd#L4275),
CC BY-NC-SA 4.0. It is a source fragment, not a standalone runnable program:
imports, parameters and optimizer initialization are omitted. It also writes
the label size as `(32)`, which is an integer; the
[`torch.randint` size argument](https://docs.pytorch.org/docs/2.14/generated/torch.randint.html)
requires a tuple such as `(32,)`. The fragment is reproduced unchanged;
the next example supplies a corrected, executable program.

```python
# Single training step for a 2-layer MLP
x = torch.randn(32, 784, device="cuda")  # Input batch
y = torch.randint(0, 10, (32), device="cuda")  # Labels

# Forward pass
h = torch.relu(x @ W1 + b1)  # Hidden layer
logits = h @ W2 + b2  # Output layer
loss = F.cross_entropy(logits, y)

# Backward pass
loss.backward()

# Parameter update
optimizer.step()
```

#### Forward values and parameter gradients

The executable variant keeps the source dimensions: 32 examples, 784 input
features, 256 hidden units and 10 classes. It uses CPU and float64 so the
gradient calculations can be checked without a CUDA device. Data generation
and parameter initialization happen before the step. The following blocks
continue the same program.

```python
from math import sqrt

import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, D, H, C = 32, 784, 256, 10
dtype = torch.float64
x = torch.randn(B, D, dtype=dtype)
y = torch.randint(0, C, (B,))
W1 = torch.nn.Parameter(torch.randn(D, H, dtype=dtype) / sqrt(D))
b1 = torch.nn.Parameter(torch.zeros(H, dtype=dtype))
W2 = torch.nn.Parameter(torch.randn(H, C, dtype=dtype) / sqrt(H))
b2 = torch.nn.Parameter(torch.zeros(C, dtype=dtype))
parameters = (W1, b1, W2, b2)
optimizer = torch.optim.SGD(parameters, lr=0.01, foreach=False)

optimizer.zero_grad(set_to_none=True)
a = x @ W1 + b1
h = torch.relu(a)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, y)
assert h.shape == (B, H) and logits.shape == (B, C)
loss.backward()
```

Here $A=XW_1+b_1$ and $H=\operatorname{ReLU}(A)$ have shape $32\times256$;
the logits $Z=HW_2+b_2$ have shape $32\times10$. The labels are integer
class indices, not a matrix of target probabilities. With the unweighted
mean cross-entropy used here, the derivative for one score is

$$
\begin{aligned}
P&=\operatorname{softmax}(Z),\\
G_{ij}&=\frac{P_{ij}-\mathbf1[j=y_i]}{32}.
\end{aligned}
$$

Increasing the target score lowers the loss, so its derivative is negative;
increasing a non-target score raises the loss, so its derivative is positive.
The divisor 32 comes from averaging over examples. Class weights, ignored
labels or a different reduction would change that normalization.
[`CrossEntropyLoss`](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)
documents those cases; the formula above applies to this program.

The second layer receives $G$. Its weight derivative is $H^\top G$; passing
the derivative back to its input gives $GW_2^\top$. ReLU then keeps the entries
whose preactivation was positive and zeros the others:

$$
\begin{aligned}
G_H&=GW_2^\top,\\
G_A&=G_H\odot\mathbf1[A>0].
\end{aligned}
$$

The first layer returns $X^\top G_A$ for its weight. Each bias is shared by
all 32 examples, so its derivative sums the rows of the corresponding
incoming derivative. These are the matrix-layer rules from the
[[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/01 Нейрон, градиент и backpropagation|backpropagation chapter]],
now applied to both layers of one observed step:

```python
with torch.no_grad():
    g_logits = logits.softmax(dim=-1)
    g_logits[torch.arange(B), y] -= 1
    g_logits /= B
    g_h = g_logits @ W2.T
    g_a = g_h * (a > 0)
    expected_gradients = (
        x.T @ g_a, g_a.sum(dim=0),
        h.T @ g_logits, g_logits.sum(dim=0),
    )

for parameter, expected in zip(parameters, expected_gradients):
    torch.testing.assert_close(parameter.grad, expected)
assert x.grad is None
```

The input batch has `requires_grad=False`. The first weight still needs its
input values $X$ to compute $X^\top G_A$, but the program does not request
$G_X=G_AW_1^\top$. The engine can omit that matrix product. A trainable input
or an earlier trainable layer would require it. Saved forward values and
requested input derivatives are different requirements.

`backward()` has populated `.grad`; it has not updated any parameter. Plain
SGD without momentum or weight decay applies $\theta\leftarrow\theta-0.01g$.
Check both operations separately:

```python
before_step = tuple(t.detach().clone() for t in parameters)
step_gradients = tuple(t.grad.detach().clone() for t in parameters)
loss_before = loss.item()
optimizer.step()
for parameter, before, gradient in zip(parameters, before_step, step_gradients):
    torch.testing.assert_close(parameter.detach(), before - 0.01 * gradient)

with torch.no_grad():
    after_logits = torch.relu(x @ W1 + b1) @ W2 + b2
    loss_after = F.cross_entropy(after_logits, y).item()
assert loss_after < loss_before
print(f"{loss_before:.6f} -> {loss_after:.6f}")
```

On PyTorch 2.14.1 CPU, this prints `2.644112 -> 2.476242`.
The smaller loss is a check on this fixed example, not a guarantee for every
SGD learning rate. Before another independent step, `zero_grad` clears the
old derivatives. Omitting it would add the new backward contributions to
the previous ones.

#### Match each matrix product to a recorded event

With a fixed input, there are two forward matrix products and three backward
matrix products in this eager implementation:

| Matrix product | Result shape |
|---|---|
| $XW_1$ — hidden preactivation | $32\times256$ |
| $HW_2$ — scores | $32\times10$ |
| $H^\top G$ — second weight gradient | $256\times10$ |
| $GW_2^\top$ — hidden derivative | $32\times256$ |
| $X^\top G_A$ — first weight gradient | $784\times256$ |

The next function profiles only the step. Cloning the initial values and
creating the local optimizer occur outside the recorded window. Both calls
start from identical parameter values; only the request for an input
derivative changes. The named ranges identify forward, backward and the
optimizer separately. No CUDA activity or elapsed-time result is inferred
from this CPU trace.

```python
from torch.profiler import ProfilerActivity, profile, record_function

def observed_step(input_grad):
    observed_x = x.detach().clone().requires_grad_(input_grad)
    local = tuple(torch.nn.Parameter(t.detach().clone()) for t in before_step)
    w1, bias1, w2, bias2 = local
    local_optimizer = torch.optim.SGD(local, lr=0.01, foreach=False)
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("mlp.zero_grad"):
            local_optimizer.zero_grad(set_to_none=True)
        with record_function("mlp.forward"):
            hidden = torch.relu(observed_x @ w1 + bias1)
            scores = hidden @ w2 + bias2
            observed_loss = F.cross_entropy(scores, y)
        with record_function("mlp.backward"):
            observed_loss.backward()
        with record_function("mlp.optimizer"):
            local_optimizer.step()

    products = []
    for event in prof.events():
        if event.name == "aten::mm":
            parent = event.cpu_parent
            while parent is not None and not parent.name.startswith("mlp."):
                parent = parent.cpu_parent
            assert parent is not None
            shapes = tuple(tuple(shape) for shape in event.input_shapes)
            products.append((parent.name, shapes))
    return products, observed_x.grad

fixed_input_products, no_input_gradient = observed_step(False)
trainable_input_products, input_gradient = observed_step(True)
assert len(fixed_input_products) == 5
assert len(trainable_input_products) == 6
assert no_input_gradient is None and input_gradient.shape == x.shape
for phase, shapes in fixed_input_products:
    print(phase, shapes)
```

The additional event for a trainable input multiplies $G_A$ of shape
$32\times256$ by $W_1^\top$ of shape $256\times784$. The recorded shapes
let us identify that derivative rather than guess from a timing total.
This event list is from eager PyTorch 2.14.1 on CPU; a compiled implementation
may fuse or replace operations.
[`record_shapes`](https://docs.pytorch.org/docs/2.14/profiler.html#torch.profiler.profile)
also adds overhead and can extend tensor lifetimes, so this diagnostic
capture is not an uninstrumented performance measurement.

Count only the matrix arithmetic, using $2mkn$ FLOP for an $m\times k$ by
$k\times n$ product. The two forward products total 13,008,896 FLOP. With no
input derivative, the three backward products total 13,172,736 FLOP, about
1.013 times the forward matrix work. Requesting $G_X$ adds 12,845,056 FLOP,
making the backward matrix work twice the forward matrix work. Bias, ReLU,
cross-entropy and SGD operations are excluded from these totals:

```python
def matrix_work(products):
    totals = {"mlp.forward": 0, "mlp.backward": 0}
    for phase, (left, right) in products:
        m, k = left
        other_k, n = right
        assert k == other_k
        totals[phase] += 2 * m * k * n
    return totals

fixed_work = matrix_work(fixed_input_products)
trainable_work = matrix_work(trainable_input_products)
assert fixed_work == {"mlp.forward": 13_008_896, "mlp.backward": 13_172_736}
assert trainable_work["mlp.backward"] == 2 * trainable_work["mlp.forward"]
assert trainable_work["mlp.backward"] - fixed_work["mlp.backward"] == 12_845_056
```

Five `aten::mm` events are not five CUDA kernel launches. A library operator
can invoke several kernels, while compilation can combine several operators.
The source's six-launch estimate and fixed per-launch latency are assigned
model inputs, not a measured trace of this program. To count device launches,
collect CUDA activity or Nsight data on the target GPU.

On CUDA, the host can record the autograd graph while submitted device work
is still pending. Dependencies inside a stream preserve the order of the
matrix products; returning from a Python call does not prove their completion.
[`CUDA semantics`](https://docs.pytorch.org/docs/2.14/notes/cuda.html#asynchronous-execution)
defines that distinction. The existing source timeline above shows the
separate host and device tracks; the compiler and export discussion below
explains how changing the execution strategy can change their arrangement.

<!-- END REVIEW: training step 2026-10-27 -->

<a id="capture-не-равен-optimization"></a>

В eager-режиме операторы передаются на исполнение по мере работы Python.
Это удобно для отладки и динамического управления, но каждый мелкий
оператор оплачивает диспетчеризацию. Граф позволяет видеть несколько
операторов вместе и согласовывать их исполнение. Однако сам захват графа
ещё не доказывает ускорения. Компилятор должен сохранить семантику,
выбрать подходящие преобразования и сформировать эффективную реализацию.

Разрыв графа (*graph break*) оставляет часть программы вне объединённого
фрагмента. Например, управление, зависящее от значения тензора, и некоторые
обращения к Python-состоянию могут требовать возврата в eager. Проверки
предположений компилятора (*guards*) и новые варианты форм могут вызывать
перекомпиляцию. В профиле поэтому ищут не только более быстрое ядро,
но и границы скомпилированных участков, пропуски между ними и повторные
холодные расходы. Конкретные причины и поддержка меняются между версиями;
их проверяют по журналам компилятора и
[документации `torch.compile`](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.graph_breaks_index.html).

### Fusion как устранение движения данных

Если промежуточный тензор содержит $N$ элементов по $b$ байт, отдельная
запись этого тензора и последующее чтение добавляют до $2Nb$ байт логического
обмена с глобальной памятью. Объединение двух операторов может устранить
эту материализацию и один запуск. Это оценка на уровне алгоритма: реальное
число обращений к HBM зависит также от кэшей и выбранной реализации.

Но большое объединённое ядро не обязательно лучше. Увеличение числа
одновременно нужных значений повышает потребность в регистрах, может
снижать число резидентных warps и приводить к дополнительным обращениям
к памяти. Компилятор иногда повторно вычисляет промежуточный результат,
чтобы не хранить его. Поэтому число ядер — полезное наблюдение, но не
целевая метрика оптимизации. Проверяют задержку, корректность и пик памяти.

Отдельно оценивают окупаемость компиляции. Если её дополнительная стоимость
равна $C$, а устойчивый шаг сокращается с $t_e$ до $t_c$, то при
$t_e>t_c$ грубая граница окупаемости составляет
$n>C/(t_e-t_c)$ шагов. Формула предполагает отсутствие новых компиляций
и неизменность нагрузки. Короткая разовая задача и долгий устойчивый
процесс могут поэтому предпочесть разные режимы исполнения.

Например, дополнительные 8 с компиляции при сокращении шага с 12 до 10 мс
окупаются после 4000 шагов. На 1000 шагах eager затратит 12 с,
а compiled — $8+1000\cdot0{,}010=18$ с. На 4000 шагах оба варианта
дадут 48 с; на 4001-м суммарное время compiled станет меньше.
Время каждого тёплого шага улучшилось сразу, а время всей короткой
задачи — нет. Это расчёт с условными числами, не результат запуска
`torch.compile` из предыдущего примера.

<a id="interoperability-graph-как-контракт-с-границами"></a>

### Что меняется при экспорте модели

Экспорт модели тоже меняет наблюдаемую программу. Checkpoint хранит
параметры для некоторой реализации; экспортированный граф фиксирует
вычисление в определённом наборе операторов; скомпилированный engine
содержит план исполнения с конкретными ограничениями по устройству,
формам и типам чисел. Эти артефакты нельзя считать взаимозаменяемыми.

Например, [ONNX](https://onnx.ai/onnx/intro/concepts.html) описывает граф,
его входы, выходы, типы, параметры и версии операторов. Сам по себе экспорт
этого графа не переносит весь Python-конвейер: токенизацию, подготовку
запроса, выбор следующего токена и пользовательские расширения нужно
учесть отдельно. После смены runtime сравнивают выходы на обычных и крайних
входах, а профиль собирают заново. Если eager-реализация правильна,
а engine — нет, сначала локализуют расхождение между слоями, а не
«оптимизируют» неправильное вычисление.

<a id="memory-snapshot"></a>

## Memory Snapshot: почему память остаётся занятой

У таблицы выделений памяти другой смысл, чем у графика одновременно
занятой памяти. Оператор может много раз выделить большой временный
буфер и сразу его освободить. Сумма выделенных за шаг байт будет высокой,
хотя этот буфер не определяет пик. Наоборот, небольшой тензор, ссылка
на который сохраняется на каждом шаге, способен постепенно заполнить GPU.
Для такого вопроса нужна история времени жизни выделений.

Прежде всего различают три величины. `memory_allocated()` показывает
текущий объём памяти тензоров, учитываемый CUDA-аллокатором PyTorch.
`memory_reserved()` — объём, который аллокатор удерживает у CUDA для
использования, включая свободные для повторного выделения блоки.
Показание общей занятой памяти устройства может дополнительно включать
контексты, другие процессы и выделения библиотек вне аллокатора PyTorch.
Состояния NCCL — типичный пример таких внешних выделений.

Поэтому разность `reserved - allocated` сама по себе не доказывает утечку.
Кэширование блоков позволяет избегать дорогих повторных запросов к CUDA.
И вызов `empty_cache()` не освобождает живые тензоры, на которые сохраняются
ссылки. Для подозрения на утечку сравнивают одно и то же место нескольких
шагов после прогрева: растёт ли объём живой памяти, какие блоки продолжают
существовать и откуда они были выделены. Семантика этих величин описана в
[PyTorch CUDA semantics, Memory management](https://docs.pytorch.org/docs/2.14/notes/cuda.html#memory-management).

Для различия allocated и reserved рассмотрим условный аллокатор, который
удерживает освобождённые блоки для повторного использования. Размеры в таблице
заданы в МиБ; других выделений здесь нет.

| Состояние | Allocated | Reserved |
|---|---:|---:|
| Параметры и постоянное состояние | 100 | 128 |
| Сохранены активации на 60 МиБ | 160 | 192 |
| Добавлен временный буфер на 40 МиБ | 200 | 256 |
| Временный буфер освобождён | 160 | 256 |
| Активации больше не нужны | 100 | 256 |

Пик живой памяти равен 200 МиБ, в конце живыми остаются 100 МиБ.
Оставшиеся 156 МиБ reserved не заняты этими тензорами. Если на следующем
одинаковом шаге границы allocated повторяются, сама по себе эта разность
не показывает утечку. Если allocated в одной и той же точке шага растёт,
ищут сохраняющиеся ссылки и соответствующие им выделения. Реальные размеры
сегментов и возможности их повторного использования определяет аллокатор;
таблица не задаёт правило его округления или обещание выделить из любого
свободного набора байт один сплошной буфер.

### Запись истории и два представления одного снимка

В следующем фрагменте `run_steps` обозначает исследуемые шаги после прогрева.
Синхронизация завершает выбранное окно перед сохранением. История ограничена
числом событий, чтобы длительный запуск не накопил неограниченный журнал:

```python
torch.cuda.memory._record_memory_history(max_entries=100_000)
try:
    run_steps()
    torch.cuda.synchronize()
finally:
    try:
        torch.cuda.memory._dump_snapshot("snapshot.pickle")
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)
```

Имена этих документированных функций начинаются с подчёркивания: интерфейс
остаётся внутренним и может меняться. Перед использованием сверяют сигнатуры
своей версии PyTorch. При переполнении
ограниченного журнала старые события вытесняются. Начинать запись следует
до интересующего участка, а при исследовании холодного пика — до соответствующей
инициализации. Формат и просмотрщик описаны в
[Understanding CUDA Memory Usage, API 2.14](https://docs.pytorch.org/docs/2.14/torch_cuda_memory.html).
Снимок открывают в [Memory Visualizer](https://pytorch.org/memory_viz/).

Закреплённая CPU-память из S6 требует отдельного наблюдения. В PyTorch 2.14
опция `record_pinned_host_memory=True` включает её историю в поля
`host_segments` и `host_traces`. Просмотрщик `memory_viz` пока не отображает
эти поля, поэтому отсутствие CPU-буферов на его диаграмме не означает,
что загрузчик их не создал. Поддержка и ограничение описаны в
[Including Pinned (Host) Memory](https://docs.pytorch.org/docs/2.14/torch_cuda_memory.html#including-pinned-host-memory).

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/profiling-review-2026-10/pytorch-active-memory-timeline.png]]

*PyTorch contributors, оригинальный «Active Memory Timeline» из
[Understanding CUDA Memory Usage](https://docs.pytorch.org/docs/2.8/torch_cuda_memory.html#active-memory-timeline).
Каждая полоса — отдельное выделение, её толщина соответствует размеру,
а протяжённость — времени жизни в показанной истории. Внизу раскрыт стек
выбранного блока. [Исходный PNG, PyTorch v2.8.0](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/docs/source/_static/img/torch_cuda_memory/active_memory_timeline.png),
BSD-style license репозитория PyTorch; файл не изменён.*

На этом рисунке видно постепенное накопление живых выделений, затем их
освобождение. По одному силуэту нельзя подписать точные границы forward
и backward: для этого сопоставляют порядок событий, стеки и этапы собственной
программы. Но изображение уже отвечает на вопрос, которого нет в таблице
операторов: какие большие блоки существовали **одновременно**. Если пик
создают сохранённые для backward активации, можно исследовать их время
жизни, объединение операций или activation checkpointing. Если пик
вызван одним временным буфером, нужны другие действия.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/profiling-review-2026-10/pytorch-allocator-state-history.png]]

*PyTorch contributors, оригинальный «Allocator State History» из
[той же документации](https://docs.pytorch.org/docs/2.8/torch_cuda_memory.html#allocator-state-history).
Слева выбирается событие выделения или освобождения, справа показываются
сегменты и блоки аллокатора в этот момент. [Исходный PNG](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/docs/source/_static/img/torch_cuda_memory/allocator_state_history.png),
BSD-style license PyTorch; файл не изменён. Это состояние из документации,
а не демонстрация ошибки нехватки памяти в нашей нагрузке.*

Это представление нужно для другого вопроса: как свободное место распределено
внутри зарезервированных сегментов. При OOM выбирают соответствующее событие
и проверяют размер неудавшегося запроса и состояние аллокатора. Наличие
свободных байт не всегда означает наличие подходящего блока. Однако диагноз
«фрагментация» требует этой проверки, а не одного большого значения
`memory_reserved()`. Следует также учитывать блоки, освобождение которых
ожидает завершения работы в другом stream.

Для учебной модели полезно сравнить forward с отключённым autograd и полный
шаг обучения при двух длинах контекста. В полном шаге отдельно наблюдают
активации, градиенты и обновление оптимизатора. Параметры, состояния
оптимизатора и сохранённые тензоры имеют разные времена жизни, поэтому
понижение точности части операций не обязано пропорционально уменьшать
весь пик. В [CS336 Assignment 2, §2.1.6, pp. 9–10](https://github.com/stanford-cs336/assignment2-systems/blob/ca8bc81a59b70516f7ebb2da4808daade877c736/cs336_assignment2_systems.pdf#page=9)
эта проверка доводится до конкретных размеров крупнейших выделений и
стеков операций, которые их создали.

## Когда нужен Nsight Compute

К аппаратным счётчикам переходят после выбора конкретного ядра и входа.
На этом этапе вопрос уже узок: например, почему матричное умножение
заданной формы далеко от ожидаемой границы или почему объединённая
поэлементная операция оказалась медленнее исходной.

Nsight Compute позволяет исследовать пропускную способность памяти,
использование вычислительных конвейеров, occupancy, регистры и причины
остановки выдачи инструкций. Эти показатели интерпретируют вместе.
Низкая occupancy не является самостоятельной ошибкой, если доступного
параллелизма хватает для высокой производительности. И плотная дорожка
ядер в Systems не доказывает насыщения Tensor Cores. Счётчики помогают
связать наблюдаемое ядро с моделью roofline и выбрать гипотезу для S8.

Не все счётчики собираются за один проход. Nsight Compute может повторно
исполнять ядро или приложение и восстанавливать состояние памяти;
режим сбора влияет на кэши и конкурентное исполнение. Поэтому длительность
такого сеанса нельзя подставлять вместо задержки обычной программы.
Механизмы и ограничения описаны в
[Nsight Compute Profiling Guide, Replay](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#replay).
После изменения возвращаются к полному шагу без подробного профилирования.

## Условный сквозной пример

Следующие числа — **расчётная иллюстрация**, не замеры GPU и не прогноз
для конкретной машины. Пусть один полностью последовательный шаг длится
420 мс: 110 мс занимает подготовка пакета, 18 мс — передача H2D,
275 мс — ядра и 17 мс — промежутки между запусками. Такое сложение допустимо
только потому, что в условии интервалы не перекрываются и не вложены.

Предположим, CPU-профиль связывает подготовку с рабочими процессами загрузчика,
а временная шкала показывает, что вычисления начинаются только после её
окончания. Первой проверяемой гипотезой становится подготовка следующего
пакета во время вычисления текущего. Если удаётся полностью скрыть эти
110 мс, остальные интервалы не меняются, а заполнением и опустошением
конвейера можно пренебречь, длительность устойчивого шага составит
$\max(110,18+275+17)=310$ мс. Это идеализированный ориентир, с которым
сравнивают реальный эксперимент, а не обещанный эффект включения prefetch.

После этого собственно подготовка данных перестаёт определять критический
путь. Даже бесконечно быстрый загрузчик не сократит оставшиеся 310 мс.
Зато устранение всех 17 мс промежутков могло бы дать нижнюю границу 293 мс,
если остальные расходы неизменны. Если объединение операций дополнительно
сокращает обмен с памятью, изменится и время ядер; это отдельная гипотеза,
которую проверяют новым профилем.

В настоящем эксперименте шаги имеют распределение длительности. Медианы
частей нельзя механически сложить и назвать медианой суммы, а p95 отдельных
стадий — p95 полного шага. Итоговую задержку и полезные токены в секунду
измеряют непосредственно, без профилировщика, на одинаковой последовательности
нагрузки. Помимо средней или медианы сохраняют хвост распределения:
оптимизация обычного шага может не устранить редкие паузы чтения или компиляции.

## Воспроизводимый цикл

Профилирование должно завершаться проверенной причинной гипотезой.
«Много времени в `aten::mm`» — наблюдение. «Матрица такой формы использует
неподходящую реализацию; изменение раскладки уменьшит время полного шага
без изменения результата» — проверяемая гипотеза. Между ними нужны данные
о формах, временной шкале и, при необходимости, аппаратных счётчиках.

Практический цикл состоит из шести шагов:

1. Зафиксировать нагрузку и получить исходное распределение времени без
   подробного сбора событий.
2. Собрать короткий представительный профиль и найти интервал, который
   действительно задерживает результат.
3. Сформулировать одну гипотезу и ожидаемый наблюдаемый эффект: исчезнувшее
   ожидание, меньше передач, меньше запусков или более короткое ядро.
4. Внести одно изменение и проверить корректность, включая градиенты для
   обучаемой операции и граничные формы входов.
5. Собрать профиль, подтверждающий изменение механизма, и отдельно повторить
   измерение производительности без профилировщика.
6. Сравнить полный шаг, полезный объём работы и пик памяти; принять или
   отклонить гипотезу по этим данным.

Вместе с trace сохраняют commit кода, команду и конфигурацию, формы и типы
входов, число полезных токенов и режим forward/backward/optimizer.
Для среды указывают GPU, driver, CUDA, PyTorch, compiler и версии используемых
библиотек; для распределённого запуска — host и rank. Для измерения
нужны число шагов прогрева, окно сбора, активные опции профилировщика,
правила синхронизации, состояние частот и ограничения мощности, а также
исходные результаты до и после изменения. Это позволяет отличить эффект
кода от смены среды или нагрузки.

Артефакты разных инструментов сохраняют раздельно: JSON временной шкалы,
`.nsys-rep`, снимок памяти `.pickle` и результаты обычного измерения.
Trace может содержать пути исходников, формы и пользовательские NVTX-метки,
поэтому перед публикацией проверяют отсутствие конфиденциальных данных.
Недоверенные pickle-файлы нельзя загружать произвольным Python-кодом.
Версию формата тоже фиксируют: изменившиеся имена операторов и API
снимка памяти не должны выглядеть как изменение самой модели.

## Практика и первоисточники

Для первой лабораторной достаточно одной модели и одного вопроса.
Возьмите одинаковые входы и сравните полный шаг с данными, уже размещёнными
на устройстве, и шаг с реальным конвейером. Пометьте получение пакета,
передачу, forward, backward и обновление оптимизатора. Объясните один
наблюдаемый промежуток через события CPU и GPU, измените его причину и
покажите отдельное измерение полного шага. Вторая часть — два снимка памяти
в одинаковой точке устойчивого шага и объяснение крупнейших живых выделений.
Если GPU недоступен, CPU-экспорт trace остаётся полезной проверкой кода,
но не считается выполнением CUDA-части лабораторной.

- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week02_fast_pipelines/lecture.pdf|EDLS week 2, лекция Max Ryabinin]] и [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week02_fast_pipelines/seminar/practice.ipynb|практика]] — переход от CPU-профиля к PyTorch, памяти и Nsight; [оригинальный репозиторий, закреплённая версия](https://github.com/mryab/efficient-dl-systems/tree/e632aa89ca9e6638d52e1b686095e7442faffbb0/week02_fast_pipelines).
- [Stanford CS336 Lecture 6](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_06.py) — различие измерения и профиля, выбор ядра по форме, три реализации GeLU.
- [Stanford CS336 Assignment 2, §§2.1.3–2.1.6](https://github.com/stanford-cs336/assignment2-systems/blob/ca8bc81a59b70516f7ebb2da4808daade877c736/cs336_assignment2_systems.pdf#page=3) — воспроизводимое измерение, NVTX-разметка, разбор временной шкалы и памяти.
- [PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html), [Memory Snapshot](https://docs.pytorch.org/docs/2.8/torch_cuda_memory.html), [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html), [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) — точная семантика инструментов и их ограничения.
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week06_dl_arithmetic/seminar/practice.ipynb|EDLS week 6 — практика арифметики и профилирования]], [[05 Источники/Courses/Harvard ML Systems/tinytorch/14_profiling|TinyTorch 14 — Profiling]], [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol2/performance_engineering|Harvard CS249r — Performance Engineering]] и [[05 Источники/Courses/Harvard ML Systems/vol1/frameworks|ML Frameworks]] — дополнительные задачи и подробные исходные главы о производительности и уровнях исполнения.

Связанные механизмы: [[01 Справочник/Inference/Triton и GPU kernels|Triton и GPU kernels]],
[[01 Справочник/Inference/FlashAttention|FlashAttention]],
[[01 Справочник/FFN и MoE/Dense FFN и gated activations|FFN и функции активации]].
Профили блоков применимы к [[02 Атлас моделей/Семейства/T5|T5]] и другим
Transformer-моделям, но формы, алгоритмы и последовательность событий
нужно получать из конкретной реализации.

← [[02 Areas/ML & DL/00 Учебник/10 ML Systems/06 Data pipeline, padding и packing|Data pipeline, padding и packing]] ·
[[02 Areas/ML & DL/00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению|GPU kernels и Triton]] →
