---
title: Data pipeline, padding и packing
type: textbook-chapter
status: canonical
last_updated: 2026-10-28
last_verified: 2026-10-28
source_language: mixed
---

# Data pipeline, padding и packing

Предположим, модель обрабатывает порцию данных за 20 мс, а чтение и подготовка
следующей порции занимают 28 мс. Если выполнять эти действия последовательно,
между двумя обновлениями параметров пройдёт не меньше 48 мс. Более быстрое
матричное умножение сократит только первую часть этого времени. Чтобы GPU не
ждал данные после каждого шага, следующую порцию нужно готовить заранее.

Есть и другая причина лишних вычислений. Четыре текста длиной 128, 140, 160 и
1024 токена содержат 1452 токена. Если поместить их в прямоугольную матрицу
`[4, 1024]`, в ней окажутся 2644 позиции-заполнителя. Даже непрерывно работающий
GPU может большую часть шага обрабатывать не текст, а дополнение до общей
длины — *padding*.

Это две разные задачи. Конвейер загрузки должен вовремя доставлять данные,
а формирование порций — давать модели достаточно полезной работы. При этом
ускорение не должно менять обучающую задачу: перемешивание, границы документов,
маска внимания и вклад каждого токена в функцию потерь остаются частью
алгоритма обучения.

## Что хранить

До запуска модели документ проходит несколько преобразований: чтение байтов,
декодирование текста, очистку, токенизацию и добавление служебных токенов.
Если корпус используется много раз, повторять одинаковую токенизацию на
каждом проходе необязательно. Можно один раз сохранить последовательности
целочисленных идентификаторов, а при обучении читать готовые массивы.
Случайные преобразования, которые должны меняться между эпохами, таким способом
не фиксируют: иначе изменение формата хранения незаметно изменит обучение.

Много отдельных маленьких файлов удобно изучать вручную, но открытие каждого
файла, запрос метаданных и сетевой обмен могут стоить больше чтения его
содержимого. Поэтому примеры часто объединяют в крупные части корпуса —
*шарды*. Внутри шарда чтение идёт последовательно, а между шардами можно
распределять работу. Слишком крупные шарды, в свою очередь, затрудняют
балансировку работников и возобновление с середины потока.

Формат выбирают под доступ к данным. Arrow удобен для типизированных таблиц;
Parquet позволяет читать выбранные столбцы и группы строк, но требует
декодирования сжатых блоков. Для уже токенизированного текста возможен простой
массив целых чисел с индексом границ документов. `mmap` позволяет обращаться
к файлу как к массиву без предварительного чтения всего корпуса в память,
но не делает доступ бесплатным: отсутствующая страница всё равно должна
прийти с диска или по сети. Protobuf и MessagePack решают задачу сериализации
записи; сами по себе они не задают эффективное разбиение большого корпуса.

При сжатии передаётся меньше байтов, зато CPU должен восстановить содержимое.
Если ограничение — сеть, это может помочь; если декодирование уже задерживает
обучение, дополнительное сжатие способно ухудшить результат.

Вместе с массивами сохраняют описание корпуса: контрольные суммы, тип целых
чисел и порядок байтов, границы документов, версию токенизатора и правила
BOS/EOS. Например, `uint16` подходит только тогда, когда все используемые
идентификаторы, включая специальные токены, помещаются в диапазон 0–65535.
Сам файл с числами не позволяет восстановить словарь или узнать, было ли
между документами добавлено EOS.

## Декодирование изображений и DALI

В мультимодальном обучении после чтения файла нужно ещё получить массив
пикселей, изменить размер, выбрать фрагмент, применить случайные преобразования
и нормировку. Измерение одного `Image.open` может не включать эту работу:
в семинаре EDLS вариант PIL вызывает также `convert("RGB")` и `np.asarray`.
Вариант OpenCV переводит BGR в RGB. `torchvision.io.read_image` возвращает
тензор с осями `[C, H, W]`: в ячейке ноутбука его ещё превращают в NumPy-массив
и переставляют оси в `[H, W, C]`, а в отдельном скрипте оставляют тензором.
Поэтому перед сравнением времени проверяют, какой результат каждый путь
действительно подготовил для модели.

[Семинар EDLS, §4–5](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week02_fast_pipelines/seminar/practice.ipynb)
содержит шесть способов чтения и пример DALI. Отдельный
[скрипт сравнения](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week02_fast_pipelines/seminar/image_loaders_benchmark.py)
несколько раз читает выбранный набор файлов и считает изображения в секунду.
Повторное чтение может обслуживаться файловым кэшем ОС. Эти результаты
характеризуют весь измеряемый вызов в заданном окружении, а не только
декодер и не предельную полосу диска. Числа из вывода ноутбука нельзя
переносить на другой корпус или считать ускорением полного обучения.

В исходном примере DALI вызов `fn.readers.caffe` читает закодированные
изображения из LMDB. Затем
`fn.decoders.image(encoded, device="mixed", output_type=types.RGB)`
принимает CPU-данные и возвращает изображения на GPU. Слово `mixed`
не означает отсутствие CPU-работы: конкретный путь декодирования зависит
от формата и оборудования. Следующие операции получают GPU-изображения
и могут выполнять преобразования без возврата массива на CPU.
Размещение входов и выходов определено в
[документации графа DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/pipeline.html#processing-graph-structure),
ограничения декодера — в
[описании `fn.decoders.image`](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.decoders.image.html).

Функция с `@pipeline_def` описывает граф при построении конвейера;
это не цикл, который Python заново исполняет для каждого изображения.
В семинаре `pipe.build()` подготавливает конвейер, а `pipe.run()`
получает очередную порцию его результатов. `num_threads` относится
к CPU-потокам DALI, `device_id` выбирает GPU. Эти параметры не равны
`num_workers` обычного `DataLoader`.

Место изменения размера влияет и на объём передачи. Порция из 32 RGB-изображений
640 × 640 в `uint8` занимает 37.5 МиБ без учёта служебной памяти.
Если сначала на CPU получить изображения 224 × 224 и оставить `uint8`,
передавать нужно около 4.59 МиБ. Если перед передачей преобразовать их в FP32,
получится около 18.38 МиБ. `mixed`-декодер распределяет работу между
CPU и устройством; его обмен зависит от формата и выбранного бэкенда,
включая возможный CPU fallback. Этот обмен нельзя приравнять к простому
копированию одного из трёх RGB-массивов. Одинаковое число картинок
ещё не означает одинаковую нагрузку H2D.

Перенос преобразований на GPU убирает часть CPU-работы, но добавляет ядра,
промежуточные массивы и обращения к памяти устройства. Если модель уже
ограничена ресурсами GPU, это может замедлить шаг. Сравнивают полный цикл
при одинаковых правилах выбора фрагмента, цветовых каналов, нормировки
и случайных преобразований. Профиль должен показать, сократилось ли ожидание
данных и не выросло ли время модели. Запуск галереи преобразований из
семинара сам по себе такого сравнения не даёт.

<a id="storage-и-dataloader-как-система"></a>

## Как загружать

В PyTorch разделены доступ к отдельному примеру и сборка порции данных.
`Dataset` описывает получение примера, `sampler` — выбор индексов,
`batch_sampler` — их группировку, а `collate_fn` превращает выбранные примеры
в общие тензоры. `DataLoader` организует выполнение этой работы и передачу
результата обучающему процессу. Для потокового `IterableDataset` вместо
индексации используется последовательное чтение.

При `num_workers=0` чтение, преобразования и `collate_fn` выполняются в
обучающем процессе. Несколько работников позволяют готовить разные порции
параллельно с вычислениями модели. Но каждый работник требует памяти и времени
CPU; избыточное число процессов может усилить конкуренцию за диск, каналы памяти
и внутренние потоки библиотек. Если данные уже находятся в RAM, а обработка
сводится к взятию небольшого среза, обмен между процессами иногда дороже
самой работы.

Предварительная загрузка — *prefetch* — отделяет момент подготовки порции от
момента её использования. В `DataLoader` параметр `prefetch_factor` задаёт
число заранее загружаемых порций на работника. Например, восемь работников
с коэффициентом 2 могут держать в работе до 16 таких задач. Если одна готовая
порция занимает 64 МиБ, только их содержимое даёт порядок 1 ГиБ памяти; сюда
ещё не входят исходные записи, временные массивы и дополнительные копии.
Это оценка, а не точная формула RSS процесса.

Очередь сглаживает неравномерность чтения, но не повышает предельную скорость
медленной стадии. В примере из начала главы подготовка за 28 мс и вычисление
за 20 мс могут при идеальном перекрытии давать одну порцию примерно каждые
28 мс после заполнения конвейера. Первую порцию всё равно нужно подготовить.
Если подготовка устойчиво медленнее потребления, любая заранее заполненная
очередь рано или поздно опустеет.

Для первых трёх порций расписание с одним последовательно работающим
загрузчиком выглядит так. Время отсчитывается от начала чтения;
передачу на GPU и другие расходы здесь отдельно не учитываем.

| Порция | Подготовка, мс | Вычисление, мс |
|---|---:|---:|
| 1 | 0–28 | 28–48 |
| 2 | 28–56 | 56–76 |
| 3 | 56–84 | 84–104 |

Три порции завершаются за 104 мс вместо $3(28+20)=144$ мс
при последовательном выполнении. Между вычислениями остаются паузы
по 8 мс: очередь не устранила ограничение загрузчика. Для $M$ порций,
постоянных времён подготовки $p$ и вычисления $g$ эта двухстадийная
модель даёт $p+(M-1)\max(p,g)+g$. Несколько работников могут изменить
скорость подготовки, но совместное чтение с одного диска не обязано
ускоряться пропорционально их числу.

`persistent_workers=True` сохраняет процессы между проходами по загрузчику,
чтобы не создавать их заново. Это полезно при дорогой инициализации, но означает,
что их локальное состояние тоже сохраняется. Изменение поля `dataset.epoch`
в основном процессе не обязано автоматически попасть в копии работников.
Ограниченная очередь, контроль памяти и явная передача состояния важнее
максимального числа предварительно загруженных порций.
Поведение параметров определяет [документация DataLoader 2.14](https://docs.pytorch.org/docs/2.14/data.html).

Порядок выдачи тоже влияет на результат. При `in_order=True` загрузчик
сохраняет заданный порядок, даже если одна порция готовилась дольше
следующей. `in_order=False` разрешает выдавать готовые порции раньше.
Если короткие примеры или данные одного источника обрабатываются быстрее,
они могут чаще попадать в начало обучения. При ограничении числа шагов
изменится даже увиденная часть корпуса. Поэтому эту настройку проверяют
вместе с распределением длин и источников, а не только по порциям в секунду.

<!-- BEGIN REVIEW: tf.data cache 2026-10-28 -->
<a id="cache-placement-and-epoch-state"></a>

## Caching across epochs

A corpus used for several epochs may pass through the same tokenizer or image
decoder repeatedly. A cache stores the elements produced at a chosen point in
the input pipeline. Operations before that point can be skipped on later
iterations; operations after it still run. The placement also determines which
ordering and random choices become part of the stored data.

<!-- BEGIN SOURCE: TensorFlow caching intro -->
> The `tf.data.Dataset.cache` transformation can cache a dataset, either in memory or on local storage.
> This will save some operations (like file opening and data reading) from being executed during each epoch.
<!-- END SOURCE: TensorFlow caching intro -->

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/tf-data-2026-10-28/cached-dataset.svg]]

*Scroll horizontally, or open the figure to inspect the full timeline.
The TensorFlow Authors, [Better performance with the tf.data API, Caching](https://www.tensorflow.org/guide/data_performance#caching);
[original notebook, cells 36–38](https://github.com/tensorflow/docs/blob/e4210b940833a4e4e7c5d35a19ca46cafc27c7c5/site/en/guide/data_performance.ipynb).
Copyright 2019 The TensorFlow Authors;
[Apache 2.0, full license](/licenses/tensorflow-docs-e4210b94-Apache-2.0.txt).
Complete original SVG and original English text, without crop, redrawing or
translation. The notebook uses deliberate sleeps to simulate file opening,
reading, mapping and training; this diagram is not a GPU throughput measurement.*

<!-- BEGIN SOURCE: TensorFlow caching explanation -->
> Here, the data execution time plot shows that when you cache a dataset, the transformations before the `cache` one (like the file opening and data reading) are executed only during the first epoch.
> The next epochs will reuse the data cached by the`cache` transformation.
>
> If the user-defined function passed into the `map` transformation is expensive, apply the `cache` transformation after the `map` transformation as long as the resulting dataset can still fit into memory or local storage.
> If the user-defined function increases the space required to store the dataset beyond the cache capacity, either apply it after the `cache` transformation or consider pre-processing your data before your training job to reduce resource usage.
<!-- END SOURCE: TensorFlow caching explanation -->

The upper three rows show `Open`, `Read` and `Map` work during the first epoch.
Later epochs still contain `Train` intervals, but do not repeat those upstream
operations. Caching therefore differs from prefetching: prefetching overlaps
work that is still performed, while a completed cache reuses stored elements.
A cache can also sit before a prefetch buffer; the two mechanisms have separate
memory costs.

### Count executed transformations

The following CPU example uses TensorFlow 2.21.0. Squaring four integers stands
in for deterministic preparation. A TensorFlow variable counts executions of
the mapped operation; a Python counter inside the function would also be
affected by graph tracing and would not count the runtime calls reliably.
The single-thread options keep this small demonstration's execution explicit;
they are not recommended throughput settings.

```python
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
options = tf.data.Options()
options.threading.private_threadpool_size = 1
options.experimental_optimization.apply_default_optimizations = False

def numbers(n):
    return tf.data.Dataset.range(n).with_options(options)

def collect(dataset):
    return [int(x) for x in dataset.as_numpy_iterator()]

decode_calls = tf.Variable(0, dtype=tf.int64)

@tf.autograph.experimental.do_not_convert
def prepare(x):
    with tf.control_dependencies([decode_calls.assign_add(1)]):
        return x * x

prepared = numbers(4).map(prepare).cache()
assert collect(prepared) == [0, 1, 4, 9]
assert int(decode_calls.numpy()) == 4
assert collect(prepared) == [0, 1, 4, 9]
assert int(decode_calls.numpy()) == 4

decode_calls.assign(0)
raw_cached = numbers(4).cache().map(prepare)
assert collect(raw_cached) == collect(raw_cached) == [0, 1, 4, 9]
assert int(decode_calls.numpy()) == 8
```

Both pipelines return the same values. In the first, `prepare` runs four times
to fill the cache and zero times on the second pass. In the second, the cache
contains raw IDs, so `prepare` runs on every pass: eight executions in total.
The counters establish reuse, without timing an integer multiplication or
claiming that a particular real tokenizer becomes twice as fast.

### Keep epoch-dependent work after the cache

If an image crop, masking decision or ordering is produced before `cache`, the
stored element includes that decision. A new epoch does not rerun the upstream
function. To make this visible without relying on two random draws being
different, the next example adds a known offset controlled by the epoch.
This is a deterministic stand-in for epoch-dependent preprocessing, not an
image augmentation algorithm.

```python
epoch = tf.Variable(0, dtype=tf.int64)

@tf.autograph.experimental.do_not_convert
def epoch_dependent(x):
    return x + 100 * epoch.read_value()

frozen = numbers(4).map(epoch_dependent).cache()
live = numbers(4).cache().map(epoch_dependent)
assert collect(frozen) == collect(live) == [0, 1, 2, 3]

epoch.assign(1)
assert collect(frozen) == [0, 1, 2, 3]
assert collect(live) == [100, 101, 102, 103]
```

The first pass completed both caches. Changing `epoch` now affects only the
map downstream of the cache. Changing a random seed likewise does not cause
an already cached upstream transformation to execute again. For fresh random
augmentation on each epoch, cache the deterministic representation and apply
the stochastic transformation afterwards. If an augmentation must be fixed
as part of a dataset release, storing its output can instead be intentional.

The same placement issue applies to shuffle. The API states:

> Note: `cache` will produce exactly the same elements during each iteration
> through the dataset. If you wish to randomize the iteration order, make sure
> to call `shuffle` *after* calling `cache`.

*The TensorFlow Authors, [`Dataset.cache`, TensorFlow 2.21.0](https://github.com/tensorflow/tensorflow/blob/a481b10260dfdf833a1b16007eead49c1d7febf3/tensorflow/python/data/ops/dataset_ops.py#L1513-L1570),
Apache 2.0. Original wording.*

```python
fixed_order = numbers(8).shuffle(
    8, seed=7, reshuffle_each_iteration=True).cache()
fixed_epochs = [collect(fixed_order), collect(fixed_order)]
assert fixed_epochs[0] == fixed_epochs[1]

fresh_order = numbers(8).cache().shuffle(
    8, seed=7, reshuffle_each_iteration=True)
fresh_epochs = [collect(fresh_order), collect(fresh_order)]
assert all(sorted(items) == list(range(8)) for items in fresh_epochs)
print("before cache:", fixed_epochs)
print("after cache:", fresh_epochs)
```

Before the cache, the first permutation becomes the stored order. After the
cache, each iteration asks the shuffle operation for another permutation of
the same stored elements. The seed makes the sequence reproducible in this
fixed-version example; random reshuffling does not guarantee that every two
epochs have distinct orders. The buffer contains all eight items here. With
a smaller buffer, the shuffle is local to that buffer, as in the streaming
example later in this chapter.

### Finish filling a finite cache

Receiving the last item requested by a training loop need not exhaust the
input of the cache. TensorFlow's API specifies the completion condition:

> Note: To guarantee that the cache gets finalized, the input dataset must be
> iterated through in its entirety, until it raises StopIteration. Otherwise,
> subsequent iterations may not use cached data.

*The TensorFlow Authors, [`Dataset.cache`, same source](https://github.com/tensorflow/tensorflow/blob/a481b10260dfdf833a1b16007eead49c1d7febf3/tensorflow/python/data/ops/dataset_ops.py#L1513-L1570),
Apache 2.0. Original wording.*

```python
decode_calls.assign(0)
unfinished = numbers(4).map(prepare).cache().take(2)
assert collect(unfinished) == [0, 1]
first_calls = int(decode_calls.numpy())
assert collect(unfinished) == [0, 1]
assert int(decode_calls.numpy()) > first_calls

decode_calls.assign(0)
finished = numbers(4).take(2).map(prepare).cache()
assert collect(finished) == [0, 1]
assert collect(finished) == [0, 1]
assert int(decode_calls.numpy()) == 2

assert int(numbers(4).repeat().cardinality().numpy()) == -1
assert collect(numbers(4).cache().repeat(2)) == [0, 1, 2, 3] * 2
```

In the first pipeline, `take(2)` stops the consumer while the input of `cache`
still has four items. In the tested runtime, that incomplete cache is discarded,
and the next iteration executes `prepare` again. TensorFlow logs a warning
for this case. In the second, `take(2)` defines the finite dataset before
caching, so both values and the end of the input are observed; the next pass
reuses the two prepared elements.

For the same reason, an unbounded `repeat()` before `cache` has no final
end-of-input at which to finish a finite cache. The `-1` above denotes infinite
cardinality, not a negative number of records. Cache the finite input before
repeating it when that is the intended dataset. A training loop with a fixed
`steps_per_epoch` must still establish whether the cache's input was exhausted;
the loop's epoch counter alone does not establish this.

### Persistent cache identity and memory

A file cache can survive the process that created it. Its filename is part of
the data contract: changing the upstream dataset does not invalidate an
existing completed cache under the same filename. The following uses only a
new temporary directory owned by the example.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory(prefix="bookvar-tf-cache-") as directory:
    cache_name = str(Path(directory) / "records-v1")
    assert collect(numbers(3).cache(cache_name)) == [0, 1, 2]
    # A different source with the same filename still reads the old cache.
    assert collect(numbers(5).cache(cache_name)) == [0, 1, 2]
    assert collect(numbers(5).cache(cache_name + "-v2")) == [0, 1, 2, 3, 4]
```

Cache identity therefore includes the corpus release, tokenizer or decoder,
preprocessing configuration and output schema. A versioned filename or a
separate directory can prevent an old cache from silently serving a changed
pipeline. This rule is explicit in the same `Dataset.cache` documentation;
it is not automatic content-addressed invalidation.

The representation at the cache boundary determines capacity. Ignoring record
metadata, 1000 RGB images of shape `[224, 224, 3]` occupy 150,528,000 bytes as
`uint8`, about 143.55 MiB. The same arrays in FP32 occupy 602,112,000 bytes,
about 574.22 MiB. A normalization that creates FP32 arrays before the cache
quadruples that payload; placing it afterwards reduces cached bytes while
repeating the conversion. Decoder intermediates, prefetch/shuffle buffers,
allocator overhead and the model still require separate memory. An in-memory
dataset cache is also distinct from the operating system's file-page cache.
<!-- END REVIEW: tf.data cache 2026-10-28 -->

## Передача из RAM в память GPU

Обычная оперативная память процесса допускает подкачку страниц. Для передачи
на GPU через DMA нужны страницы с устойчивым расположением в физической памяти.
Закреплённая память — *pinned*, или *page-locked memory* — не выгружается в
подкачку. При передаче из обычной памяти драйверу может понадобиться
промежуточная закреплённая область. Это отдельная работа, а не часть
матричного умножения модели.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/data-pipeline-review-2026-10/pinmem.png]]

*Vincent Moens, [PyTorch: A guide on good usage of non_blocking and pin_memory](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html),
BSD-3-Clause. Голубая стрелка обозначает закрепление CPU-тензора, оранжевая —
передачу на GPU. Диск показан как возможное место подкачки: это не означает,
что каждое копирование обязательно читает данные с диска. Исходный рисунок
сохранён без изменений.*

`DataLoader(pin_memory=True)` подготавливает закреплённые CPU-тензоры.
`tensor.to("cuda", non_blocking=True)` позволяет не ждать завершения передачи
после каждого вызова на стороне CPU. Эти настройки относятся к разным
механизмам. Более того, быстрый возврат из Python-вызова ещё не означает, что
копирование и вычисления исполняются одновременно на устройстве.

В одном CUDA stream операции упорядочены: сначала копирование, затем
зависящий от него расчёт. Чтобы передача **следующей** порции перекрывалась
с вычислением **текущей**, используют отдельный поток копирования, независимые
буферы и событие готовности. Перед использованием нового буфера вычислительный
поток должен дождаться соответствующего события. Старый CPU-буфер нельзя
перезаписывать, пока асинхронное чтение из него ещё не закончено; время жизни
GPU-буферов также нужно согласовать с потоками и аллокатором.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/data-pipeline-review-2026-10/same-stream.png]]

*PyTorch, Vincent Moens, [тот же эксперимент: pinned=True, streamed=False](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html#asynchronous-vs-synchronous-operations-with-non-blocking-true-cuda-cudamemcpyasync),
BSD-3-Clause. На GPU копирование и арифметика находятся в одном потоке и идут
последовательно. Верхняя CPU-дорожка сама по себе этого не объясняет: нужно
смотреть нижние дорожки устройства. Это исходный снимок эксперимента автора.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/data-pipeline-review-2026-10/separate-stream.png]]

*Тот же источник, pinned=True, streamed=True. На дорожке Stream 13 находится
`Memcpy HtoD`, а на Stream 7 — независимая арифметика. Их интервалы пересекаются
по горизонтальной оси времени. В этом опыте они работают с разными тензорами:
ядро не читает ещё не скопированный вход. Снимок демонстрирует возможность
перекрытия, а не универсальную величину ускорения. Оба снимка можно открыть
в исходном размере.*

## Dynamic padding

Чтобы сформировать прямоугольную порцию из последовательностей разной длины,
их дополняют до максимальной длины **в этой порции**, а не во всём корпусе.
Для длин $l_1,\ldots,l_B$ доля настоящих токенов равна

$$
\eta_{\text{pad}}=\frac{\sum_{i=1}^{B}l_i}{B L_{\max}},
\qquad L_{\max}=\max_i l_i.
$$

Для `[128, 140, 160, 1024]` получается $1452/4096\approx35{,}4\%$.
Это доля заполненных позиций, **не измеренная эффективность GPU**. Линейные
проекции и FFN примерно пропорциональны числу позиций, а обычное плотное
внимание зависит от квадрата длины. Маска может запрещать внимание к PAD,
но не обязана устранять вычисление всех запрещённых элементов. Специализированное
ядро для переменных длин может выполнять совсем другой объём работы.

Для этого же набора длин можно отдельно посчитать пары позиций внимания.
В четырёх полных квадратах размера $1024\times1024$ их 4 194 304.
Сумма квадратов настоящих длин равна

$$
\begin{aligned}
&128^2+140^2+160^2+1024^2\\
&=1\,110\,160,
\end{aligned}
$$

то есть примерно 26,5% от первого числа. Для причинного внимания
число разрешённых пар внутри отдельного документа длины $l$ равно
$l(l+1)/2$; для четырёх документов получается 555 806 против
2 099 200 пар в дополненных треугольниках. Это логический счёт пар,
не фактически выполненные FLOP: ядро может вычислять полные плитки
и лишь затем маскировать часть результатов. Число входных токенов
и число пар внимания поэтому описывают разные составляющие нагрузки.

### Маска определяется длиной, а не значением токена

В следующем примере `pad_id` используется только для заполнения памяти.
Настоящие позиции определяются исходными длинами. Поэтому код остаётся
корректным, даже если токенизатор использует один идентификатор для PAD и EOS.

```python
import torch
from torch.nn.utils.rnn import pad_sequence

def collate_lm(rows, pad_id=0):
    # Каждый input_ids уже содержит нужные BOS/EOS.
    seqs = [torch.as_tensor(row["input_ids"], dtype=torch.long) for row in rows]
    if not seqs or any(x.ndim != 1 or x.numel() < 2 for x in seqs):
        raise ValueError("Expected nonempty batch of 1-D sequences of length >= 2")
    lengths = torch.tensor([x.numel() for x in seqs])
    ids = pad_sequence(seqs, batch_first=True, padding_value=pad_id)
    valid = torch.arange(ids.shape[1])[None, :] < lengths[:, None]
    labels = ids.clone()
    labels[~valid] = -100
    return {"input_ids": ids, "attention_mask": valid, "labels": labels}

batch = collate_lm([
    {"input_ids": [11, 12, 2]},
    {"input_ids": [21, 2]},
], pad_id=2)
assert batch["attention_mask"].tolist() == [[True, True, True], [True, True, False]]
assert batch["labels"].tolist() == [[11, 12, 2], [21, 2, -100]]
```

Значение EOS `2` в короткой строке остаётся обучающей целью. Только последняя
добавленная позиция исключена из потерь. Проверка `ids != pad_id` скрыла бы
также настоящий EOS и изменила задачу обучения.

Здесь принят интерфейс causal-LM, который **внутри** сопоставляет
`logits[:, :-1]` с `labels[:, 1:]`. Если функция потерь этого не делает,
сдвиг нужно выполнить самостоятельно, но только один раз. Маска внимания
определяет доступный контекст, а `labels == -100` — позиции, исключённые
из cross-entropy. Эти две маски не взаимозаменяемы. При SFT к маске потерь
добавляют правила для префикса и реплик пользователя; их нельзя восстановить
из одной лишь длины текста.

## Bucketing

Динамическое дополнение мало помогает, если короткие тексты постоянно
попадают в одну порцию с длинными. *Bucketing* группирует примеры близких длин.
Рассмотрим восемь текстов и порции по два примера:

| Способ группировки | Пары длин | Всего позиций после padding |
|---|---|---:|
| Исходный порядок | (8, 64), (9, 60), (10, 56), (12, 52) | 464 |
| Группировка по длине | (8, 9), (10, 12), (52, 56), (60, 64) | 282 |

Настоящих токенов в обоих случаях 271. Доля заполненных позиций возрастает
с $271/464\approx58{,}4\%$ до $271/282\approx96{,}1\%$. Это тот же набор
данных и то же число порций. Отношение 464/282 описывает изменение числа
позиций, но ещё не является измеренным ускорением модели.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/edls/optimal-sequence-processing.png]]

*Max Ryabinin, [Efficient DL Systems, слайд 31](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week02_fast_pipelines/lecture.pdf#page=31).
Слева показаны последовательности с заполнителями; справа — две порции,
дополненные до разных длин. Слайд упоминает packing, но правая схема сама по
себе не показывает склеивание документов. Полный исходный слайд сохранён;
лицензия закреплённой версии репозитория — MIT.*

Постоянная сортировка всего корпуса делает порядок обучения зависимым от
длины текста, а часто и от его жанра или языка. Поэтому обычно сначала
перемешивают индексы, сортируют небольшие окна по длине, формируют порции,
а затем перемешивают порядок порций. Размер окна задаёт компромисс между
случайностью порядка и количеством заполнителей.

Число примеров тоже может меняться. Но ограничение
$\sum_i l_i\le T_{\text{budget}}$ ограничивает только число **настоящих**
токенов. Для прямоугольной порции нужны также ограничения на $B L_{\max}$,
а для плотного внимания — оценка члена $B L_{\max}^2$. Иначе один длинный
текст может вызвать нехватку памяти, хотя сумма длин укладывается в бюджет.

## Packing

Вместо дополнения каждой последовательности можно хранить настоящие токены
подряд, сохранив отдельно границы документов. Для длин `[7, 10, 8, 5]`
прямоугольный тензор содержит 40 позиций, а упакованный — 30. Массив
накопленных длин `[0, 7, 17, 25, 30]` сообщает, где начинается и заканчивается
каждая последовательность. В реализациях attention такой массив часто
называется `cu_seqlens`.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/data-pipeline-review-2026-10/padding-packing.png]]

*Angel Li, [PyTorch: Using Variable Length Attention](https://docs.pytorch.org/tutorials/intermediate/variable_length_attention_tutorial.html),
BSD-3-Clause. Слева четыре строки с padding, справа те же токены без
заполнителей и с границами 7, 17, 25, 30. Упаковка меняет хранение;
независимость документов обеспечивает реализация внимания, использующая
границы. Рисунок сохранён без изменений.*

Такое переменное представление следует отличать от заполнения контейнеров
фиксированной длины несколькими документами. Во втором случае остаётся
задача выбора документов, которые поместятся вместе: например, в строку
длиной 1024 можно уложить несколько коротких текстов, но часть места на краю
всё ещё может пропасть. Разрезание длинного документа, перенос его остатка
в следующую строку и добавление EOS — отдельные решения.

Само склеивание не сохраняет смысл обучения автоматически. Обычная causal
mask запрещает смотреть в будущее, но начало документа B всё ещё видит весь
документ A. Для непрерывного потока предобучения это может быть сознательным
выбором. Для независимых примеров, например разных диалогов SFT, нужен другой
режим.

<a id="packing-без-leakage"></a>

## Как сохранить независимость документов при упаковке

Пусть упакованная строка содержит два документа: `[11, 12, EOS]` и
`[21, 22, 23, EOS]`. Обозначим через $s_i$ номер документа, содержащего
позицию $i$. Независимое причинное внимание разрешает связь запроса $i$
с ключом $j$ только при двух условиях:

$$
\operatorname{allow}(i,j)=[s_i=s_j]\land[j\le i].
$$

Первая проверка запрещает переход между документами, вторая — чтение будущего.
Получается блочно-диагональная причинная маска. EOS отмечает границу как токен,
но сам по себе не накладывает такой запрет на attention.

Для документов длиной 3 и 4 маска имеет вид

$$
\left[\begin{array}{ccc|cccc}
1&0&0&0&0&0&0\\
1&1&0&0&0&0&0\\
1&1&1&0&0&0&0\\ \hline
0&0&0&1&0&0&0\\
0&0&0&1&1&0&0\\
0&0&0&1&1&1&0\\
0&0&0&1&1&1&1
\end{array}\right].
$$

Строка соответствует запросу, столбец — ключу. Единица разрешает связь.
Два треугольника сохраняют причинность внутри документов; нулевые
прямоугольники между ними запрещают переход через границу.

Для той же строки удобно явно выписать метаданные:

| Поле | Значения |
|---|---|
| `input_ids` | 11, 12, EOS, 21, 22, 23, EOS |
| `segment_ids` | 0, 0, 0, 1, 1, 1, 1 |
| `position_ids` | 0, 1, 2, 0, 1, 2, 3 |
| `labels` до внутреннего сдвига | −100, 12, EOS, −100, 22, 23, EOS |

Игнорирование метки на первой позиции B исключает задачу «по EOS документа A
предсказать начало B». При этом последнее слово A по-прежнему должно
предсказывать EOS. Сброс позиций воспроизводит отдельную обработку документов,
если модель использует ту же позиционную схему. Одного сброса недостаточно
для изоляции: без ограничения внимания B всё ещё может читать значения из A.

Для проверки небольшой маски достаточно такого кода:

```python
segment_ids = torch.tensor([0, 0, 0, 1, 1, 1, 1])
n = segment_ids.numel()
same_document = segment_ids[:, None] == segment_ids[None, :]
causal = torch.ones(n, n, dtype=torch.bool).tril()
allow = same_document & causal
assert not allow[3:, :3].any()
assert allow[5].nonzero().flatten().tolist() == [3, 4, 5]
```

Это проверка логики, а не рекомендуемый способ хранить огромную маску:
плотная матрица занимает $O(T^2)$ памяти. Ядра для переменных длин могут
использовать границы последовательностей и не вычислять междокументные
блоки. Простая передача плотной маски обычному ядру не гарантирует такого
сокращения работы. Кроме того, соглашения API различаются: в одном интерфейсе
`True` означает разрешённую связь, в другом — запрещённую. Маску нужно
преобразовать согласно документации выбранного оператора.

Сравнивают выходы модели при отдельной обработке документов и при их
упаковке с отключённым dropout и одинаковой обработкой позиций. Затем
изменяют токены только в A: результаты для B должны остаться прежними в
пределах численного допуска. Если меняются, граница документа где-то потеряна.
Такой тест проверяет весь путь, а не только правильность нарисованной маски.

### Проверка на одном операторе внимания

Пусть все запросы и ключи равны
нулю, а значения $V$ документа A равны `[1, 2, 3]`, документа B —
`[10, 20, 30, 40]`. Все разрешённые оценки внимания одинаковы,
поэтому softmax даёт равные веса. Независимое причинное внимание
возвращает для A `[1, 1.5, 2]`, для B — `[10, 15, 20, 25]`:
например, третья позиция B усредняет только 10, 20 и 30.

Если оставить одну общую causal mask, первая позиция B получит
$(1+2+3+10)/4=4$ вместо 10. Запрета смотреть в будущее здесь
недостаточно: все три позиции A уже находятся в прошлом.

Следующий CPU-пример использует `allow` из предыдущего блока.
У `scaled_dot_product_attention` булево значение `True` разрешает связь;
dropout явно выключен. Соглашение о маске приведено в
[документации SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html).

```python
import torch.nn.functional as F

q = torch.zeros(1, 1, 7, 1, dtype=torch.float64)
k = q.clone()
v = torch.tensor([1, 2, 3, 10, 20, 30, 40],
                 dtype=torch.float64).reshape(1, 1, 7, 1)

def attend(q, k, v, mask=None):
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask,
        is_causal=(mask is None), dropout_p=0.0,
    )

packed = attend(q, k, v, allow)
separate = torch.cat([
    attend(q[:, :, :3], k[:, :, :3], v[:, :, :3]),
    attend(q[:, :, 3:], k[:, :, 3:], v[:, :, 3:]),
], dim=2)
torch.testing.assert_close(packed, separate)

changed_v = v.clone()
changed_v[:, :, :3] += 100
isolated = attend(q, k, changed_v, allow)
torch.testing.assert_close(packed[:, :, 3:], isolated[:, :, 3:])

causal_only = attend(q, k, v)
assert causal_only[0, 0, 3, 0].item() == 4.0
print(packed.flatten().tolist())
```

После прибавления 100 к значениям A правильная маска оставляет все
выходы B прежними. С общей causal mask первый выход B стал бы равен
$(101+102+103+10)/4=79$. Сравнение с отдельными вызовами проверяет
эквивалентность упаковки, а изменение A — отсутствие связи с B.
Этот опыт проверяет только оператор внимания. Проверка всей модели
дополнительно должна охватывать эмбеддинги, позиции, маски потерь
и обратный проход; плотная CPU-маска не заменяет быстрое varlen-ядро.

## Как нормировать функцию потерь

После изменения группировки в разных порциях может оказаться разное число
обучающих целей. Среднее от средних тогда не равно среднему по всем токенам.
Если первая порция содержит 100 учитываемых целей со средней потерей 0,5,
а вторая — 10 со средней потерей 2, простое среднее даёт 1,25. Среднее по
токенам другое:

$$
L=\frac{100\cdot0{,}5+10\cdot2}{110}\approx0{,}636.
$$

При накоплении градиентов нормировку задают по всему окну накопления:

$$
L=\frac{\sum_b\sum_{t\in\mathcal V_b}\ell_{b,t}}
{\sum_b |\mathcal V_b|},
$$

где $\mathcal V_b$ — позиции с учитываемыми целями **после сдвига меток**.
Это не обязательно все непустые входные токены: начало документа, префикс
инструкции и реплики пользователя могут не участвовать в потерях. Если число
целей известно заранее, каждый микрошаг может делать backward от своей суммы
потерь, делённой на общий знаменатель. При распределённом обучении нужно
согласовать этот знаменатель с усреднением градиентов между процессами;
локальное среднее каждого процесса не равно глобальному при разных длинах.

### Нормировка при усреднении градиентов DDP

Пусть работают $R$ процессов, $S_r$ — сумма потерь по всем учитываемым
целям процесса $r$ в окне накопления, а $N$ — общее число этих целей
на всех процессах; предполагается $N>0$. При стандартном усреднении
градиентов DDP локальная
функция потерь должна быть

$$
L_r=\frac{R S_r}{N}.
$$

Тогда после синхронизации получится

$$
\frac1R\sum_{r=0}^{R-1}\nabla L_r
=\frac{\sum_r\nabla S_r}{N},
$$

то есть градиент среднего по всем целям. Множитель $R$ компенсирует
деление в DDP. Для двух процессов с суммами 50 и 20 и общим числом
целей 110 локальные множители дают потери $100/110$ и $40/110$;
их среднее равно $70/110$, как в исходном токенном расчёте.
Если поделить каждую сумму только на 110, DDP дополнительно уменьшит
градиент вдвое.

$N$ считают по реально участвующим целям после сдвига и маскирования;
согласование одного этого счётчика не требует передавать сами тексты.
Для окна накопления знаменатель общий для всех микрошагов.
Формула предполагает обычное усреднение DDP. Если пользовательский
обработчик коммуникаций меняет правило редукции, множитель нужно пересчитать. См.
[примечание PyTorch о сумме потерь и DDP](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html).

Токенная нормировка здесь — выбранная задача. Равный вес документов тоже
возможен, но тогда это другой критерий, и его следует сохранить при переходе
от отдельных примеров к packing. В [[02 Areas/ML & DL/00 Учебник/10 ML Systems/05 Численные форматы и mixed precision|главе о mixed precision]]
эта же нормировка сочетается с накоплением градиентов и loss scaling.

<a id="базовый-путь-загрузки-без-доказанного-перекрытия"></a>

## Базовый цикл загрузки

Следующий фрагмент связывает `collate_lm` с загрузчиком. `dataset` и
`token_budget_sampler` обозначают уже подготовленный набор данных и
расписание порций; модель использует описанный выше внутренний сдвиг меток.
Это фрагмент подключения данных, а не полный цикл оптимизации.

```python
from functools import partial
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_sampler=token_budget_sampler,
    collate_fn=partial(collate_lm, pad_id=pad_id),
    num_workers=4,
    prefetch_factor=2,
    persistent_workers=True,
    pin_memory=True,
)
for cpu_batch in loader:
    batch = {k: v.to("cuda", non_blocking=True) for k, v in cpu_batch.items()}
    loss = model(**batch).loss
    # backward и обновление параметров выполняются в обучающем цикле.
```

Четыре работника — начальная настройка для измерения, не универсальный
рецепт. При `num_workers=0` параметры `prefetch_factor` и
`persistent_workers` убирают. На системах с запуском работников через
`spawn` точка входа должна быть защищена `if __name__ == "__main__":`,
а функции и классы — доступны для сериализации.

Копирование и модель в этом фрагменте используют текущий поток CUDA,
поэтому перекрытие H2D с вычислениями не заявляется. Двойная буферизация и
зависимости потоков разобраны в
[[02 Areas/ML & DL/00 Учебник/10 ML Systems/02 GPU, CUDA и иерархия памяти|главе о GPU и CUDA]].

## Разделение данных и возобновление обучения

Для индексируемого набора данных `DistributedSampler` распределяет индексы
между процессами обучения. `set_epoch(epoch)` меняет порядок перемешивания
между эпохами. При `drop_last=False` sampler может дополнить список
повторными индексами, чтобы разделить его поровну; при `drop_last=True`
отбрасывается хвост. У самого `DataLoader` также есть `drop_last`, но он
относится к неполной порции. Эти два решения действуют на разных этапах.

Например, пять индексов `[0, 1, 2, 3, 4]` нужно распределить между двумя
процессами без перемешивания. Sampler с `drop_last=False` дополняет список
индексом 0: процесс 0 получает `[0, 2, 4]`, процесс 1 — `[1, 3, 0]`.
Если затем `DataLoader(batch_size=2, drop_last=True)` отбросит неполные
порции, в обучение попадут `[0, 2]` и `[1, 3]`: исчезнут и индекс 4,
и добавленная копия 0. При `drop_last=True` уже у sampler результат
деления был бы тем же без дополнения. Для потоковых наборов последнее
неполное множество может отбрасываться отдельно у каждого работника;
оценка `len(loader)` поэтому не доказывает, сколько примеров прочитано.

У потокового `IterableDataset` каждый работник получает собственную копию
итератора. Если все копии читают один и тот же поток, данные дублируются.
Разделение нужно задать явно — по шардам, учитывая и номер процесса обучения,
и номер работника. Пусть $r$ — номер процесса, $w$ — номер работника в нём.
При $R$ процессах и $W$ работниках на процесс глобальный номер читателя
равен $rW+w$, а всего читателей $RW$.
Такой способ требует одинакового $W$; изменение топологии меняет разбиение.

При двух процессах и двух работниках на процесс десять шардов можно
разделить по остатку номера шарда от деления на четыре:

| Процесс, работник | Глобальный читатель | Номера шардов |
|---|---:|---|
| 0, 0 | 0 | 0, 4, 8 |
| 0, 1 | 1 | 1, 5, 9 |
| 1, 0 | 2 | 2, 6 |
| 1, 1 | 3 | 3, 7 |

Каждый шард назначен ровно одному читателю. Если учитывать только
номер работника, оба процесса повторят одно и то же разбиение.
Равное число шардов при этом не означает равного числа токенов:
размеры шардов и длины документов могут различаться.

В потоковом чтении перемешивание часто ограничено буфером: из него выбирается
случайная запись и заменяется следующей из потока. Это не равномерная
перестановка всего корпуса. Размер буфера и порядок шардов влияют на то,
какие примеры будут соседствовать, что важно при смеси языков или источников.
Практический интерфейс описан в
[Hugging Face Datasets: Stream](https://huggingface.co/docs/datasets/stream).

Для воспроизводимого продолжения недостаточно сохранить веса и номер шага.
Нужны версия корпуса, порядок шардов, положение чтения, состояния генераторов
случайных чисел и, при точном восстановлении, ещё не использованная часть
буфера перемешивания и упаковки. Иначе после перезапуска часть примеров будет
повторена или пропущена. Отдельно проверяют, не пересекаются ли train и
validation: их нельзя получать независимыми `take` и `skip` от по-разному
перемешанных потоков.

У готового загрузчика нужно проверить, какую часть состояния он сохраняет.
Например, `IterableDataset.state_dict()` в Hugging Face Datasets хранит
текущий шард и положение внутри него. Однако при возобновлении после
`.shuffle()` содержимое shuffle-буфера теряется и буфер заполняется заново.
Это ограничение прямо указано в
[разделе о восстановлении итератора](https://huggingface.co/docs/datasets/stream#save-a-dataset-checkpoint-and-resume-iteration).

Пусть буфер вмещает две записи и сначала содержит `[a, b]`.
После выдачи `a` его место занимает `c`, после выдачи `c` — запись `d`.
К моменту checkpoint прочитаны `[a, b, c, d]`, выданы только `a` и `c`,
а в буфере остались `[d, b]`. Позиции чтения «после d» недостаточно:
продолжение с `e` пропустит обе ожидающие записи. Повторное
чтение с начала может вернуть уже использованные записи. Для точного
продолжения нужен либо сохранённый буфер вместе с состоянием генератора
случайных чисел, либо возможность воспроизвести его тем же чтением и теми же
случайными выборами.
Если система гарантирует только приблизительное продолжение,
это следует отличать от точного воспроизведения порядка примеров.

<a id="проверка-throughput"></a>

## Как измерить скорость обработки данных

Сначала сравнивают три нагрузки с одинаковыми формами: данные уже на GPU,
готовые тензоры в RAM и полный путь от хранилища. Первая задаёт ориентир для
самой модели, вторая добавляет передачу, третья — чтение и преобразования.
После этого по одному меняют число работников, предварительную загрузку,
формат хранения и способ группировки. Иначе невозможно понять, что именно
устранило задержку.

Измеряют не только порции в секунду, но и настоящие входные токены, учитываемые
целевые токены и шаги оптимизатора за единицу времени. Если новая группировка
изменила число целей в шаге, рост `steps/s` ещё ничего не говорит о скорости
обучения на том же объёме данных. Вместе с результатом сохраняют распределение
длин, долю padding, потребление RAM/VRAM и настройки загрузчика.

Длительность `next(iterator)` показывает, сколько обучающий процесс ждал
готовую порцию, а не сколько времени все работники суммарно читали данные.
Аналогично сумма CPU-, CUDA- и H2D-интервалов может считать перекрывающуюся
работу дважды. Следующая глава показывает, как прочитать временную трассу,
связать простой с конкретной стадией и проверить гипотезу повторным измерением.

## Практика и первоисточники

- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week02_fast_pipelines/lecture.pdf|Efficient DL Systems, Week 2]] и [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week02_fast_pipelines/seminar/practice.ipynb|полный семинар]] — хранение, динамическое дополнение, потоковое чтение, декодирование изображений и DALI.
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week02_fast_pipelines/homework/README|EDLS: практическое задание]] — сравнение группировки и упаковки последовательностей.
- [[05 Источники/Courses/Harvard ML Systems/tinytorch/05_dataloader|TinyTorch 05 — DataLoader]] — самостоятельная реализация набора данных, итерации, перемешивания и порций.
- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol1/data_engineering|Harvard CS249r — Data Engineering]] — более широкий разбор хранения, качества данных и воспроизводимости.
- [PyTorch: DataLoader](https://docs.pytorch.org/docs/2.14/data.html), [передача через pinned memory](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html) и [variable-length attention](https://docs.pytorch.org/tutorials/intermediate/variable_length_attention_tutorial.html) — интерфейсы и исходные эксперименты. Доступность конкретного ядра attention зависит от установленной версии и GPU.

← [[02 Areas/ML & DL/00 Учебник/10 ML Systems/05 Численные форматы и mixed precision|Численные форматы и mixed precision]] ·
[[02 Areas/ML & DL/00 Учебник/10 ML Systems/07 Profiling ML-нагрузки|Profiling ML-нагрузки]] →
