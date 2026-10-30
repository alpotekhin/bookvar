---
title: Измерение производительности и roofline
type: textbook-chapter
status: canonical
last_updated: 2026-10-26
last_verified: 2026-10-26
source_language: mixed
source_unit_id:
  - lecture-05-roofline-performance
  - lecture-05-matrix-mystery
  - lecture-05-performance-recap-transition
  - lecture-06-benchmark-profile-experiment
  - lecture-06-section-169-add-dim-2048
  - lecture-06-section-199-matmul-dim-2048
  - lecture-06-section-202-matmul-dim-128
  - assignment-02-task-benchmarking-script
  - assignment-02-benchmark-synchronization
---

<a id="cs336-systems-roofline"></a>

# Измерение производительности и roofline

Два варианта матричного умножения дают одинаковый результат. Один вызов
Python возвращается быстрее другого. Достаточно ли этого, чтобы выбрать
первый вариант? Если вычисление отправлено на GPU, ответ пока неизвестен:
CPU мог только поставить работу в очередь. Нужно дождаться результата
и определить, что именно мы сравниваем — запуск операции, её выполнение
на устройстве или весь путь от входных данных до готового ответа.

Даже после правильного измерения остаётся второй вопрос: почему получено
именно такое время? Матрицы могут не помещаться в кэш, ядро — запускать
слишком мало блоков, а CPU — не успевать отправлять очередную работу.
Одного процента «загрузки GPU» недостаточно, чтобы различить эти причины.

## Что входит в измерение

Пусть приложение получает текст, токенизирует его, передаёт тензоры на GPU,
выполняет модель и возвращает ответ. Время одного матричного умножения
и задержка приложения относятся к разным участкам этой последовательности.
Оба числа полезны, но отвечают на разные вопросы.

![[02 Areas/ML & DL/Assets/Sources/Harvard ML Systems/vol1/benchmarking/b2b9aaeaa637a9a665f0f4e9229daeb5bc5a4f2d.svg]]

*Harvard ML Systems, [Benchmarking Granularity](https://mlsysbook.ai/vol1/benchmarking/benchmarking.html#fig-granularity),
Harvard Edge ML Systems Book contributors,
[CC BY-NC-SA 4.0](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/LICENSE.md).
Слева направо граница расширяется от слоёв модели до приложения.
В правой части явно присутствуют вычисления вне нейросети — Non-AI Compute.
Оригинальный SVG сохранён целиком, без изменений.*

| Граница измерения | Что включено | Для чего нужно число |
|---|---|---|
| Ядро на GPU | Выполнение одного ядра на устройстве | Сравнить способы реализации операции |
| Оператор или модель | Все необходимые ядра и зависимости между ними | Проверить выигрыш в реальном вычислительном графе |
| Запрос | Подготовка, очередь, передача, модель, формирование ответа | Узнать задержку для пользователя |
| Обучение до заданного качества | Шаги обучения, загрузка данных, оценивание и прочие оговорённые расходы | Сравнить стоимость получения нужной модели |

Если измерение начинается после токенизации, её ускорение никак не изменит
результат опыта. Если вход уже находится в памяти GPU, число ничего не
скажет о цене передачи с CPU. Это не ошибка само по себе: границу выбирают
под вопрос и указывают рядом с результатом.

Ускорение отдельного участка также нельзя переносить на весь путь.
Если матричное умножение занимало 20% последовательного времени, его
ускорение втрое даст

$$
S=\frac{1}{0.8+0.2/3}\approx1.15.
$$

Остальные 80% работы сохранились. При перекрытии операций вместо простой
суммы нужно рассматривать зависимости и критический путь — самую длинную
по времени цепочку зависимых операций, определяющую готовность результата.

## До таймера: одинаковая задача и правильный результат

Сравнение начинается с определения операции. Для $C=AB$ нужно задать формы
матриц, типы входов и выхода, размещение в памяти, допустимую ошибку и то,
требуются ли градиенты. Две матрицы одинаковой формы могут по-разному
размещаться в памяти: например, одна хранится построчно, а другая получена
транспонированием. Измерять их как одну и ту же нагрузку нельзя.

Результат новой реализации сравнивают с понятным эталоном до измерения
скорости. Для обучаемой операции проверяют и обратный проход. Полезно
включить размер, при котором последний блок заполнен не целиком:
ошибка маскирования часто не видна на удобной степени двойки.
Численный допуск выбирают исходя из точности и порядка вычислений,
а не подгоняют под ошибку уже написанного ядра.

Смена FP32 на BF16 может быть полезной оптимизацией, но меняет условия
сравнения. Для целой модели недостаточно совпадения нескольких выходных
тензоров: нужно проверить качество на согласованном наборе данных.
При обучении более быстрый шаг ещё не означает меньшее время до нужного
качества — могли измениться размер пакета и число необходимых шагов.

Вместе с результатом сохраняют версии программы и библиотек, модель GPU,
режим точности, формы и strides тензоров. Для переменных длин запросов
нужна сама выборка или правило её получения. Иначе другой исследователь
воспроизведёт похожий код, но не ту нагрузку, на которой получено ускорение.

## Два способа измерить CUDA-вычисление

Сначала отделим зависимости данных от времени отправки операций.
Рассмотрим $z=x\odot y+2$, где x и y — два тензора из единиц,
а $\odot$ обозначает поэлементное умножение.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/systems-visuals-2026-09/d2l-asyncgraph.svg]]

*Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola,
[Dive into Deep Learning — Asynchronous Computation, fig. 13.2.2](https://d2l.ai/chapter_computational-performance/async-computation.html#fig-asyncgraph),
[CC BY-SA 4.0](https://github.com/d2l-ai/d2l-en/blob/23d7a5aecceee57d1292c56e90cce307f183bb0a/LICENSE).
Оригинальный SVG. Узел «_» — промежуточный результат умножения.*

Создание x и y не зависит друг от друга, но умножение требует оба входа,
а прибавление двойки — результат умножения. Поставить все операции в очередь
можно раньше, чем завершится последняя из них. Однако получить численное
значение z до окончания этих зависимостей нельзя.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/systems-visuals-2026-09/d2l-threading.svg]]

*Те же авторы, [D2L, fig. 13.2.3 — Interactions of the frontend and backend](https://d2l.ai/chapter_computational-performance/async-computation.html#fig-threading),
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), оригинальный SVG без изменений.*

На верхней полосе отправка и выполнение чередуются. Ниже они разделены:
Frontend быстро выдаёт задания, а Backend продолжает их выполнять.
Перед печатью z появляется ожидание wait. Это схема асинхронного исполнения,
а не трасса конкретной реализации PyTorch: её смысл — показать, почему
возврат управления и готовность результата происходят в разные моменты.
Таймер, остановленный после отправки последнего задания, пропустит это ожидание.

### Время от вызова до готового результата

Обычный таймер CPU измеряет прошедшее реальное время. Для асинхронной
операции нужны две границы: перед стартом закончить прежнюю работу,
перед остановкой дождаться новой. Минимальный фрагмент выглядит так:

```python
import time
import torch

# run() и входные данные уже подготовлены; устройство CUDA выбрано.
torch.cuda.synchronize()
t0 = time.perf_counter()
y = run()
torch.cuda.synchronize()
wall_ms = 1000 * (time.perf_counter() - t0)
```

Без первого ожидания в интервал может попасть работа предыдущей итерации.
Без второго таймер может остановиться до завершения вычислений. С обоими
ожиданиями интервал включает вызов Python, отправку работы, её выполнение
и расходы на ожидание завершения. Это время готовности результата
в выбранных условиях, а не чистая длительность арифметических инструкций.

Синхронизация здесь ждёт работу всех CUDA streams текущего устройства,
в том числе постороннюю для нашего опыта работу этого процесса.
Поэтому измерение проводят без такой нагрузки. Для нескольких GPU
нужно охватить завершение работы каждого участвующего устройства;
одного ожидания на случайно выбранном GPU недостаточно.

### CUDA events: интервал на временной шкале устройства

CUDA event — отметка в очереди выполнения. Поместим первую отметку перед
операцией, вторую после неё и дождёмся второй. Разность отметок даст
время между ними на GPU. В [CS336 Lecture 6](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_06.py)
этим способом сравниваются поэлементное сложение и матричное умножение.

Ниже — небольшой измеритель для функции `run()`, которая отправляет всю
свою работу в **текущий CUDA stream текущего устройства**. Для нескольких
streams нужны зависимости в обе стороны: работа начинается после
начальной отметки и завершается до конечной. Пример этих зависимостей
не создаёт и предназначен только для одного stream.

```python
import torch

def cuda_samples(run, warmup=20, repeats=50, inner=20):
    if warmup < 0 or repeats < 1 or inner < 1:
        raise ValueError("Invalid repetition count")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Создать обе отметки на устройстве вне измеряемой серии.
    start.record()
    end.record()
    end.synchronize()

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    samples_ms = []
    for _ in range(repeats):
        start.record()
        for _ in range(inner):
            run()
        end.record()
        end.synchronize()
        samples_ms.append(start.elapsed_time(end) / inner)
    return samples_ms
```

Входы подготавливаются до вызова измерителя. Например, для сложения
с константой можно использовать один вход и заранее выделенный выход:

```python
import statistics

if not torch.cuda.is_available():
    raise RuntimeError("This example requires an NVIDIA CUDA device")

x = torch.randn(1 << 20, device="cuda", dtype=torch.float32)
out = torch.empty_like(x)

def run():
    return torch.add(x, 1.0, out=out)

torch.testing.assert_close(run(), x + 1.0)
samples = cuda_samples(run)
print("Median group-average time, ms:", statistics.median(samples))
```

Каждый элемент `samples` — **среднее время двадцати вызовов внутри одной
группы**. Это снижает относительную цену двух событий, но сглаживает
разброс отдельных вызовов. Поэтому p99 этого списка нельзя выдавать
за p99 задержки запроса. `inner=1` меняет единицу наблюдения, но не
превращает изолированный опыт с ядром в измерение сервиса.

Интервал событий тоже не обязательно равен сумме длительностей ядер.
Если CPU медленно отправляет короткие операции, GPU может простаивать
между ними уже после начальной отметки. Эти промежутки попадут в интервал.
Работа другого stream может конкурировать за ресурсы. События задают
границу наблюдения, а причины времени затем выясняет профилировщик.

Для повторяемых сравнений полезен готовый
[`torch.utils.benchmark.Timer`](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html).
Он поддерживает прогрев, синхронизацию ускорителя и группировку повторений
через `blocked_autorange`. Его параметры, включая число CPU-потоков,
тоже входят в условия опыта. Пример выше нужен, чтобы понимать, что
измеряет такой инструмент; результаты запуска на GPU здесь не приводятся.

## Прогрев и кэш — разные условия опыта

Первое выполнение может включать инициализацию CUDA, загрузку библиотек,
компиляцию и подбор алгоритма. После смены формы тензора часть этих расходов
может появиться снова. Отдельно меняются частоты устройства и его температура.
Несколько прогревочных вызовов помогают отделить запуск от устойчивой работы,
но число 20 в примере — параметр, а не гарантия достижения устойчивого режима.
Это проверяют по последовательности измерений.

Повторное использование одного массива, как в коде выше, задаёт ещё одно
условие: его данные могут остаться в кэше. Если цель — узнать скорость
повторных обращений к этому массиву, такое измерение уместно. Если цель —
оценить пропускную способность HBM на новых данных, оно может дать неверную
интерпретацию. Нужно контролировать рабочий набор и проверять фактический
обмен с памятью, а не просто делить размер массива на время.

Создание случайного массива перед таймером не гарантирует холодный кэш:
сама генерация уже затронула память GPU. `torch.cuda.empty_cache()` управляет
свободными блоками кэширующего аллокатора PyTorch, а не очищает аппаратный L2.
Условия холодного и повторного доступа описывают явно и одинаково применяют
к сравниваемым реализациям.
[Документация PyTorch: empty_cache](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.empty_cache.html).

Холодный запуск при этом не является «плохим измерением». Для приложения,
которое редко просыпается ради одного запроса, он может быть главным
показателем. Его просто измеряют отдельным сценарием, не смешивая
инициализацию с многократным выполнением уже подготовленной модели.

## Разброс времени и неопределённость оценки

Пусть в условном опыте 980 запросов завершились за 20 мс, а 20 запросов —
за 500 мс. Среднее время равно

$$
\bar t=\frac{980\cdot20+20\cdot500}{1000}=29.6\text{ мс}.
$$

Медиана равна 20 мс, а p99 — 500 мс: редкие медленные запросы почти
не изменили типичное время, но сильно ухудшили хвост. Для наглядности
здесь используется квантиль по рангу: p99 — элемент с номером
$\lceil0.99N\rceil$ в отсортированной выборке. Библиотеки могут использовать
интерполяцию, поэтому при малой выборке важно указать и способ расчёта.

Если требование — не менее 99% ответов за 50 мс, этот опыт его не выполняет:
порог превышен у 2% запросов. Доля превышений позволяет непосредственно
проверить требование. Ошибки и тайм-ауты нельзя незаметно вычеркнуть
из знаменателя и затем сообщить задержку только успешных ответов.

**Квантиль и доверительный интервал отвечают на разные вопросы.** Квантиль
описывает разброс задержек самих запросов. Доверительный интервал показывает
неопределённость оценки выбранного показателя по конечной выборке.
Можно достаточно точно знать среднее и одновременно иметь тяжёлый хвост.
А можно получить нестабильную оценку p99 просто потому, что запросов мало.
При 200 наблюдениях верхний процент — примерно два наблюдения;
такая выборка даёт мало сведений о редких задержках.

Процентили нескольких запусков нельзя усреднить, чтобы получить общий p99.
Для общей оценки нужны исходные наблюдения или совместимые гистограммы;
при этом смешивание разных нагрузок само по себе может скрыть различия.
Показатели отдельных сценариев поэтому сохраняют рядом с агрегатом.

Повторы внутри одного прогретого процесса не заменяют независимые запуски.
Они разделяют состояние кэшей, температуру, размещение памяти и соседнюю
нагрузку. Сначала сохраняют отдельные времена и порядок выполнения,
затем сравнивают результаты нескольких запусков. Bootstrap по отдельным
наблюдениям предполагает подходящую независимость; для коррелированных
серий нужно учитывать группы или временные блоки, а не искусственно
увеличивать число независимых образцов.

Для A/B-сравнения порядок вариантов чередуют или случайно меняют, чтобы
один из них не измерялся всегда на более горячем устройстве. При этом
сохраняют одинаковую политику кэша. Отчёт содержит и разброс, и величину
эффекта. Переход 10 → 7 мс означает уменьшение времени на 30%, но
ускорение в $10/7\approx1.43$ раза; эти две формулировки не взаимозаменяемы.

## Почему сервис нужно проверять под нагрузкой

Одиночный запрос на свободном GPU почти ничего не говорит об очереди.
Если один клиент отправляет следующий запрос только после ответа,
медленная система сама уменьшает частоту поступления новых запросов.
Это замкнутая нагрузка (*closed loop*). Она полезна для некоторых сценариев,
но может скрыть поведение при независимом потоке поступлений.

В открытой нагрузке (*open loop*) моменты поступления задаются независимо
от завершения предыдущих запросов. Например, если в упрощённой системе
приходит 100 одинаковых запросов в секунду, а завершается только 80,
очередь без ограничений растёт в среднем на 20 запросов в секунду.
Скорость выполнения одного запроса не изменилась, но время ожидания
будет увеличиваться. Ограничение очереди заменит часть ожиданий отказами,
а не добавит вычислительной мощности.

В отчёте поэтому разделяют предложенную нагрузку, завершённую работу,
задержки и ошибки. Для LLM дополнительно фиксируют входные и выходные
длины, долю повторных префиксов и правило остановки генерации.
Одинаковое число запросов не означает одинаковое число токенов работы.
TTFT, межтокенные задержки и время полного ответа измеряют отдельно.
Подробный протокол относится к главе
[[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/58b Benchmarking, SLO и эксплуатация inference|об измерении inference]].

## Roofline: сколько работы приходится на один байт

Предположим, что за измеренный интервал выполнено $F$ арифметических операций
и через выбранный уровень памяти передано $Q$ байт. Устройство способно
выполнять не больше $P_{\rm peak}$ операций в секунду и передавать
не больше $BW$ байт в секунду. Даже при идеальном перекрытии потребуется
не меньше времени, чем требует более длительная из этих двух работ:

$$
T\ge\max\left(\frac{F}{P_{\rm peak}},\frac{Q}{BW}\right).
$$

Поскольку производительность равна $F/T$, получаем:

$$
\begin{aligned}
I&=\frac{F}{Q},\\
P=\frac{F}{T}&\le\min(P_{\rm peak},BW\cdot I).
\end{aligned}
$$

Величина $I$ называется **арифметической интенсивностью** и измеряется
в FLOP/байт. Она показывает, сколько вычислений приходится на переданные
данные. Один FMA — умножение с накоплением — принято считать двумя FLOP;
тот же способ подсчёта должен использоваться и для пика устройства.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/performance-review-2026-09/jax-roofline.png]]

*Оригинальная схема Jacob Austin и соавторов,
[How To Scale Your Model — Visualizing Rooflines](https://jax-ml.github.io/scaling-book/roofline/#visualizing-rooflines),
[исходный PNG](https://github.com/jax-ml/scaling-book/blob/9e2bbee091bf0919a28211a2c6bc5e5d42851dbd/assets/img/roofline-improved.png),
[MIT](https://github.com/jax-ml/scaling-book/blob/9e2bbee091bf0919a28211a2c6bc5e5d42851dbd/LICENSE).
Обе оси логарифмические. Сравниваются две полосы памяти BW₁ и BW₂
при одном вычислительном пике; это схема зависимости, не измерение
конкретного GPU. Изображение также используется в материалах CS336.*

Слева предел растёт вместе с интенсивностью: данные пока поступают слишком
медленно, чтобы загрузить арифметические блоки полностью. Справа увеличение
интенсивности уже не поднимает предел: достигнута горизонтальная граница
$P_{\rm peak}$. Точка пересечения, *ridge point*, равна

$$I^*=\frac{P_{\rm peak}}{BW}.$$

На рисунке переход от BW₁ к большей BW₂ поднимает наклонную границу.
Для Algo 1 это допускает ускорение. Для Algo 2, находящегося справа
от обеих точек пересечения, увеличение полосы не меняет верхнюю границу.
Положение слева или справа указывает на более строгий из **двух пределов
этой модели**. Оно ещё не доказывает, что реальное ядро насыщает
соответствующий ресурс: ядро может работать далеко ниже обеих границ.

Roofline — название этой ломаной, похожей на линию крыши. Модель была
сформулирована в работе
[Williams, Waterman, Patterson, 2009](https://doi.org/10.1145/1498765.1498785).
Она полезна именно как ограничение и способ поставить следующий опыт,
а не как обещание достичь паспортной скорости.

### Численный пример: что изменилось после оптимизации

Возьмём **условное** устройство с пиком 120 TFLOP/s в выбранном формате
и полосой HBM 1.5 TB/s. Здесь T обозначает десятичный множитель $10^{12}$,
а байт — восемь бит. Точка пересечения равна $120/1.5=80$ FLOP/байт.

Два варианта выполняют одинаковые 72 GFLOP, но передают разный объём данных:

| Величина | Вариант A | Вариант B |
|---|---:|---:|
| Обмен с HBM | 3.6 GB | 0.9 GB |
| Интенсивность | 20 FLOP/байт | 80 FLOP/байт |
| Нижняя граница времени вычислений | 0.6 мс | 0.6 мс |
| Нижняя граница времени передачи | 2.4 мс | 0.6 мс |
| Предел roofline | 30 TFLOP/s | 120 TFLOP/s |

Допустим, в нашем расчётном примере время A равно 3 мс, а B — 1 мс.
Тогда производительность составляет 24 и 72 TFLOP/s, а ускорение — 3×.
Все эти времена заданы для объяснения; это не результаты аппаратного опыта.

У A достигается 80% его границы roofline, у B — только 60%, но B всё равно
быстрее втрое. Относительная близость к пределу и абсолютная скорость —
разные показатели. Оптимизация изменила сам предел за счёт уменьшения
обмена. Оставшийся разрыв нельзя объяснить одним словом «эффективность»:
нужно выяснить, сколько времени заняли инструкции, ожидания и неполная
загрузка устройства.

### Какую память мы считаем

В предыдущем примере $Q$ относится к HBM. Для L2 или shared memory будет
другой объём обращений и другая полоса. Один и тот же блок данных может
много раз читаться из L2, но редко запрашиваться из HBM. Поэтому
многоуровневый roofline использует $I_L=F/Q_L$ для каждого уровня $L$,
а не один и тот же размер входного массива для всех линий.

Пусть операция выполняет прежние 72 GFLOP. Предположим, что через HBM
проходит 0.9 GB, а между L2 и вычислителями — 3.6 GB: данные используются
повторно, но не каждое обращение требует чтения из HBM. Вычислительный пик
остаётся равным 120 TFLOP/s; условные полосы — 1.5 TB/s у HBM и 4.5 TB/s
у L2. Ограничения времени получаются такими:

| Ресурс | $I$, FLOP/байт | Минимум, мс |
|---|---:|---:|
| Вычисления | — | 0.6 |
| HBM | 80 | 0.6 |
| L2 | 20 | 0.8 |

Самое строгое ограничение теперь даёт L2. При идеальном перекрытии
$T\ge0.8$ мс, а $P\le90$ TFLOP/s. Если ошибочно поделить все 3.6 GB
на полосу HBM, получится 2.4 мс: обращения к кэшу будут приписаны
внешней памяти. Если, наоборот, оставить для L2 только 0.9 GB, исчезнет
ограничение, связанное с повторными чтениями.

Это важно при интерпретации необычно высокой «полосы памяти». Если разделить
логические чтения на время, а данные фактически пришли из кэша, результат
может превысить паспортную полосу HBM. Противоречия нет: числитель
не соответствовал выбранной границе памяти.

В [Nsight Compute — Roofline Charts](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline-charts)
описаны графики с отдельными ограничениями L1, L2 и DRAM. Для каждого
уровня нужно проверить определение трафика: какие чтения и записи входят
в счётчик и к какому интервалу они относятся. Числа в нашей таблице заданы
для объяснения, а не получены из такого профиля.

Вычислительный пик тоже выбирают под операцию: тип операндов, накопление,
обычные или матричные инструкции, плотные или поддерживаемые разреженные
вычисления. Нельзя сравнивать обычное FP32-сложение с FP8-пиком Tensor Cores
и трактовать получившийся процент как качество реализации.

### Последовательность операций: где суммировать время

Пусть два ядра выполняются строго последовательно: второе использует
результат первого, а реализация не запускает их части параллельно.
Сохраним условные 120 TFLOP/s и 1.5 TB/s. Первое ядро выполняет
72 GFLOP и передаёт 3.6 GB; второе — 288 GFLOP и 0.3 GB.
Для каждого отдельно действует прежняя формула. Все времена в таблице
указаны в миллисекундах:

| Ядро | $F_j/P_{\rm peak}$ | $Q_j/BW$ |
|---|---:|---:|
| Первое | 0.6 | 2.4 |
| Второе | 2.4 | 0.2 |

В каждой строке большая из двух оценок равна 2.4 мс. Поэтому последовательное
выполнение требует не менее 4.8 мс. Если сначала
сложить FLOP и байты, а затем применить roofline один раз, получится
$\max(3.0,2.6)=3.0$ мс. Это тоже нижняя граница, но более слабая:
она допускает перекрытие вычислений второго ядра с обменом первого,
которого в данном расписании нет. Свободные арифметические блоки во время
первого ядра не могут заранее выполнить зависимое второе.

Для этой последовательности полезнее считать

$$
T_{\rm seq}\ge\sum_j
\max\left(\frac{F_j}{P_{\rm peak}},\frac{Q_j}{BW}\right).
$$

Расчёт можно воспроизвести без GPU. Код ниже оценивает ограничения времени;
он не выполняет ядра и не измеряет их скорость.

```python
peak = 120e12       # FLOP/s
bandwidth = 1.5e12  # byte/s

def lower_bound_ms(flop, traffic_bytes):
    return 1000 * max(flop / peak, traffic_bytes / bandwidth)

kernels = [(72e9, 3.6e9), (288e9, 0.3e9)]
per_kernel = [lower_bound_ms(f, q) for f, q in kernels]
sequential_ms = sum(per_kernel)
pooled_ms = lower_bound_ms(
    sum(f for f, q in kernels),
    sum(q for f, q in kernels),
)
print([round(t, 1) for t in per_kernel])  # [2.4, 2.4]
print(round(sequential_ms, 1))           # 4.8
print(round(pooled_ms, 1))               # 3.0
```

Объединение ядер, конвейеризация и изменение порядка вычислений могут
поменять зависимости и объём обмена. Тогда пересчитывают и расписание,
и $Q_j$; нельзя сохранить прежние числа и считать перекрытие уже достигнутым.
В [JAX Scaling Book — Where Does the Time Go?](https://jax-ml.github.io/scaling-book/roofline/#where-does-the-time-go)
обсуждаются оценки через максимум и сумму, а затем их уточнение по
областям перекрытия. Здесь расписание задано явно, чтобы различить две
границы на конкретных числах.

## Матричное умножение: размер и повторное использование

Для $C=AB$, где формы равны $M\times K$ и $K\times N$,
арифметическая работа приближённо равна $2MKN$ FLOP. Сначала рассмотрим
идеализированный обмен: A и B прочитаны по одному разу, C записана один раз,
старое значение C не требуется. Если входы и выход занимают по $s$ байт
на элемент, получаем

$$
\begin{aligned}
Q_{\rm ideal}&=s(MK+KN+MN),\\
I_{\rm ideal}&=\frac{2MKN}{s(MK+KN+MN)}.
\end{aligned}
$$

Для квадратных BF16-матриц с $M=N=K=4096$ и BF16-выходом:

$$
\begin{aligned}
F&=2\cdot4096^3\approx137.44\text{ GFLOP},\\
Q_{\rm ideal}&=2\cdot3\cdot4096^2\\
&=100\,663\,296\text{ байт}=96\text{ MiB},\\
I_{\rm ideal}&\approx1365.33\text{ FLOP/байт}.
\end{aligned}
$$

Накопление при этом может выполняться в FP32: оно не означает, что
выход обязательно хранится в FP32. Если формат выхода другой,
его вклад нужно пересчитать отдельно. Аналогично, в операции
$C\leftarrow\alpha AB+\beta C$ при ненулевом $\beta$ потребуется
ещё и прежнее значение C.

Идеальный счёт описывает обязательные данные при холодных входах и записи
результата за рассматриваемую границу. Это не предсказание реального HBM-трафика
одного запуска: повторные чтения, промежуточные буферы и попадания в кэш
меняют обмен. Размер массива в памяти и байты, прошедшие через счётчик
профилировщика за интервал, — разные величины.

Теперь сохраним матрицу весов $K=N=4096$, а $M$ будем считать числом
обрабатываемых векторов токенов. При тех же двухбайтовых входах и выходе:

| Число векторов $M$ | $I_{\rm ideal}$, FLOP/байт | Откуда получается повторное использование |
|---:|---:|---|
| 1 | ≈1.00 | Вся матрица весов нужна одному вектору |
| 128 | ≈120.47 | Те же веса используются для 128 векторов |
| 4096 | ≈1365.33 | Много вычислений на одну загрузку весов |

Это объясняет различие между применением линейного слоя к одному новому
токену и сразу ко многим токенам. Но это ещё не расчёт целой LLM:
внимание, KV-кэш, остальные слои и фактическая реализация добавляют работу.
Подробный учёт модели продолжится в
[[04 Арифметика Transformer и MoE|следующей главе]].

## Почему соседние размеры дают разную скорость

В формуле $2N^3$ для квадратного умножения нет резких скачков, а на графике
измеренной производительности они есть. Roofline этого не запрещает:
он ограничивает скорость сверху, но не описывает каждый выбор библиотечного
ядра и размещение работы на SM.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/l05-p47.png]]

*Stanford CS336, [Lecture 5, слайд 47](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_05.pdf),
по разбору Horace He,
[What Shapes Do Matrix Multiplications Like?](https://www.thonking.ai/p/what-shapes-do-matrix-multiplications).
Слева — TFLOP/s для разных сторон квадратной матрицы; легенда разделяет
размеры по кратности. Справа — учебная схема доступа к выровненным
и невыровненным строкам. Полный слайд сохранён без изменений; график
относится к опыту автора, не к измерениям Bookvar.*

Здесь стоит различить несколько механизмов. Длина строки определяет,
как начинаются следующие строки в памяти и какие загрузки можно использовать.
Выходная матрица разбивается на блоки фиксированной формы;
неполные крайние блоки тоже требуют выполнения. Число блоков
может перестать помещаться в прежнее число волн исполнения.
Наконец, сама библиотека может выбрать для новой формы другое ядро.

### Переход от 1792 к 1793

В опыте Horace He профилировщик показал одно и то же CUTLASS-ядро
с выходными блоками $256\times128$ около рассматриваемой границы.
Число блоков для стороны N равно

$$
G(N)=\left\lceil\frac{N}{256}\right\rceil
\left\lceil\frac{N}{128}\right\rceil.
$$

При $N=1792$ получается $7\cdot14=98$ блоков, а при $N=1793$ —
$8\cdot15=120$. Число полезных FLOP выросло лишь примерно на 0.17%,
но понадобились ещё 22 выходных блока, причём новые блоки заполнены
не полностью.

У A100 в этом разборе 108 SM. **Если выбранное ядро размещает только
один блок на SM**, 98 блоков укладываются в одну волну, а 120 требуют
второй: сначала до 108, затем оставшиеся 12. Значительная часть устройства
в конце простаивает. Это *wave quantization*, или ступенчатое изменение
числа волн.

Неполные блоки и новая волна возникли одновременно, но это не одно и то же.
Если бы на SM помещалось по два таких блока, вместимость волны составила
бы 216 и переход 98 → 120 сам по себе не добавил бы волну. Продолжительность
также не обязана увеличиться ровно вдвое: блоки могут стоить разное время,
а загрузки и расписание — перекрываться.

В [руководстве NVIDIA по GEMM](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#wave-quant)
раздельно разобраны заполнение блоков и число волн. Для своей программы
нужно записать выбранное ядро, размер его сетки и ресурсные ограничения.
Сам по себе зубец на графике не доказывает причину. Проверяют несколько
близких форм, в том числе $N-1$, $N$ и $N+1$, сохраняя точность,
размещение данных и условия прогрева.

## От времени к причине: профилирование

![[02 Areas/ML & DL/Assets/Sources/Harvard ML Systems/vol1/benchmarking/682e143e450db0d85ad5c608e3a125ac5176e0ba.svg]]

*Harvard ML Systems, [Isolation vs. Representativeness](https://mlsysbook.ai/vol1/benchmarking/benchmarking.html#fig-benchmark-tradeoffs),
Harvard Edge ML Systems Book contributors,
[CC BY-NC-SA 4.0](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/LICENSE.md).
Это качественная схема компромисса, не измеренная зависимость:
изолированный опыт упрощает поиск причины, полный — лучше воспроизводит
работу приложения. Оригинальный SVG не изменён.*

Микроизмерение позволяет менять одну операцию, оставляя входы и остальные
условия неизменными. В полном приложении одновременно меняются очереди,
кэши и подготовка данных, поэтому выигрыш труднее объяснить. Но именно
полный опыт показывает, помогла ли оптимизация пользователю. На практике
нужны оба уровня, а не выбор одного из них навсегда.

Таймер отвечает на вопрос «сколько заняло». Профилировщик помогает выяснить,
где прошло это время. Сначала полезна общая временная шкала: когда CPU
запускает операции, где выполняются копирования, когда GPU занят и
какой участник ждёт другого. Для этого используют PyTorch Profiler или
Nsight Systems. Затем отдельное дорогое ядро изучают подробнее в Nsight Compute.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/a2-nsight-trace.png]]

*Stanford CS336 course staff, [Assignment 2 — Nsight Systems, с. 5](https://github.com/stanford-cs336/assignment2-systems/blob/ca8bc81a59b70516f7ebb2da4808daade877c736/cs336_assignment2_systems.pdf#page=5),
[MIT](https://github.com/stanford-cs336/assignment2-systems/blob/ca8bc81a59b70516f7ebb2da4808daade877c736/LICENSE).
Снимок из задания курса: сохранена вся область рисунка, без перерисовки.
Для чтения названий ядер изображение можно открыть в полном размере.*

Горизонтальная ось — время. Внизу дорожки PyTorch и NVTX связывают
интервал с вызовом модели и её слоёв; выше видна последовательность
CUDA-ядер. Ещё выше идут разные показатели устройства: GPU Active,
SM Warp Occupancy и DRAM Bandwidth. Их нельзя читать как три названия
одной «загрузки»: занятое устройство может мало передавать из DRAM,
а размещённые warps могут ждать выполнения инструкции.

Чтобы исследовать паузу, сначала находят её на дорожке ядер, затем
сопоставляют тот же интервал с вызовами и передачами. Большая общая доля
GPU Active ещё не говорит, что каждый слой выполняется эффективно.
Числа на снимке принадлежат опыту авторов задания; это не замер
кода из этой главы.

Если между короткими ядрами большие промежутки, проверяют отправку работы
с CPU и точки синхронизации. Если ядро непрерывно читает много данных,
смотрят фактическую полосу и повторное использование. Если малы и скорость
вычислений, и обмен, это ещё не диагноз: возможны малая сетка, зависимости
инструкций, ожидание памяти или ограничения размещения блоков.
Изолированный процент occupancy не заменяет эту последовательность проверок.

Профилирование может менять опыт. Для сбора счётчиков Nsight Compute
способен повторять ядро, управлять частотами и очищать кэши между повторами.
Поэтому нельзя без проверки соединять время из прогретого обычного запуска
с трафиком из профиля, полученного при холодном кэше. Настройки replay,
cache control и clocks фиксируют вместе с измерением.
[Nsight Compute — Reproducibility](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#reproducibility).

Оптимизацию проверяют ещё раз без диагностического инструмента и на той
границе, ради которой она делалась. Выигрыш в одном ядре мог исчезнуть
из-за новой перестановки тензора, копирования или роста очереди.

<!-- BEGIN REVIEW: fleet measurement 2026-10-26 -->

<a id="fleet-mfu-scaling-and-preserved-progress"></a>

## MFU and scaling: measuring the complete training job

A training step can keep every GPU busy while spending much of its time
recomputing activations or waiting inside communication kernels. A short
profile can also omit checkpoint saves and restarts. To compare training
systems, first specify the model work being counted, the devices available
to do that work, and the interval over which progress is measured.

### MFU and HFU count different work

The Ultra-Scale Playbook explains why counting every executed operation
can reward extra work. The following two paragraphs are reproduced in their
original English from its [activation recomputation section](https://huggingface.co/spaces/nanotron/ultrascale-playbook),
Nouamane Tazi and coauthors, [Apache-2.0](/licenses/ultrascale-Apache-2.0.txt):

<!-- BEGIN SOURCE: Ultra-Scale HFU and MFU -->
When you’re measuring how efficient your training setup is at using your
GPU/TPU/accelerator, you usually want to take recomputation into account to
compute total FLOPS (Floating point operations per second) and compare it to
theoretical maximum FLOPS of the GPU/TPU/accelerator. Taking recomputation into
account when calculating FLOPS for a training step gives a value called
“hardware FLOPS” which is the real number of operations performed on the
accelerator. Dividing this number by the duration of the training step and the
maximum accelerator FLOPS yields the **_Hardware FLOPS Utilization (HFU)._**

However, what really matters at the end of the day is the start-to-end time
needed to train a model on a given dataset. So when comparing various
GPU/TPU/accelerator together, if one of these accelerator provide for instance
enough memory to skip recomputation and thus perform less operation per second
(lower HFU) but for a faster training, it should be rewarded not punished. Thus,
an alternative is to compute what is called **_Model FLOPS Utilization (MFU)_**
which, in contrast to HFU, only takes into account the required operations for
the forward+backward passes through the model, and do not include recomputation
in the measured FLOPs. This value is thus more specific to the model than the
training implementation.
<!-- END SOURCE: Ultra-Scale HFU and MFU -->

For a fixed set of devices and one complete step, write this as

$$
\begin{aligned}
\mathrm{MFU}&=\frac{F_{\mathrm{model}}}
{t_{\mathrm{step}}\sum_i P_{\mathrm{peak},i}},\\
\mathrm{HFU}&=\frac{F_{\mathrm{executed}}}
{t_{\mathrm{step}}\sum_i P_{\mathrm{peak},i}}.
\end{aligned}
$$

$F_{\mathrm{model}}$ is a declared count of the model's forward and backward
work for the global batch. $F_{\mathrm{executed}}$ additionally includes work
such as activation recomputation. Neither numerator is the fraction of time
with a running kernel reported by `nvidia-smi`.

For example, suppose the model requires 600 TFLOP, recomputation adds
200 TFLOP, and the devices could execute 1000 TFLOP during the measured
step. MFU is 60% and HFU is 80%. Removing recomputation while keeping the
same step time would reduce HFU to 60% without reducing MFU. If the step
also becomes faster, MFU rises. Whether removing recomputation is feasible
depends on the memory needed to retain the activations.

For a dense Transformer, $6PD$ is a common approximation to the model
training work: $P$ counts parameters under the stated convention and $D$
counts tokens in the global batch. It omits terms such as attention's
sequence-length-dependent work; the
[[02 Areas/ML & DL/00 Учебник/10 ML Systems/04 Арифметика Transformer и MoE|model arithmetic chapter]]
derives the fuller count. Specify whether padding contributes to $D$.
Discarding padding from a reporting numerator does not make dense kernels
skip those positions. For MoE, total stored parameters cannot simply replace
the active parameter count in the dense estimate.

Count the global batch once. A log copied to every rank does not represent
that many independent copies of the model work. Tensor- and pipeline-parallel
ranks divide that work; data-parallel replicas process different portions of
the batch. The denominator must include every device allocated to the measured
job, including devices waiting for dependencies.

Choose a peak for the operation type and numerical format being reported.
Dividing dense BF16 work by a sparse FP8 advertising peak changes the meaning
of the result. A mixed-format step needs an explicit counting convention;
tokens per second and time to the quality target should accompany the ratio.
The [[02 Areas/ML & DL/00 Учебник/10 ML Systems/05 Численные форматы и mixed precision#Почему после смены формата меняется знаменатель MFU|mixed-precision chapter]]
returns to this denominator when changing formats.

### Strong scaling and weak scaling answer different questions

In **strong scaling**, keep the model and global batch fixed and increase
the number of devices. If $n_0$ devices take $T_0$ seconds and $n$ devices
take $T_n$, the speedup and scaling efficiency are

$$
S=\frac{T_0}{T_n},\qquad
\eta_{\mathrm{strong}}=\frac{S}{n/n_0}.
$$

The ideal step time is $T_0n_0/n$. Each device receives less work as the job
grows, while communication and synchronization do not necessarily shrink.
The experiment asks how much sooner the same batch can finish.

In a **weak-scaling** experiment, increase the total work with the resources.
For data parallelism, one possible protocol holds the model, sequence length
and tokens per device fixed, and increases the global batch with $n$.
The ideal step time then remains $T_0$, and $\eta_{\mathrm{weak}}=T_0/T_n$.
This protocol changes the global batch and can change optimization behavior:
equal step times do not establish equal time to the same model quality.

Some model-parallel studies instead increase model size with the device
count. That also changes GEMM shapes and arithmetic intensity. Read the
experimental conditions before interpreting a curve labelled “weak scaling.”

NVIDIA's [Megatron-LM, Training Speed and Scalability](https://github.com/NVIDIA/Megatron-LM/blob/73a28a1078a8da8e6062199f7f1079a52173ab77/README.md#training-speed-and-scalability)
provides both kinds of plot. Its complete setup paragraph is retained below,
followed by the original explanations and figures. These are reported NVIDIA
results, not measurements reproduced in Bookvar. The source is pinned to
`73a28a1`; the figures themselves identify Megatron-Core r0.8.0.

<!-- BEGIN SOURCE: Megatron scaling setup -->
Our codebase is capable of efficiently training large language models (i.e., models with hundreds of billions of parameters) with both model and data parallelism. To demonstrate how our software scales with multiple GPUs and model sizes, we consider GPT models ranging from 2 billion parameters to 462 billion parameters. All models use a vocabulary size of 131,072 and a sequence length of 4096. We vary hidden size, number of attention heads, and number of layers to arrive at a specific model size. As the model size increases, we also modestly increase batch size. Our experiments use up to 6144 [H100](https://www.nvidia.com/en-us/data-center/h100/) GPUs. We perform fine-grained overlapping of data-parallel (`--overlap-grad-reduce --overlap-param-gather`), tensor-parallel (`--tp-comm-overlap`) and pipeline-parallel communication (enabled by default) with computation to improve scalability. The reported throughputs are measured for end-to-end training and include all operations including data loading, optimizer steps, communication, and even logging. Note that we did not train these models to convergence.
<!-- END SOURCE: Megatron scaling setup -->

<!-- BEGIN SOURCE: Megatron weak scaling -->
Our weak scaled results show superlinear scaling (MFU increases from 41% for the smallest model considered to 47-48% for the largest models); this is because larger GEMMs have higher arithmetic intensity and are consequently more efficient to execute.
<!-- END SOURCE: Megatron weak scaling -->

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/fleet-measurement-2026-10/megatron-weak-scaling.png]]

*NVIDIA, [original weak-scaling PNG](https://github.com/NVIDIA/Megatron-LM/blob/73a28a1078a8da8e6062199f7f1079a52173ab77/images/weak_scaling.png),
copied in full without changes, 4737 × 2125;
[repository BSD-3-Clause terms](https://github.com/NVIDIA/Megatron-LM/blob/73a28a1078a8da8e6062199f7f1079a52173ab77/LICENSE).
Both axes are logarithmic. Green annotations give model size; the dashed
blue line is perfect linear scaling relative to the baseline.*

The green points cross above the dashed reference as the model grows.
That does not show the same small model becoming more efficient merely
because more GPUs were assigned to it. The workload changed: each point
has a different parameter count, and the setup also changes batch size.
Larger matrix multiplications can use the device more efficiently, as in
the earlier arithmetic-intensity example.

<!-- BEGIN SOURCE: Megatron strong scaling -->
We also strong scaled the standard GPT-3 model (our version has slightly more than 175 billion parameters due to larger vocabulary size) from 96 H100 GPUs to 4608 GPUs, using the same batch size of 1152 sequences throughout. Communication becomes more exposed at larger scale, leading to a reduction in MFU from 47% to 42%.
<!-- END SOURCE: Megatron strong scaling -->

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/fleet-measurement-2026-10/megatron-strong-scaling.png]]

*NVIDIA, [original strong-scaling PNG](https://github.com/NVIDIA/Megatron-LM/blob/73a28a1078a8da8e6062199f7f1079a52173ab77/images/strong_scaling.png),
copied in full without changes, 4737 × 2125, same BSD-3-Clause terms.
The model is labelled 177B at every point. The setup holds the global batch
at 1152 sequences, unlike the preceding plot.*

The rightmost green points fall below the linear reference. More devices
still produce higher aggregate throughput, but the gain is less than
proportional to the resource increase. A lower MFU therefore need not mean
a slower job. At fixed model work, fixed precision and identical devices,
the MFU ratio equals strong-scaling efficiency:

$$
\frac{\mathrm{MFU}_n}{\mathrm{MFU}_{n_0}}
=\frac{T_0}{T_n}\frac{n_0}{n}
=\eta_{\mathrm{strong}}.
$$

This identity does not apply unchanged to the first plot, where model size
changes. Nor does it make falling MFU a universal law: a new device count
can enable a different partition, remove memory pressure, or change kernels.

### Reconstructing the 8-to-128-GPU calculation

Harvard's [Measurement at Scale](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/performance_engineering/performance_engineering.qmd#L2419-L2550)
uses a 70B model, 8 versus 128 GPUs, and step times of 180 versus 245 ms.
Its source code labels those times as *scenario assumptions*. The calculation
below makes the approximation explicit with $6PD$ and a chosen dense peak of
989 TFLOP/s per device; it is not an H100 benchmark.

| Quantity | Local baseline | Larger job |
|---|---:|---:|
| GPUs | 8 | 128 |
| Global tokens per step | 2204 | 35446 |
| Assumed step time, ms | 180 | 245 |
| Approximate model work, PFLOP | 0.92568 | 14.88732 |
| Peak budget during the step, PFLOP | 1.42416 | 31.01504 |
| MFU | 65.00% | 48.00% |

The device count increases 16-fold, while tokens increase about 16.08-fold.
Tokens per device are nearly constant: 275.5 versus about 276.92. This is
close to a weak-scaling experiment with a larger global batch, not a
strong-scaling experiment on the same batch.

```python
def dense_training_mfu(parameters, tokens, devices, step_seconds, peak_per_device):
    model_flops = 6 * parameters * tokens
    peak_budget = devices * peak_per_device * step_seconds
    return model_flops / peak_budget

local = dense_training_mfu(70e9, 2204, 8, 0.180, 989e12)
fleet = dense_training_mfu(70e9, 35446, 128, 0.245, 989e12)
print(f"MFU: {100 * local:.2f}% -> {100 * fleet:.2f}%")
print(f"Relative decrease: {100 * (1 - fleet / local):.2f}%")
print(f"Percentage-point decrease: {100 * (local - fleet):.2f}")
```

The outputs are 65.00% → 48.00%, a 26.15% relative decrease and a
17.00-percentage-point decrease. These are two descriptions of the same
change, not interchangeable percentages. The MFU ratio includes both
the work ratio and the device/time ratio; using only $180/245$ loses the
small difference in tokens per device.

This ratio alone does not attribute the decrease to InfiniBand. Padding,
microbatch sizes, kernel choices, recomputation and rank imbalance can also
change. Hold these conditions fixed where possible, compare traces from
the participating ranks, and measure collectives separately before assigning
a cause. The [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/44a Processes, collectives и DDP#ddp-overlap-readiness-streams-compilation|DDP overlap explanation]]
shows why a late rank can extend a communication interval even when the
network itself has not slowed down.

### Aggregate work and capacity, not percentages

For a resource pool observed over a window, add the non-duplicated model
work and divide by the capacity available over that same window:

$$
U=\frac{\sum_j F_{\mathrm{model},j}}
{\int_{t_0}^{t_1}\sum_{i\in\mathcal A(t)}P_{\mathrm{peak},i}\,dt}.
$$

$\mathcal A(t)$ specifies the devices included in the report. Using allocated
devices counts their idle time against the job; using only active devices
answers a different question. If the allocation changes, split the interval
at those changes. The numerical format and operation convention must remain
compatible across numerator and denominator.

For example, consider two independent jobs over the same one-second window.
One has a 100-TFLOP/s peak and reports 80% MFU; the other has a
400-TFLOP/s peak and reports 20%. Their model work is 80 and 80 TFLOP.
The pool ratio is $(80+80)/(100+400)=32\%$, not the unweighted average
of 50%. For one distributed job, do not add an identical global FLOP estimate
from every rank; it is still only one global step.

Duration matters too. On an unchanged allocation, 80% for ten seconds and
20% for ninety seconds give $(0.8\cdot10+0.2\cdot90)/100=26\%$.
A mean of the two dashboard percentages would again report 50%.
Preserve the work, capacity and duration needed to reconstruct the aggregate.

### Preserved progress, checkpointing and recovery

A fast steady-state step does not tell how much training survives a restart.
Google's [ML Goodput Measurement](https://github.com/AI-Hypercomputer/ml-goodput-measurement/blob/fa782faaeb10b994557c7850b2e1bdc8cc5ec414/README.md#retrieve-badput-breakdown),
Google LLC, [Apache-2.0](/licenses/goodput-fa782faa-Apache-2.0.txt), distinguishes the following costs. The selected
paragraphs are reproduced without translation:

<!-- BEGIN SOURCE: Goodput badput definition -->
Badput breakdown is dictionary representation of various sources of Badput
mapped to its corresponding value. Badput is the percentage of time spent by the
job doing work that is not training to the total lifetime of the job. This
includes time spent doing device initialization, training preparation,
program startup, checkpoint loading, compilation or re-compilation, data loading,
checkpoint saving, custom badput events, wasted progress and time lost due
to disruptions.
<!-- END SOURCE: Goodput badput definition -->

<!-- BEGIN SOURCE: Goodput data loading -->
 - Data Loading Time (DATA_LOADING_SYNC)

  This is the time spent on loading each batch of data for the training at a
  step to continue. This should be a small contribution to Badput if parallel
  data loading is used.

  Async data loading is accumulated overlapping with training steps and is
  non-blocking, therefore is not unproductive time. The time spent on overlapped
  data loading is stored in BadputType.DATA_LOADING_ASYNC, but does **not**
  affect overall Goodput of the workload.
<!-- END SOURCE: Goodput data loading -->

<!-- BEGIN SOURCE: Goodput wasted progress -->
 - Wasted Progress due to Disruption (WASTED_PROGRESS_FROM_DISRUPTION)

 Based on checkpointing frequency, a disruption may result in time lost in the
 form of wasted progress, i.e. time that was spent on productive training but
 lost after restart as well as time lost for the infrastructure to restart the
 workload.
<!-- END SOURCE: Goodput wasted progress -->

The last paragraph mentions two effects: lost training progress and time
spent restarting. The library lists infrastructure recovery as a separate
bucket immediately afterward. Assign each interval once when constructing
a breakdown; counting recovery in both buckets would inflate badput.
Likewise, asynchronous work that overlaps useful training is not an additional
serial interval to subtract. It can still compete for resources and slow
the training steps, which requires a separate comparison.

Consider this one-hour accounting example on a fixed allocation. The rows
are disjoint assumed intervals, not measured values from the library:

| State | Seconds | What remains after recovery |
|---|---:|---|
| Training whose progress is preserved | 2700 | Updates remain on the final training trajectory |
| Training discarded after a failure | 300 | Updates must be repeated after restore |
| Startup and compilation | 120 | No training updates |
| Blocking checkpoint saves | 180 | Saved state, no concurrent updates in this example |
| Infrastructure recovery and restore | 300 | No training updates |
| Total | 3600 | 2700 seconds of preserved training |

The GPUs execute training for 3000 seconds, but only 2700 seconds contribute
progress that remains after recovery. The productive-time fraction is
$2700/3600=75\%$. Counting all executed training would give
$3000/3600\approx83.33\%$ and hide the 300 seconds of lost progress.
The retained intervals include any replay needed to reach the final state;
an update lost before restart is not counted a second time as preserved work.

If all training intervals have the same model-work rate and steady-state
MFU is 50%, preserved model work divided by the full hour's peak capacity
is $0.50\cdot0.75=37.5\%$. That factorization relies on the constant rate
and fixed allocation. With variable step costs or elastic resources, use
the actual work and capacity integral instead. This progress-normalized
ratio is an explicitly defined accounting metric, not a claim that the
library calculates MFU or that a 75% goodput fraction means 75% of peak FLOP/s.

```python
preserved = 2700
discarded = 300
startup = 120
checkpoint = 180
recovery = 300
elapsed = preserved + discarded + startup + checkpoint + recovery
productive_fraction = preserved / elapsed
print(f"Productive time: {100 * productive_fraction:.2f}%")  # 75.00%
print(f"Preserved-work utilization: {100 * 0.50 * productive_fraction:.2f}%")
# Preserved-work utilization: 37.50%
```

This definition concerns training time and retained progress. In serving,
“goodput” often counts requests that meet a latency SLO instead; the
[[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/58b Benchmarking, SLO и эксплуатация inference|inference benchmarking chapter]]
defines that different numerator. Checkpoint intervals and recovery state
are developed in the [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/44h Fault tolerance и fleet orchestration|fault-tolerance chapter]].

For time-to-quality comparisons, use the benchmark's actual timing rules.
[MLPerf Training, Run Results and Quality measure](https://github.com/mlcommons/training_policies/blob/14bc89fe00691a5181f401bd67dc242dc928b205/training_rules.adoc#run-results)
defines a contiguous wall-clock interval, quality criteria and explicit
exceptions for initialization and offline evaluation. Its Result files
section requires framework logs and MLLOG lines. A production run can have
different checkpoint and monitoring requirements, but there is no universal
10–20% deduction that converts its benchmark result into production capacity.
Measure those expenses over the intended workload and reporting window.

<!-- END REVIEW: fleet measurement 2026-10-26 -->

## Мощность, энергия и итоговый отчёт

Мощность измеряется в ваттах, энергия — в джоулях:

$$E=\int_{t_0}^{t_1}P(t)\,dt.$$

Меньшая мощность не обязательно означает меньшую энергию на задачу.
Если выполнение ускорилось вдвое, а средняя мощность выросла в 1.5 раза,
энергия этой работы стала равна $1.5/2=0.75$ от исходной. Условия качества
и граница измерения должны оставаться одинаковыми: показания платы GPU,
всего сервера и стойки — не одна величина.

Короткое ядро может завершиться между двумя отсчётами телеметрии. Тогда
измеряют достаточно длинную серию и сообщают интервал наблюдения,
число выполненных операций и то, вычиталась ли энергия простоя.
Для обучения важна энергия до нужного качества, для сервиса — энергия
на полезно завершённую работу при согласованной нагрузке.

Хороший отчёт позволяет восстановить опыт. В нём остаются:

1. задача, эталон и условия проверки качества;
2. входы, формы, типы, размещение и граница таймера;
3. версии программы, библиотек и оборудования;
4. прогрев, кэш, порядок A/B, число повторов и исходные наблюдения;
5. показатель с единицами, разброс, ошибки и ограничения вывода.

Уменьшение времени на 30% относится к указанной нагрузке и границе таймера.
В отчёте должны остаться исходное и новое время, а также результат
проверки качества: без них нельзя установить, что именно ускорилось.

## Источники и практика

- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol1/benchmarking|Harvard ML Systems — Benchmarking]]:
  границы измерения, постановка опыта и отчётность.
- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol2/performance_engineering|Harvard ML Systems — Performance Engineering]]:
  арифметическая интенсивность и связь аппаратных пределов с оптимизацией.
- [How To Scale Your Model — All About Rooflines](https://jax-ml.github.io/scaling-book/roofline/):
  вывод ограничений времени и примеры матричных операций.
- [Stanford CS336, Lecture 5](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_05.pdf):
  аппаратные пределы, блочное умножение и зависимость скорости от размеров.
- [Stanford CS336, Lecture 6](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_06.py):
  код измерений и профили поэлементных и матричных операций.
- [Stanford CS336, Assignment 2](https://github.com/stanford-cs336/assignment2-systems/blob/ca8bc81a59b70516f7ebb2da4808daade877c736/cs336_assignment2_systems.pdf):
  задание на прогрев, синхронизацию и измерение прямого/обратного прохода.
- [PyTorch Benchmark](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)
  и [CUDA events](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.Event.html):
  готовые инструменты и точные условия измерения.
- [Horace He — What Shapes Do Matrix Multiplications Like?](https://www.thonking.ai/p/what-shapes-do-matrix-multiplications):
  исходный разбор графиков, использованных в CS336, с кодом опыта.
- [NVIDIA — Matrix Multiplication Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html):
  выравнивание, заполнение блоков и wave quantization.
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week01_intro/seminar.ipynb|EDLS Week 1 — семинар]]
  и [[05 Источники/Courses/Harvard ML Systems/tinytorch/19_benchmarking|TinyTorch 19 — Benchmarking]]:
  материалы для самостоятельного воспроизведения измерений.

← [[02 Areas/ML & DL/00 Учебник/10 ML Systems/02 GPU, CUDA и иерархия памяти|GPU, CUDA и иерархия памяти]] ·
[[02 Areas/ML & DL/00 Учебник/10 ML Systems/04 Арифметика Transformer и MoE|Арифметика Transformer и MoE]] →
