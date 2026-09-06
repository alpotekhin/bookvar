---
title: GPU kernels и Triton — от программы к измерению
type: textbook-chapter
status: draft
last_verified: 2026-09-06
source_language: mixed
source_unit_id:
  - lecture-05-coalescing-failures
  - lecture-05-operator-fusion
  - lecture-05-tiling-derivation
  - lecture-05-performance-recap-transition
  - lecture-06-kernel-fusion-failure-mode
  - lecture-06-section-244-naive-gelu
  - lecture-06-gelu-profiler-comparison
  - lecture-06-section-247-builtin-gelu
  - lecture-06-section-250-compiled-gelu
  - lecture-06-triton-softmax-worked-example
  - lecture-06-figure-step-321-rendering-1
  - lecture-06-figure-step-353-rendering-1
  - lecture-06-matmul-tiling-derivation
  - assignment-02-task-torch-compile
primary_sources:
  - https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_06.py
  - https://github.com/triton-lang/triton
  - https://triton-lang.org/main/getting-started/tutorials/
  - https://openai.com/index/triton/
---

<a id="cs336-systems-triton"></a>

# GPU kernels и Triton — от программы к измерению

В PyTorch одна строка может описывать десятки арифметических операций. GPU при
этом видит не математическую строку, а последовательность kernels, чтений и
записей, которую создали dispatcher, compiler и backend. Пока операция велика,
эта разница незаметна: один GEMM занимает device достаточно долго. У цепочки
коротких elementwise-операций запуск и промежуточные записи в HBM могут стоить
дороже самой арифметики.

Эта глава проходит один и тот же инженерный цикл несколько раз:

1. записать ясную reference-функцию;
2. определить shapes, dtype и допустимую ошибку;
3. посчитать минимальные чтения, записи и FLOPs;
4. измерить warmed execution;
5. проверить profiler trace;
6. объединить работу в Triton program;
7. снова проверить correctness, latency и несколько неудобных shapes.

Triton здесь не замена CUDA и не обещание автоматического ускорения. Это язык,
на котором программист задаёт блоки данных и операции над ними, а compiler
берёт на себя значительную часть отображения на threads, warps и инструкции.
Чтобы написать быстрый kernel, всё равно нужно понимать memory hierarchy,
tiling, occupancy и численную точность.

## От одной формулы PyTorch к нескольким проходам по HBM

Рассмотрим tanh approximation для GeLU:

$$
\operatorname{GELU}(x)=\frac{x}{2}
\left(1+\tanh\left(\sqrt{\frac{2}{\pi}}
(x+0.044715x^3)\right)\right).
$$

Наивная PyTorch-запись удобна как спецификация. Но если каждый оператор
запускается отдельно, степени, умножения, сложения и `tanh` читают input или
предыдущий intermediate из HBM и записывают следующий. У каждого прохода почти
одинаковый объём данных, хотя полезной арифметики мало.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/l05-p31.png]]

*Stanford CS336 Lecture 5, p. 31: общая разница между несколькими отдельными
kernels и одной fused-операцией. Считать нужно переходы через HBM, а не только
число математических операторов. [Pinned PDF](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_05.pdf).*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/l05-p32.png]]

*На следующем слайде эта схема получает конкретный вычислительный граф:
$\sin^2(x)+\cos^2(x)$ в наивной записи запускает пять CUDA kernels и
материализует промежуточные тензоры. Источник: Stanford CS336 Lecture 5, p. 32,
[pinned PDF](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_05.pdf); полный 220-dpi render без crop.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/l05-p33.png]]

*Stanford CS336 Lecture 5, p. 33: compiler объединяет пять pointwise-операций
предыдущего графа в один CUDA kernel. Арифметика та же, но исчезают
materialization и повторные reads. Три слайда оставлены последовательностью:
общая модель переноса данных → конкретный FX graph → результат fusion.*

Для GeLU Lecture 6 сравнивает три реализации:

- исходную композицию PyTorch operators;
- builtin `torch.nn.functional.gelu`;
- `torch.compile` исходной композиции.

Сначала outputs сравниваются, затем измеряется steady state и только потом
читается профиль. Builtin может уже вызывать специализированный kernel;
compiler может fuse граф, но платит cold compile и guards. Поэтому “меньше
kernels” — механизм, а не результат: результатом остаётся корректная функция с
меньшим временем на заданном наборе shapes.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/gelu-profile-naive.png]]

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/gelu-profile-builtin.png]]

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/gelu-profile-compiled.png]]

*Это три последовательных вывода профилировщика из Stanford CS336 Lecture 6,
шаги трассы 245, 248 и 251: одинаковый `dim=16384`, но девять запусков в eager
режиме против одного библиотечного или скомпилированного ядра. Арифметика GeLU
сохраняется, а отдельные запуски и промежуточные проходы через HBM исчезают.
Времена — наблюдение на машине курса, не обещание ускорения на другом GPU.
[Закреплённая версия исходного кода](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_06.py#L265-L302).*

## Контракт собственного kernel

До Triton-кода фиксируют:

- входные и выходные shapes, strides и допустимую non-contiguity;
- dtype operands, accumulator и output;
- точную математическую семантику, включая approximation mode;
- поведение на tail, пустых размерах, NaN/Inf и aliasing;
- reference implementation и `atol`/`rtol`;
- формы для benchmark, в том числе не кратные block size.

Сравнение только одного output недостаточно для differentiable operator.
Необходимы forward, backward по каждому входу и, когда возможно,
`gradcheck` в более высокой точности. Порядок редукции на GPU отличается от
reference, поэтому побитовое равенство float tensors обычно неверный контракт.

## Что именно задаёт Triton program

CUDA-программист мыслит thread blocks и threads. В Triton основная единица —
program instance: один запуск функции обрабатывает целый block индексов. Grid
говорит, сколько instances нужно; `tl.program_id(axis)` выбирает текущий.

Для одномерного elementwise kernel:

```python
@triton.jit
def kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = operation(x)
    tl.store(y_ptr + offsets, y, mask=mask)
```

Здесь `offsets` — вектор логических индексов, а `mask` запрещает последнему
экземпляру программы читать и записывать элементы за границей. Размер блока
`BLOCK` известен компилятору и потому объявлен `tl.constexpr`; фактическая длина
`n` приходит во время запуска. `BLOCK` часто выбирают степенью двойки, но длина
массива не обязана ей делиться. Ошибка в маске может не проявиться на удобном
размере и привести к обращению к памяти за границей массива.

Этот уровень не отменяет аппаратную иерархию. Compiler должен назначить
операции lanes/warps, разместить живые значения в registers/shared memory и
сгенерировать memory transactions. Большой block повышает работу на launch, но
может увеличить register pressure; маленький block создаёт больше launches и
может терять reuse.

## Первый полный пример: softmax одной строки

Для $X\in\mathbb{R}^{M\times N}$ softmax по строкам равен

$$
y_{ij}=\frac{\exp(x_{ij}-m_i)}
{\sum_{k=1}^{N}\exp(x_{ik}-m_i)},
\qquad m_i=\max_k x_{ik}.
$$

Вычитание maximum не меняет результат в точной арифметике, но предотвращает
overflow экспоненты. Наивная композиция `max → subtract → exp → sum → divide`
может прочитать элементы порядка пяти раз и создать несколько intermediates.
Если строка помещается в один Triton block, program читает её один раз,
выполняет reductions on chip и один раз записывает output.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/triton-softmax.png]]

*Оригинальная схема [Stanford CS336 Lecture 6, step 321](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_06.py),
`images/triton-softmax.png`: один program обрабатывает одну строку, а mask
закрывает padded columns. Рисунок перенесён без перерисовки.*

Порядок программы:

1. `row = tl.program_id(0)` выбирает строку;
2. `tl.arange` строит offsets до ближайшей подходящей `BLOCK_SIZE`;
3. masked load подставляет $-\infty$ за границей, чтобы padding не изменил max;
4. `tl.max`, `tl.exp` и `tl.sum` выполняют стабильный softmax;
5. masked store пишет только $N$ настоящих элементов.

Корректность проверяют на случайной матрице, одинаковых значениях, одном
доминирующем logit, отрицательных значениях и widths `BLOCK-1`, `BLOCK`,
`BLOCK+1`. Последний случай может потребовать другую стратегию: одна строка уже
не помещается в выбранный block.

## Reduction, которая не помещается в один tile

Пусть в строке 4096 элементов, а program обрабатывает по 1024. Нельзя сложить
четыре независимых результата, забыв про их место в общей редукции. Для простой
суммы program держит vector accumulator, проходит tiles циклом и затем делает
финальную `tl.sum`.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/triton-row-sum.png]]

*Оригинальная схема [Stanford CS336 Lecture 6, step 353](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_06.py),
`images/triton-row-sum.png`: те же lanes последовательно получают несколько
tiles, а masked accumulator переживает цикл. Рисунок показывает, что tile и
полная строка — разные масштабы.*

Для softmax длинной строки состояния сложнее: нужно сохранить running maximum
и нормировочную сумму и при появлении нового максимума перемасштабировать старый
вклад. Именно этот online-softmax invariant станет основой FlashAttention.

## Tiled GEMM: reuse вместо идеализированной загрузки всей матрицы

Для $C=AB$, $A\in\mathbb{R}^{M\times K}$,
$B\in\mathbb{R}^{K\times N}$, наивное вычисление каждого $C_{mn}$ независимо
читает строку A и колонку B заново. Идея “загрузить A и B целиком on chip” дала
бы максимальный reuse, но большие матрицы не помещаются. Tiling выбирает
промежуточный масштаб.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/l05-p42.png]]

*Stanford CS336 Lecture 5, p. 42: output tile C связывается с row tile A и
column tile B. Один загруженный элемент участвует в нескольких FMA внутри
tile. [Pinned PDF](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_05.pdf).*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/l05-p43.png]]

*Stanford CS336 Lecture 5, p. 43: неполный крайний tile объясняет wasted work и
masked loads. Он следует за основной схемой, потому что tail — следствие
выбранного разбиения, а не отдельная оптимизация.*

Program получает координаты output tile $(p_m,p_n)$, создаёт offsets по M, N и
K, затем проходит K-blocks:

$$
C_{tile}\leftarrow C_{tile}+A_{m,k}B_{k,n}.
$$

Accumulator обычно шире input dtype. A/B tiles переиспользуются для многих
элементов C; arithmetic intensity растёт примерно с tile size, пока capacity и
occupancy не становятся новым ограничением. Выбор `BLOCK_M`, `BLOCK_N`,
`BLOCK_K`, `num_warps` и stages — совместный: увеличить один размер без учёта
register/shared-memory pressure нельзя.

После GEMM можно применить GeLU/ReLU до store и не материализовать отдельный
activation tensor. Но fusion имеет предел. Если epilogue увеличит число живых
registers и уменьшит resident blocks, сэкономленные bytes могут не окупить
потерю occupancy.

## Autotuning — поиск по корректным кандидатам

Triton позволяет связать несколько configurations с key shapes и измерить их.
Autotuner не освобождает от design:

- search space задаёт человек;
- compile и tuning cost исключаются из steady-state benchmark;
- winner одной shape не переносится автоматически на соседние;
- все candidates должны соблюдать correctness и resource limits;
- cache autotuning results должен быть привязан к GPU, software version и key.

Полезная матрица shapes включает производственные размеры и границы tiles.
Например, вместе с 4096 проверяют 4095 и 4097. Иначе tuner выбирает
конфигурацию, превосходную только на идеально кратном benchmark.

## Как читать профиль Triton kernel

Профиль отвечает на последовательность вопросов:

1. **Запустился ли наш kernel?** Имя и число launches должны соответствовать
   grid; fallback или graph break меняют эксперимент.
2. **Есть ли CPU gaps?** Если GPU ждёт dispatch, работа ниже уровня kernel ещё
   не является главным ограничением.
3. **Каков achieved bandwidth/FLOP/s?** Сопоставить с roofline для нужного
   memory level и dtype.
4. **Хватает ли работы?** Grid, waves, tail tiles и active warps объясняют
   underfill.
5. **Не заплатили ли registers/shared memory?** Spills и низкая occupancy могут
   отменить reuse.
6. **Какие shapes регрессировали?** Сравнить не только среднее, но и каждую
   заранее объявленную форму.

`torch.compile` и Triton не являются взаимоисключающими. Compiler может
сгенерировать Triton kernels для captured graph; вручную написанный kernel
нужен, когда общая система не выводит желаемый layout/schedule или когда его
контракт стоит сделать явным. Библиотечный kernel остаётся обязательным
baseline: собственная реализация оправдана измеренным выигрышем или необходимой
семантикой, а не самим фактом авторства.

## Переход к FlashAttention

Softmax показал, как объединить reductions одной строки; tiled GEMM — как
переиспользовать blocks матриц. Attention объединяет обе задачи. Полная матрица
$QK^\top$ не помещается on chip, а нормировка одной query row зависит от всех
key tiles. Online softmax позволяет обновлять максимум, знаменатель и weighted
value по мере обхода tiles. Следующая глава разбирает точный forward и backward:
[[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/56 FlashAttention|FlashAttention]].

## Практика и первоисточники

- [Stanford CS336 Lecture 5, pinned `8b59b507`, pp. 30–54](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_05.pdf) — fusion, recomputation, tiling, wave quantization and FlashAttention transition.
- [Stanford CS336 Lecture 6, pinned `8b59b507`](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_06.py) — executable GeLU comparison, Triton elementwise kernel, softmax, row reduction, and tiled matmul.
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/) — current language/API examples; verify against the installed Triton version.
- [Tillet, Kung, Cox, Triton](https://openai.com/index/triton/) and [Triton repository](https://github.com/triton-lang/triton) — programming model and implementation.
- [[02 Areas/ML & DL/06 Практика/21 Профилировать и ускорить Transformer kernel|Capstone: профилировать и ускорить Transformer kernel]].

← [[02 Areas/ML & DL/00 Учебник/10 ML Systems/07 Profiling ML-нагрузки|Profiling ML-нагрузки]] ·
[[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/56 FlashAttention|FlashAttention]] →
