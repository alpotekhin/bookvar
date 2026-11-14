---
title: GPU kernels и Triton — от программы к измерению
type: textbook-chapter
status: draft
last_updated: 2026-11-04
last_verified: 2026-11-04
source_language: mixed
reference_terms:
  - anchor: kernel-contract
    label: "Контракт GPU-ядра"
    label_en: "GPU kernel contracts"
  - anchor: triton-program-grid
    label: "Программа Triton, grid и маски"
    label_en: "Triton programs, grids and masks"
  - anchor: kernel-fusion-hbm
    label: "Fusion и промежуточный трафик HBM"
    label_en: "Fusion and intermediate HBM traffic"
  - anchor: triton-row-softmax
    label: "Softmax строки и маска −∞"
    label_en: "Row softmax and −∞ masks"
  - anchor: chunked-reductions
    label: "Редукции по частям и online softmax"
    label_en: "Chunked reductions and online softmax"
  - anchor: local-reduction-tree-work-depth-synchronization
    label: "Дерево редукции: работа и глубина"
    label_en: "Reduction trees: work and depth"
  - anchor: tiled-matmul-reuse
    label: "Блочный GEMM и повторное использование"
    label_en: "Tiled GEMM and data reuse"
  - anchor: indexed-gather-scatter-and-gradients
    label: "Gather, scatter-add и градиенты"
    label_en: "Gather, scatter-add and gradients"
  - anchor: kernel-numerics-backward
    label: "Численная корректность и backward"
    label_en: "Numerical correctness and backward"
  - anchor: fused-linear-cross-entropy
    label: "Linear cross-entropy без полной матрицы логитов"
    label_en: "Linear cross-entropy without full logits"
  - anchor: triton-autotuning
    label: "Автоподбор конфигурации Triton"
    label_en: "Triton configuration autotuning"
  - anchor: kernel-measurement
    label: "Границы измерения GPU-ядра"
    label_en: "GPU kernel measurement boundaries"
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
primary_sources:
  - https://github.com/rusty1s/pytorch_scatter/blob/f514c10f920b5aeed2eb162092f0ad20d3edee52/torch_scatter/scatter.py
  - https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/_torch_docs.py
  - https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/_tensor_docs.py
  - https://github.com/onnx/onnx/blob/3646b038c23824207d064d8625a908b1bf8710f4/onnx/defs/doc_strings.cc
  - https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
  - https://github.com/NVIDIA/cuda-samples/blob/5443602d89ed99aede2e4b7bf329daddeadb320e/cpp/2_Concepts_and_Techniques/reduction/reduction_kernel.cu
  - https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_06.py
  - https://github.com/triton-lang/triton
  - https://triton-lang.org/main/getting-started/tutorials/
  - https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week06_dl_arithmetic/seminar/practice.ipynb
  - https://openai.com/index/triton/
  - https://arxiv.org/html/1805.02867v2
  - https://arxiv.org/html/2410.10989v2
  - https://github.com/linkedin/Liger-Kernel/blob/6ce504c3286178ba6537624252f0a1ee88423b53/src/liger_kernel/ops/fused_linear_cross_entropy.py
---

<a id="cs336-systems-triton"></a>

# GPU kernels и Triton — от программы к измерению

Профиль может показать, что несколько умножений и одна активация занимают
много времени. Арифметика здесь не обязательно трудна для GPU:
дорогим оказывается способ её организации. Каждая отдельная операция читает
тензор, создаёт промежуточный результат и передаёт его следующей операции.
Для большого матричного умножения стоимость одного запуска обычно мала по
сравнению с вычислениями; для цепочки коротких операций заметными становятся
и запуск, и повторные обращения к памяти.

В [[02 Areas/ML & DL/00 Учебник/10 ML Systems/07 Profiling ML-нагрузки|предыдущей главе]]
мы искали причину задержки на временной шкале. Следующий шаг — изменить
реализацию найденной операции, сохранив её смысл. Для этого нужны два описания
одной задачи. Математическое задаёт результат и допустимую ошибку. Аппаратное
объясняет, какие данные обрабатываются вместе, где находятся промежуточные
значения и какие части работы можно выполнять параллельно. Ядро GPU
(*kernel*) связывает эти описания: это функция, один запуск которой исполняет
множество параллельных экземпляров работы на устройстве.

Triton позволяет описывать операции над блоками данных, а не отдельные
действия каждого аппаратного потока. Компилятор распределяет эти операции
между потоками, варпами и инструкциями. Но он не выбирает за программиста
математический алгоритм, границы блоков, численный контракт и критерий успеха.
Поэтому начнём с простого сложения векторов, где видна вся адресация, затем
перейдём к объединению операций, редукциям и матричному умножению. Во всех
случаях отдельно проверим адресацию, промежуточные результаты и объём
переносимых данных.

<a id="контракт-собственного-kernel"></a>

<a id="kernel-contract"></a>

## Контракт собственного ядра

Пусть требуется вычислить $z_i=x_i+y_i$. Формула ещё не определяет полную
операцию. Могут ли входы находиться на разных GPU? Разрешён ли транспонированный
тензор? Что делать с пустым массивом? Нужно ли поддерживать градиент? Если
обёртка принимает произвольный тензор, а ядро считает его элементы смежными,
то даже идеальная арифметика даёт неверный ответ.

Для учебного сложения ниже контракт намеренно узок: два одномерных смежных
тензора FP32 одинаковой формы на одном CUDA-устройстве; результат выделяется
отдельно; автоматическое дифференцирование не поддерживается. Пустой вход
обрабатывает Python-обёртка без запуска ядра. При расширении контракта придётся
явно добавить шаги по памяти (`strides`), другие типы данных или обратный
проход, а не просто убрать проверки.

В более общей операции заранее фиксируют формы, размещение, типы входа,
аккумулятора и результата, допустимость пересечения областей памяти, поведение
на NaN/Inf и математическое приближение. Эталон и допуски `atol`/`rtol`
выбирают до подбора быстрых параметров. Иначе можно незаметно ускорить уже
другую задачу — например, заменить точную GeLU её приближением или забыть
часть строки при редукции.

<a id="что-именно-задаёт-triton-program"></a>

<a id="triton-program-grid"></a>

## Как программа Triton распределяет работу

В обычном CUDA-ядре основная точка зрения — отдельный поток: он получает свои
индексы и выполняет последовательность инструкций. В Triton удобно мыслить
экземпляром программы (*program instance*), который обрабатывает логический
блок элементов. Сетка запуска (`grid`) задаёт количество экземпляров;
`tl.program_id(axis)` сообщает координату текущего экземпляра.

Если длина массива равна 1000, а размер блока — 256, нужны четыре экземпляра.
Они работают с индексами 0–255, 256–511, 512–767 и 768–1023. Последний диапазон
содержит только 232 настоящих элемента. Лишние 24 позиции можно оставить в
логическом блоке, но обращаться по их адресам нельзя. Для этого используется
маска: тензор логических значений той же формы, что и адреса.

Здесь важно не отождествлять блок из 256 элементов с 256 потоками. Несколько
элементов может обрабатывать один поток; конкретное отображение зависит от
компилятора, операции, типа чисел и `num_warps`. Аналогично четыре экземпляра
программы — не четыре запуска с CPU. Вызов `kernel[(4,)](...)` ставит в очередь
одно ядро с сеткой из четырёх экземпляров. Значения, загруженные через
`tl.load`, также не обязаны сначала попадать в разделяемую память: компилятор
выбирает размещение в регистрах и необходимые обмены для данной операции.

### Полный первый пример: сложение векторов

В следующем примере есть обе части реализации: функция, исполняемая на GPU,
и Python-обёртка, проверяющая контракт и задающая запуск. Это учебная версия
шаблона из [официального Vector Addition tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
(Philippe Tillet, OpenAI и участники Triton;
[MIT](https://github.com/triton-lang/triton/blob/1b24a39939875de5333e3990bec3ddcbfc64eb80/LICENSE)).
Она требует совместимых PyTorch, Triton и CUDA GPU; ниже не заявляется
проверка исполнения на таком устройстве.

```python
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, z_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < n
    x = tl.load(x_ptr + offsets, mask=valid, other=0.0)
    y = tl.load(y_ptr + offsets, mask=valid, other=0.0)
    tl.store(z_ptr + offsets, x + y, mask=valid)


def add_vectors(x, y):
    assert x.is_cuda and y.is_cuda and x.device == y.device
    assert x.ndim == y.ndim == 1 and x.shape == y.shape
    assert x.dtype == y.dtype == torch.float32
    assert x.is_contiguous() and y.is_contiguous()
    assert not x.requires_grad and not y.requires_grad
    z = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return z
    block = 256
    with torch.cuda.device(x.device):
        add_kernel[(triton.cdiv(n, block),)](
            x, y, z, n, BLOCK=block, num_warps=4
        )
    return z
```

`tl.arange(0, BLOCK)` создаёт не Python-цикл, а вектор индексов, над которым
дальнейшие операции выполняются поэлементно. `BLOCK: tl.constexpr` означает,
что размер логического блока известен при компиляции. Для этой формы
`tl.arange` он задаётся степенью двойки; длина входа `n` такой быть не обязана.
Операция `x_ptr + offsets` строит вектор адресов, причём смещения считаются
в элементах, не в байтах. `tl.load` получает значения по разрешённым адресам,
сложение образует блок результата, а `tl.store` записывает только настоящие
элементы.

У маски две разные роли. Во-первых, она исключает недопустимые обращения к
памяти. Во-вторых, параметр `other` определяет значение в исключённых позициях
самого логического блока. В сложении это значение не влияет на сохраняемый
результат; при редукции влияет, поэтому оставить его неопределённым уже нельзя.
Не заменяют маску выражением `tl.where(valid, tl.load(ptr), 0)`: выбор значения
не защищает уже выполненное чтение. Семантика исключённых позиций описана в
[документации `tl.load`](https://triton-lang.org/main/python-api/generated/triton.language.load.html).

Первые проверки должны включать длины 0, 1, 255, 256, 257 и 1000. Случай 256
не проверяет хвост вообще; 257 проверяет почти полностью пустой второй блок.
Такую проверку на GPU можно записать после определения функций:

```python
for n in (0, 1, 255, 256, 257, 1000):
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn_like(x)
    torch.testing.assert_close(add_vectors(x, y), x + y)
```

Последовательные смещения помогают соседним потокам обращаться к соседним
адресам и объединять запросы к памяти (*coalescing*). При большом шаге между
элементами тот же объём полезных данных может потребовать больше транзакций.
Поэтому для несмежного входа недостаточно правильно умножить индексы на stride:
нужно ещё понять цену получившегося доступа. Копирование через `contiguous()`
может восстановить удобное размещение, но его время и дополнительная память
входят в стоимость всей операции.

Сложение само по себе редко оправдывает ручное ядро: библиотечная реализация
уже решает очень простую задачу. Для FP32 минимальный логический обмен — два
чтения и одна запись, то есть $12n$ байт на $n$ сложений. Здесь почти нечего
повторно использовать. Польза примера в другом: теперь можно заменить одно
сложение последовательностью вычислений, не меняя способ загрузки и записи.

<a id="kernel-fusion-hbm"></a>

## От одной формулы PyTorch к нескольким проходам по HBM

Если после сложения нужна нелинейная функция, не обязательно сначала записывать
сумму в глобальную память. Каждый экземпляр программы уже держит нужный блок
значений и может продолжить вычисление. Такое объединение операций называют
*fusion*. Оно сохраняет математическую задачу, но устраняет промежуточные
тензоры и часть запусков. Особенно наглядна активация GeLU, приближение которой
через гиперболический тангенс имеет вид

$$
\begin{gathered}
a(x)=\sqrt{\frac{2}{\pi}}(x+0.044715x^3),\\
\operatorname{GELU}_{\tanh}(x)=\frac{x}{2}\left(1+\tanh a(x)\right).
\end{gathered}
$$

Наивная PyTorch-запись удобна как спецификация. Но если каждый оператор
запускается отдельно, степени, умножения, сложения и `tanh` читают вход или
предыдущий промежуточный результат из глобальной памяти и записывают следующий.
Логический объём этих обращений велик по сравнению с одной загрузкой входа и
одной записью выхода. Не каждое обращение обязательно доходит до HBM — часть
обслуживается кэшами, — поэтому это модель трафика, а не измерение аппаратных
счётчиков. Fusion устраняет саму необходимость материализовать промежуточные
тензоры вне ядра.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/l05-p31.png]]

*Tatsunori Hashimoto и команда Stanford CS336, Lecture 5, p. 31:
слева результаты каждого этапа возвращаются в память, справа цепочка
заканчивается одной записью. Формы обозначают значения на разных стадиях,
а не разные форматы чисел. [Закреплённый PDF](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_05.pdf#page=31).*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/l05-p32.png]]

*На следующем слайде эта схема получает конкретный вычислительный граф:
$\sin^2(x)+\cos^2(x)$ в наивной записи запускает пять ядер CUDA и
материализует промежуточные тензоры. Источник: Tatsunori Hashimoto и команда
Stanford CS336, Lecture 5, p. 32,
[закреплённый PDF](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_05.pdf#page=32); полная страница, без обрезки.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/l05-p33.png]]

*Tatsunori Hashimoto и команда Stanford CS336, Lecture 5, p. 33:
компилятор объединяет пять поэлементных операций
предыдущего графа в одно ядро CUDA. Арифметика та же, но исчезают
записи промежуточных результатов и повторные чтения. Три слайда показывают
общую схему переноса данных, конкретный граф FX и результат объединения.
[Закреплённый PDF](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_05.pdf#page=33).*

Для GeLU Lecture 6 сравнивает три реализации:

- исходную композицию операций PyTorch;
- встроенную `torch.nn.functional.gelu(..., approximate="tanh")`;
- `torch.compile` исходной композиции.

Сначала сравнивают результаты, затем измеряют время после прогрева и изучают
профиль. Встроенная функция может уже вызывать специализированное ядро.
Компилятор способен объединить граф, но первоначальная компиляция и проверки
условий применимости имеют свою стоимость. Уменьшение числа ядер объясняет
возможное ускорение, но не доказывает его: нужны корректный результат и меньшее
измеренное время на заданном наборе форм тензоров.

В сохранённой трассе курса эти варианты дают следующую картину. Здесь
`dim=16384` означает матрицу $16384\times16384$, а не вектор из 16384 элементов.

| Реализация | Наблюдение в трассе |
|---|---|
| Наивная композиция PyTorch | 9 запусков CUDA: промежуточные результаты передаются между ядрами |
| Встроенная GeLU, `approximate="tanh"` | 1 запуск CUDA: активация вычисляется библиотечным ядром |
| Та же композиция после `torch.compile` | 1 запуск CUDA: захваченная цепочка объединена в сгенерированное ядро |

Это наблюдение Percy Liang и команды
[Stanford CS336, Lecture 6, шаги 245, 248 и 251 сохранённой трассы](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/var/traces/lecture_06.json),
а не новый запуск Bookvar. Оно объясняет механизм выигрыша, но не устанавливает
время работы на другой машине. Без `approximate="tanh"` сравнивались бы разные
функции: [встроенная GeLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.gelu.html)
по умолчанию использует другую формулу.

Fusion нельзя понимать как произвольное алгебраическое упрощение. Равенство
$\sin^2x+\cos^2x=1$ не разрешает компилятору безусловно заменить всю программу
константой при заданной семантике чисел с плавающей точкой. Подобным образом
математически верная запись $\tanh a=(e^{2a}-1)/(e^{2a}+1)$ может получить
`inf/inf` на большом положительном аргументе. Сохранить формулу на бумаге
недостаточно: нужны устойчивые численные операции на всём заявленном диапазоне.

В трансформере тот же принцип применяется к SwiGLU:
$\operatorname{SiLU}(g)\odot u=g\sigma(g)\odot u$.
Вместо промежуточного тензора SiLU программа загружает блоки $g$ и $u$,
вычисляет нелинейность и сразу записывает произведение. Такое объединение
разбирается в [семинаре EDLS, неделя 6](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week06_dl_arithmetic/seminar/practice.ipynb).
Сами предшествующие проекции $XW_g$ и $XW_u$ при этом могут оставаться отдельными
GEMM: объединение поэлементного участка не означает, что весь FFN стал одним
ядром. Именно граница доступных данных определяет разумный масштаб fusion.

<a id="первый-полный-пример-softmax-одной-строки"></a>

<a id="triton-row-softmax"></a>

## Разбор алгоритма: softmax одной строки

Для $X\in\mathbb{R}^{M\times N}$ softmax по строкам равен

$$
\begin{aligned}
m_i&=\max_k x_{ik},\\
y_{ij}&=\frac{\exp(x_{ij}-m_i)}
{\sum_{k=1}^{N}\exp(x_{ik}-m_i)}.
\end{aligned}
$$

В отличие от GeLU, значение одного выхода теперь зависит от всей строки.
Разделить строку между независимыми программами и нормировать каждый кусок
отдельно нельзя: получится несколько распределений вместо одного. Поэтому
естественное начальное разбиение — один экземпляр программы на одну строку.
Строки независимы, а редукции внутри строки компилятор организует между
участвующими потоками.

Вычитание максимума не меняет результат в точной арифметике, но делает все
аргументы экспоненты неположительными. Для конечных входов хотя бы один
аргумент равен нулю, поэтому знаменатель положителен и экспонента не
переполняется. Например, строка $(5,5,5)$ даёт $(1/3,1/3,1/3)$, а прибавление
1000 ко всем её элементам не меняет ответ. Непосредственное вычисление
$e^{1005}$ в FP32 при этом уже невозможно.

Наивная композиция `max → subtract → exp → sum → divide` в модели отдельных
операций читает $5MN+2M$ и записывает $3MN+2M$ элементов. Если строка помещается
в ресурсы одного экземпляра программы, достаточно прочитать $MN$ и записать
$MN$ элементов, удерживая промежуточные значения на чипе. Отношение примерно
четыре к одному относится к логическому объёму обмена при больших $N$, а не к
гарантированному ускорению: редукции, экспоненты и размещение блоков тоже стоят
времени. Подсчёт разобран в [официальном Fused Softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html).

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/triton-softmax.png]]

*Percy Liang и команда Stanford CS336, Lecture 6,
[step 321, исходный рисунок](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/images/triton-softmax.png):
строка 1 проходит загрузку, вычитание максимума, экспоненту с суммированием и
нормировку внутри одного экземпляра программы. Остальные строки обрабатываются
другими экземплярами. Маски и устройство редукции на этой схеме не показаны;
это логическая последовательность, не аппаратная временная шкала.*

Ниже — простой вариант «одна программа на строку», продолжающий импорты из
примера сложения. Вход — двумерный FP32-тензор с единичным шагом по столбцам;
расстояние между началами строк передаётся отдельно. Значения должны быть
конечными, градиенты не поддерживаются. Ограничение $1\le N\le4096$ выбрано
для учебной обёртки, а не является универсальным пределом GPU или Triton.

```python
@triton.jit
def row_softmax_kernel(x_ptr, y_ptr, x_stride, y_stride,
                       n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < n_cols
    x = tl.load(x_ptr + row * x_stride + cols,
                mask=valid, other=-float("inf")).to(tl.float32)
    numerator = tl.exp(x - tl.max(x, axis=0))
    y = numerator / tl.sum(numerator, axis=0)
    tl.store(y_ptr + row * y_stride + cols, y, mask=valid)


def row_softmax(x):
    assert x.is_cuda and x.ndim == 2 and x.dtype == torch.float32
    assert x.stride(1) == 1 and not x.requires_grad
    rows, cols = x.shape
    assert 1 <= cols <= 4096
    y = torch.empty((rows, cols), device=x.device, dtype=x.dtype)
    if rows == 0:
        return y
    with torch.cuda.device(x.device):
        row_softmax_kernel[(rows,)](
            x, y, x.stride(0), y.stride(0), cols,
            BLOCK=triton.next_power_of_2(cols), num_warps=4
        )
    return y
```

При $N=5$ программа строит блок из восьми позиций. Три лишние позиции получают
$-\infty$: максимум остаётся максимумом настоящей строки, а после экспоненты
их вклад в сумму равен нулю. Подставить нули при загрузке неверно. Даже если
они случайно не изменят максимум, экспоненты заполнителей добавятся в
знаменатель и сумма пяти записанных вероятностей окажется меньше единицы.
Маска записи этого не исправит: ошибка уже возникла в общей нормировке.

Возьмём пять нулевых логитов. Без padding каждое значение softmax равно
$1/5$. В блоке из восьми позиций правильные заполнители $-\infty$
после экспоненты дают три нуля: знаменатель по-прежнему равен 5.
При заполнении нулями все восемь экспонент равны единице. В память
запишутся только пять результатов, но каждый будет равен $1/8$,
а их сумма — $5/8=0{,}625$. Ограничение записи не устраняет вклад
ложных элементов в редукцию.

Следующий самостоятельный CPU-пример проверяет именно этот расчёт;
он не исполняет Triton-ядро:

```python
import torch

logits = torch.zeros(5, dtype=torch.float64)
correct_block = torch.full((8,), -torch.inf, dtype=logits.dtype)
correct_block[:5] = logits
wrong_block = torch.zeros(8, dtype=logits.dtype)
wrong_block[:5] = logits

correct = correct_block.softmax(0)[:5]
wrong = wrong_block.softmax(0)[:5]
torch.testing.assert_close(correct, torch.full_like(logits, 1 / 5))
torch.testing.assert_close(wrong, torch.full_like(logits, 1 / 8))
assert correct.sum().item() == 1.0
assert wrong.sum().item() == 0.625
```

Тот же код объясняет значение stride. Адрес элемента $(i,j)$ равен
`base + i * stride_row + j * stride_col`. В обёртке `stride_col == 1`, поэтому
последнее умножение отсутствует. Срез через одну строку можно обработать,
передав увеличенный `stride_row`; транспонирование обычно нарушит требование
единичного шага по столбцам и будет отклонено. Автоматическое скрытое
копирование здесь не выполняется.

Проверяют случайные матрицы, равные логиты, отрицательные значения, один
доминирующий логит и размеры 255, 256, 257. Для каждого входа сравнивают
полный результат с `torch.softmax(x, dim=-1)`, а дополнительно — неотрицательность
и сумму строки, близкую к единице. Последние два свойства полезны, но не
заменяют сравнение: перестановка вероятностей внутри строки тоже их сохраняет.
`tl.exp` использует приближённую экспоненту, поэтому допуск выбирают с учётом
диапазона входов и требуемой точности, а не требуют побитового совпадения.

У строк из одних $-\infty$, строк с $+\infty$ или NaN должны быть отдельно
определены правила. Вычитание максимума в таких случаях может само дать NaN.
В частности, полностью замаскированная строка attention требует явного
решения вызывающей операции; формула стабильного softmax сама по себе не
задаёт распределение для отсутствующих допустимых ключей. Учебная обёртка
такие входы не обещает поддерживать.

Ближайшая степень двойки тоже имеет стоимость. При переходе от 4096 к 4097
столбцам логический блок вырос бы до 8192, хотя полезных данных почти столько
же. Возрастают объём редукции и давление на регистры. Текущий официальный
tutorial идёт дальше учебного варианта: ограничивает число одновременно
запущенных программ и даёт каждой обходить несколько строк. Это другая
организация сетки, а не изменение определения softmax. Простой вариант выше
следует [разбору CS336 Lecture 6](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_06.py),
не копирует весь механизм настройки из обновляемой документации.

<a id="reduction-которая-не-помещается-в-один-tile"></a>

<a id="chunked-reductions"></a>

## Редукция, которая не помещается в один блок

Пусть в строке 4096 элементов, а программа обрабатывает по 1024. Для суммы
не требуется держать все 4096 значений одновременно: достаточно накапливать
частичные суммы. Один экземпляр программы может хранить вектор из 1024
аккумуляторов, загрузить очередную порцию, прибавить её поэлементно и в конце
выполнить `tl.sum`. Это по-прежнему одна программа на строку, но уже несколько
итераций внутри неё. Последняя неполная порция заполняется нулями — нейтральным
элементом сложения, в отличие от $-\infty$ для максимума.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/triton-row-sum.png]]

*Percy Liang и команда Stanford CS336, Lecture 6,
[step 353, исходный рисунок](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/images/triton-row-sum.png):
те же позиции вектора последовательно получают
несколько блоков, а аккумулятор сохраняется между итерациями. Рисунок показывает, что блок и
полная строка — разные масштабы. Обозначения `t0`–`t3` здесь удобно читать как
позиции учебного блока; это не утверждение, что Triton всегда выделяет по
одному аппаратному потоку на каждый элемент.*

На рисунке первые две порции дают аккумуляторы $(8,10,6,7)$, последняя
добавляет $(5,3,0,0)$, а финальная сумма равна 39. Данные не исчезают при
переходе к следующей порции: от них остаётся достаточное состояние для
продолжения вычисления. При очень длинной строке и малом числе строк одной
программы на строку может не хватить для загрузки GPU. Тогда возможна
двухэтапная редукция: несколько программ записывают частичные суммы, а другое
ядро объединяет их. Дополнительный обмен может окупаться большей параллельностью.

<!-- BEGIN REVIEW: local reduction tree 2026-10-27 -->

<a id="local-reduction-tree-work-depth-synchronization"></a>

### A reduction tree: work, depth and synchronization

The row-sum example ends with `tl.sum(accumulator)`. When the accumulator is
distributed across several lanes, the reduction must combine their partials.
Consider eight values, each initially
available at one of the positions labelled $p_0$ through $p_7$ in the animation.
These labels identify abstract processing positions, not a prescribed mapping
of a Triton block to CUDA threads.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/reduction-primitives-2026-10-27/binomial-tree.gif|Binomial-tree reduction: eight inputs become four partial sums, then two, then 36]]

*Flogr, [Binomial tree](https://commons.wikimedia.org/w/index.php?title=File:Binomial_tree.gif&oldid=824382071),
2018, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Complete original GIF, 901 × 512, four frames; no crop, redraw or relabelling.
Read upwards: a blue edge carries a value into the next partial sum. The final
frame retains the intermediate values, so the tree can also be read as a
static calculation. The animation illustrates dependencies, not elapsed GPU time.*

The input is $(2,3,5,1,7,6,8,4)$. In the first round, four independent additions
produce $2+3=5$, $5+1=6$, $7+6=13$ and $8+4=12$. A second round combines adjacent
partials, giving $5+6=11$ and $13+12=25$. Only then can the final addition produce
$11+25=36$.

| Round | New partial sums | Independent additions |
| --- | --- | ---: |
| 1 | 5, 6, 13, 12 | 4 |
| 2 | 11, 25 | 2 |
| 3 | 36 | 1 |

There are seven additions altogether, just as in a sequential sum of eight
inputs. What changes is their dependency structure. A running sequential
accumulator needs the result of one addition before starting the next, making
a chain of seven. The tree's longest chain contains three additions. With
enough participants, additions on the same row can proceed independently.

For $N=2^k$ inputs, the addition count and dependency depth are

$$
\begin{aligned}
W&=N/2+N/4+\cdots+1=N-1,\\
D&=\log_2 N.
\end{aligned}
$$

$W$ counts work; $D$ counts dependent addition stages. If at most $P$ additions
can run simultaneously and each takes one idealized time unit, every schedule
must take at least $\max(\lceil W/P\rceil,D)$ units. For this eight-input tree,
four workers can finish in three rounds. With two workers, its first row needs
two rounds, followed by one round for each remaining row: four in total.
One worker still needs seven. None of these counts includes loads, instruction
dispatch or synchronization latency; they are not GPU speedup measurements.

Mark Harris's [Optimizing Parallel Reduction in CUDA, slides 3–5 and 29–31](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
develops the distinction between work and parallel depth, then combines local
sequential accumulation with a reduction tree. Its benchmark numbers and
warp-synchronous shared-memory code describe the historical G80 example;
they are not current hardware contracts.

The tree can be traced on CPU without pretending to execute concurrent threads:

```python
def reduction_trace(values):
    current = list(values)
    if not current:
        raise ValueError("this trace requires at least one input")
    levels = [current.copy()]
    additions = 0
    while len(current) > 1:
        next_level = []
        for i in range(0, len(current) - 1, 2):
            next_level.append(current[i] + current[i + 1])
            additions += 1
        if len(current) % 2:
            next_level.append(current[-1])
        current = next_level
        levels.append(current.copy())
    return levels, additions

levels, additions = reduction_trace([2, 3, 5, 1, 7, 6, 8, 4])
assert levels == [[2, 3, 5, 1, 7, 6, 8, 4],
                  [5, 6, 13, 12], [11, 25], [36]]
assert additions == 7 and len(levels) - 1 == 3

# An odd element is carried forward, not discarded or added twice.
assert reduction_trace([3, 1, 4, 1, 5]) == (
    [[3, 1, 4, 1, 5], [4, 5, 5], [9, 5], [14]], 4
)
```

Carrying an unpaired element forward avoids unnecessary additions. A GPU
implementation may instead pad to a convenient power of two with the identity
element. For five inputs, a full eight-leaf sum tree has seven addition nodes
rather than the unpadded four; the three extra inputs are zeros. A maximum
requires a different identity, normally $-\infty$. Neither padding nor a mask
allows a required input to disappear from the reduction.

### Shared memory, registers and the boundary of a barrier

Each tree stage consumes results from the preceding stage. In a shared-memory
implementation, a participant writes its partial before another participant
can use it. For the first four values above, $p_0$ must read the completed
$5+1=6$, not the old input 5. Reading that old value after computing $2+3=5$
would produce 10 instead of 11. A block-wide synchronization point orders
the shared-memory stages; it must be reached by every thread in the block,
including those that did not perform an addition in that stage.

The pinned [CUDA Samples `reduce2`](https://github.com/NVIDIA/cuda-samples/blob/5443602d89ed99aede2e4b7bf329daddeadb320e/cpp/2_Concepts_and_Techniques/reduction/reduction_kernel.cu)
uses `cg::sync(cta)` after loading shared memory and after every stage.
It pairs `sdata[tid]` with `sdata[tid + s]`, starting with half the block
size and halving $s$. This pairs different leaves from the adjacent-pair
animation but computes the same exact integer sum. Its power-of-two block
assumption is part of the algorithm.

Within a warp, a reduction can exchange register values directly. The following
generic helper is copied unchanged from that source; the source also contains
an architecture-specific integer specialization not reproduced here.

```cpp
template <class T> __device__ __forceinline__ T warpReduceSum(unsigned int mask, T mySum)
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        mySum += __shfl_down_sync(mask, mySum, offset);
    }
    return mySum;
}
```

*NVIDIA Corporation, CUDA Samples, revision `5443602`,
[original helper](https://github.com/NVIDIA/cuda-samples/blob/5443602d89ed99aede2e4b7bf329daddeadb320e/cpp/2_Concepts_and_Techniques/reduction/reduction_kernel.cu#L78-L85),
[full BSD-3-Clause copyright and conditions](/licenses/nvidia-cuda-samples-5443602-BSD-3-Clause.txt).
This is a device-function excerpt, not a complete kernel or an executed CUDA test.*

For FP32 values in a full NVIDIA warp of 32 participating lanes, use the same
`0xffffffff` mask in every lane. The offsets are 16, 8, 4, 2 and 1. When
`lane_id + offset < 32`, the shuffle reads the current value from that
higher-ID lane. If the source would lie outside the warp, it returns the
caller's own value instead. Lane 0 finishes with the sum of all 32 inputs;
the helper does not promise that every lane receives that sum.

All 32 lanes execute the five stages, giving 160 lane-level additions,
including work whose result is not needed by lane 0. The minimal binary
tree counted earlier would need 31 additions. The source loop has five
warp-wide stages, rather than a chain of 160 scalar additions. The exact
machine instruction count depends on the value type, compiler and target.
Useful arithmetic, work performed by lanes and machine instructions are
different counts.

The full-mask example requires all 32 lanes to participate. Merely substituting
a sparse mask does not make this exact offset schedule correct for an arbitrary
subset: a requested source lane may not participate. The
[CUDA Programming Guide, Warp `__sync` Intrinsic Constraints](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-sync-intrinsic-constraints)
specifies the participation rules. Register shuffles do not order unrelated
shared-memory accesses. In particular, `volatile` alone is not a substitute
for the synchronization needed between shared-memory stages.

A block barrier does not synchronize separate blocks. One ordinary multi-block
reduction can write a partial sum per block, then launch a second kernel to
reduce those partials. The dependency must be established, for example by
issuing both kernels to the same stream; a launch is not a device-wide CPU
wait. More specialized cooperative or atomic algorithms have their own
requirements. Spinning while waiting for every block is unsafe if the waiting
blocks occupy resources needed to run the blocks they are waiting for.

Local sequential work can reduce the size of the tree. In the preceding
row-sum figure, four accumulators finish with $(13,13,6,7)$. Combining these
partials needs only three additions: $(26,13)$, then 39. The full row still
had to be read. Increasing work per participant trades a shorter collective
stage for less parallel work, more live state and a longer local chain.
That trade-off explains why neither one participant per element nor the
largest possible block is an unconditional performance choice.

Changing the tree also changes floating-point rounding. The operation counts
above are exact integer calculations, not a promise of bitwise agreement
between different floating-point reductions. The numerical checks below must
still cover the implemented order and accumulator precision.

### Maximum and normalization state across chunks

<!-- END REVIEW: local reduction tree 2026-10-27 -->

Для softmax достаточное состояние сложнее. Пусть уже обработаны элементы с
максимумом $m$ и нормировочной суммой $\ell=\sum e^{x-m}$. Для новой порции $B$
обновление имеет вид

$$
\begin{aligned}
m'&=\max\left(m,\max_{x\in B}x\right),\\
\ell'&=e^{m-m'}\ell+\sum_{x\in B}e^{x-m'}.
\end{aligned}
$$

Первое слагаемое переводит накопленную сумму к новому масштабу. Например,
если прежний максимум был 5, а новый равен 8, старый вклад нужно умножить на
$e^{-3}$: складывать суммы, нормированные разными максимумами, нельзя.

Для строки $(4,5,7,8)$ обработаем сначала $(4,5)$, затем $(7,8)$.
После первой порции максимум равен 5, а сумма —
$\ell=1+e^{-1}\approx1{,}367879$.
Во второй порции максимум становится равным 8. В новом масштабе
вклад уже просмотренных элементов составляет
$e^{-3}\ell=e^{-4}+e^{-3}\approx0{,}068103$.
Новая порция добавляет $e^{-1}+1\approx1{,}367879$;
общий знаменатель теперь равен $\ell'\approx1{,}435982$.

Четыре числителя в том же масштабе —
$(e^{-4},e^{-3},e^{-1},1)$. Деление на $\ell'$ даёт примерно
$$
\begin{aligned}
(&0{,}012755,\;0{,}034671,\\
 &0{,}256187,\;0{,}696387).
\end{aligned}
$$
Если забыть множитель $e^{-3}$, знаменатель получится
$2(1+e^{-1})\approx2{,}735759$ и сумма этих вероятностей не будет единицей.
Это тот же пересчёт масштаба, что в алгоритме 3 и его блочном обобщении
у Milakov и Gimelshein,
[Online normalizer calculation for softmax, §3–3.1](https://arxiv.org/html/1805.02867v2#S3).

После обхода известен знаменатель всей строки, но ещё не все выходные
вероятности. Если логиты не сохранены на чипе, их придётся прочитать ещё раз
или хранить промежуточные данные. В FlashAttention ситуация выгоднее:
нужна не сама матрица вероятностей, а их взвешенная сумма с $V$, которую
можно обновлять вместе с $m$ и $\ell$. Именно это различие делает потоковый
softmax полезным строительным блоком внимания.

<a id="tiled-gemm-reuse-вместо-идеализированной-загрузки-всей-матрицы"></a>

<a id="tiled-matmul-reuse"></a>

## Блочное матричное умножение и повторное использование данных

Для $C=AB$, $A\in\mathbb{R}^{M\times K}$,
$B\in\mathbb{R}^{K\times N}$, наивное вычисление каждого $C_{mn}$ независимо
читает строку $A$ и столбец $B$ заново. Соседний $C_{m,n+1}$ использует ту же
строку $A$, а $C_{m+1,n}$ — тот же столбец $B$. Поэтому потенциальное повторное
использование уже содержится в формуле, но независимые вычисления его не
организуют. Кэш может помочь, однако алгоритм не должен полагаться только на
случайное сохранение нужных элементов. Если бы $A$ и $B$ целиком помещались в памяти
чипа, повторные чтения из HBM не понадобились бы. Большие матрицы не помещаются,
поэтому их разбивают на небольшие плитки (*tiles*), доступные для повторного
использования во время вычисления одного блока результата.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/l05-p42.png]]

*Tatsunori Hashimoto и команда Stanford CS336, Lecture 5, p. 42:
блок $C$ связан с полосой блоков $A$ и полосой блоков $B$. Тёмные блоки —
текущая пара, оранжевый блок — накапливаемый результат. Зелёные строка и столбец
показывают вычисление одного его элемента. На слайде квадратная матрица имеет
размер $N$, а блок — $T$; в формулах ниже используются общие размеры $M,N,K$.
[Закреплённый PDF](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_05.pdf#page=42).*

Один экземпляр программы отвечает за прямоугольник результата
$B_M\times B_N$. Чтобы продвинуться на $B_K$ позиций по общей размерности,
он загружает блок $A$ размера $B_M\times B_K$ и блок $B$ размера
$B_K\times B_N$, перемножает их и прибавляет результат к своему аккумулятору.
Следующая пара блоков обновляет **тот же** аккумулятор. Только после обхода
всей размерности $K$ готовый блок $C$ записывается в глобальную память.

Числовой пример с двумя шагами по $K$:

$$
A=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix},
\qquad
B=\begin{bmatrix}1&2\\3&4\\5&6\end{bmatrix}.
$$

Пусть блок результата имеет размер $2\times2$, а $B_K=2$.
На первом шаге берём первые два столбца $A$ и первые две строки $B$:

$$
\begin{bmatrix}1&2\\4&5\end{bmatrix}
\begin{bmatrix}1&2\\3&4\end{bmatrix}
=\begin{bmatrix}7&10\\19&28\end{bmatrix}.
$$

Например, левый верхний элемент равен $1\cdot1+2\cdot3=7$.
Это ещё не ответ: третья координата общей размерности не обработана.
В последнем шаге остаются один столбец и одна строка; маски дают нули
во второй, лишней позиции:

$$
\begin{bmatrix}3&0\\6&0\end{bmatrix}
\begin{bmatrix}5&6\\0&0\end{bmatrix}
=\begin{bmatrix}15&18\\30&36\end{bmatrix}.
$$

К прежнему аккумулятору прибавляется новый вклад:

$$
\begin{aligned}
C&=\begin{bmatrix}7&10\\19&28\end{bmatrix}
+\begin{bmatrix}15&18\\30&36\end{bmatrix}
\\&=\begin{bmatrix}22&28\\49&64\end{bmatrix}.
\end{aligned}
$$

Аккумулятор не обнуляется между шагами. Обнуление или запись только
последнего произведения потеряли бы вклад первых двух координат $K$.
Размеры $2\times2$ здесь выбраны для ручного расчёта; они не являются
предложением конфигурации Tensor Cores.

На одном таком шаге загружено $B_MB_K+B_KB_N$ элементов и выполнено
$2B_MB_NB_K$ FLOP, считая умножение и сложение за две операции. Если каждый
входной элемент занимает $s$ байт, интенсивность относительно этих загрузок
равна

$$
\begin{aligned}
I_{\text{tile}}&\approx
\frac{2B_MB_NB_K}{s(B_MB_K+B_KB_N)}\\
&=\frac{2B_MB_N}{s(B_M+B_N)}.
\end{aligned}
$$

Это оценка без записи результата, краёв и межблочного кэша, а не полная
модель времени GEMM. Она уже объясняет два важных эффекта. При
$B_M=B_N=T$ получаем $I\approx T/s$: расширение блока результата повышает
повторное использование. А одно только увеличение $B_K$ в этой оценке
сокращается в числителе и знаменателе — оно меняет длину цикла, конвейер и
ресурсы, но не создаёт такой же выигрыш в повторном использовании.

Например, $B_M=B_N=64$, $B_K=32$ и FP16-входы дают 4096 загруженных
элементов, 8192 байта и 262144 FLOP, то есть 32 FLOP/байт для одного шага.
Аккумулятор $64\times64$ в FP32 сам содержит ещё 16 КиБ логических значений.
Они распределяются между потоками, но суммарное давление на регистры никуда
не исчезает. Так возникает предел идеи «сделаем плитку побольше».

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/l05-p43.png]]

*Tatsunori Hashimoto и команда Stanford CS336, Lecture 5, p. 43,
с заимствованным на слайде примером NVIDIA: неполный крайний блок объясняет лишнюю работу
и необходимость чтения с маской. Он следует за основной схемой: крайний блок — следствие
выбранного разбиения, а не отдельная оптимизация.
[Закреплённый слайд](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_05.pdf#page=43).*

На слайде достаточно изменить ширину результата с 256 до 257, чтобы при
блоках $128\times128$ потребовались не четыре, а шесть блоков. В двух новых
блоках полезен только один столбец. Маска гарантирует корректность, но не
превращает неполный блок в бесплатный: логические формы и часть работы остаются
прежними. Это эффект округления числа блоков (*tile quantization*).

Адресация строится из тех же приёмов, что и в сложении векторов. Программа
получает координаты блока результата $(p_m,p_n)$, создаёт векторы индексов
$m_i=p_mB_M+i$, $n_j=p_nB_N+j$ и $k_l=qB_K+l$. Адреса очередной пары блоков
равны

$$
\begin{aligned}
\operatorname{addr}A_{il}&=A_{\mathrm{ptr}}+m_i s_{Am}+k_l s_{Ak},\\
\operatorname{addr}B_{lj}&=B_{\mathrm{ptr}}+k_l s_{Bk}+n_j s_{Bn}.
\end{aligned}
$$

Две размерности адресов получаются широковещательным расширением
(`broadcasting`): `m[:, None]` и `k[None, :]` образуют сетку $B_M\times B_K$.
При загрузке $A$ нужны обе проверки $m_i<M$ и $k_l<K$; при загрузке $B$ —
$k_l<K$ и $n_j<N$. Заполнители равны нулю, поэтому не изменяют произведение.
На каждом шаге выполняется

$$
C_{tile}\leftarrow C_{tile}+A_{m,k}B_{k,n}.
$$

В Triton блочное умножение выражается через `tl.dot`; аккумулятор FP32
позволяет не округлять частичную сумму до FP16 после каждого шага. При записи
нужна маска по $M$ и $N$, даже если загрузки уже были замаскированы. Полная
адресация и цикл приведены в [Matrix Multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html).
Выбранное отображение на матричные инструкции зависит от типа входа, формы
блока и целевого GPU: наличие `tl.dot` в исходнике само по себе не доказывает
достижение пиковой производительности Tensor Cores.

### Повторное использование между программами

До сих пор мы повторно использовали данные внутри одного блока результата.
Но соседним программам тоже нужны общие блоки $A$ или $B$. Если назначать
близкие идентификаторы программ пространственно близким блокам результата,
возрастает вероятность, что общий вход ещё находится в L2. Это другой уровень
повторного использования: программы не обмениваются своим локальным
аккумулятором, а обращаются к общему кэшу устройства.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/triton-grouped-vs-row-major.png]]

*Philippe Tillet, OpenAI и участники Triton,
[Matrix Multiplication — L2 Cache Optimizations](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#l2-cache-optimizations):
сверху первые девять выходных блоков лежат в одной строке и требуют 9 блоков
$A$ и 81 блока $B$; снизу участок $3\times3$ требует по 27 блоков каждого
входа. Числа 0–8 показывают логическое назначение работы, а не гарантию
последовательного исполнения GPU. Оригинальный PNG без изменений,
[MIT, закреплённый источник](https://github.com/triton-lang/triton/blob/1b24a39939875de5333e3990bec3ddcbfc64eb80/docs/getting-started/tutorials/grouped_vs_row_major_ordering.png).*

Схема сравнивает рабочий набор для группы выходных блоков: 90 входных блоков
против 54 при благоприятном повторном использовании. Она не утверждает,
что суммарный трафик любой матрицы обязательно уменьшится в отношении 90:54.
Реальный выигрыш зависит от ёмкости L2 и от того, какие программы одновременно
исполняются. Поэтому изменение порядка программ проверяют измерениями,
сохраняя прежними формулу и набор выходных блоков.

### Почему крупные блоки и объединение операций не всегда ускоряют

Параметры `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `num_warps` и число стадий конвейера
выбирают совместно. Больший блок результата повышает повторное использование,
но увеличивает аккумулятор. Дополнительные стадии могут перекрыть загрузку
следующей порции вычислением текущей, но требуют держать больше данных
одновременно. Эти ресурсы ограничивают число резидентных программ и варпов
на мультипроцессоре — показатель, называемый *occupancy*. Максимальная
occupancy не является самостоятельной целью: важен баланс между достаточным
параллелизмом и полезной работой каждого блока.

После GEMM можно применить GeLU/ReLU до записи и не создавать отдельный
тензор активаций. Но объединение имеет предел. Если заключительные операции
потребуют больше регистров, число одновременно размещённых блоков может
сократиться. При нехватке регистров компилятор может выгружать значения
в память (*register spilling*), и часть устранённых обращений появится снова
в другой форме. Если сетка мала, слишком крупные блоки также оставят часть
мультипроцессоров без работы. Выигрыш от уменьшения обмена с памятью тогда
окажется меньше потерь от недостаточной параллельной загрузки GPU.

Есть и алгоритмическая граница. Операцию, требующую результатов многих
независимых программ, нельзя просто дописать в конец каждого блока и считать
глобально синхронизированной. Потребуется иной алгоритм или ещё одно ядро.
Таким образом, fusion — не требование «всё в один запуск», а выбор разумной
границы, внутри которой доступны нужные данные и помещается рабочее состояние.

<!-- BEGIN REVIEW: indexed movement 2026-10-27 -->

<a id="indexed-gather-scatter-and-gradients"></a>

## Indexed gather and scatter

An embedding lookup selects rows by token or item ID. A sparse graph layer
selects neighbours by node ID. In both cases, the addresses depend on an
integer index tensor. The same array can be read several times, skipped or
reordered without changing its stored layout. This differs from the fixed
strides used to walk a dense GEMM tile.

For a one-dimensional input $x=(10,20,30,40)$ and indices $I=(2,0,2,3)$,
gather returns $y=(30,10,30,40)$. Output positions 0 and 2 both read input
position 2; no output reads input position 1. The input is unchanged.
In coordinates, $y_k=x_{I_k}$.

### Following an index along one axis

The original PyTorch explanation specifies how this extends to more axes:

<!-- BEGIN SOURCE: PyTorch gather coordinates -->

Gathers values along an axis specified by `dim`.

For a 3-D tensor the output is specified by:

```text
out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
```

`input` and `index` must have the same number of dimensions.
It is also required that `index.size(d) <= input.size(d)` for all
dimensions `d != dim`. `out` will have the same shape as `index`.
Note that `input` and `index` do not broadcast against each other.
When `index` is empty, we always return an empty output with the same shape
without further error checking.

<!-- END SOURCE: PyTorch gather coordinates -->

*PyTorch contributors, [`torch.gather`, v2.14.1 source](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/_torch_docs.py#L4812-L4849),
[license and copyright notices](/licenses/pytorch-v2.14.1-LICENSE.txt).
Original English; documentation roles and the literal-block delimiter are
converted to Markdown. All examples below use non-negative `torch.int64`
indices and CPU tensors.*

For `dim=1` in a matrix, the output row stays fixed while the index chooses
the input column. With input `[[1, 2], [3, 4]]` and index
`[[0, 0], [1, 0]]`, output row 0 reads column 0 twice, giving `[1, 1]`.
Output row 1 reads columns 1 and 0, giving `[4, 3]`. With the same arrays
and `dim=0`, columns stay fixed and the indices choose rows instead:
the result is `[[1, 2], [3, 2]]`.

```python
import torch

data = torch.tensor([[1, 2], [3, 4]])
index = torch.tensor([[0, 0], [1, 0]], dtype=torch.int64)
assert torch.gather(data, 1, index).tolist() == [[1, 1], [4, 3]]
assert torch.gather(data, 0, index).tolist() == [[1, 2], [3, 2]]

# The output follows the index shape, not the input shape.
larger = torch.tensor([[1, 2], [3, 4], [5, 6]])
assert torch.gather(larger, 0, index).shape == (2, 2)
```

ONNX has two different operators with “Gather” in their names.
[`GatherElements-13`](https://github.com/onnx/onnx/blob/3646b038c23824207d064d8625a908b1bf8710f4/onnx/defs/doc_strings.cc#L4343-L4396)
uses the coordinate rule above and returns the index tensor's shape.
[`Gather-13`](https://github.com/onnx/onnx/blob/3646b038c23824207d064d8625a908b1bf8710f4/onnx/defs/doc_strings.cc#L4397-L4461)
selects whole slices: each scalar index replaces the selected axis with a
slice of rank one less than the input. For data shape `[3, 2]`, index shape
`[2, 2]` and axis 0, its output shape is `[2, 2, 2]`; `GatherElements`
instead returns `[2, 2]`. The general `Gather` output shape is
`data.shape[:axis] + indices.shape + data.shape[axis+1:]`, after normalizing
the axis. Similar names do not establish shape compatibility.

ONNX also permits in-range negative index values for these operators.
Core PyTorch `gather` requires non-negative index values. An exporter or a
hand-written port must preserve the source API's indexing contract rather
than substitute an operator by name.

### Scatter-add combines values with the same destination

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/indexed-movement-2026-10-27/scatter-add.svg|Scatter-add: eight indexed inputs accumulate into four outputs 8, 7, 5 and 4]]

*Matthias Fey, [PyTorch Scatter, `scatter`](https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html),
[complete original SVG, revision `f514c10`](https://github.com/rusty1s/pytorch_scatter/blob/f514c10f920b5aeed2eb162092f0ad20d3edee52/docs/source/_figures/add.svg),
[MIT copyright and permission notice](/licenses/pytorch-scatter-f514c10-MIT.txt).
No crop, redraw or relabelling. The top row gives destination indices;
curved arrows group input values by destination. Each `add` combines a
group. The drawing specifies the mathematical grouping, not a GPU schedule.*

<!-- BEGIN SOURCE: PyTorch Scatter reduction -->

Reduces all values from the `src` tensor into `out` at the
indices specified in the `index` tensor along a given axis
`dim`.
For each value in `src`, its output index is specified by its index
in `src` for dimensions outside of `dim` and by the
corresponding value in `index` for dimension `dim`.
The applied reduction is defined via the `reduce` argument.

<!-- END SOURCE: PyTorch Scatter reduction -->

*Original English: Matthias Fey, [`scatter` docstring](https://github.com/rusty1s/pytorch_scatter/blob/f514c10f920b5aeed2eb162092f0ad20d3edee52/torch_scatter/scatter.py#L92-L176),
MIT; only Sphinx attribute markup is converted to Markdown.*

The figure's input values are $(5,1,7,2,3,2,1,3)$ and its indices are
$(0,0,1,0,2,2,3,3)$. Starting from zeros, destination 0 receives 5, then 1,
then 2, giving the successive totals 5, 6 and 8. Destination 1 receives
only 7; destination 2 receives $3+2=5$; destination 3 receives $1+3=4$.
The result is $(8,7,5,4)$. Index order need not be sorted, and equal
values at different source positions still make separate contributions.

For an initial output $b$, the operation is

$$
\operatorname{out}_j=b_j+\sum_{k:I_k=j}\operatorname{src}_k.
$$

An output with no incoming values keeps $b_j$. Starting from four tens
would therefore give $(18,17,15,14)$, not $(8,7,5,4)$.
The core PyTorch operation below makes both the destination size and its
initial contents explicit. It reproduces the figure without requiring the
optional `torch_scatter` package.

```python
import torch

src = torch.tensor([5, 1, 7, 2, 3, 2, 1, 3])
index = torch.tensor([0, 0, 1, 0, 2, 2, 3, 3])
out = torch.zeros(5, dtype=src.dtype)
out.scatter_add_(0, index, src)
assert out.tolist() == [8, 7, 5, 4, 0]  # destination 4 is untouched

base = torch.full((4,), 10, dtype=src.dtype)
assert base.scatter_add_(0, index, src).tolist() == [18, 17, 15, 14]

# Assignment is appropriate here because all destination indices are unique.
assigned = torch.full((4,), -1, dtype=src.dtype)
assigned.scatter_(0, torch.tensor([2, 0]), torch.tensor([7, 9]))
assert assigned.tolist() == [9, -1, 7, -1]
```

The underscore marks an in-place update. These examples use fresh destination
buffers; modifying a tensor already needed by autograd requires separate
lifetime and aliasing checks. The package `torch_scatter.scatter` can
broadcast its index argument and can infer the destination axis size as
`index.max() + 1`. Core
[`Tensor.scatter_add_`](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/_tensor_docs.py#L4432-L4488)
does not broadcast `index` with `src`; they and the destination have the same
rank, `index.size(d) <= src.size(d)` on every axis, and
`index.size(d) <= out.size(d)` outside the selected axis. Its documented
backward requires `src.shape == index.shape`. The examples satisfy these
conditions rather than rely on package-specific broadcasting.

Assignment and addition have different repeated-index contracts. The original
PyTorch warning for `scatter_` without a reduction reads:

<!-- BEGIN SOURCE: PyTorch repeated scatter assignment -->

> When indices are not unique, the behavior is non-deterministic (one of the
> values from `src` will be picked arbitrarily) and the gradient will be
> incorrect (it will be propagated to all locations in the source that
> correspond to the same index)!

<!-- END SOURCE: PyTorch repeated scatter assignment -->

*PyTorch contributors, [`Tensor.scatter_`, v2.14.1](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/_tensor_docs.py#L4303-L4341),
[license](/licenses/pytorch-v2.14.1-LICENSE.txt).
This warning concerns assignment with colliding destinations; it does not
describe the mathematical rule of `scatter_add_`.*

An ordinary parallel read-modify-write also cannot implement scatter-add
safely. If two participants both read an initial 0, add 5 and 1 independently,
then write 5 and 1, one contribution is lost. The completed sum must be 6.
Atomic addition, or an algorithm that groups and reduces each destination
before writing, prevents that lost update. PyTorch Scatter documents atomic
GPU implementations. Atomic addition still leaves the floating-point order
unspecified: adding all contributions correctly does not guarantee a
bitwise-identical sum across schedules. Sorting and segmented reduction can
change that trade-off, but require their own grouping work and storage.

### Why gather's backward uses scatter-add

Return to $x=(10,20,30,40)$, $I=(2,0,2,3)$ and the gathered
$y=(30,10,30,40)$. Suppose the incoming gradient is $g=(1,2,3,4)$,
as it would be for $L=y_0+2y_1+3y_2+4y_3$.
Input $x_2$ contributed to both $y_0$ and $y_2$, so the chain rule adds
their two contributions, $1+3=4$. Input $x_0$ contributed only to $y_1$,
giving 2. Input $x_1$ was never selected, giving 0; input $x_3$ gives 4.
Thus $\nabla_x L=(2,0,4,4)$.

For a fixed integer index tensor, the general derivative is

$$
\frac{\partial L}{\partial x_j}=\sum_{k:I_k=j}g_k.
$$

This is scatter-add into an initially zero gradient buffer. Replacing it
with repeated-index assignment would discard a path through the computation.
Conversely, for $z=b+\operatorname{scatter\_add}(I,s)$, the gradient of
each update is the output gradient at its destination:
$\partial L/\partial s_k=(\nabla_z L)_{I_k}$, a gather. The gradient with
respect to the initial $b$ is the output gradient itself. Integer indices
are held fixed; these rules do not differentiate a discrete routing decision.

```python
import torch

index = torch.tensor([2, 0, 2, 3])
x = torch.tensor([10., 20., 30., 40.], dtype=torch.float64,
                 requires_grad=True)
g = torch.tensor([1., 2., 3., 4.], dtype=x.dtype)
y = x.gather(0, index)
dx, = torch.autograd.grad((y * g).sum(), (x,))
expected = torch.zeros_like(x).scatter_add_(0, index, g)
torch.testing.assert_close(dx, expected)
assert dx.tolist() == [2., 0., 4., 4.]

# Reversing the arrows does not reconstruct an omitted or duplicated input.
reconstructed = torch.zeros_like(x).scatter_add_(0, index, y.detach())
assert reconstructed.tolist() == [10., 0., 60., 40.]

updates = torch.tensor([5., 7., 11., 13.], dtype=x.dtype,
                       requires_grad=True)
initial = torch.ones(4, dtype=x.dtype, requires_grad=True)
z = initial.scatter_add(0, index, updates)
ds, db = torch.autograd.grad((z * g).sum(), (updates, initial))
torch.testing.assert_close(ds, g.gather(0, index))
torch.testing.assert_close(db, g)
```

Writing gather as a fixed selection matrix $G_I$ makes the distinction from
an inverse explicit: $y=G_Ix$ and $\nabla_x L=G_I^\top g$.
The transpose sends contributions back along every selected path.
However, $(G_I^\top G_Ix)_j=c_jx_j$, where $c_j$ counts occurrences of $j$.
For the example $c=(1,0,2,1)$: the missing coordinate is lost and the repeated
coordinate is doubled. The ONNX descriptions call `GatherElements` and
`ScatterElements` inverses, but arbitrary repeated or omitted indices do not
define an invertible map. A full one-to-one permutation is the special case
where reversing the index mapping can reconstruct all coordinates.

The same accumulation appears in
[[02 Areas/ML & DL/00 Учебник/02 Рекомендательные системы/03 Ranking и нейронные рекомендательные модели|embedding-bag gradients]]:
repeated IDs contribute repeatedly to one table row. That chapter develops
pooling and table sharding; here the destination is an element of a local
tensor, not a device or process.

### Address patterns and the cost of an index

For the one-dimensional gather, the address calculation becomes

$$
\begin{aligned}
\operatorname{addr}(x_{I_k})&=x_{\mathrm{ptr}}+I_k s_x,\\
\operatorname{addr}(y_k)&=y_{\mathrm{ptr}}+k s_y.
\end{aligned}
$$

The source address needs the loaded index value. Output writes can remain
contiguous while source reads visit distant regions. Adjacent or repeated
indices can still share memory transactions or benefit from cache; indexed
access does not imply a universal slowdown or the absence of reuse.

For a concrete transaction-count model, assume 32 active lanes, one FP32
read per lane, a 32-byte-aligned base, and the 32-byte access units documented
for NVIDIA compute capability 6.0 and above. Indices 0–31 occupy four units;
indices $I_k=32k$ occupy 32. Both request 128 useful bytes, but cover 128
versus 1024 bytes of access units. These counts omit index loads and output
writes and are not measured HBM traffic or an eightfold runtime prediction.
[NVIDIA's coalescing explanation, §10.2.1.1–10.2.1.4](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)
also shows why caching can change the observed penalty.

```python
def fp32_access_units(indices, base_bytes=0):
    return {(base_bytes + 4 * i) // 32 for i in indices}

assert len(fp32_access_units(range(32))) == 4
assert len(fp32_access_units([32 * k for k in range(32)])) == 32
assert len(fp32_access_units(reversed(range(32)))) == 4
assert len(fp32_access_units([7] * 32)) == 1
```

Sorting indices may improve source locality or group repeated destinations,
but a gather must then restore the requested output order. Sorting,
temporary storage and the inverse permutation enter the cost. In scatter-add,
a skewed index distribution can concentrate atomic updates on a few
destinations. It offers reuse but also contention. The index distribution,
element width, cache residency and surrounding computation belong in a
performance experiment; the operator name alone does not predict its time.

These local tensor operations are separate from `torch.distributed.gather`
and `torch.distributed.scatter`, which transfer tensors between processes.
A sharded embedding lookup can require both indexed local accesses and
inter-device exchanges, but their contracts and costs must be analysed
separately. Process-level examples continue in
[[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/44a Processes, collectives и DDP|Processes, collectives и DDP]].

<!-- END REVIEW: indexed movement 2026-10-27 -->

<a id="kernel-numerics-backward"></a>

## Численная корректность и обратный проход

Смена разбиения меняет порядок сложений. В точной арифметике
$(a+b)+c=a+(b+c)$, но округление после каждой операции нарушает эту
ассоциативность. Поэтому даже корректные GPU-редукции могут не совпадать
побитово с последовательной суммой на CPU или с другим библиотечным ядром.
Более точный аккумулятор уменьшает ошибку накопления, но не восстанавливает
информацию, потерянную при преобразовании самих входов в FP16/BF16.

У GEMM отдельно проверяют точность умножения и точность аккумулятора.
FP32-вход с FP32-аккумулятором ещё не означает, что каждое произведение
вычисляется с полной FP32-точностью. Для NVIDIA параметр `input_precision`
в `tl.dot` управляет, в частности, использованием TF32; доступные режимы и
значения по умолчанию нужно сверять с [текущей документацией `tl.dot`](https://triton-lang.org/main/python-api/generated/triton.language.dot.html).
Если Triton и эталон используют разные режимы, тест сравнивает не только
алгоритмы, но и разные численные контракты. Связь форматов и накопления
подробнее разобрана в [[02 Areas/ML & DL/00 Учебник/10 ML Systems/05 Численные форматы и mixed precision|главе о mixed precision]].

Условие близости обычно имеет вид

$$
|y-y_{\mathrm{ref}}|\le\mathrm{atol}
+\mathrm{rtol}\,|y_{\mathrm{ref}}|.
$$

Абсолютный допуск нужен около нуля; относительный — при большом масштабе
результата. Числа выбирают под тип данных, длину редукции и использование
операции, а не увеличивают до исчезновения ошибки теста. Помимо случайных
данных полезны почти сокращающиеся большие значения, одинаковые входы,
редкие экстремумы и границы блоков. Для матричного умножения независимо
меняют $M$, $N$ и $K$: квадратная матрица кратного размера не проверяет
ни хвост общей размерности, ни неполный выходной блок.

Для обучения проверка прямого прохода — только половина задачи. Простой
вызов Triton-ядра не создаёт автоматически нужное правило дифференцирования
для PyTorch. Операцию интегрируют с autograd, определяют обратный проход и
проверяют производные по каждому входу. Наши две учебные обёртки отклоняют
входы с `requires_grad=True` именно потому, что такого правила в них нет.

Например, для softmax $y$ и входящего градиента $g_j=\partial L/\partial y_j$
производная по логитам равна

$$
\frac{\partial L}{\partial x_j}
=y_j\left(g_j-\sum_k g_k y_k\right).
$$

Например, при $y=(0{,}25,0{,}75)$ и $g=(2,0)$ скалярная сумма
$\sum_k g_ky_k$ равна $0{,}5$. Производные по двум логитам равны
$0{,}25(2-0{,}5)=0{,}375$ и
$0{,}75(0-0{,}5)=-0{,}375$. Их сумма равна нулю:
одинаковый сдвиг обоих логитов не меняет softmax и потому
не меняет потери. Если забыть общее слагаемое и оставить только
поэлементное $y_jg_j$, получится $(0{,}5,0)$ — другое правило
дифференцирования, несмотря на правильный forward.

Обратный проход снова содержит редукцию по строке и поэлементный участок:
та же организация блоков может быть полезна, но нужно решить, где взять $y$.
Можно сохранить его для backward, а можно восстановить по входу, повторив
часть вычислений. Первый вариант расходует память между проходами, второй —
арифметику и чтения. Это выбор между сохранением и пересчётом, а не бесплатное
продолжение ускоренного forward.

Градиенты сравнивают с эталонным autograd на небольших входах; там, где
операция и поддерживаемые типы позволяют, дополнительно используют `gradcheck`
или проверку направленной производной в высокой точности. Численная
производная FP64-эталона сама по себе не доказывает корректность иного
FP16-ядра: необходимо проверить именно реализованный backward. Затем
проверяют блок модели, один полный шаг оптимизатора и короткую траекторию
обучения. Допустимое локальное отклонение и сохранение поведения обучения —
связанные, но разные уровни проверки.

<a id="fused-linear-cross-entropy"></a>

## Выходной слой и cross-entropy без полной матрицы логитов

После последнего блока языковой модели скрытые состояния ещё нужно
спроецировать на словарь. Пусть после сдвига осталось $P$ позиций,
размер скрытого состояния равен $d$, а словарь содержит $V$ токенов.
Для матрицы $H\in\mathbb R^{P\times d}$ и весов
$W\in\mathbb R^{d\times V}$ логиты равны $X=HW$.
Уже одна матрица $X$ содержит $PV$ чисел.

При $P=32768$, $V=128000$ и двух байтах на число это
8 388 608 000 байт, или 7.8125 ГиБ. Здесь посчитаны только логиты:
ни веса, ни состояния оптимизатора, ни сохранённые активации не вошли.
В прямой композиции могут появляться дополнительные большие результаты
`log_softmax`, softmax и градиент логитов. Их фактическое число зависит
от реализации и компиляции. Оценка «четыре полные матрицы» из учебного
разбора EDLS не описывает обязательный расход любого PyTorch CE.

### Что даёт объединение cross-entropy

Для одной учитываемой позиции с целью $t$ достаточно получить

$$
\begin{aligned}
L_i&=m+\log\ell-x_{it},\\
\frac{\partial L_i}{\partial x_{ij}}&=p_{ij}-[j=t],
\end{aligned}
$$

где $m$ и $\ell$ — максимум и нормировочная сумма строки из предыдущего
раздела, а $p_{ij}=\exp(x_{ij}-m)/\ell$. Необязательно сохранять
полный отдельный тензор вероятностей. После первого обхода строки
известны $m$, $\ell$ и потеря. Во втором обходе можно восстанавливать
вероятности по частям и сразу писать градиент поверх уже использованных
логитов. Такой путь уменьшает одновременное число больших буферов,
но исходная матрица логитов $[P,V]$ всё ещё существует.

Перезапись допустима только внутри операции, которая управляет временем
жизни этого буфера. Если те же логиты нужны для другой потери, метрик
или последующего autograd, уничтожать их нельзя. Это не общая
рекомендация присвоить новое значение входу произвольной PyTorch-функции.

### Fused linear cross-entropy: обрабатывать и проекцию по частям

Liger FLCE объединяет не только операции потери, но и выходную
проекцию. В оригинальном объяснении Pin-Lun Hsu и соавторов,
[Liger Kernel, §3.2](https://arxiv.org/html/2410.10989v2#S3.SS2),
последовательность описана так; английский текст сохранён:

> The main idea of FLCE is shown in Figure 1. The 3D hidden states
> (shifted already to align with their next ground truth tokens) are
> flattened into a 2D matrix by collapsing the batch size and sequence
> length dimensions into a single dimension. The linear projection
> head is applied sequentially on the chunked hidden states.
>
> The generated output logits are passed to the non-fused Liger CE
> kernel to compute the partial loss and return the chunked logits
> gradient for deriving the chunked hidden states gradients and the
> accumulated projection head gradients.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/triton-review-2026-10-23/liger-flce-original.png]]

*Pin-Lun Hsu и соавторы, [Liger Kernel, Figure 1](https://arxiv.org/html/2410.10989v2#S3.SS2),
arXiv:2410.10989v2, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
[Оригинальный PNG](https://arxiv.org/html/2410.10989v2/liger_flce_v3.png)
перенесён без изменений. Оранжевый фрагмент — часть скрытых состояний;
зелёный — её логиты, которые затем используются для хранения градиентов.
Веса выходного слоя общие для всех фрагментов, поэтому их градиенты
нужно складывать. Внутренний CE-kernel объединяет операции самой
cross-entropy; подпись «non-fused CE» отличает его от объединения с linear.*

Пусть одновременно обрабатываются $C=512$ позиций. Их скрытые состояния
$H_c$ дают логиты $X_c=H_cW$ размера $[512,128000]$.
Такой BF16-буфер занимает 131 072 000 байт, или 125 МиБ,
вместо 7.8125 ГиБ всей матрицы. После вычисления потери и градиентов этой порции
буфер можно использовать для следующей. Это оценка одного буфера,
не пик памяти всего шага и не значение, автоматически выбираемое Liger.

Обозначим через $G_c$ градиент потерь по логитам порции. Обратный проход
выходной проекции состоит из двух матричных произведений:

$$
\begin{aligned}
\nabla_{H_c}L&=G_cW^\top,\\
\nabla_W L&=\sum_c H_c^\top G_c.
\end{aligned}
$$

Первый результат относится только к своим строкам $H_c$; второй
накапливается в одной общей матрице весов. Нельзя заменить сумму
градиентов весов последним слагаемым. Размер этой матрицы $[d,V]$
не уменьшается от разбиения токенов, как не исчезает и градиент входа
$[P,d]$. Экономится прежде всего промежуточное пространство словаря.

Для среднего по токенам знаменатель также общий. Если три цели
разделены на порции по две и одной, их средние потери имеют веса
$2/3$ и $1/3$, а не $1/2$ и $1/2$. При `ignore_index` считают
только учитываемые цели. В
[закреплённой реализации Liger](https://github.com/linkedin/Liger-Kernel/blob/6ce504c3286178ba6537624252f0a1ee88423b53/src/liger_kernel/ops/fused_linear_cross_entropy.py)
веса хранятся в форме `[V,d]`: наша матрица $W$ соответствует
`weight.T`. Число неигнорируемых целей считается до цикла по порциям и передаётся
CE-kernel. Более общие случаи — веса классов, label smoothing,
soft-capping и Z-loss — требуют собственных производных и нормировки.

### Исполняемый CPU-эталон разбиения

Этот пример показывает формулы для обычной cross-entropy без bias,
весов классов и сглаживания меток. Он **не является Liger-кодом**:
создаёт отдельные тензоры логитов и их градиентов, а не переиспользует
один GPU-буфер. Поэтому им проверяется математика разбиения,
не экономия памяти или производительность CUDA.
Матрицы имеют совместимые размеры; каждая цель — индекс от 0 до $V-1$
или `-100`, обозначающий исключённую из потерь позицию.

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def chunked_linear_ce(h, w, target, chunk_size):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    n_valid = (target != -100).sum().item()
    if n_valid == 0:
        raise ValueError("mean loss needs at least one valid target")
    loss = h.new_zeros(())
    dh, dw = torch.zeros_like(h), torch.zeros_like(w)
    for start in range(0, h.shape[0], chunk_size):
        stop = min(start + chunk_size, h.shape[0])
        hc, tc = h[start:stop], target[start:stop]
        valid = tc != -100
        safe_target = torch.where(valid, tc, 0)
        rows = torch.arange(stop - start, device=h.device)
        z = hc @ w
        lse = torch.logsumexp(z, dim=1)
        loss += (lse - z[rows, safe_target])[valid].sum() / n_valid
        g = (z - lse[:, None]).exp()
        g[rows, safe_target] -= 1
        g[~valid] = 0
        g /= n_valid
        dh[start:stop] = g @ w.T
        dw += hc.T @ g
    return loss, dh, dw

h = torch.tensor([[1., 0.], [0., 1.], [1., 1.]],
                 dtype=torch.float64, requires_grad=True)
w = torch.tensor([[1., 0., -1., 0.], [0., 1., 0., -1.]],
                 dtype=torch.float64, requires_grad=True)
target = torch.tensor([0, 1, 3])
full_loss = F.cross_entropy(h @ w, target)
full_dh, full_dw = torch.autograd.grad(full_loss, (h, w))
loss, dh, dw = chunked_linear_ce(h, w, target, chunk_size=2)
torch.testing.assert_close(loss, full_loss)
torch.testing.assert_close(dh, full_dh)
torch.testing.assert_close(dw, full_dw)
```

Логиты первых двух строк — $(1,0,-1,0)$ и $(0,1,0,-1)$;
верный класс в обоих случаях имеет логит 1. Их нормировочная сумма
после вычитания максимума равна $1+2e^{-1}+e^{-2}$,
а потеря — $\log(1+2e^{-1}+e^{-2})$.
Третья строка даёт $(1,1,-1,-1)$ с целевым логитом $-1$:
её потеря равна $2+\log(2+2e^{-2})$.
Первые две строки дают одинаковую потерю примерно 0.626523,
третья — 2.820075. Среднее по трём целям равно примерно 1.357707.
Среднее двух средних было бы около 1.723299: одна последняя строка
получила бы столько же веса, сколько первые две вместе. Сравнение
`dh` и `dw` проверяет не только скалярную потерю, но и обе ветви
обратного прохода проекции.

В Liger для скалярной итоговой потери часть градиентов можно вычислить
уже в forward и сохранить для последующего умножения на входящий
градиент loss. При произвольном векторе потерь `reduction="none"`
входящие веса строк ещё неизвестны. Их нужно учесть **до** суммирования
$\nabla_W$, либо пересчитать соответствующую часть в backward.
После суммирования восстановить разные веса отдельных строк уже нельзя.
Этот случай отдельно обработан в `LigerFusedLinearCrossEntropyFunction`
указанной версии. Прямой перенос трюка для скалярной потери на любую
другую функцию потерь некорректен.

Меньшие порции уменьшают временный буфер, но делают GEMM меньше
и увеличивают число запусков. Размер порции поэтому подбирают вместе
со скоростью полного шага. FLCE также не означает «весь выходной слой
в одном CUDA-kernel»: проекции и CE могут оставаться разными ядрами,
хотя они объединены в одну операцию autograd и обрабатывают данные частями.

<a id="autotuning--поиск-по-корректным-кандидатам"></a>

<a id="triton-autotuning"></a>

## Автоматический подбор параметров

Когда алгоритм и его контракт уже проверены, остаётся выбрать подходящее
разбиение. Triton позволяет указать несколько конфигураций и измерить их
для заданных размеров входа. Например, для одного GEMM выгоден широкий блок
результата, а для узкой матрицы почти весь такой блок будет заполнителем.
Автоматический поиск не определяет задачу вместо разработчика:

- пространство кандидатов задаёт человек;
- время компиляции и подбора отделяют от времени работы после прогрева;
- лучшая конфигурация одного размера не обязана быть лучшей для соседнего;
- каждый кандидат должен давать корректный результат и соблюдать ограничения памяти;
- кэш результатов подбора привязывают к GPU, версии ПО и параметрам входа.

Набор проверок включает рабочие размеры и границы блоков. Например, вместе с
4096 проверяют 4095 и 4097. Иначе поиск может выбрать конфигурацию, которая
быстра только на идеально кратном размере.

Декоратор `triton.autotune` выбирает конфигурацию по времени, а не проверяет
математическую правильность каждой реализации. Тесты корректности нужны
отдельно. Параметр `key` задаёт аргументы, изменение которых вызывает новый
подбор; набор ключей и механизм диспетчеризации должны учитывать те свойства
входа, для которых меняется удачная конфигурация. Произвольное повторное
использование результата для другой формы или размещения данных недопустимо.

Во время поиска ядро исполняется многократно. Для чистой записи результата
это обычно безопасно, но ядро с `out += value` изменит буфер несколько раз.
Если `out` изначально равен нулю, а `value` — единице, после двух пробных
запусков получится 2 вместо результата одного вызова, равного 1.
При сбросе перед каждым запуском оба измерения выполняют одну и ту же
задачу. Для состояния оптимизатора с ненулевым начальным значением
нужно восстановление этого значения, а не безусловное обнуление.
Для такого случая нужны сброс или восстановление состояния, например
`reset_to_zero` и `restore_value`, описанные в
[документации autotune](https://triton-lang.org/main/python-api/generated/triton.autotune.html).
Особенно осторожно настраивают ядра оптимизатора: запуск кандидатов не должен
незаметно выполнить несколько настоящих обновлений параметров модели.

<a id="как-читать-профиль-triton-kernel"></a>

<a id="kernel-measurement"></a>

## От профиля ядра к измеренному результату

Успешная компиляция доказывает, что компилятор принял программу, а не что она
быстрее библиотеки. Для сравнения сначала фиксируют формы, типы, размещение
входов и границы измеряемой операции. Время готового kernel не равно времени
обёртки, если одна обёртка выделяет выход или копирует несмежный вход, а другая
получает подготовленные буферы. Оба измерения полезны, но отвечают на разные
вопросы и должны называться отдельно.

Первый вызов может включать JIT-компиляцию, подбор конфигурации и другие
разовые затраты. Их измеряют отдельно от времени после прогрева. Сам GPU
работает асинхронно: таймер вокруг Python-вызова без ожидания завершения
часто измеряет лишь отправку работы. Для времени на устройстве используют
события соответствующего потока или проверенный benchmark helper; для
полной задержки — корректно синхронизированное измерение внешней границы.
Профилировщик помогает объяснить результат, но финальную серию измерений
полезно проводить без его накладных расходов.

Для сложения длины $n$ можно сообщить эффективную пропускную способность
$12n/t$ байт/с, где $t$ — время в секундах. Для fused softmax FP32 аналогичная
модель даёт $8MN/t$. Это **полезные байты по модели**, не показание счётчика
физического трафика HBM: кэш, повторное использование одного входа и
внутренние передачи меняют фактический поток данных. Для GEMM обычно
используют $2MNK/t$ FLOP/с. Пик, с которым сравнивают эту величину, должен
соответствовать точности вычислений и наличию либо отсутствию структурной
разреженности.

Далее профиль отвечает на последовательность вопросов:

1. **Запустилось ли нужное ядро?** Имя и число запусков с CPU должны
   соответствовать вызовам, а сетка — числу экземпляров программы внутри
   запуска. Возврат к другой реализации или разрыв графа меняют эксперимент.
2. **Ждёт ли GPU процессор?** Если есть паузы в подаче работы с CPU,
   внутренняя арифметика ядра может не быть главным ограничением.
3. **Каковы достигнутые пропускная способность и FLOP/s?** Их сопоставляют с
   roofline для соответствующего уровня памяти и типа чисел.
4. **Хватает ли параллельной работы?** Проверяют сетку, волны исполнения,
   неполные блоки и активные варпы.
5. **Хватает ли регистров и разделяемой памяти?** Выгрузка регистров в память
   и снижение числа резидентных варпов могут устранить ожидаемый выигрыш.
6. **На каких размерах стало медленнее?** Сравнивают не только среднее, но и каждую
   заранее объявленную форму.

`torch.compile` и Triton можно использовать вместе. Компилятор способен
генерировать ядра Triton для захваченного графа. Ручная реализация полезна,
когда автоматическая система не выбирает нужное размещение данных или порядок
вычислений либо требуется особая операция. Сравнение с библиотечным ядром
остаётся обязательным: собственную реализацию выбирают по измеренному
выигрышу или необходимой функциональности.

Итог эксперимента — таблица по всем заявленным формам: корректность,
устойчивая оценка времени и разброс, версия GPU/драйвера/PyTorch/Triton,
выбранная конфигурация и условия прогрева. Если ускорение есть только на
части форм, это основание для диспетчеризации с библиотечным запасным путём,
а не для усреднения, скрывающего регрессии. Если ядро ускорено, но полный шаг
модели не изменился, нужно вернуться к
[[02 Areas/ML & DL/00 Учебник/10 ML Systems/07 Profiling ML-нагрузки|профилю полного шага]]:
возможно, устранённая работа не лежала на критическом пути.

## Переход к FlashAttention

Softmax показал, как объединить редукции одной строки; блочное матричное
умножение — как повторно использовать данные. Внимание объединяет обе задачи.
При длинной последовательности полная матрица $QK^\top$ не помещается в памяти
чипа, а нормировка строки
запроса зависит от всех блоков ключей. Потоковый softmax позволяет обновлять
максимум, знаменатель и взвешенную сумму значений по мере их обхода. Прямой
и обратный проходы разобраны в отдельной главе:
[[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/56 FlashAttention|FlashAttention]].

Для обучения этот переход шире одного attention kernel. Fusion уменьшает
промежуточный обмен; пересчёт меняет объём сохранённых активаций; более
экономичный backward может позволить увеличить пакет. В
[задании EDLS «Optimizing Transformer Training»](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week06_dl_arithmetic/homework/README.md)
эти решения проверяются уже в составе RMSNorm, SwiGLU, функции потерь и
полного трансформера. Быстрое отдельное ядро имеет смысл только внутри
правильного шага обучения: с теми же целевыми токенами, нормировкой потерь,
градиентами и правилом обновления весов.

Конвейер подготовки данных определяет входы и формы; профилирование
показывает затраты, а реализация ядер меняет стоимость конкретных операций.
В [[02 Areas/ML & DL/00 Учебник/10 ML Systems/09 Компиляция графов — PyTorch, JAX и CUDA Graphs|следующей главе]]
разобраны компиляция графов, guards, состояние JAX и CUDA Graphs.
После вычислительных основ можно перейти к обучению языковой модели на корпусе:
[[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/01 Данные и pre-training|данные и предобучение]],
[[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/42 Next-token prediction|предсказание следующего токена]]
и [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/44b Gradient checkpointing и offload|сохранение и пересчёт активаций]].

## Практика и первоисточники

- [Stanford CS336 Lecture 5, версия `8b59b507`, с. 30–54](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_05.pdf) — объединение операций, пересчёт, блочное разбиение, волны исполнения и переход к FlashAttention.
- [Stanford CS336 Lecture 6, версия `8b59b507`](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_06.py) — исполняемые примеры GeLU, поэлементного ядра, softmax, суммы строки и блочного GEMM.
- [Учебные примеры Triton](https://triton-lang.org/main/getting-started/tutorials/) — актуальные примеры языка и API; совместимость сверяют с установленной версией Triton.
- [EDLS, неделя 6: kernel fusion и torch.compile](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week06_dl_arithmetic/seminar/practice.ipynb) — поэлементное объединение SwiGLU, компиляция и проверка в обучении.
- [Tillet, Kung, Cox, Triton](https://openai.com/index/triton/) и [репозиторий Triton](https://github.com/triton-lang/triton) — модель программирования и реализация компилятора.
- [[02 Areas/ML & DL/06 Практика/21 Профилировать и ускорить Transformer kernel|Capstone: профилировать и ускорить Transformer kernel]].


<details>
<summary>Перенесённые разделы о компиляции графов</summary>

Разделы перенесены целиком в следующую главу. Старые якоря сохранены:

<a id="tracing-a-data-dependent-branch"></a>

- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/09 Компиляция графов — PyTorch, JAX и CUDA Graphs#Tracing a data-dependent branch|Tracing a data-dependent branch]]

<a id="a-python-branch-between-compiled-regions"></a>

- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/09 Компиляция графов — PyTorch, JAX и CUDA Graphs#A Python branch between compiled regions|A Python branch between compiled regions]]

<a id="exporting-an-explicit-conditional"></a>

- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/09 Компиляция графов — PyTorch, JAX и CUDA Graphs#Exporting an explicit conditional|Exporting an explicit conditional]]

<a id="torch-compile-graphs-guards-recompilation"></a>
<a id="torchcompile-graphs-guards-and-recompilation"></a>

- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/09 Компиляция графов — PyTorch, JAX и CUDA Graphs#torch-compile-graphs-guards-recompilation|torch.compile: graphs, guards and recompilation]]

<a id="what-dynamo-captures"></a>

- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/09 Компиляция графов — PyTorch, JAX и CUDA Graphs#What Dynamo captures|What Dynamo captures]]

<a id="why-a-compiled-function-needs-guards"></a>

- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/09 Компиляция графов — PyTorch, JAX и CUDA Graphs#Why a compiled function needs guards|Why a compiled function needs guards]]

<a id="graph-breaks-divide-one-call-into-compiled-regions"></a>

- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/09 Компиляция графов — PyTorch, JAX и CUDA Graphs#Graph breaks divide one call into compiled regions|Graph breaks divide one call into compiled regions]]

<a id="dynamic-shapes-and-changing-scalar-values"></a>

- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/09 Компиляция графов — PyTorch, JAX и CUDA Graphs#Dynamic shapes and changing scalar values|Dynamic shapes and changing scalar values]]

<a id="jax-tracing-state-and-compiled-control-flow"></a>

- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/09 Компиляция графов — PyTorch, JAX и CUDA Graphs#JAX: tracing, state and compiled control flow|JAX: tracing, state and compiled control flow]]

<a id="a-python-update-disappears-from-the-recorded-program"></a>

- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/09 Компиляция графов — PyTorch, JAX и CUDA Graphs#A Python update disappears from the recorded program|A Python update disappears from the recorded program]]

<a id="state-as-an-argument-and-a-result"></a>

- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/09 Компиляция графов — PyTorch, JAX и CUDA Graphs#State as an argument and a result|State as an argument and a result]]

<a id="refs-are-traceable-mutable-state"></a>

- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/09 Компиляция графов — PyTorch, JAX и CUDA Graphs#Refs are traceable mutable state|Refs are traceable mutable state]]

<a id="what-a-cached-specialization-can-reuse"></a>

- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/09 Компиляция графов — PyTorch, JAX и CUDA Graphs#What a cached specialization can reuse|What a cached specialization can reuse]]

<a id="a-runtime-predicate-needs-a-compiled-conditional"></a>

- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/09 Компиляция графов — PyTorch, JAX и CUDA Graphs#A runtime predicate needs a compiled conditional|A runtime predicate needs a compiled conditional]]

<a id="a-loop-carries-fixed-shaped-state"></a>

- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/09 Компиляция графов — PyTorch, JAX и CUDA Graphs#A loop carries fixed-shaped state|A loop carries fixed-shaped state]]

<a id="donation-changes-buffer-ownership"></a>

- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/09 Компиляция графов — PyTorch, JAX и CUDA Graphs#Donation changes buffer ownership|Donation changes buffer ownership]]

<a id="device-placement-and-collective-communication"></a>

- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/09 Компиляция графов — PyTorch, JAX и CUDA Graphs#Device placement and collective communication|Device placement and collective communication]]

<a id="cuda-graphs-capture-replay"></a>
<a id="cuda-graphs-capture-and-replay"></a>

- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/09 Компиляция графов — PyTorch, JAX и CUDA Graphs#cuda-graphs-capture-replay|CUDA Graphs: capture and replay]]

<a id="why-cuda-graphs"></a>

- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/09 Компиляция графов — PyTorch, JAX и CUDA Graphs#Why CUDA Graphs?|Why CUDA Graphs?]]

<a id="pytorch-api"></a>

- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/09 Компиляция графов — PyTorch, JAX и CUDA Graphs#PyTorch API|PyTorch API]]

<a id="follow-the-buffers-through-two-replays"></a>

- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/09 Компиляция графов — PyTorch, JAX и CUDA Graphs#Follow the buffers through two replays|Follow the buffers through two replays]]

<a id="capture-constraints-and-dynamic-workloads"></a>

- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/09 Компиляция графов — PyTorch, JAX и CUDA Graphs#Capture constraints and dynamic workloads|Capture constraints and dynamic workloads]]

<a id="memory-pools-are-part-of-the-cost"></a>

- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/09 Компиляция графов — PyTorch, JAX и CUDA Graphs#Memory pools are part of the cost|Memory pools are part of the cost]]

<a id="read-an-actual-before-and-after-trace"></a>

- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/09 Компиляция графов — PyTorch, JAX и CUDA Graphs#Read an actual before-and-after trace|Read an actual before-and-after trace]]

<a id="account-for-overlap-and-capture-amortization"></a>

- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/09 Компиляция графов — PyTorch, JAX и CUDA Graphs#Account for overlap and capture amortization|Account for overlap and capture amortization]]

</details>

← [[02 Areas/ML & DL/00 Учебник/10 ML Systems/07 Profiling ML-нагрузки|Profiling ML-нагрузки]] ·
[[02 Areas/ML & DL/00 Учебник/10 ML Systems/09 Компиляция графов — PyTorch, JAX и CUDA Graphs|Компиляция графов]] →
