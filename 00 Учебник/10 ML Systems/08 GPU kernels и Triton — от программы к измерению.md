---
title: GPU kernels и Triton — от программы к измерению
type: textbook-chapter
status: draft
last_updated: 2026-10-27
last_verified: 2026-10-27
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
  - https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/notes/cuda.md
  - https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
  - https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/performance_engineering/performance_engineering.qmd
  - https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/user_guide/torch_compiler/compile/programming_model.dynamo_core_concepts.md
  - https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/user_guide/torch_compiler/compile/programming_model.recompilation.md
  - https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/user_guide/torch_compiler/torch.compiler_dynamo_overview.md
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

<!-- BEGIN REVIEW: tracing control flow 2026-10-27 -->

## Tracing a data-dependent branch

Suppose a function multiplies its input by 2 when the sum is positive and
by 3 otherwise. A recorded execution on `[1]` will contain a multiplication
by 2. Before reusing that recording on `[-1]`, the runtime needs a way to
preserve the other branch or reject that reuse. The tensor shape and dtype
are identical in these two calls, but their required computations differ.

The complete **Tracing Silent Failure** example below comes from
[Harvard ML Systems, ML Frameworks — Hybrid approaches: JIT compilation](https://mlsysbook.ai/contents/vol1/frameworks/frameworks.html#sec-ml-frameworks-hybrid-approaches-jit-compilation-8954),
Vijay Janapa Reddi and contributors,
[source revision `2bd97c5`](https://github.com/harvard-edge/cs249r_book/blob/2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd/book/quarto/contents/vol1/frameworks/frameworks.qmd),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
The function, call and comments are unchanged; only `import torch` is added
so the example can run separately on CPU.

TorchScript is a historical example here. PyTorch 2.14 documents it as
[deprecated](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/notes/cpu_threading_torchscript_inference.md)
and directs new export work to `torch.export`.

```python
import torch

def conditional_forward(x):
    if x.sum() > 0:  # Data-dependent condition
        return x * 2
    else:
        return x * 3


traced = torch.jit.trace(conditional_forward, torch.tensor([1.0]))
# Tracing captures ONLY the x.sum() > 0 branch
# If input later has sum <= 0, traced version
# still executes x * 2 branch
```

The positive example selects the first branch. Its recorded graph contains
`aten::mul` with a constant 2, without the sum, comparison or conditional.
On `[-1]`, the Python function returns `[-3]`, while `traced` returns `[-2]`.
The latter is a valid multiplication; the error is the missing choice of
which multiplication to perform.

| Input | Python | Trace |
|---|---|---|
| `[1]` | `[2]` | `[2]` |
| `[-1]` | `[-3]` | `[-2]` |
| `[0]` | `[0]` | `[0]` |
| `[1,-1]` | `[3,-3]` | `[2,-2]` |

A zero input conceals the wrong branch because both formulas return zero.
The last input also has sum zero, but its nonzero entries expose the difference.
Correctness tests need cases that distinguish the alternatives, not only cases
that happen to produce the same result.

```python
branch_cases = [
    ([1.], [2.]),
    ([-1.], [-3.]),
    ([0.], [0.]),
    ([1., -1.], [3., -3.]),
]
for values, expected in branch_cases:
    value = torch.tensor(values)
    torch.testing.assert_close(conditional_forward(value), torch.tensor(expected))
    torch.testing.assert_close(traced(value), value * 2)

assert not torch.equal(traced(torch.tensor([-1.])), torch.tensor([-3.]))
trace_nodes = {node.kind() for node in traced.graph.nodes()}
assert "aten::mul" in trace_nodes
assert trace_nodes.isdisjoint({"aten::sum", "aten::gt", "prim::If"})
```

Tracing emits a `TracerWarning` when the tensor condition is converted to a
Python boolean. That warning does not prevent construction of the artifact.
The default trace checker can run the function more than once, but uses the
original example unless other checking inputs are supplied. Repeating a
positive input still exercises only the positive branch. The
[`check_inputs` contract](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/jit/_trace.py#L918-L930)
allows another input to expose the disagreement:

```python
try:
    torch.jit.trace(
        conditional_forward, torch.tensor([1.]),
        check_inputs=[(torch.tensor([-1.]),)],
    )
except torch.jit.TracingCheckError:
    pass  # the additional input exposes incompatible traces
else:
    raise AssertionError("the negative checking input must reject this trace")
```

Passing a finite set of checks does not prove correctness for all possible
inputs. Disabling `check_trace` would remove a check, not restore the missing
conditional. The Harvard chapter also demonstrates how historical TorchScript
scripting retains both branches in its IR. That comparison explains the old
API; it does not reverse TorchScript's current deprecation status.

### A Python branch between compiled regions

Default `torch.compile` can leave a data-dependent Python decision outside
compiled regions, evaluate it for the current call, and continue on the
selected branch. The recording is no longer an unconditional replay of the
first branch. This CPU example uses `backend="eager"` to inspect correctness
without invoking an optimizing backend:

```python
compiled_branch = torch.compile(conditional_forward, backend="eager")
for values, expected in branch_cases:
    torch.testing.assert_close(
        compiled_branch(torch.tensor(values)), torch.tensor(expected)
    )
```

All four cases now agree with Python. This result demonstrates supported
fallback around the branch, not fusion, a single captured graph or a speedup.
When capturing from a fresh compiler cache, `fullgraph=True` requires a single
graph and rejects this ordinary data-dependent Python `if`. To test that
requirement independently of previously cached variants, reset the cache first:

```python
torch.compiler.reset()
strict_branch = torch.compile(
    conditional_forward, backend="eager", fullgraph=True
)
try:
    strict_branch(torch.tensor([-1.]))
except torch._dynamo.exc.Unsupported:
    pass
else:
    raise AssertionError("fresh fullgraph capture must reject the Python if")
```

The reset is part of this isolated capture experiment, not a step to insert
into a steady-state timing loop. Without it, this sequence can reuse a
previously compiled variant instead of exercising fresh strict capture.

### Exporting an explicit conditional

An exported program must represent its supported computation without falling
back to arbitrary Python between regions. A condition on dynamic tensor data
cannot be decided from shape metadata alone. The original PyTorch
[Dynamic Data-Dependent Control Flow](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/user_guide/torch_compiler/export/programming_model.md#dynamic-data-dependent-control-flow)
example replaces Python `if` with an explicit `torch.cond`. Both classes below
are copied unchanged from the source; their branches use sine and cosine:

```python
class M_old(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x.sin()
        else:
            return x.cos()

class M_new(torch.nn.Module):
    def forward(self, x):
        return torch.cond(
            pred=x.sum() > 0,
            true_fn=lambda x: x.sin(),
            false_fn=lambda x: x.cos(),
            operands=(x,),
        )
```

*PyTorch contributors, v2.14.1,
[full copyright and BSD-style license](/licenses/pytorch-v2.14.1-LICENSE.txt).
Original source language and code retained.*

`pred` is a scalar boolean tensor. `operands` supplies the input to either
branch. The graph retains both branch functions and the conditional operator;
the runtime selects one using the current predicate. For this example,
the branches produce new tensors with matching shape and dtype.

```python
exported_branch = torch.export.export(M_new(), (torch.ones(1),), strict=False)
exported_module = exported_branch.module()
python_branch = M_old()
for scalar in (1., -1., 0.):
    value = torch.tensor([scalar])
    torch.testing.assert_close(exported_module(value), python_branch(value))
assert "higher_order.cond" in exported_branch.graph_module.code
```

For `[1]`, the result is approximately `[0.841471]`; for `[-1]`, it is
`[0.540302]`; for `[0]`, it is `[1]`. The two latter calls use cosine.
The export was created with shape `[1]`; retaining the conditional does not
automatically authorize other shapes. Changing tensor data and changing a
specialized shape are different operations.

`strict=False` selects non-strict export tracing. It does not permit an
arbitrary Python `if` on unknown tensor data: exporting `M_old` still fails.
The supported branch structure is expressed by `torch.cond` in `M_new`.
PyTorch labels [control-flow operators](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/higher_order_ops/index.md)
as prototype features; supported
input/output structures, mutation and training behavior must be checked for
the chosen version. These examples test CPU forward behavior, not deployment
to a particular accelerator or an arbitrary conditional backward.

<!-- END REVIEW: tracing control flow 2026-10-27 -->

<!-- BEGIN REVIEW: torch.compile 2026-10-26 -->

<a id="torch-compile-graphs-guards-recompilation"></a>

## torch.compile: graphs, guards and recompilation

The earlier GeLU example showed fewer kernels after compilation. To understand
when that result can be reused, follow the Python function into Dynamo,
the frontend of `torch.compile`. Dynamo does not require every statement
to become a GPU instruction. It extracts tensor computations, leaves other
work in Python, and records the conditions under which this combination is
valid. The original *Dynamo Overview* diagram shows those paths separately.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/torch-compile-review-2026-10/torch-dynamo-pipeline.png|PyTorch Dynamo: guards, transformed Python bytecode, FX graphs and a compiler backend]]

*PyTorch contributors,
[Dynamo Overview](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/user_guide/torch_compiler/torch.compiler_dynamo_overview.md),
[unchanged original diagram](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/_static/img/dynamo/TorchDynamo.png),
[copyright and license notices](/licenses/pytorch-v2.14.1-LICENSE.txt).
`PyCodeObject` is the function's Python code object; a frame supplies the
current invocation's arguments and execution state. The right-hand branch
passes FX graphs to the backend. The transformed Python code calls its
result and retains the Python work outside those graphs. This is a compiler
structure diagram, not a kernel-count or speedup measurement.*

### What Dynamo captures

<!-- BEGIN SOURCE: PyTorch Dynamo tracing -->

`torch.compile`'s frontend (Dynamo) is a custom Python bytecode interpreter designed to allow graph compilation
in PyTorch programs while retaining the full flexibility of Python. Given a function to be compiled, Dynamo
interprets Python bytecode to extract sequences of PyTorch operations into 1 or more FX graphs that may be further optimized by a backend.

<!-- END SOURCE: PyTorch Dynamo tracing -->

*Original English: PyTorch contributors,
[Dynamo Core Concepts, “Dynamo Tracing”](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/user_guide/torch_compiler/compile/programming_model.dynamo_core_concepts.md#dynamo-tracing),
v2.14.1; [license](/licenses/pytorch-v2.14.1-LICENSE.txt).*

An FX graph represents operations and the values passed between them.
For `relu(x * scale + 1)`, its tensor path contains multiplication, addition
and ReLU. It does not yet say whether these will become one kernel, several
library calls, or another implementation. A backend receives this graph and
returns a callable that executes it. The usual Inductor backend can lower and
combine operations into generated kernels or library calls; capturing three
FX operations alone proves neither fusion nor a performance improvement.

In the usual compiled training path, AOTDispatcher (also called AOTAutograd)
constructs the backward graph; Inductor can then compile forward and backward
computations. Backward assumptions, including its autocast context, must agree
with the actual training step. They cannot be changed arbitrarily after
compilation. The versioned
[autograd-semantics page](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/user_guide/torch_compiler/torch.compiler_backward.md)
explains those assumptions and the corresponding configuration. The frontend
experiment below deliberately returns `graph_module.forward` unchanged:
it inspects capture and reuse, not AOTAutograd or Inductor code generation.

### Why a compiled function needs guards

<!-- BEGIN SOURCE: PyTorch Dynamo guards -->

`torch.compile` makes some assumptions about runtime values as we trace through code. During tracing, we generate "guards",
which are runtime checks for these assumptions. Guards are run in future calls to the compiled function to determine if we
can reuse previously compiled code. Examples of runtime checks are constant values, types, and object IDs.

<!-- END SOURCE: PyTorch Dynamo guards -->

*Original English: same source, “Guards”. The following experiment adapts the
custom-backend example from
[Dynamo Overview, “What is a guard?”](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/user_guide/torch_compiler/torch.compiler_dynamo_overview.md#what-is-a-guard).
The backend records `gm.code` instead of printing a tabular graph, so no
optional table-formatting package is needed.*

```python
import torch

compile_records = []

def inspect_backend(gm, example_inputs):
    compile_records.append(gm.code)
    return gm.forward

def affine_relu(x, scale):
    return torch.relu(x * scale + 1)

compiled_affine = torch.compile(
    affine_relu, backend=inspect_backend, dynamic=False, fullgraph=True
)
x = torch.tensor([-1.0, 0.0, 2.0])
torch.testing.assert_close(compiled_affine(x, 2.0), torch.tensor([0.0, 1.0, 5.0]))
assert len(compile_records) == 1

# Same metadata and Python constant, different tensor contents: reuse.
torch.testing.assert_close(
    compiled_affine(torch.tensor([-2.0, 1.0, 3.0]), 2.0),
    torch.tensor([0.0, 3.0, 7.0]),
)
assert len(compile_records) == 1

# A different Python float invalidates the value specialization.
torch.testing.assert_close(compiled_affine(x, 3.0), torch.tensor([0.0, 1.0, 7.0]))
assert len(compile_records) == 2

# dynamic=False also specializes this example on shape.
torch.testing.assert_close(compiled_affine(torch.ones(4), 3.0), torch.full((4,), 4.0))
assert len(compile_records) == 3

# The first version is still cached; the latest version need not match.
torch.testing.assert_close(compiled_affine(x, 2.0), torch.tensor([0.0, 1.0, 5.0]))
assert len(compile_records) == 3
```

The first entry, `compile_records[0]`, contains this FX function in the
PyTorch 2.14.1 CPU run:

<!-- BEGIN OBSERVED: first Dynamo FX function -->

```text
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    mul = l_x_ * 2.0;  l_x_ = None
    add = mul + 1;  mul = None
    relu = torch.relu(add);  add = None
    return (relu,)
```

<!-- END OBSERVED: first Dynamo FX function -->

`scale` is no longer an argument to this function: `2.0` has been embedded
in its multiplication. Passing a Python scale of `3.0` cannot safely reuse
that function. The guard on the original call prevents it. Assignments of
`None` release intermediate Python references; they are not a claim that
each assignment frees GPU memory or creates a kernel boundary.

The callback counts backend invocations, not kernel launches. On the first
call, Dynamo can specialize `scale` to the Python float `2.0`, and `x` to
the observed tensor metadata. Replacing only the numbers inside a tensor
does not invalidate those particular assumptions. Changing the Python float
or the specialized shape does. A failed guard does not mean that the answer
may now be wrong: it prevents reuse of code whose assumptions no longer hold.

The fifth call also explains why a guard failure for one cached version need
not trigger recompilation. Another cached version may match. New tracing is
needed when none of the available variants is applicable, subject to the
compiler's cache limits and fallback policy. Tensor guards can also cover
dtype, device, stride, `requires_grad` and other execution properties;
shape alone is not a full compatibility test.

### Graph breaks divide one call into compiled regions

<!-- BEGIN SOURCE: PyTorch Dynamo graph breaks -->

Dynamo traces your code and attempts to capture your PyTorch code into a single computation graph of PyTorch
operators (FX graph). However, this is not always possible. When encountering code that can't be traced, a "**graph break**" occurs.
In the default `torch.compile` settings, a graph break involves compiling the FX graph that has been determined so far,
running the unsupported code in regular Python, then resuming tracing after the unsupported code with a new FX graph.

Graph breaks are a feature that allows Dynamo to run over arbitrary Python code and carve out functional subgraphs that can each be individually optimized.

<!-- END SOURCE: PyTorch Dynamo graph breaks -->

*Original English: PyTorch contributors, “Graph Breaks” in the same
[Dynamo Core Concepts source](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/user_guide/torch_compiler/compile/programming_model.dynamo_core_concepts.md#graph-breaks).*

This example places an explicit boundary between a square and an addition.
It uses the documented `torch._dynamo.graph_break()` to make the split
reproducible, rather than relying on an operation that a later compiler
version might learn to capture.

```python
import torch

piece_records = []

def record_piece(gm, example_inputs):
    piece_records.append(gm.code)
    return gm.forward

def with_boundary(x):
    y = x.square()
    torch._dynamo.graph_break()
    return y + 1

split_function = torch.compile(with_boundary, backend=record_piece, dynamic=False)
v = torch.tensor([-2.0, 0.0, 3.0])
torch.testing.assert_close(split_function(v), torch.tensor([5.0, 1.0, 10.0]))
assert len(piece_records) == 2
torch.testing.assert_close(split_function(v + 1), (v + 1).square() + 1)
assert len(piece_records) == 2  # two cached regions, no new backend call

@torch.compile(backend="eager", fullgraph=True)
def strict_boundary(x):
    y = x.square()
    torch._dynamo.graph_break()
    return y + 1

try:
    strict_boundary(v)
except torch._dynamo.exc.Unsupported:
    pass
else:
    raise AssertionError("fullgraph=True must reject this explicit graph break")
```

The first FX graph returns the squared tensor, which becomes an input to the
resumed region. The second graph adds one. A second invocation can reuse both
regions while still crossing their boundary; a graph break is not necessarily
a new compilation on every call. [`fullgraph=True`](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/user_guide/torch_compiler/compile/programming_model.fullgraph_true.md)
instead requires capture as one graph and rejects this explicit break.
It does not make the unsupported operation capturable.

With a normal optimizing backend, a break can prevent fusion across that
boundary and retain Python/dispatch work between regions. It can also be an
intentional trade-off: the official
[“Graph Breaking to Reduce Recompilation Costs” example](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/user_guide/torch_compiler/compile/programming_model.recompilation.md#graph-breaking-to-reduce-recompilation-costs)
keeps a large stable computation separate from a small expression that
specializes on a changing Python constant. Removing every break is therefore
not an unconditional optimization. Measure the affected workload and compile
cost rather than counting breaks alone.

### Dynamic shapes and changing scalar values

The first example used `dynamic=False` to expose shape specialization.
With the default `dynamic=None`, Dynamo may first specialize and, after a
shape mismatch, try a more general graph with symbolic sizes. `dynamic=True`
asks for dynamic shapes upfront where possible. It does not promise one graph
for every Python program, rank, stride, shape-dependent branch or
[size-zero and size-one case](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/user_guide/torch_compiler/compile/dynamic_shapes_zero_one_specialization.md).

Python scalar values are a separate source of specialization. The original
[“Wrapping Constants with Tensors” section](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/user_guide/torch_compiler/compile/programming_model.recompilation.md#wrapping-constants-with-tensors)
demonstrates changing floats, including a learning rate. A tensor can carry
a changing value as data rather than as an exact Python constant. For the
tensor arithmetic in this example, both variations can be handled together:

```python
import torch

dynamic_records = []

def record_dynamic(gm, example_inputs):
    dynamic_records.append(gm.code)
    return gm.forward

@torch.compile(backend=record_dynamic, dynamic=True, fullgraph=True)
def tensor_scale(x, scale):
    return torch.relu(x * scale + 1)

for length, scale_value in [(3, 2.0), (5, 3.0), (7, 4.0)]:
    x = torch.arange(length, dtype=torch.float32) - 1
    scale = torch.tensor(scale_value)
    actual = tensor_scale(x, scale)
    torch.testing.assert_close(actual, torch.relu(x * scale + 1))
assert len(dynamic_records) == 1
```

This is a CPU frontend experiment with PyTorch 2.14.1. All three custom
backends return the captured graph unchanged; no Inductor kernel, GPU
execution, CUDA graph or speedup is tested. Symbolic shape handling does
not remove other guards, and moving a scalar into a tensor is useful only
when the operations consuming it support that representation.

For a real model, `TORCH_LOGS=graph_breaks,recompiles,guards` exposes these
different causes. Separate the first compilation, new variants and steady
execution in timing results. A backend compilation, an FX operation and a
device kernel are different things to count. CUDA Graphs, below, address
another boundary: replaying device work after the kernels and their arguments
have already been chosen.

<!-- END REVIEW: torch.compile 2026-10-26 -->

<!-- BEGIN REVIEW: JAX runtime 2026-10-27 -->
## JAX: tracing, state and compiled control flow

`jax.jit` also records a program before executing its compiled version, but
ordinary Python state changes are outside that recorded program. A counter
makes the consequence visible: a method that returns 1, 2, 3 in Python can
return 1, 1, 1 after compilation.

The original explanations and code marked below are by The JAX Authors,
from [Stateful computations](https://docs.jax.dev/en/latest/101/state.html),
[Just-in-time compilation](https://docs.jax.dev/en/latest/201/jit.html) and
[Control flow](https://docs.jax.dev/en/latest/201/control-flow.html),
Apache-2.0, pinned to JAX 0.12.0.1
([source](https://github.com/jax-ml/jax/tree/dc4af684446e65ee98eee8090e1ff6f8211473fb/docs)).
Their English is retained; MyST cross-references and code-cell fences are
converted to ordinary Markdown. Additional worked checks are identified by
their surrounding explanation. The [license](/licenses/jax-dc4af684-Apache-2.0.txt)
is included with the published material.

Run the examples in a fresh Python process. The first block configures four
CPU devices before JAX initializes its backend, so that the collective near
the end can actually run. These are logical CPU devices in one process,
not four GPUs or a multi-host experiment.

```python
import jax
jax.config.update("jax_num_cpu_devices", 4)
import jax.numpy as jnp
import numpy as np
from jax import lax
```

### A Python update disappears from the recorded program

<!-- BEGIN SOURCE: JAX untraceable counter -->
Let's start with a simple stateful program: a counter.

```python
import jax
import jax.numpy as jnp

class Counter:
  """A simple counter."""

  def __init__(self):
    self.n = 0

  def count(self) -> int:
    """Increments the counter and returns the new value."""
    self.n += 1
    return self.n

counter = Counter()

for _ in range(3):
  print(counter.count())
```

The `n` attribute maintains the counter's *state* between calls, updated as a
side effect. As plain Python this works fine, but it falls apart under JAX
transformations. We can see exactly how by looking at what tracing records,
using the jaxpr-inspection idiom from [Transformations](https://docs.jax.dev/en/latest/101/transformations.html):

```python
counter = Counter()
jax.jit(counter.count).trace().jaxpr
```

The recorded program takes no input, performs no operations, and always
returns 1. The side effect `self.n += 1` ran once, at trace time; the *value*
`self.n` produced was captured as a constant; and the state update itself is
nowhere in the jaxpr. Any transformation working from this traced view is
working with a function that returns 1, forever. (Under `jax.jit`, which
caches traces, that's precisely what you'd observe: the compiled counter
returns 1 on every call.)
<!-- END SOURCE: JAX untraceable counter -->

Starting from a fresh counter makes the repeated-call result explicit:

```python
broken_counter = Counter()
fast_broken_count = jax.jit(broken_counter.count)
broken_values = [int(fast_broken_count()) for _ in range(3)]
assert broken_values == [1, 1, 1]
assert broken_counter.n == 1
```

`self.n` is a Python integer. During tracing, Python reads 0, adds 1 and
changes the attribute. The resulting compiled function has no state input
and returns the constant 1. Repeating that executable cannot advance the
Python attribute. Changing the class's field after tracing is not a way to
update the executable's inputs.

### State as an argument and a result

<!-- BEGIN SOURCE: JAX threaded counter -->
The fix is to make the state part of the function's interface: take the
current state as an argument, and return the updated state alongside the
output.

```python
CounterState = int

class CounterV2:

  def count(self, n: CounterState) -> tuple[int, CounterState]:
    # You could just return n+1 once, but here we separate its role as
    # the output and as the counter state for didactic purposes.
    return n + 1, n + 1

  def reset(self) -> CounterState:
    return 0

counter = CounterV2()
state = counter.reset()

for _ in range(3):
  value, state = counter.count(state)
  print(value)
```

The caller now keeps track of the state explicitly. In exchange, `count` is
pure and hence easy to trace, and its jaxpr looks quite different:

```python
jax.jit(counter.count).trace(0).jaxpr
```

The state flows visibly through the recorded program: in as the argument, out
as a result. Nothing is baked in or hidden, so every transformation handles
this function correctly.
<!-- END SOURCE: JAX threaded counter -->

Now one compiled addition can process successive values:

```python
state = counter.reset()
fast_count = jax.jit(counter.count)
threaded_values = []
for _ in range(3):
    value, state = fast_count(state)
    threaded_values.append(int(value))
assert threaded_values == [1, 2, 3]
assert int(state) == 3
```

The same interface is used for training: `(parameters, optimizer_state,
batch) -> (new_parameters, new_optimizer_state)`. Momentum, running layer
statistics and random keys must have an explicit path through the program.
The [worked gradient example](/textbook/neural-networks/backpropagation/)
computes individual and mean gradients from such a parameter pytree, then
returns the parameters after an SGD update without mutating the inputs.

### Refs are traceable mutable state

JAX also has a distinct mutable-array type, a *ref*, whose reads and writes
are intercepted by tracing. The following original example updates a ref
captured by a compiled function:

<!-- BEGIN SOURCE: JAX ref update -->
Threading values is the standard approach, but it can be awkward. If a
function deep in your call stack wants to update normalization statistics or
record a metric, every function along the way must plumb that state in and out
of its signature.

For cases like this, JAX has **refs**: mutable arrays that can be read and
written in place. Create one with [`jax.new_ref`](https://docs.jax.dev/en/latest/_autosummary/jax.ref.new_ref.html):

```python
x_ref = jax.new_ref(jnp.zeros(3))  # new array ref, with initial value [0., 0., 0.]

@jax.jit
def f():
  x_ref[1] += 1.  # indexed add-update

print(x_ref)  # Ref([0., 0., 0.])
f()
f()
print(x_ref)  # Ref([0., 2., 0.])
```
<!-- END SOURCE: JAX ref update -->

Reading `x_ref[...]` produces an ordinary array. Unlike the Python integer
update above, the indexed ref update is intercepted by tracing and occurs
when the compiled program runs:

```python
np.testing.assert_array_equal(x_ref[...], [0., 2., 0.])
f()
np.testing.assert_array_equal(x_ref[...], [0., 3., 0.])
```

Refs have transformation-specific restrictions. In particular, `vmap` cannot
map writes to one shared closed-over ref: several logical instances would
write the same location. Passing a batched ref as an argument gives each
instance its own entry. A ref cannot be returned from `jit` or a `scan`
body, and the same ref cannot be supplied twice to a jitted function.
The complete [Refs and purity](https://docs.jax.dev/en/latest/101/state.html#refs-and-purity)
discussion distinguishes internal refs from mutable inputs; the latter have
more restricted transformation rules. These rules do not apply to arbitrary
Python object mutation.

### What a cached specialization can reuse

<!-- BEGIN SOURCE: JAX captured rank branch -->
A jaxpr records the function *as executed on the JAX types it was traced with*.
Python-level control flow and side effects are resolved during tracing. If
a function branches on a static property like rank, the jaxpr knows only the
branch that was taken:

```python
def log2_if_rank_2(x):
  if x.ndim == 2:
    return jnp.log(x) / jnp.log(2.0)
  else:
    return x

jax.jit(log2_if_rank_2).trace(jnp.array([1, 2, 3])).jaxpr
```

For a rank-1 input, the compiled function is just the identity. That's fine,
because the cached executable is only reused for inputs of the same JAX type; a
rank-2 input triggers a fresh trace that takes the other branch.
<!-- END SOURCE: JAX captured rank branch -->

For example, `[1,2,3]` remains unchanged, whereas a matrix `[[1,2],[4,8]]`
becomes `[[0,1],[2,3]]`. Knowing the shape lets Python choose a branch while
tracing. Knowing only a scalar's dtype does not tell Python whether its
runtime value is positive.

<!-- BEGIN SOURCE: JAX cache keys -->
Suppose we define `f = jax.jit(g)`. When `f` is first invoked, it gets
compiled and the resulting executable is cached. The cache key includes:

- the JAX types of the arguments — new shapes or dtypes trigger a fresh
  trace and compile;
- the values of any static arguments; and
- the object identity of the function `g` itself.

<!-- END SOURCE: JAX cache keys -->

A fresh lambda or `partial` inside a repeated loop is a different function
object, even if its source text is unchanged. Keep the transformed callable
outside the loop so it can reuse a specialization.

The following diagnostic records *traces*, not kernel launches. Its Python
list stores only shape, dtype and static metadata, never a tracer. The list
is deliberately an observation of tracing; it would be unsuitable as model
state. All numerical arguments have explicit array dtypes, avoiding another
specialization distinction between weakly typed Python scalars and arrays.

```python
trace_signatures = []

@jax.jit(static_argnames="scale")
def scaled_sum(x, scale):
    trace_signatures.append((x.shape, str(x.dtype), scale))
    return jnp.sum(x) * scale

requests = [
    (jnp.array([1., 2.], dtype=jnp.float32), 2, 6.),
    (jnp.array([3., 4.], dtype=jnp.float32), 2, 14.),
    (jnp.array([1., 2., 3.], dtype=jnp.float32), 2, 12.),
    (jnp.array([1, 2, 3], dtype=jnp.int32), 2, 12.),
    (jnp.array([1., 2.], dtype=jnp.float32), 3, 9.),
    (jnp.array([5., 6.], dtype=jnp.float32), 2, 22.),
]
for x, scale, expected in requests:
    np.testing.assert_allclose(scaled_sum(x, scale), expected)
assert trace_signatures == [
    ((2,), "float32", 2), ((3,), "float32", 2),
    ((3,), "int32", 2), ((2,), "float32", 3),
]

rank_compiled = jax.jit(log2_if_rank_2)
np.testing.assert_array_equal(rank_compiled(jnp.array([1, 2, 3])), [1, 2, 3])
np.testing.assert_allclose(rank_compiled(jnp.array([[1., 2.], [4., 8.]])),
                           [[0., 1.], [2., 3.]], atol=1e-6)
```

| Calls | What changed | Recorded traces so far |
|---|---|---|
| First, then second | Values changed; shape, dtype and `scale` stayed fixed | 1 |
| Third | Length 2 became length 3 | 2 |
| Fourth | `float32` became `int32` | 3 |
| Fifth | Static `scale` changed from 2 to 3 | 4 |
| Sixth | Reused the first signature | 4 |

This is an in-memory cache example in one process, not an exhaustive cache-key
specification. Sharding, backend and compilation options also constrain the
executable. Static arguments must be hashable; marking every changing data
value static trades the tracing error for many specializations. Variable
sequence lengths are often padded to a small set of bucket shapes to bound
that number. JAX's persistent compilation cache is a separate mechanism for
reuse across process restarts.

A closed-over Python value does not provide the same update path as an
argument. Here changing `gain` leaves the cached multiplier at 2:

```python
gain = {"value": 2.0}

@jax.jit
def hidden_gain(x):
    return x * gain["value"]

sample = jnp.array(3., dtype=jnp.float32)
np.testing.assert_allclose(hidden_gain(sample), 6.)
gain["value"] = 5.0
np.testing.assert_allclose(hidden_gain(sample), 6.)

@jax.jit
def explicit_gain(x, multiplier):
    return x * multiplier

np.testing.assert_allclose(explicit_gain(sample, jnp.float32(gain["value"])), 15.)
```

Changing an unrelated input shape could force a new trace that reads the
new gain. That would make the result depend on which specialization ran,
rather than on the function's visible inputs. Pass changing values as
arguments or use supported refs; do not rely on incidental retracing to
refresh hidden state.

### A runtime predicate needs a compiled conditional

The counter fails silently; a Python `if` on an abstract scalar normally
fails during tracing. Python needs a concrete Boolean to decide which
statements to execute. Under `jit`, `x` represents a scalar of known dtype,
not its future value. `lax.cond` records callable branches and a predicate
in the program instead.

```python
def python_branch(x):
    if x > 0:
        return x
    return 2 * x

try:
    jax.jit(python_branch)(jnp.float32(2.))
except jax.errors.TracerBoolConversionError:
    pass
else:
    raise AssertionError("Python unexpectedly selected an abstract branch")

@jax.jit
def compiled_branch(x):
    return jax.lax.cond(x > 0, lambda z: z, lambda z: 2 * z, x)

for value, expected in [(2., 2.), (-2., -4.), (0., 0.)]:
    np.testing.assert_allclose(compiled_branch(jnp.float32(value)), expected)
```

For `-2`, the predicate is false and the second branch returns `-4`. Both
branches are traced and must return the same pytree structure, shapes and
dtypes; selecting a branch does not allow the result shape to vary with
data. An array mask such as `x[x > 0]` has a data-dependent length and
usually needs a fixed-shape alternative under `jit`.

`jnp.where(predicate, a, b)` selects between already expressed array values;
it is not Python short-circuit evaluation. Also, batching a `cond` over a
batch of predicates with `vmap` can turn it into selection. Do not infer
that each unselected branch's arithmetic is skipped in every transformed
version. [The control-flow API](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html)
specifies that boundary.

### A loop carries fixed-shaped state

JAX's four structured control-flow primitives are `cond`, `while_loop`,
`fori_loop` and `scan`. After the conditional, a cumulative sum illustrates
how a loop passes state from one iteration to the next.

<!-- BEGIN SOURCE: JAX scan example -->
The most commonly used of the four is [`jax.lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html): a loop with a
fixed number of iterations that carries state from step to step, optionally
consuming a per-step slice of an input array and stacking per-step outputs.

Python equivalent:

```python
def scan(f, init, xs):
  carry = init
  ys = []
  for x in xs:
    carry, y = f(carry, x)
    ys.append(y)
  return carry, np.stack(ys)
```

```python
def cumsum_step(carry, x):
  new_carry = carry + x
  return new_carry, new_carry   # (next state, this step's output)

final, cumulative = lax.scan(cumsum_step, 0.0, jnp.arange(1., 5.))
print(final)
print(cumulative)
```
<!-- END SOURCE: JAX scan example -->

The four carry values are 1, 3, 6, 10. The final carry is the scalar 10;
the stacked outputs are `[1,3,6,10]`. The carry's pytree structure, shape and
dtype must stay fixed across iterations. A Python loop inside `jit` can
instead be unrolled while tracing, producing a larger recorded program.
`scan` represents a loop with a traced body; that does not guarantee
constant total compilation time for arbitrary bodies and backends.

The sum of these outputs is $4x_1+3x_2+2x_3+x_4$. Reverse-mode
differentiation through `scan` therefore gives `[4,3,2,1]`:

```python
np.testing.assert_allclose(final, 10.)
np.testing.assert_allclose(cumulative, [1., 3., 6., 10.])
np.testing.assert_allclose(scan(cumsum_step, 0., np.arange(1., 5.))[1], cumulative)

def sum_cumulative(xs):
    return jax.lax.scan(cumsum_step, 0., xs)[1].sum()

np.testing.assert_allclose(jax.grad(sum_cumulative)(jnp.arange(1., 5.)),
                           [4., 3., 2., 1.])
```

`lax.while_loop` permits a runtime-dependent iteration count but does not
generally support reverse-mode differentiation. A `fori_loop` with static
bounds can lower to `scan` and support reverse mode. The loop API and its
derivative rule are separate constraints; an executable loop does not by
itself imply an executable gradient through it.

### Donation changes buffer ownership

An immutable-array update returns a value; it does not promise that every
intermediate gets a separate physical allocation. XLA can reuse storage
within a compiled computation. Across a `jit` boundary, donation additionally
allows an input buffer that is no longer needed to back an output.

```python
@jax.jit(donate_argnums=0)
def add_one_donated(x):
    return x + 1

donated_input = jax.device_put(np.arange(4, dtype=np.float32))
donated_output = add_one_donated(donated_input).block_until_ready()
np.testing.assert_array_equal(donated_output, [1., 2., 3., 4.])
assert donated_input.is_deleted()
try:
    np.asarray(donated_input)
except RuntimeError:
    pass
else:
    raise AssertionError("A consumed donated buffer remained usable")
```

The caller relinquishes the input; any alias to that same consumed array is
also unusable. Reusing it is a programming error, not an in-place update that
leaves both old and new versions readable. Not every donated buffer can be
used: an incompatible output shape or dtype can make donation ineffective
and produce a warning. The result above establishes one CPU ownership case,
not a measured reduction in a model's peak GPU memory.

### Device placement and collective communication

`vmap` adds batching rules to a function. It can operate on arrays that are
already sharded, but it does not choose a device mesh or a distributed
reduction for the application. `jit` can compile computations with sharded
arrays and compiler-generated communication; `shard_map` exposes per-shard
code and explicit collectives. A placement specification and a communication
operation answer different questions.

An array deliberately placed on one device is committed there. For two
arrays committed to different single-device meshes, an addition requires
an explicit move to a common placement:

```python
devices = jax.devices("cpu")
assert len(devices) == 4
left = jax.device_put(jnp.float32(1.), devices[0])
right = jax.device_put(jnp.float32(2.), devices[1])
try:
    left + right
except ValueError:
    pass
else:
    raise AssertionError("Committed arrays on different devices were combined")
right_on_left = jax.device_put(right, devices[0])
np.testing.assert_allclose(left + right_on_left, 3.)
```

The JAX Authors' original figure and example below show an all-reduce over
four shards. The complete image is copied without cropping or relabeling
from [Manual parallelism with shard_map](https://docs.jax.dev/en/latest/201/shard-map.html#psum),
Apache-2.0.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/jax-runtime-2026-10-27/psum.png]]

*The JAX Authors, Illustration of a psum computation,
[pinned original](https://github.com/jax-ml/jax/blob/dc4af684446e65ee98eee8090e1ff6f8211473fb/docs/_static/shard_map_01_illustration_of_a_psum_computation.png),
Apache-2.0. Each device starts with a different row and receives the same
column-wise sum. The arrow gives the input/output relation, not a ring
schedule or a latency measurement.*

The source defines `Auto = jax.sharding.AxisType.Auto` in its setup. Here the
four-device CPU setup above supplies the devices:

```python
Auto = jax.sharding.AxisType.Auto
```

<!-- BEGIN SOURCE: JAX psum program -->
```python
mesh1d = jax.make_mesh((4,), ('i',), (Auto,))
jax.set_mesh(mesh1d)

@jax.shard_map(in_specs=jax.P('i'), out_specs=jax.P(None))
def f1(x_block):
  print('BEFORE:\n', x_block)
  y_block = jax.lax.psum(x_block, 'i')
  print('AFTER:\n', y_block)
  return y_block
```

```python
x = jnp.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 1, 2])
y = f1(x)
print('FINAL RESULT:\n', y)
```

The prints show that each function application starts with its own chunk of
the argument value `x_block`. After the `psum`, each function application has
the same value of `y_block`, computed by summing the applications' `x_block`
values together.
<!-- END SOURCE: JAX psum program -->

`P('i')` splits the length-16 input into four rank-1 blocks of length 4.
The first output coordinate is $3+5+5+9=22$; the others are 20, 12 and 17.
This is not the scalar sum of all 16 values, which is 71. The collective
adds equal *positions within the shards*. Its result is replicated across
the mesh axis `i`, so `P(None)` exposes one logical vector of length 4
with four physical replicas.

```python
expected_columns = np.asarray(x).reshape(4, 4).sum(axis=0)
np.testing.assert_array_equal(y, [22, 20, 12, 17])
np.testing.assert_array_equal(y, expected_columns)
assert y.shape == (4,) and len(y.addressable_shards) == 4
for shard in y.addressable_shards:
    np.testing.assert_array_equal(shard.data, expected_columns)

single_mesh = jax.make_mesh((1,), ("single",), devices=devices[:1])
with jax.set_mesh(single_mesh):
    single_device_input = jax.device_put(np.asarray(x), devices[0])
    row_sums = jax.vmap(lambda row: row.sum())(single_device_input.reshape(4, 4))
np.testing.assert_array_equal(row_sums, [9, 22, 21, 19])
assert len(row_sums.sharding.device_set) == 1
assert not np.array_equal(row_sums, y)
```

For this comparison the input and mesh context explicitly use CPU device 0;
the result is four independent row sums there, not the collective's column
sum. Without that placement, the ambient mesh set by the source example can
give newly created arrays a four-device replicated sharding. Moving just the
input to one CPU while leaving the four-device context active is also
incompatible with this computation in the tested version. That replication
still does not turn a row sum into a cross-shard column reduction.
A distributed mean gradient
likewise needs the intended reduction and denominator; mapping a gradient
function does not choose those semantics. The
[collectives chapter](/textbook/pretraining/processes-collectives-ddp/) derives
that distinction for training.

The JAX checks here use JAX/JAXlib 0.12.0.1 on four logical CPU devices in
one process. They establish numerical results, cache-specialization
observations, state and ownership behavior, placement errors and actual CPU
all-reduce semantics. They do not establish GPU performance, network cost,
multi-host correctness, or equivalence of arbitrary JAX and PyTorch programs.
For timing, asynchronous dispatch also requires a completed result: warm up
the desired specialization, then call `.block_until_ready()` inside the
measurement. Returning a `jax.Array` alone does not prove device completion.

<!-- END REVIEW: JAX runtime 2026-10-27 -->

<!-- BEGIN REVIEW: CUDA Graphs 2026-10-26 -->

<a id="cuda-graphs-capture-replay"></a>

## CUDA Graphs: capture and replay

Fusion changes what happens inside a kernel and which intermediate tensors
must be materialized. It does not remove the host-side work needed to submit
the remaining kernels. A GPU can finish a short kernel before the CPU has
prepared the next launch. In that case the empty interval in the GPU timeline
comes from the submission path, not from the arithmetic in the kernel.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/cuda-graphs-review-2026-10/launch-vs-replay.png|Five individual CPU launches and the same five GPU kernels submitted as one CUDA graph]]

*Vinh Nguyen, Michael Carilli et al.,
[Accelerating PyTorch with CUDA Graphs, Figure 1](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/),
2021. [Original PNG at the pinned website revision](https://github.com/pytorch/pytorch.github.io/blob/d1050dad1fe8ff2b28e1672d6b9f3eaa33e91320/assets/images/cuda-image-2.png),
[BSD-3-Clause notice](/licenses/pytorch-website-d1050dad-BSD-3-Clause.txt).
Copied unchanged. A–E remain five kernels in both rows: replay is not kernel
fusion. The drawing is a schematic, not a measured time scale.*

The following explanation and API example are the original English from
PyTorch 2.14.1, *CUDA semantics*, sections “CUDA Graphs”, “Why CUDA Graphs?”
and “PyTorch API”. Documentation roles and internal references are rendered
as ordinary code and links; the wording and the example are retained.

<!-- BEGIN SOURCE: PyTorch CUDA Graphs overview -->

A CUDA graph is a record of the work (mostly kernels and their arguments) that a
CUDA stream and its dependent streams perform.
For general principles and details on the underlying CUDA API, see
[Getting Started with CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/) and the
[Graphs section](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs) of the CUDA C Programming Guide.

PyTorch supports the construction of CUDA graphs using [stream capture](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#creating-a-graph-using-stream-capture), which puts a
CUDA stream in *capture mode*. CUDA work issued to a capturing stream doesn't actually
run on the GPU. Instead, the work is recorded in a graph.

After capture, the graph can be *launched* to run the GPU work as many times as needed.
Each replay runs the same kernels with the same arguments. For pointer arguments this
means the same memory addresses are used.
By filling input memory with new data (e.g., from a new batch) before each replay,
you can rerun the same work on new data.

### Why CUDA Graphs?

Replaying a graph sacrifices the dynamic flexibility of typical eager execution in exchange for
**greatly reduced CPU overhead**. A graph's arguments and kernels are fixed, so a graph replay
skips all layers of argument setup and kernel dispatch, including Python, C++, and CUDA driver
overheads. Under the hood, a replay submits the entire graph's work to the GPU with
a single call to [cudaGraphLaunch](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1g1accfe1da0c605a577c22d9751a09597).  Kernels in a replay also execute slightly faster
on the GPU, but eliding CPU overhead is the main benefit.

You should try CUDA graphs if all or part of your network is graph-safe (usually this means
static shapes and static control flow, but see the other [constraints](https://docs.pytorch.org/docs/2.14/notes/cuda.html#constraints))
and you suspect its runtime is at least somewhat CPU-limited.


### PyTorch API

> [!warning] Original API notice
> This API is in beta and may change in future releases.

PyTorch exposes graphs via a raw `torch.cuda.CUDAGraph` class
and two convenience wrappers,
`torch.cuda.graph` and
`torch.cuda.make_graphed_callables`.

`torch.cuda.graph` is a simple, versatile context manager that
captures CUDA work in its context.
Before capture, warm up the workload to be captured by running
a few eager iterations. Warmup must occur on a side stream.
Because the graph reads from and writes to the same memory addresses in every
replay, you must maintain long-lived references to tensors that hold
input and output data during capture.
To run the graph on new input data, copy new data to the capture's input tensor(s),
replay the graph, then read the new output from the capture's output tensor(s).
Example:

```python
g = torch.cuda.CUDAGraph()

# Placeholder input used for capture
static_input = torch.empty((5,), device="cuda")

# Warmup before capture
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        static_output = static_input * 2
torch.cuda.current_stream().wait_stream(s)

# Captures the graph
# To allow capture, automatically sets a side stream as the current stream in the context
with torch.cuda.graph(g):
    static_output = static_input * 2

# Fills the graph's input memory with new data to compute on
static_input.copy_(torch.full((5,), 3, device="cuda"))
g.replay()
# static_output holds the results
print(static_output)  # full of 3 * 2 = 6

# Fills the graph's input memory with more data to compute on
static_input.copy_(torch.full((5,), 4, device="cuda"))
g.replay()
print(static_output)  # full of 4 * 2 = 8
```

<!-- END SOURCE: PyTorch CUDA Graphs overview -->

*PyTorch contributors,
[CUDA semantics, source revision for v2.14.1](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/notes/cuda.md#cuda-graphs),
[license and copyright notices](/licenses/pytorch-v2.14.1-LICENSE.txt).
The snippet assumes `import torch`, as in the preceding examples. It is
retained as CUDA code, not presented as a GPU run performed for this chapter.*

### Follow the buffers through two replays

During warmup, multiplication runs eagerly on the side stream. The two
`wait_stream` calls order the streams: warmup waits for earlier work on the
current stream, and the current stream waits for warmup before capture.
The graph context then records the multiplication with the addresses of
`static_input` and `static_output`. Capture and warmup are separate steps.

Copying five threes into `static_input` changes its contents while keeping
its storage. Replay writes five sixes into `static_output`. The next `copy_`
puts five fours at the same input address; replay overwrites the same output
buffer with five eights. No Python multiplication is dispatched again.

Replacing `static_input.copy_(new_batch)` with `static_input = new_batch`
changes which tensor the Python name refers to. It does not patch the input
pointer stored in the graph. If the original allocation remains alive, replay
still reads that original allocation. If its lifetime is broken, replay may
access invalid memory or memory reassigned to another tensor. The output has
the same lifetime issue: `saved = static_output` is another reference to a
buffer that the next replay will overwrite. Clone or otherwise consume an
output before a later replay if that output must remain available.

The following CPU-only model makes those two aliasing errors visible. Lists
stand for retained input and output allocations; this is not a CUDA graph
implementation and does not test CUDA memory management.

```python
buffer = [3] * 5
captured_buffer = buffer
output = [0] * 5

def replay_fixed_buffer():
    output[:] = [2 * value for value in captured_buffer]

replay_fixed_buffer()
aliased_result = output
saved_result = output.copy()
assert saved_result == [6] * 5

buffer[:] = [4] * 5
replay_fixed_buffer()
assert output == [8] * 5
assert aliased_result == [8] * 5  # the old result was overwritten
assert saved_result == [6] * 5   # an independent copy survives

buffer = [9] * 5                # rebind the name, not the captured allocation
replay_fixed_buffer()
assert output == [8] * 5        # still reads the retained fours
assert buffer is not captured_buffer
```

### Capture constraints and dynamic workloads

Ordinary Python work is not part of replay. A Python branch chosen during
capture is not reconsidered on each replay, and an update to a Python counter
is not repeated. A CUDA tensor's `.item()` would require a CPU–GPU
synchronization and cannot be used inside ordinary stream capture. Move
necessary host decisions outside the captured region or use a supported
device-side mechanism; suppressing the error does not preserve the program.

In the documented static-shape path, shapes, layouts and addresses must remain
compatible with the captured operations. This does not require identical
input values. CUDA RNG operations are supported under the documented generator
rules; “same graph” does not mean “the same random sample every time”. During
capture, unrelated CUDA work in other threads of the process is also forbidden.
These are capture requirements, not a ban on the application doing other work
between replays.

PyTorch 2.14.1 also documents a specific exception for data-dependent control:
GPU predicates expressed through `torch.cond` can use CUDA graph conditional
nodes. Its example uses `torch.compile(..., backend="cudagraphs")`; the same
source states that this path is not yet supported by the Inductor backend.
This does not make an arbitrary Python `if` capturable. The versioned
[Data Dependent Control Flow section](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/notes/cuda.md#data-dependent-control-flow)
contains the complete CUDA example and the stated backend boundary.

For a partly dynamic model, `make_graphed_callables` can wrap graph-safe
functions or modules while the surrounding control runs eagerly. The returned
callables are autograd-aware and manage their static buffers and warmup.
Capturing a forward pass alone with the raw API does not automatically capture
a backward pass or an optimizer step. Conversely, the official
[whole-network example](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/notes/cuda.md#whole-network-capture)
captures forward, backward and SGD together. Its warmup already updates the
model, so the documentation explicitly requires real batches for that warmup.

### Memory pools are part of the cost

<!-- BEGIN SOURCE: PyTorch graph memory management -->

A captured graph acts on the same virtual addresses every time it replays.
If PyTorch frees the memory, a later replay can hit an illegal memory access.
If PyTorch reassigns the memory to new tensors, the replay can corrupt the values
seen by those tensors.  Therefore, the virtual addresses used by the graph must be
reserved for the graph across replays. The PyTorch caching allocator achieves this
by detecting when capture is underway and satisfying the capture's allocations
from a graph-private memory pool. The private pool stays alive until its
`torch.cuda.CUDAGraph` object and all tensors created during capture
go out of scope.

Private pools are maintained automatically. By default, the allocator creates a
separate private pool for each capture. If you capture multiple graphs,
this conservative approach ensures graph replays never corrupt each other's values,
but sometimes needlessly wastes memory.

<!-- END SOURCE: PyTorch graph memory management -->

*Original English: PyTorch contributors,
[Graph memory management, v2.14.1 source](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/notes/cuda.md#graph-memory-management),
[license](/licenses/pytorch-v2.14.1-LICENSE.txt). Only documentation-role markup
has been changed to inline code.*

Serving engines can keep several captured shapes and select a compatible
graph, padding and masking the unused work where the model permits it.
This trades repeated setup for retained buffers, extra work on padded entries,
and a finite set of supported shapes. A new shape does not become compatible
by giving its tensor the same Python variable name. Prefill is not inherently
excluded, and MoE is not inherently excluded: the question is whether the
chosen implementation has a capturable schedule for that case. The serving
engine must still handle uncaptured cases correctly.

The versioned [pool-sharing section](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/notes/cuda.md#sharing-memory-across-captures)
explains when several captures can share storage. Concurrent replays and
dependencies between graph outputs affect whether sharing is safe. In
particular, one replay can overwrite another graph's output in a shared pool.
Saving memory by sharing a pool is not permission to run those graphs in an
arbitrary order or concurrently.

### Read an actual before-and-after trace

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/cuda-graphs-review-2026-10/mask-rcnn-before.png|Mask R-CNN backbone before graph capture: CPU submissions and gaps between GPU kernels]]

*Vinh Nguyen, Michael Carilli et al.,
[Accelerating PyTorch with CUDA Graphs, Figure 3](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/),
2021: Mask R-CNN backbone, per-GPU batch size 1. The original range reads
31.662 ms. [Unchanged source PNG](https://github.com/pytorch/pytorch.github.io/blob/d1050dad1fe8ff2b28e1672d6b9f3eaa33e91320/assets/images/cuda-image-4.png),
[BSD-3-Clause notice](/licenses/pytorch-website-d1050dad-BSD-3-Clause.txt).*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/cuda-graphs-review-2026-10/mask-rcnn-after.png|Mask R-CNN backbone after graph capture: one graph launch and densely scheduled GPU kernels]]

*Same authors and article, Figure 4. The captured range reads 6.078 ms.
[Unchanged source PNG](https://github.com/pytorch/pytorch.github.io/blob/d1050dad1fe8ff2b28e1672d6b9f3eaa33e91320/assets/images/cuda-image-5.png),
[BSD-3-Clause notice](/licenses/pytorch-website-d1050dad-BSD-3-Clause.txt).
The two screenshots have different scales and layouts; compare their labelled
durations, not their widths on this page.*

Before capture, the GPU rows contain gaps while the host repeatedly submits
operations. After capture, the graph launch is followed by a denser sequence
of GPU kernels. The green `cudaEventSynchronize` interval in the second trace
is a host wait for completion, not additional GPU arithmetic. The authors
report about 5× for the captured backbone and about 1.7× for the whole model.
Those are different measurement boundaries. The screenshots do not establish
the same speedup for a transformer, another batch size or another machine.

### Account for overlap and capture amortization

CPU submission and GPU execution can overlap. Adding every host launch cost
to every device kernel time overestimates a pipelined eager execution.
Consider five dependent kernels, each requiring 2 time units on the GPU,
while each CPU submission takes 10 units. Kernel 1 is ready at time 10 and
finishes at 12; the next ones finish at 22, 32, 42 and 52. A single graph
submission taking 10 units followed by the same five kernels finishes at 20
in this simplified model. The GPU has not performed fewer kernels.

This CPU model assumes one ordered GPU stream, a serialized host submission
path, no copies and no additional per-node graph overhead. The time units are
illustrative, not measurements of CUDA:

```python
def launch_bound_finish(kernels, launch_time, kernel_time):
    cpu_ready = gpu_done = 0
    for _ in range(kernels):
        cpu_ready += launch_time
        gpu_done = max(gpu_done, cpu_ready) + kernel_time
    return gpu_done

eager_time = launch_bound_finish(5, 10, 2)
replay_time = 10 + 5 * 2
capture_time = 200
assert eager_time == 52 and replay_time == 20

# Six iterations do not repay capture; seven do.
assert capture_time + 6 * replay_time > 6 * eager_time
assert capture_time + 7 * replay_time < 7 * eager_time

# When long kernels already hide submission, this model predicts no gain.
assert launch_bound_finish(5, 2, 10) == 2 + 5 * 10
```

With setup cost $C$, steady eager time $T_e$ and replay time $T_g$,
capture pays for itself after $N$ iterations only when

$$
C+NT_g<NT_e.
$$

The example has $C=200$ and $T_e-T_g=32$, so the first integer satisfying
the inequality is 7. If fresh-input copies or other required work make
$T_g\ge T_e$, increasing $N$ cannot repay the setup in this model. A real
comparison therefore measures the whole input-update/replay/output-consumption
path as well as the graph's device time. It also reports warmup and capture
separately. A graph replay does not repair a slow data loader, an inefficient
kernel, a communication dependency or a memory-capacity limit.

<!-- END REVIEW: CUDA Graphs 2026-10-26 -->

<a id="как-читать-профиль-triton-kernel"></a>

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
модели не изменился, нужно вернуться к профилю S7: возможно, устранённая
работа не лежала на критическом пути.

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

Конвейер S6 определяет, какие
данные и формы получает модель; профиль S7 показывает, где тратится время;
разбиение и ядра S8 меняют цену конкретных вычислений. Следующий вопрос — как
из этих вычислений складывается обучение языковой модели на корпусе:
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

← [[02 Areas/ML & DL/00 Учебник/10 ML Systems/07 Profiling ML-нагрузки|Profiling ML-нагрузки]] ·
[[02 Areas/ML & DL/00 Учебник/_index#Углублённые модули|Выбрать углублённый модуль]] →
