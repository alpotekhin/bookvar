---
title: "Развёртывание на периферийных устройствах и федеративное обучение"
type: textbook-chapter
status: canonical
last_updated: 2026-10-27
last_verified: 2026-10-27
primary_sources:
  - https://arxiv.org/abs/2007.11622v5
  - https://proceedings.mlr.press/v54/mcmahan17a.html
  - https://developers.google.com/edge/litert/conversion/tensorflow/build/ondevice_training
  - https://mlsysbook.ai/vol2/edge_intelligence/edge_intelligence.html
  - https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
---

# Развёртывание на периферийных устройствах и федеративное обучение

Клавиатура телефона предсказывает следующее слово по уже набранному тексту.
Общая языковая модель может хорошо знать язык и при этом плохо предсказывать
имена, сокращения и обороты конкретного пользователя. Часть этой информации
можно передать модели как контекст. Другой способ адаптации состоит в том,
чтобы обновлять её параметры на примерах с устройства. Если такие обновления
полезны и другим пользователям, сервер может объединять результаты локального
обучения, не собирая сами сообщения.

В этом примере отдельно выбираются место инференса, место обучения и способ
обмена результатами. Установленная на телефоне модель может годами выполнять
только прямой проход. Локальный адаптер может оставаться личным и никогда не
передаваться на сервер. При федеративном обучении устройства участвуют в
обучении общей модели; отдельная персонализация поверх неё возможна, но не
является обязательной частью алгоритма.

Предполагается знакомство с обратным распространением, SGD и Adam, а также с
[[02 MLOps|версионированием и выпуском моделей]]. Ограничение размера самой
модели разобрано в главе
[[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/57b Сжатие моделей — pruning, distillation и low-rank|о сжатии моделей]].
Здесь к памяти инференса добавятся состояние обучения, локальные данные и
обмен между устройствами.

<a id="сначала-выбрать-где-находится-вычисление-и-состояние"></a>

## Локальный инференс, персонализация и совместное обучение

Допустим, приложение распознаёт предметы на фотографиях. При облачном
инференсе оно отправляет изображение серверу и получает ответ. При локальном
инференсе изображение обрабатывает установленная модель. Второй вариант
позволяет работать без сети, но требует, чтобы модель, её операции и рабочие
буферы помещались на поддерживаемом устройстве.

Ни один вариант сам по себе не требует обучения в приложении. Можно
сохранить пользовательские названия предметов, историю или несколько
эталонных векторов признаков и сравнивать с ними новые изображения.
Если такого представления достаточно для задачи, менять веса необязательно.
Потребность в градиентном обучении проверяют на отложенных локальных данных,
сравнивая его с этим более простым вариантом.

| Вариант | Что делает устройство | Что передаётся серверу |
|---|---|---|
| Облачный инференс | Готовит вход и показывает ответ | Входные данные |
| Локальный инференс | Выполняет прямой проход неизменной модели | Для самого инференса обмен не нужен |
| Персонализация без градиентов | Хранит контекст, предпочтения или эталонные признаки | Обмен определяется приложением |
| Локальное обучение | Обновляет голову, адаптер или другие параметры | Может ничего не передавать |
| Федеративное обучение | Обучается на локальных примерах по общему протоколу | Обновление модели и предусмотренные протоколом метаданные |

Федеративное обучение (*federated learning*, FL) определяет, как обучать общую
модель на распределённых данных. Это не требование запускать модель на
телефоне: участниками могут быть организации с собственными серверами.
Отсутствие передачи исходных примеров также не означает, что передаваемые
обновления безопасно раскрывать. Веса, градиенты и факт участия могут сообщать
информацию о данных. Защита обновлений рассматривается отдельно от алгоритма
усреднения.

## Ресурсная модель периферийного устройства

При инференсе нужны веса, входы и временные буферы операций. При обучении
нужно ещё вычислить градиенты и обновить параметры. Состояние оптимизатора
сохраняется между шагами, а часть результатов прямого прохода нужна для
обратного распространения.

Для полного дообучения с FP32-весами, FP32-градиентами и двумя FP32-массивами
моментов Adam четыре массива требуют $16P$ байт для $P$ параметров:

$$
\begin{aligned}
M_{\text{train}}&=M_W+M_G+M_{m_1}+M_{m_2}\\
&\quad+M_A+M_{\text{workspace}}\\
&\quad+M_{\text{application}}.
\end{aligned}
$$

При $P=10^7$ один массив занимает 40 MB, четыре массива занимают 160 MB
(около 152.6 MiB). Это ещё не пиковая память процесса: не учтены активации
$M_A$, рабочая память операций и остальное приложение. Если реализация
хранит дополнительную рабочую копию весов в другом формате, её тоже прибавляют.
Автоматическое приведение типов отдельных операций не доказывает наличие
такой копии.

<!-- BEGIN REVIEW: Harvard mixed-state memory 2026-10-26 -->
<a id="harvard-mixed-state-memory"></a>
### FP16 и FP32 в одном состоянии обучения

The Harvard Edge Intelligence lab uses another accounting convention:
FP16 weights and gradients, with two FP32 Adam moments. Its complete
“Math Peek: Training Memory Amplification” explanation is reproduced below.

<!-- BEGIN SOURCE: Harvard training memory Math Peek -->
> **Formula:**
>
> $$
> \begin{aligned}
> M_{\text{train}} &= M_{\text{weights}} + M_{\text{grad}}\\
> &\quad+ M_{\text{opt}} + M_{\text{act}}
> \end{aligned}
> $$
>
> **Where:**
>
> - **$M_{\text{weights}}$**: Model parameters in FP16 = $P \times 2$ bytes
> - **$M_{\text{grad}}$**: Gradient storage (same size as weights) = $P \times 2$ bytes
> - **$M_{\text{opt}}$**: Adam stores momentum + variance in FP32 = $P \times 4 \times 2$ bytes
> - **$M_{\text{act}}$**: Activations cached for backward pass, scales with batch size $B$
>
> **Amplification factor** over inference ($M_{\text{inf}} = P \times 2$):
>
> $$
> \begin{aligned}
> \text{Amplification}
> &= \frac{M_{\text{train}}}{M_{\text{inf}}}\\
> &= 1 + 1 + 4 + \frac{M_{\text{act}}}{P \times 2}\\
> &\approx 5\text{-}9\times
> \end{aligned}
> $$
<!-- END SOURCE: Harvard training memory Math Peek -->

*Harvard Edge ML Systems Book contributors,
[Lab 11, Math Peek: Training Memory Amplification](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/labs/vol2/lab_11_edge_intelligence.py#L881-L901),
CC BY-NC-SA 4.0. Original English; the equations are split across lines.
The complete Python source is retained in
[[05 Источники/Courses/Harvard ML Systems/labs/vol2/lab_11_edge_intelligence|the local lab archive]].*

The lower endpoint in the quoted approximation does not follow from the
listed arrays: $1+1+4=6$ before any activation storage. For these dtypes
the parameter-related subtotal is $12P$ bytes. The quoted “5–9×” is retained
as an issue in the source, not used as a general bound.

For the same ten million parameters used above, the individual arrays are:

| State | Bytes per parameter | Size at $P=10^7$ |
|---|---:|---:|
| FP16 weights | 2 | 20 MB |
| FP16 gradients | 2 | 20 MB |
| Two FP32 Adam moments | 8 | 80 MB |
| Parameter-related subtotal | 12 | 120 MB |

With 32 MB of saved activations, the subtotal becomes 152 MB. Relative to
the 20 MB of weights alone, this is $152/20=7.6$ times as much memory.
If an implementation additionally keeps an FP32 master copy of the weights,
that copy adds 40 MB: the subtotal is then 192 MB, or $9.6$ times the
weights-only baseline. Workspaces, inputs and the rest of the application
still have to be added. These are explicit accounting examples, not
measurements of the Marimo simulator or a phone.

The earlier 160 MB figure assumes all four parameter-related arrays are
FP32. The 120 MB figure assumes two FP16 arrays and two FP32 arrays.
Neither budget includes activations. To compare two training modes, the
dtype and presence of each array must be specified; the label “FP16
training” alone does not determine the memory footprint.
<!-- END REVIEW: Harvard mixed-state memory 2026-10-26 -->

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/edge-review-2026-10/tinytl-memory-figure1.png]]

*Han Cai, Chuang Gan, Ligeng Zhu, Song Han,
[TinyTL, Figure 1, PDF p. 2](https://arxiv.org/pdf/2007.11622v5#page=2),
CC BY 4.0. Перенесён рисунок без перерисовки; удалены только поля страницы.
Слева сравниваются инференс с batch size 1 и обучение с batch size 16;
такое сравнение не задаёт отношение памяти при одинаковом размере батча.
Справа обе сети сравниваются при batch size 16.*

В правой части рисунка MobileNetV2-1.4 имеет в 4.3 раза меньше памяти
параметров, чем ResNet-50, но память активаций отличается лишь в 1.1 раза.
В такой конфигурации уменьшение числа весов почти не устраняет основную
статью расходов при обучении. Это наблюдение относится к показанным сетям и
режиму измерения, а не ко всем архитектурам.

Для замороженного извлекателя признаков и обучаемой головы бюджет другой.
Голова, переводящая вектор из 128 признаков в 10 классов, имеет
$128\cdot10+10=1290$ параметров. Её четыре FP32-массива занимают 20 640 байт.
Однако 40 MB замороженной основы остаются в памяти, а прямой проход основы
по-прежнему требует рабочих буферов. Экономия касается состояния, необходимого
для обучения изменяемой части.

Можно заранее вычислить признаки, если основа не меняется. Тысяча векторов
длины 128 в FP32 занимает $1000\cdot128\cdot4=512\,000$ байт. Это позволяет
повторно обучать голову без повторного прохода по основе. После смены основы
старые признаки могут стать несовместимыми, поэтому такой кэш связывают с
версией извлекателя, предобработки и схемы признаков.

Кроме памяти измеряют время шага, энергию, нагрев и задержку активного
инференса. Наличие ускорителя для прямого прохода не гарантирует поддержку
обратного: часть операций может выполняться на CPU или требовать другого
формата модели. Условия «простой, зарядка, нетарифицируемая сеть» ограничивают
влияние фонового обучения на пользователя. Это политика конкретного
приложения, а не свойство FL. Если приложение разрешает обучение только при
таких условиях, далее нужно учитывать, кто из пользователей под них попадает.

<!-- BEGIN REVIEW: Micro arena 2026-10-27 -->
## Microcontroller inference: planning a tensor arena

A model can fit in Flash while its intermediate tensors exceed the available
SRAM. The Harvard Smart Doorbell example assigns its keyword-spotting model
an ARM Cortex-M4 target with 256 KB of RAM and 1 MB of Flash. These are the
chosen scenario's limits, not specifications shared by all Cortex-M4 devices.
Its three framework decisions are reproduced below, with index markup removed:

> 1. **Fixed memory arena**: The application supplies a contiguous tensor arena, and the framework plans and reuses buffers from that arena rather than relying on dynamic allocation during inference.
> 2. **Kernel selection**: Only the specific kernels used by the model (for example, Conv2D, DepthwiseConv) need to be linked or registered with the runtime.
> 3. **Compact interpreter execution**: The MCU runs a small C/C++ interpreter over a flat model representation, with the model and arena bound at initialization rather than assembled dynamically at runtime.

*Vijay Janapa Reddi and Harvard ML Systems contributors,
[Frameworks for the edge: TinyML and micro-runtimes](https://github.com/harvard-edge/cs249r_book/blob/2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd/book/quarto/contents/vol1/frameworks/frameworks.qmd#L1893),
CC BY-NC-SA 4.0. Original English; the source's unversioned Python/PyTorch
footprint estimates are not used as deployment measurements.*

TensorFlow Lite Micro (TFLM) still interprets a serialized graph and dispatches
operator kernels. Supplying a fixed buffer does not turn the graph into a
fully ahead-of-time compiled C program. The application owns the arena, while
the interpreter plans allocations inside it. Stack, application state and the
model buffer require separate accounting.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/micro-runtime-2026-10-27/model-buffer-and-arena.png]]

*The TensorFlow Authors,
[Pre-allocated tensors RFC, Background](https://github.com/tensorflow/tflite-micro/blob/1cc91a97b40e1dddcecd3b0fbb30b6643e52f164/tensorflow/lite/micro/docs/rfc/001_preallocated_tensors.md#background),
[Apache 2.0](/licenses/tflm-Apache-2.0.txt). Complete original 271×229 PNG,
shown at native size. This is the existing model-buffer/arena distinction,
not the RFC's proposed external-buffer API. The RFC says that proposal's
expected benefits did not justify its added complexity.*

Constant weight data can point into the serialized model buffer. Keeping that
buffer in Flash or ROM depends on the target's storage and linker arrangement;
the figure does not establish that every deployment avoids a RAM copy.
Runtime tensor descriptors, operator state and any mutable tensors still need
memory. Therefore neither the .tflite file size nor the weight bytes alone
give the required arena size.

The original TFLM memory-management diagram divides the arena into three
regions:

```text
--------------------------------------------------------------------------------
|        |                     |                                               |
|  HEAD  |<--  TEMPORARY    -->|                    TAIL                       |
|        |                     |                                               |
--------------------------------------------------------------------------------
* Lowest Address                                               Highest Address *
```

*The TensorFlow Authors,
[Memory Management in TensorFlow Lite Micro](https://github.com/tensorflow/tflite-micro/blob/1cc91a97b40e1dddcecd3b0fbb30b6643e52f164/tensorflow/lite/micro/docs/memory_management.md#tensor-arena),
Apache 2.0. Original text diagram, unchanged.*

The **head** holds planned non-persistent tensor buffers and operator scratch
space. The planner can assign the same bytes to buffers whose lifetimes do
not overlap. The **temporary** region provides scoped allocations, including
structures needed while preparing operators; reset returns its allocation
position toward the head. The **tail** grows from the other end and holds
persistent runtime structures and state. Reusing head storage does not free
the tail, and a tensor's descriptor can remain after its temporary data is
no longer needed.

### Lifetimes determine which buffers can share storage

Consider an assigned out-of-place execution plan with three operators, numbered
0–2, followed by the application reading the result at step 3. All sizes below
are in KiB, where 1 KiB is 1024 bytes. A lifetime includes both endpoints:
an operator may still read its input while producing its output.

| Buffer | KiB | Steps | Offset, KiB |
|---|---:|---|---:|
| `input` | 8 | 0 | 12 |
| `a` | 12 | 0–1 | 0 |
| `b` | 8 | 1–2 | 12 |
| `output` | 4 | 2–3 | 0 |

Separate buffers would occupy $8+12+8+4=32$ KiB. At step 0, `input` and `a`
need 20 KiB; at step 1, `a` and `b` also need 20 KiB. At step 2, `b` and
`output` need 12 KiB. Thus 20 KiB is a lower bound on this plan's tensor-data
storage. The listed offsets attain it: `b` reuses the old input's bytes after
step 0, and `output` reuses the beginning of `a` after step 1.

If another branch reads the original input at step 2, its lifetime extends
through that step. At step 1, `input`, `a` and `b` then coexist and require
28 KiB. The old assignment of `input` and `b` to offset 12 becomes invalid.
Their sizes have not changed; the last-use dependency has.

The following checks a supplied layout. It does not search for an optimal
layout or implement TFLM's GreedyMemoryPlanner. It rejects two buffers that
overlap in both lifetime and storage, and returns the end of the allocated
head. In-place kernel aliases would require a different, explicitly allowed
contract; they are excluded from this example.

```python
KiB = 1024

def check_arena_plan(buffers, offsets, alignment=16):
    if not buffers or set(buffers) != set(offsets):
        raise ValueError("Each nonempty buffer set needs exactly its offsets")
    if type(alignment) is not int or alignment <= 0:
        raise ValueError("Alignment must be a positive integer")
    for name, (size, first, last) in buffers.items():
        offset = offsets[name]
        if (any(type(v) is not int for v in (size, first, last, offset))
                or size <= 0 or first < 0 or last < first or offset < 0
                or offset % alignment):
            raise ValueError("Invalid size, lifetime or aligned offset")
    names = list(buffers)
    for i, left in enumerate(names):
        size_l, first_l, last_l = buffers[left]
        for right in names[i + 1:]:
            size_r, first_r, last_r = buffers[right]
            live_overlap = max(first_l, first_r) <= min(last_l, last_r)
            byte_overlap = max(offsets[left], offsets[right]) < min(
                offsets[left] + size_l, offsets[right] + size_r)
            if live_overlap and byte_overlap:
                raise ValueError(f"Concurrent buffers overlap: {left}, {right}")
    return max(offsets[name] + values[0] for name, values in buffers.items())

buffers = {"input": (8*KiB, 0, 0), "a": (12*KiB, 0, 1),
           "b": (8*KiB, 1, 2), "output": (4*KiB, 2, 3)}
offsets = {"input": 12*KiB, "a": 0, "b": 12*KiB, "output": 0}
assert check_arena_plan(buffers, offsets) == 20*KiB
branch_buffers = dict(buffers, input=(8*KiB, 0, 2))
try:
    check_arena_plan(branch_buffers, offsets)
except ValueError:
    pass
else:
    raise AssertionError("A still-live input cannot be overwritten by b")
branch_offsets = dict(offsets, b=20*KiB)
assert check_arena_plan(branch_buffers, branch_offsets) == 28*KiB
```

This dependence on simultaneously live tensors is also shown in the source
RFC's complete six-layer memory-plan example. A greedy placement is not
guaranteed to attain the live-byte lower bound for every graph: alignment,
fragmentation, kernel scratch buffers and allowed aliases affect the result.

### Initialization peak, recording output and whole-device RAM

The required arena must fit during preparation as well as invocation. In an
assigned phase ledger, suppose setup uses 28 KiB of temporary storage plus
4 KiB of persistent tail, while invocation uses a 20 KiB head plus the same
4 KiB tail. The two phase totals are 32 and 24 KiB. Taking the maximum gives
32 KiB; adding both phases' non-persistent storage would instead count bytes
that need not coexist. Conversely, using the 24 KiB invocation snapshot alone
would miss this example's setup peak. These numbers are chosen bookkeeping
inputs, not a recording from a microcontroller or a universal TFLM allocation
sequence.

For a separate chosen 256 KiB SRAM budget, reserve 64 KiB for application
state, 32 KiB for sensor/I/O buffers and 16 KiB for stack. A 32 KiB arena then
leaves 112 KiB unassigned. Code, constants, interrupt stacks, DMA constraints,
alignment and any copied model data must be accounted for in the actual
linker/runtime configuration; the remainder is not measured free memory.

```python
phases = {"prepare": {"head": 0, "temporary": 28*KiB, "tail": 4*KiB},
          "invoke": {"head": 20*KiB, "temporary": 0, "tail": 4*KiB}}
phase_totals = {name: sum(parts.values()) for name, parts in phases.items()}
assert phase_totals == {"prepare": 32*KiB, "invoke": 24*KiB}
arena_capacity = max(phase_totals.values())
sram = 256*KiB
other_ram = 64*KiB + 32*KiB + 16*KiB
assert sram - arena_capacity - other_ram == 112*KiB
assert phase_totals["invoke"] < arena_capacity
```

TFLM's documentation supplies a recording interpreter for inspecting real
arena allocations. Both original C++ fragments are retained below. Model
data, resolver, headers and build setup are omitted. For portable C++, the
array bound also needs a compile-time constant, such as `constexpr size_t`;
alignment and actual capacity must match the target. No C++ build or board
execution is claimed here.

```cpp
// Buffer for the tensor arena:
size_t tensor_arena_size = 2048;
uint8_t tensor_arena[tensor_arena_size];

// Interpreter using the shared tensor arena above:
tflite::MicroInterpreter interpreter(
  tflite::GetModel(my_model_data), ops_resolver,
  tensor_arena, tensor_arena_size);

// Invoke one time which will allocate internals:
if (interpreter.Invoke() != kTfLiteOk) {
  MicroPrintf("Exception during invoke()!");
}
```

```cpp
// Add an include to the recording API:
#include "recording_micro_interpreter.h"

// Simply change the class name from 'MicroInterpreter' to 'RecordingMicroInterpreter':
tflite::RecordingMicroInterpreter interpreter(
  tflite::GetModel(my_model_data), ops_resolver,
  tensor_arena, tensor_arena_size);

// Invoke one time which will allocate internals:
if (interpreter.Invoke() != kTfLiteOk) {
  MicroPrintf("Exception during invoke()!");
}

// Print out detailed allocation information:
interpreter.GetMicroAllocator().PrintAllocations();
```

*The TensorFlow Authors,
[Recording Memory APIs](https://github.com/tensorflow/tflite-micro/blob/1cc91a97b40e1dddcecd3b0fbb30b6643e52f164/tensorflow/lite/micro/docs/memory_management.md#recording-memory-apis),
Apache 2.0. Complete original fragments, unchanged apart from the Markdown
language label. Recording itself needs extra memory and working debug logging.*

The documentation's separate threshold-test output reports a head of 7744
bytes and a tail of 1824 bytes, totaling 9568 bytes. That output is not a
result of the small 2048-byte setup shown above, nor a measurement of the KWS
scenario. An allocation report also does not include all process/device RAM.
Preparation, repeated invocation and the actual application must all succeed
with the chosen arena and board configuration.

### A registered operator is not every possible kernel configuration

TFLM's `MicroMutableOpResolver` stores a bounded set of registrations. Registering
the needed operation types permits their lookup without including an unrelated
full framework runtime. It does not verify that a selected kernel supports
every dtype, shape or quantization configuration. The
[porting guide's complete FAQ](https://github.com/tensorflow/tflite-micro/blob/1cc91a97b40e1dddcecd3b0fbb30b6643e52f164/tensorflow/lite/micro/docs/porting_reference_ops.md#frequently-asked-questions)
distinguishes preparation-time scratch requests from invocation-time use,
disallows dynamic tensor storage, and explains reference-kernel fallbacks
for configurations an optimized kernel does not support. A missing operation
registration is not automatically repaired by such a fallback.

Conversion success, successful arena allocation and correct inference are
separate checks. Validate supported shapes and dtypes, operator preparation,
input/output quantization and reference-output tolerances on the target build.
The CPU layout checker above establishes only its supplied buffer plans.
This TFLM inference setup also differs from the LiteRT on-device-training
signatures later in this chapter; neither runtime name licenses a blanket
claim that every edge deployment supports, or forbids, training.

<!-- END REVIEW: Micro arena 2026-10-27 -->

<a id="ограничить-изменяемое-состояние"></a>

## Почему замороженные веса могут уменьшить память обратного прохода

TinyTL начинает объяснение с производных линейного слоя. Для одного объекта
пусть строка $a$ является входом, $W$ — матрицей весов, $b$ — смещением,
$z=aW+b$, а $g=\partial L/\partial z$ — градиентом, пришедшим от следующего
слоя. Тогда

$$
\begin{gathered}
\frac{\partial L}{\partial a}=gW^\top,\\
\frac{\partial L}{\partial W}=a^\top g,\\
\frac{\partial L}{\partial b}=g.
\end{gathered}
$$

Вход $a$ нужен для вычисления градиента весов. Для градиента смещения он не
нужен. Если $W$ заморожена, её градиент не вычисляется, и сохранять полный
$a$ именно для этой операции больше не требуется. Градиент по входу при этом
может оставаться необходимым: например, если в предыдущем слое обучается
смещение.

Пусть $a=(2,-1)$, $W=((1,3),(2,4))$, $b=(0.5,-0.5)$ и $g=(0.1,-0.2)$.
Прямой проход даёт $z=(0.5,1.5)$, а обратный:

$$
\begin{gathered}
\frac{\partial L}{\partial a}=(-0.5,-0.6),\\
\frac{\partial L}{\partial W}=
\begin{pmatrix}0.2&-0.4\\-0.1&0.2\end{pmatrix},\\
\frac{\partial L}{\partial b}=(0.1,-0.2).
\end{gathered}
$$

При обучении только $b$ средняя матрица не нужна. Но если за линейным слоем
стоит ReLU, для её производной нужно знать, какие входы были положительными.
Теоретически достаточно битовой маски вместо FP32-активаций. Реальная
среда исполнения может хранить маску иначе или сохранять исходный тензор.
Установка `requires_grad=False` сама по себе не гарантирует, что среда
упакует эту маску по одному биту на элемент.
Для других нелинейностей набор сохраняемых величин также отличается.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/edge-review-2026-10/tinytl-activation-figure2.png]]

*Han Cai et al.,
[TinyTL, Figure 2, PDF p. 4](https://arxiv.org/pdf/2007.11622v5#page=4),
CC BY 4.0. Все три части и легенда сохранены; удалены поля страницы.
Закрашенные синие блоки обозначают сохраняемые карты признаков,
закрашенные жёлтые элементы — обучаемые параметры. Контурные элементы
показывают соответственно несохраняемые карты и замороженные параметры.*

В варианте (a) обучаются все веса мобильного блока; для их градиентов нужны
крупные промежуточные карты. В варианте (b) обучаются смещения. Возможность
изменять представление при этом ограничена. Вариант (c) добавляет обучаемую
остаточную ветвь: она вычисляет поправку к выходу замороженной основной ветви.

В TinyTL вход дополнительной ветви сначала уменьшается по пространственному
разрешению. В ней используется групповая свёртка без большого расширения
числа каналов, характерного для исходного inverted bottleneck. В конце
результат увеличивается до нужного разрешения и прибавляется к основному
выходу. Градиент обучаемых весов дополнительной ветви требует её уменьшенных
активаций. Память основной ветви не исчезает полностью: остаются веса,
данные для её операций и состояния нужных нелинейностей.
Подробный вывод находится в
[TinyTL, разделы 3.1–3.3](https://arxiv.org/pdf/2007.11622v5#page=3).

Голова, обучение смещений, обычные адаптеры и низкоранговые обновления
ограничивают разные части модели. У низкоранговой поправки $W+AB$ мало
обучаемых параметров, но градиент по $A$ или $B$ всё равно может потребовать
входных активаций. Поэтому сравнивают пиковую память полного шага при
одинаковом размере батча, наборе обучаемых слоёв и среде исполнения.
Число обучаемых параметров само по себе не заменяет такое сравнение.

Малый батч влияет и на нормализацию. Batch Normalization в режиме обучения
оценивает статистики по батчу и пространственным позициям, а также обновляет
накопленные оценки. При малом и неоднородном локальном наборе они могут быть
нестабильными. В экспериментах TinyTL применяется Group Normalization:
её статистики вычисляются по группам каналов каждого объекта, без зависимости
от других объектов батча. Это не означает, что замена нормализации сохраняет
качество любой уже обученной модели.

## Локальное обучение должно переживать перезапуск приложения

В руководстве LiteRT обучение на устройстве объясняется через четыре операции:
`train` меняет параметры, `infer` выполняет прямой проход, `save` записывает
изменённые веса, `restore` загружает их. Эти операции экспортируются как
отдельные сигнатуры модели. Исходная формулировка сохранена на английском:

<!-- source-excerpt: litert-signatures -->
> LiteRT models typically have only a single exposed function method (or signature) that allows you to call the model to run an inference. For a model to be trained and used on a device, you must be able to perform several separate operations, including train, infer, save, and restore functions for the model.
<!-- /source-excerpt -->

*Google AI Edge, [On-device training with LiteRT, Build a model for on-device training](https://developers.google.com/edge/litert/conversion/tensorflow/build/ondevice_training#build_a_model_for_on-device_training), CC BY 4.0. Оригинальный текст; без перевода.*

После установки приложение создаёт интерпретатор с исходными весами.
Несколько вызовов `train` меняют веса в оперативной памяти. Завершение
процесса без `save` теряет эти изменения. После сохранения и перезапуска
новый интерпретатор сначала снова содержит исходные веса; продолжить
персонализацию позволяет вызов `restore` до следующего инференса.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/edge-review-2026-10/litert-checkpoint.png]]

*Google AI Edge,
[On-device training with LiteRT, Restore the trained weights](https://developers.google.com/edge/litert/conversion/tensorflow/build/ondevice_training#restore_the_trained_weights),
CC BY 4.0. Оригинальный рисунок скопирован без изменений.
Это логиты одного примера до и после загрузки checkpoint. Изменение логитов
подтверждает изменение поведения, но не доказывает улучшение качества.*

В этом руководстве `save` сохраняет веса модели. Полное продолжение обучения
с Adam дополнительно требует его моментов, счётчика шагов и согласованной
конфигурации. Также отдельно сохраняются версия основы, формат адаптера,
версия предобработки и состояние учёта приватности, если оно ведётся локально.
Произвольный файл с весами не содержит всего этого автоматически.

Сам пример LiteRT обучает небольшой классификатор Fashion-MNIST и для
демонстрации повторного обучения использует те же обучающие данные.
Поэтому его график loss не является экспериментом по персонализации на новых
пользовательских данных. API и поддержку операций нужно проверять для целевой
версии среды исполнения; здесь Android-приложение не запускалось.

При непрерывном обучении возникает ещё одно состояние — локальная выборка.
Если брать только последние наблюдения, модель может забывать прежние условия.
Буфер повторного обучения (*experience replay*) сохраняет ограниченную
выборку старых примеров и смешивает её с новыми. Он не устраняет забывание
автоматически: результат зависит от ёмкости, правила отбора и изменения данных.
Буфер признаков занимает меньше места, но связывает данные с версией
извлекателя. Шифрование, срок хранения, очистка и согласие на использование
данных задаются и для такого буфера.

## Федеративное обучение как протокол

Сервер федеративного обучения хранит общую модель $\theta^t$.
В начале раунда он выбирает клиентов и передаёт им эту версию. Клиенты
обучаются на локальных данных и возвращают параметры либо разности
$u_k=\theta_k-\theta^t$. Сервер объединяет принятые результаты и получает
кандидат следующей версии. Внешний выпуск модели пользователям может
проходить реже, чем внутренние обучающие раунды: каждый раунд не обязан
немедленно становиться пользовательским релизом.

Для клиента $k$ с $n_k$ примерами локальная функция потерь равна

$$
F_k(\theta)=\frac{1}{n_k}\sum_{i=1}^{n_k}\ell(\theta;x_{ki},y_{ki}).
$$

Если требуется одинаково учитывать каждый пример, общая цель имеет вид

$$
F(\theta)=\sum_{k=1}^{K}\frac{n_k}{N}F_k(\theta),
\qquad N=\sum_{k=1}^{K}n_k.
$$

Пользователь с тысячей примеров получает в этой цели больший вес, чем
пользователь с десятью. Если требуется одинаково учитывать каждого
пользователя, задают другую цель, например $\frac{1}{K}\sum_k F_k$.
Обе цели определены содержательно; выбор между ними нельзя заменить словом
«справедливо».

### Взвешенное среднее и выпадение клиента

Рассмотрим вариант FedAvg, в котором сервер нормирует веса по множеству
$S_t$ действительно принятых клиентов:

$$
\begin{gathered}
\alpha_k=\frac{n_k}{\sum_{j\in S_t}n_j},\\
\theta^{t+1}=\sum_{k\in S_t}\alpha_k\theta_k
=\theta^t+\sum_{k\in S_t}\alpha_k u_k.
\end{gathered}
$$

Последнее равенство использует одну базовую версию и $\sum_k\alpha_k=1$.
Это явно выбранное здесь правило работы с принятыми участниками.
При частичном участии распределение выбранных клиентов также влияет на
ожидаемое обновление; нормировка по принятым примерам сама по себе не делает
его несмещённым градиентом всей популяции. В исходной статье FedAvg
глобальная целевая функция и алгоритм заданы отдельно; приведённая формула
не является буквальной копией её псевдокода обработки выпадений.

Пусть два клиента имеют 100 и 300 примеров и вернули $\theta_1=(1,3)$,
$\theta_2=(5,7)$. Тогда веса равны 0.25 и 0.75, а новая модель:

$$
0.25(1,3)+0.75(5,7)=(4,6).
$$

Равные веса клиентов дали бы $(3,5)$. Если второй клиент не прислал результат,
для выбранного правила оставшийся клиент получает вес 1 и результат $(1,3)$.
Если протокол требует нескольких участников для безопасной агрегации, такой
раунд отменяется. Публиковать одиночный результат вместо защищённого агрегата
нельзя без изменения обещанной модели защиты.

```python
import numpy as np

def average_clients(models, counts):
    models = np.asarray(models, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if (models.ndim != 2 or counts.ndim != 1
            or len(models) == 0 or len(models) != len(counts)):
        raise ValueError("Нужны непустые векторы одинаковой длины и их веса")
    if (not np.isfinite(models).all() or not np.isfinite(counts).all()
            or np.any(counts <= 0)):
        raise ValueError("Параметры должны быть конечными, веса — положительными")
    weights = counts / counts.sum()
    if not np.isfinite(weights).all() or weights.sum() == 0:
        raise ValueError("Сумма весов не представима в выбранном формате")
    return weights @ models

np.testing.assert_allclose(average_clients([[1, 3], [5, 7]], [100, 300]), [4, 6])
np.testing.assert_allclose(average_clients([[1, 3]], [100]), [1, 3])
```

Функция иллюстрирует только агрегирование уже проверенных результатов.
Она не аутентифицирует клиентов, не проверяет версии и не обеспечивает
защиту от вредоносных обновлений.

### Общая инициализация и согласованность параметров

У нейросети можно переставить скрытые нейроны вместе с соответствующими
связями и сохранить ту же функцию. Поэтому усреднение двух независимо
обученных сетей по координатам не эквивалентно усреднению их предсказаний.
Одинаковая архитектура и размер массива ещё не означают, что координаты
двух моделей согласованы.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/edge-review-2026-10/fedavg-initialization-figure1.png]]

*H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Agüera y Arcas,
[Communication-Efficient Learning of Deep Networks from Decentralized Data, Figure 1](https://proceedings.mlr.press/v54/mcmahan17a.html),
AISTATS 2017, PMLR 54:1273–1282, CC BY 4.0.
Оригинальные графики из PDF p. 4; удалены поля и окружающий текст.
Шкалы по вертикали различаются.*

В статье две сети обучались на непересекающихся IID-подвыборках MNIST по
600 примеров: 240 шагов SGD, размер батча 50, learning rate 0.1.
Горизонтальная ось задаёт смешивание параметров
$\lambda w+(1-\lambda)w'$; в оригинале коэффициент обозначен $\theta$.
При $\lambda=0$ и $\lambda=1$ получаются исходные модели.
Слева независимая инициализация приводит к большому loss промежуточной модели.
Справа общая инициализация позволяет получить меньшее значение loss
при усреднении. Это конкретный эксперимент, а не гарантия качества среднего
для произвольных сетей.

В FedAvg все участники раунда начинают с общей версии. Неодинаковые
адаптеры тоже нужно согласовать: совпадают основа, размещение обучаемых
модулей, размерности и смысл координат. Адаптер ранга 4 и адаптер ранга 8
нельзя без дополнительного определения просто усреднить как два массива.

### Локальные эпохи и расхождение клиентов

Один локальный шаг и несколько локальных шагов отличаются местом вычисления
градиента. При одном шаге все клиенты используют одну точку $\theta^t$.
Взвешенное среднее полных локальных градиентов совпадает с градиентом
соответствующей взвешенной цели. На втором шаге каждый клиент уже находится
в собственной точке и вычисляет градиент в ней.

Рассмотрим две скалярные функции:

$$
\begin{gathered}
F_1(w)=\frac12 w^2,\\
F_2(w)=2(w-4)^2,\\
F(w)=\frac14 F_1(w)+\frac34 F_2(w).
\end{gathered}
$$

Производные равны $F'_1(w)=w$, $F'_2(w)=4(w-4)$.
При $w_0=2$ и шаге 0.1 первый клиент получает $1.8$, второй $2.8$.
Их среднее $0.25\cdot1.8+0.75\cdot2.8=2.55$ совпадает с одним шагом
по общей функции.

После ещё одного локального шага клиенты получают $1.62$ и $3.28$,
среднее равно $2.865$. Два последовательных шага по общей функции дали бы
сначала $2.55$, затем $2.92125$. Разница появилась уже на второй итерации:
локальные градиенты вычислялись в разных точках.

```python
def local_steps(start, curvature, target, steps, rate=0.1):
    value = float(start)
    for _ in range(steps):
        value -= rate * curvature * (value - target)
    return value

one = 0.25 * local_steps(2, 1, 0, 1) + 0.75 * local_steps(2, 4, 4, 1)
two = 0.25 * local_steps(2, 1, 0, 2) + 0.75 * local_steps(2, 4, 4, 2)
central = local_steps(2, 3.25, 48 / 13, 2)
assert abs(one - 2.55) < 1e-12
assert abs(two - 2.865) < 1e-12
assert abs(central - 2.92125) < 1e-12
```

Если клиенты почти дойдут до своих минимумов, они вернут 0 и 4, а среднее
будет 3. Минимум общей функции находится в $48/13\approx3.6923$.
Пример показывает, почему всё более точное решение локальных задач не
обязательно лучше решает общую задачу. Он не устанавливает сходимость
произвольного федеративного алгоритма.

В статье FedAvg $E$ означает число локальных эпох, $B$ — размер мини-батча.
Для $n_k=100$, $B=32$, $E=2$ при сохранении неполного последнего батча
получаются $2\lceil100/32\rceil=8$ шагов, а не два. Если последний батч
отбрасывают, число шагов меняется. Лимит по времени может прервать эпоху.
В протоколе фиксируют, какие именно объекты и шаги учитываются при взвешивании.

Неодинаковые распределения клиентов (*non-IID*) усиливают проблему:
различаются классы, язык, частоты наблюдений и связь между входом и меткой.
Для ограничения расхождения можно уменьшить число локальных шагов,
регуляризовать отклонение от общей модели или отделить персональные параметры.
Выбор проверяют по общей цели и группам клиентов, а не только по локальному loss.

<a id="система-важнее-средней-формулы"></a>

## Выбор участников, задержки и сетевой обмен

Межорганизационное обучение (*cross-silo*) обычно работает с небольшим числом
известных организаций. Участники могут иметь стабильное соединение и
согласованное расписание, хотя это не устраняет различия данных.
В межустройственном варианте (*cross-device*) сервер видит лишь доступную
часть большого парка. Телефон может исчезнуть из сети посреди раунда,
перезагрузиться или прислать результат после его закрытия.

Простой синхронный протокол заранее задаёт идентификатор раунда, базовую
версию, срок приёма, минимальное число участников и условия отмены.
Например, раунд 42 обучает основу версии 17. Координатор принимает обновления
только этого раунда и этой основы. Повтор того же сообщения после сетевого
сбоя не учитывается второй раз. Обновление с NaN, неверной размерностью
или недопустимым весом отклоняется. Результат от версии 16 не становится
совместимым с версией 17 после замены поля в заголовке.

При обучении с частичным участием можно выбрать больше кандидатов, чем нужно
результатов, и завершить раунд после первых $K$ ответов либо по сроку.
Такой выбор ограничивает ожидание отстающих клиентов (*stragglers*).

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/edge-review-2026-10/harvard-stragglers.svg]]

*Harvard Edge ML Systems Book contributors,
[Edge Intelligence, Client scheduling](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/edge_intelligence/edge_intelligence.qmd#L2804-L2848),
CC BY-NC-SA 4.0. Оригинальная последовательность приёма первых $K$
обновлений скопирована без изменений. Это один вариант политики завершения,
не обязательное правило FedAvg.*

Ускорение раунда может изменить состав участников. Допустим, из
10 000 устройств 2000 прошли условия допуска, 200 были выбраны,
150 завершили локальное обучение и 120 результатов были приняты до срока.
Доля принятых составляет 60% выбранных, но только 1.2% всего парка.
Если сравнивать лишь первый процент, можно не заметить, что определённая
группа почти никогда не проходит начальные условия.

Поэтому эти количества считают отдельно для классов устройств, регионов
или других заранее определённых групп. Перевзвешивание принятых ответов
может изменить их вклад, но не восстанавливает неизвестные данные клиентов,
которые ни разу не участвовали. Расширение срока, квоты участия и изменение
условий допуска тоже меняют протокол; их эффект проверяют по времени,
охвату и качеству.

Объём обмена оценивают до выбора сжатия. Передача 10 миллионов FP32-параметров
занимает 40 MB в одном направлении. Если 100 клиентов за раунд каждый
скачивают модель и загружают такой же по размеру результат, получается
$2\cdot100\cdot40=8000$ MB, то есть 8 GB на раунд.
Сто раундов требуют 800 GB. Это полезная нагрузка без заголовков, повторов,
распределения ключей и другого трафика.

Если в обе стороны передаются только 100 тысяч параметров адаптера, расчёт
даёт 80 MB на раунд и 8 GB за сто раундов. Первоначальная загрузка основы
сюда не включена. Квантование и разреживание уменьшают размер отдельных
сообщений, но могут изменить число раундов до заданного качества. Сравнение
по байтам одного обновления не устанавливает выигрыш полного обучения.

<a id="приватность-данные-не-ушли-информация-могла-уйти"></a>

## Приватность обновлений

### Что скрывает безопасная агрегация

Если серверу нужны только сумма и общий вес, отдельные обновления можно
скрыть с помощью безопасной агрегации (*secure aggregation*). В упрощённой
схеме пары клиентов согласуют случайные маски: один прибавляет маску,
другой вычитает. В сумме маски сокращаются.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/edge-review-2026-10/harvard-secure-agg.svg]]

*Harvard Edge ML Systems Book contributors,
[Edge Intelligence, Federated privacy](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/edge_intelligence/edge_intelligence.qmd#L3046-L3065),
CC BY-NC-SA 4.0. Оригинальная схема двух клиентов скопирована без изменений.
Показано сокращение масок; гарантии при сговоре и выпадении участников
этой схемой не определяются.*

Для трёх скалярных обновлений $u_1=5$, $u_2=7$, $u_3=-2$ возьмём парные
маски $r_{12}=3$, $r_{13}=1$, $r_{23}=8$. Клиенты отправят:

$$
\begin{gathered}
z_1=5+3+1=9,\\
z_2=7-3+8=12,\\
z_3=-2-1-8=-11.
\end{gathered}
$$

Их сумма равна 10, как и сумма исходных обновлений. При вычислениях по
модулю 97 последний клиент отправит 86 вместо $-11$, а
$(9+12+86)\bmod97=10$.

```python
updates = [5, 7, -2]
masked = [(5 + 3 + 1) % 97, (7 - 3 + 8) % 97, (-2 - 1 - 8) % 97]
assert masked == [9, 12, 86]
assert sum(masked) % 97 == sum(updates) % 97 == 10
# Без третьего клиента часть масок остаётся:
assert sum(masked[:2]) % 97 == 21
assert sum(updates[:2]) % 97 == 12
```

Это арифметическая иллюстрация, не реализация криптографического протокола.
В реальной системе согласуют ключи, кодирование вещественных значений,
диапазон сумм и восстановление после выпадения клиентов. Если третий клиент
исчез, сумма двух сообщений больше не равна сумме двух обновлений: остаются
маски, связанные с исчезнувшим участником. Дополнительные этапы протокола
должны либо устранить их при выполнении пороговых условий, либо отменить раунд.
См. [Bonawitz et al., Practical Secure Aggregation](https://doi.org/10.1145/3133956.3133982).

Безопасная агрегация не скрывает опубликованную сумму и не доказывает
корректность вклада клиента. Если сервер знает все слагаемые, кроме одного,
он может вычесть известные из суммы. Поэтому существенны допустимый сговор,
минимальное число участников и правила повторных запросов к пересекающимся
группам. Проверка формы и конечности обновления также не защищает от
целенаправленного отравления модели. Скрытие индивидуальных обновлений
может ограничить доступные серверу методы их проверки.

### Дифференциальная приватность и накопление гарантии

Дифференциальная приватность (DP) ограничивает изменение распределения
публикуемого результата при изменении защищаемых данных. Сначала определяют
соседство наборов $D$ и $D'$. При защите записи они отличаются одной записью.
При защите пользователя изменяется весь его вклад, который может содержать
много записей и несколько участий в обучении.

Определение требует для любого допустимого события $A$:

$$
\begin{gathered}
\Pr[\mathcal M(D)\in A]\\
\le e^\varepsilon\Pr[\mathcal M(D')\in A]+\delta.
\end{gathered}
$$

Вероятности относятся к случайности механизма. Параметр $\delta$ — добавочный
член этой границы; его не следует автоматически описывать как вероятность
«раскрытия всех данных». Определение, свойства и композиция изложены в
[Dwork и Roth, Definition 2.4 и раздел 3.5](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf).

Для ограничения вклада одного клиента весь его вектор обновления $u$
можно обрезать по норме:

$$
\bar u=u\min\left(1,\frac{C}{\|u\|_2}\right),
$$

а нулевой вектор оставить нулём. Например, $(3,4)$ имеет норму 5; при $C=2$
получается $(1.2,1.6)$ с нормой 2. Для суммы таких векторов добавление
или удаление одного клиента меняет результат не более чем на $C$.
При замене одного клиента другим верхняя граница равна $2C$:
в одном измерении переход от $-C$ к $+C$ достигает этой границы.

В коде вектор сначала делится на максимальный модуль координаты, чтобы
при расчёте нормы не возводить в квадрат очень большие исходные значения.
Эта промежуточная нормировка сохраняет направление вектора.

```python
def clip_update(update, bound):
    update = np.asarray(update, dtype=float)
    if (update.ndim != 1 or not np.isfinite(update).all()
            or not np.isfinite(bound) or bound <= 0):
        raise ValueError("Нужен конечный вектор и положительная граница")
    scale = np.max(np.abs(update), initial=0.0)
    if scale == 0:
        return update.copy()
    direction = update / scale
    norm = np.linalg.norm(direction)
    return update.copy() if scale <= bound / norm else direction * (bound / norm)

np.testing.assert_allclose(clip_update([3, 4], 2), [1.2, 1.6])
np.testing.assert_allclose(clip_update([0, 0], 2), [0, 0])
assert 2 * 2 == abs(2 - (-2))  # замена клиента при C = 2
```

Для среднего чувствительность выводят заново. Если ровно $m$ клиентов имеют
равные веса и заменяется один из них, граница равна $2C/m$.
Если число клиентов, веса или знаменатель зависят от защищаемых данных,
это рассуждение уже нельзя перенести без дополнительных условий.
В частности, формула FedAvg с разными $n_k$ не получает чувствительность $C$
только потому, что каждый вектор был обрезан до нормы $C$.

К результату добавляют шум, согласованный с чувствительностью и требуемой
гарантией. Обрезка нормы без шума сама по себе DP не обеспечивает.
При центральной DP доверенный агрегатор может видеть индивидуальные данные
или обновления до добавления шума; если от него тоже требуется защита,
нужна иная модель доверия, например корректное сочетание безопасной
агрегации и распределённого добавления шума. Перенести шум между клиентом
и сервером без изменения анализа нельзя.

Гарантию учитывают по всем публикациям, в которых участвуют защищаемые
данные. Десять механизмов с гарантиями $(0.2,10^{-6})$ для того же
пользователя по простой теореме композиции дают не хуже $(2,10^{-5})$.
Это верхняя граница учебного примера, а не результат расчёта реального FL.
Более точные методы учёта и усиление приватности при выборке требуют условий
на конкретный механизм и способ выбора участников.

В описании эксперимента сохраняют единицу приватности, соседство,
ограничение вклада, распределение шума, правило выборки, число публикаций
и состояние учёта накопленной гарантии (*privacy accountant*).
Без этих величин запись «использовали DP» не задаёт проверяемого утверждения.
Пример пользовательской защиты языковой модели см.
[McMahan et al., Learning Differentially Private Recurrent Language Models](https://arxiv.org/abs/1710.06963).

## Версионирование распределённой адаптации

В парке одновременно существуют несколько общих релизов и множество
локальных состояний. Идентификатора общей модели недостаточно, чтобы
воспроизвести ответ конкретного устройства или решить, можно ли принять
его обновление.

| Состояние | Зачем оно нужно |
|---|---|
| `base_model_version` | Указывает точную основу и смысл координат обновления |
| `adaptation_algorithm_version` | Задаёт обучаемые слои, оптимизатор и ограничения |
| `feature_and_schema_version` | Связывает ввод и кэш признаков с предобработкой |
| `federated_round` | Отделяет результаты разных раундов и повторы сообщений |
| `device_tier_policy` | Фиксирует ограничения выбранного класса устройств |
| `local_adapter_version` | Обозначает персональные параметры поверх основы |
| `privacy_accountant_state` | Не позволяет забыть предыдущие публикации при перезапуске |
| `rollback_baseline` | Указывает совместимую проверенную резервную конфигурацию |

Например, устройство загрузило основу 17 и обучило адаптер `A17`.
После установки основы 18 нельзя считать пару `(18, A17)` проверенной:
изменился извлекатель признаков, относительно которого обучалась поправка.
Политика выпуска может отключить адаптер, заново обучить его или выполнить
проверенную миграцию. Простое сохранение имени файла не подтверждает
совместимость.

Асинхронные федеративные методы допускают запаздывающие обновления по
собственным правилам. Синхронный FedAvg не превращается в такой метод после
удаления проверки версии. Допустимое запаздывание, вес устаревшего обновления
и точку, относительно которой вычислена разность, задают в алгоритме.

<a id="distributed-validation-и-rollback"></a>

## Оценивание на устройствах и откат

Обучающая выборка клиента не должна одновременно служить единственным
свидетельством улучшения его персонализации. Приложение может отложить
часть примеров, а при временной зависимости данных отделить последующие
наблюдения. Сравниваются общая модель, адаптированная и проверенная резервная
модель на одних и тех же допустимых тестовых примерах.

Агрегировать нужно величины, соответствующие целевой метрике.
Пусть первый клиент правильно классифицировал 9 из 10 объектов,
второй — 50 из 100. Средняя точность по пользователям равна
$(0.9+0.5)/2=0.7$. Точность по всем объектам равна
$(9+50)/(10+100)\approx0.5364$. Поэтому для второго показателя передают
или защищённо агрегируют суммы правильных ответов и числа примеров,
а не усредняют проценты с одинаковым весом.

```python
correct = np.array([9, 50])
total = np.array([10, 100])
per_user = np.mean(correct / total)
per_example = correct.sum() / total.sum()
assert abs(per_user - 0.7) < 1e-12
assert abs(per_example - 59 / 110) < 1e-12
```

Оба показателя описывают только наблюдавшихся клиентов и их тестовые данные.
Они не учитывают отсутствующую группу автоматически. Рядом с качеством
фиксируют охват, размер выборок, версии моделей и правила допуска.
Для метрик тоже нужна модель приватности: точный показатель маленькой группы
может раскрывать сведения даже без передачи отдельных примеров.

Без меток доступны расход памяти, сбои, время операций и расхождения с
резервной моделью. Расхождение показывает изменение поведения, но не
определяет, какая модель права. Высокая уверенность адаптированной модели
также не заменяет проверку правильности. Запаздывающие метки учитываются
отдельно, с привязкой к версии, которая сделала предсказание.

Для установки обновления можно использовать две копии: активную и
неактивную. Приложение скачивает кандидат в неактивную копию, проверяет
доверенную подпись, контрольную сумму, совместимость и контрольные примеры.
Затем оно согласует переключение с инференсом и обучением, атомарно меняет
активную ссылку и сохраняет прежнюю совместимую конфигурацию на контрольное
окно. Адаптер, предобработка и локальный checkpoint участвуют в переключении
вместе с основой.

Локальные условия отказа могут сработать без сети: ошибка загрузки,
несовместимый формат, превышение ресурсов или заданное правило резервного
поведения. Удалённый отзыв неизвестного устройству релиза без связи
доставить нельзя. Сервер может остановить новые раунды и выпуск версии,
но уже отключённое устройство узнает об отзыве лишь после связи либо
по заранее установленному сроку действия. Это разные механизмы остановки.

## Разные правила для разных классов устройств

Один класс устройств может выполнять полное локальное обучение головы,
другой — редкое обновление небольшого адаптера, третий — только инференс.
Правила определяются измерениями конкретных конфигураций: ранга адаптера,
размера батча, рабочих буферов и допустимой длительности. Низкий ранг сам по
себе не доказывает, что обучение помещается в память.

Разные классы могут участвовать в разных раундах или иметь совместимые
общие обучаемые части и отдельные персональные части. Если размеры
адаптеров различаются, нужен явно определённый способ их согласования;
обычная формула среднего предполагает одинаковое пространство параметров.

Отдельно проверяют качество у устройств, которым обучение недоступно.
Если общую модель улучшают только обновления мощных телефонов, суммарная
метрика может скрывать ухудшение у остальных. Сравнение по классам включает
и качество, и возможность участия, и расход ресурсов, а резервная модель
остаётся работоспособной в пределах заявленной функции приложения.

Сброс персонализации удаляет локальный адаптер и предусмотренные политикой
буферы. Он не вычитает автоматически прежний вклад пользователя из уже
обученной общей модели. Локальный сброс, прекращение дальнейшего участия
и удаление влияния из общей модели требуют разных технических процедур.

<a id="production-checklist"></a>

## Контракт запуска

| Часть системы | Что фиксируется перед включением |
|---|---|
| Локальное обучение | Изменяемые параметры, память полного шага, поддерживаемые операции, пределы времени и энергии |
| Локальное состояние | Совместимость основы и адаптера, содержимое checkpoint, восстановление после перезапуска, срок хранения буфера |
| Федеративный раунд | Базовая версия, правило выбора, число и веса принятых клиентов, срок, порог отмены, дедупликация |
| Защита | Модель доверия и сговора, защищаемая единица, параметры и история DP, обработка выпадений |
| Оценивание | Отложенные данные, способ усреднения метрик, охват и результаты по группам |
| Выпуск | Проверка кандидата на каждом поддерживаемом классе, совместимый резервный комплект, локальные условия отката и доставка отзыва |

Приведённые CPU-примеры подтверждают арифметику, но не выполнение этого
контракта на реальном парке. Проверки среды исполнения, качества
персонализации, криптографического протокола и энергопотребления требуют
отдельных экспериментов.

## Практика и первоисточники

- Han Cai et al., [TinyTL](https://arxiv.org/abs/2007.11622v5), разделы 3.1–3.3:
  память обратного прохода, обучение смещений и остаточная ветвь.
  [Код авторов](https://github.com/mit-han-lab/tinyml/tree/0e58f314dfe12967d6165f45238b1ea77b0f7a14/tinytl).
- Google AI Edge, [On-device training with LiteRT](https://developers.google.com/edge/litert/conversion/tensorflow/build/ondevice_training):
  экспорт `train/infer/save/restore` и восстановление весов. Это пример API,
  не проверка целевого устройства.
- McMahan et al., [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://proceedings.mlr.press/v54/mcmahan17a.html),
  раздел 2 и Figure 1: вывод FedAvg из локальных шагов SGD.
- TensorFlow Federated, [Building Your Own Federated Learning Algorithm, Part 2](https://www.tensorflow.org/federated/tutorials/custom_federated_algorithms_2),
  раздел Federated training: рассылка модели, локальные вычисления, среднее.
  В упрощённом коде этого раздела клиенты имеют равные веса; код не является
  реализацией взвешенного примера выше.
- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol2/edge_intelligence|Harvard CS249r — Edge Intelligence]]:
  полный материал курса для дальнейшего чтения; здесь сверены выбранные
  разделы об архитектуре, выборе участников и приватности, не вся глава курса.
- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol1/ml_ops|Harvard CS249r — ML Operations]] и
  [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol2/ops_scale|ML Operations at Scale]].
- Bonawitz et al., [Practical Secure Aggregation for Privacy-Preserving Machine Learning](https://doi.org/10.1145/3133956.3133982).
- Dwork и Roth, [The Algorithmic Foundations of Differential Privacy](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf),
  Definition 2.4, post-processing и раздел 3.5 о композиции.
- Kairouz et al., [Advances and Open Problems in Federated Learning](https://arxiv.org/abs/1912.04977) и
  Li et al., [Federated Learning: Challenges, Methods, and Future Directions](https://doi.org/10.1109/MSP.2020.2975749)
  как дальнейшее чтение по вариантам алгоритмов и открытым задачам.

← [[02 MLOps]] · Далее: [[00 Учебник/19 Deployment, Reliability и MLOps/04 Безопасность и приватность ML-систем|Безопасность и приватность ML-систем]]
