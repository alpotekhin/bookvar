---
title: "Компиляция графов: PyTorch, JAX и CUDA Graphs"
type: textbook-chapter
status: draft
last_updated: 2026-11-04
last_verified: 2026-11-04
source_language: mixed
reference_terms:
  - anchor: compile-tracing-a-data-dependent-branch
    label: "Tracing и зависимое от данных ветвление"
    label_en: "Tracing and data-dependent branches"
  - anchor: compile-exporting-an-explicit-conditional
    label: "Экспорт явного условного оператора"
    label_en: "Exporting explicit conditionals"
  - anchor: torch-compile-graphs-guards-recompilation
    label: "Dynamo: графы, guards и перекомпиляция"
    label_en: "Dynamo: graphs, guards and recompilation"
  - anchor: compile-graph-breaks-divide-one-call-into-compiled-regions
    label: "Graph breaks и eager-участки"
    label_en: "Graph breaks and eager regions"
  - anchor: compile-dynamic-shapes-and-changing-scalar-values
    label: "Динамические формы и скалярные аргументы"
    label_en: "Dynamic shapes and scalar arguments"
  - anchor: compile-jax-tracing-state-and-compiled-control-flow
    label: "Tracing и состояние JAX"
    label_en: "JAX tracing and state"
  - anchor: compile-state-as-an-argument-and-a-result
    label: "Явная передача состояния в JAX"
    label_en: "Explicit state threading in JAX"
  - anchor: compile-refs-are-traceable-mutable-state
    label: "JAX refs и отслеживаемые изменения"
    label_en: "JAX refs and traceable mutations"
  - anchor: compile-what-a-cached-specialization-can-reuse
    label: "Кэш специализаций и скрытые константы"
    label_en: "Specialization caches and hidden constants"
  - anchor: compile-a-runtime-predicate-needs-a-compiled-conditional
    label: "JAX cond и ограничения форм"
    label_en: "JAX cond and shape constraints"
  - anchor: compile-a-loop-carries-fixed-shaped-state
    label: "Scan, состояние цикла и градиенты"
    label_en: "Scan, loop-carried state and gradients"
  - anchor: compile-donation-changes-buffer-ownership
    label: "Donation и владение буфером"
    label_en: "Donation and buffer ownership"
  - anchor: compile-device-placement-and-collective-communication
    label: "JAX sharding и psum"
    label_en: "JAX sharding and psum"
  - anchor: cuda-graphs-capture-replay
    label: "CUDA Graphs: capture и replay"
    label_en: "CUDA Graphs: capture and replay"
  - anchor: compile-follow-the-buffers-through-two-replays
    label: "Адреса, алиасы и время жизни буферов"
    label_en: "Buffer addresses, aliases and lifetimes"
  - anchor: compile-memory-pools-are-part-of-the-cost
    label: "Частные пулы памяти CUDA Graphs"
    label_en: "CUDA Graphs private memory pools"
  - anchor: compile-account-for-overlap-and-capture-amortization
    label: "Окупаемость capture с учётом перекрытия"
    label_en: "Capture amortization with overlap"
source_unit_id:
  - assignment-02-task-torch-compile
primary_sources:
  - https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/notes/cuda.md
  - https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
  - https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/performance_engineering/performance_engineering.qmd
  - https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/user_guide/torch_compiler/compile/programming_model.dynamo_core_concepts.md
  - https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/user_guide/torch_compiler/compile/programming_model.recompilation.md
  - https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/docs/source/user_guide/torch_compiler/torch.compiler_dynamo_overview.md
  - https://github.com/jax-ml/jax/tree/dc4af684446e65ee98eee8090e1ff6f8211473fb/docs
---

# Компиляция графов: PyTorch, JAX и CUDA Graphs

Одна функция Python может каждый раз выполнять одну и ту же цепочку
тензорных операций. Библиотека всё равно должна разобрать вызовы, выбрать
реализации и передать работу устройству. Компиляция позволяет записать
вычисления и повторно использовать подготовленную программу. При этом
нужно определить, какие решения зависят от конкретных значений входа,
какие формы разрешено менять и где хранится состояние между вызовами.

В [[02 Areas/ML & DL/00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению|предыдущей главе]]
ядро получало указатели и размеры и вычисляло одну операцию. Здесь
рассматривается программа, которая вызывает такие операции. Встречающееся
в названиях API слово *graph* обозначает разные представления этой программы:

| Представление | Запись и повторное использование |
|---|---|
| Dynamo | Тензорные операции из захваченного участка Python. Повторный вызов использует граф, если выполнены guards; иначе возможна новая специализация. |
| JAX | Операции над абстрактными значениями и явным состоянием. Для повторного вызова должны подходить специализация, структуры аргументов и статические значения. |
| CUDA Graph | Работа CUDA и аргументы её запусков. Повторный запуск требует совместимости записанных операций, форм и адресов буферов. |

Ниже сохранены исходные англоязычные объяснения PyTorch и JAX. Разбор
начинается с неверно записанной ветки Python: по ней видно, почему успешный
tracing ещё не доказывает сохранение смысла функции. Затем следуют guards
и graph breaks, состояние и управление потоком в JAX, адреса буферов и
окупаемость CUDA Graphs. Эти ограничения определяют, какую часть реальной
нагрузки удастся компилировать или воспроизводить без изменения результата.

<!-- BEGIN REVIEW: tracing control flow 2026-10-27 -->

<a id="compile-tracing-a-data-dependent-branch"></a>

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

<a id="compile-exporting-an-explicit-conditional"></a>

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

<a id="compile-graph-breaks-divide-one-call-into-compiled-regions"></a>

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

<a id="compile-dynamic-shapes-and-changing-scalar-values"></a>

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
<a id="compile-jax-tracing-state-and-compiled-control-flow"></a>

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

<a id="compile-state-as-an-argument-and-a-result"></a>

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

<a id="compile-refs-are-traceable-mutable-state"></a>

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

<a id="compile-what-a-cached-specialization-can-reuse"></a>

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

<a id="compile-a-runtime-predicate-needs-a-compiled-conditional"></a>

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

<a id="compile-a-loop-carries-fixed-shaped-state"></a>

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

<a id="compile-donation-changes-buffer-ownership"></a>

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

<a id="compile-device-placement-and-collective-communication"></a>

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

<a id="compile-follow-the-buffers-through-two-replays"></a>

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

<a id="compile-memory-pools-are-part-of-the-cost"></a>

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

<a id="compile-account-for-overlap-and-capture-amortization"></a>

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

## Связь с профилем и реализацией ядер

Граф может включать библиотечные операции и сгенерированные ядра Triton.
Проверка guards, повторный tracing и захват CUDA Graph относятся к разным
этапам выполнения. При измерении их разовые затраты отделяют от повторных
вызовов; обязательные копирования и обработку результатов оставляют внутри
заявленной границы измерения. Подробный разбор этой границы находится в
[[02 Areas/ML & DL/00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению#kernel-measurement|главе о GPU-ядрах]],
а чтение временной шкалы — в
[[02 Areas/ML & DL/00 Учебник/10 ML Systems/07 Profiling ML-нагрузки|главе о профилировании]].

Далее можно перейти к
[[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/01 Данные и pre-training|данным и предобучению]]
или к
[[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/44a Processes, collectives и DDP|процессам, коллективным операциям и DDP]].

← [[02 Areas/ML & DL/00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению|GPU kernels и Triton]] ·
[[02 Areas/ML & DL/00 Учебник/_index#Углублённые модули|Выбрать углублённый модуль]] →
