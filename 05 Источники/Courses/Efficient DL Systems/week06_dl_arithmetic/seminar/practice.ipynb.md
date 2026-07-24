---
title: "Week 6: kernel fusion, torch.compile, GPU memory and Liger kernels"
type: external-resource
status: imported-source
language: original
source_kind: notebook
source_commit: e632aa89ca9e6638d52e1b686095e7442faffbb0
---

> [!note] Original course material
> The material below is preserved in its original language from
> [`week06_dl_arithmetic/seminar/practice.ipynb`](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week06_dl_arithmetic/seminar/practice.ipynb) in *Efficient Deep Learning Systems* at commit
> `e632aa89ca9e6638d52e1b686095e7442faffbb0`. License: [MIT](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/LICENSE). Bookvar changed only the
> publication markup and link paths.



### Seminar outline
1. Kernel Fusion
    - Why fusing operations matters
    - SwiGLU fusion
2. Internals of torch.compile
    - Basic torch.compile example
    - Understanding What Dynamo Captures
    - Graph Breaks
    - Loops
    - Extra. The 3 stages of torch.compile.
        - TorchDynamo: bytecode capture and FX graphs
        - AOTAutograd: forward and backward tracing
        - TorchInductor: Triton/C++ kernel generation
3. GPU Memory Hierarchy
4. Efficient Cross Entropy
    - Liger Kernel Cross Entropy
    - Fused Linear Cross Entropy

```python
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch version: {torch.__version__}")
```

## 1. Kernel Fusion

### Why Kernel Fusion Matters

**Problem:**
- GPU compute is much faster than memory
- For H100: 800 TFLOPS (25 TFLOPS for CUDA Cores) compute vs 2.4 TB/s memory bandwidth
- For elementwise ops, we're almost always memory-bound

**Why Fusion Helps:**
- Reduces memory traffic:
    - Unfused ops read/write HBM for each operation
    - Fused ops keep intermediates in registers
- Reduces kernel launch overhead:
    - Each CUDA kernel launch has ~5-10μs overhead
    - Fusing N ops into 1 kernel eliminates N-1 launches

<p float="left">
<img src="https://github.com/mryab/efficient-dl-systems/raw/e632aa89ca9e6638d52e1b686095e7442faffbb0/week06_dl_arithmetic/seminar/images/fused_kernels1.png" width="400"/>
<img src="https://github.com/mryab/efficient-dl-systems/raw/e632aa89ca9e6638d52e1b686095e7442faffbb0/week06_dl_arithmetic/seminar/images/fused_kernels2.png" width="400"/>
</p>

### SwiGLU fusion

SwiGLU is used in modern LLMs:

$$\text{SwiGLU}(x, y) = x \odot \text{SiLU}(y), \quad \text{SiLU}(x) = x \cdot \sigma(x)$$

The elementwise multiplication and SiLU are perfect fusion candidates.

```python
import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings


def swiglu_unfused(gate, up):
    """Each operation launches a separate CUDA kernel"""
    return F.silu(gate) * up


@torch.compile
def swiglu_compiled(gate, up):
    """torch.compile fuses silu + mul into a single kernel"""
    return F.silu(gate) * up


@triton.jit
def _silu(x):
    return x * tl.sigmoid(x)

@triton.jit
def _swiglu_forward_kernel(a_ptr, b_ptr, c_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    a_ptr += program_id * stride
    b_ptr += program_id * stride
    c_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # sigmoid requires type float32
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
    c_row = _silu(a_row) * b_row
    tl.store(c_ptr + col_offsets, c_row, mask=mask)


def swiglu_forward(a, b):
    ori_shape = a.shape

    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    c = torch.empty_like(a)
    n_rows = a.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _swiglu_forward_kernel[(n_rows,)](
        a,
        b,
        c,
        c.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return a, b, c.view(*ori_shape)
```

```python
def benchmark(fn, gate, up, warmup=10, iters=1000):
    for _ in range(warmup):
        _ = fn(gate, up)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()

    for _ in range(iters):
        _ = fn(gate, up)

    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event) / iters


gate = torch.randn(4096, 4096, device=device)
up = torch.randn(4096, 4096, device=device)

for _ in range(10):
    _ = swiglu_unfused(gate, up)
    _ = swiglu_compiled(gate, up)
    _ = swiglu_forward(gate, up)
torch.cuda.synchronize()

print(f"Unfused:       {benchmark(swiglu_unfused, gate, up):.3f} ms")
print(f"torch.compile: {benchmark(swiglu_compiled, gate, up):.3f} ms")
print(f"Triton:        {benchmark(swiglu_forward, gate, up):.3f} ms")
```

**OpenAI gpt-oss SwiGLU variant:**

$$\text{SwiGLU}_{\text{gpt-oss}}(x, y) = x \cdot \sigma(\alpha x) \cdot (y + 1)$$

Using the identity $\sigma(z) = \frac{1}{2}(1 + \tanh(\frac{z}{2}))$, this becomes:

$$\frac{x}{2} \cdot \left(1 + \tanh\left(\frac{\alpha x}{2}\right)\right) \cdot (y + 1)$$

**Optimized form (5 ops: FMUL, FMUL, TANH, FFMA, FFMA):**

$$\text{Let } h = \frac{x}{2}, \quad s = h \cdot \tanh(\alpha h) + h$$
$$\text{SwiGLU}_{\text{gpt-oss}}(x, y) = s \cdot y + s$$

```
h = x * 0.5           // FMUL #1
t = alpha * h         // FMUL #2  
t = tanh(t)           // TANH
s = h * t + h         // FFMA #1  (fused multiply-add)
out = s * y + s       // FFMA #2  (fused multiply-add)
```

```python
ALPHA = 1.0

def swiglu_gptoss_unfused(x, y):
    """GPT-OSS style: x * sigmoid(a*x) * (y + 1) - unfused"""
    return x * torch.sigmoid(ALPHA * x) * (y + 1)
```

```python
torch._dynamo.reset()
torch._logging.set_logs(output_code=True)

@torch.compile
def swiglu_gptoss_inspect(x, y):
    return x * torch.sigmoid(ALPHA * x) * (y + 1)

x = torch.randn(4096, 4096, device=device)
y = torch.randn(4096, 4096, device=device)
_ = swiglu_gptoss_inspect(x, y)

torch._logging.set_logs()
```

```python
torch._dynamo.reset()
torch._logging.set_logs(output_code=True)

@torch.compile
def swiglu_gptoss_tanh_compiled(x, y):
    x_half = 0.5 * x
    silu_x = x_half * torch.tanh(ALPHA * x_half) + x_half
    return silu_x * y + silu_x

x = torch.randn(4096, 4096, device=device)
y = torch.randn(4096, 4096, device=device)
_ = swiglu_gptoss_tanh_compiled(x, y)

torch._logging.set_logs()
```

```python
x = torch.randn(4096, 4096, device=device)
y = torch.randn(4096, 4096, device=device)

print(f"Unfused (eager):     {benchmark(swiglu_gptoss_unfused, x, y):.3f} ms")
print(f"Compiled (sigmoid):  {benchmark(swiglu_gptoss_inspect, x, y):.3f} ms")
print(f"Compiled (tanh):     {benchmark(swiglu_gptoss_tanh_compiled, x, y):.3f} ms")
```

### Memory Traffic Analysis

For inputs `gate` and `up` of shape `(batch, seq, hidden_dim)`:

**Unfused SwiGLU:**

| Step | Operation | Memory Access |
|------|-----------|---------------|
| 1 | sigmoid(gate) | read, write |
| 2 | gate * sigmoid | 2x read, write |
| 3 | silu * up | 2x read, write |
| **Total** | **3 kernels** | **8 * (batch, seq, hidden_dim) reads/writes** |

**Fused SwiGLU:**

| Step | Operation | Memory Access |
|------|-----------|---------------|
| 1 | silu(gate) * up | 2x read, write |
| **Total** | **1 kernel** | **3 * (batch, seq, hidden_dim) reads/writes** |

All operations (sigmoid, multiply, multiply) happen in registers - no intermediate writes to HBM!

```python
# H100 SXM specs
MEMORY_BANDWIDTH_TB_S = 2.4
CUDA_CORE_TFLOPS = 25

def calculate_theoretical_time(
    shape, 
    n_reads, 
    n_writes, 
    flops_per_element,
    dtype_bytes=4
):
    """
    Calculate theoretical execution time based on memory and compute.
    
    Args:
        shape: tensor shape
        n_reads: number of tensor reads
        n_writes: number of tensor writes  
        flops_per_element: FLOPs per element (e.g., mul=1, add=1, sigmoid≈10)
        dtype_bytes: bytes per element (FP32=4, bf16=2)
    
    Returns:
        memory_time_ms, compute_time_ms, is_memory_bound
    """
    n_elements = 1
    for dim in shape:
        n_elements *= dim

    total_bytes = (n_reads + n_writes) * n_elements * dtype_bytes
    total_gb = total_bytes / 1e9
    memory_time_ms = total_gb / (MEMORY_BANDWIDTH_TB_S * 1000) * 1000  # TB/s → GB/ms

    total_flops = n_elements * flops_per_element
    total_tflops = total_flops / 1e12
    compute_time_ms = total_tflops / CUDA_CORE_TFLOPS * 1000

    is_memory_bound = memory_time_ms > compute_time_ms
    return memory_time_ms, compute_time_ms, is_memory_bound


shape = (4096, 4096)

# Unfused: 3 separate kernels
# Kernel 1: sigmoid(gate) - read 1, write 1, ~10 FLOPs (exp, div, etc.)
# Kernel 2: gate * sigmoid - read 2, write 1, 1 FLOP
# Kernel 3: silu * up - read 2, write 1, 1 FLOP
unfused_reads = 1 + 2 + 2  # = 5
unfused_writes = 1 + 1 + 1  # = 3
unfused_flops = 10 + 1 + 1  # = 12

mem_t, comp_t, is_mem = calculate_theoretical_time(shape, unfused_reads, unfused_writes, unfused_flops)
print(f"\n**Unfused:**")
print(f"  Memory: {unfused_reads} reads + {unfused_writes} writes = {unfused_reads + unfused_writes}x tensor")
print(f"  Memory time:  {mem_t:.4f} ms")
print(f"  Compute time: {comp_t:.4f} ms")
print(f"  Bound by: {'MEMORY' if is_mem else 'COMPUTE'} ({mem_t/comp_t:.1f}x ratio)")

fused_reads = 2
fused_writes = 1
fused_flops = 12

mem_t, comp_t, is_mem = calculate_theoretical_time(shape, fused_reads, fused_writes, fused_flops)
print(f"\n**Fused:**")
print(f"  Memory: {fused_reads} reads + {fused_writes} writes = {fused_reads + fused_writes}x tensor")
print(f"  Memory time:  {mem_t:.4f} ms")
print(f"  Compute time: {comp_t:.4f} ms")
print(f"  Bound by: {'MEMORY' if is_mem else 'COMPUTE'} ({mem_t/comp_t:.1f}x ratio)")
```

## 2. Internals of torch.compile

The [`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html) allows you to compile your existing PyTorch code into optimized kernels, automatically fusing operations like we saw in the previous section. It often achieves significant speedups with just a single line change: wrapping your model or function with `torch.compile()`.

### Basic torch.compile example

```python
def simple_fn(x):
    x = x * 2
    x = x + 1
    x = torch.relu(x)
    x = x * 0.5
    return x

compiled_fn = torch.compile(simple_fn)

x = torch.randn(1000, 1000, device=device)

out1 = simple_fn(x)
out2 = compiled_fn(x)  # First call triggers compilation

print(f"Results match: {torch.allclose(out1, out2)}")
```

### Understanding What Dynamo Captures

**TorchDynamo** is the first stage of `torch.compile`. It intercepts Python bytecode at runtime and extracts PyTorch operations into an FX graph - a simple intermediate representation that captures the sequence of tensor operations without Python overhead. For details, see the [Dynamo Deep-Dive](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamo_deepdive.html) documentation.

We can use `torch._dynamo.explain()` to see what Dynamo captures:

```python
torch._dynamo.reset()

explanation = torch._dynamo.explain(simple_fn)
x = torch.randn(1000, 1000, device=device)

result = explanation(x)
print(result)
```

- **Graph Count: 1** - The entire function was captured in a single graph (good!)
- **Graph Break Count: 0** - No graph breaks occurred (good!)
- **Op Count: 4** - Four operations captured: `mul`, `add`, `relu`, `mul`
- **Guards** - Conditions that must remain true for the cached compilation to be reused (e.g., tensor shapes, dtypes, device). If guards fail, Dynamo recompiles.

| Guard | What it checks |
|-------|----------------|
| `SHAPE_ENV` | Symbolic shape constraints are satisfied |
| `DETERMINISTIC_ALGORITHMS` | `torch.use_deterministic_algorithms()` unchanged |
| `GRAD_MODE` | `torch.is_grad_enabled()` unchanged |
| `DEFAULT_DEVICE` | Default device hasn't changed |
| `GLOBAL_STATE` | Global PyTorch state unchanged |
| `TORCH_FUNCTION_STATE` | `__torch_function__` dispatch unchanged |
| `TENSOR_MATCH` | Input tensor properties match (shape, dtype, device, strides) |
| `MODULE_MATCH` | The `torch` module is the same object |
| `BUILTIN_MATCH` | `torch.relu` is the same builtin function |

### Graph Breaks

**Graph breaks** occur when Dynamo encounters code it can't capture into the graph. Common causes:
- **Data-dependent control flow** - value depends on tensor data
- **Unsupported operations** - certain Python built-ins or dynamic features
- **Non-compilable function calls** - functions Dynamo can't trace into

When a graph break happens, Dynamo splits execution: compiled code runs up to the break, then Python takes over, then compilation may resume after.

```python
torch._dynamo.reset()

def fn_with_break(x):
    x = x * 2
    if x.sum() > 0:
        x = x + 1
    else:
        x = x - 1
    return x

explanation = torch._dynamo.explain(fn_with_break)
x = torch.randn(10, 10, device=device)
result = explanation(x)
print(result.graph_break_count)
print(result.break_reasons)
```

```python
torch._dynamo.reset()

def fn_without_break(x):
    x = x * 2
    if x.shape[0] > 5:
        x = x + 1
    else:
        x = x - 1
    return x

explanation = torch._dynamo.explain(fn_without_break)
x = torch.randn(10, 10, device=device)
result = explanation(x)
print(result.graph_break_count)
print(result.break_reasons)
```

```python
torch._dynamo.reset()

def fn_with_print(x):
    x = x * 2
    print(f"Shape: {x.shape}")
    x = x + 1
    return x

explanation = torch._dynamo.explain(fn_with_print)
x = torch.randn(10, 10, device=device)
result = explanation(x)
print(result.graph_break_count)
print(result.break_reasons)
```

```python
torch._dynamo.reset()

def fn_with_item(x):
    x = x * 2
    val = x[0, 0].item()
    if val > 0:
        x = x + 1
    return x

explanation = torch._dynamo.explain(fn_with_item)
x = torch.randn(10, 10, device=device)
result = explanation(x)
print(result.graph_break_count)
print(result.break_reasons)
```

```python
torch._dynamo.reset()

def fn_with_tolist(x):
    x = x * 2
    shape_list = list(x.shape)
    data_list = x[0].tolist()
    x = x + 1
    return x

explanation = torch._dynamo.explain(fn_with_tolist)
x = torch.randn(10, 10, device=device)
result = explanation(x)
print(result.graph_break_count)
print(result.break_reasons)
```

### Loops

```python
def loop_update(tensors, value):
    for t in tensors:
        t.mul_(0.9).add_(value)

def foreach_update(tensors, value):
    torch._foreach_mul_(tensors, 0.9)
    torch._foreach_add_(tensors, value)

loop_compiled = torch.compile(loop_update)
foreach_compiled = torch.compile(foreach_update)

tensors_loop = [torch.randn(1000, 1000, device="cuda") for _ in range(10)]
tensors_foreach = [t.clone() for t in tensors_loop]

torch._dynamo.reset()
explanation = torch._dynamo.explain(loop_compiled)(tensors_loop, 0.1)
print(f"Loop version - Graph breaks: {explanation.graph_break_count}")

torch._dynamo.reset()
explanation = torch._dynamo.explain(foreach_compiled)(tensors_foreach, 0.1)
print(f"Vectorized version - Graph breaks: {explanation.graph_break_count}")
```

```python
torch._dynamo.reset()
torch._logging.set_logs(output_code=True)

def loop_update(tensors, value):
    for t in tensors:
        t.mul_(0.9).add_(value)

def foreach_update(tensors, value):
    torch._foreach_mul_(tensors, 0.9)
    torch._foreach_add_(tensors, value)

loop_compiled = torch.compile(loop_update)
foreach_compiled = torch.compile(foreach_update)

tensors_a = [torch.randn(1000, 1000, device="cuda") for _ in range(5)]
tensors_b = [t.clone() for t in tensors_a]
```

```python
loop_compiled(tensors_a, 0.1)
```

```python
torch._dynamo.reset()
foreach_compiled(tensors_b, 0.1)

torch._logging.set_logs()
```

```python
print(f"Python loop:      {benchmark(loop_update, tensors_loop, 0.1):.3f} ms")
print(f"torch._foreach_:  {benchmark(foreach_update, tensors_foreach, 0.1):.3f} ms")
```

## Extra. The 3 stages of torch.compile.

1. **Graph Acquisition (TorchDynamo + AOTAutograd)** - TorchDynamo intercepts Python bytecode at runtime and extracts the computational operations into an FX graph. AOTAutograd then traces both forward and backward passes ahead-of-time, producing separate graphs for each.

2. **Graph Lowering** - The high-level FX graph is lowered into a more primitive representation. Operations are decomposed into simpler ops, and the graph is normalized into a form suitable for optimization and code generation.

3. **Graph Compilation (TorchInductor)** - The backend compiler takes the lowered graph and generates optimized kernels. TorchInductor produces Triton code for GPU, applying optimizations like fusion, memory planning, and efficient scheduling.

Let's interactively explore what happens at each stage when `compiled_fn(x)` runs.

#### Stage 1: Graph Acquisition (TorchDynamo)

TorchDynamo captures the Python bytecode and builds an FX graph. We can inspect this using a custom backend.

```python
def inspect_graph(gm, example_inputs):
    """
    Custom backend that prints the FX graph captured by Dynamo.
    
    Args:
        gm: torch.fx.GraphModule containing the captured FX graph.
            - gm.graph: the FX graph representation
            - gm.forward: callable to execute the graph
        example_inputs: List[Tensor] - the actual inputs that triggered compilation
    
    Returns:
        A callable that executes the graph (here we just return eager execution)
    """
    print(gm.graph)
    return gm.forward

# The `backend` parameter specifies which compiler will process the captured graph.
# The default is `"inductor"` (TorchInductor), which generates optimized Triton/C++ kernels.
torch._dynamo.reset()
inspect_compiled = torch.compile(simple_fn, backend=inspect_graph)

x = torch.randn(1000, 1000, device=device)
_ = inspect_compiled(x)
```

**AOTAutograd: Forward and Backward Graphs**

The graph above is just what Dynamo captured. AOTAutograd then takes this and generates separate forward and backward graphs:

```python
from torch._functorch.aot_autograd import aot_function
from functorch.compile import make_boxed_func

def fw_compiler(gm, example_inputs):
    print("=== Forward Graph ===")
    print(gm.graph)
    print()
    return make_boxed_func(gm.forward)

def bw_compiler(gm, example_inputs):
    print("=== Backward Graph ===")
    print(gm.graph)
    return make_boxed_func(gm.forward)

aot_fn = aot_function(simple_fn, fw_compiler=fw_compiler, bw_compiler=bw_compiler)

x = torch.randn(1000, 1000, device=device, requires_grad=True)

out = aot_fn(x)

out.sum().backward()
```

#### Stage 2: Graph Lowering (Decomposition)

The FX graph is then lowered - high-level ops are decomposed into primitives. We can see this using `torch._decomp`:

```python
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions

decompositions = get_decompositions([
    torch.ops.aten.relu,
])

x = torch.randn(1000, 1000, device=device)
lowered_graph = make_fx(simple_fn, decomposition_table=decompositions)(x)

print(lowered_graph.graph)
```

#### Stage 3: Graph Compilation (TorchInductor)

Finally, TorchInductor generates optimized Triton kernels. We can see the generated code:

```python
torch._dynamo.reset()

torch._logging.set_logs(output_code=True)

compiled_with_debug = torch.compile(simple_fn)

x = torch.randn(1000, 1000, device=device)
_ = compiled_with_debug(x)

torch._logging.set_logs()
```

For further reading:
1. [Torch Compiler Troubleshooting](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_troubleshooting.html)
2. [The Missing Manual](https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit?tab=t.0#heading=h.ivdr7fmrbeab)

### From Elementwise to Matmuls

We've seen that elementwise operations like SiLU are memory-bound - each element requires memory access with minimal computation.

But what about matrix multiplication? 

In the lecture, when calculating memory time for `matmul(A, B)` with `(N, N)` matrices in bf16, we used:

$$T_{memory} = \frac{N \times N \times 3 \times 2}{\text{bandwidth}}$$

Where:
- $N \times N$ - number of elements
- $3$ - two reads (A, B) and one write (C)  
- $2$ - bytes per bf16 element

**But wait...**

To compute a single output element $C[i,j] = \sum_k A[i,k] \cdot B[k,j]$, we need to read an entire row of A and entire column of B - that's $2N$ memory reads per output element. For all $N^2$ output elements, shouldn't memory be $2N^3$ reads, not just $3N^2$?

**Why do we only count each matrix element once?**

## 3. GPU Memory Hierarchy

Modern GPUs have a memory hierarchy, from fastest/smallest to slowest/largest:

- **HBM (VRAM)** - The main GPU memory you see in specs.
- **L2 Cache** - Shared across all SMs. Automatically caches HBM accesses (like CPU caches).
- **L1 Cache / Shared Memory** - Per-SM fast memory. On modern GPUs, L1 and shared memory are unified.
- **Registers** - Fastest storage, private to each thread. When we "keep intermediates in registers" for kernel fusion, this is what we mean.

GPU caches work almost like CPU caches - when you access memory, a whole **cache line** (128 bytes) is fetched:
- If data is in L1 - return immediately
- If L1 miss, check L2 - load to L1 and return
- If L2 miss - fetch from HBM into L2, then L1

![Memory Hierarchy](https://github.com/mryab/efficient-dl-systems/raw/e632aa89ca9e6638d52e1b686095e7442faffbb0/week06_dl_arithmetic/seminar/images/mem_hierarchy.png)

For further reading: [Inside NVIDIA GPUs: Anatomy of high performance matmul kernels](https://www.aleksagordic.com/blog/matmul)

### Cache Behavior in Matmul

Let's check cache hit rates for matrix multiplication:

```python
# Profile with Nsight Compute
!ncu --metrics lts__t_sector_hit_rate.pct,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors.sum python profile_matmul.py
```

**Nsight Compute Results:**
<pre style="background-color: #f5f5f5; color: #333; padding: 12px; border-radius: 6px; font-family: monospace;">
<span style="color: #569cd6;">sm80_xmma_gemm_f32f32_f32f32_f32_nn_n...</span> (16, 32, 1)x(256, 1, 1)
Device 0, CC 9.0

Metric                       Metric Unit    Metric Value
───────────────────────────────────────────────────────────
dram__bytes_read.sum              Mbyte        <b>341.94</b>
dram__bytes_write.sum             Mbyte         <b>53.44</b>
lts__t_sector_hit_rate.pct            %        <b>89.53%</b>
lts__t_sectors.sum               sector    <b>107,804,539</b>
</pre>

**Analysis:**
- **DRAM reads**: 342 MB (inputs A, B with some cache misses, with 100% hitrate would be 128 MB)
- **DRAM writes**: 53 MB (output C, ~64 MB with write coalescing)
- **Total L2 traffic**: 107M sectors x 32 bytes = 3.45 GB
- **Hit rate**: ~90% - only 10% of L2 accesses go to HBM

Why ~90% of hitrate for matmul? Each element of A and B is loaded once from HBM, then reused ~N times from L2/L1 cache for computing multiple output elements. The tiled algorithm ensures high data reuse within a single kernel.

This is why we count memory as $3N^2$ (like we load it once, which is still not very accurate, or it is?).

```python
# Profile with Nsight Compute
!ncu --metrics lts__t_sector_hit_rate.pct,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors.sum python profile_matmul_membound.py
```

**Nsight Compute Results (Memory-Bound Matmul):**
<pre style="background-color: #f5f5f5; color: #333; padding: 12px; border-radius: 6px; font-family: monospace;">
<span style="color: #569cd6;">cutlass_80_simt_sgemm_128x32_8x5_nn_align1</span> (64, 1, 16)x(128, 1, 1)
Device 0, CC 9.0

Metric                       Metric Unit    Metric Value
───────────────────────────────────────────────────────────
dram__bytes_read.sum              Mbyte         <b>16.86</b>
dram__bytes_write.sum             Mbyte          <b>4.38</b>
lts__t_sector_hit_rate.pct            %         <b>55.59%</b>
lts__t_sectors.sum               sector       <b>938,029</b>
</pre>

For this memory-bound matmul (A: 2048×2048 @ B: 2048×2), we read 16.86 MB from HBM - almost exactly the size of matrix A (16 MB). Unlike compute-bound matmuls that reread data multiple times through cache, memory-bound kernels with small outputs read each input element essentially once.

## 4. Efficient Cross Entropy

### The Cross Entropy Memory Problem

**Notation:**
- N = batch_size × sequence_length (number of tokens)
- V = vocabulary size

**Standard Cross Entropy Memory:**

Forward pass:
- `logits` (N, V) - input from final linear layer
- `log_softmax(logits)` (N, V) - saved for backward

Backward pass:
- `softmax = exp(log_softmax)` (N, V) - needed to compute gradient
- `grad_logits` (N, V) - output gradient: `softmax - one_hot(target)`

**Peak memory:** 4x (N, V) tensors during backward!

Example: bf16 with shape `(1, 32768, 128000)` → **8 GB per tensor**.

### Liger Kernel Cross Entropy

Code: [Liger Kernel Cross Entropy](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/cross_entropy.py).

**Key Optimizations:**

**1. In-place Gradient Storage**

Instead of allocating a separate gradient tensor, store the gradient directly in the input logits:

$$\nabla_x L = \text{softmax}(x) - \text{one\_hot}(\text{target})$$

```python
# Forward pass computes both: loss and gradient
# Gradient overwrites the logits in-place
logits[i] = softmax(logits[i]) - (1 if i == target else 0)
```

**2. Online Softmax**

Compute softmax statistics in a streaming fashion without materializing the full probability vector:

```python
m, d = -inf, 0.0  # running max and denominator
for chunk in blocks(logits):
    m_new = max(m, chunk.max())
    d = d * exp(m - m_new) + sum(exp(chunk - m_new))
    m = m_new

lse = m + log(d)  # log-sum-exp
loss = lse - logits[target]  # = -log(softmax[target])
```

Result: Peak memory reduced from 4x to 1x (N, V).

Let's take a look at snapshots now.

```python
!python3 ./cross_entropy/profile_vanilla_ce.py
```

![Vanilla Snapshot](https://github.com/mryab/efficient-dl-systems/raw/e632aa89ca9e6638d52e1b686095e7442faffbb0/week06_dl_arithmetic/seminar/images/vanilla_ce.png)

```python
!python3 ./cross_entropy/compiled_ce_snapshot.py
```

![Compiled Snapshot](https://github.com/mryab/efficient-dl-systems/raw/e632aa89ca9e6638d52e1b686095e7442faffbb0/week06_dl_arithmetic/seminar/images/compiled_ce.png)

```python
!python3 ./cross_entropy/profile_liger_ce.py
```

![Vanilla Snapshot](https://github.com/mryab/efficient-dl-systems/raw/e632aa89ca9e6638d52e1b686095e7442faffbb0/week06_dl_arithmetic/seminar/images/liger_ce.png)

### Fused Linear Cross Entropy (FLCE)

Liger CE reduces peak memory from 4x to 1x, but still materializes the full logits tensor. FLCE avoids this by fusing the linear projection with cross-entropy computation.

Code: [Liger Fused Linear Cross Entropy](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py)

![Liger Fused Linear Cross Entropy](https://github.com/mryab/efficient-dl-systems/raw/e632aa89ca9e6638d52e1b686095e7442faffbb0/week06_dl_arithmetic/seminar/images/liger_fused.avif)

For further optimization of cross-entropy with large vocabularies, see Apple's Cut Cross-Entropy: [Cut Your Losses
in Large-Vocabulary Language Models](https://arxiv.org/pdf/2411.09009v1).

```python

```
