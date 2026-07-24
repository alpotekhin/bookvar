---
title: "Week 1: CUDA operations in PyTorch. Introduction to benchmarking."
type: external-resource
status: imported-source
language: original
source_kind: notebook
source_commit: e632aa89ca9e6638d52e1b686095e7442faffbb0
---

> [!note] Original course material
> The material below is preserved in its original language from
> [`week01_intro/seminar.ipynb`](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week01_intro/seminar.ipynb) in *Efficient Deep Learning Systems* at commit
> `e632aa89ca9e6638d52e1b686095e7442faffbb0`. License: [MIT](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/LICENSE). Bookvar changed only the
> publication markup and link paths.



In this seminar, we'll learn a bit more about the things one needs to keep in mind when using GPU computations, both in general and in PyTorch. We'll also see a couple of examples on benchmarking code in Python; again, there are some caveats with CUDA.

First, let's import PyTorch and get some information about the currently used device:

```python
import torch

torch.cuda.is_available()
```

```text
True
```

```python
torch.cuda.get_device_properties(0)
```

```text
_CudaDeviceProperties(name='Tesla T4', major=7, minor=5, total_memory=15095MB, multi_processor_count=40, uuid=2465eb53-250b-43ad-af03-85dd8f0b4eca, pci_bus_id=0, pci_device_id=4, pci_domain_id=0, L2_cache_size=4MB)
```

## Memory allocation
As discussed in the lecture, GPU memory is separate from the CPU memory and needs to be explicitly allocated, triggering a host-device synchronization. PyTorch uses a caching memory allocator to repurpose already available but unused memory fragments.

See this example: we allocate the memory inside the function scope, so the tensor is deleted as soon as the scope is left:

```python
def allocate_empty_tensor(dim_size):
    a = torch.zeros(4096, dim_size, dtype=torch.float32, device="cuda")
```

```python
allocate_empty_tensor(2048)
```

Printing the allocated memory size gives us an expected number of zero bytes:

```python
torch.cuda.memory_allocated()
```

```text
0
```

However, the GPU memory is still in use by the process; this is shown by `torch.cuda.memory_reserved` or `nvidia-smi` from the terminal. As a result, working in a shared GPU environment can leave a lot of unused yet allocated memory.

```python
torch.cuda.memory_reserved()
```

```text
33554432
```

```python
!nvidia-smi
```


```text
Mon Jan 19 14:04:06 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   68C    P0             31W /   70W |     150MiB /  15360MiB |      4%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
+-----------------------------------------------------------------------------------------+
```


Let's clear the cache now and see what happens:

```python
torch.cuda.empty_cache()
torch.cuda.memory_reserved()
```

```text
0
```

Note that this operation triggers a CPU-GPU synchronization, and thus using it in your code can significantly hurt the performance. It's almost always better to carefully manage the lifetime of your GPU tensors and avoid excessive allocations than to empty the cache.

Now, let's see how this cache is reused by allocating two tensors in a row: first a larger one, then a smaller one.

```python
allocate_empty_tensor(2048)
torch.cuda.memory_reserved()
```

```text
33554432
```

```python
allocate_empty_tensor(1024)
torch.cuda.memory_reserved()
```

```text
33554432
```

As expected, we reuse the cache, since the array fits into the allocated chunk.

However, if we attempt to do this with a larger tensor (3072 elements in the second dimension instead of 2048), we observe something different:

```python
allocate_empty_tensor(3072)
torch.cuda.memory_reserved()
```

```text
83886080
```

What happened? The chunk of memory that was allocated for a 4096x2048 array did not fit a tensor of size 4096x3072: thus, PyTorch needed to allocate an additional segment of a sufficient size while keeping the previous one allocated (in case the user creates a smaller tensor later at some point).

In practice, this means that if your code is dealing with tensors of dynamic size (changing batch sizes, sequence lengths or image resolutions), it is recommended to first warm up the cache by allocating tensors for the largest expected input. Otherwise, in the worst case you allocate a quadratic amount of memory with respect to the largest input size instead of a linear one.

You can also view more detailed allocation statistics by running `torch.cuda.memory_stats()`

```python
memory_stats = torch.cuda.memory_stats()
print(memory_stats["active.all.allocated"])
print(memory_stats["active.all.current"])
print(memory_stats["active.all.peak"])
print(memory_stats["reserved_bytes.all.current"])
```


```text
4
0
1
83886080
```


```python
torch.cuda.empty_cache()
print(torch.cuda.memory_stats()["reserved_bytes.all.current"])
```


```text
0
```


## Benchmarking intro
The simplest way to check the performance impact of any change is to compare the runtime of code with and without it. Here, we will consider a simple way of doing this in Python.

First, let's define two functions that compute a batched version of a dot product for two matrices:

```python
def batched_dot_mul_sum(a, b):
    """Computes batched dot by multiplying and summing"""
    return a.mul(b).sum(-1)


def batched_dot_bmm(a, b):
    """Computes batched dot by reducing to bmm"""
    a = a.reshape(-1, 1, a.shape[-1])
    b = b.reshape(-1, b.shape[-1], 1)
    return torch.bmm(a, b).flatten(-3)


# Input for benchmarking
x = torch.randn(10000, 64)

# Ensure that both functions compute the same output
assert batched_dot_mul_sum(x, x).allclose(batched_dot_bmm(x, x))
```

To conduct microbenchmarks by running the code hundreds of times and measuring the average execution time, you can use the built-in [timeit](https://docs.python.org/3/library/timeit.html) module. Simply create a Timer object with relevant arguments and call `.timeit()`:

```python
import timeit

t0 = timeit.Timer(
    stmt="batched_dot_mul_sum(x, x)",
    setup="from __main__ import batched_dot_mul_sum",
    globals={"x": x},
)

t1 = timeit.Timer(
    stmt="batched_dot_bmm(x, x)",
    setup="from __main__ import batched_dot_bmm",
    globals={"x": x},
)

print(f"mul_sum(x, x):  {t0.timeit(100) / 100 * 1e6:>5.1f} us")
print(f"bmm(x, x):      {t1.timeit(100) / 100 * 1e6:>5.1f} us")
```


```text
mul_sum(x, x):  499.3 us
bmm(x, x):      1228.2 us
```


In IPython, there exist line and cell timeit [magics](https://ipython.readthedocs.io/en/stable/interactive/magics.html#cell-magics). They can be more convenient for smaller cases but allow you a bit less control over the setup.

```python
%timeit batched_dot_mul_sum(x, x)
```


```text
369 µs ± 43 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
```


```python
%timeit batched_dot_bmm(x, x)
```


```text
859 µs ± 7.88 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
```


[torch.utils.benchmark](https://pytorch.org/docs/stable/benchmark_utils.html) copies the API of timeit while helping the user avoid common mistakes.

First, let's run it on the same code and data:

```python
import torch.utils.benchmark as benchmark

t0 = benchmark.Timer(
    stmt="batched_dot_mul_sum(x, x)",
    setup="from __main__ import batched_dot_mul_sum",
    globals={"x": x},
)

t1 = benchmark.Timer(
    stmt="batched_dot_bmm(x, x)",
    setup="from __main__ import batched_dot_bmm",
    globals={"x": x},
)

print(t0.timeit(100))
print(t1.timeit(100))
```


```text
<torch.utils.benchmark.utils.common.Measurement object at 0x7eed5d745f10>
batched_dot_mul_sum(x, x)
setup: from __main__ import batched_dot_mul_sum
  380.06 us
  1 measurement, 100 runs , 1 thread
<torch.utils.benchmark.utils.common.Measurement object at 0x7eed50ce0500>
batched_dot_bmm(x, x)
setup: from __main__ import batched_dot_bmm
  930.35 us
  1 measurement, 100 runs , 1 thread
```


In addition, we can set the number of threads for CPU computations:

```python
num_threads = torch.get_num_threads()
print(f"Benchmarking on {num_threads} threads")

t0 = benchmark.Timer(
    stmt="batched_dot_mul_sum(x, x)",
    setup="from __main__ import batched_dot_mul_sum",
    globals={"x": x},
    num_threads=num_threads,
    label="Multithreaded batch dot",
    sub_label="Implemented using mul and sum",
)

t1 = benchmark.Timer(
    stmt="batched_dot_bmm(x, x)",
    setup="from __main__ import batched_dot_bmm",
    globals={"x": x},
    num_threads=num_threads,
    label="Multithreaded batch dot",
    sub_label="Implemented using bmm",
)

print(t0.timeit(100))
print(t1.timeit(100))
```


```text
Benchmarking on 1 threads
<torch.utils.benchmark.utils.common.Measurement object at 0x7eed5de2ba40>
Multithreaded batch dot: Implemented using mul and sum
setup: from __main__ import batched_dot_mul_sum
  420.22 us
  1 measurement, 100 runs , 1 thread
<torch.utils.benchmark.utils.common.Measurement object at 0x7eed50ce0500>
Multithreaded batch dot: Implemented using bmm
setup: from __main__ import batched_dot_bmm
  872.06 us
  1 measurement, 100 runs , 1 thread
```


We can also change the thread count globally for PyTorch and measure the impact of that:

```python
prev_num_threads = num_threads
torch.set_num_threads(2)

num_threads = torch.get_num_threads()
print(f"Benchmarking on {num_threads} threads")

t0 = benchmark.Timer(
    stmt="batched_dot_mul_sum(x, x)",
    setup="from __main__ import batched_dot_mul_sum",
    globals={"x": x},
    num_threads=num_threads,
    label="Multithreaded batch dot",
    sub_label="Implemented using mul and sum",
)

t1 = benchmark.Timer(
    stmt="batched_dot_bmm(x, x)",
    setup="from __main__ import batched_dot_bmm",
    globals={"x": x},
    num_threads=num_threads,
    label="Multithreaded batch dot",
    sub_label="Implemented using bmm",
)

print(t0.timeit(100))
print(t1.timeit(100))
# in this case, we don't get any speedup, likely due to the overhead

torch.set_num_threads(prev_num_threads)
```


```text
Benchmarking on 2 threads
<torch.utils.benchmark.utils.common.Measurement object at 0x7eed5d7471d0>
Multithreaded batch dot: Implemented using mul and sum
setup: from __main__ import batched_dot_mul_sum
  333.36 us
  1 measurement, 100 runs , 2 threads
<torch.utils.benchmark.utils.common.Measurement object at 0x7eed5dfc8680>
Multithreaded batch dot: Implemented using bmm
setup: from __main__ import batched_dot_bmm
  661.51 us
  1 measurement, 100 runs , 2 threads
```


```python
# by the way, what CPU do we have?
!lscpu
```


```text
Architecture:                x86_64
  CPU op-mode(s):            32-bit, 64-bit
  Address sizes:             46 bits physical, 48 bits virtual
  Byte Order:                Little Endian
CPU(s):                      2
  On-line CPU(s) list:       0,1
Vendor ID:                   GenuineIntel
  Model name:                Intel(R) Xeon(R) CPU @ 2.20GHz
    CPU family:              6
    Model:                   79
    Thread(s) per core:      2
    Core(s) per socket:      1
    Socket(s):               1
    Stepping:                0
    BogoMIPS:                4399.99
    Flags:                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pg
                             e mca cmov pat pse36 clflush mmx fxsr sse sse2 ss h
                             t syscall nx pdpe1gb rdtscp lm constant_tsc rep_goo
                             d nopl xtopology nonstop_tsc cpuid tsc_known_freq p
                             ni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2ap
                             ic movbe popcnt aes xsave avx f16c rdrand hyperviso
                             r lahf_lm abm 3dnowprefetch ssbd ibrs ibpb stibp fs
                             gsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invp
                             cid rtm rdseed adx smap xsaveopt arat md_clear arch
                             _capabilities
Virtualization features:     
  Hypervisor vendor:         KVM
  Virtualization type:       full
Caches (sum of all):         
  L1d:                       32 KiB (1 instance)
  L1i:                       32 KiB (1 instance)
  L2:                        256 KiB (1 instance)
  L3:                        55 MiB (1 instance)
NUMA:                        
  NUMA node(s):              1
  NUMA node0 CPU(s):         0,1
Vulnerabilities:             
  Gather data sampling:      Not affected
  Indirect target selection: Vulnerable
  Itlb multihit:             Not affected
  L1tf:                      Mitigation; PTE Inversion
  Mds:                       Vulnerable; SMT Host state unknown
  Meltdown:                  Vulnerable
  Mmio stale data:           Vulnerable
  Reg file data sampling:    Not affected
  Retbleed:                  Vulnerable
  Spec rstack overflow:      Not affected
  Spec store bypass:         Vulnerable
  Spectre v1:                Vulnerable: __user pointer sanitization and usercop
                             y barriers only; no swapgs barriers
  Spectre v2:                Vulnerable; IBPB: disabled; STIBP: disabled; PBRSB-
                             eIBRS: Not affected; BHI: Vulnerable
  Srbds:                     Not affected
  Tsa:                       Not affected
  Tsx async abort:           Vulnerable
```


## Benchmarking GPU code
As we discussed in the lecture, CUDA kernel execution and PyTorch GPU operations are asynchronous.
This means that it is possible to launch several kernels in a row before receiving results from the first one. As a consequence, naive benchmarking without synchronization is likely to give you unrealistic results.

Let's try the same example, but with slightly larger matrices on the GPU:

```python
import timeit

x = torch.randn(10000, 1024, device="cuda")

t0 = timeit.Timer(
    stmt="batched_dot_mul_sum(x, x)",
    setup="from __main__ import batched_dot_mul_sum",
    globals={"x": x},
)

t1 = timeit.Timer(
    stmt="batched_dot_bmm(x, x)",
    setup="from __main__ import batched_dot_bmm",
    globals={"x": x},
)

# Ran each twice to show difference before/after warmup
print(f"mul_sum(x, x):  {t0.timeit(100) / 100 * 1e6:>5.1f} us")
print(f"mul_sum(x, x):  {t0.timeit(100) / 100 * 1e6:>5.1f} us")
print(f"bmm(x, x):      {t1.timeit(100) / 100 * 1e6:>5.1f} us")
print(f"bmm(x, x):      {t1.timeit(100) / 100 * 1e6:>5.1f} us")
```


```text
mul_sum(x, x):  196.9 us
mul_sum(x, x):   20.9 us
bmm(x, x):      1046.4 us
bmm(x, x):       25.7 us
```


First, we see that the difference between the first and the second runs of timeit is quite noticeable, which happens because of initializing the CUDA context and loading the kernels to perform the GPU computations we need.

Second, the runtimes of two methods seem too small.

Let's run the same test with `torch.utils.benchmark` to see a different set of results.

```python
t0 = benchmark.Timer(
    stmt="batched_dot_mul_sum(x, x)",
    setup="from __main__ import batched_dot_mul_sum",
    globals={"x": x},
)

t1 = benchmark.Timer(
    stmt="batched_dot_bmm(x, x)",
    setup="from __main__ import batched_dot_bmm",
    globals={"x": x},
)

# Run only once since benchmark module does warmup for us
print(t0.timeit(100))
print(t1.timeit(100))
```


```text
<torch.utils.benchmark.utils.common.Measurement object at 0x7eed5de2ba40>
batched_dot_mul_sum(x, x)
setup: from __main__ import batched_dot_mul_sum
  505.35 us
  1 measurement, 100 runs , 1 thread
<torch.utils.benchmark.utils.common.Measurement object at 0x7eed5dfc8680>
batched_dot_bmm(x, x)
setup: from __main__ import batched_dot_bmm
  189.37 us
  1 measurement, 100 runs , 1 thread
```


Now we have a more realistic set of measurements, which is caused by explicitly triggering the CPU-GPU synchronization and awaiting the results after launching the multiplication. The runtime is comparable to what we had in a previous set of benchmarks, but pay attention that now the matrices being multiplied are 16 times bigger.

Let's implement microbenchmarking by ourselves, using nothing but [`time.perf_counter()`](https://docs.python.org/3/library/time.html#time.perf_counter) and CUDA methods given by PyTorch. First, we'll run `batched_dot_mul_sum` on the GPU with and without synchronization and see how this affects the results:

```python
from time import perf_counter

import numpy as np

execution_times = []

for _ in range(100):
    start_time = perf_counter()
    batched_dot_mul_sum(x, x)
    execution_times.append(perf_counter() - start_time)

np.mean(execution_times).item()
```

```text
2.9924039998832087e-05
```

Compare this with the result that does not compute anything:

```python
execution_times = []

for _ in range(100):
    start = perf_counter()
    execution_times.append(perf_counter() - start)

np.mean(execution_times).item()
```

```text
1.3220000255387277e-07
```

Finally, let's explicitly call `torch.cuda.synchronize` at the end of each iteration to see the difference.

```python
execution_times = []

for _ in range(100):
    start_time = perf_counter()
    batched_dot_mul_sum(x, x)
    torch.cuda.synchronize()
    execution_times.append(perf_counter() - start_time)

np.mean(execution_times).item()
```

```text
0.0008708237700079735
```

The same thing applies to actual models. Let's take `Linear` as the simplest example:

```python
x = torch.randn(10000, 512, device="cuda")
linear = torch.nn.Linear(512, 1024, bias=False, device="cuda")
N_ITERS = 200

execution_times = []

for _ in range(N_ITERS):
    start = perf_counter()
    result = linear(x)
    execution_times.append(perf_counter() - start)

np.mean(execution_times).item()
```

```text
5.557694499884747e-05
```

```python
execution_times = []

for _ in range(N_ITERS):
    start = perf_counter()
    result = linear(x)
    torch.cuda.synchronize()
    execution_times.append(perf_counter() - start)

np.mean(execution_times).item()
```

```text
0.005169328135005457
```

```python
execution_times = []

start = perf_counter()
for _ in range(N_ITERS):
    result = linear(x)
torch.cuda.synchronize()

(perf_counter() - start) / N_ITERS
```

```text
0.0025256919999998217
```

Of course, `torch.utils.benchmark` also works:

```python
t0 = benchmark.Timer(stmt="linear(x)", globals={"x": x, "linear": linear})
print(t0.timeit(100))
```


```text
<torch.utils.benchmark.utils.common.Measurement object at 0x7eed50d4dcd0>
linear(x)
  2.61 ms
  1 measurement, 100 runs , 1 thread
```


## CUDA Streams
To execute several operations concurrently, you may use CUDA streams in PyTorch. Below, you can see an example of their usage.

```python
cuda = torch.device("cuda")
s = torch.cuda.Stream()  # Create a new stream.
A = torch.empty((1000, 1000), device=cuda).normal_(0.0, 1.0)
with torch.cuda.stream(s):
    # sum() may start execution before normal_() finishes!
    B = torch.sum(A)
```

```python
s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()
# Initialize cuda tensors here. E.g.:
A = torch.rand(1000, 1000, device="cuda")
B = torch.rand(1000, 1000, device="cuda")
# Wait for the above tensors to initialize.
torch.cuda.synchronize()

execution_times = []

for _ in range(100):
    start = perf_counter()
    with torch.cuda.stream(s1):
        C = torch.mm(A, A)
    with torch.cuda.stream(s2):
        D = torch.mm(B, B)
    # Wait for C and D to be computed.
    torch.cuda.synchronize()
    execution_times.append(perf_counter() - start)

np.mean(execution_times).item()
```

```text
0.0010837480900011088
```

```python
# next, let's compute C and D sequentially

execution_times = []

for _ in range(100):
    start = perf_counter()
    C = torch.mm(A, A)
    D = torch.mm(B, B)
    # Wait for C and D to be computed.
    torch.cuda.synchronize()
    execution_times.append(perf_counter() - start)

np.mean(execution_times).item()
# the speed is even higher in this case
```

```text
0.0010658494500034976
```

As you can see in this example, the usefulness of streams can be limited: concurrent kernels need to underutilize the GPU and yet take long enough to compute. It might still be a good idea if you are trying to compute some expression and fetch data for the next computation simultaneously (for example, when training samples are large, with distributed training, or in case of offloading).
The example below demonstrates how to use streams for that purpose.

```python
compute_stream = torch.cuda.Stream()
h2d_stream = torch.cuda.Stream()

A = torch.rand(10240, 10240, device="cuda")
B = [torch.rand(4096, 4096, device="cpu", pin_memory=True) for _ in range(100)]
```

```python
def sequential_execution():
  torch.matmul(A, A)
  for matrix in B:
    matrix.to("cuda", non_blocking=True)
```

```python
%%timeit
sequential_execution()
torch.cuda.synchronize()
```


```text
1.07 s ± 35.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```


```python
def stream_execution():
  with torch.cuda.stream(compute_stream):
    torch.matmul(A, A)
  with torch.cuda.stream(h2d_stream):
    for matrix in B:
      matrix.to("cuda")
```

```python
%%timeit
stream_execution()
torch.cuda.synchronize()
```


```text
576 ms ± 42.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```


## Debugging asynchronous code
Finding sources of errors in GPU-reliant code in PyTorch can also be difficult. Let's see this on a standard example of an off-by-one error and incorrect index for the embedding layer:

```python
%%writefile incorrect_index.py
import os

import torch
import torch.nn as nn

device = os.getenv("DEVICE", "cuda")
assert device in {"cpu", "cuda"}

embedding = nn.Embedding(1024,32).to(device)
# 1024 > 1023 (largest index in the created embedding layer)
input = torch.full((1,1),1024,dtype=torch.long, device=device)

# out-of-bounds access
embedding_for_index = embedding(input)

result = torch.sigmoid(embedding_for_index)
loss = result.sum()
print(loss.item())
print(loss)
```


```text
Overwriting incorrect_index.py
```


If we attempt to run the code as is, we'll see an error after the point in which we trigger the CPU-GPU synchronization (`loss.item`). This gives us no clues about what exactly caused an error:

```python
!CUDA_LAUNCH_BLOCKING=1 python incorrect_index.py
```


```text
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1478: indexSelectSmallIndex: block: [0,0,0], thread: [0,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1478: indexSelectSmallIndex: block: [0,0,0], thread: [1,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1478: indexSelectSmallIndex: block: [0,0,0], thread: [2,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1478: indexSelectSmallIndex: block: [0,0,0], thread: [3,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1478: indexSelectSmallIndex: block: [0,0,0], thread: [4,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1478: indexSelectSmallIndex: block: [0,0,0], thread: [5,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1478: indexSelectSmallIndex: block: [0,0,0], thread: [6,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1478: indexSelectSmallIndex: block: [0,0,0], thread: [7,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1478: indexSelectSmallIndex: block: [0,0,0], thread: [8,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1478: indexSelectSmallIndex: block: [0,0,0], thread: [9,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1478: indexSelectSmallIndex: block: [0,0,0], thread: [10,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1478: indexSelectSmallIndex: block: [0,0,0], thread: [11,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1478: indexSelectSmallIndex: block: [0,0,0], thread: [12,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1478: indexSelectSmallIndex: block: [0,0,0], thread: [13,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1478: indexSelectSmallIndex: block: [0,0,0], thread: [14,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1478: indexSelectSmallIndex: block: [0,0,0], thread: [15,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1478: indexSelectSmallIndex: block: [0,0,0], thread: [16,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1478: indexSelectSmallIndex: block: [0,0,0], thread: [17,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1478: indexSelectSmallIndex: block: [0,0,0], thread: [18,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1478: indexSelectSmallIndex: block: [0,0,0], thread: [19,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1478: indexSelectSmallIndex: block: [0,0,0], thread: [20,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1478: indexSelectSmallIndex: block: [0,0,0], thread: [21,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1478: indexSelectSmallIndex: block: [0,0,0], thread: [22,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1478: indexSelectSmallIndex: block: [0,0,0], thread: [23,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1478: indexSelectSmallIndex: block: [0,0,0], thread: [24,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1478: indexSelectSmallIndex: block: [0,0,0], thread: [25,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1478: indexSelectSmallIndex: block: [0,0,0], thread: [26,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1478: indexSelectSmallIndex: block: [0,0,0], thread: [27,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1478: indexSelectSmallIndex: block: [0,0,0], thread: [28,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1478: indexSelectSmallIndex: block: [0,0,0], thread: [29,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1478: indexSelectSmallIndex: block: [0,0,0], thread: [30,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1478: indexSelectSmallIndex: block: [0,0,0], thread: [31,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
Traceback (most recent call last):
  File "/content/incorrect_index.py", line 14, in <module>
    embedding_for_index = embedding(input)
                          ^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/sparse.py", line 192, in forward
    return F.embedding(
           ^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/functional.py", line 2542, in embedding
    return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```


(though an experienced person might get a hint from the failed assertions printed before the error)

There are two ways to find the actual line that caused an exception:

* First, you can just move everything to CPU. This is a valid approach, but it involves making changes to your code or input arguments and sometimes can be difficult (for example, if the error occurs after a long chain of operations).

* Second, you may use the `CUDA_LAUNCH_BLOCKING` environment variable when starting your code. This will force synchronization for all GPU-related operations, making your code slower but allowing to see the exact source of the error.

```python
!DEVICE=cpu python incorrect_index.py
```


```text
Traceback (most recent call last):
  File "/content/incorrect_index.py", line 14, in <module>
    embedding_for_index = embedding(input)
                          ^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/sparse.py", line 192, in forward
    return F.embedding(
           ^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/functional.py", line 2542, in embedding
    return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
IndexError: index out of range in self
```


## Precision of floating point operations
Let's have a look at this code snippet and its results. In essence, it computes the sixth power of a random matrix (with a fixed seed) on two different devices.

```python
torch.manual_seed(1337)
x = torch.randn(5000, 5000)
torch.use_deterministic_algorithms(False)


def matrix_power(x):
    y = x @ x @ x @ x @ x @ x
    return y.sum().item()


print(matrix_power(x))
print(matrix_power(x.cuda()))
```


```text
27654770130944.0
27654807879680.0
```


(in case you are wondering, `torch.use_deterministic_algorithms(True)` won't help)

If we do the same using numpy (in two ways), we also get a different result:

```python
print(matrix_power(x.numpy()))
np.linalg.matrix_power(x.numpy(), 6).sum()
```


```text
27654791102464.0
```


```text
np.float32(27654774000000.0)
```

Takeaway: numerical precision of floating point computations can vary between libraries, environments and devices, and from the user side, it is often hard to resolve this issue altogether. Usually, this happens due to a different summation order in code or due to inherent nondeterminism of hardware.

However, note that the relative error is small enough, which makes such blatant discrepancies less of a problem in regular deep learning code.

## CUDA Graphs

In some situations, the bottleneck of your code might be not the GPU compute performance and not even the memory bandwidth, but the latency of kernel execution. Launching each kernel takes CPU time, and if the kernels themselves run very fast, these launches can become a problem.

Fortunately, [CUDA graphs](https://github.com/pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) can address this problem. CUDA graphs capture a sequence of specific operations (i.e., kernel launches) that can later be replayed on new inputs with essentially a single kernel launch. Below, you can see an example of using graphs with a function that consists of many fast-running operations:

```python
def slow_function(x):
  for _ in range(500):
    y = x*2
    z = torch.sigmoid(y)
    a = z + 5.0
    b = torch.nn.functional.relu(a)
    x = b
  return b

def my_function2(x):
    for _ in range(59):
        y = x*float(1000)
        b = torch.sigmoid(y)
    return b

# the code below is a modified version of https://pytorch.org/docs/master/notes/cuda.html#cuda-graph-semantics

# Placeholder input used for capture
static_input = torch.ones((6,), device="cuda")
print(slow_function(static_input))
```


```text
tensor([6.0000, 6.0000, 6.0000, 6.0000, 6.0000, 6.0000], device='cuda:0')
```


Below, you can see two examples of running complex_function: directly (the standard way) or by capturing `func_graph`, copying new inputs into `static_input` and `replay()`ing it. As you can see, the second version can be alsmost 10x faster compared to default eager execution.

```python
%%timeit
slow_function(static_input)
torch.cuda.synchronize()
```


```text
19.7 ms ± 2.27 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
```


```python
func_graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(func_graph):
  static_output = slow_function(static_input)
static_output.zero_()

# Fills the graph's input memory with new data to compute on
static_input.copy_(torch.full((6,), 3, device="cuda"))
func_graph.replay()
print(static_output)
```


```text
tensor([6.0000, 6.0000, 6.0000, 6.0000, 6.0000, 6.0000], device='cuda:0')
```


```python
%%timeit
func_graph.replay()
torch.cuda.synchronize()
```


```text
3 ms ± 23.7 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```
