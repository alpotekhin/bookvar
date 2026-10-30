---
title: "Serving engines — vLLM, SGLang, TensorRT-LLM, and FlashInfer"
type: textbook-chapter
status: canonical
content_mode: attributed-original-excerpts
locale: en
translation_status: current
canonical_revision: "fee6d97ff4f17a23b48f6b3a73e7311d7699f1da456c414094d4ba934062f255"
translation_reviewed: "2026-10-29"
translation_of: "00 Учебник/14 Inference и оптимизация/55c Serving engines — vLLM, SGLang, TensorRT-LLM и FlashInfer.md"
last_updated: 2026-10-29
last_verified: 2026-10-29
---

<a id="serving-engines--vllm-sglang-tensorrt-llm-и-flashinfer"></a>

# Serving engines — vLLM, SGLang, TensorRT-LLM, and FlashInfer

Loading weights and a tokenizer does not yet make a model into a service. A single `generate` call can process a prompt and emit an answer sequentially, but a production system must concurrently accept requests of different lengths, change batch membership after every step, allocate memory for growing KV caches, stream tokens to clients, and release resources when sequences finish. A serving engine is the program that coordinates all these operations with GPU execution.

vLLM, SGLang, TensorRT-LLM, and FlashInfer often appear in the same comparison table even though they occupy different levels of the stack. vLLM and SGLang provide complete runtimes and servers. TensorRT-LLM includes a PyTorch execution backend as well as a TensorRT-engine
path, with optimized operators closely coupled to NVIDIA's stack. FlashInfer is primarily a library of kernels for attention, sampling, and MoE that another runtime may employ. A meaningful comparison begins by separating the layers of an inference system.

<a id="пять-уровней-одной-системы"></a>

## Five layers of one system

It is more useful to begin with the questions the system must answer than with product names.

| Layer | Principal question | Example mechanisms |
|---|---|---|
| API and frontend | how should requests be accepted, validated, and returned as token streams? | OpenAI-compatible HTTP API, tokenizer pool |
| Scheduler and cache manager | which requests should run now, and where should their state reside? | continuous batching, PagedAttention, RadixAttention |
| Model executor | how should the model be partitioned and one step launched on the devices? | workers, CUDA Graphs, tensor/pipeline parallelism |
| Attention and operator backends | how should attention, GEMM, normalization, and sampling execute? | FlashAttention, FlashInfer, CUTLASS, Triton kernels |
| Hardware runtime | how should streams, memory, collectives, and graphs be managed? | CUDA, NCCL, TensorRT |

The layers interact but do not substitute for one another. A fast attention kernel does not solve request queueing. A sophisticated scheduler cannot compensate for a kernel that moves the same tensor between HBM and SRAM several times. A high prefix-cache hit rate is of little use if slow tokenization blocks the frontend's engine loop.


The following sections reproduce the developers' explanations at pinned revisions.
Defaults and implementation names refer to those revisions. Bookvar's worked
examples are labeled separately; quoted performance claims are not local measurements.

<a id="vllm-scheduler-и-блочная-память-в-центре-архитектуры"></a>

<a id="vllm-граница-между-сервером-и-исполнением-модели"></a>

## vLLM: Architecture Overview

*vLLM contributors, [Architecture Overview, v0.31.0](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/docs/design/arch_overview.md), [Apache-2.0](/licenses/vllm-db9527a4-Apache-2.0.txt). The following original sections are reproduced without translating or rewriting their prose. Heading levels and links are adapted to this book.*

### V1 Process Architecture

vLLM V1 uses a multi-process architecture to separate concerns and maximize throughput. Understanding this architecture is important for properly sizing CPU resources in your deployment. The key processes are:

#### API Server Process

The API server process handles HTTP requests (e.g., the OpenAI-compatible API), performs input processing (tokenization, multi-modal data loading), and streams results back to clients. It communicates with the engine core process(es) via ZMQ sockets.

By default, there is **1 API server process**, but when data parallelism is used, the API server count automatically scales to match the data parallel size. This can also be manually configured with the `--api-server-count` flag. Each API server connects to **all** engine cores via ZMQ in a many-to-many topology, enabling any API server to route requests to any engine core. Each API server process uses multiple CPU threads for media loading (controlled by `VLLM_MEDIA_LOADING_THREAD_COUNT`, default 8).

The code can be found in [vllm/entrypoints/launchers/api_server](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/entrypoints/launchers/api_server) and [vllm/v1/utils.py](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/utils.py).

#### Engine Core Process

The engine core process runs the scheduler, manages KV cache, and coordinates model execution across GPU workers. It runs a busy loop that continuously schedules requests and dispatches work to the GPU workers.

There is **1 engine core process per data parallel rank**. For example, with `--data-parallel-size 4`, there are 4 engine core processes.

The code can be found in [vllm/v1/engine/core.py](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/engine/core.py) and [vllm/v1/engine/utils.py](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/engine/utils.py).

#### GPU Worker Processes

Each GPU is managed by a dedicated worker process. The worker process loads model weights, executes forward passes, and manages GPU memory. Workers communicate with the engine core process that owns them.

There is **1 worker process per GPU**. The total number of GPU worker processes equals `tensor_parallel_size x pipeline_parallel_size` per engine core.

The code can be found in [vllm/v1/executor/multiproc_executor.py](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/executor/multiproc_executor.py) and [vllm/v1/worker/gpu_worker.py](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/worker/gpu_worker.py).

#### DP Coordinator Process (conditional)

When using data parallelism (`--data-parallel-size > 1`), an additional coordinator process manages load balancing across DP ranks and coordinates synchronized forward passes for MoE models.

There is **1 DP coordinator process** (only when data parallelism is enabled).

The code can be found in [vllm/v1/engine/coordinator.py](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/engine/coordinator.py).

#### Process Count Summary

For a deployment with `N` GPUs, `TP` tensor parallel size, `DP` data parallel size, and `A` API server count:

| Process Type | Count | Notes |
| - | - | - |
| API Server | `A` (default `DP`) | Handles HTTP requests and input processing |
| Engine Core | `DP` (default 1) | Scheduler and KV cache management |
| GPU Worker | `N` (= `DP x PP x TP`) | One per GPU, executes model forward passes |
| DP Coordinator | 1 if `DP > 1`, else 0 | Load balancing across DP ranks |
| **Total** | **`A + DP + N` (+ 1 if DP > 1)** | |

For example, a typical single-node deployment with 4 GPUs (`vllm serve -tp=4`) has:

- 1 API server + 1 engine core + 4 GPU workers = **6 processes**

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/serving-engines-review-2026-10/vllm-v1-processes-tp4.png]]

<!-- source-caption:start -->
*vLLM contributors. [V1 Process Architecture — TP=4](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/docs/design/arch_overview.md). [Apache-2.0](/licenses/vllm-db9527a4-Apache-2.0.txt).*
<!-- source-caption:end -->

A data parallel deployment with 8 GPUs (`vllm serve -tp=2 -dp=4`) has:

- 4 API servers + 4 engine cores + 8 GPU workers + 1 DP coordinator = **17 processes**

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/serving-originals-2026-10/vllm-tp2-dp4.png]]

<!-- source-caption:start -->
*vLLM contributors. [V1 Process Architecture — TP=2, DP=4](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/docs/design/arch_overview.md). [Apache-2.0](/licenses/vllm-db9527a4-Apache-2.0.txt).*
<!-- source-caption:end -->

For CPU resource sizing recommendations, see
[CPU Resources for GPU Deployments](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/docs/configuration/optimization.md#cpu-resources-for-gpu-deployments).

### Worker

A worker is a process that runs the model inference. vLLM follows the common
practice of using one process to control one accelerator device, such as GPUs.
For example, if we use tensor parallelism of size 2 and pipeline parallelism of
size 2, we will have 4 workers in total. Workers are identified by their
`rank` and `local_rank`. `rank` is used for global orchestration, while
`local_rank` is mainly used for assigning the accelerator device and accessing
local resources such as the file system and shared memory.

### Model Runner

Every worker has one model runner object, responsible for loading and running
the model. Much of the model execution logic resides here, such as preparing
input tensors and capturing cudagraphs.

### Model

Every model runner object has one model object, which is the actual
`torch.nn.Module` instance. See [huggingface_integration](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/docs/design/huggingface_integration.md) for how various
configurations affect the class we ultimately get.

### Sharding and Quantization at Initialization

Certain features require
changing the model weights. For example, tensor parallelism needs to shard the
model weights, and quantization needs to quantize the model weights. There are
two possible ways to implement this feature. One way is to change the model
weights after the model is initialized. The other way is to change the model
weights during the model initialization. vLLM chooses the latter. The first
approach is not scalable to large models. Suppose we want to run a 405B model
(with roughly 810GB weights) with 16 H100 80GB GPUs. Ideally, every GPU should
only load 50GB weights. If we change the model weights after the model is
initialized, we need to load the full 810GB weights to every GPU and then shard
the weights, leading to a huge memory overhead. Instead, if we shard the weights
during the model initialization, every layer will only create a shard of the
weights it needs, leading to a much smaller memory overhead. The same idea
applies to quantization. Note that we also add an additional argument `prefix`
to the model's constructor so that the model can initialize itself differently
based on the prefix. This is useful for non-uniform quantization, where
different parts of the model are quantized differently. The `prefix` is
usually an empty string for the top-level model and a string like `"vision"`
or `"language"` for the sub-models. In general, it matches the name of the
module's state dict in the checkpoint file.

<a id="sglang-radixattention-и-перекрытие-cpugpu"></a>

<a id="sglang-структура-программы-становится-частью-расписания"></a>

<a id="sglang-язык-программ-и-самостоятельный-сервер"></a>

## SGLang: RadixAttention and CPU/GPU overlap

*Lianmin Zheng et al., [SGLang and RadixAttention, 17 January 2024](https://github.com/lm-sys/lm-sys.github.io/blob/88c036dd1a78c8f2525782f98ee4493a63d7a532/blog/2024-01-17-sglang.md), and the SGLang Team, [SGLang v0.4, 4 December 2024](https://github.com/lm-sys/lm-sys.github.io/blob/88c036dd1a78c8f2525782f98ee4493a63d7a532/blog/2024-12-04-sglang-v0-4.md); [MIT](/licenses/lmsys-88c036dd-MIT.txt). Original mechanism explanations follow.*

### RadixAttention

The first excerpt describes the original 2024 implementation, including its
one-token pages. That page size is not a universal default for later SGLang
releases. Paging controls placement; the radix tree discovers matching prefixes.

To systematically exploit these reuse opportunities, we introduce RadixAttention, a novel technique for automatic KV cache reuse during runtime. Instead of discarding the KV cache after finishing a generation request, our approach retains the KV cache for both prompts and generation results in a radix tree. This data structure enables efficient prefix search, insertion, and eviction. We implement a Least Recently Used (LRU) eviction policy, complemented by a cache-aware scheduling policy, to enhance the cache hit rate.

A radix tree is a data structure that serves as a space-efficient alternative to a trie (prefix tree). Unlike typical trees, the edges of a radix tree can be labeled with not just single elements, but also with sequences of elements of varying lengths. This feature boosts the efficiency of radix trees. In our system, we utilize a radix tree to manage a mapping. This mapping is between sequences of tokens, which act as the keys, and their corresponding KV cache tensors, which serve as the values. These KV cache tensors are stored on the GPU in a paged layout, where the size of each page is equivalent to one token. Considering the limited capacity of GPU memory, we cannot retain infinite KV cache tensors, which necessitates an eviction policy. To tackle this, we implement an LRU eviction policy that recursively evicts leaf nodes.
Furthermore, RadixAttention is compatible with existing techniques like continuous batching and paged attention.
For multi-modal models, the RadixAttention can be easily extended to handle image tokens.

### Overlapped CPU scheduler and GPU worker

While LLM inference runs on GPUs, there is substantial work that also needs to be done by the CPU, such as batch scheduling, memory allocation, and prefix matching. An unoptimized inference engine can spend as much as [half of its time on CPU overhead](https://mlsys.wuklab.io/posts/scheduling_overhead/). SGLang has been known for its efficient batch scheduler from the start. In this new version, we pushed it to the extreme and achieved a near zero-overhead batch scheduler. This idea is simple and has been proposed in [NanoFlow](https://arxiv.org/abs/2408.12757). Basically, we can overlap the CPU scheduling with the GPU computation. The scheduler runs one batch ahead and prepares all the metadata required for the next batch. By doing this, we can keep the GPUs always busy and hide expensive overheads such as the radix cache operations. The related code is [here](https://github.com/sgl-project/sglang/blob/85e1a6f3aa5a2288ca85fe3fe922c733b6533fa7/python/sglang/srt/managers/scheduler.py#L399). The implementation details involve resolving dependencies by creating future tokens and carefully scheduling CUDA events and synchronization. Below is an illustration of the overlapped CPU scheduler and GPU worker.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/serving-originals-2026-10/sglang-overlap-scheduler.jpg]]

*The SGLang Team, Zero-Overhead Batch Scheduler, SGLang v0.4;
[MIT](/licenses/lmsys-88c036dd-MIT.txt). “Zero-overhead” is the authors' optimization
name, not a claim that the CPU performs no work.*

SGLang also runs as a standalone HTTP server: a client-side language-model
program is not required. See the pinned [v0.5.21 OpenAI-compatible API](https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/docs/docs/basic_usage/openai_api.mdx)
and [overlap loop](https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/srt/managers/scheduler.py#L1941).

<a id="tensorrt-llm-два-пути-исполнения-внутри-одного-проекта"></a>

## TensorRT-LLM: Architecture Overview

*NVIDIA Corporation & affiliates, [Architecture Overview, v1.2.1](https://github.com/NVIDIA/TensorRT-LLM/blob/376f7e1bd8ed543f75014309e3fd4b237e9b0e73/docs/source/developer-guide/overview.md), [Apache-2.0](/licenses/tensorrt-llm-376f7e1b-LICENSE.txt). The original page body, code, and diagram are retained.*

This is the PyTorch/PyExecutor path. Building a TensorRT engine is a
[separate workflow](https://github.com/NVIDIA/TensorRT-LLM/blob/376f7e1bd8ed543f75014309e3fd4b237e9b0e73/docs/source/legacy/architecture/workflow.md).
The 22% figure below is the authors' result on selected models and hardware;
it was not reproduced for this book.

The `LLM` class is a core entry point for the TensorRT LLM, providing a simplified `generate()` API for efficient large language model inference. This abstraction aims to streamline the user experience, as demonstrated with TinyLlama:

```python
from tensorrt_llm import LLM

# Initialize the LLM with a specified model
llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Generate text using the model
output = llm.generate("Hello, my name is")
```

The `LLM` class automatically manages essential pre and post-processing steps, including tokenization (encoding input prompts into numerical representations) and detokenization (decoding model outputs back into human-readable text).

Internally, the `LLM` class orchestrates the creation of a dedicated `PyExecutor(Worker)` process on each rank.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/serving-engines-review-2026-10/tensorrt-llm-pyexecutor.png]]

<!-- source-caption:start -->
*NVIDIA Corporation & affiliates. [TensorRT LLM Architecture Overview](https://github.com/NVIDIA/TensorRT-LLM/blob/376f7e1bd8ed543f75014309e3fd4b237e9b0e73/docs/source/developer-guide/overview.md). [Apache-2.0](/licenses/tensorrt-llm-376f7e1b-LICENSE.txt).*
<!-- source-caption:end -->

This `PyExecutor` operates in a continuous background loop, designed for the efficient, asynchronous processing of inference requests.

The `PyExecutor`'s functionality is built upon several key components:

- `Scheduler`: Responsible for determining which active requests are ready for execution at each processing step.

- `KVCacheManager`: Manages the allocation, deallocation, and maintenance of the Key-Value (KV) Cache. This is a critical optimization for Transformer models, significantly enhancing performance during autoregressive text generation by storing previously computed attention keys and values.

- `ModelEngine`: Handles the loading and highly efficient execution of the language model on the GPU hardware.

- `Sampler`: Takes the raw outputs (logits) from the ModelEngine and applies appropriate sampling strategies (e.g., greedy, top-k, top-p, beam search) to generate the final output tokens.

During each iteration of its background loop, the `PyExecutor` performs the following sequence of operations:

- Request Fetching: Retrieves new inference requests from an internal request queue, if available.

- Scheduling: Interacts with the `Scheduler` to identify and prioritize requests that are ready to be processed in the current step.

- Resource Preparation: Coordinates with the `KVCacheManager` to ensure that the necessary Key-Value (KV) Cache resources are allocated for the selected requests.

- Model Execution: Invokes the `ModelEngine` to perform a forward pass on the scheduled requests, predicting the next output tokens.

- Output Handling: Updates the partial outputs for ongoing requests and finalizes the results for any requests that have reached completion, returning them to the user.


### Runtime Optimizations

TensorRT LLM enhances inference throughput and reduces latency by integrating a suite of runtime optimizations, including CUDA Graph, [Overlap Scheduler](https://github.com/NVIDIA/TensorRT-LLM/blob/376f7e1bd8ed543f75014309e3fd4b237e9b0e73/docs/source/features/overlap-scheduler.md), [Speculative decoding](https://github.com/NVIDIA/TensorRT-LLM/blob/376f7e1bd8ed543f75014309e3fd4b237e9b0e73/docs/source/features/speculative-decoding.md), etc.

#### CUDA Graph

CUDA Graphs drastically reduce the CPU-side overhead associated with launching GPU kernels, which is particularly impactful in PyTorch-based inference where Python's host-side code can be a bottleneck. By capturing a sequence of CUDA operations as a single graph, the entire sequence can be launched with one API call, minimizing CPU-GPU synchronization and driver overhead.

To maximize the "hit rate" of these cached graphs, TensorRT LLM employs CUDA Graph padding. If an incoming batch's size doesn't match a captured graph, it's padded to the nearest larger, supported size for which a graph exists. While this incurs minor overhead from computing "wasted" tokens, it's often a better trade-off than falling back to slower eager mode execution. This optimization has a significant impact, demonstrating up to a 22% end-to-end throughput increase on certain models and hardware.

#### Overlap Scheduler

The Overlap Scheduler maximizes GPU utilization by hiding CPU-bound latency behind GPU computation.

The key strategy is to launch the GPU's work for the next step (n+1) immediately, without waiting for the CPU to finish processing the results of the current step (n). This allows the CPU to handle tasks like checking stop criteria or updating responses for one batch while the GPU is already executing the model for the subsequent batch.

This concurrent execution pipeline is illustrated in the `PyExecutor`'s logic:

```python
# Schedule and launch GPU work for the current step (n)
scheduled_batch, _, _ = self._schedule()
batch_outputs = self._forward_step(scheduled_batch, previous_tensors_device)
sample_state = self._sample_async(scheduled_batch, batch_outputs)

# While the GPU is busy, process the CPU-bound results from the previous step (n-1)
if self.previous_batch is not None:
    self._process_previous_batch()
```

This approach effectively reduces GPU idle time and improves overall hardware occupancy. While it introduces one extra decoding step into the pipeline, the resulting throughput gain is a significant trade-off. For this reason, the Overlap Scheduler is enabled by default in TensorRT LLM.

<a id="flashinfer-библиотека-ядер-а-не-ещё-один-scheduler"></a>

<a id="flashinfer-что-движок-передаёт-библиотеке-ядер"></a>

## FlashInfer: KV-Cache Layout

*FlashInfer contributors, [KV-Cache Layout, v0.7.1](https://github.com/flashinfer-ai/flashinfer/blob/0b46115d238d29ea2e92c6b9b2cb077d40fbc352/docs/tutorials/kv_layout.rst), [Apache-2.0](/licenses/flashinfer-0b46115d-LICENSE.txt). Original sections are reproduced with reStructuredText converted to Markdown. Figures are from the [official illustration repository](https://github.com/flashinfer-ai/web-data/tree/d7cc3d229f23d7c0a81c1e45d8591f4337c18efb/tutorials), [Apache-2.0](/licenses/flashinfer-web-data-d7cc3d22-Apache-2.0.txt).*

### Layout: NHD/HND

FlashInfer provides two layouts for last 3 dimensions in KV-Cache: `NHD` and `HND`:

- `NHD`: the last 3 dimensions are organized as `(seq_len, num_heads, head_dim)`.
- `HND`: the last 3 dimensions are organized as `(num_heads, seq_len, head_dim)`.

The `NHD` layout is more natural because it's consistent with the output of
$xW_k$ and $xW_v$ without transpose. The `HND` layout is more friendly
for GPU implementation when KV-Cache uses low-precision data type (e.g. fp8).
In practice we don't observe significant performance difference between these two layouts
on fp16 kV-Cache and we prioritize `NHD` layout for better readability. FlashInfer implements
Attention kernels on both layouts and we provide an option to select between them (`NHD`
by default).


### Ragged Tensor

We use Ragged Tensor to store the variable length Q/K/V tensors in FlashInfer for batch prefill self-attention:

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/serving-engines-review-2026-10/flashinfer-ragged.png]]

<!-- source-caption:start -->
*FlashInfer contributors. [Ragged Tensor](https://github.com/flashinfer-ai/flashinfer/blob/0b46115d238d29ea2e92c6b9b2cb077d40fbc352/docs/tutorials/kv_layout.rst). [Apache-2.0](/licenses/flashinfer-web-data-d7cc3d22-Apache-2.0.txt).*
<!-- source-caption:end -->

In Ragged Tensor, all requests's Q/K/V are packed into a single `data` tensor without padding,
we use a `indptr` array (`num_requests+1` elements, the first element is always zero)
to store the information of variable sequence lengths of each request
(`indptr[i+1]-indptr[i]` is the sequence length of request `i`), the `data` tensor has
shape `(indptr[-1], num_heads, head_dim)` when the layout is `NHD`.

We can use `data[indptr[i]:indptr[i+1]]` to slice the keys (or values) of request `i`.


> `indptr` arrays across the flashinfer library should be of type `int32`. Arrays of type `int64` can cause indexing errors.

#### FlashInfer APIs

FlashInfer provides `flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper` to compute
the prefill attention between queries stored in ragged tensor and keys/values stored in ragged
KV-Cache.

### Mask Layout (2D Ragged Tensor)

The aforementioned Ragged Tensor can be generalized to multiple "ragged" dimensions. For example,
the attention mask in FlashInfer is a 2D ragged tensor for batch size greater than 1:

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/serving-originals-2026-10/flashinfer-mask-layout.png]]

<!-- source-caption:start -->
*FlashInfer contributors. [Mask Layout](https://github.com/flashinfer-ai/flashinfer/blob/0b46115d238d29ea2e92c6b9b2cb077d40fbc352/docs/tutorials/kv_layout.rst). [Apache-2.0](/licenses/flashinfer-web-data-d7cc3d22-Apache-2.0.txt).*
<!-- source-caption:end -->

When number of requests is greater than 1, different requests might have different query length and kv length.
To avoid padding, we use a 2D ragged tensor to store attention mask. The input `qo_indptr` and
`kv_indptr` arrays (both with length `num_requests+1`) are used to store the information of
variable sequence lengths of each request,
`qo_indptr[i+1]-qo_indptr[i]` is the query length of request `i` (`qo_len[i]`),
`kv_indptr[i+1]-kv_indptr[i]` is the kv length of request `i` (`kv_len[i]`).

The mask arrays of all requests are flattened (with query as the first dimension, and kv as last dimension)
and concatenated into a single 1D array: `mask_data`. FlashInfer will create a `mask_indptr` array implicitly
to store the start offset of each request's mask in the flattened mask array: `mask_indptr[1:] = cumsum(qo_len * kv_len)`.

`mask_data` has shape `(mask_indptr[-1],)`, we can use `mask_data[mask_indptr[i]:mask_indptr[i+1]]` to slice the flattened
mask of request `i`.

To save memory, we can further pack the flattened boolean mask array into a bit-packed array (1 bit per element, 8 elements
are packed together as a `uint8`) with "little" bit-order (see [numpy.packbits](https://numpy.org/doc/stable/reference/generated/numpy.packbits.html)
for more details). FlashInfer accepts both boolean mask and bit-packed mask. If boolean mask is provided, FlashInfer will pack it into bit-packed
array internally.

The original Mask Layout API subsection uses the old name `begin_forward`
and is omitted here. In the pinned [v0.7.1 prefill implementation](https://github.com/flashinfer-ai/flashinfer/blob/0b46115d238d29ea2e92c6b9b2cb077d40fbc352/flashinfer/prefill.py),
`plan` prepares execution and `run` performs it.

### Page Table Layout

When KV-Cache is dynamic (e.g. in append or decode stage), packing all keys/values is not
efficient because the sequence length per request changes over time. [vLLM](https://arxiv.org/pdf/2309.06180.pdf)
proposes to organize KV-Cache as a Page Table. In FlashInfer, we treat the page table as
a block sparse matrix (each used page can be viewed as an non-zero block in block sparse matrix)
and uses the [CSR format](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html)
to index the pages in KV-Cache.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/serving-originals-2026-10/flashinfer-page-layout.png]]

<!-- source-caption:start -->
*FlashInfer contributors. [Page Table Layout](https://github.com/flashinfer-ai/flashinfer/blob/0b46115d238d29ea2e92c6b9b2cb077d40fbc352/docs/tutorials/kv_layout.rst). [Apache-2.0](/licenses/flashinfer-web-data-d7cc3d22-Apache-2.0.txt).*
<!-- source-caption:end -->

For each request, we keep a record of its `page_indices`, `last_page_len` which
tracks the pages used by this request and the number of entries in the last page. The KV
sequence length of request `i` is `page_size * (len(page_indices[i]) - 1) + last_page_length[i]`.


> The `last_page_len` of each request must be greater than zero, and less than or equal to `page_size`.

The overall `kv_indptr` array (with length `num_requests+1`) can be computed as:
`[0, len(page_indices[0]), len(page_indices[0])+len(page_indices[1]), ...]`.
The overall `kv_page_indices` array (with length `kv_indptr[-1]`) is the concatenation of all requests' `page_indices`.
The overall `kv_last_page_lens` array (with length `num_requests`) is the concatenation of all requests' `last_page_length`.

The `kv_data` tensor could either be a single 5-D tensor or a tuple of 4-D tensors,
when stored in a single tensor, `kv_data` has shape:

```python
kv_cache_nhd = torch.empty(max_num_pages, 2, page_size, num_heads, head_dim, dtype=torch.bfloat16) # NHD layout
kv_cache_hnd = torch.empty(max_num_pages, 2, num_heads, page_size, head_dim, dtype=torch.bfloat16) # HND layout
```

when stored in a tuple of tensors, `kv_data = (k_data, v_data)`, and each one of them has shape:

```python
k_cache_nhd = torch.empty(max_num_pages, page_size, num_heads, head_dim, dtype=torch.bfloat16) # NHD layout
k_cache_hnd = torch.empty(max_num_pages, num_heads, page_size, head_dim, dtype=torch.bfloat16) # HND layout
v_cache_nhd = torch.empty(max_num_pages, page_size, num_heads, head_dim, dtype=torch.bfloat16) # NHD layout
v_cache_hnd = torch.empty(max_num_pages, num_heads, page_size, head_dim, dtype=torch.bfloat16) # HND layout
```


where `max_num_pages` is the maximum number of pages used by all requests, `page_size` is the number of tokens
we fit into each page. `2` in single tensor storage means K/V (first one for keys, the second one for values).


> `indptr` arrays across the flashinfer library should be of type `int32`. Arrays of type `int64` can cause indexing errors. This is also true of the `kv_page_indices` and `kv_last_page_lens` arrays.


### Multi-head Latent Attention Page Layout

Multi-head Latent Attention (MLA) is a new attention mechanism proposed in [DeepSeek v2](https://arxiv.org/abs/2405.04434) and was
used in later DeepSeek models. MLA unifies key cache and value cache into a single tensor, so there is no need to store them separately.
Compared to multi-head attention or grouped query attention, the KV-Cache of MLA do not have the `num_heads` dimension,
so there is no distinction like `NHD` and `HND` layout.

MLA separates RoPE (Rotary Positional Encoding) dimensions and other head dimensions. We use `kpe` (key w/ positional encoding) and `ckv` (compressed key/value)
to name these two components. User can store them in a single Paged KV-Cache:

```python
head_dim_ckv = 512
head_dim_kpe = 64
mla_paged_kv_cache = torch.empty(max_num_pages, page_size, head_dim_ckv + head_dim_kpe, dtype=torch.bfloat16)
ckv = mla_paged_kv_cache[:, :, :head_dim_ckv]  # Slicing here does not copy or move data
kpe = mla_paged_kv_cache[:, :, head_dim_ckv:]  # Slicing here does not copy or move data
```


and `ckv` and `kpe` can then be fed into the MLA attention kernel `flashinfer.mla.BatchMLAPagedAttentionWrapper`.

#### FlashInfer APIs

`flashinfer.page.append_paged_kv_cache` can append a batch of keys/values (stored as ragged tensors) to the paged KV-Cache
(the pages for these appended keys/values must be allocated prior to calling this API).

`flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper` and `flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper` implements the decode attention
and prefill/append attention between queries stored in ragged tensors and keys/values stored in paged KV-Cache.

### FAQ

**How do FlashInfer manages KV-Cache?**

FlashInfer itself is not responsible for managing the page-table (pop and allocate new pages, etc.) and we leave the strategy
  to the user: different serving engines might have different strategies to manage the page-table. FlashInfer is only responsible
  for computing the attention between queries and keys/values stored in KV-Cache.

<a id="примеры-kv-cache-и-границы-запросов"></a>

<a id="что-происходит-с-одним-запросом"></a>

## Worked examples: cache timing and request boundaries

The following examples are Bookvar additions, separate from the reproduced
documentation. They test state transitions and attention semantics on CPU,
not serving throughput.

<a id="когда-появляется-kv-выбранного-токена"></a>

### When a sampled token acquires its KV state

Let the prompt token IDs be [7, 3, 9], and let successive selections be 12,
4, and EOS. Ordinary autoregressive execution without speculation gives:

| Model call | New input tokens and positions | Tokens with computed KV after the call | Selected continuation |
|---|---|---|---|
| Prefill | [7, 3, 9] at [0, 1, 2] | [7, 3, 9] | 12 |
| First decode | [12] at [3] | [7, 3, 9, 12] | 4 |
| Second decode | [4] at [4] | [7, 3, 9, 12, 4] | EOS |

Token 12 can be streamed immediately after prefill even though its own keys
and values have not been computed. They are created when 12 becomes the next
input. EOS requires no further forward pass when it terminates generation.
With chunked prefill, the first answer is sampled only after the required
prompt context has been processed. Received-token count, processed-token count,
and populated-cache length therefore have different boundaries.

<a id="проверка-упакованного-внимания-на-cpu"></a>

### Packed attention must preserve request boundaries

The lengths below follow FlashInfer's ragged-tensor illustration. One head
has width four. The packed result must equal three independent attention calls.

```python
import numpy as np

rng = np.random.default_rng(7)
lengths = np.array([5, 3, 4])
indptr = np.r_[0, lengths.cumsum()]  # [0, 5, 8, 12]
q, k, v = rng.normal(size=(3, 12, 4))

def attention(q, k, v, allowed):
    scores = q @ k.T / np.sqrt(q.shape[-1])
    scores = np.where(allowed, scores, -np.inf)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ v

owner = np.repeat(np.arange(len(lengths)), lengths)
causal = np.tril(np.ones((12, 12), dtype=bool))
same_request = owner[:, None] == owner[None, :]
packed = attention(q, k, v, causal & same_request)

separate = []
for start, end in zip(indptr[:-1], indptr[1:]):
    local_mask = np.tril(np.ones((end - start, end - start), dtype=bool))
    separate.append(attention(q[start:end], k[start:end], v[start:end], local_mask))
np.testing.assert_allclose(packed, np.concatenate(separate), rtol=1e-12, atol=1e-12)

wrong = attention(q, k, v, causal)  # Incorrectly allows reading earlier requests.
assert not np.allclose(packed[5:], wrong[5:])
print("Request boundaries preserved; a global causal mask gives different outputs.")
```

The first request cannot expose the wrong mask because no foreign tokens precede
it. The error begins with request two. An optimized kernel need not materialize
a square mask; offsets and traversal rules can enforce the same boundary.
This NumPy check verifies the mathematical packing rule, not FlashInfer execution.

<!-- serving-node-completion: host-loading start -->
## Host placement, thread pools and loading weights

The process diagrams above include work that does not run on the GPU:
tokenization, request handling, scheduling and checkpoint loading. A GPU
worker waiting for CPU metadata cannot compensate by using a faster
attention kernel. CPU execution also matters when the model itself runs
on a CPU or when some operators fall back to it.

### CPU affinity and memory placement are separate decisions

In a NUMA machine, processor groups have different paths to memory.
Accessing memory attached to another NUMA node can use an interconnect
that local accesses do not need. Binding a worker to a CPU set constrains
where its threads run; a memory policy constrains where new memory is
allocated. Binding the threads alone does not prove that their weights
and inputs are local.

The [PyTorch Performance Tuning Guide](https://github.com/pytorch/tutorials/blob/5672cd98a7671eda4d9122b10f8eec580f74d03e/recipes_source/recipes/tuning_guide.py#L215)
shows the following Linux launch pattern:

```sh
numactl --cpunodebind=N --membind=N python <pytorch_script>
```

`N` is a NUMA node discovered on the target host, not a GPU number.
The command is a template, not a recommendation to bind every deployment
to node 0. Inspect the allowed CPU set, memory capacity and GPU/NIC locality
first. On some machines one socket contains multiple NUMA domains;
container CPU restrictions can further reduce the usable set.

Consider an illustrative host with eight usable physical cores on each of
two NUMA nodes. Two workers can each use eight cores and local weights,
or both can compete on the same eight-core set. Those arrangements have
the same process count but different contention.

| Worker arrangement | What needs checking |
|---|---|
| One worker per node, local allocation | The model and working set fit each node; local bandwidth is sufficient |
| Two workers bound to the same node | Shared cores and memory bandwidth can become the bottleneck |
| Threads on node 0, weights allocated on node 1 | Remote-memory traffic can remain despite CPU pinning |
| One large worker spanning both nodes | Capacity and parallelism may improve, but cross-node traffic must be measured |

There is no fixed cross-node slowdown to apply to all four arrangements.
Measure the actual request mix, thread placement, remote-memory traffic
and end-to-end latency. A larger model may require memory from several
nodes; locality cannot be obtained by assigning capacity that does not
exist. Pinning is also not a hard real-time guarantee: kernel activity,
interrupts, competing work and memory contention still matter.

### Count thread pools, not just server processes

PyTorch distinguishes the number of OpenMP threads from affinity:

<!-- source-excerpt: pytorch-openmp-affinity -->
> OpenMP is utilized to bring better performance for parallel computation tasks.
> `OMP_NUM_THREADS` is the easiest switch that can be used to accelerate computations. It determines number of threads used for OpenMP computations.
> CPU affinity setting controls how workloads are distributed over multiple cores. It affects communication overhead, cache line invalidation overhead, or page thrashing, thus proper setting of CPU affinity brings performance benefits. `GOMP_CPU_AFFINITY` or `KMP_AFFINITY` determines how to bind OpenMP* threads to physical processing units. Detailed information can be found [here](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/performance_tuning/tuning_guide.html).
<!-- /source-excerpt -->

*PyTorch contributors, [Utilize OpenMP](https://github.com/pytorch/tutorials/blob/5672cd98a7671eda4d9122b10f8eec580f74d03e/recipes_source/recipes/tuning_guide.py#L229),
BSD-3-Clause. Complete original explanatory paragraph; comment prefixes
and reStructuredText links converted to Markdown. The licence is retained
with the source snapshot; [upstream licence](https://github.com/pytorch/tutorials/blob/5672cd98a7671eda4d9122b10f8eec580f74d03e/LICENSE).*

Four model processes each starting sixteen compute threads can make
64 threads runnable on a sixteen-core allocation. That does not mean
four times the useful computation: threads wait, are rescheduled and
compete for caches and memory bandwidth. The API server, tokenizer and
runtime can have additional pools. Conversely, a pool with many sleeping
threads is not proof of CPU contention; inspect runnable work and traces.

For four equal, simultaneously active compute processes on sixteen
available cores, four compute threads per process is a starting point
for an experiment, not an optimum. Keep some capacity for request handling
if it shares those cores. Compare several thread counts at the same offered
load, including p95/p99 latency and goodput. Set controls before the relevant
runtime creates its pools, and verify which runtime actually honours them.
`OMP_NUM_THREADS` does not configure every thread in the service.

### A mapped checkpoint is not a ready model

Checkpoint loading crosses several boundaries: file metadata, tensor
storage, module parameters, device memory, and runtime initialization.
Mikayla Gawarecki's PyTorch recipe explains what memory mapping changes:

<!-- source-excerpt: pytorch-loading-mmap -->
> The `mmap` keyword argument to `torch.load` attempts to solve the above two
> problems. As its name implies, the `mmap` keyword argument to `torch.load`
> makes use of an [mmap call](https://man7.org/linux/man-pages/man2/mmap.2.html)
> which maps a file on disk into virtual memory and lets the OS handle loading and
> unloading into physical memory automatically. When this flag is passed, tensor
> storages will be memory-mapped.
<!-- /source-excerpt -->

*Mikayla Gawarecki, [Tips for Loading an nn.Module from a Checkpoint](https://github.com/pytorch/tutorials/blob/5672cd98a7671eda4d9122b10f8eec580f74d03e/recipes_source/recipes/module_load_state_dict_tips.py#L79),
BSD-3-Clause. Original paragraph, markup adapted. Its two preceding problems
are insufficient RAM for eagerly loaded storages and waiting for the whole
checkpoint before starting per-tensor processing.*

Returning from `torch.load(..., mmap=True)` does not mean every weight has
been read from storage. Access can fault pages in later. On a warm page
cache this can be inexpensive; on a cold or remote file system it can move
latency into the first use. GPU execution still needs weights on the device.
Record whether the experiment includes downloading, file reads, transfer,
conversion, compilation and the first representative forward pass.

The same recipe combines three controls:

| Control | Allocation or work it avoids | What remains |
|---|---|---|
| `torch.load(..., mmap=True)` | Eagerly materializing every checkpoint storage | Reading required pages and moving tensors to their execution device |
| Constructing the module on `meta` | Allocating and randomly initializing parameters that will be overwritten | Supplying real storages before execution |
| `load_state_dict(..., assign=True)` | Copying into a second preallocated set of module parameters | Matching names/shapes and preserving the correct parameter references |

This small CPU example adapts the recipe to a complete executable model.
It checks outputs and storage sharing, not loading speed. The temporary
checkpoint is created by the example itself; it is not an untrusted file.

<!-- checked-example: serving-mmap-loading -->
```python
import tempfile
from pathlib import Path
import torch

def make_model():
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(),
                               torch.nn.Linear(8, 2))

torch.manual_seed(7)
reference = make_model().eval()
x = torch.randn(3, 4)
with tempfile.TemporaryDirectory() as directory:
    checkpoint = Path(directory) / 'weights.pt'
    torch.save(reference.state_dict(), checkpoint)
    state = torch.load(checkpoint, mmap=True, weights_only=True,
                       map_location='cpu')
    with torch.device('meta'):
        restored = make_model()
    assert all(parameter.is_meta for parameter in restored.parameters())
    restored.load_state_dict(state, assign=True)
    restored.eval()
    assert not any(parameter.is_meta for parameter in restored.parameters())
    for name, parameter in restored.named_parameters():
        assert parameter.data_ptr() == state[name].data_ptr()
    with torch.inference_mode():
        torch.testing.assert_close(restored(x), reference(x))
```

The example requires PyTorch 2.1 or later. For further training, create an
optimizer after assigning parameters unless the chosen version's documented
parameter-swap behaviour is explicitly enabled: an optimizer can otherwise
retain references to the old parameter objects. A serving engine may also
repack, shard or quantize weights, so its final storage need not match this
small CPU example.

Safetensors addresses a different issue: storing tensor metadata and bytes
without serializing arbitrary Python objects. Its offset-based layout also
permits selective access to tensors or supported slices. It does not remove
disk traffic, device transfer or format conversion. The format's authors
make the limitation explicit:

<!-- source-excerpt: safetensors-zero-copy -->
> Zero-copy: No format is really zero-copy in ML, it needs to go from disk to RAM/GPU RAM (that takes time). On CPU, if the file is already in cache, then it can
> truly be zero-copy, whereas on GPU there is not such disk cache, so a copy is always required
> but you can bypass allocating all the tensors on CPU at any given point.
> SafeTensors is not zero-copy for the header. The choice of JSON is pretty arbitrary, but since deserialization is <<< of the time required to load the actual tensor data and is readable I went that way, (also space is <<< to the tensor data).
<!-- /source-excerpt -->

*Hugging Face Safetensors contributors, [Notes](https://github.com/huggingface/safetensors/blob/e246a2560645b7525f5775669ed816eb57c5bcc8/README.md#notes),
Apache-2.0. One complete original bullet; line wrapping only.
This does not certify model quality, provenance or the security of code
loaded separately from its tensor file.*

These distinctions return to the readiness check: a process has to load
the expected artefact, complete required device preparation and pass its
configured initialization checks before it accepts normal traffic.
A short metadata-loading time is not an end-to-end cold-start result.
<!-- serving-node-completion: host-loading end -->

<a id="где-находится-triton"></a>

## Where Triton fits

Triton sits one layer lower still: it is a language and compiler for GPU kernels expressed through tiles and program instances. It can implement fused softmax, quantization, sampling, or a specialized attention backend. Triton itself does not maintain a request queue, accept HTTP, or decide whose KV cache to evict.

“Moving from vLLM to Triton” therefore conflates levels: a runtime can use kernels written in Triton. NVIDIA Triton Inference Server is a separate product and is not the Triton programming language. The path from memory traffic to tiled matrix multiplication is developed in [[02 Areas/ML & DL/01 Справочник/Inference/Triton и GPU kernels|Triton and GPU kernels]].

<a id="как-выбирать-runtime"></a>

<a id="как-сравнивать-движки"></a>

<a id="что-остаётся-стабильным-а-что-нужно-перепроверять"></a>

## How to choose a runtime

Start from the workload, not the project name.

1. **Model and hardware.** Do the weights fit? Which data types, quantization modes, and attention backends are actually supported on the target GPU?
2. **Request shape.** Are long prompts or long decode phases dominant? Are shared prefixes, branching, multi-turn chat, or many LoRA adapters common?
3. **SLO.** Is TTFT, stable TPOT, maximum throughput, or goodput under tail-latency constraints most important?
4. **Scaling.** Are tensor/expert parallelism, several replicas, disaggregated prefill/decode, or heterogeneous hardware required?
5. **Operational cost.** How important are an OpenAI-compatible API, observability, rolling upgrades, version stability, and ease of debugging?

Then compare candidates with the same model, precision, sampling constraints, and trace of the length distribution. Peak throughput on one synthetic fixed prompt does not establish which runtime will sustain a production SLO.

<a id="связанные-темы"></a>

<a id="практика-и-первоисточники"></a>

<a id="llamacpp-полезный-контраст-cpu-first-runtime"></a>

<a id="llamacpp-другой-набор-условий-развёртывания"></a>

## Sources and further reading

Related mechanisms and implementation walkthroughs:

- [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/6868a52cb337450658006ad2f9281c2eb63f1666/README.md): CPU, GPU and mixed execution.
- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/58b Benchmarking, SLO и эксплуатация inference|Benchmarking and SLOs]]: compare engines on the same models and workloads.
- [[02 Areas/ML & DL/01 Справочник/Inference/vLLM — анатомия inference engine|Inside vLLM by Aleksa Gordić]]: a sequential implementation walkthrough; [original article](https://www.aleksagordic.com/blog/vllm).
- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention|KV-cache and PagedAttention]] and [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/56 FlashAttention|FlashAttention]].

- [vLLM architecture, v0.31.0](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/docs/design/arch_overview.md).
- [SGLang RadixAttention](https://github.com/lm-sys/lm-sys.github.io/blob/88c036dd1a78c8f2525782f98ee4493a63d7a532/blog/2024-01-17-sglang.md) and [overlap scheduling](https://github.com/lm-sys/lm-sys.github.io/blob/88c036dd1a78c8f2525782f98ee4493a63d7a532/blog/2024-12-04-sglang-v0-4.md).
- [TensorRT-LLM overview, v1.2.1](https://github.com/NVIDIA/TensorRT-LLM/blob/376f7e1bd8ed543f75014309e3fd4b237e9b0e73/docs/source/developer-guide/overview.md).
- [FlashInfer KV-cache layout, v0.7.1](https://github.com/flashinfer-ai/flashinfer/blob/0b46115d238d29ea2e92c6b9b2cb077d40fbc352/docs/tutorials/kv_layout.rst).
- [CS336 Lecture 10](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_10.py).
- [[05 Источники/Courses/Harvard ML Systems/vol1/model_serving|Harvard ML Systems: Model Serving]].
- [[05 Источники/Courses/Harvard ML Systems/vol2/inference|Harvard ML Systems: Inference]].
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week08_inference_software/lecture.pdf|Efficient DL Systems: inference software lecture]], [seminar](https://github.com/mryab/efficient-dl-systems/tree/e632aa89ca9e6638d52e1b686095e7442faffbb0/week08_inference_software).
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week08_inference_software/seminar.ipynb|Local seminar notebook]] and [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week08_inference_software/homework/homework_week8.ipynb|inference engine assignment]].
- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180).
- Zheng et al., [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104); pinned v0.5.21 [HTTP server](https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/srt/entrypoints/http_server.py) and [scheduler](https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/srt/managers/scheduler.py).
- NVIDIA, [TensorRT-LLM LLM API, v1.2.1](https://github.com/NVIDIA/TensorRT-LLM/blob/376f7e1bd8ed543f75014309e3fd4b237e9b0e73/tensorrt_llm/llmapi/llm.py).
- FlashInfer, [prefill API implementation, v0.7.1](https://github.com/flashinfer-ai/flashinfer/blob/0b46115d238d29ea2e92c6b9b2cb077d40fbc352/flashinfer/prefill.py).

**Previous:** [[00 Учебник/14 Inference и оптимизация/55b Scheduling — continuous batching, chunked prefill и prefix caching|Scheduling]].
**Next:** [[00 Учебник/14 Inference и оптимизация/56 FlashAttention|FlashAttention]].
