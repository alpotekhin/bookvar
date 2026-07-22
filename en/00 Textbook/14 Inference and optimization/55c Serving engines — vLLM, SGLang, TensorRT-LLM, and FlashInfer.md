---
title: "Serving engines — vLLM, SGLang, TensorRT-LLM, and FlashInfer"
type: textbook-chapter
status: active
locale: en
translation_of: "00 Учебник/14 Inference и оптимизация/55c Serving engines — vLLM, SGLang, TensorRT-LLM и FlashInfer.md"
last_updated: 2026-07-22
last_verified: 2026-07-22
---

# Serving engines — vLLM, SGLang, TensorRT-LLM, and FlashInfer

Loading weights and a tokenizer does not yet make a model into a service. A single `generate` call can process a prompt and emit an answer sequentially, but a production system must concurrently accept requests of different lengths, change batch membership after every step, allocate memory for growing KV caches, stream tokens to clients, and release resources when sequences finish. A serving engine is the program that coordinates all these operations with GPU execution.

vLLM, SGLang, TensorRT-LLM, and FlashInfer often appear in the same comparison table even though they occupy different levels of the stack. vLLM and SGLang provide complete runtimes and servers. TensorRT-LLM combines compilation with a highly optimized runtime closely coupled to NVIDIA's stack. FlashInfer is primarily a library of kernels for attention, sampling, and MoE that another runtime may employ. A meaningful comparison begins by separating the layers of an inference system.

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

## What happens to one request?

Consider a request that has passed HTTP validation. A processor applies the chat template, tokenizes the text, and constructs an internal sampling-parameter description. The scheduler queues the request and allocates KV blocks on a later step. The model executor performs prefill; the sampler selects the first token; and the request returns to the scheduler as decode work. The loop continues until EOS, a stop condition, or the length limit.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/vllm-engine_loop.png]]

*The vLLM V1 engine loop: the scheduler constructs work, the model executor performs a step, and the output processor updates request state. Source: Aleksa Gordić, [Inside vLLM](https://www.aleksagordic.com/blog/vllm); [original image](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2025-vllm-anatomy/engine_loop.png).*

Complexity arises because different requests occupy the loop at once. Some are still consuming long prompts, some emit their next token, others have finished, and still others await free KV blocks. The runtime's central object is therefore a changing set of requests and states, not an immutable batch.

## vLLM: scheduling and block-based memory at the center

Historically, vLLM grew out of the PagedAttention work. Its original problem was systemic: contiguous KV-cache allocation caused internal and external fragmentation, reducing feasible batch size and therefore throughput. A block table separated a token sequence's logical address space from its cache's physical placement. The scheduler could add and remove requests without relocating large contiguous buffers.

Modern vLLM contains more components, but this relationship remains important. The engine core contains the scheduler and cache manager; the model executor controls one or more workers; and the model runner prepares input tensors and metadata for the attention backend. The server frontend is kept outside the hot GPU loop.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/vllm-engine_constructor.png]]

*Components constructed around `LLM`: configuration, processor, engine core, model executor, scheduler, and KV-cache manager. Source: Aleksa Gordić, [Inside vLLM](https://www.aleksagordic.com/blog/vllm); [original image](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2025-vllm-anatomy/engine_constructor.png).*

Before the first request, a worker loads weights, selects an attention backend, measures occupied memory, and assigns the remaining budget to the KV cache. The engine may then capture CUDA Graphs for common execution sizes. Replay reduces CPU-to-GPU launch overhead but requires prepared buffers and an allowed set of shapes. Eager execution and graph replay are thus two execution modes of the same model runner, not different models.

At every step, the scheduler sets a token budget. Decode typically needs one new token position per request, while long prefills can be chunked. The cache manager verifies block capacity and creates a slot mapping from logical positions to physical KV-cache locations. The model runner packs tokens from different requests into one input, while attention metadata retains sequence boundaries.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/vllm-fwd_pass.png]]

*One forward pass for requests with different histories. Sequences are packed together, and metadata maps positions to KV blocks. Source: Aleksa Gordić, [Inside vLLM](https://www.aleksagordic.com/blog/vllm); [original image](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2025-vllm-anatomy/fwd_pass.png).*

This design suits ordinary completion and chat workloads: many independent requests, a changing batch, a long decode phase, and a need for predictable memory management. A class-by-class reading belongs in [[02 Areas/ML & DL/01 Справочник/Inference/vLLM — анатомия inference engine|the anatomy of vLLM]], because module names and APIs change faster than scheduling and cache principles.

## SGLang: program structure becomes part of scheduling

Not every LLM workload is a set of independent prompts. Multi-turn chat repeats a system prompt and history; few-shot evaluation reuses the same demonstrations; tree search creates several continuations of a common prefix; and an agent may return to an earlier state after a tool call. A runtime that sees only a flat request queue discovers these relationships too late.

The original SGLang work proposed both a language for describing language-model programs and a runtime able to exploit their structure. Its central reuse mechanism is RadixAttention: the KV cache is stored as a radix tree over token prefixes. Tree edges correspond to token sequences, nodes reference cache entries, and longest-prefix matching finds the already computed portion of a new request.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/sglang-program-figure2.png]]

*A branching language-model program and runtime opportunities, as shown in Figure 2 of Lianmin Zheng et al., [SGLang: Efficient Execution of Structured Language Model Programs](https://papers.nips.cc/paper_files/paper/2024/file/724be4472168f31ba1c9ac630f15dec8-Paper-Conference.pdf).*

When a request is inserted, the tree may split an existing edge where prefixes diverge. Finishing a request need not immediately remove its cache; nodes remain candidates for reuse. Under memory pressure, the runtime evicts leaves with an LRU policy. A cache-aware scheduler prefers requests with longer matches while still accounting for starvation and queue fairness.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/sglang-radixattention-figure3.png]]

*Nine radix-tree states during successive requests, insertions, and evictions. Source: Zheng et al., SGLang paper, Figure 3.*

RadixAttention and PagedAttention address different problems. Paging controls physical block placement and fragmentation; the radix tree indexes prefix content and discovers shared spans. A runtime may use both a block allocator for memory and a radix tree for prefix reuse. See [[02 Areas/ML & DL/01 Справочник/Inference/SGLang и RadixAttention|SGLang and RadixAttention]] for a step-by-step account.

## TensorRT-LLM: graph optimization and runtime as one stack

TensorRT-LLM serves the same ultimate purpose but emphasizes compilation and specialized NVIDIA GPU implementations. Model preparation or engine building selects precision, quantization, parallelism, and optimized plugins. The runtime executes that engine, manages in-flight batching and a paged KV cache, while the executor API organizes requests and distributed execution.

Consequently, “vLLM versus TensorRT-LLM” cannot be reduced to one tokens/s figure. The outcome depends on optimized-plugin support for the model, acceptable build time, the desired dynamism of the PyTorch ecosystem, available precision and parallelism modes, and the real mix of prompt and output lengths. TensorRT-LLM can exploit a known NVIDIA configuration particularly well; vLLM is often convenient as a rapidly evolving open runtime with a broad model interface. These are engineering profiles, not a universal ranking.

TensorRT-LLM documentation changes frequently, so supported plugins, build flags, and server options are not frozen here. Check the current [Architecture Overview](https://nvidia.github.io/TensorRT-LLM/architecture/overview.html) and [Executor](https://nvidia.github.io/TensorRT-LLM/advanced/executor.html) documentation for the deployed version.

## FlashInfer: a kernel library, not another scheduler

FlashInfer provides optimized kernels and APIs for prefill, decode, append, sampling, cascade attention, sparse and paged KV caches, and MoE operations. It need not own an HTTP server or a global request queue. A serving runtime can select FlashInfer as its attention backend while retaining responsibility for the next batch.

This division clarifies the boundary of responsibility. The scheduler constructs a ragged or paged representation of active sequences; the kernel library receives data pointers, page tables, and lengths, then executes attention. FlashInfer supports NHD and HND layouts: axis order for page, token/head, and head dimension affects addressing and kernel suitability. Attention's mathematical definition is unchanged, but layout determines which memory regions are read together.

FlashInfer is most intelligible after FlashAttention and PagedAttention. FlashAttention explains an IO-aware algorithm, PagedAttention explains dynamic request-memory management, and FlashInfer packages specialized implementations for serving workloads. Current layouts and signatures are documented in the official [KV-cache layout tutorial](https://docs.flashinfer.ai/tutorials/kv_layout.html).

## Where Triton fits

Triton sits one layer lower still: it is a language and compiler for GPU kernels expressed through tiles and program instances. It can implement fused softmax, quantization, sampling, or a specialized attention backend. Triton itself does not maintain a request queue, accept HTTP, or decide whose KV cache to evict.

“Moving from vLLM to Triton” therefore conflates levels: a runtime can use kernels written in Triton. NVIDIA Triton Inference Server is a separate product and is not the Triton programming language. The path from memory traffic to tiled matrix multiplication is developed in [[02 Areas/ML & DL/01 Справочник/Inference/Triton и GPU kernels|Triton and GPU kernels]].

## How to choose a runtime

Start from the workload, not the project name.

1. **Model and hardware.** Do the weights fit? Which data types, quantization modes, and attention backends are actually supported on the target GPU?
2. **Request shape.** Are long prompts or long decode phases dominant? Are shared prefixes, branching, multi-turn chat, or many LoRA adapters common?
3. **SLO.** Is TTFT, stable TPOT, maximum throughput, or goodput under tail-latency constraints most important?
4. **Scaling.** Are tensor/expert parallelism, several replicas, disaggregated prefill/decode, or heterogeneous hardware required?
5. **Operational cost.** How important are an OpenAI-compatible API, observability, rolling upgrades, version stability, and ease of debugging?

Then compare candidates with the same model, precision, sampling constraints, and trace of the length distribution. Peak throughput on one synthetic fixed prompt does not establish which runtime will sustain a production SLO.

## What remains stable, and what must be rechecked

The durable content is the layer decomposition, engine loop, relationship between scheduling and the KV cache, distinction between paging and prefix indexing, and boundary between a runtime and a kernel library. These survive class renaming.

The following change quickly:

- vLLM and SGLang internal classes;
- server arguments and CLIs;
- available attention backends;
- supported quantization formats;
- compatibility among CUDA Graphs, compilation, and execution modes;
- current benchmark leaders.

Such details belong in dated reference cards and labs. The textbook should teach which questions to ask of a new runtime release and how to interpret its architecture.

## Sources and further reading

- Stanford CS336, [Lecture 10: Inference](https://cs336.stanford.edu/).
- Aleksa Gordić, [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://www.aleksagordic.com/blog/vllm).
- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180).
- Zheng et al., [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104).
- [TensorRT-LLM documentation](https://nvidia.github.io/TensorRT-LLM/).
- [FlashInfer documentation](https://docs.flashinfer.ai/).
- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention|KV cache, batching, and PagedAttention]]
- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/56 FlashAttention|FlashAttention]]
