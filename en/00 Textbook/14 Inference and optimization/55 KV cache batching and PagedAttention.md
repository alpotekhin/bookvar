---
title: "KV cache, batching, and PagedAttention"
type: textbook-chapter
status: canonical
locale: en
translation_of: "00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention.md"
last_updated: 2026-07-20
last_verified: 2026-07-22
primary_sources:
  - https://cs336.stanford.edu/
  - https://arxiv.org/abs/2309.06180
  - https://docs.vllm.ai/en/latest/design/paged_attention/
  - https://www.anyscale.com/blog/continuous-batching-llm-inference
---

# KV cache, batching, and PagedAttention

During training, a Transformer receives rectangular token batches and processes
all positions in parallel. A serving system faces a different workload:
requests arrive at different times, have different prompt lengths, and finish
after different numbers of generation steps. Model parameters are shared, while
history memory belongs to each request. The KV cache, continuous batching, and
PagedAttention address three successive problems: avoid recomputing old
projections, avoid waiting for the longest response, and avoid reserving a
maximum-size contiguous memory region for every request.

## Prefill and decode use the device differently

The server first processes the complete input prompt in **prefill**, then adds
the response one token at a time in **decode**.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/prefill-decode.png]]

*The two phases of autoregressive inference in Stanford CS336,
[Lecture 10: Inference](https://cs336.stanford.edu/). Prefill constructs the
prompt's KV cache in parallel; every decode step reads the accumulated cache and
appends one key–value pair.*

For a prefill of length $S$, activation matrices have many rows. Large matrix
multiplications generally utilize tensor cores well, so this phase is often
compute-bound. At a decode step, the activation matrix has only one new row per
request. The device rereads layer parameters and attention history while doing
relatively little arithmetic. With a small batch, decode is more often limited
by memory bandwidth.

This distinction produces two user-facing metrics:

- **TTFT** (*time to first token*) primarily includes queueing and prefill;
- **TPOT/ITL** (*time per output token / inter-token latency*) describes subsequent decode steps.

One average “request latency” hides this difference and offers no guidance about
which mechanism to optimize.

## What the KV cache stores

For one causal-attention head at position $t$,

$$
q_t=x_tW_Q,\qquad k_t=x_tW_K,\qquad v_t=x_tW_V,
$$

$$
o_t=\operatorname{softmax}\left(
\frac{q_tK_{1:t}^{\top}}{\sqrt{d_h}}+M
\right)V_{1:t}.
$$

Once a new token appears, the keys and values at earlier positions do not
change. Without caching, the server would recompute
$k_1,v_1,\ldots,k_{t-1},v_{t-1}$ at every step. The KV cache retains them after
their first computation. Token $t$ requires only the new pair $(k_t,v_t)$,
which is appended to the history before attention is evaluated over all stored
positions.

Caching removes repeated linear projections, but it does not make attention
constant-time: the new query $q_t$ still compares against every key in
$K_{1:t}$. The cache also grows linearly with sequence length.

## How much memory history consumes

Let $L$ be the layer count, $S$ the current sequence length, $H_{kv}$ the number
of key/value heads, $d_h$ the head dimension, and $b$ the bytes per number. One
request then needs

$$
M_{KV}=2LSH_{kv}d_hb.
$$

The factor of two accounts for $K$ and $V$. For 32 layers, eight KV heads,
$d_h=128$, BF16, and an 8,192-token context:

$$
M_{KV}=2\cdot32\cdot8192\cdot8\cdot128\cdot2
\approx1\ \text{GiB}.
$$

That is the history memory for one request, excluding model parameters and
workspace buffers. GQA lowers $H_{kv}$, MLA compresses the history
representation, and KV-cache quantization lowers $b$. All three increase the
number of concurrent sequences, but they alter architectural or numerical
properties and should not be conflated with memory-placement algorithms.

## Why static batches leave the device idle

With static batching, requests begin together and the next batch waits for the
longest response to finish. In the diagram, yellow cells are prompt processing,
blue cells generation, red cells EOS, and the blank region after EOS contains no
useful work.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/anyscale-static-batching.png]]

*Requests of different lengths within a fixed batch. Anyscale diagram from
[Continuous Batching for LLM Inference](https://www.anyscale.com/blog/continuous-batching-llm-inference),
also used in Stanford CS336.*

**Continuous batching** changes batch membership between decode iterations. A
completed sequence immediately releases its slot for a new request. Members are
at different sequence positions, so kernels receive lists of actual lengths and
cache addresses rather than one dense rectangle.

A larger batch amortizes model-weight reads and raises throughput. It can also
make a request wait longer for the scheduler and increase its TPOT because of
neighboring work. A serving system should therefore optimize **goodput**—the
number of requests that meet both TTFT and TPOT constraints—not abstract
tokens/s.

## Why continuous batching is not enough

Suppose each request receives a contiguous buffer sized for its maximum length.
The actual answer is usually shorter, but no other request can use the reserved
tail. If buffers instead match the current length and are repeatedly extended,
adjacent free space may be unavailable. Both policies waste memory.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/paged-attention-fragmentation.png]]

*Internal fragmentation consists of reserved but unused slots inside a buffer;
external fragmentation consists of free gaps between buffers. Figure by Kwon et
al., [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180),
as shown in CS336.*

The vLLM authors emphasize that KV-cache memory is the dynamic part of server
capacity. When fragmented, it can reject a new request even though the total
number of free bytes would otherwise suffice.

## PagedAttention: logical sequences, physical blocks

PagedAttention imports the idea of paged virtual memory. A sequence's cache is
divided into **logical blocks** containing a fixed number of tokens. A block
table maps them to any available **physical blocks** in a shared pool. The next
logical block need not be physically adjacent to the previous one.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/paged-attention-blocks.png]]

*An attention request reads three physically discontiguous blocks while
preserving logical token order. Figure from the
[PagedAttention paper](https://arxiv.org/abs/2309.06180), reused in
[Stanford CS336](https://cs336.stanford.edu/).*

As a response grows, the manager allocates another free block. Internal waste is
limited to the sequence's final block, and external fragmentation between long
contiguous buffers disappears. The cost is a block table and an attention kernel
that gathers $K$ and $V$ from blocks without first copying them into a contiguous
array.

The next diagram maps one sequence's logical blocks to physical locations. The
table is part of request state; releasing the sequence returns its physical
blocks to the pool.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/paged-attention-logical.png]]

*Logical and physical KV-cache blocks. Source: Kwon et al.,
[PagedAttention](https://arxiv.org/abs/2309.06180); image reproduced in Stanford
CS336.*

## Sharing a prefix

Paged placement allows several continuations to reference the same physical
blocks for a shared prefix. This is useful for parallel samples, beam search,
and recurring system prompts. Read-only blocks need no copies. After branches
diverge, their new tokens go into separate blocks; a partially shared final
block uses copy-on-write.

Prefix caching and PagedAttention are related but distinct. Prefix caching asks
whether already computed history may be reused for a matching prefix;
PagedAttention determines how that history is placed and addressed. Paging makes
sharing cheaper, but does not discover semantically equivalent prompts or remove
the need for exact token and model-parameter matches.

## The scheduler couples computation and memory

At every iteration, the server selects work for the next batch:

- prefill for new requests;
- decode for active sequences;
- allocation of blocks for the next token;
- eviction, suspension, or recomputation under memory pressure;
- cache reuse for matching prefixes.

A long prefill utilizes compute well but can postpone the next token of an
active dialogue. **Chunked prefill** divides the input and interleaves chunks
with decode work. Chunk size controls the trade-off: larger chunks make matrix
multiplication more efficient; smaller chunks better protect inter-token
latency.

Scheduling must anticipate memory as well. The server may admit many long
prompts quickly, but their KV caches continue growing at each decode step.
Admission control should therefore consider not only current occupancy but also
an allowable continuation budget.

## What to measure

A minimum server profile includes:

| metric | mechanism it exposes |
|---|---|
| p50/p95 TTFT | queueing, prefill batch size, chunked prefill |
| p50/p95 TPOT or ITL | decode batch and scheduler pauses |
| input/output tokens/s | phase-specific throughput |
| useful/reserved KV bytes | actual memory use and fragmentation |
| block utilization | waste in final blocks |
| preemption/recompute rate | overly aggressive admission control |
| prefix-cache hit rate | effective reuse of shared prefixes |

Measurements should use length and arrival-rate distributions representative of
the product. One batch of identical requests measures kernel speed, but does not
exercise continuous batching, fragmentation, or scheduling.

## Sources and further reading

- Stanford CS336, [Lecture 10: Inference](https://cs336.stanford.edu/) — a unified account of prefill/decode, arithmetic intensity, batching, and paging.
- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — the original algorithm and fragmentation analysis.
- vLLM, [PagedAttention design](https://docs.vllm.ai/en/latest/design/paged_attention/) — the mapping from blocks to implementation data structures.
- Anyscale, [Continuous Batching for LLM Inference](https://www.anyscale.com/blog/continuous-batching-llm-inference) — a visual comparison of static and continuous batching.
