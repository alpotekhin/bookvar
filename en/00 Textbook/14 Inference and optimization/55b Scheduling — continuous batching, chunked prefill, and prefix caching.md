---
title: "Scheduling: continuous batching, chunked prefill, and prefix caching"
type: textbook-chapter
status: canonical
locale: en
translation_of: "00 Учебник/14 Inference и оптимизация/55b Scheduling — continuous batching, chunked prefill и prefix caching.md"
last_updated: 2026-07-22
last_verified: 2026-07-22
primary_sources:
  - https://www.usenix.org/system/files/osdi22-yu.pdf
  - https://arxiv.org/abs/2308.16369
  - https://arxiv.org/abs/2403.02310
  - https://arxiv.org/abs/2309.06180
  - https://www.aleksagordic.com/blog/vllm
---

# Scheduling: continuous batching, chunked prefill, and prefix caching

Batching in an ordinary neural network assumes that examples begin and finish together. This assumption fails for autoregressive serving: requests arrive at different times, inputs have different lengths, and output lengths are not known in advance. If batch membership is fixed until generation ends, short requests release compute lanes that cannot be reused until the longest request finishes. If long inputs are admitted into ongoing generation without limits, a single prefill can interrupt token delivery to every active client.

An LLM scheduler therefore performs several tasks at once: selecting requests for the next pass, allocating their KV-cache memory, preventing starvation, and negotiating the throughput–latency trade-off. [Orca](https://www.usenix.org/system/files/osdi22-yu.pdf) established the modern basis for continuous batching; [SARATHI](https://arxiv.org/abs/2308.16369) and [Sarathi-Serve](https://arxiv.org/abs/2403.02310) studied chunked prefill systematically; and [PagedAttention](https://arxiv.org/abs/2309.06180) addresses block-based memory and preemption. [Aleksa Gordić's account](https://www.aleksagordic.com/blog/vllm) is useful for following a concrete vLLM loop, but its class names and engine parameters are not the algorithm itself.

## Why static batching wastes capacity

Imagine a server that groups four requests and starts generation. The first ends after 20 tokens, the second after 40, and the last after 200. Under static batching, released rows are either padded or perform no useful work, while new requests wait for the entire group. A larger batch improves initial GPU utilization but magnifies losses from length variation and queueing.

Three intervals can be separated: waiting for a batch to form, useful execution, and waiting for the longest peers. The first and third may be acceptable for offline throughput; in an interactive service they directly increase the next request's TTFT. The batch boundary must therefore be drawn around the smallest unit of autoregressive work, not around a complete request.

## Orca: iteration-level scheduling

Orca introduced **iteration-level scheduling**. Rather than running a fixed group to completion, the scheduler invokes the model for one iteration. It then removes finished sequences, admits new ones, and forms the next batch. This regime is now generally called **continuous batching** or **in-flight batching**.

Request state persists between iterations: current length, KV cache, sampling parameters, and completion status. A batch becomes a transient view of the active set rather than a lifelong cohort. Different sequences may occupy different positions at each step, so the attention kernel must know their true lengths and cache addresses.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/vllm-fwd_pass.png]]

*One vLLM iteration: requests of different lengths are packed without right padding, and positions map to slots in the paged KV cache. Original diagram by Aleksa Gordić, from “Run forward pass” in [Inside vLLM](https://www.aleksagordic.com/blog/vllm). No explicit license is stated on the page.*

Orca combined this idea with **selective batching**: operations compatible with heterogeneous requests ran as a batch, whereas incompatible operations could execute separately. Modern packed-attention kernels and block tables have enlarged the compatible set, but the principle remains: operation semantics determine batching, not a desire to force every workload into a rectangular tensor.

Continuous batching raises throughput because a newly freed slot can promptly accept another request and model-weight reads are amortized across active sequences. Concurrency cannot grow indefinitely, however. KV-cache occupancy is proportional to the sum of active lengths, and a large batch lengthens each iteration and increases TPOT. Schedulers consequently operate under token and block budgets, not merely a maximum request count.

## What the scheduler selects at each step

A stable abstract scheduling loop has five stages.

1. Update state: retire finished requests and account for released blocks.
2. Choose tokens from the waiting queue and active set within compute and memory budgets.
3. Let the KV-cache manager locate reusable blocks and allocate new slots.
4. Assemble packed inputs in the model runner and execute one forward pass.
5. After sampling, stream new tokens to clients and advance request states.

The selection policy determines observable service quality. Strict FCFS protects old requests from starvation but lets a long prefill block short work. Always prioritizing decode keeps active responses smooth but can give newcomers excessive TTFT. Shortest-job-first policies must predict unknown output lengths and may indefinitely postpone large jobs. Practical systems use request age, waiting-time limits, client priority, and separate prefill/decode budgets.

## Fairness, preemption, and overload

A system is overloaded when requests arrive faster than it can complete them within the SLO. Its queue then grows without bound; local reordering cannot create compute capacity. Scheduling must be paired with admission control and backpressure. Rejecting or deferring some work is fairer than accepting everything and violating latency for every client.

Even at a sustainable average rate, several long sequences may exhaust memory. **Preemption** then suspends an active request. PagedAttention describes two recovery strategies. With swapping, KV blocks move to CPU memory and later return; with recomputation, the cache is freed and the prefix is recomputed when work resumes. Swapping consumes PCIe/NVLink bandwidth and CPU memory, whereas recomputation consumes GPU time. Prefix length, link speed, and the current bottleneck determine the better option.

Preemption alone does not ensure fairness. If newer short jobs repeatedly displace one request, its tail latency becomes unbounded. Aging, request-class quotas, and limits on repeated preemptions are useful safeguards. Evaluation must include p95/p99 TTFT and TPOT, not only means.

## Why a long prefill disrupts decode

Continuous batching can admit a new request between iterations, but its prefill may contain thousands of tokens and take much longer than an ordinary decode step. While the GPU processes that prompt, active clients receive no new tokens and their ITL spikes.

Deferring all prefills until decode work is absent starves new requests. Mixing a complete prefill with decode still lets the long prompt determine iteration time. SARATHI and Sarathi-Serve instead divide long prefills into bounded pieces: **chunked prefill**.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-advanced/physics-scheduling/gordic-chunked-prefill.png]]

*A long input is divided into prefill chunks, with active requests continuing to decode between them. Original diagram by Aleksa Gordić, from “Chunked prefill” in [Inside vLLM](https://www.aleksagordic.com/blog/vllm); the mechanism is studied in [SARATHI](https://arxiv.org/abs/2308.16369) and [Sarathi-Serve](https://arxiv.org/abs/2403.02310).*

For a prompt of $S$ tokens and chunk size $C$, the scheduler performs approximately $\lceil S/C\rceil$ passes instead of one prefill. Intermediate keys and values remain cached after each chunk, and the next chunk attends to the processed left context. The mathematical result is unchanged provided the attention kernel handles positions and causal masking correctly.

SARATHI calls an effective composition a **decode-maximal batch**: one prefill chunk is included and the remaining token budget is filled with decode requests. The compute-intensive chunk improves GPU utilization, while decode “rides along” at low incremental cost. Sarathi-Serve develops this into stall-free scheduling so that new prefills do not create long pauses in ongoing generation.

The choice of $C$ is a trade-off. A large chunk uses GEMM more efficiently and reduces launches but worsens the maximum ITL. A small chunk improves responsiveness and fairness but adds scheduling overhead, repeated reads, and less favorable matrix shapes. It must be tuned for the hardware, model, prompt-length distribution, and TTFT/TPOT targets rather than treated as a universal constant.

## Prefix caching: avoid recomputing shared input

Many requests share a beginning: a system instruction, agent template, few-shot examples, a long document, or conversation history. An ordinary KV cache eliminates repetition within one request but may be released when that request completes. **Prefix caching** associates tokens in each full block with their KV data so that a later request with the same beginning can skip that part of prefill.

In a block implementation, token sequences are divided into complete chunks. Each block receives a key derived from the previous block's key, its token IDs, and any additional parameters that influence computation. Chaining is necessary because an identical local fragment following a different prefix has different positions and attention history. An incomplete final block generally cannot be reused as a normal cache unit because its contents may still be extended.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/vllm-prefix_pt1.png]]

*The first request with a long shared prefix: complete blocks receive chained hash keys and KV data. Original diagram sequence by Aleksa Gordić, from “Prefix Caching” in [Inside vLLM](https://www.aleksagordic.com/blog/vllm).*

The first request has no matches. The scheduler allocates physical blocks, executes prefill, and associates computed blocks with keys. Completing the request need not destroy the data immediately: a block may enter the free queue while remaining eligible for reuse until actually reassigned.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/vllm-prefix_pt2.png]]

*Construction of the key chain and registration of computed blocks after the first prefill. Diagram by Aleksa Gordić, [Inside vLLM](https://www.aleksagordic.com/blog/vllm). No explicit license is stated.*

When a second request arrives, the engine constructs the same chain and locates the longest matching prefix. Its physical blocks become occupied again, their reference counts increase, and the forward pass receives only uncomputed tokens. This improves TTFT and reduces prefill load; decode after the common prefix is no cheaper because the requests' new tokens diverge.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/vllm-prefix_pt3.png]]

*A subsequent request finds the longest cache hit and reuses physical KV blocks. Diagram by Aleksa Gordić, from “Prefix Caching” in [Inside vLLM](https://www.aleksagordic.com/blog/vllm).*

Prefix-cache capacity is finite, so blocks are evicted when memory is scarce. Policy value depends on repetition frequency, shared-prefix size, and recency. SGLang generalizes flat lookup with RadixAttention, organizing prefixes from branching requests in a radix tree and incorporating match length into scheduling. The core correctness condition remains: reuse is valid only when every input affecting $K$ and $V$ matches, including model, adapter, positions, and prompt-processing configuration.

## How the mechanisms interact

Continuous batching, PagedAttention, chunked prefill, and prefix caching solve different parts of one problem. Continuous batching changes the compute batch dynamically. PagedAttention places growing histories and enables block sharing. Chunked prefill bounds the duration of a heavy iteration. Prefix caching removes repeated computation of a common beginning.

They compete for the same budgets. Retaining old prefix blocks may raise future hit rate but leaves less memory for active requests. A large batch improves throughput but lengthens an iteration. Decode priority stabilizes TPOT while raising TTFT. Frequent preemption frees memory but discards expensive prefill work. The scheduler must therefore be evaluated on a realistic request distribution, jointly measuring throughput, TTFT, TPOT, and tail percentiles.

Three workloads make a useful policy test. Short-prompt, long-answer chat is decode-dominated, favoring a large stable batch and low TPOT. Long-document summarization is prefill-dominated; chunking protects ongoing generations but may increase a new document's TTFT. Agent systems with shared instructions and repeated context benefit strongly from prefix caching and cache-aware scheduling. One configuration cannot optimize all three.

## Stable mechanism, changing API

Scheduler class names, block-table representations, CLI flags, and vLLM's internal call order change across releases. The durable model needs only a waiting queue, a running set, a token budget, physical KV blocks, prefill chunks, decode tokens, a priority policy, and preemption conditions. Concrete configuration must be checked against the installed engine version.

This separation keeps the explanation from aging with the API. Orca remains the source of iteration-level scheduling even when a server calls it continuous batching. SARATHI remains the source of chunked-prefill analysis even when the feature is enabled by default. PagedAttention explains block-based memory even if a particular kernel changes. The history matters because it reveals which constraint each mechanism removes and what cost it introduces.

## Sources

- Yu et al., [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/system/files/osdi22-yu.pdf) — iteration-level scheduling and selective batching.
- Agrawal et al., [SARATHI](https://arxiv.org/abs/2308.16369) — chunked prefill and decode-maximal batching.
- Agrawal et al., [Sarathi-Serve](https://arxiv.org/abs/2403.02310) — stall-free scheduling and the throughput–latency trade-off.
- Kwon et al., [PagedAttention](https://arxiv.org/abs/2309.06180) — block-based KV-cache management, scheduling, and preemption.
- Aleksa Gordić, [Inside vLLM](https://www.aleksagordic.com/blog/vllm) — a detailed walkthrough of continuous batching, chunked prefill, and prefix caching.
