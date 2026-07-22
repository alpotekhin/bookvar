---
title: "Distributed inference and disaggregated prefill/decode serving"
type: textbook-chapter
status: canonical
locale: en
translation_of: "00 Учебник/14 Inference и оптимизация/58a Распределённый inference и disaggregated serving.md"
last_updated: 2026-07-22
last_verified: 2026-07-22
primary_sources:
  - https://cs336.stanford.edu/spring2025/
  - https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin
  - https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html
---

# Distributed inference and disaggregated prefill/decode serving

A single accelerator is a natural boundary for a small model, but not for one
whose weights and KV cache exceed device memory, nor for a service handling
thousands of independent requests. Distributing inference across GPUs solves two
different problems. One is to **place and execute a single model instance** by
partitioning its computation across devices. The other is to **increase service
capacity** by running multiple instances and distributing requests among them.
These goals must not be conflated: a scheme that reduces the latency of one
forward pass may increase per-request cost through communication, while
replication can scale request throughput without helping a model that does not
fit on one GPU.

[Stanford CS336 Lecture 10](https://cs336.stanford.edu/spring2025/) begins with
the distinction between the two phases of generation. During **prefill**, all
prompt tokens are processed in parallel, and the large matrix multiplications
usually make good use of GPU compute units. During **decode**, each request
produces only one new token per step: the model parameters must be read again,
while arithmetic intensity is much lower. The phases therefore differ not only
in duration but also in how resources should be allocated.

## Four principal forms of parallelism

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/vllm-multiprocexecutor.png]]

*The transition from a local worker to a multiprocess executor: each process
owns a device and participates in the model executor's collective operations.
Source: Aleksa Gordić, [Inside vLLM](https://www.aleksagordic.com/blog/vllm),
[original image](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2025-vllm-anatomy/multiprocexecutor.png).*

### Model replication: data parallelism

With **data parallelism (DP)**, every GPU group holds a complete model instance,
and a router assigns requests to replicas. Unlike training, inference requires
no gradient exchange: replicas can serve requests independently. DP can
therefore increase the maximum request rate almost linearly until an external
component—tokenization, networking, weight storage, or load balancing—becomes
the next bottleneck.

The cost is a full copy of the weights and runtime memory at every replica. DP
does not reduce the memory required by one instance and does little for the
latency of one request. On the other hand, one replica can usually fail without
stopping the others, and each can build continuous batches independently. For a
model that fits on one GPU, this is the simplest way to raise throughput.

### Partitioning operations within a layer: tensor parallelism

**Tensor parallelism (TP)** partitions the large matrices of a layer across
GPUs. Different devices may, for example, store different columns of a
projection; each computes its part before a collective operation combines the
partial results. DistServe calls this intra-operator parallelism.

TP reduces per-GPU weight memory and can shorten a large matrix multiplication,
but almost every Transformer block invokes a collective such as `all-reduce`,
`reduce-scatter`, or `all-gather`. It is consequently sensitive to interconnect
bandwidth and latency. TP is often effective within an NVLink/NVSwitch node,
whereas its benefit may disappear across a slower network. A wider TP degree
also leaves less work on each device, eventually producing kernels too small to
use the GPU efficiently. Maximum TP is thus not an objective: the degree is a
constraint chosen from memory, latency, and topology.

### Partitioning the model by depth: pipeline parallelism

**Pipeline parallelism (PP)** places consecutive groups of layers on different
devices. Stages exchange activations rather than partial results from every
matrix operation; DistServe calls this inter-operator parallelism. PP can place
a very deep model with fewer frequent collectives than TP, but an individual
request still traverses the stages sequentially.

Throughput emerges when stages process different microbatches concurrently. If
stage times differ, faster stages wait for the slowest, creating a **pipeline
bubble**. Prefill microbatches differ in input-token count, while the effective
decode batch changes as requests finish, so equal layer counts do not guarantee
balanced load. PP generally adds activation-transfer latency to one pass, but
can raise rate capacity once the pipeline is full.

### Partitioning experts: expert parallelism

In a mixture-of-experts model, **expert parallelism (EP)** assigns different
experts to different GPUs. The router selects experts per token, after which an
all-to-all sends tokens to their owners and returns the results. Unlike TP, the
communication volume depends on routing and load balance. If a few popular
experts receive disproportionate traffic, other GPUs idle while the overloaded
experts determine layer latency.

The [Megatron Core Parallelism Strategies Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
shows that TP, PP, DP, and EP are composable. A 64-GPU allocation might contain
several replicas, each using TP within a node, PP across layer groups, and EP for
MoE blocks. The product of the parallel degrees determines the distributed
instance size, though communication groups can overlap. Inference configurations
must be evaluated along the actual data path, not merely by whether the weights
fit.

### Long sequences as a separate partitioning axis

For very long contexts, attention memory and computation can be partitioned
along the sequence dimension. Libraries use *sequence parallelism* and *context
parallelism* for related but non-identical schemes: some shard activations among
tensor-parallel ranks; others divide the context tokens themselves and exchange
K/V blocks. During prefill this can make an otherwise unplaceable input fit.
The decode benefit is less direct: the query has one position while the stored
KV cache is distributed, so partial attention results still have to be combined
at every token.

This axis is justified when context length, rather than weights, dominates
memory. It should not be layered automatically on TP and PP: another
communication dimension may add more latency than it saves in computation. A
report should state exactly what is partitioned—parameters, layers, experts,
prefill tokens, or the decode KV cache—because “parallelism” alone says nothing
about the data path.

## Communication determines the useful configuration

A distributed pass has a hard lower bound: it cannot finish before its required
communication. TP exchanges partial activations in nearly every layer; PP sends
full activations at stage boundaries; EP moves router-selected tokens twice; DP
normally exchanges only control data among otherwise independent replicas.
These patterns map differently onto cluster topology.

A practical rule is to keep the most frequent synchronous collectives within
the fastest communication domain. TP is commonly confined to one
NVLink/NVSwitch node; PP or DP may cross nodes; EP should avoid sending
all-to-all traffic through congested network tiers. This is not universal: long
prefill supplies enough computation to hide some communication, whereas the
small decode step is more exposed to every collective's latency.

## Why colocated prefill and decode interfere

A conventional serving engine mixes new prefill work with ongoing decode work.
This can improve average GPU utilization, but the phases then perturb each
other's user-facing latency. A long prefill delays the next token of an existing
response; a large decode batch delays a new request's prefill and first token.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-advanced/distributed-benchmarking/distserve-fig2-prefill-decode-interference.png]]

*Figure 2 from Yinmin Zhong et al., [DistServe: Disaggregating Prefill and
Decoding for Goodput-optimized Large Language Model Serving](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf),
OSDI 2024. For a 13B model, the authors measure batch execution time: adding one
prefill to a decode batch slows decode, while decode work lengthens prefill; the
effect is stronger for a 1,024-token input. This image is cropped from the
original vector figure in the public USENIX PDF.*

Chunked prefill limits the longest blocking unit but does not remove competition
for memory and compute units. Decode priority smooths responses already in
progress at the cost of queueing new requests; prefill priority reverses the
trade-off. A shared TP/PP configuration also has to compromise between phases
whose best batch sizes and parallel degrees differ.

## Separate prefill and decode pools

**Prefill/decode disaggregation** assigns the phases to different model
instances. The prefill pool receives the prompt, constructs its KV cache, and
produces the first token. Attention state is then transferred to a decode pool,
which continues generation. Each pool can scale independently: prefill against
the TTFT objective, decode against TPOT and the number of live sequences.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-advanced/distributed-benchmarking/distserve-fig6-runtime-architecture.png]]

*Figure 6 from [DistServe](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf).
The controller sends a request to a prefill instance, after which its KV cache
is transferred to a decode instance. This crop comes from the vector figure in
the USENIX PDF; the architecture and labels belong to the paper's authors.*

Disaggregation removes direct phase interference but adds a mandatory segment
to the critical path: **KV transfer**. Every layer's K and V tensors for every
input token must move. For $L$ layers, $n_{kv}$ KV heads, head dimension $d_h$,
prompt length $S$, and $b$ bytes per element, the volume is approximately

$$
M_{KV}=2L S n_{kv}d_h b.
$$

The factor of two accounts for K and V. GQA and MQA reduce $n_{kv}$, lowering
both decode memory and transfer cost. With long contexts, KV transfer can
consume a material fraction of TTFT; a fast network does not make it free.

DistServe exploits a useful property of pipeline parallelism: a layer's KV is
needed by its corresponding decode stage, not by every GPU. It colocates
matching prefill and decode segments within a node so the transfer traverses
NVLink rather than the inter-node network. Under bursts, a decode instance
pulls a ready cache from the prefill instance; prefill memory acts as a buffer
and prevents a push storm from exhausting decode memory.

## Choosing resources for each phase

Prefill often benefits from TP: a long matrix uses the compute units well, and
shorter execution directly improves TTFT. Once the GPU is saturated, however,
larger batches merely lengthen the queue. PP can increase rate capacity when its
stages remain occupied, though it raises the latency of an individual prompt.

Decode must retain many KV caches while producing tokens at a steady cadence.
Replication raises aggregate capacity; TP is needed when the model or cache does
not fit, or when one step exceeds the TPOT budget. Yet overly wide TP adds
synchronization at every token. MoE decode may also require EP, which saves
expert memory but makes all-to-all traffic and router imbalance operational
concerns.

There is consequently no universal rule such as “TP=8 for a large model.” The
design jointly selects prefill and decode replica counts, TP/PP/EP within each
pool, placement on the network topology, and routing policy. DistServe derives
this plan from prompt and output-duration profiles, arrival rate, and the two
SLO constraints. A workload change requires replanning: a configuration tuned
for short chat may be inefficient for long-document summarization.

## Routing, queues, and backpressure

Once the physical layout is fixed, online scheduling remains. A naive router
sends a request to the prefill replica with the shortest queue and its finished
cache to the least busy decode replica. Request counts are a poor proxy for work,
however: one 32K-token prompt can cost more than dozens of short ones, and a
long-context decode request consumes more KV memory. At minimum, a scheduler
should consider queued token counts, free cache blocks, and request age. PP also
depends on microbatch composition: prompts with similar execution times reduce
pipeline bubbles.

Disaggregation turns two local queues into a coupled producer–consumer system.
If prefill creates KV caches faster than decode consumes them, intermediate
memory fills even while prefill compute is idle. Continuing admission merely
moves the queue from the scheduler into HBM. **Backpressure** is required:
prefill must slow admission when decode lacks memory or network capacity.
Otherwise a brief burst can trigger eviction, repeated prefill, and still more
load.

Scaling the decode pool also raises the question of whether an existing cache
can be reassigned. Transfer is valid only when model revision, KV format,
parallel layout, and positional configuration match; sharing a model name is
not enough. Under TP/PP, the cache is sharded by ranks and layers, so the router
must know both the replica address and its shard mapping. Changing TP degree at
runtime usually requires repartitioning the state or running prefill again.

These constraints explain why disaggregated serving is not simply two
Deployments and a network call. It is one control plane for queues, memory, and
state compatibility. Its stability must be tested under long prompts, bursty
traffic, network slowdown, and decode-worker loss—not only a uniform synthetic
stream.

## When disaggregation is worthwhile

Disaggregation is most useful for sustained traffic with distinct phase compute
profiles and independent, strict TTFT and TPOT objectives. It permits separate
scaling and latency isolation. A small one- or two-GPU deployment, short prompts,
or sparse requests may lose more to duplicated weights and KV transfer than it
gains from isolation. A decode-instance failure also affects the prefill results
assigned to it, requiring rerouting, bounded queues, and memory controls.

The right comparison is not raw kernel tokens/s. Measure the share of requests
meeting TTFT and TPOT simultaneously; include every GPU in both pools, KV network
traffic, replica memory, and behavior under bursts. DistServe calls the resulting
measure *per-GPU goodput*: SLO-compliant requests served per second per allocated
GPU.

## Sources and further reading

- Stanford CS336, [Lecture 10: Inference](https://cs336.stanford.edu/spring2025/) — the prefill/decode distinction, arithmetic intensity, and serving workloads.
- Zhong et al., [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin), OSDI 2024 — interference, independent resource allocation, placement, and KV transfer.
- NVIDIA, [Megatron Core Parallelism Strategies Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html) — definitions and composition of DP, TP, PP, EP, and context parallelism.
- Shoeybi et al., [Megatron-LM](https://arxiv.org/abs/1909.08053) — the original tensor-parallel partitioning of Transformer layers.
