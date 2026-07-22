---
title: "Parallelism and collective communication in LLM inference"
type: textbook-chapter
status: canonical
locale: en
translation_of: "00 Учебник/14 Inference и оптимизация/58a Распределённый inference и disaggregated serving.md"
last_updated: 2026-07-23
last_verified: 2026-07-23
primary_sources:
  - https://jax-ml.github.io/scaling-book/sharding/
  - https://jax-ml.github.io/scaling-book/inference/
  - https://arxiv.org/abs/1909.08053
  - https://arxiv.org/abs/2104.04473
  - https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html
  - https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html
---

# Parallelism and collective communication in LLM inference

Distributed inference is needed for two different reasons. A single model
instance may not fit in one accelerator's memory, or its latency may be too high;
one request must then be computed by several GPUs. Alternatively, the model may
fit comfortably on one device while one instance cannot sustain the incoming
request rate. In that case, the service runs independent replicas and assigns
whole requests to them. Both designs use multiple GPUs, but they move different
data, depend on different links, and affect single-user latency differently.

This chapter therefore asks one question throughout: **which tensor is sharded,
where does each shard live before an operation, and where must the result live
after it?** The acronyms DP, TP, PP, SP, CP, and EP become useful only after that
question has an answer. Tracking tensor placement explains why a system needs
all-reduce, all-gather, reduce-scatter, all-to-all, or point-to-point transfers;
why tensor-parallel ranks usually stay inside an NVLink domain; and why a layout
that works well for prefill may perform poorly during decode.

## From one Transformer block to a GPU group

Let activations be $X\in\mathbb{R}^{B\times D}$, where $B$ is the number of
tokens processed together and $D$ is model width. A linear layer computes
$Y=XW$. On one GPU, $X$, $W$, and $Y$ are local. If $W$ does not fit, or the
matrix multiplication must be accelerated, it can be partitioned by columns:

$$
W=[W_1\;W_2\;\ldots\;W_p],\qquad
Y=[XW_1\;XW_2\;\ldots\;XW_p].
$$

Each of $p$ ranks receives the complete $X$, stores only $W_i$, and produces a
fragment of $Y$. No communication is required while the next operation accepts
that layout. If it requires a complete $Y$, the fragments must be gathered.
Partitioning the contracting dimension instead makes every rank produce a
partial sum, so a reduction is required. Most communication in a distributed
Transformer can be derived from these two matrix-multiplication cases.

[How to Scale Your Model](https://jax-ml.github.io/scaling-book/sharding/)
uses a particularly clear discipline: write both the logical tensor dimensions
and the axes of the device mesh along which they are partitioned. A tensor may be
replicated or sharded over batch, hidden width, sequence, heads, or experts. Two
tensors with the same global shape can therefore have entirely different
physical layouts and communication requirements.

## Collective operations as layout transformations

A collective is called coherently by every rank in a process group. It is not an
incidental network request after the computation; it is part of the distributed
algorithm. Until the collective completes, the next layer often does not have a
mathematically valid input.

### AllGather: shards become complete copies

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/distributed-serving-2026/nccl-allgather.png]]

*Official NCCL diagram. Every rank contributes one fragment and receives the
concatenation in rank order. Source: NVIDIA,
[Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html),
from the BSD-licensed NCCL documentation.*

If rank $i$ holds $x_i$, every rank holds
$[x_0,x_1,\ldots,x_{p-1}]$ after AllGather. The local output is $p$ times the
size of one shard. AllGather is needed when the next operator cannot consume the
partitioned representation. Sequence parallelism, for example, can keep
LayerNorm activations split by sequence and restore the layout expected by a
tensor-parallel linear layer only when necessary.

### ReduceScatter: sum and keep the result sharded

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/distributed-serving-2026/nccl-reducescatter.png]]

*ReduceScatter reduces corresponding input elements and leaves one result shard
on each rank. Source: official NCCL documentation.*

Suppose each rank computed a partial output $y_i$ with the same logical shape and
the complete answer is $y=\sum_i y_i$. If the following operator accepts a
partitioned $y$, replicating the whole sum is wasteful. ReduceScatter performs
the reduction while retaining only $1/p$ of the result locally. It reduces
activation memory and often forms a more efficient pair with a later AllGather
than an unconditional AllReduce.

### AllReduce: the complete reduction on every rank

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/distributed-serving-2026/nccl-allreduce.png]]

*AllReduce returns the reduced value to every rank. Conceptually, it can be
decomposed into ReduceScatter followed by AllGather. Source: official NCCL
documentation.*

Classic Megatron tensor parallelism uses AllReduce after a row-parallel
projection: ranks calculate partial sums over a partitioned contracting
dimension, while the following residual path expects the same complete
activation on every rank. A rough ring model for an $M$-byte message on $p$
ranks is

$$
T_{AR}\approx 2(p-1)\alpha+2\frac{p-1}{p}\frac{M}{\beta},
$$

where $\alpha$ is per-step latency and $\beta$ is effective bandwidth. NCCL uses
topology- and size-dependent algorithms, so this is not a performance predictor.
It does expose the two limiting regimes: small decode activations are sensitive
to latency, whereas larger prefill messages are more bandwidth-sensitive.

### AllToAll: a personalized permutation among ranks

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/distributed-serving-2026/nccl-alltoall.png]]

*In AllToAll, each rank sends a different fragment to every destination and
receives a different fragment from every source. Source: official NCCL
documentation.*

AllToAll is not a reduction. It changes ownership. In a mixture-of-experts
layer, the router assigns each token to one or more experts; tokens initially
held by different ranks must move to the ranks that own those experts. A second
AllToAll returns the outputs. Expert parallelism therefore stresses bisection
bandwidth and is more sensitive to token imbalance than dense tensor
parallelism.

Pipeline parallelism, by contrast, mostly uses point-to-point send/receive
between adjacent stages. A message follows a specific pipeline edge rather than
being exchanged by all ranks, and its cost appears as transfer latency and idle
time when stages are imbalanced.

## Data parallelism: independent serving replicas

During training, data parallelism means identical weights, different
mini-batches, and gradient synchronization. Ordinary inference has no gradients.
It is clearer to think in terms of **request-level replication**: each replica
contains a complete executable model instance, and a router assigns entire
requests to replicas.

DP increases aggregate request capacity almost linearly until a shared frontend,
tokenizer, storage layer, or network becomes the next bottleneck. It neither
reduces memory required by one instance nor accelerates one forward pass. Its
advantages are operational: replicas do not synchronize at every layer, build
continuous batches independently, and provide natural failure boundaries.

Replicas are not completely stateless. Each owns a KV cache and a local prefix
cache. Routing solely to the shortest queue balances compute but may destroy
prefix locality; routing solely to the largest cache match creates hot replicas.
In MoE deployments, DP can also denote specialized groups where attention is
replicated while experts are partitioned. A statement such as `DP=8` is therefore
incomplete without the model placement and request-routing policy.

## Tensor parallelism: partitioning a layer's matrices

Megatron-LM uses a conjugate pair of column- and row-parallel linear layers. The
first MLP matrix is split along its output dimension. GeLU or SwiGLU is applied
locally because each fragment is independent. The second matrix is split along
its input dimension; ranks produce partial sums that are then reduced. Q, K, and
V can similarly be split by heads, followed by a row-parallel output projection.

The arrangement is attractive because no communication is needed between the
two large MLP matrix multiplications, and synchronization occurs at a few known
points in each block. Those points nevertheless repeat for every Transformer
layer. If TP crosses a slow inter-node fabric, tens of collectives accumulate in
every generated token.

Prefill presents many tokens to each matrix multiplication. Kernels are large,
and some communication can be overlapped with useful work. Decode contributes
one token per active sequence. Its local GEMMs are shorter, while collective
latency remains. On the other hand, TP distributes weights across several HBM
channels and can reduce the lower bound imposed by reading parameters. Decode
can therefore benefit from wider TP even while memory-bound, but only until
communication costs more than the saved HBM traffic.

A practical TP choice begins with three measurements:

1. whether weights, runtime buffers, and the target KV budget fit;
2. whether measured step latency improves from TP1 to TP2, TP4, and TP8;
3. whether all TP ranks remain in a fast NVLink/NVSwitch domain.

Cost must also be considered. TP8 may minimize latency but deliver fewer tokens
per second per GPU than two TP4 replicas. There is no universally optimal TP
degree independent of workload and SLO.

## Pipeline parallelism: partitioning depth

PP places consecutive layer groups on different stages. Activations travel from
stage 0 to stage 1 and onward. Unlike TP, most layers need no all-rank collective;
communication occurs at stage boundaries. This can place a model that exceeds
one node and can use slower inter-node links for relatively compact activations.

A single request is not parallel across depth: its next stage waits for the
previous stage. Throughput appears when several microbatches occupy different
stages simultaneously. During pipeline fill and drain, some devices are idle—the
pipeline bubble. In online inference, bubble size depends on variable prompt
lengths, sequences leaving the decode batch, and unequal layer costs, not merely
on the number of stages.

An equal number of layers per stage does not guarantee balance. Embeddings and
the output head have different cost, MoE layers depend on routing, and attention
cost changes with context. Megatron Core therefore supports custom layouts and
virtual stages. Serving layouts must be tested on the target input/output length
distribution rather than only uniform synthetic sequences.

PP is commonly added when a model still does not fit under a sensible
intra-node TP degree. A TP8×PP4 instance has four stage groups, each implemented
by eight GPUs. A request crosses four stages, and the layers within each stage
still execute TP collectives.

## Sequence parallelism and context parallelism are not synonyms

Library terminology varies, but Megatron uses these methods for distinct
purposes. **Sequence parallelism (SP)** complements TP: operators such as
LayerNorm and dropout, which do not couple tokens, operate on a partitioned
sequence dimension. Paired ReduceScatter and AllGather operations avoid keeping
all relevant activations replicated.

**Context parallelism (CP)** partitions the network input and all activations by
sequence, including attention. Linear layers remain local, but each query must
see keys and values from its permitted causal prefix. Ranks must exchange KV
blocks or combine partial attention results.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/distributed-serving-2026/megatron-context-parallel-overview.png]]

*A Transformer layer under TP2×CP2. Communication around attention belongs to
CP; communication around linear blocks belongs to TP. AG and RS denote AllGather
and ReduceScatter. Source: NVIDIA,
[Megatron Core Context Parallelism](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html),
Figure 1.*

For long prefill, CP reduces per-rank activation memory and distributes very long
contexts. During decode, the new query has sequence length one while the historic
KV cache is long. Query distribution must be distinguished from KV sharding: KV
can be partitioned by heads, batch, or context, but partial attention outputs
still need to be combined at every token. CP is not a free decode acceleration
and is especially sensitive to inter-rank latency.

## Expert parallelism: parameters stay, tokens move

A dense MLP applies one parameter set to every token. An MoE has $E$ experts and
a router typically chooses $k\ll E$ of them per token. EP partitions experts to
avoid replicating every parameter. Each MoE layer then performs:

1. routing to expert indices and weights;
2. grouping tokens by destination;
3. AllToAll dispatch to expert owners;
4. local grouped GEMMs;
5. a reverse AllToAll;
6. restoration of token order and weighted top-k combination.

Compute is determined by activated parameters, capacity by all resident experts,
and communication by routed tokens. These are different quantities. Average
balance does not guarantee low tail latency: one overloaded expert delays the
collective group. Capacity factor, padding or token dropping, expert replication,
and Expert Parallel Load Balancing alter quality, memory, and latency and must be
reported in a benchmark.

EP also composes with TP. If each expert is itself tensor-parallel, the system
first dispatches tokens to an expert group and then runs collectives inside that
group. DeepSeek-style deployments may combine EP, expert TP, attention DP, and
PP. The product of degrees is not a sufficient network description; readers need
a rank map and the sequence of collectives in dense and MoE layers.

## Topology turns an algorithm into performance

Data moves through several domains: HBM inside a GPU, PCIe, NVLink/NVSwitch
inside a server, and InfiniBand or RoCE between servers. Their bandwidth and
latency differ dramatically. NUMA placement, NIC count, leaf-spine
oversubscription, and NCCL's actual route matter as much as the advertised link.

Frequent synchronous collectives should occupy the fastest domain. This leads to
a useful, non-universal rule of thumb:

- keep TP inside an NVLink/NVSwitch node;
- place EP within a high-bisection-bandwidth domain;
- allow PP to cross nodes when activation transfers are modest;
- place independent DP replicas in separate network or failure domains.

The workload can reverse a simple rule. Long prefill has large GEMMs and may hide
communication; short decode steps expose every synchronization. Large messages
are bandwidth-bound, while small messages are governed by latency and launch
overhead. Naming the interconnect is not a substitute for benchmarking the
collective message sizes and process groups used by the model.

A useful diagnostic order is: establish the single-GPU kernel and HBM roofline;
microbenchmark the required collectives; inspect layer timelines; only then run
end-to-end serving. Low GPU compute utilization does not prove insufficient
traffic. A rank may be waiting for a collective, a remote expert, or the next
pipeline stage.

## Choosing a DP×TP×PP×CP×EP composition

Start with the smallest group that fits weights and a realistic KV budget. For a
dense model, use intra-node TP first and add PP if required. Turn remaining GPUs
into independent service replicas. Add CP only when long-context memory or
latency demands it. MoE architecture requires experts, but EP degree and expert
placement remain engineering choices.

For every candidate, record which weights are replicated or sharded, where
activations and KV live, the collective sequence of dense and MoE layers, node
and NIC boundaries, maximum resident KV tokens, and measured TTFT, ITL,
throughput, and tokens per second per GPU.

“32 GPUs” is almost meaningless. “Four replicas, each TP8, with every TP group
inside one NVSwitch node and a cache-aware request router” makes the data path
understandable. An MoE deployment may require a still richer statement:
attention DP4, EP32, expert TP1, redundant shared experts, and a distinct
communication group for token dispatch.

Parallelism describes how model instances use accelerators and interconnects. It
does not decide whether prefill and decode belong in the same instance. They can
share one distributed replica or run in independently scaled pools with
different layouts. The latter introduces KV transfer between instances and turns
routing, backpressure, and autoscaling into coupled problems. That is the subject
of the next chapter.

## Sources and further reading

- Austin et al., [Sharded Matrices](https://jax-ml.github.io/scaling-book/sharding/) — deriving collectives from distributed tensor layouts.
- Austin et al., [All About Transformer Inference](https://jax-ml.github.io/scaling-book/inference/) — roofline analysis for prefill, decode, and KV-cache sharding.
- Shoeybi et al., [Megatron-LM](https://arxiv.org/abs/1909.08053) — the original tensor-parallel Transformer layout.
- Narayanan et al., [Efficient Large-Scale Language Model Training Using Megatron-LM](https://arxiv.org/abs/2104.04473) — composing tensor, pipeline, and data parallelism.
- NVIDIA, [Megatron Core Parallelism Strategies](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html) — current DP, TP, PP, CP, and EP terminology.
- NVIDIA, [NCCL Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) — exact definitions of the communication primitives.

**Previous:** [[58 Speculative decoding|Speculative decoding]]

**Next:** [[58a2 Disaggregated prefill and decode serving|Disaggregated prefill and decode serving]]
