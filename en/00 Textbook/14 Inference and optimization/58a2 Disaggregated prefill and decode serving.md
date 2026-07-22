---
title: "Disaggregated prefill and decode serving"
type: textbook-chapter
status: canonical
locale: en
translation_of: "00 Учебник/14 Inference и оптимизация/58a2 Раздельное обслуживание prefill и decode.md"
last_updated: 2026-07-23
last_verified: 2026-07-23
primary_sources:
  - https://jax-ml.github.io/scaling-book/inference/
  - https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf
  - https://arxiv.org/abs/2311.18677
  - https://arxiv.org/abs/2407.00079
  - https://docs.nvidia.com/dynamo/design-docs/disaggregated-serving
  - https://sgl-project.github.io/advanced_features/pd_disaggregation.html
---

# Disaggregated prefill and decode serving

One generation request imposes two nearly opposite workloads on the system. The
model first consumes the entire prompt and constructs its KV cache. It then runs
many short forward passes, appending exactly one token at each step. The first
phase exposes substantial token-level parallelism and often uses tensor cores
well. The second repeatedly reads model weights and an ever-growing KV cache for
one autoregressive step. Running both phases in one GPU pool is operationally
simple and avoids moving state, but the phases compete for scheduler slots, HBM,
and a single parallel layout.

**Prefill/decode disaggregation** breaks that coupling. Prefill workers accept
prompts and produce KV state; decode workers receive it and continue generation.
This is not an unconditional acceleration. It replaces local interference with a
distributed protocol that transfers large state, coordinates two queues, and
requires joint capacity planning. We need a quantitative model of both phases
before deciding whether that exchange is worthwhile.

## One request, from admission to the final token

A simplified online request follows these stages:

1. the frontend validates input and performs chat templating and tokenization;
2. a router chooses a replica and considers prefix-cache overlap;
3. the scheduler places the prompt in a waiting queue;
4. prefill computes layer activations, keys, values, and final-position logits;
5. sampling selects the first output token, ending time to first token;
6. the request occupies a slot and KV blocks in a decode batch;
7. every decode step reads weights and KV, samples a token, and appends its KV;
8. EOS, a stop condition, cancellation, or a length limit releases the blocks.

The user observes at least two distinct latency classes. **Time to first token**
includes frontend work, queuing, and prompt computation:

$$
TTFT=T_{frontend}+T_{queue,P}+T_{prefill}+T_{first\ sample}.
$$

**Inter-token latency** or **time per output token** describes pauses after the
first token. For an $O$-token output,

$$
E2E=TTFT+\sum_{i=2}^{O}ITL_i.
$$

An average ITL can hide visible stalls: a user notices a long gap even if nearby
tokens arrived quickly. A serving evaluation therefore needs tail percentiles
and SLO attainment for TTFT and ITL, not only aggregate tokens per second.

## Why prefill is often compute-bound

All $S$ prompt tokens are known during prefill. Linear layers receive a large
token dimension and reuse loaded weights across many rows. Arithmetic intensity
grows with the number of tokens. With a sufficiently long prompt, the large
matrix multiplications can approach the compute roofline; tiled attention also
does substantial work for each block loaded from HBM.

Not every prefill is compute-bound. Short prompts, small batches, launch
overhead, quantization, or an inefficient tensor-parallel layout can make it
memory- or communication-bound. The system-level distinction still holds:
prefill has independent token positions and can form large matrix operations. A
single request has no comparable parallelism during decode.

Prefill cost is driven mainly by input length. Attention work grows rapidly with
context length; the linear blocks scale linearly in tokens but dominate model
parameters. A long prompt may occupy a GPU for the time of tens or hundreds of
ordinary decode steps. A scheduler that inserts it into an active decode batch
can therefore create a visible pause in already-streaming answers.

## Why decode is often memory-bandwidth-bound

After prefill, the next token is unknown until the current step finishes. One
request contributes only one new position, so the engine can increase token
batch size only by combining independent requests. Each step reads the model's
weights, while attention additionally reads every request's private KV history.

For a request with $S$ cached positions,

$$
M_{KV}=2L S n_{kv}d_h b,
$$

where $L$ is layer count, $n_{kv}$ is the number of KV heads, $d_h$ is head
dimension, $b$ is bytes per element, and the factor two accounts for keys and
values. MQA, GQA, MLA, and KV quantization reduce this quantity. They influence
not only capacity but also attention latency and the cost of moving KV between
serving pools.

For a modest decode batch, a useful lower bound is

$$
T_{step}\gtrsim
\frac{M_{weights}+\sum_{r\in batch}M_{KV,r}}
{\text{aggregate HBM bandwidth}}.
$$

Larger batches reuse weights more efficiently, but every additional request
brings its own KV state. Throughput consequently has diminishing returns, and
HBM capacity limits maximum concurrency. A large batch is efficient for the
operator but makes each step longer and can worsen ITL. An engine continually
chooses a point on this latency-throughput Pareto frontier.

## Interference in a shared GPU pool

Modern unified engines use iteration-level scheduling. After one forward pass,
the scheduler can remove completed sequences, admit new ones, and construct the
next batch. This is much more efficient than static batching, yet prefill and
decode still have different duration and resource profiles.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/distributed-serving-2026/distserve-fig2-prefill-decode-interference.png]]

*DistServe Figure 2 measures a 13B model after adding one prefill job to decode
requests. Decode waits for the longer prompt computation, while sharing also
slows prefill. Source: Zhong et al.,
[OSDI 2024](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf), cropped
from the original figure without content changes.*

Three policies are common, but none eliminates the conflict. Prefill priority
reduces admission delay and TTFT but creates ITL spikes for active streams.
Decode priority produces smoother streams while prompts wait. Chunked prefill
splits a long prompt and interleaves its chunks with generation. It bounds one
blocking interval but introduces a chunk-size tradeoff: small chunks underuse the
GPU and repeatedly read previous KV; large chunks recreate long stalls.

A shared pool also couples model placement. Prefill may prefer a narrower TP
degree and large compute-efficient matrix multiplications, while decode may use
wider TP to reduce weight-read latency. Decode retains many long-lived KV caches;
prefill needs only temporary state for current prompts. One layout is necessarily
a compromise.

## What disaggregation changes

A disaggregated architecture has at least two independently scalable model
instance types. A prefill instance consumes tokens, computes KV, and produces the
first token. A decode instance must receive compatible state, reserve memory
blocks, and admit the request into a continuous batch. Model weights normally
exist in both pools. The benefit comes from specialization and provisioning, not
from storing only one copy of the model.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/distributed-serving-2026/distserve-fig6-runtime-architecture.png]]

*DistServe Figure 6. The controller first assigns a prefill instance and then a
decode instance; KV cache becomes explicit transferred state. Source: Zhong et
al., [DistServe](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf),
cropped from the original figure.*

Disaggregation offers four potential advantages:

1. long prompt work no longer interrupts decode steps directly;
2. capacity can scale separately for TTFT and ITL targets;
3. phases can use different TP/PP/DP layouts or hardware;
4. prefill workers need not retain every active generation's KV cache.

It also imposes four unavoidable costs:

1. model weights are duplicated across the pools;
2. KV movement enters the request's critical path;
3. routing must coordinate prefill and decode destinations;
4. failure and backpressure propagate between stages.

Unified and disaggregated deployments must therefore be compared using all
allocated GPUs, network resources, and SLO attainment. Measuring only decode
tokens per second ignores the cost of prefill and transfer and produces a false
efficiency claim.

## KV transfer is a first-class serving stage

The transferred byte count is close to the prompt's KV footprint. Its lower
latency bound is

$$
T_{transfer}\ge
\frac{M_{KV}}{BW_{effective}}+T_{coordination}.
$$

A nominal 400-Gbit/s NIC does not guarantee 50 GB/s of useful KV movement.
Effective bandwidth depends on PCIe paths, GPUDirect RDMA, NIC count, NUMA
placement, memory registration, message fragmentation, concurrent flows, and
silent fallback to TCP. Under TP or PP, KV consists of shards: the system moves
a coordinated set of layer/head fragments among corresponding ranks rather than
one tensor between two processes.

Several transfer protocols are possible. With **push**, prefill sends KV as soon
as computation ends. It is simple, but a burst of completions can exhaust decode
memory. With **pull**, decode reserves capacity and fetches KV when ready;
DistServe uses prefill GPU memory as a queueing buffer. Pull provides explicit
backpressure but requires source KV to remain alive until acknowledgement.

**Layer-wise streaming** begins moving early-layer K/V while prefill computes
later layers. Mooncake describes overlapping computation and transfer. It helps
only if the network finishes near the final layer; otherwise the remaining tail
still increases TTFT. A remote or shared cache can expose blocks without
immediate eager copying, but autoregressive attention is usually too
bandwidth-sensitive to read remote KV on every step. Remote tiers are more
useful for bootstrap, prefix reuse, or offload.

Matching a model name is insufficient for KV compatibility. Revision,
dtype/quantization, layer layout, KV-head count, block size, positional scheme,
parallel-rank mapping, and backend serialization must agree. Different TP
degrees require explicit resharding. Some systems support heterogeneous layouts,
but the transformation belongs in latency and bandwidth measurements.

## Routing: load meets state locality

Least-loaded routing can be adequate for stateless services. LLM workers own
valuable state. A router must choose where prefill can exploit a cached prefix
and where future decode KV can reside. These objectives conflict.

Always selecting maximum prefix overlap creates a hotspot around popular system
prompts. Always selecting the shortest queue scatters identical prefixes and
increases recomputation. A practical score combines cache overlap, queued
tokens, estimated service time, free KV blocks, and topology. Under PD
disaggregation, routing chooses a pair or chain: a prefill worker, a decode
worker, and a viable transfer path.

Reserving decode capacity before prefill avoids completing expensive work only
to discover that no destination can hold it. Early reservations, however, keep
memory idle while prefill runs. Late assignment sees fresher load and improves
utilization but increases the chance of waiting or rerouting. The right policy
depends on joint input/output length distributions and burstiness.

KV-aware routing also needs an index of block ownership. Events may be exact,
durable, or approximate. Stale cache-hit information causes misses and
recomputation; stale capacity information causes admission failures. The control
plane rarely offers transactional consistency, so workers must validate router
decisions and the protocol must tolerate divergence.

## Two coupled queues and backpressure

After separation, prefill is a producer and decode is a consumer of long-lived
state. If prefill finishes faster than decode releases KV, an intermediate
backlog grows. Idle prefill tensor cores do not imply safe admission: the system
may be limited by downstream memory or generation rate.

Backpressure can use several signals:

- queued input tokens and predicted TTFT in the prefill pool;
- active and queued KV tokens and predicted ITL in decode;
- bytes awaiting transfer and measured network throughput;
- reserved, received, and evictable KV blocks;
- rejected, cancelled, and recomputed request rates.

Counting requests alone is a common mistake. A 128-token and a 128K-token prompt
represent different work. Decode requests with output budgets of 32 and 4096
tokens hold memory for different durations. Scheduling should use token units
and the joint distribution of input and output lengths.

Mooncake highlights an additional feedback effect. Admission based on current
decode load is delayed by prefill duration. A controller may accept many requests
while decode is idle; they arrive downstream as a wave. It then rejects new work,
prefill drains, decode eventually drains, and the cycle repeats. Predicting
future load and coordinating admission reduces these out-of-phase oscillations.

## Independently scalable does not mean independently controlled

Separate pools allow distinct replica counts, but their scaling decisions remain
coupled by workload. Prefill capacity depends mostly on arrival rate and input
lengths. Decode capacity depends on output rate, active context lengths, and KV
memory. A shift toward longer prompts requires more P capacity; longer reasoning
outputs require more D capacity.

Plain GPU utilization is a poor autoscaling signal. Compute-bound prefill can be
highly utilized while meeting TTFT. Memory-bound decode can show moderate compute
utilization while violating ITL. Better signals differ by component:

| Pool | Primary signals |
|---|---|
| Prefill | queued input tokens, ISL, TTFT percentiles, prefix-hit rate |
| Decode | active KV tokens, context lengths, ITL, free KV blocks |
| Transfer | pending bytes, transfer percentiles, RDMA throughput/fallback |
| Frontend | arrival rate, cancellations, deadlines and SLO class |

[NVIDIA Dynamo Planner](https://docs.nvidia.com/dynamo/components/planner/planner-guide)
separates longer-term capacity prediction from faster load reaction. A
performance model estimates P/D replica counts for target TTFT and ITL, while a
short-interval loop responds to queues and KV utilization. A prudent operational
workflow first runs recommendations in advisory mode against replayed traces
before allowing a planner to mutate production replicas.

Scale-up is not instantaneous. The platform must schedule GPUs, load weights,
initialize NCCL, warm kernels or CUDA graphs, and register the worker. If cold
start exceeds burst duration, a reactive autoscaler arrives too late. Minimum
warm capacity, prediction, or admission control is required. Scale-down is also
stateful: a decode worker must drain active sequences or migrate KV, and a
prefill worker cannot disappear during an unacknowledged transfer.

## Splitwise, DistServe, and Mooncake: three complementary emphases

**Splitwise** begins with hardware asymmetry. Prompt computation benefits from
strong compute accelerators, whereas token generation underuses compute and is
more dependent on memory. The paper studies homogeneous and heterogeneous pools,
cost, and power. It is an excellent source for hardware specialization, though
not a complete modern cache-aware routing design.

**DistServe** optimizes per-GPU goodput under simultaneous TTFT and TPOT SLOs. It
measures phase interference, chooses parallelism independently, and accounts for
network bandwidth during placement. Its runtime uses a controller,
prefill/decode instances, and pull-based KV transmission. Its central lesson is
that unconstrained maximum throughput can select a deployment unsuitable for an
interactive service.

**Mooncake** puts KV cache at the center. In addition to P/D pools, it uses CPU
DRAM and SSD as a disaggregated cache, prefix-aware scheduling, layer-wise
transfer, hot-block replication, and overload admission. This matters for
long-context and multi-turn workloads, where recomputing a known prefix may cost
more than locating and moving it.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/distributed-serving-2026/mooncake-figure1-architecture.png]]

*Mooncake Figure 1. The Conductor jointly controls cache-aware prefill, KV block
balancing, and decode, while CPU/DRAM/SSD form a distributed cache between GPU
pools. Source: Qin et al., [Mooncake](https://arxiv.org/abs/2407.00079), original
figure from the open technical report.*

These are not three mutually exclusive products. They emphasize different
layers of one system: phase-specific hardware, SLO-aware placement, and
KV-centric state management.

## How modern stacks realize the design

vLLM and SGLang are inference engines. They execute the model and manage KV and
scheduling inside workers. Both provide PD-disaggregation mechanisms and KV
connectors, but flags and supported transports evolve quickly. Their versioned
documentation belongs in reproducible labs, not in the definition of the
fundamental architecture.

Dynamo operates above engines and coordinates a frontend, KV-aware router, P/D
workers, transfer, and a planner. In its documented flow, prefill returns
backend-specific state metadata, the router injects it into a decode request, and
NIXL uses an available transport such as NVLink or RDMA/UCX. It is a useful
production case study: metadata differs by engine, but the request lifecycle is
the same.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/distributed-serving-2026/dynamo-architecture.png]]

*The official NVIDIA Dynamo architecture combines the request path, KV-aware
routing, disaggregated workers, and the memory/transfer layer in one runtime.
Source: [ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo/blob/main/docs/assets/img/architecture.png),
Apache-2.0.*

When examining any implementation, answer seven questions:

1. who reserves decode KV memory, and when;
2. who owns source KV until acknowledgement;
3. how the P/D pair is selected and prefix locality is scored;
4. which ranks exchange blocks over which transport;
5. whether different TP/PP layouts are supported;
6. how requests recover from P, D, or transfer failure;
7. which metrics control admission and autoscaling.

Calling a deployment “disaggregated” without these answers describes only its
boxes and arrows, not its operating behavior.

## When separation pays off

PD disaggregation is attractive under sustained high load, long or heterogeneous
prompts, strict ITL targets, large decode batches, and a fast KV-transfer path.
It isolates long-prefill spikes, permits different hardware and layouts, and can
use spare prefill memory for prefix caching.

Unified serving is often better for a small installation, low request rate,
short prompts, small models, or a slow inter-node network. It keeps one set of
weights per replica, avoids KV handoff, and has a simpler failure model. Chunked
prefill may provide enough latency isolation without a separate pool.

Make the decision with a controlled comparison using the same model revision,
quality settings, and workload trace. Measure p50/p95/p99 TTFT and ITL; SLO
attainment and goodput over **all** GPUs; HBM and KV utilization in both pools;
network bytes and transfer percentiles; prefix hit and recomputation rates;
behavior under bursts, long prompts, and worker failures; and cold-start and
resize time.

If separation improves the mean but worsens tail latency through a transfer
queue, it has not solved the service problem. If it raises raw throughput while
doubling GPU count and lowering per-GPU goodput, it is likewise not an efficiency
win. Its value is predictable user-facing SLO attainment at an acceptable
capacity cost.

## Sources and further reading

- Austin et al., [All About Transformer Inference](https://jax-ml.github.io/scaling-book/inference/) — prefill, generation, continuous batching, and disaggregated serving in one narrative.
- Zhong et al., [DistServe](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf), OSDI 2024 — interference, goodput, placement, and runtime KV transfer.
- Patel et al., [Splitwise](https://arxiv.org/abs/2311.18677), ISCA 2024 — phase-specific hardware, provisioning, power, and cost.
- Qin et al., [Mooncake](https://arxiv.org/abs/2407.00079) — KV-centric scheduling, distributed caching, layer-wise transfer, and overload handling.
- NVIDIA, [Dynamo Disaggregated Serving](https://docs.nvidia.com/dynamo/design-docs/disaggregated-serving) and [Architecture Flow](https://docs.nvidia.com/dynamo/latest/design-docs/architecture-flow) — a current orchestration protocol and NIXL transfer.
- SGLang, [PD Disaggregation](https://sgl-project.github.io/advanced_features/pd_disaggregation.html) — unified-scheduling interference and a current engine implementation.

**Previous:** [[58a Distributed inference and disaggregated serving|Parallelism and collective communication]]  
**Next:** [[58b Benchmarking SLOs and inference operations|Benchmarking, SLOs, and inference operations]]
