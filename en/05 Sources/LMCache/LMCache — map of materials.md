---
title: "LMCache: architecture, storage, transfer, and operations"
type: source-note
status: active
locale: en
translation_of: "05 Источники/LMCache/LMCache — карта материалов.md"
last_verified: 2026-07-23
---

# LMCache: a distributed memory layer for LLM inference

An inference engine normally treats the key–value cache as private, temporary
state. The cache is allocated in GPU memory while a request is running and is
discarded when the request ends or the worker exits. That design is natural for
single-request decoding, but it wastes work whenever several requests process
the same long context, whenever a conversation returns to an earlier prefix, or
whenever prefill and decoding run on different workers.

LMCache changes the ownership boundary. The inference engine still executes the
Transformer, but KV tensors can be named, moved, stored, looked up, and reused
outside the lifetime of one request. In the terminology of the
[LMCache system paper](https://arxiv.org/abs/2510.09665), KV cache becomes both
a storage object and a communication medium between inference engines. This
single change connects several apparently separate serving techniques:

- prefix reuse avoids recomputing a context that another request has already
  prefetched;
- multi-tier offloading exchanges scarce GPU memory for larger CPU, local-disk,
  or remote-storage capacity;
- prefill/decode disaggregation transfers computed KV state from a prefill
  worker to a decode worker;
- CacheBlend makes selected non-prefix chunks reusable, with partial
  recomputation to repair missing cross-chunk dependencies;
- serialization and transformation layers trade precision or CPU work for
  storage capacity and transfer bandwidth.

The important idea is therefore not “put the KV cache on disk.” LMCache is a
distributed state layer with a data plane, a control plane, storage policies,
failure modes, and an economic break-even point. This chapter follows the
current multiprocess architecture described in the
[official MP documentation](https://docs.lmcache.ai/mp/index.html). Older
in-process examples remain useful historically, but the project marks that mode
as deprecated.

## 1. When moving KV is better than recomputing it

For a prompt of $N$ tokens, let $T_{\text{prefill}}(N)$ be the time required
to compute its KV tensors. Reuse is beneficial only when the complete lookup and
load path is cheaper:

$$
T_{\text{lookup}} +
T_{\text{queue}} +
T_{\text{read}} +
T_{\text{transfer}} +
T_{\text{inject}}
<
T_{\text{prefill}}(N).
$$

This inequality is more useful than an unconditional claim that “cache hits
reduce TTFT.” A hit in a slow object store may lose to recomputation for a short
prefix. A long cached document transferred over a fast local path may save most
of prefill. Contention can reverse the result: the nominal bandwidth of NVMe or
RDMA is not the request-level bandwidth after queueing and concurrent traffic.

KV capacity also grows rapidly. For a conventional attention model, an
approximate per-request footprint is

$$
C_{\text{KV}}
\approx
2 \, N \, L \, H_{\text{kv}} \, D \, b,
$$

where $L$ is the number of layers, $H_{\text{kv}}$ the number of KV heads,
$D$ the head dimension, $b$ bytes per stored element, and the factor two
accounts for keys and values. Grouped-query attention lowers
$H_{\text{kv}}$, while quantization lowers $b$; neither removes linear
growth with context length. If a storage tier of usable capacity $M$ holds
objects of mean size $\bar C$, its rough capacity is

$$
K_{\text{objects}} \lesssim \frac{M}{\bar C}.
$$

The useful capacity is lower once allocator overhead, partial chunks, replicas,
temporary transfer buffers, and eviction watermarks are included. Production
planning should use observed object sizes and reuse-distance distributions, not
only the formula.

## 2. The multiprocess architecture

In current MP mode, LMCache runs as a daemon separate from vLLM. One server can
serve several engine processes or pods on a node. The separation has three
consequences. Cache failures and Python runtime work do not have to share the
engine process; CPU cache capacity can scale independently of GPU allocation;
and several engines can reuse a common node-local cache.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/lmcache-blog-2026/mp-mode-architecture.png]]

The figure is best read as two cooperating planes:

1. **The engine-facing data path.** An LMCache connector participates in the
   engine's paged-KV lifecycle. It asks whether prompt chunks exist, retrieves
   hits, injects them into the engine's KV pages, and submits newly computed
   chunks for asynchronous storage.
2. **The storage and management path.** The MP server owns the index, CPU or
   local-storage objects, L2 adapters, eviction state, background prefetch and
   store work, metrics, and management endpoints.

The [MP architecture guide](https://docs.lmcache.ai/mp/architecture.html)
describes a `StorageManager` over an L1 manager and one or more L2 adapters.
Three background controllers make the apparently simple `lookup/store/retrieve`
interface work:

- the **StoreController** observes completed L1 writes and submits asynchronous
  L1-to-L2 stores according to policy;
- the **PrefetchController** resolves misses in L1 against L2 and loads found
  objects back into L1 before retrieval;
- the **EvictionController** monitors capacity and removes victims according to
  the configured policy and watermark.

Objects are protected by read/write locks and pass through a lifecycle rather
than being unstructured byte arrays: a writer reserves space, commits a ready
object, readers hold it while copying, and eviction can act only when lifecycle
constraints permit. Timeouts exist so that a crashed client does not retain a
lock forever.

The write path is deliberately asynchronous. After the engine produces new KV,
the request should not have to wait for a remote store to persist it. The read
path is latency-sensitive: lookup, L2 prefetch, L1 readiness, and device
injection are on the critical path to resumed computation. This asymmetry
explains why a backend that is acceptable for durable writes may still be a poor
source for interactive cache hits.

## 3. Identity, chunks, and lookup

LMCache does not identify an object merely by a filename or by raw prompt text.
The cache is divided into token-aligned chunks, hashed, and associated with
metadata required to interpret the tensor. The current
[HTTP API documentation](https://docs.lmcache.ai/mp/http_api.html) exposes an
encoded object key containing:

- the chunk hash;
- model name;
- KV rank;
- object-group identifier;
- optional `cache_salt`.

These fields prevent several invalid forms of reuse. KV produced by different
models is not interchangeable. Tensor-parallel ranks own different tensor
shards. Hybrid architectures may require separate object groups for different
attention or state layouts. A salt creates a distinct key namespace even when
the token sequence is identical.

Chunk size is a workload trade-off. Large chunks reduce indexing and transfer
overhead but turn small prompt differences into larger misses and can waste
space at boundaries. Small chunks improve matching granularity but increase
hashing, metadata, RPC, and I/O overhead. The official quickstart recommends a
production-scale chunk size such as 256 tokens rather than treating the smallest
possible block as automatically optimal.

The key is necessary but not a complete compatibility proof. Operators must
also keep model revisions, tokenizer and chat-template versions, tensor dtype,
attention layout, and serving configuration compatible. A shared model name
does not make caches produced by materially different artifacts safe to
exchange. Version boundaries should therefore be reflected in deployment
isolation or in an explicit namespace convention.

For ordinary prefix reuse, lookup walks complete chunks from the beginning of
the prompt and stops when the contiguous match ends. A partial last chunk is
normally recomputed. On an L1 miss, the prefetch controller asks configured L2
adapters, loads available chunks into L1, and keeps them read-locked until the
engine retrieves them. Cancellation must release these lookup locks; session
end is therefore part of correctness, not only cleanup.

Attention layout belongs in the keying discussion. Uniform-attention models can
share one regular KV layout. Hybrid models containing full attention,
sliding-window attention, Mamba, or linear-attention state require explicit
object-group separation and model-specific validation. The official
[LMCache recipes](https://docs.lmcache.ai/) document supported layouts and
limitations. A cache hit is useful only if tensor identity and layout are
correct; matching tokens alone is insufficient.

## 4. Storage tiers and policies

LMCache conventionally names engine GPU memory **L0**, server-local fast storage
**L1**, and persistent or remote adapters **L2**. These are roles, not universal
latency promises. For example, MP L1 can be pinned CPU DRAM, while a configured
NVMe slab accessed with GPUDirect Storage changes both its capacity and transfer
path.

The current [L2 storage guide](https://docs.lmcache.ai/mp/l2_storage.html)
lists a broad adapter surface:

| Family | Officially documented options | Appropriate question |
|---|---|---|
| Node-local memory/storage | CPU DRAM, filesystem adapters, native filesystem, raw block, DAX | Does capacity on one node cover the reuse window? |
| NIXL storage | POSIX, GDS, multi-threaded GDS, HF3FS, object, Azure Blob | Can the storage path exploit the available interconnect or direct I/O? |
| Distributed stores | Mooncake Store, Redis/Valkey, InfiniStore, Aerospike | Is cross-instance sharing worth network and service complexity? |
| Object storage | S3-compatible stores, Azure Blob, Hugging Face Buckets | Is persistence/capacity more important than first-hit latency? |
| Extension/testing | custom plugin, mock, fault-injection adapters | Is a new backend or failure policy being validated? |

This table is a taxonomy, not a benchmark ranking. Backend names do not specify
the deployed hardware, topology, concurrency, or object size. Measurements must
include end-to-end L2 queue and I/O time.

Multiple adapters can form a cascade. Store and prefetch policies decide which
adapters receive objects and where misses are searched. Prefetch concurrency
controls parallelism; buffer-only mode can make L1 a staging area rather than a
long-lived cache. L1 eviction is watermark-driven. L2 eviction is configured
per adapter, so each tier can enforce a separate capacity budget.

The capacity decision should combine hit probability with cost:

$$
E[T] =
p_{\text{L0}}T_{\text{L0}}+
p_{\text{L1}}T_{\text{L1}}+
p_{\text{L2}}T_{\text{L2}}+
p_{\text{miss}}T_{\text{prefill}}.
$$

Adding a tier is beneficial when it shifts enough probability away from more
expensive states to pay for its queueing, transfer, storage, and operational
cost. Reuse distance is often more informative than aggregate hit rate: an
object that is hit only after hours may justify inexpensive object storage but
not pinned DRAM.

## 5. From engine pages to LMCache objects

Inference engines manage KV as many small paged blocks, whereas storage systems
and networks prefer larger transfers. LMCache therefore gathers or exposes
engine blocks, moves chunk-sized objects, and scatters retrieved data into the
destination pages.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/lmcache-blog-2026/mp-transfer-paths.png]]

The MP implementation supports distinct local transfer strategies:

| Path | Mechanism | Strength | Constraint |
|---|---|---|---|
| LMCache-driven GPU path | CUDA IPC exposes device allocations; the server performs the transfer | Avoids serializing tensor payloads through RPC | Requires compatible local GPU/IPC setup and careful allocation lifetime |
| Engine-driven path | Worker gathers/scatters and participates in `PREPARE`/`COMMIT` | Works when the server cannot directly operate on engine device memory | More work remains in the engine-side process |
| POSIX shared memory | Non-GPU payload passes through shared memory | Avoids copying large payloads through ZMQ | Requires shared IPC namespace and managed slots |
| Pickle fallback | Serialized bytes travel through the protocol | Portable and simple for fallback/testing | Additional copies and serialization overhead |

The official [configuration reference](https://docs.lmcache.ai/mp/configuration.html)
distinguishes `lmcache_driven`, `engine_driven`, and `auto` routing. The choice
must not be reduced to “CUDA IPC is always faster.” Topology, tensor layout,
concurrent CUDA work, CPU launch overhead, NUMA placement, and block aggregation
all affect the result. The
[LMCache benchmark CLI](https://docs.lmcache.ai/cli/bench.html) exists to check
transfer integrity and performance for the actual path.

## 6. Cross-node sharing and prefill/decode disaggregation

Local offloading and P/D disaggregation use the same data type but solve
different scheduling problems. Offloading stores KV so a future request can
reuse prior computation. P/D disaggregation transfers KV produced for the
current request from a prefill worker to a decode worker.

In a disaggregated request:

1. a router assigns the prompt to a prefill worker;
2. prefill computes KV, potentially overlapping chunk transfer with later
   computation;
3. a decode worker obtains the complete required state;
4. decoding starts only after the destination can address the transferred KV;
5. lifecycle and failure handling release or retry partially transferred state.

NIXL-backed transport can use NVLink, RDMA, or TCP depending on deployment. P2P
sharing allows one LMCache server to read objects from another, while a
centralized L2 store offers a different topology. The current
[server reference](https://docs.lmcache.ai/cli/server.html) exposes P2P
advertisement, discovery, lookup timeouts, load timeouts, and a transfer-engine
choice.

The P/D break-even condition includes overlap:

$$
T_{\text{PD}} =
\max(T_{\text{remaining prefill}},T_{\text{KV transfer}})
+
T_{\text{handoff}}+T_{\text{decode queue}}.
$$

Disaggregation is attractive only when improved specialization, batching, or
decode scheduling outweighs handoff and queue costs. It is not a guaranteed
TTFT improvement. A deployment should separately report prefill queue, prefill
compute, transfer, handoff, decode queue, and inter-token latency.

## 7. Control plane and fleet coordination

One MP server can manage its own objects through HTTP and ZMQ interfaces. A
multi-server deployment adds the
[MP Coordinator](https://docs.lmcache.ai/mp/coordinator.html): a standalone
service that tracks registrations and heartbeats, aggregates L2 events, provides
a fleet view, and dispatches management operations.

The control surface includes lookup/status, prefetch, deletion/clear, pinning,
movement, compression, health, and completion checks. Pinning is operationally
important: it distinguishes a cache object protected for a planned workload
from an object eligible for ordinary eviction. Prefetch turns cache warming into
an explicit orchestration action rather than waiting for a user request.

Fleet coordination should not be confused with application request routing.
The coordinator knows cache instances and management state. Selection of a
prefill or decode endpoint, session affinity, admission control, and
cache-aware request routing belong to the surrounding serving stack. The
official [vLLM Production Stack](https://github.com/vllm-project/production-stack)
is one adjacent project; features implemented there should not be attributed to
the LMCache core.

## 8. CacheBlend: reuse beyond a common prefix

Prefix caching is exact because causal attention ensures that a token's KV state
depends only on the tokens before it. Moving a text chunk to a new position or
placing another retrieved document before it changes that history. Reusing the
old KV unchanged would omit cross-chunk attention and can alter model quality.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/lmcache-blog-2026/cacheblend-agent-context.jpg]]

[CacheBlend](https://arxiv.org/abs/2405.16444) addresses this RAG-shaped case.
It stores precomputed chunks, retrieves matches even when they are not a single
contiguous prefix, and selectively recomputes a subset of tokens to repair the
largest discrepancies. Retrieval and partial recomputation can overlap. The
method is therefore neither naive KV concatenation nor a claim that arbitrary
state is position-independent.

Current MP internals include progressively richer Blend protocols, including a
unified prefix/non-prefix lookup, sparse coalesced prefetch, per-token scatter
into paged KV, and RoPE correction for shifted positions. Those implementation
details matter because positional rotation and destination block placement are
part of semantic correctness.

The paper reports 2.2–3.3× lower TTFT and 2.8–5× higher throughput than full KV
recomputation on its evaluated models, datasets, and workload. These are paper
results, not universal production guarantees. A valid evaluation must compare
answer quality as well as latency, preserve document order and prompt template,
and include the selective recomputation cost.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/lmcache-blog-2026/cacheblend-results.png]]

## 9. Compression, serialization, and editable KV state

Moving KV can become bandwidth-bound. LMCache therefore permits a serializer
and deserializer on each L2 adapter. According to the official
[serde guide](https://docs.lmcache.ai/mp/serde.html), current built-ins include
FP8 and TurboQuant presets; the interface can also host lossless compression,
other lossy encodings, or authenticated encryption.

For a compression ratio $r=C_{\text{raw}}/C_{\text{encoded}}$, compression is
useful when

$$
T_{\text{encode}}+
\frac{C_{\text{raw}}}{rB}+
T_{\text{decode}}
<
\frac{C_{\text{raw}}}{B},
$$

subject to an acceptable quality change for lossy transforms. The inequality
also explains why a ratio alone is insufficient: a fast link or small object
may not repay codec overhead.

[CacheGen](https://arxiv.org/abs/2310.07240) is a specialized KV encoder and
streaming method. It exploits KV distributional properties and can adapt
compression to available bandwidth. Its paper reports 3.5–4.3× smaller KV and
3.2–3.7× lower context fetch-plus-processing delay in the evaluated setup.
TurboQuant and FP8 are different mechanisms and should not be described as
synonyms for CacheGen.

LMCache also exposes a
[KV Cache SDK](https://docs.lmcache.ai/) for retrieving KV to CPU, applying a
user-defined edit, and storing the result. Token dropping is the documented
example: reducing retained state can increase decode batch capacity. This is a
model-quality intervention, not merely lossless storage optimization, so it
requires task evaluation and must preserve block-alignment and layout
constraints.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/lmcache-blog-2026/classic-cache-vs-kv-state.png]]

## 10. Observability and production operation

A cache layer can make latency better on average while making tail behavior
harder to explain. LMCache's
[observability system](https://docs.lmcache.ai/mp/observability.html) uses an
internal event bus with metrics, logging, and tracing subscribers. The official
metric set includes:

- requested and hit tokens for prefix hit-rate calculation;
- L0↔L1 and L1↔L2 request-level throughput;
- L1 object lifetime, idle-before-eviction, and reuse gaps;
- real workload reuse gaps;
- per-adapter store/load completion and failures;
- active prefetch jobs and tier usage;
- engine deliveries grouped by worker, model, and cache salt;
- remote-backend health and ping latency.

The distinction between device-copy throughput and end-to-end adapter
throughput is essential. L2 histograms include queue, network, and disk time;
they should not be presented as raw media bandwidth. Storage traces can be
recorded for replay and regression analysis.

The recommended Kubernetes pattern is one LMCache MP server per selected node,
shared by inference pods. The
[deployment guide](https://docs.lmcache.ai/mp/deployment.html) and
[Kubernetes Operator](https://docs.lmcache.ai/mp/operator.html) cover the
DaemonSet, node-local service, connection ConfigMap, CUDA IPC requirements, and
optional Prometheus `ServiceMonitor`. Health probes are necessary but do not
replace workload metrics: a responsive server can still have a saturated L2
queue or a useless hit rate.

## 11. Multi-tenancy and security boundaries

`cache_salt` participates in object identity and enables per-tenant metrics.
With `IsolatedLRU`, each salt has a separate recency domain and a soft quota
managed through the
[quota API](https://docs.lmcache.ai/cli/quota.html). The fleet coordinator can
aggregate L2 usage and enforce corresponding eviction budgets.

Three caveats must remain explicit:

1. a quota is an eviction budget, not admission control; writes may exceed it
   until the next eviction cycle;
2. a salt is a namespace/isolation label, not proof of caller identity;
3. the reviewed official documentation does not establish built-in end-to-end
   authentication, authorization, or TLS as a general LMCache security
   boundary.

Consequently, ZMQ, HTTP management endpoints, and storage credentials should be
placed behind the deployment's trusted-network, identity, and policy controls.
For data at rest, the serde interface can wrap objects in authenticated
encryption, but key management and authorization remain external system
responsibilities. Sensitive prompt state should be assigned retention,
deletion, and audit policies just like other persisted application data.

## 12. Integrations and support status

The official [integration guide](https://docs.lmcache.ai/developer_guide/integration.html)
documents vLLM and SGLang support and labels TensorRT-LLM as coming soon. The
upstream vLLM connector and MP connector should be distinguished from deprecated
in-process examples. Installation documentation covers CUDA, ROCm, and Intel
XPU paths, but availability of a package is not evidence that every backend,
codec, and transfer mode has equal maturity on every platform.

KV reuse is not limited to text-only prompts. LMCache documents multimodal
caching through its vLLM integration, including an Ultravox example. The same
identity rule becomes stricter here: cached state depends on the processed media
inputs and their preprocessing path, not merely on visible text. Hybrid
attention recipes likewise cover models that combine ordinary attention with
sliding-window, Mamba, or linear-attention state. Both areas must be validated
per architecture rather than inferred from support for a neighboring
Transformer model.

| Capability | Status in official material reviewed 2026-07-23 | What to verify locally |
|---|---|---|
| vLLM connector | Supported; MP mode is the recommended architecture | Exact vLLM/LMCache compatibility and KV layout |
| SGLang integration | Listed as supported | Feature parity for the intended cache and transfer mode |
| TensorRT-LLM | Marked “coming soon” | Do not plan production use from this label alone |
| CUDA local transfer | CUDA IPC and GPU paths documented | IPC namespace, driver/runtime, topology |
| ROCm / Intel XPU | Installation paths documented | Supported connector, model, transfer, and backend combination |
| Kubernetes | Manual deployment and Operator documented | Security policy, storage, NUMA and observability integration |

## 13. A deployment decision guide

| Workload condition | Start with | Add only after measurement | Primary metrics |
|---|---|---|---|
| Repeated long prefixes on one node | MP server with CPU L1 | Local NVMe/GDS when reuse outlives DRAM capacity | hit tokens, L0↔L1 load time, reuse distance |
| Shared documents across nodes | Distributed L2 or P2P | Cascade/object storage for longer retention | L2 hit latency, queue time, backend failures |
| Separate prefill and decode pools | NIXL-capable transport and explicit P/D routing | More replicas after queue/topology study | transfer overlap, handoff time, TTFT and ITL |
| RAG with reordered chunks | CacheBlend with quality evaluation | More aggressive selective reuse | TTFT, recompute fraction, task quality |
| Bandwidth-limited L2 | FP8/TurboQuant or evaluated custom serde | CacheGen/custom codec when workload justifies it | encoded bytes, codec time, quality delta |
| Multi-tenant service | `cache_salt`, IsolatedLRU, quotas and per-salt metrics | Fleet coordinator and external policy controller | per-salt usage, eviction, hit rate |
| Untrusted network or sensitive prompts | External network/auth controls and retention policy | Authenticated encryption serde where appropriate | access audit, deletion verification, key lifecycle |

The minimal production experiment is not “send the same prompt twice.” It
should include the real model and prompt template, realistic prefix lengths and
reuse distances, concurrent requests, the intended storage topology, misses and
partial hits, backend slowdown or failure, and p50/p95/p99 latency. Report TTFT,
inter-token latency, throughput, GPU utilization, hit tokens rather than only
request hits, bytes moved, and answer quality for lossy or non-prefix methods.

LMCache is valuable when redundant prefill or fragmented KV ownership is a
material cost. It is not automatically useful for mostly unique short prompts,
nor does persistence remove the need for cache-aware scheduling. The correct
architecture follows from the measured reuse pattern: compute once only when
the state can be found, transferred, interpreted, and protected more cheaply
than it can be recomputed.

## Primary sources

- [LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference, 8 October 2025](https://arxiv.org/abs/2510.09665)
- [LMCache multiprocess documentation, accessed 23 July 2026](https://docs.lmcache.ai/mp/index.html)
- [MP architecture and protocol](https://docs.lmcache.ai/mp/architecture.html)
- [Persistent L2 storage and adapter reference](https://docs.lmcache.ai/mp/l2_storage.html)
- [MP observability](https://docs.lmcache.ai/mp/observability.html)
- [Multi-server coordinator](https://docs.lmcache.ai/mp/coordinator.html)
- [KV cache serialization and compression](https://docs.lmcache.ai/mp/serde.html)
- [CacheGen, 11 October 2023; SIGCOMM 2024](https://arxiv.org/abs/2310.07240)
- [CacheBlend, 26 May 2024; EuroSys 2025](https://arxiv.org/abs/2405.16444)
- [LMCache source repository, Apache-2.0](https://github.com/LMCache/LMCache)
