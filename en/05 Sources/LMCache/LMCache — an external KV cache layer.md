---
title: "LMCache — an external KV cache layer"
type: source-note
status: active
locale: en
translation_of: "05 Источники/LMCache/LMCache — an external KV cache layer.md"
last_verified: 2026-07-23
primary_sources:
  - https://arxiv.org/abs/2510.09665
  - https://docs.lmcache.ai/developer_guide/architecture.html
---

# LMCache — an external KV cache layer

PagedAttention makes KV memory inside one engine manageable. Prefix caching
allows that engine to skip repeated prefill for a prefix it still owns.
LMCache starts where those two mechanisms stop: it extracts KV tensors from the
engine, gives them an addressable lifetime beyond one request, and moves them
between GPU memory, host memory, local storage, remote storage, and other model
instances.

The distinction matters. An engine-native cache is normally fate-shared with
the worker process and constrained by its HBM budget. An external KV layer can
retain a context after the request finishes, make it visible to another replica,
or deliver it from a prefill worker to a decode worker. The engine remains the
place where attention runs; LMCache becomes the storage and transport substrate
for the intermediate state attention consumes.

## Two modes over the same data object

In **storage mode**, newly produced KV chunks are asynchronously copied out of
the engine. CPU DRAM is the first capacity tier, usually backed by pinned and
NUMA-aware memory. Colder entries can continue to NVMe or a remote backend.
When a later request matches cached tokens, the selected chunks travel back to
GPU memory and prefill is skipped for those positions.

In **transport mode**, persistence is secondary. A prefill instance produces
KV for a new prompt and transfers it directly to the decode side. This is the
data-plane operation required by prefill/decode disaggregation. NIXL, RDMA,
NVLink, or TCP may implement the channel, but all variants must carry layout
metadata as well as bytes: model revision, layer, dtype, block mapping, and
parallel rank determine whether the destination can consume the cache.

## Index, connector, worker, controller

The connector is the boundary with vLLM or SGLang. At scheduling time it asks
how many input tokens already exist externally and prepares metadata for the
model runner. At execution time it loads matching KV pages before a layer uses
them or stores newly generated pages afterwards.

LMCache does not transfer at the engine's smallest page granularity. The paper
describes grouping small pages into configurable chunks, commonly 256 tokens,
because tens-of-kilobytes copies do not saturate PCIe or network bandwidth.
Larger chunks reduce per-operation overhead but amplify useless transfer on
partial matches. Chunk size is therefore an I/O parameter and a cache-indexing
parameter at once.

The worker owns the data plane: allocation, gathering and scattering paged
blocks, asynchronous copies, storage backends, and peer transfer. The controller
provides a control plane over cache objects: lookup, move, pin, unpin, evict,
compress, and clear. Once these operations exist, a router can prefer the
replica holding the longest useful prefix, and an operator can warm or migrate
state instead of treating every request as independent.

## Compute–I/O overlap

Loading an entire context before launching the model puts storage latency
directly into TTFT. LMCache instead supports layer-wise pipelining: while layer
$l$ computes, the cache required by layer $l+1$ is fetched into a staging
buffer. The buffer only needs to hold a bounded slice of the request, and the
copy stream can overlap the compute stream.

The overlap is effective only when transfer finishes before the next layer
needs its data. Thus a cache hit is not automatically a latency win. Reusing KV
is worthwhile when

$$
T_{lookup}+T_{load}+T_{scatter} < T_{prefill\ avoided}.
$$

This inequality depends on prompt length, cache tier, effective bandwidth,
queueing, model parallel layout, and how much transfer can be hidden. Cache-hit
rate alone is therefore an incomplete production metric.

## Original sources

- LMCache team, [Architecture Overview](https://docs.lmcache.ai/developer_guide/architecture.html).
- Cheng et al., [LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference](https://arxiv.org/abs/2510.09665).
- Related textbook chapters: [[55 KV cache batching and PagedAttention|KV cache and PagedAttention]] and [[58a2 Disaggregated prefill and decode serving|disaggregated serving]].
