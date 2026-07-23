---
title: "KV cache as persistent inference state"
type: source-note
status: active
locale: en
translation_of: "05 Источники/LMCache/KV cache as persistent inference state.md"
last_verified: 2026-07-23
primary_sources:
  - https://blog.lmcache.ai/en/2026/04/28/stop-calling-it-kv-cache-its-something-much-bigger/
  - https://arxiv.org/abs/2510.09665
---

# KV cache as persistent inference state

The phrase *KV cache* originally described a request-local optimization. Keys
and values produced by earlier tokens remained in HBM so decode did not repeat
their projections. They disappeared with the request and were cheap to
reconstruct from the prompt. Under those assumptions, the conventional cache
analogy was accurate.

Long-context assistants, RAG, multi-turn conversations, agents, and
prefill/decode disaggregation changed the lifecycle. The same computed context
may now be reused across requests, sessions, replicas, or phases of one request.
Its storage hierarchy can span HBM, host DRAM, NVMe, and a remote service. It
has identifiers, placement, retention, transformation, and access policies.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/lmcache-blog-2026/classic-cache-vs-kv-state.png]]

*The LMCache blog contrasts a conventional ephemeral cache with persistent,
shared, and transformable KV state. Source: Junchen Jiang,
[Stop Calling It KV Cache](https://blog.lmcache.ai/en/2026/04/28/stop-calling-it-kv-cache-its-something-much-bigger/).*

## What the stronger abstraction gets right

Treating KV as a first-class inference object makes several systems questions
visible:

- **identity:** which exact model revision, adapter, tokenizer, positional
  configuration, and token sequence produced the object;
- **layout:** how layers, KV heads, tensor-parallel ranks, pages, dtype, and
  quantization are represented;
- **placement:** which tier or worker currently owns each segment;
- **lifecycle:** when an entry is created, pinned, replicated, migrated,
  compressed, evicted, or invalidated;
- **economics:** whether loading the object is cheaper than recomputing prefill;
- **security:** whether one tenant is allowed to discover or reuse another
  tenant's derived state.

These properties are not captured by a binary hit/miss counter. They require a
namespace and control plane, which is why LMCache exposes lookup, movement,
pinning, cleanup, and compression operations.

## Where the memory analogy can mislead

The blog deliberately argues for “model-native memory,” but KV state is not a
portable semantic summary of a document. It is produced for one model and one
position-dependent context. Reordering chunks changes causal dependencies;
switching the model or parallel layout can make the bytes unusable; a compressed
or edited cache may change outputs. Humans also cannot inspect KV tensors as
reliable facts.

For this reason, the textbook should retain two descriptions at once:

1. mathematically, KV is an exact intermediate tensor required by attention;
2. operationally, persisted KV behaves like a valuable inference-state object.

The second description improves system design, but it does not turn KV into a
general database of meaning.

## A useful cost rule

Persisting state is beneficial when expected avoided prefill exceeds storage,
transfer, and management cost:

$$
p_{reuse}\,C_{prefill}
>
C_{store}+p_{reuse}C_{load}+C_{capacity}+C_{control}.
$$

The reuse probability is workload-specific. Stable system prompts and popular
documents can justify pinning; one-off user content may not. Remote storage can
still win for very long contexts when GPU recomputation is expensive, but it
should be demonstrated by TTFT and goodput under realistic queueing rather than
by cache capacity alone.

## Original source

Junchen Jiang,
[Stop Calling It KV Cache: It's Something Much Bigger](https://blog.lmcache.ai/en/2026/04/28/stop-calling-it-kv-cache-its-something-much-bigger/), 2026-04-28.
