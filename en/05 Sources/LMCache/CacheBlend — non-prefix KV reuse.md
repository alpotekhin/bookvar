---
title: "CacheBlend — non-prefix KV reuse"
type: source-note
status: active
locale: en
translation_of: "05 Источники/LMCache/CacheBlend — non-prefix KV reuse.md"
last_verified: 2026-07-23
primary_sources:
  - https://blog.lmcache.ai/en/2026/04/01/accelerating-openclaw-agents-with-cacheblend/
  - https://arxiv.org/abs/2405.16444
---

# CacheBlend — non-prefix KV reuse

Prefix caching is exact and inexpensive to reason about: if two requests share
the same tokens from position zero, their cached states for that prefix are
reusable. Agent and RAG prompts often contain the same documents at different
positions, however. New conversation turns, tool results, and reordered
retrieval chunks break the common prefix even when most of the text is
unchanged.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/lmcache-blog-2026/cacheblend-agent-context.jpg]]

*An agent prompt contains a stable conversation prefix and retrieved material
whose position changes between turns. Original LMCache image from
[Accelerating OpenClaw Agents with CacheBlend](https://blog.lmcache.ai/en/2026/04/01/accelerating-openclaw-agents-with-cacheblend/).*

## Why cached chunks cannot simply be concatenated

Suppose the first request uses chunks $A,B,C$ and the second uses $A,C,B$.
Each token's hidden state, key, and value were produced after attending to its
original causal prefix. The cached representation of $C$ after $A,B$ is not the
representation that would have been produced after $A$. Moving the bytes to a
new position preserves the old dependency, not the new one.

Full reuse is therefore fast but inconsistent; full prefill is exact but throws
away almost all cached work. CacheBlend chooses an intermediate point. It loads
cached chunks, identifies tokens with high KV deviation under the new context,
and selectively recomputes a small fraction so cross-chunk dependencies can be
recovered. The paper calls them high-KV-deviation tokens.

The method introduces a quality–latency control absent from exact prefix
caching. If $r$ is the recomputed fraction, then the system trades approximately

$$
C_{request}(r)=C_{load}+rC_{prefill}+C_{blend}
$$

against the error produced by retaining stale contextual states. The best $r$
depends on the task, chunk boundaries, ordering change, and model.

## The LMCache OpenClaw example

The blog constructs two-turn prompts with a system prompt, a user query, and a
large supporting document. Placing the changing question before the stable
document intentionally breaks prefix reuse. In the reported example, ordinary
prefix caching reuses 48% of prompt tokens and produces a 4.055-second TTFT on
the second turn. CacheBlend reuses the system prompt and document despite their
separation, reports a 98% hit rate, and a 2.325-second TTFT.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/lmcache-blog-2026/cacheblend-results.png]]

*The article's CacheBlend result for the two-turn demonstration. These numbers
describe that workload and configuration; they are not a universal speedup.*

The comparison is useful because it exposes a real agent-context pattern, but
it should not be generalized without additional measurements. A production
evaluation needs answer-quality checks against full recomputation, TTFT
percentiles, the cost of loading chunks, recompute fraction, chunk-size
distribution, and a baseline that normalizes cache capacity.

## Where it belongs in the architecture

CacheBlend is not a replacement for PagedAttention or the external storage
layer. PagedAttention owns physical pages. LMCache locates and moves reusable
chunks. CacheBlend defines when non-prefix chunks may be reused and which tokens
must be recomputed to repair contextual inconsistency. Scheduler support is
needed to reserve pages and interleave loading with selective prefill.

## Original sources

- Jiayue Chen and LMCache Team, [Accelerating OpenClaw Agents with CacheBlend](https://blog.lmcache.ai/en/2026/04/01/accelerating-openclaw-agents-with-cacheblend/), 2026-04-01.
- Yao et al., [CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion](https://arxiv.org/abs/2405.16444).
