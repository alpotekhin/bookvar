---
title: "KV cache, batching, and PagedAttention"
type: textbook-chapter
status: canonical
last_updated: 2026-10-29
last_verified: 2026-10-25
prerequisites:
  - "[[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55a Физика LLM inference — prefill, decode и roofline]]"
next:
  - "[[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55b Scheduling — continuous batching, chunked prefill и prefix caching]]"
source_unit_id:
  - lecture-10-naive-versus-cached-inference
  - lecture-10-dynamic-workload-failure-mode
  - lecture-10-paged-attention-memory-sequence
primary_sources:
  - https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_10.py
  - https://arxiv.org/abs/2309.06180
  - https://arxiv.org/abs/2407.12391v1
  - https://docs.vllm.ai/en/latest/design/paged_attention/
  - https://huggingface.co/docs/transformers/v4.57.1/cache_explanation
  - https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html
  - https://www.anyscale.com/blog/continuous-batching-llm-inference
locale: en
translation_status: current
canonical_revision: "42438233617da20ba81611806b40ed450beaf6506e90ec181e2d43ac92580832"
translation_reviewed: "2026-10-29"
translation_of: "00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention.md"
---

<a id="kv-cache-пакетирование-и-pagedattention"></a>

# KV cache, batching, and PagedAttention

Suppose a model receives 4,000 tokens and generates one more. To select
the next token, we could submit all 4,001 tokens again. That is correct,
but almost all the work repeats: representations for the first 4,000
positions have already been computed. A KV cache retains the part
needed by future steps, so subsequent calls process only new positions.

During training, a Transformer receives rectangular token batches and
processes all positions in parallel. A server handles requests arriving
at different times, with different inputs and generation lengths.
Model parameters are shared, while history state belongs to each request.
KV caching, continuous batching and PagedAttention address three
successive problems: repeated computation of old positions, waiting
for the longest answer, and reserving a maximum-sized contiguous buffer
for every request. The [[55a Физика LLM inference — prefill, decode и roofline|previous chapter]]
examines computational cost; the [[55b Scheduling — continuous batching, chunked prefill и prefix caching|next chapter]]
examines which requests run in each iteration.

<a id="prefill-и-decode-используют-устройство-по-разному"></a>

## Prefill and decode use the device differently

The server first processes the input prompt in **prefill**, then adds
the answer one token at a time in **decode**.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-review-2026-09/prefill-decode.png]]

*The first call processes Computer science is and selects a. The next
receives a, appends its K/V and selects discipline. Baolin Li et al.,
Figure 2 in [LLM Inference Serving: Survey of Recent Advances and Opportunities](https://arxiv.org/html/2407.12391v1#S2.F2),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

For a prefill of length $S$, activation matrices contain many rows.
Large matrix multiplications often make good use of tensor cores,
so arithmetic can be the limiting resource. During decode there is only
one new row per request. The device rereads layer parameters and
attention history while performing relatively little arithmetic.
With a small batch, memory bandwidth more often limits decode.

The first output token is selected from logits at the final prefill
position. Its own K/V is not yet in the cache: it is computed when
that token enters the next iteration. After a 4,000-token prompt,
the server can therefore send the first answer token while the cache
contains exactly 4,000 positions, not 4,001.

<a id="что-именно-сохраняет-kv-cache"></a>

## What the KV cache stores

Let $x_t$ be position $t$'s input representation **at this layer**.
In the first layer it comes from the embedding; in later layers it
comes from the preceding Transformer block. For one attention head,
temporarily omitting positional transformations,

$$
\begin{aligned}
q_t&=x_tW_Q,\\
k_t&=x_tW_K,\\
v_t&=x_tW_V.
\end{aligned}
$$

$$
o_t=\operatorname{softmax}\left(
\frac{q_tK_{1:t}^{\top}}{\sqrt{d_h}}
\right)V_{1:t}.
$$

Earlier $k_i,v_i$ remain valid because the causal mask prevents position
$i$ from attending to positions on its right. Appending a token changes
neither its input nor the keys available to it. Feature normalization
and the MLP also act independently at each position. The old position's
first-layer result therefore remains unchanged; the same argument
applies successively to every later layer. Each layer has **its own**
K/V cache.

A new position needs its $q_t,k_t,v_t$. The key and value are appended;
the query is compared with all available keys, and the resulting weights
mix the values. Old queries are unnecessary: they produced old outputs
and do not enter the expression for the new $o_t$. Old MLP computations
and other operations on processed positions need not repeat either.
Nor must we retain all intermediate activations as for backpropagation.

This argument assumes unchanged parameters, causal attention and
consistent positional encoding. Dropout is disabled during inference.
Editing an early token, switching a LoRA adapter or changing the
positional transformation of stored keys can invalidate the cache.
In a bidirectional encoder, a new token may change earlier
representations, so the same argument does not apply. RoPE implementations
usually store already-rotated keys: a new position needs its actual
position number, not zero at the start of each input chunk.
The layer implementation determines the exact representation.

Caching removes repeated computation but not history reads.
The new $q_t$ still compares with $K_{1:t}$, and the result depends
on $V_{1:t}$. Full attention's work per decode step therefore grows
with context length. PagedAttention changes where these data live,
not the attention rule. See also
[Transformers 4.57.1 cache explanations](https://huggingface.co/docs/transformers/v4.57.1/cache_explanation)
for the mechanism and tensor shapes.

<a id="проверка-целый-префикс-и-обработка-по-частям"></a>

### Check a complete prefix against chunked processing

With $P$ cached positions and $C$ new tokens, Q has $C$ rows
while K/V have $P+C$. New-chunk row $i$ can attend to key positions
$j\le P+i$. For $P=3,C=2$, the mask is

$$
\begin{pmatrix}
1&1&1&1&0\\
1&1&1&1&1
\end{pmatrix}.
$$

The example has one layer and one head. It processes the sequence once
as a whole and again in chunks of three, two and one tokens.
The results should agree within floating-point tolerance.

<!-- example: cached-attention-equivalence -->
```python
import torch

torch.manual_seed(7)
dtype = torch.float64
x = torch.randn(6, 8, dtype=dtype)
wq, wk, wv = [torch.randn(8, 4, dtype=dtype) for _ in range(3)]

def attention_chunk(x_new, past=None):
    q, k, v = x_new @ wq, x_new @ wk, x_new @ wv
    old_length = 0 if past is None else past[0].shape[0]
    if past is not None:
        k = torch.cat((past[0], k), dim=0)
        v = torch.cat((past[1], v), dim=0)
    q_positions = old_length + torch.arange(q.shape[0])
    k_positions = torch.arange(k.shape[0])
    allowed = k_positions[None, :] <= q_positions[:, None]
    scores = q @ k.T / q.shape[-1] ** 0.5
    weights = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
    return weights @ v, (k, v)

full, _ = attention_chunk(x)
past, pieces, start = None, [], 0
for length in (3, 2, 1):
    out, past = attention_chunk(x[start:start + length], past)
    pieces.append(out)
    start += length
torch.testing.assert_close(torch.cat(pieces), full, rtol=1e-12, atol=1e-12)
assert past[0].shape == (6, 4)
```


Here `torch.cat` keeps the example short, but allocates a new tensor
and copies the old cache on each call. This checks mathematics, not
an efficient allocator. A serving engine writes new values into
allocated slots. This example also omits multiple layers, GQA, RoPE,
padding and the output projection; a complete model needs separate
checks for them.

A square lower-triangular mask is correct for prefill from scratch.
Applying the same diagonal to a $C\times(P+C)$ matrix would remove
access to past positions. For example, PyTorch SDPA's
`is_causal=True` uses upper-left alignment. With Q length one
and K length six, that does not mean “the latest query sees all six
keys.” Supply an explicit mask as above. For one new token, no padding,
and a cache ending at that token, causality is already satisfied and
no mask is needed. The
[PyTorch SDPA tutorial](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)
illustrates the two alignments.

<a id="кеш-в-многослойной-модели"></a>

### Caching in a multilayer model

Every decoder layer updates its cache, while the caller supplies only
unprocessed token IDs. The `attention_mask` covers both old and
new positions. With three cached tokens and two new tokens, the input
length is two and the mask length is five. Supplying all five IDs
with the old cache would duplicate the first three positions.

The next example continues the same Python session. It constructs
a randomly initialized Llama without downloading weights: two blocks,
four query heads, two KV heads and RoPE. It compares every logit and
the stored K/V, not merely the chosen next-token ID. It uses
Transformers 4.57.1 on CPU in float32.

<!-- example: cached-llama-equivalence -->
```python
from transformers import LlamaConfig, LlamaForCausalLM, DynamicCache

torch.manual_seed(19)
torch.set_num_threads(1)
lm_config = LlamaConfig(
    vocab_size=16, hidden_size=16, intermediate_size=32,
    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    max_position_embeddings=32,
    bos_token_id=1, eos_token_id=None, pad_token_id=0,
)
lm_config._attn_implementation = "eager"
lm = LlamaForCausalLM(lm_config).float().eval()
token_ids = torch.tensor([[1, 4, 5, 6, 7, 8]])

with torch.inference_mode():
    full_output = lm(token_ids, use_cache=True)
    layer_cache = DynamicCache(config=lm.config)
    cached_logits, start = [], 0
    for length in (3, 2, 1):
        end = start + length
        chunk_output = lm(
            token_ids[:, start:end],
            attention_mask=torch.ones((1, end), dtype=torch.long),
            cache_position=torch.arange(start, end),
            past_key_values=layer_cache, use_cache=True,
        )
        assert layer_cache.get_seq_length() == end
        cached_logits.append(chunk_output.logits)
        start = end

torch.testing.assert_close(
    torch.cat(cached_logits, dim=1), full_output.logits,
    rtol=2e-6, atol=1e-8,
)
for cached, reference in zip(
    layer_cache.layers, full_output.past_key_values.layers
):
    assert cached.keys.shape == cached.values.shape == (1, 2, 6, 4)
    torch.testing.assert_close(cached.keys, reference.keys)
    torch.testing.assert_close(cached.values, reference.values)
```


Shape `(1,2,6,4)` gives batch size, KV-head count, position count
and head dimension. Four query heads do not turn this into a four-head
KV cache: reuse occurs inside attention.
[LlamaAttention](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/models/llama/modeling_llama.py)
applies RoPE to keys before storing them, so checking K/V equality
also checks positional transformations in that implementation.

`DynamicCache` is mutated by the model call. Two variables referring
to the same object do not create independent generation branches.
The complete and chunked calls above use different objects.
`DynamicLayer` also extends tensors through `torch.cat`: it
is an ordinary contiguous cache, not PagedAttention. Shapes and update
rules are described in
[Cache storage implementation](https://huggingface.co/docs/transformers/v4.57.1/cache_explanation#cache-storage-implementation).

<a id="выбранный-токен-ещё-не-находится-в-кеше"></a>

### A selected token is not yet cached

A generation loop first runs forward, then selects the next ID from
its logits. That ID becomes input only to the next forward call.
For a three-token prompt, four iterations have these states:

| Call | New input positions | Cached positions after forward | Total IDs after selection |
|---|---:|---:|---:|
| Prefill | 3 | 3 | 4 |
| Decode 1 | 1 | 4 | 5 |
| Decode 2 | 1 | 5 | 6 |
| Decode 3 | 1 | 6 | 7 |

Check that order by comparing logits with a complete recomputation
of the current prefix at every step. EOS is disabled; exactly four
selections are made.

<!-- example: cached-greedy-loop -->
```python
sequence = token_ids[:, :3].clone()
to_process = sequence
generation_cache = DynamicCache(config=lm.config)

with torch.inference_mode():
    for _ in range(4):
        uncached = lm(sequence, use_cache=False).logits[:, -1:, :]
        cache_start = generation_cache.get_seq_length()
        cached = lm(
            to_process, use_cache=True,
            attention_mask=torch.ones_like(sequence),
            cache_position=torch.arange(cache_start, sequence.shape[1]),
            past_key_values=generation_cache,
        ).logits[:, -1:, :]
        torch.testing.assert_close(cached, uncached, rtol=2e-6, atol=1e-8)
        next_id = cached.argmax(-1)
        sequence = torch.cat((sequence, next_id), dim=1)
        to_process = next_id

assert sequence.shape[1] == 7
assert generation_cache.get_seq_length() == 6
```


To continue this answer, pass the last selected ID. Passing the whole
answer again would duplicate the first six cached positions.
A further forward solely to process EOS is normally unnecessary,
because selecting EOS terminates generation.

<a id="padding-cache_position-и-position_ids"></a>

### Padding, cache_position and position_ids

With left padding, a token's logical position can differ from its cache
column index. Consider requests of length three and five, padded to
five columns:

| Quantity | Short request | Long request |
|---|---|---|
| input_ids | 0, 0, 1, 4, 5 | 1, 6, 7, 8, 9 |
| attention_mask | 0, 0, 1, 1, 1 | 1, 1, 1, 1, 1 |
| RoPE position_ids | 0, 0, 0, 1, 2 | 0, 1, 2, 3, 4 |
| Next position_id | 3 | 5 |
| Next cache_position | 5 | 5 |

Padding is masked, and its `position_ids` are set to zero here.
The rectangular `DynamicCache` already has five columns in each row,
although the shorter request has only three useful tokens.
`cache_position` identifies processing columns; `position_ids`
specifies RoPE positions. Without explicit `position_ids`, this
Llama version uses `cache_position` for both. If the prefix used
unpadded numbering, later calls must continue that numbering.
`generate` normally prepares the appropriate values within its loop.

Process the five-column prefix once, then add one token per row.
Compare each result with a separate unpadded call:

<!-- example: cached-left-padding -->
```python
padded_ids = torch.tensor([[0, 0, 1, 4, 5], [1, 6, 7, 8, 9]])
padding_mask = (padded_ids != 0).long()
positions = (padding_mask.cumsum(-1) - 1).clamp(min=0)
next_ids = torch.tensor([[10], [11]])
padding_cache = DynamicCache(config=lm.config)

with torch.inference_mode():
    lm(
        padded_ids, attention_mask=padding_mask,
        position_ids=positions, cache_position=torch.arange(5),
        past_key_values=padding_cache, use_cache=True,
    )
    next_mask = torch.cat((padding_mask, torch.ones((2, 1), dtype=torch.long)), 1)
    padded_next = lm(
        next_ids, attention_mask=next_mask,
        position_ids=padding_mask.sum(-1, keepdim=True),
        cache_position=torch.tensor([5]),
        past_key_values=padding_cache, use_cache=True,
    ).logits[:, -1, :]
    for row in range(2):
        unpadded = padded_ids[row, padding_mask[row].bool()][None, :]
        unpadded = torch.cat((unpadded, next_ids[row:row + 1]), 1)
        separate = lm(unpadded, use_cache=False).logits[:, -1, :]
        torch.testing.assert_close(
            padded_next[row:row + 1], separate, rtol=2e-6, atol=1e-8,
        )
assert padding_cache.get_seq_length() == 6
```


A paged engine can store unpadded requests with individual lengths and
block tables. This example's rectangular indices do not prescribe its
internal layout. They show why useful text length cannot automatically
serve as a write index for every cache implementation.

<a id="сколько-памяти-занимает-история"></a>

## How much memory history consumes

Let $L$ denote layers, $S$ current sequence length, $H_{kv}$ key/value
heads, $d_h$ head dimension, and $b$ bytes per stored number.
One request then requires

$$
M_{KV}=2LSH_{kv}d_hb.
$$

The factor two counts K and V. For 32 layers, eight KV heads,
$d_h=128$, BF16 and 8,192 positions:

$$
\begin{aligned}
M_{KV}&=2\cdot32\cdot8192\cdot8\cdot128\cdot2\\
&=1\ \text{GiB}.
\end{aligned}
$$

This is exactly 1 GiB, or 1,073,741,824 bytes: 128 KiB per token.
Sixty-four independent requests require 64 GiB for useful K/V alone.
Parameters, working buffers, block rounding, metadata and quantization
scales add to that requirement.

For requests of different lengths, add their useful storage rather than
multiply batch size by an arbitrarily chosen length. For $B$ independent
sequences with the same homogeneous layers,

$$
M_{KV}=2LH_{kv}d_hb\sum_{i=1}^{B}S_i.
$$

For example, requests of 1,024 and 2,048 tokens contain 3,072 useful
positions. At the preceding 128 KiB per position, they require 384 MiB.
A rectangular buffer padding both to 2,048 positions would take 512 MiB:
the shorter request's padding accounts for another 128 MiB. A paged
pool instead rounds each length to its block size. With shared prefixes,
the same physical blocks must not be counted again for every reference.

The formula assumes all $L$ layers retain the entire history and have
identical KV-head counts. Ordinary MHA has $H_{kv}=H_q$; GQA lets
several query heads use one KV head, so K/V need not be stored again
for every query head. For heterogeneous layers, sum their costs
separately. Sliding windows bound retained history. MLA may store a
compressed state plus a separate positional component; substituting
query-head count into this formula would be wrong. Across GPUs,
count which layers and KV heads are placed or replicated on each device.

GQA reduces $H_{kv}$, MLA compresses the representation, and cache
quantization reduces $b$. These can support more concurrent sequences,
but change architectural or numerical properties. They are distinct
from memory-placement algorithms.

<a id="почему-статический-пакет-простаивает"></a>

## Why static batches leave capacity idle

In static batching, requests begin together and the next batch waits
for the longest answer. A slot after EOS does no useful work but
cannot receive another request until the group finishes.
The [[55b Scheduling — continuous batching, chunked prefill и prefix caching|scheduling chapter]]
develops the transition to iteration-level scheduling.

**Continuous batching** changes membership between decode iterations.
A finished sequence releases a slot that can receive a new request.
Members occupy different sequence positions, so kernels receive actual
lengths and cache addresses rather than one dense rectangle.

Larger batches amortize parameter reads and increase throughput.
They can also increase scheduling wait and TPOT through neighboring
work. Under latency constraints, **goodput** counts successfully
completed requests per unit time that meet both TTFT and TPOT limits.

<a id="почему-непрерывного-пакетирования-недостаточно"></a>

## Why continuous batching is not enough

Suppose every request receives a contiguous buffer for its maximum
length. Its actual answer is usually shorter, but another request
cannot use the reserved tail. Allocating only the current length and
repeatedly extending the buffer can instead encounter an occupied
neighbor. Enough free bytes in total do not guarantee a sufficiently
large contiguous region.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/paged-attention-fragmentation.png]]

*The paper distinguishes future-token reservation, never-used internal
tails, and external gaps between allocations. Figure 3, Kwon et al.,
[Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), copied from CS336.
The words label K/V slots, not stored text.*

Some reserved slots will be needed later but are currently unavailable
to other requests. More distant slots will never be used because
generation ends earlier. Gaps between allocated buffers may be too
small for a new request. These are separate sources of waste; a single
“allocated GiB” counter does not distinguish them.

KV-cache memory is the dynamic part of server capacity. Fragmentation
can cause a request to be rejected despite enough free bytes in total.

<a id="pagedattention-логическая-последовательность-физические-блоки"></a>

## PagedAttention: logical sequences and physical blocks

PagedAttention uses paged addressing. A sequence's cache is divided
into fixed-token-count **logical blocks**. A block table maps them
to available **physical blocks** in a shared pool. Consecutive logical
blocks need not be physically adjacent.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/paged-attention-blocks.png]]

*The query for forth reads three discontiguous physical blocks.
The logical beginning Four score and seven is in the bottom row,
although it lies physically after years ago our fathers. Figure 5,
[PagedAttention](https://arxiv.org/abs/2309.06180), reused in
[Stanford CS336 Spring 2026](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_10.py),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

As an answer grows, the manager allocates another free block.
Unused internal space remains only in the final block. External
fragmentation between equal-sized blocks disappears because any free
block from that pool can satisfy the next allocation. This concerns
the KV pool, not every allocation in GPU memory. The cost is an
address table and a kernel able to read K/V directly from the blocks;
it need not first copy the whole history into a contiguous array.

The next figure maps logical blocks to physical locations. The table
belongs to request state, but blocks belong to the shared pool.
Several requests can reference one block.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/paged-attention-logical.png]]

*Request A's logical blocks 0, 1 and 2 occupy physical blocks 7, 1
and 3. Request B uses 5 and 2; A and B share no data in this example.
Figure 7, Kwon et al., [PagedAttention](https://arxiv.org/abs/2309.06180),
reproduced in Stanford CS336, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

<a id="как-найти-токен-по-таблице-блоков"></a>

### Find a token through the block table

With block size $B$, zero-based position $t$ has logical block number
$\lfloor t/B\rfloor$ and offset $t\bmod B$. In request A, the table
is `[7,1,3]` and $B=4$. Position 8, the word brought, is in logical
block 2, physical block 3, slot 0.

<!-- example: paged-slot-address -->
```python
block_table = [7, 1, 3]
block_size = 4

def physical_slot(position):
    logical_block, offset = divmod(position, block_size)
    return block_table[logical_block] * block_size + offset

assert [physical_slot(t) for t in range(9)] == [28, 29, 30, 31, 4, 5, 6, 7, 12]
```


An actual tensor address also includes layer, KV-head and vector-component
indices. K and V may have different layouts, but the lookup rule is
the same. A physical slot number is not a RoPE position: moving block
7 to block 4 must not change token meaning or logical positions.
A KV block is also distinct from a CUDA thread block.

The operating-system analogy concerns addressing, not the whole virtual
memory mechanism. The engine manager allocates slots and prepares tables.
GPU kernels read those tables as ordinary data; no operating-system
page fault is needed to reach the next history block. Paging alone
does not imply CPU or disk offload.

<a id="softmax-остаётся-общим-для-всей-истории"></a>

### Softmax still spans the complete history

Splitting K/V into blocks does not create independent attention tasks.
Applying softmax separately in each block and averaging the outputs
would give each block equal probability mass. Suppose the first block
has scores $[0,0]$, and the second $[\ln3,\ln3]$.
Their global masses are $2/(2+6)=1/4$ and $6/(2+6)=3/4$, not
$1/2$ each. If first-block values are 1 and second-block values are 3,
the correct output is $\tfrac14\cdot1+\tfrac34\cdot3=2.5$.
Averaging separately normalized outputs would give $(1+3)/2=2$.

Partial results can be combined exactly. For block $b$, let $m_b$ be
its maximum score, $z_b=\sum_{j\in b}\exp(s_j-m_b)$ and
$u_b=\sum_{j\in b}\exp(s_j-m_b)v_j$. With $m=\max_bm_b$,

$$
o=\frac{\sum_b e^{m_b-m}u_b}{\sum_b e^{m_b-m}z_b}.
$$

Subtracting the maximum stabilizes computation. Unfilled slots in the
last block contribute to neither numerator nor denominator, so the
kernel needs the actual length. This is a mathematical combination
rule; a kernel may distribute work among threads differently.
Global normalization preserves ordinary attention, rather than
approximating it. Different floating-point operation orders can still
change rounding slightly.

The [original vLLM kernel description](https://docs.vllm.ai/en/latest/design/paged_attention/)
explains block reads and softmax reductions. The documentation labels
it historical: it explains the mechanism, not all modern attention
backends. Blocked KV storage works with different kernels, including
FlashAttention variants; these technologies are compatible.

<a id="размер-блока-полезная-память-и-резерв"></a>

### Block size: useful memory and reservation

Three independent requests of lengths 17, 31 and 40 contain 88 useful
token slots. In the earlier model, a slot across all 32 layers takes
$2\cdot32\cdot8\cdot128\cdot2=131072$ bytes, or 128 KiB.

| Block size | Blocks per request | Allocated slots | Unused tails | Useful / reserved KV |
|---|---|---:|---:|---:|
| 16 tokens | 2, 2, 3 | 112 | 24 | 11 / 14 MiB |
| 32 tokens | 1, 1, 2 | 128 | 40 | 11 / 16 MiB |

The smaller blocks save 2 MiB here but need more table entries.
With four bytes per entry and one table per logical sequence, indices
alone take 28 versus 16 bytes. Actual tables, alignment, reference
counts and allocator metadata add overhead; these numbers do not
describe the manager's full memory cost.

For $N$ independent sequences, rounding in this pool leaves fewer than
$NB$ unused slots: at most $B-1$ per sequence. This bound covers
current K/V without future-block preallocation. Speculative candidates,
different KV-layer groups and alignment can add other costs.
Reducing $B$ to one is not always beneficial: tables and lookups grow,
and kernel efficiency changes. Choose among block sizes supported by
the backend, not solely by unused-slot percentage.

Fork a 20-token prefix with 16-token blocks. The first complete block
stays shared; writing a divergent branch requires one copy of the
partial tail, reserving 16 additional slots or 2 MiB. With 32-token
blocks, copy-on-write reserves 32 slots or 4 MiB. Sharing postpones
copies but cannot let divergent branches modify one common tail.

<a id="совместное-использование-префикса"></a>

## Sharing a prefix

Paged storage allows several continuations to reference the same physical
prefix blocks. This helps multiple samples, beam search and recurring
system instructions. Read-only blocks need no copy. Once continuations
diverge, new tokens use separate blocks; the original paper handles a
partially shared final block through **copy-on-write**.
That is a branching mechanism, not a required property of every modern
prefix-cache implementation.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/inference-evaluation/paged-attention-sharing.png]]

*Two requests share translation examples at the beginning but have
different inputs and continuations. They can reference the same physical
KV blocks for the shared portion. Figure 10 from PagedAttention, copied
from Stanford CS336 at
[`8b59b50`](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_10.py#L594-L600),
based on Kwon et al., [PagedAttention](https://arxiv.org/abs/2309.06180),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

When two continuations share the last block, the first divergent write
must not change the other branch's data. The manager increments the
shared block's reference count and creates a physical copy of that
block on writing. In the diagram, fathers and mothers first share
years ago our; one branch's table then points to a new block.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/inference-evaluation/paged-attention-copy-on-write.png]]

*Complete block 7 is not copied. Partial block 1 splits: branch A1
gets a copy in block 3, while A2 keeps block 1. Figure 8 from
PagedAttention, copied from Stanford CS336 at
[`8b59b50`](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_10.py#L599-L600);
Kwon et al., [PagedAttention](https://arxiv.org/abs/2309.06180),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

Reference counts also govern completion. When A1 finishes, its private
block can return to the pool, but A2 still needs shared block 7.
Releasing the final reference makes a block reusable, without necessarily
erasing its contents immediately. With prefix caching, it may remain
an evictable cached entry until reallocated. Memory cannot be overwritten
while a running GPU kernel still reads it; the manager must coordinate
release with operation completion.

Prefix caching asks whether computed history may be reused for an
identical prefix. PagedAttention specifies placement and addressing.
Paging makes sharing cheaper, but does not discover matching inputs
or eliminate checks of exact tokens and model parameters. The same
word following different prefixes generally has different K/V;
matching one block's text is insufficient.

[vLLM V1 Automatic Prefix Caching](https://docs.vllm.ai/en/latest/design/prefix_caching/)
reuses only **complete** blocks. The partial shared tail in the
copy-on-write example is not automatically an inter-request cache hit.
The [[55b Scheduling — continuous batching, chunked prefill и prefix caching|scheduling chapter]]
explains prefix identity and what remains to compute after a hit.

<a id="планировщик-связывает-вычисления-и-память"></a>

## The scheduler couples computation and memory

Each iteration selects requests and the number of new positions to
process. Those positions need KV slots. If the shared cache already
contains part of an input, its blocks can be attached to the request
and only the remaining suffix computed. If free blocks are insufficient,
work must wait, another request's state must be released, or a different
placement strategy used. Batch selection therefore includes memory
accounting.

A long prefill can delay an active conversation's next token.
**Chunked prefill** divides the input, allowing batch membership to
be reconsidered after each chunk. Larger chunks often favor matrix
multiplication; smaller chunks offer more frequent service to decoding
requests. Token count alone does not determine iteration duration:
attention context length also matters.

The cache continues growing after the input is processed. Reserving
space for the maximum answer is not mandatory, but denser admission
raises the risk of later exhaustion. Preemption and recomputation can
erase the larger batch's benefit. The
[[55b Scheduling — continuous batching, chunked prefill и prefix caching|request scheduling chapter]]
works through this trade-off.

<a id="что-измерять"></a>

## What to measure

Distinguish memory shortage, queueing and slow execution with several
measurements:

| Quantity | Mechanism exposed |
|---|---|
| p50/p95 TTFT | Queueing, prefill-batch size, chunked prefill |
| ITL distribution and request-level TPOT separately | Decode batches and scheduling pauses; mean TPOT hides individual stalls |
| Input/output tokens/s | Phase-specific throughput |
| Useful/reserved KV bytes | Occupancy and fragmentation |
| Block utilization | Waste in final blocks |
| Preemption/recompute rate | Overly aggressive admission |
| Reused input tokens / total input tokens | Saved prefill volume; requests with any hit form a different metric |

Use lengths and arrival rates representative of the product. One batch
of identical requests does not test the arrival of new work and completion
of old work. A request stream must change batch membership, memory
allocation and waiting time.

<a id="практика-и-первоисточники"></a>

## Practice and primary sources

- [[05 Источники/Courses/Harvard ML Systems/tinytorch/18_memoization|TinyTorch 18: KVCache implementation and memory accounting]].
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week08_inference_software/lecture.pdf|Efficient DL Systems: inference engine architecture]].
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week08_inference_software/seminar.ipynb|Efficient DL Systems: inference software seminar]].
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week08_inference_software/homework/homework_week8.ipynb|Efficient DL Systems: inference engine assignment]].
- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol2/inference|Harvard CS249r: Inference]].
- [[02 Areas/ML & DL/06 Практика/14 Собрать минимальный inference engine|Build a minimal inference engine]].
- Stanford CS336 Spring 2026, [Lecture 10: Inference](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_10.py): prefill/decode, batching and the five-frame paging explanation; pinned commit `8b59b507`.
- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180): the original algorithm and fragmentation analysis.
- vLLM, [PagedAttention design](https://docs.vllm.ai/en/latest/design/paged_attention/): historical original-kernel explanation; [Automatic Prefix Caching](https://docs.vllm.ai/en/latest/design/prefix_caching/): V1's separate full-prefix-block cache.
- Anyscale, [Continuous Batching for LLM Inference](https://www.anyscale.com/blog/continuous-batching-llm-inference): static and continuous batching illustrated.

Previous: [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55a Физика LLM inference — prefill, decode и roofline|The physics of LLM inference: prefill, decode and roofline]].
Next: [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55b Scheduling — continuous batching, chunked prefill и prefix caching|Request scheduling: batching, chunked prefill and prefix caching]].
