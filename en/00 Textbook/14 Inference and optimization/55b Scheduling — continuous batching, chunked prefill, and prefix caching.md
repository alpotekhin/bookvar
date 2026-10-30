---
title: "Scheduling: continuous batching, chunked prefill, and prefix caching"
type: textbook-chapter
status: canonical
last_updated: 2026-10-30
last_verified: 2026-10-25
prerequisites:
  - "[[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention]]"
next:
  - "[[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55c Serving engines — vLLM, SGLang, TensorRT-LLM и FlashInfer]]"
source_unit_id:
  - lecture-10-section-381-handling-dynamic-workloads
  - lecture-10-dynamic-workload-failure-mode
  - lecture-10-figure-step-392-rendering-1
primary_sources:
  - https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_10.py
  - https://www.usenix.org/system/files/osdi22-yu.pdf
  - https://arxiv.org/abs/2308.16369
  - https://arxiv.org/abs/2403.02310
  - https://arxiv.org/abs/2309.06180
  - https://arxiv.org/abs/2312.07104
  - https://github.com/vllm-project/vllm/blob/23fbc2ec2f7219888bfc76cf987381f501151971/vllm/v1/core/sched/scheduler.py
  - https://github.com/vllm-project/vllm/blob/23fbc2ec2f7219888bfc76cf987381f501151971/docs/design/prefix_caching.md
  - https://github.com/sgl-project/sglang/blob/bab9c81fea62c2d20b646736c918c88fea54b7a2/python/sglang/srt/managers/schedule_policy.py
  - https://www.aleksagordic.com/blog/vllm
  - https://huggingface.co/blog/continuous_batching
locale: en
translation_status: current
canonical_revision: "53a8c18595f4e2f2205a0d43c009e443e948720dc199e50f966449a838e8e0ec"
translation_reviewed: "2026-10-30"
translation_of: "00 Учебник/14 Inference и оптимизация/55b Scheduling — continuous batching, chunked prefill и prefix caching.md"
---

<a id="scheduling-continuous-batching-chunked-prefill-и-prefix-caching"></a>

# Scheduling: continuous batching, chunked prefill, and prefix caching

Suppose a server is answering four users. One answer is almost finished,
two continue, and the fourth may require hundreds more tokens. A new
request arrives with a long document. Waiting for all four answers to
finish leaves the newcomer looking at an empty screen. Processing the
entire document immediately can pause answers already in progress.

The scheduler chooses the next batch and the number of new positions
to process for each request. It considers the waiting queue, active
sequence states and available KV blocks. The cache manager checks
whether new keys and values fit and allocates their blocks.

The [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55a Физика LLM inference — prefill, decode и roofline|inference physics chapter]]
derives prefill/decode costs. A server must distribute those passes
among requests arriving and finishing at different times. Iteration-level
scheduling admits requests before a whole batch finishes. Chunked prefill
limits new prompt work in one step; prefix caching removes repeated
computation of shared context.

<a id="почему-в-статическом-пакете-простаивают-места"></a>

<a id="почему-статический-пакет-теряет-вместимость"></a>

## Why static batches leave slots idle

Four simultaneous requests A, B, C and D need 2, 4, 6 and 8 further
generation steps. With fixed membership, the group finishes after
eight steps and generates $2+4+6+8=20$ tokens. Four slots across
eight steps allow $4\cdot8=32$ tokens, giving $20/32=62.5\%$
slot occupancy in this model.

That ratio is not GPU utilization. An implementation may remove finished
rows and compute smaller matrices. New requests still wait if admission
cannot resume until the whole group finishes. Fixed tensor shapes can
add unnecessary computation on finished rows as well.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/anyscale-static-batching.png]]

*Empty positions remain to the right of END after shorter sequences
finish. This is a fixed-group diagram, not a GPU-kernel trace.
Anyscale, reproduced in
[Stanford CS336 Lecture 10](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_10.py#L553);
[original article](https://www.anyscale.com/blog/continuous-batching-llm-inference).*

Add waiting requests E and F, each needing two steps. With static
batching they start after step eight and finish at ten. If membership
can change after every iteration, E takes A's slot from step three
and F takes an available slot from step five. E finishes at four,
F at six, and the last request in the queue at eight.

The following model assumes every prompt is already processed.
Each active request receives one token per iteration. It models
service order only, excluding prefill time, memory and pass duration.

```python
from collections import deque

def finish_steps(lengths, slots, continuous):
    if slots < 1 or any(n < 1 for n in lengths.values()):
        raise ValueError("positive lengths and slot count required")
    waiting = deque(lengths.items())
    active, finished = {}, {}
    step = 0
    while waiting or active:
        if continuous or not active:
            while waiting and len(active) < slots:
                name, remaining = waiting.popleft()
                active[name] = remaining
        step += 1
        for name in list(active):
            active[name] -= 1
            if active[name] == 0:
                finished[name] = step
                del active[name]
    return finished

lengths = dict(A=2, B=4, C=6, D=8, E=2, F=2)
static = finish_steps(lengths, 4, continuous=False)
dynamic = finish_steps(lengths, 4, continuous=True)
assert (static["E"], static["F"], max(static.values())) == (10, 10, 10)
assert (dynamic["E"], dynamic["F"], max(dynamic.values())) == (4, 6, 8)
```


All six requests finish in eight rather than ten iterations.
Converting that difference to elapsed time requires each pass's
duration, which depends on batch size. New requests also require the
omitted prefill work. The ratio $10/8$ is therefore not a measured
server speedup.

<a id="orca-планирование-на-уровне-итерации"></a>

## Orca: iteration-level scheduling

In [Orca](https://www.usenix.org/system/files/osdi22-yu.pdf), the scheduler
submits a batch for one iteration. Once results return, it removes
finished requests, admits new ones and forms another batch.
This is commonly called continuous or in-flight batching.

Between passes, each request retains processed-token state, KV cache,
sampling settings and stopping conditions. Requests share a batch only
for the current computation. Row order may change, but the mapping
from rows to requests and cache addresses must remain correct.

<a id="какие-операции-можно-объединить"></a>

### Which operations can be batched

Suppose A has K/V for 100 positions and B for 200. Each has selected
a token and must process it to predict the next one. Request P is
assigned four prefill positions. Linear layers receive:

| Request | Cached positions | New rows in this pass |
|---|---:|---:|
| A, decode | 100 | 1 |
| B, decode | 200 | 1 |
| P, initial prefill | 0 | 4 |

The same weight matrix $W$ projects each new row. Concatenate six
rows, multiply by $W$, then partition the result among A, B and P.
Their cached-history lengths do not change this multiplication's
shape: past K/V have already been computed and will instead be read
during attention.

Let $T_i$ be new positions assigned to request $i$ and $d$ the hidden
width. Normalization, linear projections and MLPs act independently
on positions, so their inputs can be packed as

$$
X_{\mathrm{packed}}\in\mathbb{R}^{(\sum_iT_i)\times d}.
$$

Attention must restrict each new position to eligible positions in
its own request. Concatenating histories under one global causal
mask would leak information across users. Sequence boundaries,
lengths, new-token positions and KV-block addresses are required.

Orca combines iteration-level scheduling with selective batching:
operations run together when doing so preserves their results.
Request isolation does not require a separate GPU-kernel launch per
sequence. A kernel can handle several lengths in one launch when
given their boundaries and cache layout. See
[CS336 Lecture 10, lines 553–575](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_10.py#L553)
for tensor shapes and selective batching.

Packing amortizes weight reads over more new positions, but each
request still has its own attention history. A larger batch increases
KV traffic and memory requirements. Both assigned work and storage
for its new K/V must fit.

<a id="новый-prefill-рядом-с-продолжающимся-decode"></a>

### A new prefill beside ongoing decode

Replacing a completed request also changes input shapes. In the next
figure, one request finishes while another continues. The replacement
has six tokens, `[<bos>, The, larg, est, ani, mal]`; the ongoing
request needs to process only `be`.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/scheduling-2026-10-25/hf-dynamic-batching.png]]

*In the rectangular replacement batch, six new positions in one row
force five pad positions in the other. Blue tokens are already cached;
their appearance does not mean they are recomputed. Rémi Ouazan Reboul,
Arthur Zucker and Luc Georges,
[Continuous batching from first principles](https://huggingface.co/blog/continuous_batching#continuous-batching).
[Unchanged original](https://huggingface.co/datasets/huggingface/documentation-images/blob/c8c3cf72b3b724ce734259e4a086d30618a9198b/blog/continuous_batching/dynamic_batching.png),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/);
licence recorded in the [image dataset card](https://huggingface.co/datasets/huggingface/documentation-images/blob/c8c3cf72b3b724ce734259e4a086d30618a9198b/README.md).*

A rectangular `[2,6,d]` input has twelve slots but only seven
useful ones. Packing gives linear layers six new prompt rows and
one continuing-answer row. The eight blue history tokens are already
computed. Attention needs their K/V, but they are not new rows.

Key count is also not seven. Appending `be` gives the ongoing
request nine keys; the new request has six. This pass can be described
by seven query rows and fifteen key columns. The decode row sees
nine positions of its own history. The six prefill rows see one,
two, through six positions of the new input. All cross-request
connections are forbidden.

This matrix describes permitted reads; it does not require allocating
a dense `7 × 15` tensor. Variable-length and paged kernels can
receive lengths and addresses separately. Packing removes padding
from new inputs, while attention boundaries preserve independence.

<a id="что-выбирает-планировщик-на-каждом-шаге"></a>

## What the scheduler selects at each step

Two counters describe an ordinary autoregressive request:

- $c_i$: positions whose K/V have been computed;
- $n_i$: known sequence tokens at the start of the pass.

During prefill, $n_i$ is prompt length while $c_i$ grows as the prompt
is processed. After complete prefill, the first selected answer token
is known but has no K/V yet. The next pass processes it and predicts
a continuation. Ordinary decode therefore has $n_i-c_i=1$.
[[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/58 Спекулятивное декодирование|Speculative decoding]]
can have several pending positions.

The scheduler assigns $q_i$ new positions. In a simple synchronous
model, $0\le q_i\le n_i-c_i$, with token budget $B$:

$$
\sum_iq_i\le B.
$$

Partial-prefill logits cannot select the answer's first token: the
remaining input has not been incorporated. Selection waits until the
entire prompt has been processed. A request can therefore participate
in several continuously batched iterations before returning anything.

<a id="токенов-может-хватить-а-памяти--нет"></a>

### Token capacity and memory capacity are separate

Let a KV block hold $b$ tokens. If request $i$ has $c_i$ computed
positions and exactly $\lceil c_i/b\rceil$ allocated blocks, the next
$q_i$ positions need

$$
\Delta m_i=
\left\lceil\frac{c_i+q_i}{b}\right\rceil-
\left\lceil\frac{c_i}{b}\right\rceil.
$$

additional blocks. This assumes full attention without shared blocks,
spare reservation or preemption. A real manager also accounts for
sharing, copy-on-write, layer groups and incomplete computations;
see the [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention|KV-cache chapter]].

Take $b=8$ and $B=12$. Decode requests have $c_A=7$ and $c_B=8$,
and prompt P has 18 tokens. With two free blocks, consider A, B,
then P:

| Assignment | Memory requirement | Remaining token budget | Free blocks |
|---|---|---:|---:|
| A: one decode position | One slot remains in A's first block | 11 | 2 |
| B: one decode position | B's first block is full; allocate another | 10 | 1 |
| P: eight prefill positions | The final free block holds eight | 2 | 0 |

Another P position would require a second block. Memory is exhausted,
so the pass contains ten new positions despite a budget of twelve.

The following code checks both budgets. Input triples are
`(name, computed positions, desired new positions)`; an external
policy has already chosen their order. This model neither preempts
requests nor changes their state while constructing the plan.

```python
def ceil_div(n, d):
    return (n + d - 1) // d

def plan_step(requests, token_budget, free_blocks, block_size):
    if token_budget < 0 or free_blocks < 0 or block_size < 1:
        raise ValueError("invalid resource budget")
    plan = {}
    for name, computed, wanted in requests:
        if name in plan or computed < 0 or wanted < 0:
            raise ValueError("invalid request")
        allocated = ceil_div(computed, block_size)
        room = (allocated + free_blocks) * block_size - computed
        count = min(wanted, token_budget, room)
        extra = ceil_div(computed + count, block_size) - allocated
        plan[name] = count
        token_budget -= count
        free_blocks -= extra
    return plan, token_budget, free_blocks

requests = [("A", 7, 1), ("B", 8, 1), ("P", 0, 18)]
assert plan_step(requests, 12, 2, 8) == (
    {"A": 1, "B": 1, "P": 8}, 2, 0
)
assert plan_step(requests, 12, 3, 8) == (
    {"A": 1, "B": 1, "P": 10}, 0, 0
)
```


With three free blocks, P processes ten rather than eight tokens.
It still cannot return an answer: eight prompt positions remain.
The next iteration checks both budgets again as decode histories grow.

<a id="план-и-завершённое-вычисление--разные-состояния"></a>

### A plan is not a completed computation

vLLM V1 does not require distinct `RUNNING_PREFILL` and
`RUNNING_DECODE` active-request classes. The
[pinned schedule implementation](https://github.com/vllm-project/vllm/blob/23fbc2ec2f7219888bfc76cf987381f501151971/vllm/v1/core/sched/scheduler.py#L591)
uses known and processed position counts. It considers running
requests first, including incomplete prefills.

During asynchronous execution, a counter may include assigned positions
before computation finishes. Their blocks are allocated, but K/V can
be read only after the executor writes them. The scheduler must
account for completion, cancellation and rejected speculative tokens.
The synchronous example above checks resource assignment, not
dependencies among unfinished operations.

<a id="fairness-preemption-и-перегрузка"></a>

## Fairness, preemption and overload

A memory-feasible batch can still leave some users waiting. Always
prioritizing existing decode can smooth active responses while delaying
newcomers' first token. Giving the entire budget to long new prompts
instead pauses ongoing answers.

FCFS admits requests in arrival order, but a long first prompt still
needs a per-iteration allocation policy. The scheduler can assign
all available budget to it or share that budget with others.

Short-prefill priority can use known input length. Short-answer priority
would have to estimate an unknown output length. Repeatedly choosing
short requests can defer long ones. Aging increases priority with
waiting time; actual waiting guarantees depend on the policy and
load limits.

<a id="что-происходит-при-нехватке-kv-блоков"></a>

### When KV blocks run out

A request that fit initially may no longer fit after hundreds of output
tokens. Preemption suspends active execution and releases its blocks.

The original [PagedAttention](https://arxiv.org/abs/2309.06180) paper
describes two recovery approaches. Swapping transfers the cache to
CPU memory and later back to the GPU. Recomputation releases it and,
on resumption, reprocesses known prompt and generated tokens.
It reconstructs K/V without changing tokens already delivered.

For example, after an eight-token prompt, a user has received three
answer tokens. If the whole cache is lost, processing the known
eleven-token sequence reconstructs the state. Logits at its final
position select the fourth answer token. The first three come from
stored request history; they are not sampled again. Recovery costs
time but does not restart the answer.

Swapping consumes link bandwidth and CPU memory; recomputation consumes
GPU execution time. A particular engine version may use only one.
In the pinned vLLM V1,
[`_preempt_request`](https://github.com/vllm-project/vllm/blob/23fbc2ec2f7219888bfc76cf987381f501151971/vllm/v1/core/sched/scheduler.py#L1571)
frees blocks, resets the processed-position count and returns the
request to the queue. On readmission, some prefix may still be cached
if not overwritten. Surviving state determines recomputation volume.

Frequent preemption repeatedly spends prefill work to free memory
and then rebuild paused histories. Reducing admitted concurrency can
improve throughput by avoiding those recomputations.

<a id="почему-перестановки-очереди-недостаточно"></a>

### Reordering cannot remove overload

When arriving work persistently exceeds service capacity, an unbounded
queue grows. A bounded queue rejects work or pushes waiting back to
the sender. Requests per second alone do not characterize load:
ten short questions and ten long-document requests impose different work.

Bound arrivals, active requests and input/output lengths to control
queue and memory use. Remove cancelled or expired requests so the
GPU stops computing results nobody will consume. The
[[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/58c Queueing и capacity planning|queueing and capacity chapter]]
derives limits from workload distributions.

<a id="почему-длинный-prefill-мешает-decode"></a>

## Why a long prefill disrupts decode

A thousands-token prefill can take much longer than one decode step.
When it joins a batch of generating requests, those clients wait
for the mixed batch's computations needed by their next tokens.
Continuous batching permits admission; it does not eliminate this
post-admission delay.

Chunked prefill splits the unprocessed prompt. Each chunk leaves K/V
in the cache and gives the scheduler another opportunity to redistribute
the next step's work.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-advanced/physics-scheduling/gordic-chunked-prefill.png]]

*Aleksa Gordić's diagram shows the first two eight-token chunks of an
18-token prompt. Four blocks are filled after pass two; the remaining
two tokens require another pass. This shows cache growth, not the
schedule of neighboring requests.
[Inside vLLM, Chunked prefill](https://www.aleksagordic.com/blog/vllm).*

If $S_{\mathrm{new}}$ input tokens remain after cache reuse, and each
pass can process a chunk of $C$, prefill needs
$\lceil S_{\mathrm{new}}/C\rceil$ passes. Fewer available token slots
or blocks can increase that count.

Under full causal attention, each chunk reads the processed prefix
and its own permitted positions. Position numbering does not restart
at zero. Correct positions and masking preserve the mathematical
result, although different matrix shapes and kernels can alter rounding.
The [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention|cache chapter]]
derives and tests the mask.

<a id="чему-помогает-разбиение-расчёт-времени"></a>

### A timing calculation

Four requests are decoding when a twelve-token prompt arrives.
Assume this illustrative mixed-pass duration:

$$
\tau(q,D)=2+q+0.5D\quad\text{ms},
$$

Here $q$ is new prefill positions and $D$ is requests assigned one
decode step each. The constant is fixed overhead. All coefficients
are chosen for the example, not measured on a GPU.

| New prompt processing | Pass composition | Pass duration | New request's first token |
|---|---|---:|---:|
| Entire prompt | 12 prefill + 4 decode | 16 ms | After 16 ms |
| Three chunks | 4 prefill + 4 decode, three times | 8 ms each | After 24 ms |

With chunks, each old request receives tokens at 8, 16 and 24 ms.
Without chunks, its next token waits 16 ms. The new request waits
longer with chunks: two extra passes and their decode work are added.
Reducing pauses for active answers can increase the newcomer's TTFT.

```python
def prefill_timeline(prompt_tokens, chunk_size, decodes):
    if prompt_tokens < 1 or chunk_size < 1 or decodes < 0:
        raise ValueError("invalid workload")
    remaining, elapsed, events = prompt_tokens, 0.0, []
    while remaining:
        chunk = min(chunk_size, remaining)
        elapsed += 2.0 + chunk + 0.5 * decodes
        events.append((chunk, elapsed))
        remaining -= chunk
    return events

assert prefill_timeline(12, 12, 4) == [(12, 16.0)]
assert prefill_timeline(12, 4, 4) == [(4, 8.0), (4, 16.0), (4, 24.0)]
assert prefill_timeline(10, 4, 4) == [(4, 8.0), (4, 16.0), (2, 22.0)]
```


GPU time need not depend linearly on positions. At small batch sizes,
added prefill can occupy otherwise unused compute capacity. With larger
batches or long context, another operation may become the bottleneck.
Choose chunk size from measurements of the target model, hardware
and workload.

<a id="sarathi-и-sarathi-serve"></a>

### SARATHI and Sarathi-Serve

[SARATHI](https://arxiv.org/abs/2308.16369) combines chunked prefill
with decode-maximal batching: one prefill chunk joins many decode
steps, sharing linear-layer execution and weight reads.

[Sarathi-Serve Algorithm 3](https://arxiv.org/abs/2403.02310) fills
the batch with ongoing decodes first, incomplete prefills second,
and new requests last while budget remains. The name *stall-free*
describes this schedule: existing generation keeps receiving decode
steps during new-prompt processing. A mixed pass can still take longer
than decode alone, so latency targets require separate verification.

With a twelve-position budget, four ongoing decodes occupy four.
An admitted prompt with seven remaining positions takes seven more.
A new request receives one position if memory admission succeeds.
The next iteration reevaluates membership: the finished prefill begins
generation and the new request continues its input chunks. This
budget produces a concrete schedule retaining old decodes on every pass.

<a id="как-меняется-пакет-за-три-прохода"></a>

### Three mixed passes

Allow five new positions per pass. Three prompts wait:
`Who ate the pie`, `I am sure this project` and
`The largest animal`. The figure explicitly defines tokenization
and lengths, including `<bos>`, as 5, 7 and 6. These are teaching
splits, not results from a specified tokenizer.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/scheduling-2026-10-25/hf-continuous-batching.png]]

*Three five-position passes: complete prefill of prompt one; its decode
with four positions of prompt two; the last three of prompt two with
the first two of prompt three. Blue is cached history and white is
new input. Matrices encode permitted attention, not physical KV
placement. Rémi Ouazan Reboul, Arthur Zucker and Luc Georges,
[Continuous batching from first principles](https://huggingface.co/blog/continuous_batching#continuous-batching).
[Unchanged original](https://huggingface.co/datasets/huggingface/documentation-images/blob/c8c3cf72b3b724ce734259e4a086d30618a9198b/blog/continuous_batching/continuous_batching.png),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

Pass one processes all five positions and selects `?`.
On pass two, `?` takes one new position; four others process
`[<bos>, I, am, sure]` from prompt two. Request one selects EOS
and finishes. Request two cannot answer yet: `this pro ject`
has not been processed.

Pass three has no ongoing decode. Three positions finish prompt two;
two start prompt three with `[<bos>, The]`. The final `ject`
position selects `will`. Request three still has four input tokens
left, so its answer has not begun. Request one's EOS is not submitted
as another input, and that request no longer needs its old KV.

On pass two, five queries read ten keys: six from request one and
four from request two. On pass three, five queries read nine keys:
seven from request two and two from request three. A square
`5 × 5` triangular mask loses cached history; a global
`5 × 10` triangular mask mixes independent requests.

Each pair in the following code gives cached and new positions for
one request. Query rows and key groups follow the same request order;
positions within each group start at zero. The dense mask is for
inspection, not an optimized paged kernel.

```python
def packed_causal_mask(requests):
    if not requests or any(c < 0 or q < 1 for c, q in requests):
        raise ValueError("nonnegative cached length and positive query length required")
    key_count = sum(c + q for c, q in requests)
    mask, key_start = [], 0
    for cached, queries in requests:
        for position in range(cached, cached + queries):
            mask.append([
                key_start <= key <= key_start + position
                for key in range(key_count)
            ])
        key_start += cached + queries
    return mask

second = packed_causal_mask([(5, 1), (0, 4)])
assert len(second) == 5 and len(second[0]) == 10
assert second[0] == [True] * 6 + [False] * 4
assert [sum(row) for row in second] == [6, 1, 2, 3, 4]

third = packed_causal_mask([(4, 3), (0, 2)])
assert len(third) == 5 and len(third[0]) == 9
assert [sum(row) for row in third] == [5, 6, 7, 1, 2]
```


The first row of pass three is `this`: it sees four cached tokens
and itself. The next row, `pro`, sees six; `ject` sees seven.
The first row of the new prompt sees only its own `<bos>`,
even though another user's context precedes it in the packed array.
The [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55c Serving engines — vLLM, SGLang, TensorRT-LLM и FlashInfer#Проверка упакованного внимания на CPU|next chapter]]
computes attention and checks request isolation.

More uniform pass durations can also help pipeline parallelism.
Alternating short decode passes with long prefills can leave pipeline
stages waiting on neighbors. Chunking reduces duration variability;
communication costs and stage dependencies remain.

<a id="почему-токеновый-бюджет-не-равен-бюджету-времени"></a>

### A token budget is not a time budget

A chunk of $q$ new positions after a $c$-position prefix contains

$$
N_{\mathrm{pairs}}=qc+\frac{q(q+1)}2
$$

permitted causal query–key pairs. At $q=4,c=16$ there are 74.
With the same four new positions and $c=16{,}000$, there are 64,010.
Linear layers process the same number of rows, but attention cost differs.

`max_num_batched_tokens` limits new positions per pass, not its
maximum duration. Context lengths, layer composition, kernels,
parallelism and hardware also affect time.

Smaller chunks reduce each pass's new prefill work but increase
launches, repeated weight reads and prefix accesses. Compare TTFT,
individual token intervals of active answers and server throughput.

<a id="prefix-caching-не-вычислять-общий-вход-повторно"></a>

## Prefix caching: avoid recomputing shared input

Requests often share a system instruction, examples, document or
conversation history. An ordinary KV cache reuses history within
one generation. Prefix caching extends reuse across requests:
if the required K/V remain available, a matching beginning need
not be processed again.

The cache stores shared-context keys and values. The new request still
processes its distinct suffix and generates its answer. Tokens and
all conditions affecting their representations must agree.
Identical visible text with different tokenization or chat templates
may not produce the same prefix.

<a id="почему-ключ-блока-зависит-от-предыдущих-блоков"></a>

### Why block keys depend on earlier blocks

vLLM's automatic prefix-cache key for a complete block includes its
token IDs, the previous prefix hash and extra inputs such as an
adapter or multimodal content. Identical local tokens after different
histories do not imply identical KV, even at equal absolute positions:
their representations depend on preceding context.

The example uses a teaching serialization different from vLLM's
internal byte format. Its namespace fixes model revision, adapter
and the permitted sharing domain.

```python
import hashlib
import json

def prefix_keys(tokens, block_size, namespace):
    if block_size < 1:
        raise ValueError("positive block size required")
    parent, keys = "", []
    for start in range(0, len(tokens) - block_size + 1, block_size):
        block = list(tokens[start:start + block_size])
        payload = json.dumps(
            [parent, block, namespace],
            sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        parent = hashlib.sha256(payload).hexdigest()
        keys.append(parent)
    return keys

ns = {"model": "model-revision-1", "adapter": None, "tenant": "team-A"}
a = prefix_keys([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 4, ns)
b = prefix_keys([1, 2, 3, 4, 5, 6, 7, 8, 90, 91, 92], 4, ns)
c = prefix_keys([0, 2, 3, 4, 5, 6, 7, 8], 4, ns)
assert a == b and len(a) == 2  # два общих полных блока
assert a[1] != c[1]           # токены второго блока те же, история другая
assert a != prefix_keys(list(range(1, 12)), 4, {**ns, "tenant": "team-B"})
```


Two inputs share eight tokens and differ in their final three.
At block size four, two complete blocks match; the partial tail
is not indexed by this full-block implementation.
A cache supporting finer granularity could reuse a shorter piece
if it protects the shared data against modification.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-scheduling-review-2026-10/vllm-prefix-cache-overview.png]]

*vLLM automatic prefix-cache structures: physical block pool, allocation
free queue, hash index and request references. The index finds content;
it does not itself allocate another physical block.
[vLLM documentation](https://github.com/vllm-project/vllm/blob/23fbc2ec2f7219888bfc76cf987381f501151971/docs/design/prefix_caching.md),
vLLM contributors, [Apache-2.0](/licenses/vllm-23fbc2ec-Apache-2.0.txt).*

<a id="как-запрос-получает-найденные-блоки"></a>

### How a request acquires matching blocks

Lookup and allocation are sequential. A matching indexed block can
also be on the free queue: its content remains, but no active request
uses it. Pin that block before allocating the new tail, or allocation
could destroy the match it is supposed to reuse.

The complete operation sequence below is reproduced from
[Automatic Prefix Caching, Block Allocation](https://github.com/vllm-project/vllm/blob/23fbc2ec2f7219888bfc76cf987381f501151971/docs/design/prefix_caching.md#block-allocation),
vLLM contributors, [Apache-2.0](/licenses/vllm-23fbc2ec-Apache-2.0.txt).

<!-- source-excerpt:scheduling-vllm-allocation:start -->
**New request:** Workflow for the scheduler to schedule a new request with KV cache block allocation:

1. The scheduler calls `kv_cache_manager.get_computed_blocks()` to get a sequence of blocks that have already been computed. This is done by hashing the prompt tokens in the request and looking up cache blocks.
2. The scheduler calls `kv_cache_manager.allocate_slots()`. It does the following steps:
    1. Compute the number of new required blocks, and return if there are no sufficient blocks to allocate.
    2. “Touch” the computed blocks. It increases the reference count of the computed block by one, and removes the block from the free queue if the block wasn’t used by other requests. This is to avoid these computed blocks being evicted. See the example in the next section for illustration.
    3. Allocate new blocks by popping the heads of the free queue. If the head block is a cached block, this also “evicts” the block so that no other requests can reuse it anymore from now on.
    4. If an allocated block is already full of tokens, we immediately add it to the cache block, so that the block can be reused by other requests in the same batch.

**Running request:** Workflow for the scheduler to schedule a running request with KV cache block allocation:

1. The scheduler calls `kv_cache_manager.allocate_slots()`. It does the following steps:
    1. Compute the number of new required blocks, and return if there are no sufficient blocks to allocate.
    2. Allocate new blocks by popping the heads of the free queue. If the head block is a cached block, this also “evicts” the block so that no other requests can reuse it anymore from now on.
    3. Append token IDs to the slots in existing blocks as well as the new blocks. If a block is full, we add it to the cache block to cache it.

**Duplicated blocks**
Assuming block size is 4 and you send a request (Request 1\) with prompt ABCDEF and decoding length 3:

```text
Prompt: [A, B, C, D, E, F]
Output: [G, H, I]

Time 0:
  Tokens: [A, B, C, D, E, F, G]
  Block Table: [0 (ABCD), 1 (EFG)]
  Cache Blocks: 0
Time 1:
  Tokens: [A, B, C, D, E, F, G, H]
  Block Table: [0 (ABCD), 1 (EFGH)]
  Cache Blocks: 0, 1
Time 2:
  Tokens: [A, B, C, D, E, F, G, H, I]
  Block Table: [0 (ABCD), 1 (EFGH), 2 (I)]
  Cache Blocks: 0, 1
```

Now block 0 and block 1 are cached, and we send the same request again (Request 2\) with greedy sampling, so that it will produce exactly the same outputs as the Request 1:

```text
Prompt: [A, B, C, D, E, F]
Output: [G, H, I]

Time 0:
  Tokens: [A, B, C, D, E, F, G]
  Block Table: [0 (ABCD), 3 (EFG)]
  Cache Blocks: 0, 1
Time 1:
  Tokens: [A, B, C, D, E, F, G, H]
  Block Table: [0 (ABCD), 3 (EFGH)]
  Cache Blocks: 0, 1, 3
```

As can be seen, block 3 is a new full block and is cached. However, it is redundant as block 1, meaning that we cached the same block twice. In v0, when detecting block 3 is duplicated, we free block 3 and let Request 2 use block 1 instead, so its block table becomes `[0, 1]` in Time 1. However, the block table in vLLM v1 is append-only, meaning that changing the block table from `[0, 3]` to `[0, 1]` is not allowed. As a result, we will have duplicated blocks for the hash key E-H. This duplication will be eliminated when the request is freed.
<!-- source-excerpt:scheduling-vllm-allocation:end -->

<a id="полное-попадание-и-последний-forward"></a>

### A full hit still needs output logits

An eight-token prompt matches two cached four-token blocks, positions
1–4 and 5–8. Selecting its first answer token requires output logits
after position eight.

KV stores layer keys and values, not the final hidden state of
position eight used by the vocabulary projection. This vLLM path
therefore processes the last position again to obtain that state
and its logits. Mathematically, cached K/V for positions 1–7 plus
one new computation for position eight would suffice.

The API adds a block-alignment constraint: the returned computed
prefix length must be divisible by block size. Seven positions
cannot be returned at block size four. The largest eligible prefix
shorter than eight has length four:

| Input portion | Action |
|---|---|
| Positions 1–4 | Reuse the stored block |
| Positions 5–8 | Run forward again |
| After position 8 | Compute logits and select the first answer token |

Four positions are recomputed because final-position logits are
needed and the reusable boundary must be block-aligned.
The pinned
[`get_computed_blocks`](https://github.com/vllm-project/vllm/blob/23fbc2ec2f7219888bfc76cf987381f501151971/vllm/v1/core/kv_cache_manager.py#L264)
states this in its original comment, vLLM contributors,
[Apache-2.0](/licenses/vllm-23fbc2ec-Apache-2.0.txt):

<!-- source-excerpt:scheduling-vllm-full-hit:start -->
```text
# NOTE: When all tokens hit the cache, we must recompute the last token
# to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
# This can trigger recomputation of an entire block, rather than just
# the single last token, because allocate_slots() requires
# num_computed_tokens to be block-size aligned. Removing this limitation
# could slightly improve performance in the future.
```
<!-- source-excerpt:scheduling-vllm-full-hit:end -->

For prompt length $S$, block size $b$ and matching-prefix length
$H$, first cap reusable positions at $S-1$, then round down to
whole blocks:

$$
\begin{aligned}
R&=b\left\lfloor\frac{\min(H,S-1)}b\right\rfloor,\\
S_{\mathrm{new}}&=S-R,
\end{aligned}
$$

$R$ counts reused positions; $S_{\mathrm{new}}$ counts positions
computed again. With $S=8,H=8,b=4$, $R=4$ and four positions run.
At prompt length nine, both blocks end before the final position:
$R=8$, so only position nine runs. At length eleven with the same
match, eight are reused and three run.

This describes the pinned implementation's local full-attention cache.
An engine with different reuse granularity or additional cached model
outputs can handle a full match differently.

```python
def local_full_block_hit(prompt_length, cached_prefix, block_size):
    if prompt_length < 1 or cached_prefix < 0 or block_size < 1:
        raise ValueError("invalid prefix")
    reusable = min(cached_prefix, prompt_length - 1) // block_size * block_size
    return reusable, prompt_length - reusable

assert local_full_block_hit(8, 8, 4) == (4, 4)
assert local_full_block_hit(11, 8, 4) == (8, 3)
assert local_full_block_hit(9, 8, 4) == (8, 1)
```


<a id="свободный-блок-не-обязательно-пустой"></a>

### A free block need not be empty

Finishing a request releases its block references. A block needed by
another active request stays occupied. With no references, it can
become allocatable while retaining old content and its key until
overwritten. These are two operations:

- releasing a reference: the request no longer needs the block;
- evicting content: remove the old index key and assign the physical
  storage to new data.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-scheduling-review-2026-10/vllm-prefix-release-shared-blocks.png]]

*After Request 0 finishes, its private blocks 4, 3 and 2 join the free
queue. Shared blocks 0 and 1 are still needed by Request 1.
Blocks 7, 8 and 9 were already free, giving queue order
7, 8, 9, 4, 3, 2.
[Original and step-by-step explanation](https://github.com/vllm-project/vllm/blob/23fbc2ec2f7219888bfc76cf987381f501151971/docs/design/prefix_caching.md#example),
vLLM contributors, [Apache-2.0](/licenses/vllm-23fbc2ec-Apache-2.0.txt).*

A hit on an unused cached block pins it to the request and removes it
from the free queue. Before allocating other blocks, verify capacity
for both those pins and the new tail. Shared-prefix computation is
saved, but the pinned blocks still occupy memory while the request runs.

Unreferenced old blocks may be overwritten for new allocations,
losing future reuse opportunities. Blocks pinned by active requests
cannot be overwritten. Reservations and manager policies further
constrain available capacity.

<a id="radixattention-и-порядок-обслуживания"></a>

### RadixAttention and service order

[SGLang](https://arxiv.org/abs/2312.07104) stores prefixes in a
compressed radix tree. Edges represent token sequences; divergent
continuations create branches. If a request matches only an edge's
beginning, split the edge at divergence. This index change does not
recompute matching K/V.

An initial `[1,2,3,4,5]` request forms one edge. Then
`[1,2,3,8]` shares the first three tokens, splitting that edge.
After `[1,2,3]`, the continuations are `[4,5]` and `[8]`.
The first three positions' shared K/V remain in place.

A third request, `[1,2,3,4,9]`, follows the common start and
matches `[4]` within `[4,5]`. That edge splits into
continuations `[5]` and `[9]` after position 4.
Four positions are reused; only the final one must be computed.

This example assumes one-token granularity. The tree is on CPU and
its KV tensors on GPU; splitting an edge changes the index, not
the shared tensors. See the
[original RadixAttention explanation](https://www.lmsys.org/blog/2024-01-17-sglang/#backend-automatic-kv-cache-reuse-with-radixattention).
In the pinned
[RadixCache](https://github.com/sgl-project/sglang/blob/bab9c81fea62c2d20b646736c918c88fea54b7a2/python/sglang/srt/mem_cache/radix_cache.py),
matches align to `page_size` when it exceeds one, and require
the same `extra_key` namespace.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-scheduling-review-2026-10/sglang-radix-tree-original.jpg]]

*The original RadixAttention example: request i on the left continues
an existing conversation; answers j, k and l on the right share examples.
Orange dashed branches are evicted while their common beginning stays.
This is one state selected by the authors, not the numeric tree above.
[LMSYS, Fast and Expressive LLM Inference](https://www.lmsys.org/blog/2024-01-17-sglang/);
[unchanged original file](https://github.com/lm-sys/lm-sys.github.io/blob/88c036dd1a78c8f2525782f98ee4493a63d7a532/public/images/blog/sglang/radix_attn_preview.jpg),
[MIT](/licenses/lmsys-88c036dd-MIT.txt).*

References protect used branches. Memory reclamation chooses unprotected
leaves. Deleting one can turn its parent into another candidate leaf.
Common beginnings still needed by other branches remain.

Cache state also affects request order. Suppose only one eight-token
shared prefix can remain cached between sequential requests.
The queue is A1, B1, A2; A1 and A2 share a prefix, while B1 has another.
Order A1–B1–A2 computes a prefix three times because B evicts A.
A1–A2–B1 computes it twice, saving eight input-token computations
but making B1 wait longer.

[SGLang policies](https://github.com/sgl-project/sglang/blob/bab9c81fea62c2d20b646736c918c88fea54b7a2/python/sglang/srt/managers/schedule_policy.py)
include longest-prefix matching, `lpm`, among other choices.
Cache-hit priority reduces repeated computation, but a continuous
stream of matching requests can defer a miss. Measure waiting time
for both hits and misses alongside saved computation.

<a id="корректность-и-изоляция"></a>

### Correctness and isolation

Matching a text fragment is insufficient. Weight revision, positions,
adapter, multimodal representations and every other input affecting
KV must agree. Some conditions may be fixed for an entire cache
instance; others must enter the key. Old KV cannot be used after
a model-weight change.

A shared cache may also reveal a prior request because hits run faster.
vLLM's `cache_salt` separates reuse domains. Cryptographic hashes
reduce dangerous collisions; appropriately chosen salts limit sharing
between groups. Neither alone proves the absence of other side channels.
Hash choice and shared-cache access rules require separate review.

<a id="как-механизмы-взаимодействуют"></a>

## How the mechanisms interact

An 18-token prompt has two reusable four-token blocks.
Ten positions remain. With chunks of four, they need three passes:
4, 4 and 2.

Four other requests continue decoding, and each pass has a budget
of eight new positions. Give each decode one position and allocate
the remaining four to our prefill. Assume enough memory and no
neighboring completion during these passes:

| Pass | New prefill positions | Total processed | User-visible result |
|---|---|---:|---|
| Before first pass | — | 8 from cache | Waiting |
| 1 | 9–12 | 12 | Waiting: six remain |
| 2 | 13–16 | 16 | Waiting: two remain |
| 3 | 17–18 | 18 | First answer token |

Pass three uses six of eight positions. The scheduler can assign
the remainder elsewhere if admission policy and memory allow.
After selecting its first answer token, our request also enters decode.
Its next pass processes that token using K/V for all eighteen
prompt positions.

A neighboring completion releases block references and a batch slot.
The next iteration chooses membership and budgets again.

| Mechanism | What it changes | What it does not resolve by itself |
|---|---|---|
| Continuous batching | Membership between iterations | Long-prefill cost |
| Blocked KV cache | History placement and sharing | Service order |
| Chunked prefill | Assigned new-prompt work per pass | Repeated computation of identical inputs |
| Prefix caching | Positions that need recomputation | Generating new answers and reading their contexts |

A prefix hit leaves the next token's context unchanged; attention
still reads its K/V. The hit alone does not reduce decode attention
work. Sharing can nevertheless free memory for more requests, while
saved prefill reduces GPU competition. Those effects can improve
aggregate generation performance.

Short-prompt, long-answer workloads accumulate growing decode histories.
Long-document workloads add prefill work controlled by chunk size.
Agents with repeated instructions and branching contexts can reuse
prefixes. Validate a configuration on the request distribution it
will actually serve.

<a id="версии-движков-и-проверка-производительности"></a>

<a id="устойчивый-механизм-и-меняющийся-api"></a>

## Engine versions and performance verification

The vLLM details above refer to `23fbc2ec`; SGLang details refer
to `bab9c81f`, checked on 14 October 2026.
Other versions may behave differently.

Record engine version, token budget, concurrency limit, block size,
queue policy, length limits, parallelism and cache settings.
Sarathi-Serve's comparisons with vLLM concern the version used in
that paper, not today's V1 implementation.

Measure the following on the target workload:

- Input tokens actually computed and the fraction reused. Requests
  with any hit do not measure the number of saved positions.
- TTFT and queue wait. A short prefill can still wait a long time
  for admission.
- Individual inter-token intervals. A request's mean TPOT can hide
  a long pause.
- Preemptions and repeated prefills. GPU activity can rise because
  completed work is being repeated.

Keep cold/warm cache conditions, arrival order, length distributions
and latency requirements comparable. The
[[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/58b Benchmarking, SLO и эксплуатация inference|inference benchmarking chapter]]
develops the protocol. Running this chapter's Python examples checks
arithmetic and teaching-model behavior; it does not measure vLLM or
SGLang latency and is not a load test.

<a id="практика-и-первоисточники"></a>

## Practice and primary sources

- [[02 Areas/ML & DL/06 Практика/14b Планировщик запросов и KV-память|Interactive scheduler and KV-memory lab]]: edit arrival times, input/output lengths, token budget and KV pool, then step through the allocations. The simulator does not run a neural network or predict GPU latency.

- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week08_inference_software/lecture.pdf|Efficient DL Systems: inference scheduling]].
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week08_inference_software/seminar.ipynb|Scheduler state and request lifecycle seminar]].
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week08_inference_software/homework/homework_week8.ipynb|Inference engine assignment]].
- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol2/inference|Harvard CS249r: Inference]].
- Stanford CS336 Spring 2026, [Lecture 10](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_10.py#L553): dynamic lengths, selective batching and memory.
- Yu et al., [Orca, §3–4](https://www.usenix.org/system/files/osdi22-yu.pdf): scheduler/executor coordination at iteration boundaries.
- Agrawal et al., [SARATHI](https://arxiv.org/abs/2308.16369) and [Sarathi-Serve, §3–4](https://arxiv.org/abs/2403.02310): mixed prefill/decode and latency trade-offs.
- Kwon et al., [PagedAttention](https://arxiv.org/abs/2309.06180): blocks, sharing and preemption recovery.
- Zheng et al., [SGLang and RadixAttention](https://arxiv.org/abs/2312.07104): radix trees and cache-aware scheduling.
- [vLLM Prefix Caching](https://github.com/vllm-project/vllm/blob/23fbc2ec2f7219888bfc76cf987381f501151971/docs/design/prefix_caching.md): the full block-lifecycle example behind the figures.
- Aleksa Gordić, [Inside vLLM](https://www.aleksagordic.com/blog/vllm): implementation walkthrough and original diagrams.
- Ouazan Reboul, Zucker and Georges, [Continuous batching from first principles](https://huggingface.co/blog/continuous_batching): new-input shapes, padding and mixed-pass sequence.

Previous: [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention|KV-cache and PagedAttention]].
Next: [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55c Serving engines — vLLM, SGLang, TensorRT-LLM и FlashInfer|Serving engine architecture]].
