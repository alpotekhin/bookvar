---
title: "Disaggregated prefill and decode serving"
type: textbook-chapter
status: canonical
locale: en
translation_status: current
canonical_revision: "7973703ca7ec77248f0fdc77321e7c5f9805164d182508234fb5c49c3e86a13d"
translation_reviewed: "2026-10-29"
translation_of: "00 Учебник/14 Inference и оптимизация/58a2 Раздельное обслуживание prefill и decode.md"
last_updated: 2026-10-29
last_verified: 2026-07-23
primary_sources:
  - https://jax-ml.github.io/scaling-book/inference/
  - https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf
  - https://arxiv.org/abs/2311.18677v2
  - https://arxiv.org/abs/2407.00079v1
  - https://docs.nvidia.com/dynamo/v1.5.1/knowledge-base/concepts/system-architecture/disaggregated-serving
  - https://docs.sglang.io/docs/advanced_features/pd_disaggregation
---

<span id="раздельное-обслуживание-prefill-и-decode"></span>

# Disaggregated prefill and decode serving

Suppose a server is generating answers for 32 requests and completes their
shared decode step in 10 ms. A new long prompt takes 120 ms to prefill.
If the server executes that prefill first, the next tokens of the active
answers arrive after roughly 130 ms instead of 10 ms. These are hypothetical
times for a sequential schedule, not measurements of an engine.

A separate GPU group can process the new prompt while the original group
continues generation. The prefill group, P, then transfers KV to the decode
group, D, that will continue its answer. This is *prefill/decode disaggregation*.
Either group can itself contain multiple GPUs using TP and PP.

State transfer now lies on the request path. For 32 layers, eight KV heads
of width 128, and BF16, an 8192-token prompt creates 1 GiB of KV.
At a useful transfer rate of 25 GiB/s, moving it takes 40 ms.
Separation prevents the long prefill from interrupting existing streams, but
the new request still needs its state at D. The tradeoff includes both delays
and the GPU and memory capacity allocated to both pools.

<span id="один-запрос-от-очереди-до-последнего-токена"></span>

## One request, from admission to the final token

The frontend validates input, applies the chat template, and tokenizes it.
The router selects P, where the request may wait. Prefill computes every layer's
K/V for the known input positions and the final position's logits. Sampling
chooses the first output token. D receives the continuation state, admits the
request into its batch, and generates the rest.

That first output token has been selected, but its K/V have not yet been
computed. For input IDs [12, 45, 77], prefill caches three positions and might
select token 98. The first D step feeds 98 at position 3, appends the fourth
KV position, and selects output token 2. Feeding the last prompt token again,
or treating 98 as already cached, produces an incorrect continuation.
The same cached-prefix/uncached-token boundary appears in
[[58 Speculative decoding|speculative decoding]].

A handoff therefore needs more than K/V bytes: request identity and version,
input and selected tokens, valid cache length, positions, sampling parameters,
and generation-constraint state. Reproducing stochastic sampling exactly also
requires a consistent random-generator state. Fields and the owner of sampling
depend on the engine; there is no universal packet containing only a model
name and KV.

TTFT ends when the client receives the first token. ITL, also called TBT in
some papers, is the gap between adjacent output tokens. For client timestamps
$t_1,\ldots,t_O$ measured from request submission,

$$
\begin{aligned}
TTFT&=t_1,\\
ITL_i&=t_i-t_{i-1}\quad(i\ge2),\\
E2E&=TTFT+\sum_{i=2}^{O}ITL_i=t_O.
\end{aligned}
$$

Here E2E ends at the last token, not a later stream-closing event. For $O>1$,
$TPOT=(t_O-t_1)/(O-1)$ averages the gaps after token 1. It is undefined for
a one-token response. Tools using a different TPOT definition need to be
reconciled with their actual timestamps.

~~~python
def token_latencies(times_ms):
    assert len(times_ms) > 0
    assert times_ms[0] >= 0
    assert all(b >= a for a, b in zip(times_ms, times_ms[1:]))
    intervals = [b - a for a, b in zip(times_ms, times_ms[1:])]
    return {
        'ttft': times_ms[0], 'e2e': times_ms[-1],
        'itl': intervals,
        'tpot': sum(intervals) / len(intervals) if intervals else None,
    }

latency = token_latencies([120., 190., 200., 210.])
assert latency['itl'] == [70., 10., 10.]
assert latency['tpot'] == 30.
assert token_latencies([120.])['tpot'] is None
~~~

The mean of 30 ms hides the initial 70 ms pause. KV handoff can fall in that
gap. Later we compare immediate emission of token 1 with buffering until D
is ready.

<span id="почему-prefill-чаще-compute-bound"></span>

## Why prefill is often compute-bound

Prefill applies the same weight matrix to many known token rows. Ignoring
activation traffic, a BF16 matrix with $W$ elements requires $2W$ bytes of
weight reads and about $2SW$ FLOP for $S$ tokens. The ratio is roughly
$S$ FLOP/byte: 8192 at $S=8192$, versus one for a single row.
A real kernel also reads and writes activations, so this is one contribution
to arithmetic intensity, not a complete roofline.

Dense causal attention processes a quadratically growing set of allowed
query–key pairs. FlashAttention reduces HBM traffic without making that
calculation linear. One constant “prefill tokens/s” therefore cannot reliably
predict all input lengths and prefix-hit fractions.

A sufficiently large input can already occupy the compute units well.
Adding a second long request increases batch duration without necessarily
improving efficiency. [DistServe §3.1](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf#page=5)
demonstrates this for a 13B model on A100. Its roughly 512-token threshold
belongs to that measurement, not every model or GPU.

Short prefill can remain memory-, launch-, or communication-bound.
[[55a The physics of LLM inference — prefill, decode, and roofline|The physics of inference]]
introduces these regimes; here they determine which prompts and batches a
separate P pool should serve.

<span id="почему-decode-чаще-memory-bandwidth-bound"></span>

## Why decode is often memory-bandwidth-bound

Ordinary autoregressive decode cannot know the next input until sampling
finishes. Each request contributes one new row per step. Independent requests
increase batch size. In the same simplified BF16 weight-read calculation,
intensity grows with request count $B$, not known prompt length $S$:
about 32 FLOP/byte for $B=32$, rather than 8192 for the prefill example.

Attention also reads each request's historical KV. Ordinary MHA/GQA with equal
K/V widths uses

$$
M_{KV}=2LSn_{kv}d_hb,
$$

where $L$ is layer count, $S$ cached positions, $n_{kv}$ KV heads,
$d_h$ head width, and $b$ bytes per element. MHA and GQA can have very
different footprints. MLA stores a different latent representation; this formula
cannot be applied unchanged. See
[[57a KV-cache compression и offload|KV-cache compression and offload]].

For a chosen placement, the read-time lower bound is approximately

$$
T_{step}\gtrsim
\frac{M_{weights}+\sum_{r\in batch}M_{KV,r}}
{BW_{HBM}}.
$$

Numerator and denominator must describe the same GPUs and actual traffic.
Dividing logical KV by aggregate HBM bandwidth is invalid if state is replicated
or work is imbalanced. Communication, additional kernel passes, and buffer
traffic add to real step time.

Large batches reuse weights, but each request adds KV and attention work.
D is constrained by capacity, step duration, and the ITL target together.
Freeing HBM alone does not make a larger batch acceptable.

<a id="интерференция-в-общем-gpu-pool"></a>

<span id="почему-новый-prefill-задерживает-активные-ответы"></span>

<span id="interference-in-a-shared-gpu-pool"></span>

## Why a new prefill delays active answers

A unified iteration-level scheduler removes completed requests and admits new
ones after each pass. Continuous batching need not wait for a whole batch to
finish generating, but the current pass must still finish before its next
tokens can be emitted.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/distributed-serving-2026/distserve-fig2-prefill-decode-interference.png]]

*Zhong et al., [DistServe, Figure 2](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf#page=5).
The authors' 13B measurement compares a decode-only batch in orange with the same
batch plus one prefill in blue. Prompt lengths are 128 on the left and 1024 on
the right. Horizontal dashed lines mark prefill running alone. Vertical arrows
compare phase latencies, not KV volume.*

Return to a 10 ms decode batch and 120 ms prefill. Executing the whole prefill
first produces a 130 ms gap. Splitting it into four 30 ms chunks and running a
decode step after each lets active answers advance every 40 ms.
Prefill now finishes after
$30+10+30+10+30+10+30=150$ ms rather than 120 ms.
These are separate sequential jobs; real mixed kernels may share resources
differently.

Chunked prefill bounds one pause but introduces execution boundaries and
repeated reads of preceding KV. It may satisfy the target SLO without a separate
P pool. PD separation is useful when isolation and a different layout justify
state transfer and extra resources.

There can also be cross-process interference. In MoE with attention DP, one
process may run a large prefill while another is ready for short decode.
Synchronized expert exchanges make the faster process wait. SGLang describes
this in [Issues with Unified Scheduling](https://docs.sglang.io/docs/advanced_features/pd_disaggregation#issues-with-unified-scheduling).
Separating phases reduces that mismatch without removing load balancing inside
each pool.

<a id="что-именно-меняет-disaggregation"></a>

<span id="размещение-p-и-d-в-разных-пулах"></span>

<span id="what-disaggregation-changes"></span>

## Placing P and D in separate pools

P and D run the same model with compatible state, but normally hold separate
weight copies. P computes input prefixes; D retains long-lived KV for active
answers. Their machine counts need not match: multiple P instances can feed
one D or vice versa.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/distributed-serving-2026/distserve-fig6-runtime-architecture.png]]

*Zhong et al., [DistServe, Figure 6](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf#page=9).
The controller selects P and D instances, each with its own model and distributed
executor. Their connecting arrow moves layer KV, not model weights per request.*

Each phase can choose its own TP, PP, replicas, and hardware, as described in
[[58a Distributed inference and disaggregated serving|parallelism and collectives]].
The choices still depend on the queues and network between them.

D may hold more requests because prefill workspaces no longer occupy its memory.
But assigning two of an eight-GPU budget to P leaves only six for D; total
decode capacity can fall. Compare the whole service, not only an improved
D process. A shared pool may be cheaper at low load because specialized
machines would sit idle.

[Splitwise §IV](https://arxiv.org/abs/2311.18677v2) also uses a mixed pool.
When a phase's queue grows, some already-loaded machines temporarily execute
both kinds of work, then return to their primary role as those requests finish.
This is Splitwise's mechanism for avoiding stranded capacity, not a required
feature of every PD system.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/pd-serving-review-2026-10/splitwise-architecture.png]]

*Patel et al., [Splitwise, Figure 10, PDF p.6](https://arxiv.org/abs/2311.18677v2),
CC BY-NC-SA 4.0. CLS controls pool placement and routing; MLS controls a machine's
queue and batches. Blue/orange icons identify machine roles and green rectangles
are GPUs. The complete figure and source caption include the mixed pool and
direct KV transfers; surrounding page text is excluded.*

<a id="kv-transfer-как-отдельная-стадия-запроса"></a>

<span id="размер-и-время-передачи-kv"></span>

<span id="kv-transfer-is-a-first-class-serving-stage"></span>

## KV transfer size and duration

The initial GQA example has a 1 GiB prompt cache:

$$
\begin{aligned}
M_{KV}&=2\cdot32\cdot8192\cdot8\cdot128\cdot2\\
&=1\,073\,741\,824\text{ bytes}.
\end{aligned}
$$

At 25 GiB/s useful bandwidth, transfer takes 40 ms. Add a hypothetical sequential
5 ms of setup/coordination for 45 ms total. NIC line rate is not useful KV
bandwidth: GPU–NIC paths, memory registration, concurrent flows, message sizes,
and network contention matter. Waiting for an available channel is additional.

~~~python
GiB = 2**30
def kv_bytes(layers, tokens, kv_heads, head_dim, item_bytes):
    return 2 * layers * tokens * kv_heads * head_dim * item_bytes

size = kv_bytes(32, 8192, 8, 128, 2)
assert size == GiB
transfer_ms = size / (25 * GiB) * 1000
assert transfer_ms == 40.
assert transfer_ms + 5. == 45.
~~~

If D already has a compatible 2048-token prefix, the missing 6144 tokens
occupy 0.75 GiB. Transferring only that suffix requires actual block matching
and reuse support at D. Prefix caching on P alone does not provide it.
Some implementations deliberately exclude D prefix reuse from receiver selection.

TP/PP distribute state among processes. Do not multiply global cache size by TP
when ranks own disjoint shards. Conversely, replicated KV heads require accounting
for actual copies and choosing which ones to send. A PP stage owns only its layers.

<span id="передача-при-разных-разбиениях"></span>

### Different source and destination layouts

If P uses TP2 and D uses TP1, each P rank may own half the KV heads.
D must receive both halves and lay them out in its own format. Identical
element precision does not imply identical storage axes: layouts may prioritize
layers, pages, or heads differently.

The example's logical axes are [K/V kind, layer, position, KV head, head
coordinate]. Senders store [layer, K/V kind, head, position, coordinate].
The integers label elements, not actual LLM states.

~~~python
import numpy as np

logical = np.arange(2 * 2 * 4 * 4 * 2).reshape(2, 2, 4, 4, 2)
source_head_shards = np.split(logical, 2, axis=3)
wire_buffers = [part.transpose(1, 0, 3, 2, 4).copy()
                for part in source_head_shards]
restored_parts = [buffer.transpose(1, 0, 3, 2, 4)
                  for buffer in wire_buffers]
destination = np.concatenate(restored_parts, axis=3)
assert np.array_equal(destination, logical)
assert sum(buffer.nbytes for buffer in wire_buffers) == logical.nbytes
~~~

TP1→TP2 would perform the reverse distribution. A paged cache additionally maps
logical positions to physical block IDs, offsets, and partially filled pages.
Moving the correct byte count without axis metadata does not restore a valid cache.

Compatibility includes weights and adapters, position representation, K/V format,
quantization scales, page size, and rank map. Permuting axes cannot repair state
computed by a different model or adapter. Precision/kernel changes also need
separate verification: losslessly moving already-computed values does not prove
bitwise-identical subsequent computation on another device.

<span id="push-pull-и-владение-памятью"></span>

### Push, pull, and memory ownership

With push, P starts writing agreed D buffers. With pull, D fetches after
allocating capacity. This distinguishes the initiator and reservation order;
both may be asynchronous. [DistServe §4.3](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf#page=9)
lets D pull as capacity permits while P temporarily holds completed KV in GPU memory.

| State | What is known | What is still unsafe |
|---|---|---|
| Prefill finished | P holds values and selected token 1 | Freeing the only source copy before transfer completes |
| D memory reserved | Destination blocks and addresses are assigned | Reading them as ready KV |
| Transfer in flight | Some ranges may already be written | Attention depending on unfinished ranges |
| Transfer acknowledged | Required writes are complete and visible to the compute stream | Continuing with stale request placement |
| Source released | D owns the continuation state | Assuming a later D failure automatically restores KV |

Suppose P starts an asynchronous write into D's block 17 and the user cancels.
Removing the request from a queue and immediately reassigning block 17 can let
a late write corrupt the new request. The transfer must complete or be safely
stopped before buffer reuse. A request-generation ID can reject an old
acknowledgment; it does not stop a write already in flight.

Splitwise uses one-sided MSCCL++ puts and a per-request semaphore for completion.
SGLang has bootstrap metadata and backend-specific transfer states. These are
implementations of the dependencies, not a common API shared by all systems.

<span id="передача-по-слоям"></span>

### Transferring layer by layer

Early-layer K/V are ready before the full prefill ends. Sending them while later
layers compute reduces the remaining transfer tail but requires layer-level
readiness coordination.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/pd-serving-review-2026-10/splitwise-layer-transfer.png]]

*Patel et al., [Splitwise, Figure 11, PDF p.6](https://arxiv.org/abs/2311.18677v2),
CC BY-NC-SA 4.0. Left: compute, whole-cache transfer, and decode run sequentially.
Right: green layer transfers overlap blue prefill. Both panels and the original
labels are retained.*

The source describes the mechanism as follows in
[§IV-C, KV-cache transfer](https://arxiv.org/abs/2311.18677v2):

<!-- source-excerpt: splitwise-layer-handoff:start -->
> In Splitwise, we optimize the KV-cache transfer by overlapping it with the computation in the prompt phase.
> As each layer in the LLM gets calculated in the prompt machine, the KV cache corresponding to that layer is also generated.
> At the end of each layer, we trigger an asynchronous transfer of the KV-cache for that layer while the prompt computation continues to the next layer.
<!-- source-excerpt: splitwise-layer-handoff:end -->

Take four hypothetical layers of 20 ms each, with 30 ms transfer per layer
over one shared channel. Layer computation finishes at 20, 40, 60, and 80 ms.
Transfer 1 occupies 20–50 ms. Transfer 2 waits and ends at 80 ms, then
transfers 3/4 end at 110/140 ms. Sequential execution would take
$80+120=200$ ms. Streaming saves 60 ms but still leaves a 60 ms tail after
prefill.

~~~python
def layer_transfer_finish(compute_ms, transfer_ms):
    assert len(compute_ms) == len(transfer_ms) and len(compute_ms) > 0
    compute_end, transfer_end = 0., 0.
    ready = []
    for compute, transfer in zip(compute_ms, transfer_ms):
        compute_end += compute
        transfer_end = max(compute_end, transfer_end) + transfer
        ready.append(transfer_end)
    return compute_end, ready

prefill_end, ready = layer_transfer_finish([20.] * 4, [30.] * 4)
assert prefill_end == 80.
assert ready == [50., 80., 110., 140.]
assert ready[-1] - prefill_end == 60.
assert layer_transfer_finish([20.] * 4, [10.] * 4)[1][-1] == 90.
~~~

Even at 10 ms transfer per layer, the final layer is not delivered until 90 ms.
The approximation $\max(\sum T_{compute},\sum T_{transfer})$ would predict
80 ms and miss the final layer's readiness dependency. This example makes D
wait for the complete cache. Layer-wise continuation on D needs its own
dependency schedule.

Frequent small transfers also have overhead. Splitwise uses sequential transfer
for shorter prompts and layer-wise transfer for longer ones. Mooncake overlaps
layer-wise KV loading/saving with prefill, including transfers to the CPU pool.
Sources still retain buffers until completion, and chunked prefill may need
early-layer KV again for later chunks.

<span id="где-заканчивается-ttft-при-передаче-kv"></span>

### First-token emission determines the latency ledger

Let $A$ end when token 1 is ready on P; $X$ is the unhidden transfer tail;
$Q_D$ is the wait to join D's batch; and $D_1$ is the first decode step,
which produces output token 2. Assume sequential intervals with client delivery
included.

P-first emits token 1 immediately. D-ready buffers it until D can proceed:

$$
\begin{aligned}
TTFT_{\mathrm{P-first}}&=A,\\
ITL_{2,\mathrm{P-first}}&=X+Q_D+D_1,\\
TTFT_{\mathrm{D-ready}}&=A+X+Q_D,\\
ITL_{2,\mathrm{D-ready}}&=D_1.
\end{aligned}
$$

Both deliver token 2 at $A+X+Q_D+D_1$.

~~~python
def first_two_tokens(a, transfer, d_queue, first_decode, policy):
    assert policy in {'p-first', 'd-ready'}
    ready = a + transfer + d_queue
    first = a if policy == 'p-first' else ready
    second = ready + first_decode
    return {'ttft': first, 'ttst': second, 'first_itl': second - first}

p_first = first_two_tokens(120., 45., 15., 10., 'p-first')
d_ready = first_two_tokens(120., 45., 15., 10., 'd-ready')
assert p_first == {'ttft': 120., 'ttst': 190., 'first_itl': 70.}
assert d_ready == {'ttft': 180., 'ttst': 190., 'first_itl': 10.}
~~~

TTST is time to the second token. P-first has lower TTFT; D-ready starts with
a smoother stream. Neither makes token 2 arrive sooner in this example.
Compare systems with the same emission policy and both latency boundaries.
If reservation, transfer, and queuing overlap, use the actual critical path
rather than their sum. Buffering both tokens until the first decode step ends
would create yet another TTFT boundary.

<a id="routing-вычислительная-нагрузка-встречается-с-locality"></a>

<span id="выбор-исполнителей-и-повторное-использование-префикса"></span>

<span id="routing-load-meets-state-locality"></span>

## Routing and prefix reuse

P's cost depends on reusable prefix state. Matching an isolated text segment is
insufficient: its K/V depend on preceding tokens. Mooncake identifies blocks
through a hash chain that includes the previous block. State identity must also
account for the model, adapters, and other computation-changing inputs.

Consider 8192 input tokens and two P workers. A has a cached prefix of 6144
but 100 ms of queued work; B has neither a prefix nor a queue. For a hand
calculation, assign 0.02 ms per unprocessed token. This is a teaching model,
not a real LLM profile.

A finishes after $100+2048\cdot0.02=140.96$ ms.
B recomputing everything takes $8192\cdot0.02=163.84$ ms, so shortest-queue
routing loses. But B can fetch the 6144 cached positions from a distributed
CPU cache. At 128 KiB/position, 0.75 GiB takes 30 ms over 25 GiB/s.
B then finishes in $30+40.96=70.96$ ms. Largest-local-prefix routing also loses.

~~~python
def prefill_candidates(tokens, workers, bytes_per_token, bw_bytes_per_ms,
                       compute_ms_per_token):
    best_prefix = max(w['prefix'] for w in workers)
    plans = []
    for w in workers:
        local_time = w['queue'] + (tokens - w['prefix']) * compute_ms_per_token
        plans.append((local_time, w['name'], 'local'))
        if best_prefix > w['prefix']:
            transfer = (best_prefix - w['prefix']) * bytes_per_token / bw_bytes_per_ms
            time = w['queue'] + transfer + (tokens - best_prefix) * compute_ms_per_token
            plans.append((time, w['name'], 'fetch'))
    return plans

workers = [{'name': 'A', 'prefix': 6144, 'queue': 100.},
           {'name': 'B', 'prefix': 0, 'queue': 0.}]
fast = min(prefill_candidates(8192, workers, 128 * 2**10, 25 * GiB / 1000, .02))
assert fast[1:] == ('B', 'fetch') and np.isclose(fast[0], 70.96)
slow = min(prefill_candidates(8192, workers, 128 * 2**10, 2.5 * GiB / 1000, .02))
assert slow[1:] == ('A', 'local') and np.isclose(slow[0], 140.96)
~~~

At 2.5 GiB/s, remote reuse loses to A's local plan. The example assumes the
cache source can serve independently of its compute queue, while receiver
waiting, transfer, and computation are sequential. Real predictors use full
input length, reusable prefix length, GPU profiles, and network state.

[Mooncake v1 §5 and Algorithm 1](https://arxiv.org/html/2407.00079v1#S5)
compares prefix locality with queues and potential transfer.
Moving popular blocks also creates more local copies for later requests,
reducing hotspots. Its migration threshold is heuristic; the minimum-time
example above explains the decision but does not implement every branch of
Algorithm 1.

D selection then depends on free blocks, active context lengths, and the
P→D path. A nominally least-loaded D can have a slow transfer path.
A matching D prefix is useful only if the backend can deduct it from transferred
state; the Dynamo limitation below makes that boundary explicit.

<span id="две-связанные-очереди-и-backpressure"></span>

## Two coupled queues and backpressure

Completed prefill produces state that D must accept and keep during generation.
If P produces it faster than D admits it, intermediate KV accumulates.
After buffers fill, more prefill only increases waiting or wasted work.

Suppose D has room for one request while three P workers simultaneously start
100 ms prefills. Unless D frees more space, it can accept only one result.
The others must remain on P, wait elsewhere, or be rejected. Idle P at admission
did not guarantee successful continuation.

Count KV positions and promises, not just requests: 128-token and 128K-token
prefixes differ greatly. With capacity 16384 positions, 8192 active and 4096
reserved, a new 8192-position cache does not fit, even though existing reserved
bytes have not arrived. A new 4096-position cache would fit exactly, leaving
no margin for growth.

~~~python
def reserve_in_order(capacity, active, reserved, incoming):
    used = active + reserved
    assert 0 <= used <= capacity
    accepted = []
    for size in incoming:
        fits = used + size <= capacity
        accepted.append(fits)
        if fits:
            used += size
    return accepted, used

accepted, used = reserve_in_order(16384, 8192, 4096, [2048, 4096, 1024])
assert accepted == [True, False, True] and used == 15360
wrong, wrong_used = reserve_in_order(16384, 8192, 0, [2048, 4096, 1024])
assert wrong == [True, True, True]
assert wrong_used + 4096 > 16384
~~~

The test checks current transferred-position capacity. Generation creates more
KV, requiring headroom, controlled admission, or rescheduling. Unknown output
length means this one check cannot reserve an entire future conversation.

<span id="ранний-допуск-и-физическое-резервирование"></span>

### Early admission and physical allocation

Predicting that a request can probably be served differs from physically
reserving D blocks. Early allocation reduces the chance of finishing expensive
prefill without a destination and enables transfer setup. Those blocks,
however, remain unavailable to others while a long prefill runs.

Late allocation uses fresher load information and does not hold that memory
during prefill. It leaves completed KV on P and may add a D wait after expensive
computation. Layer-wise transfer also needs a valid destination before it can
begin. Neither timing is universally best.

[SGLang's Decode allocation policy](https://docs.sglang.io/docs/advanced_features/pd_disaggregation#decode-allocation-policy)
documents the early default and a “prefill_complete” option. In the documentation
snapshot reviewed on 20 October 2026, the latter delays per-request KV allocation,
not the total KV pool, and requires Mooncake, the Python bootstrap registry, and
attention TP=CP=PP=1. Attention DP is allowed with attention TP=1 per process.
Staging buffers and several other features are unsupported. Those constraints
must not be generalized to arbitrary TP/CP/PP configurations.

<span id="почему-проверка-только-текущей-нагрузки-запаздывает"></span>

### Why current load alone is too late

[Mooncake v1 §6](https://arxiv.org/html/2407.00079v1#S6) addresses admission
before prefill. Looking only at current D load omits requests already in flight
on P. P first admits a large wave; its results overload D, new work is rejected,
and P empties. After D drains, another wave begins.

The forecast must include accepted in-flight requests and expected D completions.
The early Mooncake design models decode with a common assumed duration;
accurate per-request output-length prediction is left for future work.
This does not guarantee acceptance: local D checks again after prefill.

A hard advance reservation would prevent some overflow but could hold too much
memory idle. Forecast admission estimates future load without necessarily
assigning all physical blocks immediately. The tasks are related, not identical.

<a id="independent-scaling--не-независимое-управление"></a>

<span id="число-p--и-d-исполнителей"></span>

<span id="independently-scalable-does-not-mean-independently-controlled"></span>

## How many P and D instances?

Let accepted arrival rate be $\lambda$ requests/s, output length $O\ge1$,
one P sustain $\mu_P$ requests/s, and one D produce $\nu_D$ new tokens/s at
the target ITL. P has already selected token 1, so D produces
$E[O]-1$ further tokens per request. Necessary average-flow bounds are

$$
\begin{aligned}
n_P\mu_P&\ge\lambda,\\
n_D\nu_D&\ge\lambda(E[O]-1),\\
BW_{network}&\ge\lambda E[M_{transfer}].
\end{aligned}
$$

They do not model bursts, length distributions, individual congested links, or
latency percentiles. The service rates belong to measured batch sizes and
placements, not peak hardware FLOP/s.

Take 10 requests/s, 200 output tokens each, 2 requests/s per P, and
1200 new tokens/s per D at the desired latency. At least five P and two D
instances are needed. TP2 on P and TP4 on D require 18 GPUs, not seven.
With mean generation residence time 2 s, Little's-law concurrency is about
$10\cdot2=20$ active decode requests, whose KV and headroom must fit.

~~~python
from math import ceil

rate, output_tokens = 10., 200.
p_instances = ceil(rate / 2.)
d_instances = ceil(rate * (output_tokens - 1) / 1200.)
assert (p_instances, d_instances) == (5, 2)
assert p_instances * 2 + d_instances * 4 == 18
assert rate * 2. == 20.
assert rate * GiB < 25 * GiB
~~~

These are lower bounds. Five P instances run at exactly average capacity,
with no margin for random arrivals; queue stability and TTFT are not assured.
Two D instances may have spare output rate yet lack space for long contexts.
Provisioning needs the joint input/output-length distribution.

Scaling takes time: assign GPUs, load weights, initialize communication groups,
and warm kernels or CUDA graphs. A shorter burst can end before a reactive
scale-up is ready. Minimum warm capacity, prediction, and admission control
therefore complement scaling. A D worker must drain or explicitly migrate active
state before removal; P may still own unacknowledged KV after computation ends.

Useful signals differ by component: queued input tokens, input lengths, prefix
hits, and TTFT for P; active KV positions, context lengths, free blocks, and
ITL for D; pending bytes and transfer tails for the network. Compute utilization
alone can be high for healthy P or moderate for memory-bound D missing its SLO.
A planner's predictions should be checked against replayed traces before it
is allowed to resize a live service.

<span id="время-выполнения-и-ожидание-в-очереди"></span>

### Service time versus queue delay

[DistServe §3.1, equations 1–3](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf#page=6)
compares prefill TP2 and PP2 at the same GPU count using equal input lengths,
Poisson arrivals, FCFS, and constant service times. This is an M/D/1
approximation, not a general continuous-batching predictor.

Let a full pass take $D=0.2$ s. TP2 gives speedup $K=1.6$ and a
0.125 s pass. PP2 keeps the approximately 0.2 s request path but has two
0.1 s stages and can admit another request every 0.1 s. TP has lower isolated
latency; PP has higher admission capacity in this example.

For constant service time $s$ and rate $R$ with $Rs<1$,

$$
W_q=\frac{Rs^2}{2(1-Rs)}.
$$

TP uses $s=D/K$ and adds $D/K$ for the pass.
The authors' balanced-PP approximation uses $s=D/2$ and adds the full path $D$.

~~~python
def md1_wait(rate, service):
    if rate * service >= 1:
        return float('inf')
    return rate * service**2 / (2 * (1 - rate * service))

def phase_latency(rate, full_time=.2, tp_speedup=1.6):
    tp_service = full_time / tp_speedup
    return (tp_service + md1_wait(rate, tp_service),
            full_time + md1_wait(rate, full_time / 2))

assert np.allclose(phase_latency(4.), [.1875, .23333333333333334])
assert np.allclose(phase_latency(7.), [.5625, .3166666666666667])
~~~

At four requests/s, TP has lower mean TTFT: 187.5 versus 233.3 ms.
At seven, PP wins: 316.7 versus 562.5 ms. Lower isolated execution time
did not preserve the advantage under queuing.
Variable lengths, unequal stages, and a p99 constraint require a new analysis.

<span id="подбор-конфигурации-по-slo"></span>

### Selecting a configuration under SLO constraints

DistServe searches valid phase layouts and uses an execution model to find the
load meeting a target SLO fraction. Fast inter-node links permit more independent
P/D choices. Under constrained networking, it places P and D segments of the
same layers within a node, allowing their KV to move over NVLink.
This couples PP placement to the KV path; it does not require all of P and D
inside one server.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/pd-serving-review-2026-10/splitwise-provisioning.png]]

*Patel et al., [Splitwise, Figure 12, PDF p.7](https://arxiv.org/abs/2311.18677v2),
CC BY-NC-SA 4.0. Each panel plots P-machine count horizontally and D-machine count
vertically. The authors' Splitwise-HH model uses BLOOM-176B, a coding trace,
and 70 requests/s. All nine panels, color scales, and the caption are retained;
this is not a universal pool ratio.*

Rows show TTFT, TBT, and E2E; columns show p50, p90, and p99.
Slowdown is relative to an isolated request on DGX-A100, not the fastest
color on the plot. All nine constraints must hold. TTFT limits are 2×, 3×,
and 6×; TBT and E2E limits are 1.25×, 1.5×, and 5×, respectively
([Table VI, PDF p.9](https://arxiv.org/abs/2311.18677v2)).

The star is the least-cost feasible configuration in that experiment:
27 P machines and three D machines. Removing capacity from one pool can
violate another row of constraints even if one selected metric still looks
acceptable. A different workload, model, or device requires new feasible regions.

<a id="distserve-splitwise-и-mooncake-три-исследовательских-акцента"></a>

<span id="сравнение-систем-distserve-splitwise-и-mooncake"></span>

<span id="splitwise-distserve-and-mooncake-three-complementary-emphases"></span>

## Comparing DistServe, Splitwise, and Mooncake

DistServe centers on layout choice under TTFT and TPOT requirements.
Its controller assigns a short P queue and a less-loaded D; D pulls completed
KV. A workload profile determines placement and replica count, reconsidered
when the workload changes. Advanced fault tolerance is future work in the
original implementation. A controller and two pools do not establish recovery
for every failed request.

Splitwise also studies hardware choice and a mixed pool. It compares DGX-A100,
DGX-H100, and power-limited H100. Its 2024 costs and proportions are not current
purchasing recommendations. The reusable method is to profile phases, retain
the actual request trace, impose latency constraints, and search feasible
configurations. It chooses sequential or layer-wise KV transfer per request.

Mooncake v1 treats KV as shared service state. Its Conductor coordinates prefix
reuse, movement, and compute load. Distributed CPU DRAM and SSD complement VRAM,
but attention still needs data on a suitable access path. External capacity
does not replace local decode bandwidth.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/distributed-serving-2026/mooncake-figure1-architecture.png]]

*Qin et al., [Mooncake v1, Figure 1](https://arxiv.org/abs/2407.00079v1).
The Conductor's three duties are cache-aware prefill assignment, block balancing,
and decode assignment. P is above D, with a CPU KV pool and RDMA transfers
between them. TBT denotes inter-token delay. This is the original 2024
architecture, not every current deployment of Mooncake Transfer Engine.*

<span id="длинный-prefill-части-входа-и-стадии-модели"></span>

### Long prefill: input chunks and model stages

Mooncake v1 uses *chunked pipeline parallelism* (CPP): different layer groups
form stages, and the known input is divided into chunks. Output activations
move between stages. The paper compares CPP with Ring Attention and other
context-partitioning methods that exchange data inside attention. It calls
those alternatives SP, a broader use of the term than the token-local Norm/
Dropout SP of the preceding chapter.

With two stages and chunks A/B, stage 0 first processes A and saves its K/V.
It can then process B while stage 1 processes A's output. When B reaches
stage 1, that stage must already have its K/V for A.

| Interval | Stage 0 | Stage 1 |
|---|---|---|
| 1 | A, early layers | Waiting |
| 2 | B, early layers, using A's KV | A, later layers |
| 3 | Available for other work | B, later layers, using A's KV |

Causality is preserved: B sees A at every layer, but B's early layer does not
depend on A's later layer. These are already-known prompt tokens, not guesses
of future output tokens.

The following checks that order using two causal attention layers. Each layer
keeps its own K/V; the mask includes prior cached positions.
Norm, MLP, and position transforms are omitted to keep the dependency test small.

~~~python
def empty_kv(width):
    return {'k': np.empty((0, width)), 'v': np.empty((0, width))}

def causal_attention_chunk(x, weights, cache):
    wq, wk, wv = weights
    previous = len(cache['k'])
    q = x @ wq
    k = np.concatenate([cache['k'], x @ wk])
    v = np.concatenate([cache['v'], x @ wv])
    scores = q @ k.T / np.sqrt(q.shape[-1])
    allowed = np.arange(len(k))[None, :] <= previous + np.arange(len(x))[:, None]
    scores = np.where(allowed, scores, -np.inf)
    p = np.exp(scores - scores.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    cache['k'], cache['v'] = k, v
    return x + p @ v

rng = np.random.default_rng(7)
x = rng.normal(size=(8, 3))
weights = [[rng.normal(size=(3, 3)) for _ in range(3)] for _ in range(2)]
whole = causal_attention_chunk(x, weights[0], empty_kv(3))
whole = causal_attention_chunk(whole, weights[1], empty_kv(3))

cache0, cache1 = empty_kv(3), empty_kv(3)
a0 = causal_attention_chunk(x[:4], weights[0], cache0)
b0 = causal_attention_chunk(x[4:], weights[0], cache0)
a1 = causal_attention_chunk(a0, weights[1], cache1)
b1 = causal_attention_chunk(b0, weights[1], cache1)
assert np.allclose(np.concatenate([a1, b1]), whole)
~~~

The CPU executes operations sequentially and checks equivalence, not GPU
pipeline acceleration. Unequal chunk/stage times create idle slots.
Every stage must also retain or load preceding chunks' KV.
CPP, chunk size, and layer streaming therefore have coupled constraints rather
than independent multiplicative speedups.

<a id="как-идеи-реализованы-в-современных-стеках"></a>

<span id="передача-запроса-в-dynamo-и-sglang"></span>

<span id="how-modern-stacks-realize-the-design"></span>

## Request handoff in Dynamo and SGLang

In [Dynamo v1.5.1 Disaggregated Serving](https://docs.nvidia.com/dynamo/v1.5.1/knowledge-base/concepts/system-architecture/disaggregated-serving),
PrefillRouter selects P, receives continuation metadata, and forwards it in
the D request. KV itself moves directly between executors, not inside the
router's JSON response. NIXL provides a supported transport path; routing does
not require reading each block's contents.

| Backend in that documentation version | Continuation metadata |
|---|---|
| SGLang | bootstrap_info: address, port, and room_id |
| vLLM | kv_transfer_params: block identifiers and remote-worker data |
| TensorRT-LLM | opaque_state: serialized backend-internal state |

Metadata identifies and coordinates a transfer; it does not certify all bytes
are ready. D selection, allocation, and bootstrap can begin early, but attention
must still wait for its K/V dependencies. An asynchronous protocol must
distinguish a prepared request from ready data.

The [v1.5.1 router documentation](https://docs.nvidia.com/dynamo/v1.5.1/knowledge-base/modular-components/router/disaggregated-serving)
disables D prefix-match scoring and does not assume deduplication of transferred
blocks by default. Therefore the abstract option “send only a missing suffix”
cannot be attributed to that path automatically. It also states that after P
dispatch, prefill/handoff errors are returned to the request rather than silently
retried through a unified decode-only path.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/distributed-serving-2026/dynamo-architecture.png]]

*The official Dynamo overview retained from the previous English edition:
frontend/router, disaggregated executors, and transfer/memory components.
Source: [ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo/blob/main/docs/assets/img/architecture.png),
Apache-2.0. Use the versioned protocol above for exact handoff behavior; a general
architecture drawing does not establish backend-specific guarantees.*

“Mooncake” in SGLang configuration normally names a transfer mechanism,
not the original paper's entire Conductor and cache system. Choosing NIXL or
Mooncake alone does not specify admission, allocation timing, or first-token
emission. Reproducible experiments fix engine version, transport backend,
both parallel layouts, and reservation policy.

<a id="когда-разделение-выгодно"></a>

<span id="как-сравнивать-общий-и-раздельный-пулы"></span>

<span id="when-separation-pays-off"></span>

## Comparing unified and disaggregated service

Use the same weights, precision, requests, offered load, and emission policy.
Count all GPUs in both pools, networking, and allocation time. D-only throughput
does not describe service efficiency without P and handoff.

Suppose an eight-GPU unified deployment completes 10 requests/s and 80% meet
both TTFT and ITL targets. Goodput is eight qualifying requests/s, or one per
GPU. A sixteen-GPU split deployment completes 14 requests/s with 95% jointly
meeting the targets: 13.3 qualifying requests/s, but only 0.83125 per GPU.

| Deployment | All GPUs | Completed requests/s | Both SLOs met | Goodput/s | Goodput/GPU |
|---|---:|---:|---:|---:|---:|
| Unified | 8 | 10 | 80% | 8 | 1 |
| Split | 16 | 14 | 95% | 13.3 | 0.83125 |

These are metric examples, not DistServe or Splitwise measurements.
Separate 95% TTFT and 95% ITL rates do not imply 95% joint success.
Check their intersection per request. Different device prices also make GPU
count an incomplete denominator; use allocation cost over the same interval.

Long prefills and strict ITL give a reason to separate phases, but the result
depends on the full request path: P queuing, KV/D waiting, transfer tails,
capacity, rejection rates, and changing input distributions. Low steady load
or a slow network can favor a unified pool with chunked prefill.
Measure p50/p95/p99 and joint SLO attainment, HBM/KV use in both pools, bytes
and transfer tails, prefix reuse/recomputation, burst behavior, failures, and
scale-up/drain time. The
[[58b Benchmarking SLOs and inference operations|next chapter]]
details the measurement procedure.

<span id="практика-и-первоисточники"></span>

<span id="sources-and-further-reading"></span>

## Practice and primary sources

- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol2/inference|Harvard CS249r — Inference]].
- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol2/network_fabrics|Harvard CS249r — Network Fabrics]].
- [[02 Areas/ML & DL/05 Источники/LMCache/LMCache MP mode — transfer paths|LMCache — transfer paths]].
- [[02 Areas/ML & DL/05 Источники/LMCache/KV cache as persistent inference state|LMCache — persistent inference state]].
- Austin et al., [All About Transformer Inference](https://jax-ml.github.io/scaling-book/inference/) — phase costs and distributed placement.
- Zhong et al., [DistServe, OSDI 2024](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf) — §2–4, queues, placement, and transfer.
- Patel et al., [Splitwise, ISCA 2024, v2](https://arxiv.org/abs/2311.18677v2) — §IV, Figures 10–12, Table VI.
- Qin et al., [Mooncake, v1, 2024](https://arxiv.org/abs/2407.00079v1) — §3–6, Conductor, long prefill, and admission.
- NVIDIA, [Dynamo v1.5.1 architecture](https://docs.nvidia.com/dynamo/v1.5.1/knowledge-base/concepts/system-architecture/disaggregated-serving) and [routing](https://docs.nvidia.com/dynamo/v1.5.1/knowledge-base/modular-components/router/disaggregated-serving).
- SGLang, [PD Disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation) — decode allocation snapshot reviewed 20 October 2026.

**Previous:** [[58a Distributed inference and disaggregated serving|Parallelism and collective communication]]

**Next:** [[58b Benchmarking SLOs and inference operations|Benchmarking, SLOs, and inference operations]]
