---
title: "Parallelism and collective communication in LLM inference"
type: textbook-chapter
status: canonical
locale: en
translation_status: current
canonical_revision: "7219c5e6fdef1ca9cd3fa92c50097cfe77a6ad664afa3ac43e9e6d8c117d7625"
translation_reviewed: "2026-10-29"
translation_of: "00 Учебник/14 Inference и оптимизация/58a Распределённый inference и disaggregated serving.md"
last_updated: 2026-10-29
last_verified: 2026-07-23
primary_sources:
  - https://jax-ml.github.io/scaling-book/sharding/
  - https://jax-ml.github.io/scaling-book/inference/
  - https://arxiv.org/abs/1909.08053
  - https://arxiv.org/abs/2104.04473
  - https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html
  - https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html
---

<span id="параллелизм-и-коллективные-операции-в-llm-inference"></span>

# Parallelism and collective communication in LLM inference

Distributed inference addresses two different problems. A model may exceed one
accelerator's memory or take too long to answer one request. Several GPUs must
then cooperate on that request. Alternatively, the model may fit on one device
but one instance cannot handle the incoming traffic. Independent replicas can
serve different requests. Both arrangements use multiple GPUs; their data
movement and effect on single-request latency are different.

For example, 70 billion BF16 parameters occupy roughly 140 GB. They do not fit on
an 80 GB GPU, even before allocating KV. Splitting the weights over four GPUs
leaves roughly 35 GB per device, but the devices must now combine their partial
results. Two independent four-GPU replicas can serve different request sets
without exchanging weights on every generation step.

A *rank* identifies a process within a particular communication group. One GPU
can participate in several groups: the TP group of its pipeline stage and the
pipeline group connecting it to other stages, for example. “Rank 0” without a
group name does not uniquely identify a device.

<span id="от-одного-transformer-блока-к-группе-gpu"></span>

## From one Transformer block to a GPU group

Let $X\in\mathbb{R}^{B\times D}$ contain $B$ tokens of width $D$.
A linear layer computes $Y=XW$. On one device, all three tensors are local.
Splitting the output columns of the weights gives

$$
\begin{aligned}
W&=[W_1\;W_2\;\ldots\;W_p],\\
Y&=[XW_1\;XW_2\;\ldots\;XW_p].
\end{aligned}
$$

Each rank receives the complete $X$, stores $W_i$, and produces a fragment of
$Y$. If the next operation accepts that layout, no gathering is needed. Splitting
the contracting dimension instead produces partial sums that must be added.
These two cases explain most communication around Transformer linear layers.

[How to Scale Your Model](https://jax-ml.github.io/scaling-book/sharding/)
describes both the global tensor shape and its placement on a *device mesh*.
A full copy of $Y$ and a partial sum for $Y$ can have the same local shape
$B\times D$, but only the full copy contains the final values. Treating a partial
sum as a completed layer output changes the function. A column shard differs
from both: it contains final values for only some coordinates.

<span id="коллективные-операции-четыре-преобразования-размещения"></span>

## Collective operations as layout transformations

The full derivation appears in the chapters on
[[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/44a Processes, collectives и DDP|processes and collectives]]
and [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/44c Tensor и sequence parallelism|tensor and sequence parallelism]].
Here we use two processes; the original NCCL figures use four. Reduction means
elementwise addition in the examples. A different reduction, such as maximum,
must be selected explicitly.

<a id="allgather-из-shards-в-полную-копию"></a>

<span id="allgather-собрать-полный-тензор-из-фрагментов"></span>

### AllGather: shards become complete copies

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/distributed-serving-2026/nccl-allgather.png]]

*Official NVIDIA NCCL diagram: every rank contributes a fragment and receives
their concatenation in rank order. Source:
[Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html),
BSD-licensed documentation.*

Suppose rank 0 stores [1, 2] and rank 1 stores [3, 4]. Both receive
[1, 2, 3, 4]. No values are added: the fragments occupy different positions in
the result, ordered by rank within the group.

This is useful when token rows are sharded but the next TP linear implementation
expects all rows. The local data volume grows by the group size, so the gathered
buffer also belongs in the peak-memory calculation.

<span id="reducescatter-просуммировать-и-оставить-результат-разделённым"></span>

### ReduceScatter: sum and keep the result sharded

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/distributed-serving-2026/nccl-reducescatter.png]]

*ReduceScatter reduces matching elements and leaves one output shard on each
rank. Source: official [NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html).*

Now rank 0 supplies [1, 2, 3, 4] and rank 1 supplies [10, 20, 30, 40].
The elementwise sum is [11, 22, 33, 44]. ReduceScatter leaves [11, 22] on
rank 0 and [33, 44] on rank 1. Each process has finished values for only its
positions. An operator that accepts that layout can continue without replicating
the complete result.

<span id="allreduce-полная-сумма-на-каждом-rank"></span>

### AllReduce: the complete reduction on every rank

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/distributed-serving-2026/nccl-allreduce.png]]

*AllReduce returns the reduced result to every rank. ReduceScatter followed by
AllGather has the same result; the library need not use that implementation.
Source: official [NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html).*

For the same inputs, AllReduce returns [11, 22, 33, 44] to both processes.
Classic Megatron TP uses it to add partial linear outputs before the residual
connection. Replacing it with AllGather would concatenate partial sums rather
than add them.

The following calculation reproduces these results in one process. It sends
nothing between GPUs.

~~~python
import numpy as np

shards = [np.array([1, 2]), np.array([3, 4])]
gathered = np.concatenate(shards)
partials = np.array([[1, 2, 3, 4], [10, 20, 30, 40]])
reduced = partials.sum(axis=0)
scattered = np.split(reduced, 2)
assert gathered.tolist() == [1, 2, 3, 4]
assert reduced.tolist() == [11, 22, 33, 44]
assert [a.tolist() for a in scattered] == [[11, 22], [33, 44]]
assert np.array_equal(np.concatenate(scattered), reduced)
~~~

For a ring AllReduce with $M$ bytes per rank and $p$ ranks, ReduceScatter and
AllGather each take $p-1$ steps. Each step transfers a fragment of $M/p$ bytes.
The corresponding rough model is

$$
T_{AR}\approx 2(p-1)\alpha+
2\frac{p-1}{p}\frac{M}{\beta},
$$

where $\alpha$ is step latency and $\beta$ is effective bandwidth.
NCCL chooses algorithms according to topology and message size, so this is not
a measured performance prediction. It separates latency-sensitive small decode
messages from bandwidth-sensitive large prefill messages.

Take $p=8$, $\alpha=2$ microseconds, and $\beta=50\cdot10^9$ bytes/s.
With width 8192 and BF16, $M=B\cdot8192\cdot2$ bytes.

| Pass | Tokens $B$ | Message | Latency term | Bandwidth term | Total |
|---|---:|---:|---:|---:|---:|
| One-request decode | 1 | 16 KiB | 28 us | 0.573 us | 28.573 us |
| Long prefill | 8192 | 128 MiB | 28 us | 4.698 ms | 4.726 ms |

Two such reductions in each of 80 layers would require about 4.57 ms per decode
step or 756 ms per prefill without overlap. Useful overlap requires suitable
dependencies and an implementation that exploits them; a large GEMM alone
does not guarantee it. Reducing $p$ shortens the ring but increases local weight
storage and matrix work. It does not change the number of AllReduce calls per
layer in this arrangement.

<a id="alltoall-персональная-перестановка-между-всеми-ranks"></a>

<span id="alltoall-передать-каждому-процессу-свой-фрагмент"></span>

<span id="alltoall-a-personalized-permutation-among-ranks"></span>

### AllToAll: a different fragment for each destination

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/distributed-serving-2026/nccl-alltoall.png]]

*Each rank sends a distinct fragment to each destination and receives one from
each source. Source: official [NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html).*

Suppose rank 0 prepares 10 for itself and 11 for rank 1, while rank 1 prepares
20 for rank 0 and 21 for itself. Rank 0 receives [10, 20]; rank 1 receives
[11, 21]. Rows of the send matrix become columns at the receivers. There is
neither a reduction nor a full copy of all values.

In MoE, dispatch sends token vectors to expert owners and the reverse exchange
returns their outputs. Token-dependent routing can give destinations different
message sizes. Distributing equal numbers of experts across GPUs does not
ensure equal work.

Pipeline parallelism instead primarily uses point-to-point send/receive between
adjacent stages. The receiver cannot compute its layers until the previous
stage's activations arrive.

Every collective participant must agree on the group, call order, buffer sizes,
element type, and reduction operation. A skipped AllReduce or incompatible
buffer can leave the others waiting or produce invalid results; these requirements
are documented by [NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html).

<span id="data-parallelism-независимые-реплики-сервиса"></span>

## Data parallelism: independent serving replicas

Training DP uses identical weights, different minibatches, and synchronized
gradients. Dense inference has no gradients. A serving replica contains one
complete executable model instance, which may itself use TP or PP; a router
assigns whole requests to it.

If one replica sustains 100 requests/s at the required latency, two could
sustain about 200 under unchanged batching efficiency, sufficient demand, and no
shared bottleneck. Uneven request lengths, a common frontend or storage service,
and lost prefix hits can reduce that gain. Replication does not shrink one
instance's weights or shorten one forward pass. Independent dense replicas
need no per-layer synchronization.

Each replica owns KV state. A busy replica with the required long prefix can
finish a request sooner than an idle replica that must recompute it.
Routing compares queue delay with avoided work, not different model weights.
See [[55b Scheduling — continuous batching, chunked prefill, and prefix caching|batching, scheduling, and prefix caching]].

In MoE, *attention DP* may instead mean replicated attention with shared,
distributed experts. These groups communicate in MoE layers and are not
independent service copies. Their DP meaning must be specified alongside EP
groups.

<span id="tensor-parallelism-матрица-разделена-внутри-слоя"></span>

## Tensor parallelism: partitioning a layer's matrices

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/distributed/tensor-parallel-split.svg]]

*Vijay Janapa Reddi et al., Harvard ML Systems, Vol. II,
[Column Parallel / Row Parallel](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/distributed_training/images/svg/tensor-parallel-split.svg),
CC BY-NC-SA 4.0. The left partitions output coordinates; the right adds
contributions to the same coordinates. Concatenation on the left denotes the
global output: a matching row-parallel projection does not require a physical
gather. Backward annotations describe training, not inference.*

Megatron-LM splits the first MLP matrix by output coordinates and the second
by input coordinates. Inference uses the same forward computation without
backward or gradient exchanges. Consider

$$
\begin{aligned}
H&=\phi(XW_{up}+b_{up}),\\
Y&=HW_{down}+b_{out}.
\end{aligned}
$$

Use two input rows $(1,2)$ and $(3,4)$ and an intermediate width of four.
Rank 0 owns columns 0–1 of $W_{up}$; rank 1 owns columns 2–3.
Both receive the full $X$. Coordinates of $b_{up}$ follow the columns.
ReLU makes the hand calculation simple; like GeLU, it acts elementwise and
therefore obeys the same placement rule.

~~~python
import numpy as np

X = np.array([[1., 2.], [3., 4.]])
Wup = np.array([[1., 0., 1., 2.],
                [0., 1., 1., -1.]])
bup = np.array([0., .5, -.5, 1.])
Wdown = np.array([[1., 0.], [0., 1.],
                  [1., 1.], [2., -1.]])
bout = np.array([.1, -.2])

def tp_mlp(x, up, up_bias, down, out_bias, degree=2):
    assert up.shape[1] % degree == 0
    columns = np.split(up, degree, axis=1)
    biases = np.split(up_bias, degree)
    rows = np.split(down, degree, axis=0)
    hidden = [np.maximum(x @ w + b, 0.)
              for w, b in zip(columns, biases)]
    partials = [h @ w for h, w in zip(hidden, rows)]
    return hidden, partials, np.sum(partials, axis=0) + out_bias

hidden, partials, Y = tp_mlp(X, Wup, bup, Wdown, bout)
assert np.allclose(hidden[0], [[1., 2.5], [3., 4.5]])
assert np.allclose(hidden[1], [[2.5, 1.], [6.5, 3.]])
assert np.allclose(partials[0], [[1., 2.5], [3., 4.5]])
assert np.allclose(partials[1], [[4.5, 1.5], [12.5, 3.5]])
assert np.allclose(Y, [[5.6, 3.8], [15.6, 7.8]])
dense = np.maximum(X @ Wup + bup, 0.) @ Wdown + bout
assert np.allclose(Y, dense)
~~~

The first multiplication leaves $H_0=[(1,2.5),(3,4.5)]$ on rank 0 and
$H_1=[(2.5,1),(6.5,3)]$ on rank 1. ReLU needs no neighbor coordinates.
The second multiplication produces two $2\times2$ arrays contributing to the
same outputs. For the first row, they are $(1,2.5)$ and $(4.5,1.5)$.
Their sum is $(5.5,4)$; adding $b_{out}=(0.1,-0.2)$ once gives $(5.6,3.8)$.

No full $H=[H_0,H_1]$ was gathered: each slice of $W_{down}$ was already next
to its input coordinates. Had the first multiplication split the contracting
input dimension, its partial outputs would need to be added *before* the
nonlinearity. In general, $\phi(a+b)\ne\phi(a)+\phi(b)$.
[Megatron-LM, §3 and Figure 3](https://arxiv.org/abs/1909.08053v4) explains this choice.

Adding the output bias on every rank before AllReduce would add it $p$ times.
The residual has the same issue. Add both after reduction or use an equivalent
scheme that counts each contribution exactly once.

<span id="swiglu-и-attention"></span>

### SwiGLU and attention

SwiGLU needs matching output-coordinate partitions for $W_{gate}$ and $W_{up}$.
Rank $i$ computes

$$
\begin{aligned}
H_i&=\operatorname{SiLU}(XW_{gate,i})\odot(XW_{up,i}),\\
Y&=\sum_i H_iW_{down,i}.
\end{aligned}
$$

Both factors are local. Mismatched coordinate partitions would require a
permutation before their elementwise product. A gated MLP does not remove the
need for compatible placement.

In multi-head attention, heads independently calculate scores, softmax, and
weighted value sums. Each process can own a subset of heads and their Q/K/V
projections. Head outputs feed an input-partitioned $W_O$, and ranks add its
partial results. The classic Megatron Transformer forward pass has one
AllReduce for attention and one for the MLP. SP, specialized distributed
kernels, or a different architecture can change this sequence.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/tensor-sequence-review-2026-10/hf-attention-tp.png]]

*Nouamane Tazi et al., Hugging Face,
[The Ultra-Scale Playbook: Tensor Parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook/blob/c93890dd4b22356ee18ed9b863e6f2d0909729bc/assets/images/tp_full_diagram.png),
Apache-2.0. Both rows receive the same $X$ but compute different heads.
$B_1,B_2$ are slices of $W_O$: the sum combines $Y_1B_1$ and $Y_2B_2$, not the
different heads themselves. Training dropout is disabled in ordinary inference.
The complete original diagram is retained.*

Norm parameters and other small tensors may remain replicated. Dividing total
parameters by TP therefore estimates rather than exactly predicts per-rank
memory. GQA imposes another constraint: KV heads can be fewer than query heads.
Beyond the KV-head count, head-wise sharding requires replication or another
partition axis.

<span id="почему-tp-по-разному-ведёт-себя-на-prefill-и-decode"></span>

### Why prefill and decode respond differently to TP

Prefill gives matrix multiplications many token rows. Splitting the work helps
until communication and smaller, less efficient local kernels consume the gain.
Decode adds one row per active request; its shorter steps expose launch and
collective latency.

TP can still accelerate memory-bound decode: multiple GPUs read their weight
shards concurrently from separate HBM channels. Activation exchange after the
projections is the cost. [How to Scale Your Model: Generation](https://jax-ml.github.io/scaling-book/inference/#generation)
examines this regime. Repeatedly gathering weights layer by layer, as in an
FSDP execution, can cost much more than exchanging activations. That is a
restriction of a particular decode regime, not a ban on every FSDP inference
configuration or phase.

Compare memory fit for weights, buffers, and target KV; measured TP1/2/4/8
step times; and the links crossed by each collective. Then compare throughput
per GPU at the same SLO. TP8 can reduce one request's latency while serving
fewer requests than two TP4 replicas on the same eight GPUs.

<span id="pipeline-parallelism-слои-разделены-по-глубине"></span>

## Pipeline parallelism: partitioning depth

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/distributed-inference-review-2026-10/harvard-pipeline-panels.svg]]

*Reddi et al., Harvard ML Systems, Vol. II,
[Pipeline Parallelism: Request Flow](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/inference/images/svg/pipeline-parallel-routing.svg),
CC BY-NC-SA 4.0. The source's two upper panels show one request's sequential
path and different requests occupying the stages. All stages, arrows, and
schedule rows are retained; the lower throughput-formula block is excluded.
For equal stages, steady throughput is one microbatch per stage time, not four.*

PP places consecutive layers on successive stages. An 80-layer model with PP4
might assign layers 0–19, 20–39, 40–59, and 60–79. Each stage receives
activations, computes its layers, and passes the result onward. Without TP
inside a stage, device transfers occur at stage boundaries rather than after
every linear projection.

Let each of four stages take 4 ms per microbatch, including the transfer cost
already accounted for in that estimate. Three independent microbatches A, B,
and C have the following schedule:

| Stage | 0–4 | 4–8 | 8–12 | 12–16 | 16–20 | 20–24 ms |
|---|---|---|---|---|---|---|
| 0 | A | B | C | · | · | · |
| 1 | · | A | B | C | · | · |
| 2 | · | · | A | B | C | · |
| 3 | · | · | · | A | B | C |

They finish at 16, 20, and 24 ms from the schedule's start. Each spends 16 ms
from entering stage 0 to leaving stage 3. Different stages process *different*
microbatches concurrently. With $m$ microbatches, $p$ equal stages, and stage
time $\tau$, completion takes $(m+p-1)\tau$. The occupied fraction of the
schedule is $m/(m+p-1)$, or $3/6=50\%$ here. Empty fill/drain slots form the
*pipeline bubble*.

~~~python
def pipeline_finish(stage_times, microbatches):
    finish = [[0.] * microbatches for _ in stage_times]
    for stage, duration in enumerate(stage_times):
        for batch in range(microbatches):
            input_ready = finish[stage-1][batch] if stage else 0.
            stage_free = finish[stage][batch-1] if batch else 0.
            finish[stage][batch] = max(input_ready, stage_free) + duration
    return finish

finish = pipeline_finish([4., 4., 4., 4.], 3)
assert finish[-1] == [16., 20., 24.]
assert 3 / (3 + 4 - 1) == .5
~~~

Autoregressive decode adds a dependency across passes of the same request.
After A completes all stages, sampling must choose its next token before the
next pass can enter stage 0. Four unknown future tokens of A cannot fill the
four stages. Other independent requests or ready microbatches supply that work.
Varying prompt lengths, completions, and scheduler limits make real schedules
less regular.

Equal layer counts need not produce equal stage times. Embeddings, the output
head, and token-dependent MoE work can unbalance them. With stage times
2, 6, 3, and 5 ms, the first microbatch still takes 16 ms, but later completions
cannot occur faster than the 6 ms bottleneck. The average 4 ms stage time does
not determine throughput. Measure both prefill and decode when partitioning.

A TP8×PP4 instance has four groups of eight GPUs: 32 GPUs for one executable
replica, not 32 independent servers. Comparing it with smaller independent
replicas is meaningful only if those replicas fit the same model and target KV.

PP also partitions KV by layer. Stage 0 owns K/V for layers 0–19, stage 1 for
20–39, and so forth. Each stage creates and extends its own entries; it need
not send the entire accumulated cache to the next stage at every token.
However, the request needs coherent state at all stages. Losing one stage
prevents ordinary continuation even if the others remain healthy.

<span id="sequence-parallelism-и-context-parallelism--не-синонимы"></span>

## Sequence parallelism and context parallelism are not synonyms

In Megatron terminology, *sequence parallelism* reduces replicated activations
around token-local operators. LayerNorm uses all $D$ coordinates of one token
but does not use other tokens. With four rows split into two pairs, each process
can normalize its two full-width rows locally. Splitting the width itself would
make that local LayerNorm incorrect.

AllGather then collects normalized rows for the TP linear region. The expanding
and contracting projections work as above. ReduceScatter after the second
projection adds partial results and returns only the appropriate rows to each
rank, so the next Norm again sees half the tokens rather than replicated outputs.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/tensor-sequence-review-2026-10/hf-tp-sp-regions.png]]

*Tazi et al.,
[The Ultra-Scale Playbook: Sequence Parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook/blob/c93890dd4b22356ee18ed9b863e6f2d0909729bc/assets/images/tp_sp_diagram.png),
Apache-2.0. Left: activations between TP regions are replicated; forward $f$
leaves complete inputs unchanged and $f^*$ AllReduces partial outputs.
Right: $g$ AllGathers rows and $g^*$ ReduceScatters them. Norm and residuals
operate on sharded, full-width rows. Both original panels are retained.*

The following continues the earlier TP-MLP example and uses its matrices and
function.

~~~python
X4 = np.array([[1., 2.], [3., 4.], [2., -1.], [-2., 1.]])
local_rows = np.split(X4, 2, axis=0)
def layer_norm(x, eps=1e-5):
    return (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(
        x.var(axis=-1, keepdims=True) + eps)

normalized = np.concatenate([layer_norm(x) for x in local_rows])  # AllGather
_, partials, _ = tp_mlp(normalized, Wup, bup, Wdown, bout)
rows_after_rs = np.split(np.sum(partials, axis=0), 2, axis=0)
result = np.concatenate([rows + bout for rows in rows_after_rs])
reference = np.maximum(layer_norm(X4) @ Wup + bup, 0.) @ Wdown + bout
assert np.allclose(result, reference)
~~~

Each rank's output has two rows, not four. Norm, the residual, and subsequent
operators must agree on row order. This is one ordinary TP/SP layout, not every
possible distributed linear implementation. Dropout shown in training diagrams
is disabled for ordinary inference.

*Context parallelism* also shards the attention input by token position.
Long prefill can then distribute sequence activations. Local queries still need
all permitted keys, including remote ones. The implementation must exchange
K/V blocks or combine attention results from different key partitions.
Causal masks and global positions must remain correct regardless of transfer order.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/distributed-serving-2026/megatron-context-parallel-overview.png]]

*NVIDIA,
[Megatron Core Context Parallelism](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html),
Figure 1. GPU0/1 process one sequence shard with TP2; GPU2/3 process the other.
CP connects GPU0↔GPU2 and GPU1↔GPU3. Yellow AG/RS belong to TP/SP, blue operations
to CP attention. Slash-separated labels describe forward/backward training;
inference executes the forward path.*

Megatron's documented CP implementation can exchange blocks through ring
point-to-point transfers. Blue AG here means obtaining K/V from other context
shards, not gathering all weights. A collective name alone does not make this
equivalent to FSDP.

<span id="как-объединить-attention-по-двум-частям-kv"></span>

### Combining attention over two KV partitions

During decode, the new query has length one while historical KV can be large.
Send the same query to two processes that own different KV positions.
Let the four scores be $\log1,\log3,\log2,\log4$, with values 10, 20, 30, 40.
Full attention gives

$$
o=\frac{10+60+60+160}{1+3+2+4}=29.
$$

The first partition produces $70/4=17.5$, the second $220/6\approx36.667$.
Their simple average, 27.083, is wrong: their unnormalized probability masses
are four and six, not equal. Combining outputs must preserve these denominators.

For stability, process $i$ stores a maximum $m_i$, a shifted exponential sum
$\ell_i$, and an unnormalized weighted value sum $u_i$:

$$
\begin{aligned}
\ell_i&=\sum_j\exp(s_j-m_i),\\
u_i&=\sum_j\exp(s_j-m_i)v_j.
\end{aligned}
$$

Merge them by

$$
\begin{aligned}
m&=\max_i m_i,\\
a_i&=\exp(m_i-m),\\
\ell&=\sum_i a_i\ell_i,\\
u&=\sum_i a_i u_i,\\
o&=u/\ell.
\end{aligned}
$$

These are the same softmax statistics combined by [[56 FlashAttention|FlashAttention]]
over successive blocks, now held on different processes. A MAX reduction gives
the common maximum; rescaled local statistics can then be summed. A specialized
kernel may organize the communication differently.

~~~python
def merge_attention(score_parts, value_parts):
    width = value_parts[0].shape[1]
    stats = []
    for scores, values in zip(score_parts, value_parts):
        if not len(scores) or not np.isfinite(scores).any():
            stats.append((-np.inf, 0., np.zeros(width)))
            continue
        m = np.max(scores)
        weights = np.exp(scores - m)
        stats.append((m, weights.sum(), weights @ values))
    global_m = max(s[0] for s in stats)
    if not np.isfinite(global_m):
        raise ValueError('query has no unmasked keys')
    total, numerator = 0., np.zeros(width)
    for m, denominator, weighted in stats:
        if denominator == 0:
            continue
        scale = np.exp(m - global_m)
        total += scale * denominator
        numerator += scale * weighted
    return numerator / total

scores = 1000. + np.log([1., 3., 2., 4.])
values = np.array([[10.], [20.], [30.], [40.]])
answer = merge_attention(np.split(scores, 2), np.split(values, 2))
assert np.allclose(answer, [29.])
~~~

Adding 1000 to every score preserves softmax but makes direct exponentiation
overflow at ordinary precision. Empty or fully masked partitions need separate
handling to avoid subtracting negative infinity from itself. Here values have
one coordinate; in the model, $u_i$ has the head's value width.

Context-sharded KV reduces local memory but requires attention coordination at
every token. For short contexts, exchange latency can exceed saved KV-read time.
This is one distributed-decode scheme, not a claim that every Megatron CP
implementation uses this exact schedule.

<span id="expert-parallelism-параметры-распределены-токены-движутся"></span>

## Expert parallelism: parameters stay, tokens move

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/expert-hybrid-review-2026-10/edls-moe-expert-parallel.png]]

*Efficient DL Systems,
[Week 6, PDF p.130](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week06_dl_arithmetic/lecture.pdf#page=130),
MIT. GPU 1 groups assignments for its expert and computes its full matrices.
B1–B4 identify row sources; they do not require sending each entire batch.
Both SwiGLU expanding projections and the contracting projection belong to
one expert.*

A dense MLP applies the same parameters to every token. An MoE router chooses
$k$ out of $E$ experts per token. EP must deliver the input vector to every
selected expert and return its output. A token ID alone is insufficient:
the expert consumes the width-$D$ activation produced by preceding layers.

Let rank 0 own E0/E1 and rank 1 own E2/E3. Rank 0 initially holds t0/t1;
rank 1 holds t2/t3. Use these top-2 assignments, with weights already normalized
over selected experts:

| Token | Origin | Input $x$ | Experts and weights |
|---|---:|---|---|
| t0 | 0 | (1, 2) | E0: 0.75; E2: 0.25 |
| t1 | 0 | (3, 4) | E2: 0.40; E3: 0.60 |
| t2 | 1 | (5, 6) | E1: 0.50; E2: 0.50 |
| t3 | 1 | (7, 8) | E0: 0.25; E1: 0.75 |

There are eight assignments. Rank 0 keeps t0→E0 and sends t0→E2,
t1→E2, t1→E3 to rank 1. Rank 1 keeps t2→E2 and sends t2→E1,
t3→E0, t3→E1 back. Rows of the following matrix are sources; columns are
destinations:

$$
N_{send}=\begin{pmatrix}1&3\\3&1\end{pmatrix}.
$$

Each receiver groups vectors by local expert. E0 gets t0/t3, E1 gets t2/t3,
E2 gets t0/t1/t2, and E3 gets only t1: loads [2, 2, 3, 1].
Equal expert counts did not produce equal matrix work.

Replace expert MLPs with E0$(x)=x$, E1$(x)=2x$, E2$(x)=3x$,
and E3$(x)=4x$ to check the return path. Token t0 receives
$0.75x+0.25\cdot3x=1.5x=(1.5,3)$.
Token t1 receives $0.4\cdot3x+0.6\cdot4x=3.6x=(10.8,14.4)$.
Outputs must return to their original tokens even though expert grouping
changed their execution order.

~~~python
def expert_dispatch(x, origin, choices, gates, owner, expert_weights):
    ranks = int(max(owner.max(), origin.max())) + 1
    send_counts = np.zeros((ranks, ranks), dtype=int)
    received = [[] for _ in range(ranks)]
    for token in range(len(x)):
        for expert, gate in zip(choices[token], gates[token]):
            src, dst = int(origin[token]), int(owner[expert])
            send_counts[src, dst] += 1
            received[dst].append((src, token, expert, gate, x[token]))
    returned = [[] for _ in range(ranks)]
    for dst, records in enumerate(received):
        for src, token, expert, gate, vector in sorted(records, key=lambda r: r[2]):
            output = vector @ expert_weights[expert]
            returned[src].append((token, gate, output))
    result = np.zeros_like(x)
    for records in returned:
        for token, gate, output in records:
            result[token] += gate * output
    return result, send_counts

x = np.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
origin = np.array([0, 0, 1, 1])
choices = np.array([[0, 2], [2, 3], [1, 2], [0, 1]])
gates = np.array([[.75, .25], [.4, .6], [.5, .5], [.25, .75]])
owner = np.array([0, 0, 1, 1])
expert_weights = np.array([np.eye(2) * s for s in [1., 2., 3., 4.]])
outputs, counts = expert_dispatch(x, origin, choices, gates, owner, expert_weights)
assert counts.tolist() == [[1, 3], [3, 1]]
assert np.allclose(outputs, [[1.5, 3.], [10.8, 14.4],
                             [12.5, 15.], [12.25, 14.]])
~~~

This CPU code models packing and order restoration, not GPU communication.
Real implementations use communication kernels for dispatch and return and may
fuse them with other operations. The router weight is applied exactly once,
after return. Moving it to the input of an arbitrary nonlinear expert changes
the function: generally $E(gx)\ne gE(x)$.

Equal-size AllToAll messages would pad each of the four matrix entries to
three assignments, allocating 12 slots for eight assignments. Variable sizes,
compaction, or padding have different memory and latency costs. Six assignments
cross ranks here. At width two and BF16, input vectors carry 24 bytes and
outputs another 24, excluding metadata and protocol overhead. At width 8192,
the same routing carries 4096 times more activation data.

All resident experts determine parameter memory; selected experts determine
per-token computation; routing determines traffic and load imbalance.
Dropping assignments at capacity changes the computed function. Replicating the
same expert weights and redirecting assignments can preserve it but costs memory
and requires consistent placement. See the
[[02 Mixture of Experts — routing, capacity и serving|MoE chapter]]
for training-time load balancing.

EP can combine with TP within each expert. Dispatch then targets an expert
group, whose members execute a sharded MLP. Attention DP, EP, and expert TP can
reuse the same physical processes in different groupings; their degrees cannot
be multiplied blindly.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/expert-hybrid-review-2026-10/edls-moe-tensor-parallel.png]]

*The same course, [PDF p.129](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week06_dl_arithmetic/lecture.pdf#page=129),
MIT. Under TP, GPU 1 computes matrix fragments from several experts; under EP
in the preceding figure, separate owners hold full expert matrices.
Gray and green areas are matrix partitions, not complete expert replicas.
The computation diagram is retained; slide title and outer margins are removed.*

<a id="как-топология-превращает-алгоритм-в-производительность"></a>

<span id="топология-сети-и-размещение-процессов"></span>

<span id="topology-turns-an-algorithm-into-performance"></span>

## Network topology and process placement

Local HBM, NVLink/NVSwitch, and inter-node NIC transfers use different paths.
A TP group spanning nodes sends its frequent layer reductions across the
network. Aligning a node boundary with a PP boundary instead transfers
activations between model segments. EP can exchange vectors among many rank
pairs at every MoE layer. Placement must account for these actual paths,
including GPU-to-NIC access and contention on shared links.

When memory capacity allows, TP often stays inside a fast NVLink/NVSwitch
domain, EP uses a domain with sufficient aggregate cross-partition bandwidth,
PP crosses nodes when activation transfer fits its budget, and independent
replicas occupy separate nodes or failure domains. These are placement choices,
not universal rules.

For example, two eight-GPU nodes can host independent TP8 replicas. The first
replica's AllReduce never touches the second node. One TP16 instance uses both
nodes in every such exchange. The GPU count is unchanged, but inter-node work
is not. Link names cannot predict time without message sizes and measurements
on the selected groups.

Measure local kernels, representative collectives, layer schedules, and complete
request handling separately. Low compute utilization can mean waiting for a
reduction, remote expert, or previous stage. More incoming requests do not remove
that dependency, though independent work may sometimes overlap it.

<a id="как-выбирать-композицию-dptpppcpep"></a>

<span id="группы-процессов-при-сочетании-видов-параллелизма"></span>

<span id="choosing-a-dptpppcpep-composition"></span>

## Process groups in hybrid parallelism

For a dense model with independent DP, TP, and PP axes, the rank count is
$DP\cdot TP\cdot PP$. DP2×TP4×PP2 uses 16 GPUs: two replicas, two stages per
replica, and four TP ranks per stage. One possible global numbering is

$$
r=((d\cdot PP)+s)\cdot TP+t.
$$

Replica $d=0$ has TP groups [0,1,2,3] and [4,5,6,7]; pipeline connections
join corresponding ranks across these stages. Replica $d=1$ occupies [8,…,15].
This numbering illustrates the groups; it is not a required engine convention.

For each candidate, record sharded and replicated weights, activation and KV
layouts, the dense/MoE collective sequence, process groups and node crossings,
available memory and token limits, and measured TTFT/ITL/throughput at the same
SLO. CP and EP need their own group description. EP may occupy ranks treated
as DP by attention. The product DP×TP×PP×CP×EP is not a universal cluster-size
formula.

<a id="небольшой-пример-чтения-конфигурации"></a>

<span id="расчёт-памяти-tp4-tp8-и-tp16"></span>

## Memory calculation: TP4, TP8, and TP16

Return to a hypothetical 70B dense model on sixteen 80 GB GPUs. BF16 weights
occupy 140 GB; assume they are fully shardable over TP. Reserve another 8 GB per
GPU for workspaces and other overhead. GB here means $10^9$ bytes.

For KV, use 80 layers, 64 query heads, eight KV heads, head dimension 128,
BF16, and 8192 positions. With $L$ layers, $S$ positions, $n_{KV}$ KV heads,
head width $d_h$, and $b$ bytes per element,

$$
\begin{aligned}
M_{KV}&=2LSn_{KV}d_hb\\
&=2\cdot80\cdot8192\cdot8\cdot128\cdot2\\
&=2\,684\,354\,560\text{ bytes}\\
&=2.5\text{ GiB}.
\end{aligned}
$$

The first factor two accounts for K and V; BF16 supplies the final two bytes.
TP4 leaves two KV heads per rank; TP8 leaves one. For TP16, assume each KV head
is replicated across two ranks holding different query heads. Per-rank KV
therefore stops shrinking beyond TP8. Request/context sharding, discussed next,
is not included in this table.

| Placement | Weights/GPU, GB | KV budget/GPU, GB | KV/request/GPU | Requests/replica, at most | Across 16 GPUs, at most |
|---|---:|---:|---:|---:|---:|
| TP4×DP4 | 35 | 37 | 640 MiB | 55 | 220 |
| TP8×DP2 | 17.5 | 54.5 | 320 MiB | 162 | 324 |
| TP16×DP1 | 8.75 | 63.25 | 320 MiB | 188 | 188 |

~~~python
def kv_memory_bound(tp, dp, kv_heads=8, context=8192):
    assert tp * dp == 16
    local_heads = max(1, kv_heads // tp)
    local_kv = 2 * 80 * context * local_heads * 128 * 2
    available = 80e9 - 140e9 / tp - 8e9
    per_replica = int(available // local_kv)
    return local_kv, per_replica, dp * per_replica

assert kv_memory_bound(4, 4) == (671088640, 55, 220)
assert kv_memory_bound(8, 2) == (335544320, 162, 324)
assert kv_memory_bound(16, 1) == (335544320, 188, 188)
~~~

These are bounds under the stated memory model, not measured batch sizes or
throughputs. They exclude future context growth, fragmentation, unequal lengths,
kernel limits, and SLO headroom beyond the assumed 8 GB. A 162-request TP8 batch
might fit but have unacceptable step time. The table also does not prove TP8
faster than TP4; that requires the same serving workload on both.

<span id="как-избежать-репликации-kv-сверх-числа-голов"></span>

### Avoiding KV replication beyond the head count

[How to Scale Your Model: Sharding the KV Cache](https://jax-ml.github.io/scaling-book/inference/#sharding-the-kv-cache)
describes sharding by heads first, then by requests. Small query activations and
attention outputs move instead of the large historical cache.

Take two processes, requests A/B, and one shared KV head for four query heads.
After Q projection, rank 0 holds q0/q1 of both requests; rank 1 holds q2/q3.
Store all KV for A on rank 0 and B on rank 1. Rank 0 sends B's q0/q1 to
rank 1; rank 1 sends A's q2/q3 to rank 0. Each now has all query heads and
local KV for its assigned request.

The reverse permutation sends heads 0–1 of both outputs back to rank 0 and
heads 2–3 back to rank 1. Ordinary TP output projection follows. Two extra
AllToAll operations avoid KV replication. A second mesh axis can also shard KV
heads.

The following checks only these Q/output permutations. Local attention uses
the full cache of each assigned request; restored outputs must equal the
unsharded computation.

~~~python
def local_attention(q, k, v):
    scores = np.einsum('bhd,bsd->bhs', q, k) / np.sqrt(q.shape[-1])
    p = np.exp(scores - scores.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    return np.einsum('bhs,bsd->bhd', p, v)

rng = np.random.default_rng(42)
Q = rng.normal(size=(2, 4, 3))
K = rng.normal(size=(2, 5, 3))
V = rng.normal(size=(2, 5, 3))
q_by_head = np.split(Q, 2, axis=1)
q_by_request = [np.concatenate([part[b:b+1] for part in q_by_head], axis=1)
                for b in range(2)]
out_by_request = [local_attention(q, K[b:b+1], V[b:b+1])
                  for b, q in enumerate(q_by_request)]
out_by_head = [np.concatenate([out[:, h:h+2] for out in out_by_request], axis=0)
               for h in [0, 2]]
restored = np.concatenate(out_by_head, axis=1)
assert np.allclose(restored, local_attention(Q, K, V))
~~~

With one request, the batch cannot be divided among several processes. The
remaining choices include replication or context-sharded KV with merged
attention. KV placement depends on the numbers of heads and independent
requests as well as context length; TP degree alone does not describe it.

<a id="граница-этой-главы"></a>

<span id="разделение-prefill-и-decode"></span>

## Separating prefill from decode

The same distributed replica can run both phases, or separate pools can own
prefill and decode. The latter must transfer the produced KV, sometimes
resharding it. TP specifies computation *within* a pool; it does not decide
when to transfer state or reserve receiver memory. The next chapter derives
those dependencies.

<span id="практика-и-первоисточники"></span>

<span id="sources-and-further-reading"></span>

## Practice and primary sources

- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol2/inference|Harvard CS249r — Inference]].
- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol2/collective_communication|Harvard CS249r — Collective Communication]].
- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol2/network_fabrics|Harvard CS249r — Network Fabrics]].
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week08_inference_software/lecture.pdf|Efficient DL Systems — distributed inference]].
- Austin et al., [Sharded Matrices](https://jax-ml.github.io/scaling-book/sharding/) — deriving collectives from distributed tensor layouts.
- Austin et al., [All About Transformer Inference](https://jax-ml.github.io/scaling-book/inference/) — prefill, decode, and KV-cache sharding with roofline analysis.
- Shoeybi et al., [Megatron-LM](https://arxiv.org/abs/1909.08053) — the original tensor-parallel Transformer.
- Narayanan et al., [Efficient Large-Scale Language Model Training Using Megatron-LM](https://arxiv.org/abs/2104.04473) — composing tensor, pipeline, and data parallelism.
- NVIDIA, [Megatron Core Parallelism Strategies](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html) — DP, TP, PP, CP, and EP terminology.
- NVIDIA, [NCCL Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) — precise operation definitions.

**Previous:** [[58 Speculative decoding|Speculative decoding]]

**Next:** [[58a2 Disaggregated prefill and decode serving|Disaggregated prefill and decode serving]]
