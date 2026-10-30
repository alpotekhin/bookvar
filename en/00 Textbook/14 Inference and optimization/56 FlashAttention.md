---
title: "FlashAttention: exact attention with less memory traffic"
type: textbook-chapter
status: canonical
last_updated: 2026-10-19
last_verified: 2026-10-14
source_unit_id:
  - lecture-05-performance-recap-transition
  - lecture-05-flashattention-derivation
  - lecture-05-recap
  - assignment-02-task-pytorch-attention
  - assignment-02-flashattention-online-softmax
  - assignment-02-task-flash-forward
  - assignment-02-flashattention-tests
  - assignment-02-task-flash-backward
primary_sources:
  - https://arxiv.org/abs/2205.14135
  - https://arxiv.org/abs/2307.08691
  - https://arxiv.org/html/2407.08608v1
  - https://crfm.stanford.edu/2023/07/17/flash2.html
  - https://github.com/Dao-AILab/flash-attention
  - https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
  - https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html
locale: en
translation_status: current
canonical_revision: "15ba654d059b1ccd35bd07c6b041e405bef2f99d74ee03f672c205263b7589a1"
translation_reviewed: "2026-10-29"
translation_of: "00 Учебник/14 Inference и оптимизация/56 FlashAttention.md"
---

<a id="flashattention-точное-внимание-с-меньшим-обменом-данными"></a>

<a id="cs336-flashattention"></a>

# FlashAttention: exact attention with less memory traffic

Once the queries $Q$, keys $K$ and values $V$ are available, attention
requires two matrix multiplications and a softmax:

$$
\begin{aligned}
S&=\frac{QK^\top}{\sqrt d},\\
P&=\operatorname{softmax}(S),\\
O&=PV.
\end{aligned}
$$

Executing these as separate operators requires storing $S$ and $P$ between
operations. For a long sequence, these matrices are much larger than the
inputs $Q,K,V$ and output $O$. The GPU writes intermediate data to memory;
the next operator reads it back. Those transfers can cost more than
additional arithmetic on a small tile already held on chip.

FlashAttention computes the same attention without constructing complete
$S$ and $P$ matrices in device memory. It processes blocks of keys and values
and immediately adds their contributions to the output. The softmax
denominator includes every key, including keys not yet visited. As new keys
arrive, the algorithm adjusts its accumulated sums without returning to
earlier blocks.

<a id="вычислительная-сложность-не-объясняет-время-работы"></a>

## Computational complexity does not determine execution time

For one head and one example, let $Q,K,V\in\mathbb R^{N\times d}$. The two
matrix products require approximately $4N^2d$ FLOPs when multiplication and
addition are counted separately. FlashAttention retains the quadratic
dependence on $N$. It reduces transfers of intermediate data between
off-chip GPU memory and on-chip storage.

HBM provides a large off-chip memory. Thread-block shared memory and registers
are on chip and have much lower access latency, but cannot hold an entire
long-sequence attention matrix. Nor can the storage of separate
multiprocessors be treated as one buffer available to an arbitrary thread
block. Total GPU SRAM capacity is not the capacity of one computation block.

The theoretical FlashAttention analysis represents these resources with
a two-level HBM–SRAM model. An actual kernel must separately place input
tiles in shared memory, accumulators in registers, and state and pipeline
buffers in the remaining capacity. An oversized tile can spill registers
to slower memory or reduce the number of resident thread blocks. Increasing
the amount loaded at once can therefore make execution slower.

This **IO-aware** analysis considers transfers between memory levels as well
as arithmetic. See [FlashAttention, §2–3](https://arxiv.org/html/2205.14135v2#S3)
for the model and [CS336 Lecture 5, pp.49–55](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_05.pdf)
for its connection to operator fusion and tiling.

<a id="обычная-реализация-материализует-квадратные-тензоры"></a>

## A conventional implementation materializes square tensors

Consider three separate operators:

1. Matrix multiplication reads $Q,K$ and writes the $N\times N$ matrix $S$.
2. Softmax reads $S$ and writes an equally large $P$.
3. The second multiplication reads $P,V$ and writes the $N\times d$ output $O$.

For $N=8192$, $d=64$ and two bytes per element, the sizes are as follows.
These are for one head and one example, excluding gradients and library
workspaces.

| Object | Size |
|---|---:|
| One complete $S$ or $P$ matrix | 128 MiB |
| All four matrices $Q,K,V,O$ together | 4 MiB |
| One length-$N$ FP32 row-statistics vector | 32 KiB |

Writing and reading $S$ and $P$ once each transfers 512 MiB. This is a
calculation for that execution order, not a measured traffic count for every
existing kernel: intermediate dtype, rereads, caching and fusion change the
number. It identifies the opportunity. At large $N$, memory traffic can be
reduced without making every individual multiplication faster.

FlashAttention neither removes token pairs nor approximates softmax. It
avoids storing the complete square matrix between operators. If the caller
requires the attention weights $P$ themselves, returning them still takes
quadratic storage.

<a id="блочный-проход-через-sram"></a>

## A tiled pass through SRAM

A query tile $Q_i$ has $B_q$ rows; key and value tiles $K_j,V_j$ have $B_k$
rows each. Their product $Q_iK_j^\top$ is a small $B_q\times B_k$ matrix.
Its contribution can immediately enter the output tile, after which the
temporary matrix is discarded. Inputs, outputs and short row-statistics
vectors remain.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/flashattention-tiling.png]]

*Tri Dao, [FlashAttention-2, Stanford CRFM](https://crfm.stanford.edu/2023/07/17/flash2.html).
Two key/value blocks contribute to one output without storing complete
$S$ and $P$. The original diagram omits maximum subtraction for simplicity;
the numerically stable version is derived below.*

FA1 and FA2 visit tiles in different orders. The original FA1 algorithm
selects $K_j,V_j$ in the outer loop and visits query tiles $Q_i$ in the inner
loop. A loaded key/value pair can serve multiple query tiles. Accumulated
outputs and row statistics must be read and updated as execution advances
through the keys.

FA2 forward selects a query tile in the outer loop. One thread block keeps
that tile's state while traversing all $K,V$ tiles and writes the completed
output once. Other query tiles can run independently because their output
rows do not overlap. Different tasks may reread the same $K,V$. This trades
repeated loads for independent work; it does not imply that every input is
read exactly once.

[CS336 Assignment 2, Algorithm 1, p.25](https://github.com/stanford-cs336/assignment2-systems/blob/ca8bc81a59b70516f7ebb2da4808daade877c736/cs336_assignment2_systems.pdf)
and this chapter's CPU implementation use the second order. Normalization
must remain consistent across the key tiles read by each query tile.

<a id="как-softmax-считается-по-частям"></a>

## Computing softmax incrementally

Consider one row of $S$. Its entries $s_j$ score the query against the keys,
and $v_j$ are the corresponding values. Attention computes a weighted mean:

$$
o=\frac{\sum_j e^{s_j}v_j}{\sum_j e^{s_j}}.
$$

Numerator and denominator can be accumulated in blocks, but directly
evaluating $e^{s_j}$ overflows for large scores. Stable softmax subtracts
the maximum score from every entry. If a later tile raises that maximum,
the accumulated sums must be expressed relative to the new maximum.
The previous scores do not need to be retained.

<a id="пример-новый-блок-содержит-больший-максимум"></a>

### A larger maximum in the next block

Take scores $[0,\log2,\log4]$ and scalar values $[10,20,40]$. Without tiling,
weights are proportional to $[1,2,4]$, giving

$$o=\frac{1\cdot10+2\cdot20+4\cdot40}{1+2+4}=30.$$

Process the first two keys. Subtracting their maximum $\log2$ gives weights
$[1/2,1]$, denominator $\ell=1.5$ and numerator
$u=(1/2)\cdot10+1\cdot20=25$. For this portion, the output is $25/1.5=50/3$.

The third key raises the maximum to $\log4$. The earlier weights must now
be $[1/4,1/2]$. Multiplying both accumulated sums by
$e^{\log2-\log4}=1/2$ makes this adjustment without rereading those keys.
The new key has weight one:

$$
\begin{aligned}
\ell'&=\tfrac12\cdot1.5+1=1.75,\\
u'&=\tfrac12\cdot25+1\cdot40=52.5,\\
o&=52.5/1.75=30.
\end{aligned}
$$

| Processed keys | Maximum | $\ell$ | $u$ | $u/\ell$ |
|---|---|---:|---:|---:|
| First two | $\log2$ | 1.5 | 25 | $50/3$ |
| All three | $\log4$ | 1.75 | 52.5 | 30 |

Updating the maximum without rescaling the old sums would return
$(25+40)/(1.5+1)=26$. This is an algorithmic error rather than rounding:
weights measured relative to different maxima have been mixed.

<a id="объединение-двух-блоков"></a>

### Merging two blocks

For a processed key set $A$, retain maximum $m_A$, weight sum $\ell_A$ and
unnormalized output $u_A$:

$$
\begin{aligned}
m_A&=\max_{j\in A}s_j,\\
\ell_A&=\sum_{j\in A}e^{s_j-m_A},\\
u_A&=\sum_{j\in A}e^{s_j-m_A}v_j.
\end{aligned}
$$

$m_A$ and $\ell_A$ are scalars; $u_A$ has the same vector dimension as $v_j$.
The partial output is $u_A/\ell_A$. The common factor $e^{-m_A}$ cancels
between numerator and denominator.

Let the next block $B$ have statistics $m_B,\ell_B,u_B$. Express both parts
relative to $m'=\max(m_A,m_B)$:

$$
\begin{aligned}
\ell'&=e^{m_A-m'}\ell_A+e^{m_B-m'}\ell_B,\\
u'&=e^{m_A-m'}u_A+e^{m_B-m'}u_B.
\end{aligned}
$$

The identity $e^{m_A-m'}e^{s_j-m_A}=e^{s_j-m'}$ explains the rescaling.
Every key receives the weight it would have received in a full-row
calculation. The merged output is $o=u'/\ell'$. Initialize an empty row
with $m=-\infty$, $\ell=0$, $u=0$; the first nonempty block supplies a
finite maximum and positive denominator.

For a query tile, apply these updates independently to each row. The
scalars $m,\ell$ become length-$B_q$ vectors and $u$ becomes a
$B_q\times d$ matrix. Each score tile updates the maximum, rescales old
$u$ and adds the product of unnormalized weights with $V_j$. Divide by
$\ell$ only after visiting every key. Also save

$$L_i=m_i+\log\ell_i=\log\sum_j e^{S_{ij}}.$$

This logsumexp vector recovers probabilities during backward without
retaining complete score rows.

<a id="почему-память-становится-линейной"></a>

## Why memory usage becomes linear

For batch size $B$, head count $H$, sequence length $N$ and head dimension
$d$, inputs $Q,K,V$ and output $O$ contain $O(BHNd)$ elements. Saved $L$
contains $O(BHN)$. Temporary tiles never grow to $N\times N$. With fixed
$B,H,d$, this storage is linear in $N$.

Storage and number of reads are different quantities. In FA1's two-level
model, for SRAM capacity $M$ elements with $d\le M\le Nd$, HBM traffic is
$\Theta(N^2d^2/M)$, compared with $\Theta(Nd+N^2)$ for the conventional
implementation analyzed in the paper. With fixed $d,M$, the first bound
is still quadratic in $N$. Reusing data in fast memory reduces traffic;
it does not eliminate every repeated read.
[FA1, Theorem 2](https://arxiv.org/html/2205.14135v2#S3.SS2).

Mixed precision and buffer placement complicate physical accounting:
$Q,K,V$ may be BF16 while maxima, denominators and output accumulators
are FP32. Larger tiles can reduce rereads but consume more registers and
shared memory. Capacity, available parallel tasks and matrix-instruction
throughput must be considered together.

<a id="проверяемая-реализация-блочного-forward"></a>

## A checkable blocked forward pass

The following ordinary PyTorch operations implement the recurrence on CPU.
The example uses one sequence and one head, equal query/key/value lengths,
no dropout, and FP64 for correctness comparisons. The final tile may be
shorter than earlier tiles. FP64 here is not a recommendation for GPU kernels.

<!-- checked-example: en-flashattention-cpu -->
```python
import math
import torch


def blocked_attention(q, k, v, q_block=4, k_block=3, causal=False):
    if q.ndim != 2 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Expected equal [sequence, head_dim] shapes")
    if q.shape[0] == 0 or q.shape[1] == 0 or min(q_block, k_block) <= 0:
        raise ValueError("Dimensions and block sizes must be positive")
    n, d = q.shape
    outputs, logsumexps = [], []
    for i in range(0, n, q_block):
        qi = q[i:i + q_block]
        rows = torch.arange(i, i + qi.shape[0], device=q.device)
        m = q.new_full((qi.shape[0],), -float("inf"))
        ell = q.new_zeros(qi.shape[0])
        u = torch.zeros_like(qi)
        for j in range(0, n, k_block):
            if causal and j >= i + qi.shape[0]:
                break  # This entire key block is in the future.
            kj, vj = k[j:j + k_block], v[j:j + k_block]
            scores = (qi @ kj.T) / math.sqrt(d)
            if causal:
                cols = torch.arange(j, j + kj.shape[0], device=q.device)
                scores = scores.masked_fill(cols[None, :] > rows[:, None],
                                            -float("inf"))
            m_new = torch.maximum(m, scores.amax(dim=-1))
            alpha = torch.exp(m - m_new)
            p = torch.exp(scores - m_new[:, None])
            ell = alpha * ell + p.sum(dim=-1)
            u = alpha[:, None] * u + p @ vj
            m = m_new
        outputs.append(u / ell[:, None])
        logsumexps.append(m + ell.log())
    return torch.cat(outputs), torch.cat(logsumexps)


# Realize scores [0, log(2), log(4)] with d=1.
q_example = torch.ones(3, 1, dtype=torch.float64)
k_example = torch.tensor([[0.0], [math.log(2)], [math.log(4)]],
                         dtype=torch.float64)
v_example = torch.tensor([[10.0], [20.0], [40.0]], dtype=torch.float64)
out, lse = blocked_attention(q_example, k_example, v_example,
                             q_block=2, k_block=2)
torch.testing.assert_close(out, torch.full_like(out, 30.0))
torch.testing.assert_close(lse, torch.full_like(lse, math.log(7)))
print(out[-1, 0].item())  # 30.0
```

The variable `p` contains exponentials relative to the current maximum,
not the final probability matrix. Their sums enter `ell`; division occurs
outside the inner loop to avoid repeatedly normalizing partial outputs.

This code reproduces the mathematical order, not FlashAttention's speed.
Each PyTorch operation executes separately, and ordinary autograd retains
intermediates for its backward. A blocked forward with autograd enabled
therefore does not establish linear training memory. A specialized kernel
fuses operations and supplies a recomputing backward.

<a id="backward-что-требуется-пересчитать"></a>

## Recomputing the backward pass

Let $G=\partial\mathcal L/\partial O$ be the upstream gradient. Without
dropout, the derivatives are

$$
dV=P^\top G,\qquad dP=GV^\top,
$$

$$
dS_{ij}=P_{ij}\left(dP_{ij}-\sum_tP_{it}dP_{it}\right),
$$

$$
dQ=\frac{dSK}{\sqrt d},\qquad dK=\frac{dS^\top Q}{\sqrt d}.
$$

$dP$ denotes the loss derivative with respect to $P$, not a small change
in probability. Softmax backward subtracts one row statistic from every
entry of $dP$. Compute this statistic without reducing a complete
length-$N$ probability row:

$$
\begin{aligned}
D_i&=\sum_jP_{ij}dP_{ij}\\
&=\sum_jP_{ij}\sum_aG_{ia}V_{ja}\\
&=\sum_aG_{ia}\underbrace{\sum_jP_{ij}V_{ja}}_{O_{ia}}\\
&=\sum_aO_{ia}G_{ia}.
\end{aligned}
$$

Thus $D$ needs only the already known $O,G$ and contains $N$ elements per head.

<a id="пример-градиентов-для-одной-строки"></a>

### Gradients for one row

For scores $[0,\log2,\log4]$ and values $[10,20,40]$, the output was $o=30$.
Set the derivative with respect to that scalar output to $G=1$.
Then $p=[1/7,2/7,4/7]$, $dP=[10,20,40]$ and $D=oG=30$.

| Key | $p_j$ | $dV_j=p_jG$ | $dS_j=p_j(dP_j-D)$ |
|---|---:|---:|---:|
| 1 | $1/7$ | $1/7$ | $-20/7$ |
| 2 | $2/7$ | $2/7$ | $-20/7$ |
| 3 | $4/7$ | $4/7$ | $40/7$ |

The first value is below the current output. Raising its score increases
its contribution to the mean and lowers $o$, giving a negative derivative.
The third value gives a positive derivative. Score derivatives sum to zero:
adding the same constant to all scores does not change softmax.

Beyond $Q,K,V$, this row's backward needs saved $L=\log7$ and $o=30$.
Recomputing the first tile gives
$[\exp(0-\log7),\exp(\log2-\log7)]=[1/7,2/7]$.
The second tile recovers $4/7$. Subtracting $D=30$ gives the derivatives
in the table without retaining complete $P$.

<a id="восстановление-блоков-вероятностей"></a>

### Recovering probability tiles

Saved $L$ reconstructs any probability tile:

$$
\begin{aligned}
S_{ij}&=Q_iK_j^\top/\sqrt d,\\
P_{ij}&=\exp(S_{ij}-L_i[:,\mathrm{None}]).
\end{aligned}
$$

Indices $i,j$ now refer to tiles. Subtract $L_i$ row by row. The denominator
over all visible keys is already known from forward, so normalization does
not require another traversal of all keys. Then compute

$$
dV_j\mathrel{+}=P_{ij}^\top G_i,\qquad dP_{ij}=G_iV_j^\top,
$$

$$
dS_{ij}=P_{ij}\odot(dP_{ij}-D_i[:,\mathrm{None}]),
$$

$$
\begin{aligned}
dK_j&\mathrel{+}=dS_{ij}^\top Q_i/\sqrt d,\\
dQ_i&\mathrel{+}=dS_{ij}K_j/\sqrt d.
\end{aligned}
$$

An additional $Q_iK_j^\top$ product and exponentiation replace reads of saved
quadratic $P$. More arithmetic avoids storing and loading a large intermediate.
This is activation recomputation where derivative structure identifies the
precise quantities needed, rather than replaying the entire forward.

See [CS336 Assignment 2, pp.23–24](https://github.com/stanford-cs336/assignment2-systems/blob/ca8bc81a59b70516f7ebb2da4808daade877c736/cs336_assignment2_systems.pdf)
and [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/44b Gradient checkpointing и offload|activation checkpointing]].

<a id="dropout-и-воспроизводимость-пересчёта"></a>

### Dropout and reproducible recomputation

With dropout after softmax, the output uses $P\odot Z$, where $Z$ is a
random mask including the surviving entries' scale factor. Both $dV$ and
$dP$ must use the same mask. Sampling a new mask during backward would
differentiate a different function.

Storing the complete $Z$ is unnecessary.
[FA1, Appendix B.4](https://arxiv.org/html/2205.14135v2#A2.SS4) describes
saving random-generator state and reproducing the mask during recomputation.
The kernel must consistently map random values to attention positions.
Two implementations with the same seed may draw numbers in different orders,
so equal seeds alone do not establish equal masks. Compare outputs and
gradients without dropout first; test mask reproduction separately.

<a id="два-способа-накопить-градиенты"></a>

<a id="почему-в-алгоритме-два-tiled-прохода"></a>

### Two ways to accumulate gradients

$dK_j,dV_j$ collect contributions from every query, while $dQ_i$ collects
contributions from every key. A parallel task owning a key tile can accumulate
$dK_j,dV_j$ locally, but its $dQ_i$ contributions overlap those of other tasks.

[FA2, §3.2](https://arxiv.org/html/2307.08691v1#S3.SS2) uses atomic additions
for $dQ$. Atomics prevent lost updates, but addition order can vary, affecting
bitwise reproducibility. Libraries may offer other, including deterministic,
backward variants. Their costs should not be mixed in one measurement.

[CS336 Assignment 2, pp.28–29, Algorithm 2](https://github.com/stanford-cs336/assignment2-systems/blob/ca8bc81a59b70516f7ebb2da4808daade877c736/cs336_assignment2_systems.pdf)
offers a different optional teaching algorithm. One pass fixes a key tile
and accumulates $dK,dV$; another fixes a query tile and accumulates $dQ$.
Each gradient tile has a unique owner, so these accumulators need no atomics.
The cost is recomputing score and probability tiles again. This exposes the
dependencies but does not describe every FlashAttention kernel.

<a id="causal-mask-в-неполной-плитке"></a>

### A causal mask in an incomplete tile

Tiling does not change which keys are visible. The CPU code retains global
row and column positions and sets forbidden scores to $-\infty$ before
exponentiation. A wholly future tile can be skipped; a diagonal tile still
needs entrywise masking. Backward reconstruction applies the same condition.

Final CPU slices use their actual lengths. GPU kernels commonly use fixed
tile sizes and additionally guard loads and stores beyond tensor boundaries.
That bounds check does not replace the causal mask. The example does not
handle arbitrary masks with fully masked rows or align unequal query and key
lengths; those cases are covered in
[[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention|the KV-cache chapter]].

<a id="что-изменил-flashattention-2"></a>

## What FlashAttention-2 changed

Lower HBM traffic does not ensure that every GPU execution unit is occupied.
FA1 left three sources of lost performance: non-matrix work, too few
independent tasks, and communication between warps. FA2 changes all three.

<a id="меньше-нормировок-внутри-цикла"></a>

### Fewer normalizations inside the loop

Tensor Cores execute matrix products; other units compute maxima,
exponentials, sums and rescaling. FLOP counts alone do not make these costs
comparable. For example, the FA2 paper contrasts A100's theoretical
312 TFLOP/s FP16/BF16 matrix throughput with 19.5 TFLOP/s ordinary FP32
throughput. These describe that GPU and those operation types, not a
universal price for every scalar operation.

FA2 therefore retains unnormalized output $u$ rather than dividing by $\ell$
after every tile. Rescaling when the maximum changes remains necessary;
final normalization moves outside the loop. Backward stores only
$L=m+\log\ell$ instead of separate $m,\ell$ vectors. The CPU example already
uses these choices.

<a id="больше-независимых-блоков-работы"></a>

### More independent work blocks

One task per head and example yields only eight tasks for batch one and
eight heads, even with a very long sequence. Much of a large GPU can remain
idle. FA2 assigns each query tile an independent output. With $N=8192$,
$B_q=128$ and eight heads, this yields $8\cdot64=512$ tasks instead of eight.

This is the number of available tasks, not simultaneous residency or a
64-fold speedup. Registers, shared memory and GPU limits determine how many
blocks can execute together. Reordering nevertheless removes a specific
source of insufficient parallelism.

<a id="меньше-обмена-между-warp"></a>

### Less communication between warps

In sliced-K, different warps receive different key/value portions for shared
queries. Each computes only part of an output row's sum. Their results must
be combined through shared memory and synchronization.

In sliced-Q, warps receive separate query rows and share the $K,V$ tiles.
Each warp owns its output rows, eliminating this cross-warp reduction from
that portion of forward.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/flashattention2-partitioning.png]]

*Tri Dao, [FlashAttention-2, Stanford CRFM](https://crfm.stanford.edu/2023/07/17/flash2.html).
Left: warps contribute to the same output rows. Right: each warp owns rows
of $Q$ and the corresponding output. This partitions work within a thread
block; it does not distribute a model across GPUs.*

These changes do not imply that FA2 is always limited only by matrix
operations. Relative execution rates change across GPU generations.
The next optimization uses asynchronous execution as well as partitioning.

<a id="flashattention-3-одновременная-передача-и-обработка-блоков"></a>

## FlashAttention-3: overlapping transfers and computation

FA3 was designed for Hopper. TMA moves data from global memory to shared
memory; WGMMA launches matrix multiplication for a group of four warps.
Both let execution issue an operation and continue other work before it
finishes. A result cannot be read early, and a buffer cannot be overwritten
while a consumer still uses it.

<a id="загрузка-следующего-блока-во-время-вычислений"></a>

### Loading the next tile during computation

Some warps prepare data while others compute attention. A producer loads
the next $K_j,V_j$ into a shared-memory ring buffer; consumers operate on
ready tiles. Barriers indicate when loading has finished and when storage
can be reused. This division is **warp specialization**.

The next transfer can overlap current computation. A slow transfer still
makes consumers wait, and a deeper buffer consumes more memory.
Asynchrony permits overlap without removing bandwidth limits or dependencies.

<a id="softmax-одной-группы-и-матричное-умножение-другой"></a>

### Softmax in one group, matrix multiplication in another

Even with timely loads, each attention iteration has a dependency chain:
$QK^\top$, then softmax, then $PV$. Softmax waits for scores, and the second
product waits for weights. While a group computes exponentials and sums,
its matrix work is paused.

Two groups with independent query tiles can stagger their schedules.
While one computes softmax, the other executes matrix products, then they
switch roles. FA3 calls this **pingpong scheduling**.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/flashattention-review-2026-10/fa3-pingpong-original.png]]

*Jay Shah, Ganesh Bikshandi et al.,
[FlashAttention-3, Figure 1, p.6](https://arxiv.org/html/2407.08608v1#S3.F1),
CC BY 4.0. GEMM0 computes scores; GEMM1 combines weights and values.
Colors identify iterations and dashed lines show alternation between groups.
This is a dependency diagram, not a measured GPU timeline.*

Tensor Cores execute products while other units perform portions of softmax.
Barriers enforce the required ordering but have their own cost. Results
depend on tile sizes and the compiler's generated instructions.

<a id="конвейер-внутри-одной-группы"></a>

### Pipelining within one group

Neighboring iterations of one group can also overlap. Let
$\widetilde P_{ij}$ denote unnormalized weights after subtracting the current
maximum but before dividing by $\ell$, as in the CPU example's `p`.
The next product $Q_iK_{j+1}^\top$ depends on inputs, not on completion of
$\widetilde P_{ij}V_j$. It can be issued early to overlap matrix work and
softmax from adjacent iterations.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/flashattention-review-2026-10/fa3-gemm-softmax-original.png]]

*Jay Shah, Ganesh Bikshandi et al.,
[FlashAttention-3, Figure 2, p.6](https://arxiv.org/html/2407.08608v1#S3.F2),
CC BY 4.0. Numbers identify iterations. WGMMA0, softmax and WGMMA1 within
an iteration retain their dependencies while adjacent iterations overlap.
Original paper illustration, not redrawn.*

Separate buffers hold current weights and next scores. Output accumulator
$u$ contains contributions from processed keys. One pipeline step has the
following dependencies:

| Operation | Required data | Constraint on the next action |
|---|---|---|
| Launch $Q_iK_{j+1}^{\top}$ | Input tiles $Q_i,K_{j+1}$ | Next-tile softmax waits for this product |
| Launch $\widetilde P_{ij}V_j$ | Current weights and $V_j$ | Do not reuse the weight buffer before reads finish |
| Process next scores | Completed scores and old $m,\ell$ | Produces new weights and the rescaling factor for $u$ |
| Rescale $u$ to the new maximum | Rescaling factor and completed current value product | Do not change $u$ while a matrix operation still writes to it |

An asynchronous launch does not make its result immediately available for
modification. Extra intermediate storage permits the overlap.
The authors describe practical constraints in
[FA3, §3.2](https://arxiv.org/html/2407.08608v1#S3.SS2).
The following original paragraphs, **Compiler reordering** and
**Register pressure**, are by Jay Shah, Ganesh Bikshandi et al.,
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Their $B_r,B_c$ correspond to our $B_q,B_k$.

<!-- source-excerpt: fa3-pipeline-practicalities:start -->
> **Compiler reordering**
>
> The pseudocode represents an idealized execution order but the compiler (NVCC) often rearranges instructions for optimization.
> This can disrupt the carefully crafted WGMMA and non-WGMMA operation pipelining sequence, potentially leading to unexpected behavior or diminished performance gains. An analysis of the SASS code shows that the compiler generates overlapped code as expected ([Section B.2](https://arxiv.org/html/2407.08608v1)).
>
> **Register pressure**
>
> To maintain optimal performance, register spilling should be minimized.
> However, the 2-stage pipeline requires additional registers to store intermediate results and maintain context between stages.
> Specifically, an extra $\mathbf{S}_{\mathrm{next}}$ must be kept in registers, leading to extra register usage of size $B_r \times B_c \times \text{sizeof}(\text{float})$ per threadblock.
> This increased register demand may conflict with using larger block sizes (another common optimization), which is also register-hungry.
> In practice, trade-offs should be made based on profiling results.
<!-- source-excerpt: fa3-pipeline-practicalities:end -->

An extra $64\times64$ FP32 score tile requires 16 KiB of register state
across the thread block, not in one thread's register file. Doubling both
tile dimensions quadruples this requirement. If registers are insufficient,
the compiler may spill values to device-backed local memory. Extra memory
accesses can erase the gain from pipelining.

<a id="fp8-отдельный-выбор-точности"></a>

### FP8 is a separate precision choice

FP8 increases available matrix throughput but reduces input precision.
An outlier in one feature can set the scale for an entire tensor, coarsening
the representation of many smaller values.

FA3 uses blockwise scales and transforms $Q,K$ before quantization to spread
outliers across features. For orthogonal $R$,

$$
(QR)(KR)^\top=QRR^\top K^\top=QK^\top.
$$

Before quantization, this preserves scores. After quantization, it may
reduce error relative to quantizing the original distribution. The paper
uses a normalized Hadamard transform with random signs, which is cheaper
than a general dense orthogonal transform.

Data layout also needs work: the first product produces FP32 accumulators,
while the second FP8 product expects a particular operand layout. FA3
coordinates these representations and transforms value tiles within the
kernel. Reducing bit width does not eliminate layout conversion costs.

Blocked FP16/BF16 attention preserves the dense-attention formula but can
differ numerically because of rounding. FP8 adds quantization error.
Neither guarantees bitwise agreement with the original implementation.
The accuracy comparison in [FA3, §3.3 and §4.3](https://arxiv.org/html/2407.08608v1#S3.SS3)
applies to the paper's inputs and quantization choices, not every model or
quality tolerance.

<a id="реализация-и-номер-версии"></a>

## Implementations and version numbers

An algorithm name, the `flash-attn` package and PyTorch's backend are distinct.
Device, dtype and backward support depend on the implementation. The table
records the official repository as reviewed on **14 October 2026**, commit
[`1fc0221183e799363001394a12dd9e78e15f5dda`](https://github.com/Dao-AILab/flash-attention/tree/1fc0221183e799363001394a12dd9e78e15f5dda).

| Implementation | Confirmed by that commit's README |
|---|---|
| FA2, NVIDIA CUDA | Ampere, Ada, Hopper; FP16/BF16; head dimensions up to 256. Backward at dimension 256 on consumer GPUs has a separate no-dropout condition. |
| FA3, Hopper | FP16/BF16 forward and backward; FP8 forward. H100/H800 and CUDA 12.3 or newer are specified. The section is labeled a beta release. |
| FA4, CuTeDSL | Separate `flash-attn-4` package for Hopper and Blackwell; the example uses `flash_attn.cute.flash_attn_func`. |

This is not a complete compatibility matrix. ROCm has separate implementations
and constraints. FA4's presence does not mean an arbitrary PyTorch call
selects it. Check the
[pinned README](https://raw.githubusercontent.com/Dao-AILab/flash-attention/1fc0221183e799363001394a12dd9e78e15f5dda/README.md)
and the executed kernel. FA4 mechanisms require its own paper; a similar
name does not establish the FA3 execution schedule.

<a id="обучение-prefill-и-decode"></a>

## Training, prefill and decode

Training and long prefill provide many query rows whose tiles can be
distributed across workers. Ordinary decode has one new $Q$ row per
sequence. Its geometry often requires partitioning other axes, such as
subsets of keys, and then merging partial results.

Faster long prefill therefore does not establish faster per-token decode.
KV caching, Flash-Decoding and page layouts are covered in
[[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention|the KV-cache chapter]]
and [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55c Serving engines — vLLM, SGLang, TensorRT-LLM и FlashInfer|serving engines]].

<a id="flashattention-не-следует-путать-с-архитектурой-внимания"></a>

## FlashAttention is not an attention architecture

| Mechanism | What changes | Relation to dense attention |
|---|---|---|
| FlashAttention | Evaluation order and data movement | Same mathematical operation for the same mask; numerical behavior depends on implementation and dtype |
| Sliding-window attention | Keys visible to a row | Restricts the interaction set |
| Block-sparse attention | Computed blocks | Changes interaction structure; not full dense attention |
| Linear attention | Aggregation equation | A different attention mechanism |
| PagedAttention | Physical KV placement | Does not itself change logical keys or values |

Tiling does not imply sparsity. FlashAttention uses tiles to manage memory;
if the mask allows every pair, every pair remains in the computation.
The same kernel-organization principle can also serve a restricted mask.

<a id="как-понять-что-kernel-действительно-используется"></a>

## Verifying that the kernel is used

PyTorch's `scaled_dot_product_attention` selects an implementation from
device, dtype, tensor shapes, mask and other parameters. A successful call
does not reveal the chosen path. A diagnostic can temporarily allow only
the flash backend:

```python
# CUDA-only diagnostic snippet; not GPU-validated for this chapter.
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

q = torch.randn(1, 8, 1024, 64, device="cuda", dtype=torch.bfloat16)
k, v = torch.randn_like(q), torch.randn_like(q)
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    output = F.scaled_dot_product_attention(q, k, v,
                                           dropout_p=0.0, is_causal=True)
```

Unsupported parameters should not silently fall back to the math backend
here: the context manager has disabled it. However, `FLASH_ATTENTION`
does not identify FA2, FA3 or FA4 from the separate repository. Record the
PyTorch version, build, device and profiled kernel name.

SDPA applies dropout according to `dropout_p` even when its containing
module is in `eval()` mode. Pass zero explicitly for evaluation without
dropout. See [PyTorch 2.14 SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
and [sdpa_kernel](https://docs.pytorch.org/docs/2.14/generated/torch.nn.attention.sdpa_kernel.html).
The CUDA snippet above was not GPU-validated for this chapter.

Correctness checks precede speed measurements. Compare the CPU example's
output and $L$ with direct softmax and logsumexp, then compare gradients
for all three inputs. Cover multiple partitions of the same sequence,
incomplete tiles, causal and noncausal modes, and a later tile that raises
the maximum. CS336's official tests are a starting point, but their fixed
shape does not cover every such case.

GPU measurements require warmup, synchronization and matching shapes,
dtype, dropout and mask. Time forward and backward separately and measure
peak allocated memory separately from elapsed time. See
[[02 Areas/ML & DL/00 Учебник/10 ML Systems/07 Profiling ML-нагрузки|profiling ML workloads]].
A large operation-level speedup need not produce the same model-level gain:
MLPs, data movement, communication and other operations also contribute.

<a id="практика-и-первоисточники"></a>

## Practice and primary sources

- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week06_dl_arithmetic/lecture.pdf|Efficient DL Systems: activation traffic and operator fusion]].
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week06_dl_arithmetic/seminar/practice.ipynb|Efficient DL Systems FlashAttention practice]].
- [[02 Areas/ML & DL/Papers/Flash Attention|FlashAttention source note]].
- [[02 Areas/ML & DL/Papers/Flash Attention 2|FlashAttention-2 source note]].
- Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135): two-level memory model, algorithm and IO analysis; Appendix B covers recomputation and dropout.
- Dao, [FlashAttention-2](https://arxiv.org/abs/2307.08691): non-matrix work, parallelism and warp partitioning.
- Tri Dao, [FlashAttention-2 at Stanford CRFM](https://crfm.stanford.edu/2023/07/17/flash2.html): original aggregation and sliced-Q diagrams.
- Shah et al., [FlashAttention-3](https://arxiv.org/html/2407.08608v1): asynchronous schedules, register constraints and FP8; CC BY 4.0.
- [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention): official implementations, constraints and tests; the reviewed state is pinned above.
- PyTorch, [Scaled Dot Product Attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html): current API documentation; version 2.14 was used for chapter verification.
- [Stanford CS336 Lecture 5, pinned `8b59b507`, pp.49–55](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_05.pdf): tiling, fusion and online softmax.
- [Stanford CS336 Assignment 2, pinned `ca8bc81`, pp.23–31](https://github.com/stanford-cs336/assignment2-systems/blob/ca8bc81a59b70516f7ebb2da4808daade877c736/cs336_assignment2_systems.pdf): forward/backward equations, algorithms, tests and measurement conditions.

← [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55c Serving engines — vLLM, SGLang, TensorRT-LLM и FlashInfer|Serving engines]] ·
[[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/57 Квантизация языковых моделей|Quantizing language models]] →
