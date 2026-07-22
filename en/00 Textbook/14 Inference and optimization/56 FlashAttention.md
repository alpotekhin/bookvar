---
title: "FlashAttention: exact attention with less memory traffic"
type: textbook-chapter
status: canonical
locale: en
translation_of: "00 Учебник/14 Inference и оптимизация/56 FlashAttention.md"
last_updated: 2026-07-22
last_verified: 2026-07-22
primary_sources:
  - https://arxiv.org/abs/2205.14135
  - https://crfm.stanford.edu/2023/07/17/flash2.html
  - https://github.com/Dao-AILab/flash-attention
  - https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
---

# FlashAttention: exact attention with less memory traffic

The attention equation contains two matrix multiplications and a softmax:

$$
S=\frac{QK^\top}{\sqrt d},\qquad P=\operatorname{softmax}(S),\qquad O=PV.
$$

On paper, it is natural to compute $S$, then $P$, and finally multiply $P$ by $V$. On a GPU, that sequence writes the square matrices $S$ and $P$ to device memory and reads them back shortly afterward. With long contexts, a substantial portion of execution time is spent moving these intermediates rather than performing arithmetic. FlashAttention preserves the same mathematical operation but changes its evaluation order so that small tiles of the square matrix never leave fast local memory.

## Computational complexity does not determine execution time

Standard attention requires $O(N^2d)$ operations for sequence length $N$ and head dimension $d$. This notation says nothing about how often data moves between levels of the memory hierarchy.

An accelerator has large but relatively slow HBM and much smaller on-chip SRAM. Matrix units may consume operands faster than HBM can supply them. An algorithm that repeatedly writes and reads intermediate tensors can therefore become memory-bound without changing its FLOP count.

The original FlashAttention paper calls this analysis **IO-aware**: the algorithm is designed around reads and writes between HBM and SRAM, not only the asymptotic number of arithmetic operations.

## A conventional implementation materializes square tensors

Let $Q,K,V\in\mathbb R^{N\times d}$. An implementation composed of separate operators commonly performs the following steps:

1. Read $Q$ and $K$, then write $S=QK^\top$, which has $N^2$ elements.
2. Read $S$, compute a row-wise softmax, and write the $N^2$-element matrix $P$.
3. Read $P$ and $V$, then write the $Nd$-element output $O$.

At $N=8192$, a single $N^2$ matrix in FP16 occupies 128 MiB **for one head and one example**, before auxiliary buffers or the backward pass. Even when a framework fuses some operators, materializing $P$ creates a quadratic memory footprint.

FlashAttention neither sparsifies attention nor removes token pairs. It avoids storing the complete $S$ and $P$ matrices in HBM.

## A tiled pass through SRAM

Tri Dao's overview diagram on the Stanford CRFM page shows the complete forward pass. Blocks $K_j,V_j$ are loaded successively from HBM into SRAM. For each $Q_i$ block, the algorithm computes a small $Q_iK_j^\top$ tile, immediately applies a local softmax, and updates the corresponding output block. HBM holds $Q,K,V,O$ and a few row-wise statistics, but never the full attention matrix.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/flashattention-tiling.png]]

*FlashAttention forward pass: outer loops over $K,V$ and $Q$ blocks, a local $S_{ij}$ matrix in SRAM, and online updates to the softmax statistics. Original diagram by Tri Dao, [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://crfm.stanford.edu/2023/07/17/flash2.html).*

Tile dimensions are chosen so that $Q_i,K_j,V_j$, local scores, and output state fit in SRAM. Some $Q$ blocks must be read more than once, but this is cheaper than writing and rereading two complete $N\times N$ matrices.

## Computing softmax incrementally

The softmax of a row depends on the sum of exponentials over all its elements, which appears to prevent finalizing one block before examining the remaining keys. Online softmax resolves this dependency.

For the processed portion of a row, retain its maximum $m$ and normalization sum

$$
\ell=\sum_j e^{s_j-m}.
$$

Suppose the next block has maximum $m_b$ and normalization sum $\ell_b$. The combined statistics are

$$
m'=\max(m,m_b),
$$

$$
\ell'=e^{m-m'}\ell+e^{m_b-m'}\ell_b.
$$

The accumulated output numerator is rescaled by the same coefficients:

$$
u'=e^{m-m'}u+e^{m_b-m'}u_b,
\qquad o=\frac{u'}{\ell'}.
$$

Each new block thus corrects earlier contributions as if the final maximum and denominator had been known from the beginning. This is not a softmax approximation; it is a different associative evaluation order. Finite precision can change the last few bits, just as two valid GPU reductions can differ slightly.

## Why memory usage becomes linear

The quadratic **arithmetic** remains: every row of $Q$ still interacts with every row of $K$. The quadratic score matrix, however, is not stored in HBM. Inputs, outputs, and row-wise statistics are retained, so auxiliary memory grows as $O(N)$ rather than $O(N^2)$.

During backward, a conventional implementation might retain $P$ from the forward pass. FlashAttention instead saves compact normalization statistics and recomputes local tiles. This adds some FLOPs but eliminates expensive reads of a stored square matrix. On a GPU, additional arithmetic can be cheaper than additional HBM traffic.

## What FlashAttention-2 changed

The first version already reduced HBM traffic, but it did not always distribute work evenly across streaming multiprocessors and required communication among warps within a block. FlashAttention-2 improves partitioning at two scales.

1. In addition to batch and heads, it parallelizes across sequence length. This matters for long contexts with a small batch, where the earlier number of thread blocks cannot occupy every SM.
2. Within a thread block, `sliced-Q` replaces `sliced-K`: each warp receives its own portion of $Q$, while all warps can access $K,V$. Each warp can produce an independent output slice without reducing intermediate results through shared memory.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/flashattention2-partitioning.png]]

*Left: FlashAttention partitions $K,V$ across warps and then combines their results. Right: FlashAttention-2 partitions $Q$, giving each warp an independent output slice. Original diagram by Tri Dao, [Stanford CRFM](https://crfm.stanford.edu/2023/07/17/flash2.html).*

FlashAttention-3 specializes execution for Hopper, exploiting asynchrony and the lower-precision modes of newer tensor cores. FlashAttention-4 in the official repository uses CuTeDSL for Hopper and Blackwell. A higher version number does not guarantee a speedup on every GPU: the architecture, head dimensions, data type, mask, and selected kernel must all be supported.

## Training, prefill, and decode

Training and long prefill provide many rows of $Q$, allowing the tiled algorithm to use matrix units effectively while eliminating large intermediates. Decode generally has only one new row of $Q$ and reads $K,V$ from a growing KV cache. This is a different geometry. FlashDecoding and other specialized kernels split a long KV cache in parallel and merge partial results.

Thus, saying that a model “uses FlashAttention” is insufficient to characterize serving. One must identify the prefill kernel, decode kernel, KV-cache representation and placement, GQA/MQA configuration, and scheduler.

## FlashAttention is not an attention architecture

| mechanism | what it changes | result relative to dense attention |
|---|---|---|
| FlashAttention | evaluation order and data movement | the same dense attention |
| sliding-window attention | set of visible keys | a different mask and less work |
| block-sparse attention | blocks that are evaluated | approximate or structurally different attention |
| linear attention | aggregation equation itself | a different architecture |
| PagedAttention | physical placement of the KV cache | the same logical keys and values |

FlashAttention can implement a causal or sliding-window mask, but kernel efficiency and an architectural restriction on context are distinct design choices.

## Verifying that the kernel is actually used

PyTorch's [`scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) may select a flash, memory-efficient, or math backend depending on the device, data type, tensor shapes, and parameters. The absence of an error does not prove that the flash backend ran: the implementation may silently fall back.

Verification should include:

- a profiler showing the kernel that actually executed;
- peak allocated memory at several values of $N$;
- separate forward and backward timings;
- identical masks, dropout, data types, and head dimensions;
- output and gradient comparison against a reference within an appropriate numerical tolerance.

A benchmark of one attention operation is not a model-level speedup. If MLPs, communication, or decode-time KV-cache reads dominate execution, the end-to-end gain will be smaller than the local kernel gain.

## Sources and further reading

- Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135) — the IO-aware formulation, algorithm, and proof of exactness.
- Tri Dao, [FlashAttention-2 at Stanford CRFM](https://crfm.stanford.edu/2023/07/17/flash2.html) — clear diagrams of tiling and warp partitioning.
- [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) — reference implementation, constraints, and numerical-correctness tests.
- PyTorch, [Scaled Dot Product Attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — backend selection in a practical API.
