---
title: "The physics of LLM inference: prefill, decode, and roofline"
type: textbook-chapter
status: canonical
locale: en
translation_status: current
canonical_revision: "dcdc7f9712614b5342b15d03cba1beb7f641ea72b27fe6ee04730ad70fd66bdc"
translation_reviewed: "2026-10-29"
translation_of: "00 Учебник/14 Inference и оптимизация/55a Физика LLM inference — prefill, decode и roofline.md"
last_updated: 2026-10-29
last_verified: 2026-07-22
primary_sources:
  - https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_10.py
  - https://arxiv.org/abs/2309.06180
  - https://www.aleksagordic.com/blog/vllm
  - https://arxiv.org/abs/2205.14135
---

# The physics of LLM inference: prefill, decode, and roofline

Generating an answer begins with one pass over the input, followed by passes that process one newly selected token at a time. During the input pass, each weight matrix serves many positions. During generation, the same matrix is needed again for far fewer new positions. A GPU can finish the arithmetic quickly and still spend most of the step waiting for weights to arrive from memory. Fast processing of a long prompt therefore does not imply short intervals between output tokens.

Suppose one user's request runs on a model whose weights occupy 14 GB. Most of those weights must be read again for its next token. Generating for several users together lets the loaded weights serve several rows. Each user still has a separate attention history, and that history grows with the answer. Batching amortizes weight reads per request without eliminating reads of that request's history. This is the distinction developed in [Stanford CS336](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_10.py) and the [JAX Scaling Book](https://jax-ml.github.io/scaling-book/inference/).

## An autoregressive request has two phases

Suppose tokenization produces an input of $S$ tokens. During **prefill**, the model receives the entire hidden-state tensor

$$
X\in\mathbb{R}^{B\times S\times d},
$$

where $B$ is the batch size and $d$ is the model width. A causal mask prevents positions from attending to the future, but it does not force the positions to be computed sequentially: projections, normalizations, MLPs, and all permitted attention entries can be evaluated in one pass through the layers. Prefill produces the logits for the first output token and stores attention keys and values for every input position.

During **decode**, each active request contributes only one new position. The model computes its hidden state, obtains the next-token distribution, selects a token, and repeats. Output positions $y_1,y_2,\ldots$ cannot be evaluated in parallel because

$$
p(y_{1:T}\mid x)=\prod_{t=1}^{T}p(y_t\mid x,y_{<t}).
$$

A thousand-token answer therefore requires prefill for the first token and
999 subsequent ordinary decode steps. More GPUs do not remove this dependency; they can only shorten an individual step or serve more requests concurrently.

Without a KV cache, each new step would run the entire growing prefix through the Transformer again. The first source animation shows that repeated work after each token selection.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/inference-evaluation/naive-inference.webp]]

*Jacob Austin et al., [JAX Scaling Book: Inference](https://jax-ml.github.io/scaling-book/inference/). Naive autoregressive generation processes the whole prefix again. The same illustration appears in [CS336 Lecture 10](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_10.py#L162-L286).*

A KV cache separates the single prefill from the short decode passes. Each pass computes K/V only for its new position and reads previously computed K/V from memory.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/inference-evaluation/cached-inference.webp]]

*Jacob Austin et al., [JAX Scaling Book: Inference](https://jax-ml.github.io/scaling-book/inference/); also used in CS336. Red marks prefill and blue marks decode. The old keys and values are reused, not recomputed.*


## Notation and the cost of one matrix multiplication

Let $B$ be the number of sequences, $S$ the available context length, and $T$ the positions processed in the current pass. Model width is $d$, MLP width is $d_{ff}$, query/KV head counts are $H_q,H_{kv}$, head width is $d_h$, layer count is $L$, and one stored scalar occupies $b$ bytes. Thus $d=H_qd_h$. Usually $T=S$ for prefill and $T=1$ for ordinary decode. In attention formulas, $S$ includes the current K/V position if it has already been appended; it is not always the original prompt length.

For $X\in\mathbb R^{B\times d}$ and $W\in\mathbb R^{d\times d_{ff}}$, the product $Y=XW$ requires

$$
F=2Bd d_{ff}.
$$

A multiply and an add count as two FLOPs. If BF16 inputs are each read exactly once from HBM and the result written once, the traffic and intensity are

$$
\begin{aligned}
M&=2Bd+2dd_{ff}+2Bd_{ff},\\
I&=\frac{Bd d_{ff}}{Bd+dd_{ff}+Bd_{ff}}.
\end{aligned}
$$

Only when $d,d_{ff}\gg B$ does this approach $B$ FLOP/byte. Several rows share one loaded weight matrix. This is an ideal model of one GEMM, excluding normalization, nonlinearities, residual traffic, launches, TP collectives, and accelerator-cache reuse.

## How much computation does the model perform?

As a first approximation, a forward pass through dense weight matrices requires about two floating-point operations per parameter per token: one multiplication and one addition. If those linear-layer matrices contain $P$ parameters, they require

$$
F_{\text{dense}}\approx 2PN,
$$

where $N$ is the number of tokens processed by the pass. For prefill, $N=BS$; for one decode step, $N=B$. This estimate gives the right scale: it covers the dominant attention projections and MLPs, but omits attention's quadratic term, normalizations, token selection, and systems overhead.

Input embedding lookup needs separate treatment: reading one indexed row does not multiply by the whole embedding table. The output vocabulary projection generally does use the whole matrix. Tying these weights saves storage but does not remove the output projection. For MoE, token arithmetic uses the selected experts, while model storage includes every resident expert.


For one layer and one sequence, constructing $Q,K,V$ and applying the output projection costs $O(Sd^2)$, as does the MLP. The two attention matrix multiplications require approximately

$$
F_{\text{attn,prefill}}\approx 4S^2d
$$

operations if both full square products are evaluated: roughly $2S^2d$ for
$QK^\top$ and the same amount for multiplying probabilities by $V$. A causal
implementation that skips forbidden pairs performs about $2S(S+1)d$ operations
instead, since only $S(S+1)/2$ pairs are allowed. Both counts remain quadratic;
tiling and reduced HBM traffic do not remove this arithmetic. For $L$ layers
and batch size $B$, the term is multiplied by $LB$. Parameterized matrix layers dominate at moderate context lengths, but the quadratic term becomes material as $S$ grows.

At decode, the new query is compared with all $S$ cached keys and used to aggregate their values. The attention cost for one layer is then

$$
F_{\text{attn,decode}}\approx 4Sd,
$$

not $4S^2d$, because there is only one query. Decode is nevertheless not necessarily cheap. Its linear layers still require their weight matrices, and attention must read an ever-growing KV cache. Step time is governed by bytes moved as well as FLOPs.

### MLP traffic depends on the executed operations

A gated MLP has two expanding projections and one contracting projection:

$$
\begin{aligned}
U&=XW_{\mathrm{up}},\\
G&=XW_{\mathrm{gate}},\\
H&=\operatorname{SiLU}(G)\odot U,\\
Y&=HW_{\mathrm{down}}.
\end{aligned}
$$

For $N=BT$ rows, the three GEMMs require $F_{\mathrm{MLP}}=6Nd d_{ff}$. Fusion need not change this arithmetic, but can change the transfers. First assume four separate operations, all intermediate tensors stored in HBM, and every listed access reaching HBM:

| Operation | Reads | Writes | BF16 bytes |
|---|---|---|---:|
| $U=XW_{\mathrm{up}}$ | $X,W_{\mathrm{up}}$ | $U$ | $2Nd+2dd_{ff}+2Nd_{ff}$ |
| $G=XW_{\mathrm{gate}}$ | $X$ again, $W_{\mathrm{gate}}$ | $G$ | $2Nd+2dd_{ff}+2Nd_{ff}$ |
| $H=\operatorname{SiLU}(G)\odot U$ | $G,U$ | $H$ | $6Nd_{ff}$ |
| $Y=HW_{\mathrm{down}}$ | $H,W_{\mathrm{down}}$ | $Y$ | $2Nd_{ff}+2dd_{ff}+2Nd$ |

Adding the transfers gives

$$
\begin{aligned}
M_{\mathrm{MLP,separate}}&=6Nd+12Nd_{ff}\\
&\qquad{}+6dd_{ff}.
\end{aligned}
$$

Writing $U$ or $G$ does not replace its later read: both are transfers. Counting only the input and output of the whole layer omits these intermediates.

Now suppose a fused kernel reads $X$ once, computes both expanding projections, and forms $H$ without storing complete $U$ and $G$ in HBM. Only $H$ is written and later read by the contracting GEMM. With ideal one-time weight reads,

$$
\begin{aligned}
M_{\mathrm{MLP,fused}}&=4Nd+4Nd_{ff}+6dd_{ff},\\
I_{\mathrm{MLP,fused}}&=
\frac{3Nd d_{ff}}{2Nd+2Nd_{ff}+3dd_{ff}}.
\end{aligned}
$$

This is a particular fusion model, not a guarantee about every MLP implementation. Tile reloads, L2 hits and GEMM epilogues affect actual HBM traffic. The arithmetic alone does not establish that a matching kernel exists in an engine or is faster.

The [CS336 calculation](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_10.py#L182-L213) uses the same short byte expression, but its separately listed writes of $U,G$ are not accompanied by their later reads. That expression cannot be a complete ledger for the displayed separate operations. Choose the actual kernel sequence and intermediate-storage locations before interpreting a profile.

For $d=5120,d_{ff}=13824$:

| Rows $N$ | Three GEMMs | Separate traffic | Fused-model traffic | Intensity: separate / fused |
|---:|---:|---:|---:|---:|
| 1 | 0.425 GFLOP | 424.87 MB | 424.75 MB | 1.00 / 1.00 FLOP/byte |
| 64 | 27.18 GFLOP | 437.26 MB | 429.52 MB | 62.16 / 63.28 FLOP/byte |
| 1024 | 434.87 GFLOP | 626.00 MB | 502.27 MB | 694.67 / 865.80 FLOP/byte |

Here MB means $10^6$ bytes. With one row, weights dominate and eliminating intermediate transfers barely changes the estimate. With 1024 rows, activation traffic matters. The same optimization therefore need not help prefill and decode equally.

<!-- example: inference-mlp-traffic -->
```python
def mlp_cost(rows, width, hidden):
    flops = 6 * rows * width * hidden
    separate = (
        6 * rows * width
        + 12 * rows * hidden
        + 6 * width * hidden
    )
    fused = (
        4 * rows * width
        + 4 * rows * hidden
        + 6 * width * hidden
    )
    return flops, separate, fused

assert mlp_cost(64, 5120, 13824) == (
    27_179_089_920, 437_256_192, 429_522_944
)
```

The BF16 approximation $I_{\mathrm{MLP}}\approx BT$ applies only while weights dominate activation traffic. Long prefill supplies many rows naturally; $T=1$ decode needs batching to reuse weights.

### Attention reads each request's own history

For an MHA attention core, the $QK^\top$ and $AV$ products require

$$
F_{\mathrm{attn}}=4BSTd.
$$

This counts all query heads because $d=H_qd_h$. Optimistically reading Q, K and V once and writing the result once gives

$$
\begin{aligned}
M_{\mathrm{attn,MHA}}&=2bBd(S+T),\\
I_{\mathrm{attn,MHA}}&=\frac{2ST}{b(S+T)}.
\end{aligned}
$$

This is a minimum input/output ledger, not a universal FlashAttention traffic formula: tiled execution may reload K/V. FlashAttention avoids writing the full score/probability matrices to HBM; avoiding that traffic is not the same as reading K/V exactly once.

For BF16, intensity is $ST/(S+T)$. With $T=S$ it is $S/2$ under the same full-square FLOP convention; skipping causal pairs roughly halves the numerator and prefill intensity. For decode, $T=1$ gives $S/(S+1)<1$. That last bound applies to the **MHA attention core**, excluding projections, softmax, new-KV writes and communication. It is not a universal GQA bound.

With GQA, Q and output remain width $d=H_qd_h$, but stored K/V have width $H_{kv}d_h$. Under the same assumptions,

$$
\begin{aligned}
M_{\mathrm{attn,GQA}}&=2bB(Td+SH_{kv}d_h),\\
I_{\mathrm{attn,GQA}}&=\frac{2STd}{b(Td+SH_{kv}d_h)}.
\end{aligned}
$$

For long-context BF16 decode, ideal shared-K/V reuse makes intensity approach $H_q/H_{kv}$, not a value below one. Batch size cancels from this ideal per-request-history model. It still affects occupancy, launches and scheduling in real kernels. Query heads must actually reuse the loaded K/V: separate reloads for different compute groups reduce the traffic benefit.

## Roofline: arithmetic or memory sets the limit

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/performance/prefill-vs-decode-roofline.svg]]

*Vijay Janapa Reddi et al., [Harvard ML Systems: Performance Engineering](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/performance_engineering/images/svg/prefill-vs-decode-roofline.svg), CC BY-NC-SA 4.0. The long-prefill and batch-one-decode points are schematic, not measured results or claims about every batch size.*


The roofline model relates three quantities. Let a device sustain at most $C_{\max}$ FLOP/s and read HBM at $W_{\max}$ bytes/s. For an operation requiring $F$ FLOPs and moving $M$ bytes, its **arithmetic intensity** is

$$
I=\frac{F}{M}\quad\text{FLOP/byte}.
$$

Attainable performance is bounded by

$$
C\leq\min(C_{\max},IW_{\max}).
$$

The boundary

$$
I^*=\frac{C_{\max}}{W_{\max}}
$$

is the roofline ridge point. To its left, an operation is memory-bandwidth-bound: compute units wait for data. To its right, peak arithmetic throughput is the constraint.

CS336 uses nominal H100 SXM ceilings of 989 TFLOP/s for dense BF16 Tensor Core arithmetic, without structured sparsity, and 3.35 TB/s HBM bandwidth. Their ratio is about 295 FLOP/byte. The ridge belongs to that precision and device variant. It does not predict that a batch of 295 rows saturates a real GPU: matrix shapes, kernels and achieved bandwidth still matter.


![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/vllm-roofline.png]]

*Arithmetic and memory-bandwidth ceilings. Implementations may operate below either ceiling. Original diagram by Aleksa Gordić, from the Performance section of [Inside vLLM](https://www.aleksagordic.com/blog/vllm); Stanford CS336 uses the same definitions. No explicit license is stated on the page.*

Prefill turns projections into large GEMMs: each loaded weight tile serves many input rows. The number of operations per byte rises, often placing prefill near the compute-bound portion of the roofline. Decode with small $B$ resembles matrix–vector multiplication: weights are loaded for only one or a few rows. If weights occupy $M_W$ bytes, a simplified decode pass has $F\approx2PB$ while weight traffic remains about $M_W$. A larger batch raises intensity because the same weights serve several requests:

$$
I_{\text{decode}}\approx\frac{2PB}{M_W+M_{KV,\text{read}}}.
$$

The numerator here counts only linear layers. Add attention FLOPs when they are material; otherwise this ratio is not the intensity of the complete pass.

This is the principal benefit of batching during generation. It does not reduce the work per request; it amortizes weight reads. The gain ends when computation reaches the arithmetic ceiling, KV-cache memory is exhausted, or the larger batch violates latency requirements.

Roofline is an upper bound, not an exact timing model. Tile sizes, occupancy, kernel launch overhead, synchronization, inter-GPU communication, uneven sequence lengths, and the CPU scheduler all matter. Even so, it explains why equal-FLOP methods can take different amounts of time: FlashAttention accelerates exact attention by reducing HBM traffic, while weight quantization can accelerate decode by reducing bytes read.

## Lower bounds are useful sanity checks

The same quantities yield a coarse time lower bound:

$$
T_{\min}\geq\max\left(\frac{F}{C_{\max}},\frac{M}{W_{\max}}\right).
$$

For batch-one decode, begin with the weight footprint. A seven-billion-parameter model in BF16 occupies about 14 GB. If a step did nothing but read those weights once from memory with 2 TB/s bandwidth, the read alone would take at least 7 ms; actual layers, KV-cache traffic, and auxiliary work take longer. This is not a benchmark, but it rejects physically impossible claims: a substantially shorter step requires weight reuse, a smaller representation, or a different data placement.

For prefill, the corresponding check starts with input-token count and FLOPs. If linear-layer execution already dominates and its time is close to $2PS/C_{\max}$, eliminating a few gigabytes of traffic cannot yield a proportional speedup. For short decode, reducing precision from BF16 to 8 or 4 bits can reduce the dominant $M$ term—but only if the device executes that format without a costly standalone unpacking pass.

Batching changes both bounds. Weight matrices serve the whole batch, reducing bytes per request, but FLOPs grow with $B$, and aggregate KV history can become larger than the weights. Throughput therefore tends to rise rapidly with batch size and then plateau. Beyond that point, a larger batch consumes memory and worsens latency with little useful throughput gain. This plateau, rather than the maximum number of requests that fit, is a sound starting point for server tuning.

### Worked estimate: Llama 2 13B on one H100

Lecture 10 uses $S=1024,d=5120,d_{ff}=13824$, $H_q=H_{kv}=40$, $d_h=128$, $L=40$, vocabulary $V=32000$, BF16, and peak HBM bandwidth 3.35 TB/s. Its large-matrix parameter model is

$$
\begin{aligned}
P&\approx 2Vd+L(3dd_{ff}+2dH_qd_h\\
&\qquad{}+2dH_{kv}d_h).
\end{aligned}
$$

It gives $13\,015\,449\,600$ parameters and about 26.03 GB of weights, excluding small normalization parameters. The input embedding and output vocabulary matrices are stored separately. One request's full-MHA cache occupies

$$
\begin{aligned}
M_{KV,1}&=2bLSH_{kv}d_h\\
&=838\,860\,800\ \text{bytes}.
\end{aligned}
$$

Separate **resident storage** from **step traffic**. Weights and caches occupy $2P+B M_{KV,1}$ bytes, but one input lookup reads a row rather than the whole $Vd$ table. Linear layers use $P_{\mathrm{linear}}=P-Vd$ parameters. Reading those weights, all used K/V, and one embedding row per request once gives

$$
M_{\mathrm{read}}=2(P-Vd)+B M_{KV,1}+2Bd.
$$

Repeated tokens could reuse embedding rows from cache; here all counted reads are assumed to reach HBM. New-KV writes, other activations and overhead are omitted. Consequently $M_{\mathrm{read}}/W_{\max}$ estimates only the chosen reads at peak bandwidth, not exact generation time.

| $B$ | Resident weights + KV | Counted step reads | Reads at 3.35 TB/s | $B/t$, tokens/s |
|---:|---:|---:|---:|---:|
| 1 | 26.87 GB | 26.54 GB | 7.92 ms | 126 |
| 64 | 79.72 GB | 79.39 GB | 23.70 ms | 2701 |
| 256 | 240.78 GB | 240.45 GB | 71.78 ms | 3567 |

GB is decimal. The 256-request case cannot fit in 80 GB HBM. Even 64 requests are not established as feasible: almost all nominal memory is already occupied, before workspace, temporary tensors and future cache growth. The last column is not measured or promised throughput.

<!-- example: inference-llama-budget -->
```python
def decode_budget(batch, kv_heads=40, context=1024):
    width, hidden, layers = 5120, 13824, 40
    vocab, query_heads, head_dim = 32000, 40, 128
    params = 2 * vocab * width + layers * (
        3 * width * hidden
        + 2 * width * query_heads * head_dim
        + 2 * width * kv_heads * head_dim
    )
    kv_bytes = 2 * 2 * layers * context * kv_heads * head_dim
    allocated = 2 * params + batch * kv_bytes
    read = (
        2 * (params - vocab * width)
        + batch * kv_bytes
        + 2 * batch * width
    )
    seconds = read / 3.35e12
    return allocated, read, seconds, batch / seconds

allocated, read, seconds, tokens_s = decode_budget(64)
assert allocated == 79_717_990_400
assert read == 79_390_965_760
assert round(seconds * 1000, 2) == 23.70
```

The simplified CS336 estimate counts the full resident weights and KV as reads as well. This explains the small difference: 26.87 GB is batch-one residency, not a mandatory full-table input-embedding read. The accounting convention should remain explicit even when the timing difference is small.

Replacing MHA with $H_{kv}=8$ reduces both the caches and K/V projection matrices. In this hypothetical calculation, $B=64$ occupies 33.41 GB, reads 33.09 GB and needs 9.88 ms for those reads at peak bandwidth. At $B=256$, the respective numbers are 65.63 GB, 65.30 GB and 19.49 ms. This is a **different architecture**, not an allocator setting that speeds up the original Llama 2. A real comparison must control workload and identify the architecture, kernels, available memory and achieved bandwidth.

## Model bandwidth utilization: bytes per measured step

Suppose the selected model of a decode step requires 14 GB of reads and the measured step takes 14 ms. The ratio is 1 TB/s. On a device whose peak is 2 TB/s, this is 50% **model bandwidth utilization (MBU)**. Megha Agarwal et al. give this example for a 7B model in 16-bit precision, neglecting its small KV cache, in [MosaicML / Databricks](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices).

Let $M_{\mathrm{est}}$ be estimated bytes for one complete step, $t$ the duration of that same step, and $W_{\max}$ one device's peak memory bandwidth:

$$
\operatorname{MBU}_{\mathrm{model}}
=\frac{M_{\mathrm{est}}}{tW_{\max}}.
$$

The example gives $14\cdot10^9/(0.014\cdot2\cdot10^{12})=0.5$, with decimal GB/TB. Its numerator comes from a traffic model, not hardware counters. State whether it counts all resident parameters or used matrices, occupied KV positions or the entire pool, and whether writes/intermediates are included. The Llama example above shows why a whole input embedding table should not automatically enter every step's reads.

### Converting step time to throughput requires batch size

An ordinary decode step processes one position for each of $B$ sequences. With constant batch size, no speculation and only these steps in the measurement window, token rate is $R=B/t$. Substituting $t=B/R$ gives

$$
\operatorname{MBU}_{\mathrm{model}}
=\frac{R M_{\mathrm{est}}}{B W_{\max}}.
$$

Here $M_{\mathrm{est}}$ belongs to the entire batch. Dividing by $B$ preserves weight sharing. Four requests can use the same 14 GB of weights in a 20 ms step: $R=4/0.020=200$ tokens/s. With negligible KV, MBU is $200\cdot14/(4\cdot2000)=35\%$. Omitting the batch divisor produces 140%, incorrectly charging the shared weight read to every token.

<!-- example: inference-mbu-batch -->
```python
model_bytes = 14_000_000_000  # weights only; KV ignored in this example
peak_bytes_s = 2_000_000_000_000
for batch, seconds in [(1, 0.014), (4, 0.020)]:
    tokens_s = batch / seconds
    mbu = model_bytes / (seconds * peak_bytes_s)
    from_throughput = tokens_s * model_bytes / (batch * peak_bytes_s)
    assert abs(mbu - from_throughput) < 1e-12
    print(batch, f"{tokens_s:.2f}", f"{100 * mbu:.2f}%")
```

| Batch $B$ | Stated step time | Token rate | Weight-only MBU |
|---:|---:|---:|---:|
| 1 | 14 ms | 71.43 tokens/s | 50.00% |
| 4 | 20 ms | 200.00 tokens/s | 35.00% |

These are stipulated times for checking the formula, not benchmark measurements. The second row has higher throughput but lower MBU: one weight load supports four rows and more arithmetic. Maximizing MBU alone is not a batching objective; latency and memory constraints remain.

With continuous batching, $B$, KV lengths and work composition change. An iteration containing a prefill chunk is not pure decode. Speculative verification processes several candidates, with an accepted-token count different from processed positions. Count bytes for the actual iterations rather than reconstructing them from client-average TPOT and one “average batch size.”

### Selecting the numerator for KV, quantization and MoE

For a dense model with one-time reads, start with $M_{\mathrm{est}}=M_W+\sum_j M_{KV,j}$, as in [JAX Scaling Book](https://jax-ml.github.io/scaling-book/inference/#theoretical-estimates-for-llm-latency-and-throughput). $M_{KV,j}$ is the used history of request $j$, not its reserved cache capacity. A short request does not read every unoccupied block. Shared prefixes require another assumption: do different requests reread their shared blocks, or reuse the data? One physical prefix copy does not guarantee one read per step.

Quantized weights include packed values and the scales, zero points and other metadata required by their format. Parameter count times four bits omits those fields. If a separate kernel unpacks a matrix into HBM, that write and the subsequent read also create traffic.

For MoE, the selected expert union depends on routing across the batch. Consider 2 GiB of shared weights and four 1 GiB experts; each of two tokens chooses one expert. Six GiB are resident. With ideal reuse, routes $(0,0)$ need 3 GiB of weight reads; routes $(0,1)$ need 4 GiB.

| Routes | Expert union | One-time weight reads |
|---|---|---:|
| 0, 0 | {0} | 3 GiB |
| 0, 1 | {0, 1} | 4 GiB |

Using all resident 6 GiB overcounts both cases. Adding per-token active parameters also gives $3+3$ GiB and double-counts shared weights and possibly the same expert. The union of used matrices is the intended model. A kernel may still reload some of those matrices; the estimate does not prove one-time HBM reads. Here $1\ \mathrm{GiB}=2^{30}$ bytes.

<!-- example: inference-mbu-moe -->
```python
GiB = 2**30
shared_bytes = 2 * GiB
expert_bytes = [GiB] * 4
for routes in [(0, 0), (0, 1)]:
    selected = set(routes)
    read_bytes = shared_bytes + sum(expert_bytes[e] for e in selected)
    print(routes, read_bytes // GiB, "GiB")
```

### Model bytes and hardware counters answer different questions

A profiler can count actual DRAM reads and writes. Over the same interval,

$$
U_{\mathrm{HBM}}
=\frac{M_{\mathrm{DRAM,read}}+M_{\mathrm{DRAM,write}}}
{tW_{\max}}.
$$

If an illustrative 14 ms step transfers 27 GB on a 2 TB/s device, that ratio is $27/28\approx96.43\%$, while model MBU based on 14 GB remains 50%. The device may be busy transferring reloads and intermediate results. High memory utilization can coexist with unnecessary traffic. Cache reuse produces the opposite discrepancy: a logical tensor read need not reach HBM.

[Nsight Compute: Memory and Caches](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#memory) describes the memory levels; its [metrics structure](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-structure) distinguishes peak percentages over active cycles and elapsed cycles. These percentages are not automatically model bytes divided by whole-step time. Counter collection may [replay kernels](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#replay) and alter cache state. Record the collection method, interval and peak definition; see [[00 Учебник/10 ML Systems/07 Profiling ML-нагрузки|profiling ML workloads]].

Model MBU above 100% calls for checking the assumptions, not a claim of exceeding physical peak. Possible causes include counting all MoE experts rather than selected ones or using total batch throughput without dividing by $B$. A Nsight percentage with its own sustained-peak denominator is yet another metric whose definition must be checked.

### Summed GPU bandwidth cannot remove placement imbalance

Suppose two GPUs, each with 1 TB/s bandwidth, work in parallel on one step. The first must read 12 GB, the second 2 GB. They need at least 12 ms and 2 ms respectively. Dividing totals gives $(12+2)/(1+1)=7$ ms, which is unattainable with this placement: unused bandwidth on GPU 2 cannot read weights from GPU 1's HBM.

For fixed placement,

$$
t\geq\max_i\frac{M_i}{W_i}.
$$

At an exactly 12 ms step, aggregate utilization would be $14/(0.012\cdot2000)\approx58.33\%$. Individual ratios are 100% and 16.67%. The low average reflects imbalance, not slow reads on the loaded GPU. Communication, omitted here, adds further constraints.

<!-- example: inference-mbu-ranks -->
```python
rank_bytes = [12_000_000_000, 2_000_000_000]
rank_peaks = [1_000_000_000_000, 1_000_000_000_000]
lower_bound = max(m / w for m, w in zip(rank_bytes, rank_peaks))
aggregate_ideal = sum(rank_bytes) / sum(rank_peaks)
aggregate_mbu = sum(rank_bytes) / (lower_bound * sum(rank_peaks))
assert lower_bound == 0.012
assert aggregate_ideal == 0.007
assert abs(aggregate_mbu - 7 / 12) < 1e-12
print(f"{1000 * lower_bound:.0f} ms", f"{100 * aggregate_mbu:.2f}%")
```

Summing peaks is an aggregate description of simultaneously working devices, with placement stated. Replicated weights/KV belong in the traffic of every reader. Pipeline stages do not execute as one common memory channel: latency analysis must include stage ordering, microbatches and pipeline occupancy rather than divide bytes by the sum of all stage bandwidths.

### Why throughput reaches a plateau

With small batches, weight-loading time changes little while served rows increase. After compute becomes limiting, layer time grows approximately with batch size and $B/t$ stops rising quickly. Appendix A of the JAX Scaling Book demonstrates this on a TPU layer with two matrix multiplications. It is not a measurement of a whole LLM's MBU or a recommended H100 batch size.

<!-- BEGIN SOURCE: JAX inference Appendix A -->
#### Appendix A: How real is the batch size > 240 rule?

The simple rule we provided above, that our batch size must be greater than 240 tokens to be compute-bound, is roughly true but ignores some ability of the TPU to prefetch the weights while other operations are not using all available HBM, like when doing inter-device communication.

Here's an empirical plot of layer time (in microseconds) for a small Transformer with d<sub>model</sub> 8192, d<sub>ff</sub> 32768, and only 2 matmuls per layer. This comes from [this Colab notebook](https://colab.sandbox.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing). You'll see that step time increases very slowly up until around batch 240, and then increases linearly.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/mbu-review-2026-10/jax-batch-scaling-latency.png]]

Here's the actual throughput in tokens / us. This makes the argument fairly clearly. Since our layer is about 600M parameters sharded 4 ways here, we'd expect a latency of roughly 365us at minimum.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/mbu-review-2026-10/jax-batch-scaling-throughput.png]]

So at least in this model, we do in fact see throughput increase until about BS240 per data parallel shard.
<!-- END SOURCE: JAX inference Appendix A -->

*Jacob Austin et al., [JAX Scaling Book: All About Transformer Inference, Appendix A](https://jax-ml.github.io/scaling-book/inference/#appendix-a-how-real-is-the-batch-size--240-rule), [pinned source](https://github.com/jax-ml/scaling-book/blob/9e2bbee091bf0919a28211a2c6bc5e5d42851dbd/inference.md#appendix-a-how-real-is-the-batch-size--240-rule). Original English passage and two complete unchanged PNGs; [MIT license](/licenses/jax-scaling-book-MIT.txt), copyright Google LLC and Maruan Al-Shedivat.*

The first vertical axis is microseconds per layer; the second is tokens per microsecond. Its plateau near 0.26 tokens/µs corresponds to approximately 260,000 tokens/s **for this layer**, not a complete model pass. The red mark near $B=240$ belongs to this experimental configuration. The author's predicted 365 µs is not a measured point: the blue time curve lies above that estimate.

Even with high MBU, reducing model bytes and actual transfers may help: change a weight format, KV representation or reload schedule, subject to answer-quality checks. Compare MBU together with model configuration, throughput and latency on the same workload. The [[06 Практика/13 Оптимизировать один Transformer step|measurement practice]] records the conditions needed.

## The KV cache trades recomputation for memory

For causal attention at a new position $t$,

$$
\begin{aligned}
q_t&=x_tW_Q,\\
k_t&=x_tW_K,\\
v_t&=x_tW_V,
\end{aligned}
$$

$$
o_t=\operatorname{softmax}\left(\frac{q_tK_{1:t}^{\top}}{\sqrt{d_h}}+M\right)V_{1:t}.
$$

Keys and values at earlier positions do not change once created. The KV cache retains $K_{1:t-1}$ and $V_{1:t-1}$ at every layer, so the next step computes only $k_t$ and $v_t$. Without it, every step would process the whole prefix again, making total generation cost grow much faster.

The computation saved is paid for in memory. Let $L$ be the number of layers, $S$ the current sequence length, $H_{kv}$ the number of key/value heads, $d_h$ the head dimension, and $b$ the bytes per element. Then

$$
M_{KV,1}=2LSH_{kv}d_hb.
$$

The factor of two accounts for separate $K$ and $V$ arrays. For $B$ equal-length sequences,

$$
M_{KV,B}=2BLSH_{kv}d_hb.
$$

With $L=32$, $H_{kv}=8$, $d_h=128$, $S=8192$, and BF16, this is about 1 GiB per request. Eight such requests need roughly 8 GiB for attention history alone, excluding weights, temporary tensors, CUDA Graphs, and bookkeeping. MQA and GQA reduce $H_{kv}$, cache quantization reduces $b$, and MLA changes the history representation itself; none removes linear growth in $S$.

A conventional server loses still more memory if it reserves one contiguous buffer for each request's maximum possible output. Actual lengths are unknown, so reservations remain partly empty and gaps between buffers are difficult to reuse.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/paged-attention-fragmentation.png]]

*Figure 2 from Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180): actual keys and values occupy the left-hand memory map; internal reservations and external fragmentation appear on the right. Authors: Woosuk Kwon et al. Image source: [ar5iv HTML](https://ar5iv.labs.arxiv.org/html/2309.06180).*

PagedAttention addresses placement by dividing logical history into blocks and mapping them onto arbitrary physical blocks. This raises effective batch capacity but does not reduce the useful KV-cache bytes. Paging removes allocator waste; GQA and lower precision shrink the data itself.

## Latency is not reducible to tokens per second

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/vllm-latency_diagram.png]]

*Aleksa Gordić, [Inside vLLM: Performance](https://www.aleksagordic.com/blog/vllm). TTFT runs from submission to the first token, ITL between successive tokens, and complete latency to the last token. These are observed intervals, not a diagram of internal server stages. No explicit license is stated on the page.*

At least two independent quantities matter to a user. **TTFT** (*time to first token*) runs from request submission to receipt of the first token:

$$
\begin{aligned}
\operatorname{TTFT}&=T_{front}+T_{queue}+T_{schedule}\\
&\qquad{}+T_{prefill}+T_{sample,1}.
\end{aligned}
$$

Here the terms denote disjoint portions of the critical path: $T_{front}$ includes tokenization, transport through delivery of the first token, and other frontend work not counted elsewhere; $T_{sample,1}$ accounts for selecting that token. Overlapping work must not be added twice.

**TPOT** (*time per output token*) is the mean interval during subsequent generation. The related **ITL** (*inter-token latency*) retains each individual interval and exposes tail behavior. Here $T$ counts output tokens, rather than positions in one forward pass. For an answer of $T$ tokens,

$$
T_{E2E}=\operatorname{TTFT}+(T-1)\operatorname{TPOT},\qquad T>1,
$$

an exact identity when all metrics use the same client timestamps through the
last token. An average still conceals pauses in particular iterations. For
$T=1$, TPOT is undefined and $T_{E2E}=TTFT$; post-processing after the last token
belongs to a separately defined extended E2E metric. Interactive chat may tolerate moderate TTFT but feel erratic when ITL spikes; offline summarization cares more about aggregate throughput.

For a request sent at time zero, suppose tokens arrive at 0.30, 0.34, 0.38 and 0.78 seconds. TTFT is 300 ms; the three ITLs are 40, 40 and 400 ms; mean TPOT is 160 ms. Completion time is $300+3\cdot160=780$ ms. The average hides the long pause before the last token, so retain the ITL distribution.

If a streaming client returns several tokens in one chunk, the chunk arrival does not reveal each token's timestamp. Chunk-to-chunk intervals cannot be labeled token ITL without qualification. State where timestamps were taken and their granularity.

**Throughput** is the number of requests or tokens completed per unit time. It can rise with batch size while per-request latency deteriorates. Serving systems therefore use **goodput**: work completed per unit time while meeting constraints such as TTFT below 500 ms and TPOT below 50 ms. A system with the highest tokens/s may have lower goodput if its queues violate the SLO.

## One request competes for several resources

Before model execution, a request has already passed through CPU tokenization and queueing. During prefill it competes for arithmetic units and workspace; during decode, for HBM bandwidth and room for the growing cache. Multi-GPU configurations add tensor-parallel collectives and pipeline-parallel activation transfers. Logit processing, sampling, and detokenization follow the forward pass. With small models or very short batches, this auxiliary work can be a material fraction of latency.

Measurements must therefore vary workload as well as model: input/output length distributions, arrival rate, concurrent clients, and repeated-prefix frequency. A tokens-per-second result without these conditions does not transfer to another service.

## Optimizations implied by the computational model

The two-phase physics classifies methods by the bottleneck they remove. Batching reuses weights during decode; quantization reduces weight and cache traffic; FlashAttention avoids unnecessary intermediate-matrix transfers; PagedAttention reduces wasted memory; chunked prefill limits the prompt positions admitted to one iteration; and disaggregated prefill/decode permits each regime to use suitable resources and parallelism.

A token-count limit is not a hardware-independent time limit; the kernel, context and other work in that iteration still determine duration.

None is an unconditional speedup. Large batches improve throughput but may worsen TPOT. Small KV blocks reduce fragmentation but complicate addressing. Quantization helps only when an efficient kernel supports the format. Phase disaggregation introduces KV-cache transfer over the network. The correct starting question is whether the server is limited by arithmetic, memory bandwidth, memory capacity, communication, or scheduling. That answer connects this chapter's equations to the mechanisms that follow.

## Practice and primary sources

- [[05 Источники/Courses/Harvard ML Systems/vol2/inference|Harvard CS249r: Inference]].
- [[05 Источники/Courses/Harvard ML Systems/vol2/performance_engineering|Harvard CS249r: Performance Engineering]].
- [[05 Источники/Courses/Efficient DL Systems/week08_inference_software/lecture.pdf|Efficient DL Systems: inference software]].
- [[05 Источники/Courses/Efficient DL Systems/week06_dl_arithmetic/lecture.pdf|Efficient DL Systems: arithmetic and memory transfers]].

- Stanford CS336, [Lecture 10: Inference](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_10.py) — computational model, roofline, KV cache, and serving metrics.
- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — KV-cache memory analysis, fragmentation, and PagedAttention.
- Aleksa Gordić, [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://www.aleksagordic.com/blog/vllm) — request processing and the latency and roofline diagrams.
- Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135) — IO-aware analysis of exact attention.
