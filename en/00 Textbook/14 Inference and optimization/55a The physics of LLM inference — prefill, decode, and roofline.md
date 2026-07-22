---
title: "The physics of LLM inference: prefill, decode, and roofline"
type: textbook-chapter
status: canonical
locale: en
translation_of: "00 Учебник/14 Inference и оптимизация/55a Физика LLM inference — prefill, decode и roofline.md"
last_updated: 2026-07-22
last_verified: 2026-07-22
primary_sources:
  - https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_10.py
  - https://arxiv.org/abs/2309.06180
  - https://www.aleksagordic.com/blog/vllm
  - https://arxiv.org/abs/2205.14135
---

# The physics of LLM inference: prefill, decode, and roofline

To a user, a language-model response appears to be one operation. To an accelerator, it consists of two markedly different regimes. The model first processes all input tokens together and constructs their internal representations. It then invokes the same layers repeatedly to produce one new token at a time. The first regime presents the GPU with large matrix multiplications and generally keeps much of its compute hardware busy. In the second, billions of parameters must be read again for comparatively little arithmetic. “Accelerating inference” is therefore not one problem: reducing time to first token and accelerating subsequent generation call for different techniques.

This distinction is best established before examining vLLM, SGLang, or TensorRT-LLM. Engines change schedulers and software interfaces, but not autoregressive dependence, data movement through the memory hierarchy, or bandwidth limits. The computational model and notation below follow [Stanford CS336: Inference](https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_10.py); memory management is checked against [PagedAttention](https://arxiv.org/abs/2309.06180), and the boundaries of a practical serving loop against [Inside vLLM](https://www.aleksagordic.com/blog/vllm).

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

Even a short forward pass must therefore be repeated a thousand times for a thousand-token answer. More GPUs do not remove this dependency; they can only shorten an individual step or serve more requests concurrently.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/vllm-latency_diagram.png]]

*The intervals that make up request latency: queueing, prefill, and sequential decode steps. Original diagram by Aleksa Gordić, from the Performance section of [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://www.aleksagordic.com/blog/vllm). No explicit license is stated on the page.*

## How much computation does the model perform?

As a first approximation, a forward pass through dense parameters requires about two floating-point operations per parameter per token: one multiplication and one addition. For a model with $P$ parameters, the linear layers therefore require

$$
F_{\text{dense}}\approx 2PN,
$$

where $N$ is the number of tokens processed by the pass. For prefill, $N=BS$; for one decode step, $N=B$. This estimate gives the right scale: it covers the dominant attention projections and MLPs, but omits attention's quadratic term, normalizations, token selection, and systems overhead.

For one layer and one sequence, constructing $Q,K,V$ and applying the output projection costs $O(Sd^2)$, as does the MLP. The two attention matrix multiplications require approximately

$$
F_{\text{attn,prefill}}\approx 4S^2d
$$

operations: roughly $2S^2d$ for $QK^\top$ and the same amount for multiplying probabilities by $V$. For $L$ layers and batch size $B$, this term is multiplied by $LB$. Parameterized matrix layers dominate at moderate context lengths, but the quadratic term becomes material as $S$ grows.

At decode, the new query is compared with all $S$ cached keys and used to aggregate their values. The attention cost for one layer is then

$$
F_{\text{attn,decode}}\approx 4Sd,
$$

not $4S^2d$, because there is only one query. Decode is nevertheless not necessarily cheap. Its linear layers still require reading the entire model, and attention must read an ever-growing KV cache. Step time is governed by bytes moved as well as FLOPs.

## Roofline: arithmetic or memory sets the limit

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

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/vllm-roofline.png]]

*Roofline and representative prefill and decode workloads. Original diagram by Aleksa Gordić, from the Performance section of [Inside vLLM](https://www.aleksagordic.com/blog/vllm); Stanford CS336 uses the same definitions. No explicit license is stated on the page.*

Prefill turns projections into large GEMMs: each loaded weight tile serves many input rows. The number of operations per byte rises, often placing prefill near the compute-bound portion of the roofline. Decode with small $B$ resembles matrix–vector multiplication: weights are loaded for only one or a few rows. If weights occupy $M_W$ bytes, a simplified decode pass has $F\approx2PB$ while weight traffic remains about $M_W$. A larger batch raises intensity because the same weights serve several requests:

$$
I_{\text{decode}}\approx\frac{2PB}{M_W+M_{KV,\text{read}}}.
$$

This is the principal benefit of batching during generation. It does not reduce the work per request; it amortizes weight reads. The gain ends when computation reaches the arithmetic ceiling, KV-cache memory is exhausted, or the larger batch violates latency requirements.

Roofline is an upper bound, not an exact timing model. Tile sizes, occupancy, kernel launch overhead, synchronization, inter-GPU communication, uneven sequence lengths, and the CPU scheduler all matter. Even so, it explains why equal-FLOP methods can take different amounts of time: FlashAttention accelerates exact attention by reducing HBM traffic, while weight quantization can accelerate decode by reducing bytes read.

## Lower bounds are useful sanity checks

The same quantities yield a coarse time lower bound:

$$
T_{\min}\geq\max\left(\frac{F}{C_{\max}},\frac{M}{W_{\max}}\right).
$$

For batch-one decode, begin with the weight footprint. A seven-billion-parameter model in BF16 occupies about 14 GB. If a step did nothing but read those weights once from memory with 2 TB/s bandwidth, the read alone would take at least 7 ms; actual layers, KV-cache traffic, and auxiliary work take longer. This is not a benchmark, but it rejects physically impossible claims: a substantially shorter step requires weight reuse, a smaller representation, or a different data placement.

For prefill, the corresponding check starts with input-token count and FLOPs. If $2PS$ approaches the device's compute limit, eliminating a few gigabytes of traffic cannot yield a proportional speedup. For short decode, reducing precision from BF16 to 8 or 4 bits can reduce the dominant $M$ term—but only if the device executes that format without a costly standalone unpacking pass.

Batching changes both bounds. Weight matrices serve the whole batch, reducing bytes per request, but FLOPs grow with $B$, and aggregate KV history can become larger than the weights. Throughput therefore tends to rise rapidly with batch size and then plateau. Beyond that point, a larger batch consumes memory and worsens latency with little useful throughput gain. This plateau, rather than the maximum number of requests that fit, is a sound starting point for server tuning.

## The KV cache trades recomputation for memory

For causal attention at a new position $t$,

$$
q_t=x_tW_Q,\qquad k_t=x_tW_K,\qquad v_t=x_tW_V,
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

At least two independent quantities matter to a user. **TTFT** (*time to first token*) runs from request submission to receipt of the first token:

$$
\operatorname{TTFT}=T_{queue}+T_{schedule}+T_{prefill}+T_{sample,1}.
$$

**TPOT** (*time per output token*) is the mean interval during subsequent generation. The related **ITL** (*inter-token latency*) retains each individual interval and exposes tail behavior. For an answer of $T$ tokens,

$$
T_{E2E}\approx\operatorname{TTFT}+(T-1)\operatorname{TPOT},
$$

although an average conceals pauses in particular iterations. Interactive chat may tolerate moderate TTFT but feel erratic when ITL spikes; offline summarization cares more about aggregate throughput.

**Throughput** is the number of requests or tokens completed per unit time. It can rise with batch size while per-request latency deteriorates. Serving systems therefore use **goodput**: work completed while meeting constraints such as TTFT below 500 ms and TPOT below 50 ms. A system with the highest tokens/s may have lower goodput if its queues violate the SLO.

## One request competes for several resources

Before model execution, a request has already passed through CPU tokenization and queueing. During prefill it competes for arithmetic units and workspace; during decode, for HBM bandwidth and room for the growing cache. Multi-GPU configurations add tensor-parallel collectives and pipeline-parallel activation transfers. Logit processing, sampling, and detokenization follow the forward pass. With small models or very short batches, this auxiliary work can be a material fraction of latency.

Measurements must therefore vary workload as well as model: input/output length distributions, arrival rate, concurrent clients, and repeated-prefix frequency. A tokens-per-second result without these conditions does not transfer to another service.

## Optimizations implied by the computational model

The two-phase physics classifies methods by the bottleneck they remove. Batching reuses weights during decode; quantization reduces weight and cache traffic; FlashAttention avoids unnecessary intermediate-matrix transfers; PagedAttention reduces wasted memory; chunked prefill bounds the duration of a compute-heavy iteration; and disaggregated prefill/decode permits each regime to use suitable resources and parallelism.

None is an unconditional speedup. Large batches improve throughput but may worsen TPOT. Small KV blocks reduce fragmentation but complicate addressing. Quantization helps only when an efficient kernel supports the format. Phase disaggregation introduces KV-cache transfer over the network. The correct starting question is whether the server is limited by arithmetic, memory bandwidth, memory capacity, communication, or scheduling. That answer connects this chapter's equations to the mechanisms that follow.

## Sources

- Stanford CS336, [Lecture 10: Inference](https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_10.py) — computational model, roofline, KV cache, and serving metrics.
- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — KV-cache memory analysis, fragmentation, and PagedAttention.
- Aleksa Gordić, [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://www.aleksagordic.com/blog/vllm) — request processing and the latency and roofline diagrams.
- Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135) — IO-aware analysis of exact attention.
