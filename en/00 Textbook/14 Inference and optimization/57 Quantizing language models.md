---
title: "Quantizing language models"
type: textbook-chapter
status: canonical
locale: en
translation_of: "00 Учебник/14 Inference и оптимизация/57 Квантизация языковых моделей.md"
last_updated: 2026-09-15
last_verified: 2026-09-15
primary_sources:
  - https://cs336.stanford.edu/
  - https://huggingface.co/blog/hf-bitsandbytes-integration
  - https://arxiv.org/abs/2210.17323
  - https://arxiv.org/abs/2306.00978
  - https://arxiv.org/abs/2211.10438
---

# Quantizing language models

During token-by-token generation, an accelerator reads almost every model parameter at every step but performs little arithmetic per byte loaded. Representing parameters with four bits instead of sixteen reduces both their footprint and the traffic required to feed compute units. Yet “a 4-bit model” specifies neither numerical error nor actual speed. One must also know how values are grouped, where scales are stored, what precision activations and the KV cache retain, and whether the target device has a kernel for the chosen format.

## Replacing a real number with a code and scale

For a signed, symmetric, uniform $b$-bit grid with $b\ge2$,

$$
q_{\max}=2^{b-1}-1,\qquad
s=\frac{\max_i|w_i|}{q_{\max}},
$$

$$
q_i=\operatorname{clip}\left(\operatorname{round}\frac{w_i}{s},
-q_{\max},q_{\max}\right),\qquad
\hat w_i=sq_i.
$$

Here $q_i$ is a compact integer code and $s$ is the scale that reconstructs the approximation $\hat w_i$. The difference $w_i-\hat w_i$ is quantization error. An all-zero group is handled separately with $q_i=0$ and a nonzero scale, for example $s=1$, to avoid division by zero.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/hf-quantization-rounding.png]]

*Continuous values mapped to discrete levels and reconstructed through a scale. Illustration from Hugging Face, [Making LLMs even more accessible with bitsandbytes](https://huggingface.co/blog/hf-bitsandbytes-integration), also used in [Stanford CS336: Inference](https://cs336.stanford.edu/).*

Let $w=(-1.0,-0.3,0.2,0.9)$ and use a symmetric 3-bit grid: $q_{\max}=3$ and $s=1/3$. Then

$$
q=(-3,-1,1,3),\qquad
\hat w=(-1,-1/3,1/3,1).
$$

If an outlier of magnitude 10 is added to the same group, the step becomes $10/3$ and nearly all small values round to zero. Bit width is therefore only one parameter; the set of values sharing a scale is equally important.

## Symmetric and asymmetric grids

Asymmetric quantization adds a zero point $z$:

$$
q_i=\operatorname{round}(w_i/s)+z,
\qquad
\hat w_i=s(q_i-z).
$$

It can use the code range more efficiently when the distribution is shifted away from zero, but requires storing and applying $z$. The tensor distribution and available kernels determine the choice; asymmetric quantization is not universally superior.

Scale and zero point may be assigned:

- once per tensor (*per-tensor*);
- once per output channel (*per-channel*);
- once per group of $G$ adjacent weights (*group-wise*);
- once per hardware-defined block (*block-wise*).

Smaller groups adapt better to local outliers but add metadata and scaling operations. For 128 four-bit weights and one 16-bit scale, the true storage cost is

$$
4+\frac{16}{128}=4.125\ \text{bits/weight}.
$$

A 16-bit zero point raises this to 4.25 bits. Alignment, headers, and format tables further increase file size slightly.

## What exactly is quantized?

`W4A16` means 4-bit weights and FP16/BF16 activations. In `W8A8`, both matrix-multiplication operands use eight bits. These modes address different constraints.

| mode | what is reduced | where it is commonly useful | principal difficulty |
|---|---|---|---|
| W8A16 | parameters | memory-bound decode | availability of a fast weight-only kernel |
| W4A16 | parameters, more aggressively | capacity and decode | low-bit error and group scales |
| W8A8 | weights and activations | large prefill matrix multiplications | dynamic activation outliers |
| FP8 | floating-point weights/activations | modern tensor cores | choosing a scaling recipe |
| KV8/KV4 | KV cache | long contexts and large batches | accumulation of attention error |

**Weight-only** quantization is a natural fit for decode: it reduces the dominant read traffic while retaining activations in a convenient precision. Large prefill multiplications can be compute-bound, so unpacking W4 into FP16 may not accelerate them. W8A8 and FP8 can use low-precision tensor cores, but activations depend on the input and contain rare high-magnitude channels.

The KV cache is a third, independent object. Quantizing weights does not reduce request history; the KV format must be selected separately. Error in keys changes attention scores, error in values changes the weighted sum itself, and the effect can grow with context length.

## PTQ: quantizing an already trained model

**Post-training quantization** does not repeat full model training. A small calibration set is passed through the model to observe representative activations and choose transformation parameters.

Simple nearest-point rounding minimizes local weight error. A layer, however, computes $Y=XW$, so equally inaccurate weights may affect its output very differently depending on activations $X$.

### GPTQ

[GPTQ](https://arxiv.org/abs/2210.17323) processes input channels sequentially: these are weight columns in the paper's convention $Y=W_{\mathrm{paper}}X_{\mathrm{paper}}$, but rows of $W$ in this chapter's convention $Y=XW$. It compensates for the introduced error in the remaining weights using approximate second-order information. Its objective is to preserve layer outputs on calibration activations. The method became a major recipe for 3–4-bit weight-only PTQ.

### AWQ

[AWQ](https://arxiv.org/abs/2306.00978) builds on the observation that a small fraction of channels are particularly salient: their activations have large magnitudes, so errors in the associated weights strongly perturb the output. Rather than retaining scattered weights at mixed precision, it chooses channel scaling before quantization.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/awq-schema.png]]

*Left: ordinary INT3 rounding sharply degrades perplexity. Center: retaining 1% of weights in FP16 helps but is inconvenient for hardware. Right: AWQ rescales salient channels and quantizes the entire matrix uniformly. Original Figure 2 from Lin et al., [AWQ](https://arxiv.org/abs/2306.00978), reproduced in Stanford CS336.*

The diagram illustrates an important engineering principle: a useful numerical technique must preserve regularity for the device. Arbitrarily scattered FP16 values inside an INT3 matrix can retain quality while destroying an efficient dense kernel.

### SmoothQuant

For W8A8, activation outliers are often harder to quantize than weights. [SmoothQuant](https://arxiv.org/abs/2211.10438) transfers part of the channel scale from activations into weights using an equivalent transformation. For a positive vector $s$ over input channels,

$$
XW=(X\operatorname{diag}(s)^{-1})(\operatorname{diag}(s)W).
$$

Activations become more uniform and weights less so, but static weights are generally easier to quantize per channel. The equality holds before quantization; the advantage arises because error is distributed more favorably between the two transformed operands.

## QAT: exposing the model to error during training

In **quantization-aware training**, the forward pass simulates rounding while a high-precision master copy of the parameters is retained. Because the derivative of `round` is zero almost everywhere, training generally uses a straight-through estimator: the backward pass approximately propagates gradients through the operation.

QAT costs more than PTQ but lets parameters adapt to the grid. It is especially useful at very low bit widths, for activation quantization, and when training directly for a hardware format. A “QAT model” still needs supported packing and a suitable deployment kernel.

## NF4 and QLoRA address fine-tuning

[QLoRA](https://arxiv.org/abs/2305.14314) stores a frozen base model in 4-bit NF4 and trains LoRA adapters. NF4 uses nonuniform levels selected for approximately normally distributed weights, while double quantization compresses the scales themselves.

This primarily reduces **fine-tuning** memory. After training, one may keep base and adapter separate, merge the adapter and run a new PTQ procedure, or use a runtime that supports the composite representation. A QLoRA checkpoint is not automatically an optimal serving artifact.

## File format is not compute format

### FP8: eight bits do not specify one range

In common FP8 formats, one bit stores the sign. **E4M3** allocates four bits
to the exponent and three to the mantissa. The E4M3FN variant used by CUDA
and TensorRT has a maximum finite value of 448, a minimum normal value of
$2^{-6}$, and subnormals down to $2^{-9}$. **E5M2** allocates five exponent
bits and two mantissa bits: its corresponding limits are 57,344, $2^{-14}$,
and $2^{-16}$. E5M2 trades precision for range; E4M3 has more significant
levels within each exponent interval. The exact variant matters because
IEEE-like Inf/NaN and finite-only encodings allocate extreme codes differently.

A tensor is usually scaled: $\hat x=s\,\operatorname{FP8}(x/s)$.
A single per-tensor scale is inexpensive, but a shared maximum magnitude
can make small channels lose resolution. Per-channel or block scales follow
local ranges more closely while changing the layout and kernel contract.
A recipe must specify scale selection, weight and activation formats, and
the separate KV-cache format. A speedup measured for one recipe on one
device does not establish the same gain on another accelerator.

### FP4, MXFP4 and NVFP4

FP4 E2M1 has one sign bit, two exponent bits, and one explicit mantissa bit.
It has no Inf/NaN encoding; saturating overflow maps to the largest finite magnitude. Its finite magnitudes are $\{0,0.5,1,1.5,2,3,4,6\}$. This small codebook alone
cannot describe the range of a model tensor.

[OCP MX v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
defines MXFP4 blocks of 32 E2M1 values with a shared eight-bit E8M0
power-of-two scale. Their payload costs $4+8/32=4.25$ bits per element,
before padding. [NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
instead combines a 16-value block, an FP8 E4M3 block scale, and a global FP32
tensor scale:

$$
\hat x=s_{\mathrm{global}}s_{\mathrm{block}}x_{\mathrm{E2M1}}.
$$

The one-dimensional payload costs $4+8/16=4.5$ bits per element plus the
global scale. These layouts are not interchangeable. In the
[Transformer Engine 2.15 training recipe](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.15/user-guide/features/low_precision_training/nvfp4/nvfp4.html),
activations and gradients use 16-element blocks, while weights default to
16×16 blocks so their rowwise and columnwise quantized versions agree.
Storage estimates must specify which recipe they describe. E2M1 has a maximum magnitude of 6 and E4M3 scales a maximum of 448. A power-of-two scale simplifies multiplication but its rounding can itself add error.

| format | element layout | scale granularity | scale type | finite maximum before scaling |
|---|---|---|---|---:|
| FP8 E4M3FN | 1+4+3 bits | tensor/channel/block, recipe-dependent | recipe-dependent, often FP32 | 448 |
| FP8 E5M2 | 1+5+2 bits | tensor/channel/block | recipe-dependent | 57,344 |
| MXFP4 | E2M1, 4 bits | 32 elements | E8M0, 8 bits | 6 |
| NVFP4 | E2M1, 4 bits | 16 elements, or 16×16 weights | E4M3 block + FP32 global | 6 |

### A shared scale couples neighboring values

Consider two 16-element blocks. Their nonzero values are $(0.5,1,2,6)$ and
$(8,16,32,96)$; all remaining entries are zero. Effective scales $s_A=1$ and
$s_B=16$ give both blocks the same E2M1 codes $(0.5,1,2,6)$, with exact
reconstruction. This is an illustrative block-scaling example, not the full
NVFP4 scale-selection recipe.

Now put all 32 entries under one scale $s=16$. The first block's nonzero
values become $(0.03125,0.0625,0.125,0.375)$ before rounding, map to
$(0,0,0,0.5)$, and reconstruct as $(0,0,0,8)$. The mean squared error over
all 32 entries is $(0.25+1+4+4)/32=0.2890625$, instead of zero. Both scales
are exactly representable: the loss comes from forcing neighbors to share
a range, not from rounding the scale. A real NVFP4 implementation additionally
chooses a global scale, rounds local scales, and obeys its kernel's numerical
contract.

Low precision also applies during pretraining. The [Nemotron 3 Super report,
3 April 2026 revision](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)
describes NVFP4 pretraining. This does not mean every parameter, optimizer
state, and operation uses four bits. The training recipe and the released
checkpoint format are separate facts; success for one model does not
establish stability for another training run.

### Worked execution: weight-only versus weights and activations

Take a projection $W\in\mathbb R^{4096\times4096}$. BF16 weights occupy
32 MiB. Four-bit codes occupy 8 MiB, plus
$16{,}777{,}216/128\cdot2=256$ KiB of FP16 scales for groups of 128.
For decode with one activation row, the BF16 input occupies only 8 KiB:
weight traffic dominates. A fused W4A16 kernel can read approximately
8.25 MiB instead of 32 MiB, unpack groups into registers, and accumulate
in FP16 or FP32, depending on the implementation.

With a 256-token prefill, the BF16 activation matrix occupies 2 MiB and the
GEMM reuses weights much more extensively. A weight-only kernel still saves
HBM traffic, but dequantizes values and often supplies FP16 operands to
tensor cores. W8A8 stores 16 MiB of INT8 weights and 1 MiB of INT8 activations;
a supported INT8 tensor-core kernel can reduce both traffic and multiplication
cost. FP8 W8A8 is a different numerical format with its own scales and kernel,
not an alternative interpretation of these INT8 codes. The decision requires
measurements of unpacking and scaling, accumulator precision, tile occupancy,
and end-to-end prefill and decode separately.

### Rotations: where they appear and where they disappear

For an orthogonal $R$, a linear layer can be rewritten without changing its
function:

$$XW=(XR)(R^\top W).$$

The transformed weights $W'=R^\top W$ can be prepared offline. The online
rotation $XR$ disappears only if it can be algebraically absorbed into
neighboring weights or implemented inside a fused kernel. A fast Hadamard
transform costs $O(d\log d)$; an online dense learned rotation costs $O(d^2)$
and can consume the quantization gain if left unfused.

[SpinQuant](https://arxiv.org/abs/2405.16406) places rotations in the residual
stream and inside attention and MLP blocks, using orthogonal pairs that
cancel around compatible operations. Learned rotations are initialized
from Hadamard transforms and optimized on calibration loss using
orthogonality-preserving Cayley optimization. Remaining online Hadamard
operations require efficient kernels or fusion. Rotating Q alone does not
preserve attention: the Q/K transformation must be coordinated, while RoPE
and the head layout constrain where it can be applied or absorbed.

Quantization changes the equivalence. INT4, MXFP4, and NVFP4 have different
grids and scaling rules, so a rotation must be evaluated for the actual
format; “rotation always improves FP4” does not follow from INT4 results.

Validation includes full-precision equivalence before quantization,
$R^\top R\approx I$, profiler checks for unexpected dense rotation kernels,
and the exact packed layout and scales after absorption. Compare
no rotation, offline-only rotation, and online-plus-fused rotation with the
same weight, activation, and KV formats. Then repeat quality and latency
measurements: reduced kurtosis is not itself a product metric.

A weight may be stored in four bits yet unpacked into FP16 before every matrix multiplication. Such an artifact saves disk and HBM capacity, but speed depends on whether unpacking, scaling, and multiplication are fused in one kernel. Implementations all called INT4 may use different:

- group sizes and packing orders;
- symmetric or asymmetric codes;
- scale data types;
- layouts for tensor-parallel shards;
- supported GPU or CPU instructions.

Selection should therefore proceed backward from deployment: target device → available kernel → supported layout → algorithm that produces that artifact. The [vLLM quantization table](https://docs.vllm.ai/en/latest/features/quantization/) is more useful than a generic list of methods because it records combinations that a runtime and hardware actually support.

## Why average perplexity is insufficient

A small mean loss can conceal severe degradation in uncommon regimes:

- numbers, rare tokens, and lower-resource languages;
- programming and exact mathematical steps;
- JSON, function calling, and constrained decoding;
- long contexts, especially with separate KV quantization;
- close answer alternatives separated by small logit differences.

The calibration set must cover the input forms expected in production. Calibration on short English prose alone does not establish quality for long Russian tool-using conversations.

## Validating a deployable artifact

Compare the original and quantized models with the same tokenizer, chat template, and decoding configuration. Check:

1. parameter size **including metadata** and peak HBM use;
2. prefill throughput separately from decode throughput;
3. TTFT and TPOT at several batch sizes and sequence lengths;
4. perplexity across multiple domains;
5. tasks with unambiguous evaluation: code, mathematics, JSON, and tool calls;
6. long-context behavior with both original and quantized KV caches;
7. profiler evidence that the intended low-precision kernel executes;
8. saving and reloading the exact artifact that will be deployed.

A W4 speedup for batch-one decode does not prove a gain for large batches or prefill. A four-times-smaller file does not prove a fourfold speedup: scales, the KV cache, activations, and communication remain.

## Sources and further reading

- Stanford CS336 Spring 2026, [Lecture 10: Inference](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_10.py) — the relationship between quantization, memory-bound decode, and hardware efficiency.
- Hugging Face, [bitsandbytes integration](https://huggingface.co/blog/hf-bitsandbytes-integration) — an illustrated introduction to codes, scales, and outliers.
- Frantar et al., [GPTQ](https://arxiv.org/abs/2210.17323) — second-order PTQ.
- Lin et al., [AWQ](https://arxiv.org/abs/2306.00978) — activation-aware scaling for weight-only quantization.
- Xiao et al., [SmoothQuant](https://arxiv.org/abs/2211.10438) — transferring quantization difficulty from activations to weights for W8A8.
- Dettmers et al., [QLoRA](https://arxiv.org/abs/2305.14314) — NF4 and memory-efficient parameter-efficient fine-tuning.

- Efficient DL Systems, [week 9 lecture at pinned commit](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week09_inference_algorithms/lecture.pdf) — PTQ/QAT, GPTQ, SmoothQuant, low-precision formats, and rotations.
- Liu et al., [SpinQuant](https://arxiv.org/abs/2405.16406) — learned rotations for W4A4KV4.
- NVIDIA CUDA Math API, [E2M1](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-math-api/cuda_math_api/struct____nv__fp4__e2m1.html) — scalar layout and finite values.
- Open Compute Project, [MX v1.0, Table 1 and §5.3](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) — MXFP4 block and scale definitions.
- NVIDIA, [Transformer Engine 2.15 NVFP4](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.15/user-guide/features/low_precision_training/nvfp4/nvfp4.html) — scaling and one-/two-dimensional layouts.

### Practice

- [[05 Источники/Courses/Harvard ML Systems/tinytorch/15_quantization|TinyTorch 15 — Quantization]].
- [[05 Источники/Courses/Harvard ML Systems/tinytorch/16_compression|TinyTorch 16 — Compression]].
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week09_inference_algorithms/seminar.ipynb|Efficient DL Systems quantization seminar]].
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week09_inference_algorithms/homework/homework.ipynb|Efficient DL Systems quantization assignment]].
- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol1/model_compression|Harvard CS249r — Model Compression]].
- [[02 Areas/ML & DL/06 Практика/15 Реализовать W8A8 и SmoothQuant|Implement W8A8 and SmoothQuant]].

**Next:** [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/57a KV-cache compression и offload|KV-cache compression and offload]].
