---
title: "Quantizing language models"
type: textbook-chapter
status: canonical
locale: en
translation_of: "00 Учебник/14 Inference и оптимизация/57 Квантизация языковых моделей.md"
last_updated: 2026-07-22
last_verified: 2026-07-22
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

For a signed, symmetric, uniform $b$-bit grid,

$$
q_{\max}=2^{b-1}-1,\qquad
s=\frac{\max_i|w_i|}{q_{\max}},
$$

$$
q_i=\operatorname{clip}\left(\operatorname{round}\frac{w_i}{s},
-q_{\max},q_{\max}\right),\qquad
\hat w_i=sq_i.
$$

Here $q_i$ is a compact integer code and $s$ is the scale that reconstructs the approximation $\hat w_i$. The difference $w_i-\hat w_i$ is quantization error.

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

[GPTQ](https://arxiv.org/abs/2210.17323) quantizes columns sequentially and compensates for the introduced error in the remaining weights using approximate second-order information. Its objective is to preserve layer outputs on calibration activations. The method became a major recipe for 3–4-bit weight-only PTQ.

### AWQ

[AWQ](https://arxiv.org/abs/2306.00978) builds on the observation that a small fraction of channels are particularly salient: their activations have large magnitudes, so errors in the associated weights strongly perturb the output. Rather than retaining scattered weights at mixed precision, it chooses channel scaling before quantization.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/awq-schema.png]]

*Left: ordinary INT3 rounding sharply degrades perplexity. Center: retaining 1% of weights in FP16 helps but is inconvenient for hardware. Right: AWQ rescales salient channels and quantizes the entire matrix uniformly. Original Figure 1 from Lin et al., [AWQ](https://arxiv.org/abs/2306.00978), reproduced in Stanford CS336.*

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

- Stanford CS336, [Lecture 10: Inference](https://cs336.stanford.edu/) — the relationship between quantization, memory-bound decode, and hardware efficiency.
- Hugging Face, [bitsandbytes integration](https://huggingface.co/blog/hf-bitsandbytes-integration) — an illustrated introduction to codes, scales, and outliers.
- Frantar et al., [GPTQ](https://arxiv.org/abs/2210.17323) — second-order PTQ.
- Lin et al., [AWQ](https://arxiv.org/abs/2306.00978) — activation-aware scaling for weight-only quantization.
- Xiao et al., [SmoothQuant](https://arxiv.org/abs/2211.10438) — transferring quantization difficulty from activations to weights for W8A8.
- Dettmers et al., [QLoRA](https://arxiv.org/abs/2305.14314) — NF4 and memory-efficient parameter-efficient fine-tuning.
