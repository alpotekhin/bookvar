---
title: "Quantizing language models"
type: textbook-chapter
status: canonical
last_updated: 2026-10-29
last_verified: 2026-10-19
source_unit_id:
  - lecture-02-fp4-block-scaling
  - lecture-10-section-311-activation-aware-quantization-awq
  - lecture-10-figure-step-318-rendering-1
primary_sources:
  - https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_10.py
  - https://huggingface.co/blog/hf-bitsandbytes-integration
  - https://arxiv.org/abs/2210.17323
  - https://arxiv.org/abs/2306.00978
  - https://arxiv.org/abs/2211.10438
  - https://arxiv.org/html/2305.14314v1
  - https://arxiv.org/html/2405.16406v4
  - https://arxiv.org/abs/2208.07339
  - https://github.com/huggingface/transformers/blob/1d9654e78a23fb167ef539cbd699dfbd3e4830cf/docs/source/en/quantization/concept_guide.md
locale: en
translation_status: current
canonical_revision: "70bd3d2b286af4c5b04150e0a27bf527bd18de4ac6f829d2b7eaeb437f010d17"
translation_reviewed: "2026-10-29"
translation_of: "00 Учебник/14 Inference и оптимизация/57 Квантизация языковых моделей.md"
---

<a id="квантизация-языковых-моделей"></a>

# Quantizing language models

During token-by-token generation, a dense model reads almost all its
parameters but performs little arithmetic per byte loaded. Representing
parameters with four bits instead of sixteen reduces storage and transfer
volume. Numerical error and speed also depend on group sizes, scale
metadata, activation and KV-cache precision, and the target device's
kernel support.

<a id="действительное-число-заменяется-кодом-и-масштабом"></a>

## Replacing a real number with a code and scale

Take weights $(-1.0,-0.3,0.2,0.9)$. Allocate three bits each and use
symmetric integer codes from $-3$ to $3$. The largest magnitude is one,
so adjacent codes are separated by $s=1/3$.

| Weight $w_i$ | $w_i/s$ | Integer code $q_i$ | Reconstructed weight $\hat w_i=sq_i$ |
|---:|---:|---:|---:|
| $-1.0$ | $-3.0$ | $-3$ | $-1$ |
| $-0.3$ | $-0.9$ | $-1$ | $-1/3$ |
| $0.2$ | $0.6$ | $1$ | $1/3$ |
| $0.9$ | $2.7$ | $3$ | $1$ |

The file can store four short codes and one shared scale rather than four
original numbers. Reconstruction returns an approximation: $0.2$ became
$1/3$. The magnitude and sign of that error affect matrix multiplication.

For a signed symmetric uniform $b$-bit grid with $b\ge2$:

$$
q_{\max}=2^{b-1}-1,\qquad s=\frac{\max_i|w_i|}{q_{\max}},
$$

$$
\begin{aligned}
q_i&=\operatorname{clip}\bigl(\operatorname{round}(w_i/s),\\
&\qquad{}-q_{\max},q_{\max}\bigr),\\
\hat w_i&=sq_i.
\end{aligned}
$$

This uses $2^b-1$ levels: INT8 uses $[-127,127]$, although its signed
container can also hold $-128$. The symmetric grid represents zero exactly.
Other implementations use every container code; their rounding and
saturation rules must be checked separately.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/hf-quantization-rounding.png]]

*Tim Dettmers and Hugging Face,
[Making LLMs even more accessible with bitsandbytes](https://huggingface.co/blog/hf-bitsandbytes-integration).
The figure calculates $\alpha=127/\max|w_i|$, multiplies by $\alpha$, then
rounds. Our scale is $s=1/\alpha$. Bottom-row labels are themselves rounded;
apparent agreement with the top row does not imply zero reconstruction error.*

Adding a weight of magnitude ten to the group changes the step to $10/3$.
All four earlier weights now round to zero. They have not become less
relevant to the model: a larger neighbor consumed the shared range.
Smaller groups can limit this effect.

<a id="кодирование-и-восстановление"></a>

### Encoding and reconstruction

This CPU example implements that rounding. Codes remain ordinary Python
integers; it does not yet pack a three-bit file. For an all-zero group it
sets $s=1$, an arbitrary nonzero scale because every code and reconstructed
value remains zero.

<!-- checked-example: quant-symmetric -->
```python
import math


def symmetric_quantize(values, bits):
    if not isinstance(bits, int) or isinstance(bits, bool) or bits < 2:
        raise ValueError("bits must be an integer >= 2")
    values = [float(x) for x in values]
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("Expected a non-empty finite group")
    qmax = 2 ** (bits - 1) - 1
    amax = max(abs(x) for x in values)
    scale = amax / qmax if amax else 1.0
    codes = [max(-qmax, min(qmax, round(x / scale))) for x in values]
    restored = [scale * q for q in codes]
    return codes, scale, restored


codes, scale, restored = symmetric_quantize([-1, -0.3, 0.2, 0.9], 3)
assert codes == [-3, -1, 1, 3]
assert scale == 1 / 3
assert symmetric_quantize([0, 0], 3) == ([0, 0], 1.0, [0.0, 0.0])
assert symmetric_quantize([-1, -0.3, 0.2, 0.9, 10], 3)[0] == [0, 0, 0, 0, 3]

```

Python `round` resolves exact halfway cases toward the nearest even integer.
All numerical examples here use that rule. Rounding behavior is part of
the contract when comparing another kernel.

<a id="симметричная-и-асимметричная-сетка"></a>

## Symmetric and asymmetric grids

For $(-1,0,1,2)$, the positive range is larger than the negative range.
A symmetric four-bit grid chooses $s=2/7$ and leaves some negative levels
unused. An asymmetric grid can distribute sixteen codes, zero through
fifteen, across $[-1,2]$. The step is $s=3/15=0.2$, and zero corresponds
to code $z=5$:

$$
\begin{aligned}
(-1,0,1,2)&\longrightarrow(0,5,10,15)\\
&\longrightarrow(-1,0,1,2).
\end{aligned}
$$

These deliberately chosen values lie exactly on the grid; arbitrary
values still incur rounding error. The number $z$ is the **zero point**:

$$
\begin{aligned}
q_i&=\operatorname{clip}\bigl(\operatorname{round}(w_i/s)+z,\\
&\qquad{}q_{\min},q_{\max}\bigr),\\
\hat w_i&=s(q_i-z).
\end{aligned}
$$

Range selection normally ensures that zero is representable. With
$a=\min(0,\min_iw_i)$, $b=\max(0,\max_iw_i)$ and $a<b$, one choice is

$$
\begin{aligned}
s&=\frac{b-a}{q_{\max}-q_{\min}},\\
z&=\operatorname{clip}\bigl(\operatorname{round}(q_{\min}-a/s),\\
&\qquad{}q_{\min},q_{\max}\bigr).
\end{aligned}
$$

An asymmetric grid stores and applies $z$. Matrix multiplication therefore
requires corrections: $\sum q_xq_w$ becomes
$\sum(q_x-z_x)(q_w-z_w)$. Some corrections can be precomputed, but kernel
support determines which ones are cheap. Using code levels more densely
does not guarantee a faster product.

<a id="общий-масштаб-для-тензора-канала-или-группы"></a>

### Sharing a scale across a tensor, channel or group

Under $Y=XW$, columns of $W$ are output channels and rows are input channels.
A scale can cover the whole tensor, an output channel, or $G$ weights within
a channel. PyTorch `Linear.weight` stores the transpose,
$[d_{\mathrm{out}},d_{\mathrm{in}}]$. “Quantize rows” is ambiguous without
a shape convention.

Smaller groups follow local ranges more closely but add metadata. For
128 four-bit weights and one 16-bit scale:

$$
4+\frac{16}{128}=4.125\ \text{bits/weight}.
$$

A 16-bit zero point raises this to 4.25 bits per weight. That is one storage
scheme, not a universal asymmetric-quantization cost: zero points may be
packed differently. Alignment and incomplete groups add further overhead.

<a id="упаковка-int4"></a>

### Packing INT4

Hugging Face explains two four-bit codes per byte in
[int4 and weight packing](https://github.com/huggingface/transformers/blob/1d9654e78a23fb167ef539cbd699dfbd3e4830cf/docs/source/en/quantization/concept_guide.md#int4-and-weight-packing).
The first two explanatory paragraphs are reproduced below.
Copyright 2024 The HuggingFace Team,
[Apache-2.0](/licenses/transformers-1d9654e7-Apache-2.0.txt).

<!-- source-excerpt: hf-int4-packing:start -->
> int4 quantization further reduces the model size and memory usage (halving it compared to int8). The same affine or symmetric quantization principles apply, mapping the float32 range to the 16 possible values representable by int4 ( $[-8, 7]$ for signed int4).
>
> A key aspect of int4 quantization is **weight packing**. Since most hardware can't natively handle 4-bit data types in memory, two int4 values are typically packed together into a single int8 byte for storage and transfer. For example, the first value might occupy the lower 4 bits and the second value the upper 4 bits of the byte (`packed_byte = (val1 & 0x0F) | (val2 << 4)`).
<!-- source-excerpt: hf-int4-packing:end -->

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/quantization-review-2026-10/hf-weight-packing.png]]

*Hugging Face,
[Quantization concepts: weight packing](https://huggingface.co/docs/transformers/main/quantization/concept_guide#int4-and-weight-packing).
Two four-bit codes occupy one byte. Original documentation-images figure,
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

For example, $-3$ and $5$ have four-bit two's-complement representations
`1101` and `0101`. Put the first in the low nibble to obtain
`01011101`, or `0x5D`. Unpacking must interpret `1101` as signed again:
$13-16=-3$.

<!-- checked-example: quant-int4-packing -->
```python
def pack_int4_pair(low, high):
    if not all(isinstance(x, int) and -8 <= x <= 7 for x in (low, high)):
        raise ValueError("Expected two signed INT4 codes")
    return (low & 0x0F) | ((high & 0x0F) << 4)


def unpack_int4_pair(byte):
    if not isinstance(byte, int) or not 0 <= byte <= 255:
        raise ValueError("Expected one byte")
    low, high = byte & 0x0F, (byte >> 4) & 0x0F
    return tuple(x - 16 if x >= 8 else x for x in (low, high))


assert pack_int4_pair(-3, 5) == 0x5D
assert unpack_int4_pair(0x5D) == (-3, 5)

```

These functions only pack two codes. They do not specify a Tensor Core
matrix layout, group/scale order or checkpoint format. Reconstruction is
usually fused with multiplication to avoid writing an entire unpacked
FP16 matrix to HBM.

<a id="что-именно-квантуется"></a>

## What exactly is quantized?

`W4A16` denotes four-bit weights and FP16/BF16 activations. In `W8A8`,
both matrix-multiplication operands use eight bits. These address different
constraints.

| Mode | What shrinks | Typical use | Main difficulty |
|---|---|---|---|
| W8A16 | Parameters | Memory-bound decode | A fast weight-only kernel |
| W4A16 | Parameters more aggressively | Capacity and decode | Low-bit error and group scales |
| W8A8 | Weights and activations | Large prefill matrix products | Dynamic activation outliers |
| FP8 | Floating-point weights/activations | Supported Tensor Cores | Scaling recipe |
| KV8/KV4 | KV cache | Long contexts and large batches | Attention error |

Weight-only quantization reduces decode's dominant read traffic while
leaving activations in FP16/BF16. A sufficiently large prefill projection
may instead be compute-bound; unpacking W4 to FP16 may not speed it up.
W8A8 or FP8 can use low-precision matrix instructions, but activations
depend on input and may contain rare large channels.

The KV cache is an independent third object. Quantizing weights does not
shrink request history. Key error changes attention scores; value error
changes the weighted sum. The result depends on input, context length and
format. It does not follow that error increases monotonically with tokens.

<a id="ptq-квантизация-уже-обученной-модели"></a>

## PTQ: quantizing an already trained model

In post-training quantization, the original model is already trained.
Round-to-nearest (RTN) needs no text at all: weights and a grid suffice.
More accurate methods use calibration examples to estimate which errors
matter to layer outputs.

For $Y=XW$, the error is $X(W-\hat W)$. Minimizing weight error need not
minimize output error. With input $(10,1)$, an error of $0.1$ in the first
weight changes a scalar output by one; the same error in the second changes
it by $0.1$. Input correlations matter too: errors in different weights
can reinforce or compensate one another.

Calibration usually runs a small representative dataset through the model.
For each layer, collect required input statistics, choose ranges, scales or
weight transformations, then evaluate on a separate set. Preserving
calibration outputs alone does not establish quality on other languages,
domains or sequence lengths.

<a id="статические-и-динамические-масштабы"></a>

### Static and dynamic scales

Weight scales can be prepared offline because weights remain fixed during
inference. Activation scales can be static, estimated from calibration and
then frozen, or dynamic, calculated from current activations, perhaps per
token. Dynamic scales adapt to each input but add runtime reduction and
conversion work.

The maximum does not always minimize average error. Clipping saturates a
few large values at the extreme code while providing a smaller step for
the others. Ranges may be selected by quantiles, mean squared error or
layer-output error. The appropriate criterion depends on the task:
losing a rare but significant outlier can be worse than a larger mean error.

### GPTQ

[GPTQ, §3–4](https://arxiv.org/abs/2210.17323) chooses quantized weights to
preserve a layer's outputs on calibration activations:

$$
\min_{\hat W\ \text{on the chosen grid}}\|XW-X\hat W\|_F^2.
$$

Consider one output channel with two weights:

$$
\begin{aligned}
X&=\begin{pmatrix}1&1\\1&2\end{pmatrix},\\
w&=\begin{pmatrix}0.4\\0.35\end{pmatrix},\\
y&=Xw=\begin{pmatrix}0.75\\1.10\end{pmatrix}.
\end{aligned}
$$

Use a grid with step one. First round $0.4$ to zero. Leaving the second
weight unchanged gives outputs $(0.35,0.70)$. Both calibration examples
lose $0.4$, and squared errors sum to $0.32$.

The second weight is still free. Adding $\delta$ to it changes outputs by
$(\delta,2\delta)$. Minimize

$$
(-0.4+\delta)^2+(-0.4+2\delta)^2.
$$

The derivative is $10\delta-2.4=0$, giving $\delta=0.24$.
The second weight becomes $0.59$. Outputs $(0.59,1.18)$ differ from the
original by $(-0.16,0.08)$, with squared-error sum $0.032$.

| State | Weights | Outputs | Squared-error sum |
|---|---|---|---:|
| Original layer | $(0.4,0.35)$ | $(0.75,1.10)$ | $0$ |
| First weight rounded | $(0,0.35)$ | $(0.35,0.70)$ | $0.32$ |
| Second weight compensates | $(0,0.59)$ | $(0.59,1.18)$ | $0.032$ |
| Second weight also rounded | $(0,1)$ | $(1,2)$ | $0.8725$ |
| Independent rounding of both | $(0,0)$ | $(0,0)$ | $1.7725$ |

Compensation does not eliminate later rounding: the intermediate output
is more accurate than the final one. It changed the second code choice
and reduced final error in this example. A sequential procedure does not
promise the globally best combination of codes.

In general, inverse Hessian information gives the correction. Under our
$Y=XW$ convention, the quadratic error has $H=2X^\top X$.
For a selected weight $w_q$ and quantized value $\hat w_q$:

$$
\delta w=-\frac{w_q-\hat w_q}{[H^{-1}]_{qq}}(H^{-1})_{:,q}.
$$

The paper stores inputs as columns,
$Y_{\mathrm{paper}}=W_{\mathrm{paper}}X_{\mathrm{paper}}$,
so its Hessian is $2XX^\top$ and it processes weight columns. Those are
rows of $W$ in this chapter.

GPTQ makes the procedure practical for large layers with a shared input-channel
order across outputs, blocked updates and Cholesky-based computation.
Adding $\lambda I$ to $H$, called damping, helps with ill-conditioned
calibration matrices. It does not build the full language-model loss Hessian.

<!-- checked-example: quant-gptq-step -->
```python
import numpy as np

calibration_x = np.array([[1.0, 1.0], [1.0, 2.0]])
original_w = np.array([0.4, 0.35])
target_y = calibration_x @ original_w
inverse_hessian = np.linalg.inv(2 * calibration_x.T @ calibration_x)

corrected_w = original_w.copy()
q0 = round(corrected_w[0])
error0 = corrected_w[0] - q0
corrected_w -= error0 / inverse_hessian[0, 0] * inverse_hessian[:, 0]
np.testing.assert_allclose(corrected_w, [0.0, 0.59])

squared_error = lambda weights: np.sum((calibration_x @ weights - target_y) ** 2)
np.testing.assert_allclose(squared_error([0, 0.35]), 0.32)
np.testing.assert_allclose(squared_error(corrected_w), 0.032)
np.testing.assert_allclose(squared_error(np.rint(corrected_w)), 0.8725)
np.testing.assert_allclose(squared_error(np.rint(original_w)), 1.7725)

```

This is one quadratic compensation step, not an optimized GPTQ implementation.
In the [authors' gptq.py](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/gptq.py),
the variable `Hinv` contains an upper Cholesky factor after preparation,
not raw $H^{-1}$. Substituting it directly into the equation above would
change the algorithm.

### AWQ

[AWQ, §3](https://arxiv.org/html/2306.00978v5#S3) identifies sensitive input
channels from large calibration activations. As a diagnostic, its authors
retained a small portion of associated weights in FP16. Quality improved,
but scattered FP16 values inside an INT3 matrix are inconvenient for a
dense kernel.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/topics-53-59-source-first/awq-schema.png]]

*Ji Lin et al., [AWQ, Figure 2](https://arxiv.org/abs/2306.00978).
Left: ordinary INT3 rounding. Center: the diagnostic that keeps 1% of weights
in FP16. Right: AWQ scaling with uniform low-bit weight storage.
The center improves perplexity but complicates dense execution.
Also discussed in [Stanford CS336 Lecture 10](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_10.py#L480-L487).*

AWQ itself retains a uniform low-bit format. It enlarges weights of selected
input channels and inversely scales the activations. Before quantization,
the product is unchanged:

$$
\begin{gathered}
XW=(XD^{-1})(DW),\\
D=\operatorname{diag}(s_1,\ldots,s_d),\quad s_j>0.
\end{gathered}
$$

After rounding it becomes $(XD^{-1})Q(DW)$. A sensitive channel gains a finer
effective step after inverse scaling. Enlarging weights can also raise the
group maximum and worsen other channels' rounding, so scales must be selected.

Take $X=(10,1)$ and $w=(0.14,1.75)^\top$, with output $3.15$.
Symmetric INT4 has step $1.75/7=0.25$, rounds the weights to $(0.25,1.75)$,
and produces $4.25$.

Double the first weight and halve its input:
$X'=(5,1)$, $w'=(0.28,1.75)^\top$. The maximum and step remain unchanged.
Rounding still gives $(0.25,1.75)$, but the transformed input yields
$5\cdot0.25+1.75=3$. Absolute error falls from $1.10$ to $0.15$ without
a separate FP16 weight.

<!-- checked-example: quant-awq-step -->
```python
awq_x = np.array([10.0, 1.0])
awq_w = np.array([0.14, 1.75])
channel_scale = np.array([2.0, 1.0])
plain_w = np.array(symmetric_quantize(awq_w, 4)[2])
scaled_w = np.array(symmetric_quantize(awq_w * channel_scale, 4)[2])
np.testing.assert_allclose(awq_x @ awq_w, 3.15)
np.testing.assert_allclose(awq_x @ plain_w, 4.25)
np.testing.assert_allclose((awq_x / channel_scale) @ scaled_w, 3.0)

```

The factor two was chosen manually for this example. AWQ constructs
candidates from activation statistics and selects scaling strength by
calibration output error. It searches for a transformation rather than
fine-tuning original weights by gradient descent. The name alone does
not specify group size, packing or inference kernel.

<a id="выбросы-активаций-и-llmint8"></a>

### Activation outliers and LLM.int8()

Weight-only quantization leaves activations in FP16/BF16. W8A8 must round
them too. A few large input channels can create such a broad shared range
that ordinary values lose most available levels.
[LLM.int8()](https://arxiv.org/abs/2208.07339) observed these outliers
concentrated in a small subset of features in its studied models.

LLM.int8() splits the product along input channels. For outlier set $J$:

$$
XW=X_{:,\bar J}W_{\bar J,:}+X_{:,J}W_{J,:}.
$$

The main product uses INT8; the small outlier product uses FP16.
Their outputs are added. This mixed-precision operation pays for outlier
handling, a second product and merging. Better numerical behavior alone
does not prove a speedup over an optimized FP16 kernel.

For example, $X=(1024,1)$ and $w=(1/1024,1)^\top$ produce two.
Quantize each complete vector with its own symmetric INT8 grid: the small
entry of each rounds to zero and the product becomes zero.
Keeping the first channel in FP16 preserves its contribution
$1024\cdot(1/1024)=1$; the remaining INT8 channel contributes another one.
These numbers are deliberately FP16-representable. General values still
round in the high-precision branch.

### SmoothQuant

[SmoothQuant, §3–4](https://arxiv.org/html/2211.10438v6#S3) transfers part of
the activation range into weights so that ordinary INT8 GEMM can handle
both operands. It changes scales rather than keeping a separate outlier path.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/quantization-review-2026-10/smoothquant-intuition.png]]

*Guangxuan Xiao et al.,
[SmoothQuant: authors' illustration](https://github.com/mit-han-lab/smoothquant/blob/c61476d728e42ae0d8a35e7e78494edcac3237b5/figures/intuition.png),
[MIT](/licenses/smoothquant-c61476d7-MIT.txt).
Left: an activation outlier alongside a small weight range. Right: more
balanced ranges. Lines are quantization levels, not additional bits.*

Again $X'=XD^{-1}$ and $W'=DW$, but the scales use both operands' ranges.
Let $a_j=\max_t|X_{tj}|$ be the calibration activation maximum for input
channel $j$, and $b_j=\max_k|W_{jk}|$ its weight-row maximum.
One recipe is

$$
s_j=\frac{a_j^\alpha}{b_j^{1-\alpha}},\qquad 0\le\alpha\le1.
$$

At $\alpha=0$, weights are mainly equalized; at $\alpha=1$, activations.
Intermediate values distribute the range between them. Implementations
bound tiny or zero denominators and scales to avoid division by zero.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/quantization-review-2026-10/smoothquant-migrate.jpg]]

*Guangxuan Xiao et al.,
[SmoothQuant: distributions before and after transformation](https://github.com/mit-han-lab/smoothquant/blob/c61476d728e42ae0d8a35e7e78494edcac3237b5/figures/migrate.jpg),
[MIT](/licenses/smoothquant-c61476d7-MIT.txt).
Activations are arranged by token and channel; weights by input and output
channel. Vertical scales differ: activation outliers shrink while weight
ranges grow. EDLS week 9, slides 37–39, discusses the same figure.*

Use three bits for a short calculation. Let $X=(100,1)$ and
$w=(0.01,1)^\top$, giving output two. Independently quantizing each complete
operand symmetrically rounds the second input and first weight to zero,
so the product becomes zero.

For $\alpha=1/2$, $s=(100,1)$. Then both transformed operands are $(1,1)$,
represented exactly, and their product remains two.

<!-- checked-example: quant-smoothquant-step -->
```python
sq_x = np.array([100.0, 1.0])
sq_w = np.array([0.01, 1.0])
sq_scale = np.sqrt(np.abs(sq_x) / np.abs(sq_w))
plain_x = np.array(symmetric_quantize(sq_x, 3)[2])
plain_w = np.array(symmetric_quantize(sq_w, 3)[2])
smooth_x = np.array(symmetric_quantize(sq_x / sq_scale, 3)[2])
smooth_w = np.array(symmetric_quantize(sq_w * sq_scale, 3)[2])
np.testing.assert_allclose(sq_scale, [100.0, 1.0])
np.testing.assert_allclose(sq_x @ sq_w, 2.0)
np.testing.assert_allclose(plain_x @ plain_w, 0.0)
np.testing.assert_allclose(smooth_x @ smooth_w, 2.0)

```

Exact compensation follows from deliberately chosen numbers. Real W8A8
layers retain rounding errors in both operands, and new inputs may exceed
calibration ranges.

Per-input-channel activation scales complicate GEMM. Channel $j$ is inside
$\sum_jX_{tj}W_{jk}$, so its coefficient generally cannot be pulled outside
the product. Per-token activation scales and per-output-channel weight
scales can instead be applied to accumulated outputs. SmoothQuant makes a
coarser activation grid accurate enough while retaining this execution order.

Input scaling can often be absorbed into a preceding LayerNorm/RMSNorm and
associated linear layers, with weights transformed offline. This is what
[smooth_ln_fcs](https://github.com/mit-han-lab/smoothquant/blob/c61476d728e42ae0d8a35e7e78494edcac3237b5/smoothquant/smooth.py)
implements. If several projections share one normalized input, their
transformations must agree; changing only one branch independently would
break equivalence.

<a id="qat-модель-видит-ошибку-во-время-обучения"></a>

## QAT: exposing the model to error during training

In quantization-aware training, forward uses reconstructed quantized
$\hat w$ while the trainable master $w$ remains real-valued.
This is **fake quantization**. It simulates the future format's error;
a teaching implementation may execute ordinary FP32 operations.

Take $w=0.26$ with fixed step $s=0.1$. Forward yields $\hat w=0.3$.
For $\mathcal L=(\hat w-1)^2$, the derivative with respect to $\hat w$
is $-1.4$. The rounding derivative with respect to $w$ is zero almost
everywhere: a small change around $0.26$ leaves code three unchanged.
Applying that derivative literally would give no useful parameter update.

A straight-through estimator substitutes an approximate derivative.
For example, within the permitted range, take
$\partial\hat w/\partial w\approx1$. With learning rate $0.1$, the master
weight becomes $0.26-0.1(-1.4)=0.40$. Next forward produces $0.4$, reducing
loss from $0.49$ to $0.36$.

This is an approximate gradient, not the true derivative of piecewise
constant rounding. Clipped implementations often zero gradients outside
the valid range. STE details, learned scales and fake-quantization schedules
depend on the method. See [Bengio et al., 2013](https://arxiv.org/abs/1308.3432)
for the gradient-estimation formulation.

QAT adapts weights to a chosen format's errors but requires data, backward
and training memory. It can fine-tune a model or operate from the start of
training. A numerical emulator does not itself create a compact file:
deployment still needs packing, compatible scales and an inference kernel.

<a id="nf4-и-qlora-решают-задачу-дообучения"></a>

## NF4 and QLoRA address fine-tuning

QLoRA freezes a quantized base model and trains small LoRA matrices.
With reconstructed base weights $\hat W$, the linear layer is

$$
Y=X\hat W+XAB.
$$

Updating $A,B$ requires backward through the layer, but not gradients and
optimizer states for every base weight. Four-bit codes are reconstructed
into a compute dtype, commonly BF16; this does not imply four-bit training
matrix products throughout.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/quantization-review-2026-10/qlora-training-original.png]]

*Tim Dettmers, Artidoro Pagnoni, Ari Holtzman and Luke Zettlemoyer,
[QLoRA, Figure 1, p.3](https://arxiv.org/html/2305.14314v1#S1.F1),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Green arrows show gradient flow; blue arrows show parameter updates.
Gradients pass through the frozen base to adapters and earlier layers,
but base weights are not updated. Pink arrows move optimizer states
between CPU and GPU.*

<a id="nf4-четыре-бита-как-индекс-таблицы"></a>

### NF4: four bits index a table

NF4 uses sixteen predefined levels rather than a uniform integer grid.
A code indexes table $C$. With block scale $s$, choose

$$
\begin{aligned}
q_i&=\arg\min_{k\in\{0,\ldots,15\}}|w_i/s-C_k|,\\
\hat w_i&=sC_{q_i}.
\end{aligned}
$$

Levels cluster near zero and become sparser toward the range limits.
QLoRA constructs them from normal-distribution quantiles and includes zero
explicitly. This targets a particular weight distribution rather than
being universally best for arbitrary activations. Unlike FP4 E2M1, an NF4
code is not divided into sign, exponent and mantissa. NF4 is a representation;
QLoRA is a fine-tuning method using it. NF4 weights can also serve inference
without fine-tuning.

<a id="double-quantization-сжатие-масштабов"></a>

### Double Quantization compresses scales

QLoRA's authors calculate the scale overhead of small blocks and compress
the scales themselves. The complete **Double Quantization** passage from
[§3](https://arxiv.org/html/2305.14314v1#S3) is reproduced below, Dettmers et al.,
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Reference [13] belongs to the paper's bibliography.

<!-- source-excerpt: qlora-double-quantization:start -->
> We introduce *Double Quantization* (DQ), the process of quantizing the quantization constants for additional memory savings.
> While a small blocksize is required for precise 4-bit quantization [13], it also has a considerable memory overhead. For example, using 32-bit constants and a blocksize of 64 for $\mathbf{W}$, quantization constants add $32/64 = 0.5$ bits per parameter on average. Double Quantization helps reduce the memory footprint of quantization constants.
>
>
> More specifically, Double Quantization treats quantization constants $c_2^{\text{FP32}}$ of the first quantization as inputs to a second quantization. This second step yields the quantized quantization constants $c_2^{\text{FP8}}$ and the second level of quantization constants $c_1^{\text{FP32}}$. We use 8-bit Floats with a blocksize of 256 for the second quantization as no performance degradation is observed for 8-bit quantization, in line with results from Dettmers and Zettlemoyer [13].
> Since the $c_2^{\text{FP32}}$ are positive, we subtract the mean from $c_2$ before quantization to center the values around zero and make use of symmetric quantization. On average, for a blocksize of 64, this quantization reduces the memory footprint per parameter from $32/64=0.5$ bits, to $8/64 + 32/(64\cdot256) = 0.127$ bits, a reduction of 0.373 bits per parameter.
<!-- source-excerpt: qlora-double-quantization:end -->

For the stated block sizes, codes and two scale levels cost

$$
4+\frac8{64}+\frac{32}{64\cdot256}\approx4.127\ \text{bits/weight}.
$$

This is the selected recipe's amortized payload. Global parameters,
alignment, incomplete blocks and other model tensors are separate.
“4-bit” does not promise a file of exactly half a byte per parameter.

<a id="paged-optimizers-и-выпуск-модели"></a>

### Paged optimizers and deployment

QLoRA also moves optimizer states between CPU and GPU to tolerate memory
spikes. Transfer costs remain, as do activation storage and backward.
This is training-memory management, distinct from PagedAttention's
inference KV-cache management.

After fine-tuning, keep base and adapter separate or merge them and
requantize for the target runtime. The merged $W+AB$ need not lie on the
old NF4 grid. A QLoRA checkpoint is not automatically the fastest deployment
representation.

<a id="формат-файла-не-равен-формату-вычисления"></a>

## File format is not compute format

<a id="gguf-контейнер-и-типы-отдельных-тензоров"></a>

### GGUF: a container with per-tensor types

[GGUF](https://github.com/ggml-org/ggml/blob/facdc44a67da17df4433adcc468b424f48e9ded2/docs/gguf.md)
stores model metadata, tensor descriptions and tensor bytes. Each tensor
description specifies its shape, type and data offset. The `.gguf`
extension therefore does not mean four-bit: the container supports F16/F32
and different quantized representations. The executor selects its compute
path from those types and the device.

In [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/8ab29faec401829db0e617f40af2c942f4a9eaab/tools/quantize/README.md),
converting a model to GGUF and quantizing it are separate stages.
A BF16 GGUF can, for example, become a `Q4_K_M` artifact. That name
identifies a recipe that can mix tensor types, not exactly four bits for
every parameter. The tool permits separate output-matrix and embedding
types. Measure the resulting file's size, speed and quality rather than
infer them from its filename suffix.

<a id="fp8-восемь-бит-не-задают-один-диапазон"></a>

### FP8: eight bits do not specify one range

E4M3 allocates one sign bit, four exponent bits and three mantissa bits.
The E4M3FN variant used by CUDA/TensorRT has maximum finite magnitude 448,
minimum normal $2^{-6}$ and subnormals down to $2^{-9}$.
E5M2 uses five exponent bits and two mantissa bits, with corresponding
limits 57,344, $2^{-14}$ and $2^{-16}$. It trades precision for range.
The exact variant matters: IEEE-like Inf/NaN and finite-only encodings
allocate extreme codes differently.

Tensors are normally scaled: $\hat x=s\,\operatorname{FP8}(x/s)$.
One scale per tensor costs little metadata, but a shared large maximum
reduces small-value resolution. Channel or block scales follow local ranges
more closely while changing layout and kernel requirements. A recipe
must identify scale selection, weight and activation formats, and the
independent KV format. A result on one accelerator does not establish
the same gain on another.

<a id="fp4-mxfp4-и-nvfp4"></a>

### FP4, MXFP4 and NVFP4

Scalar E2M1 has one sign bit, two exponent bits and one explicit mantissa bit.
CUDA defines finite magnitudes $\{0,0.5,1,1.5,2,3,4,6\}$, no Inf/NaN,
and saturation at six. Local scaling extends this small range.

OCP MX v1.0's MXFP4 combines 32 E2M1 elements with one eight-bit E8M0
power-of-two scale. Codes plus scale cost $4+8/32=4.25$ bits per element
before alignment. Power-of-two scaling simplifies multiplication but scale
rounding can add error.

NVFP4 uses 16 E2M1 elements, an FP8 E4M3 block scale and an FP32 global
scale: $x\approx s_{\mathrm{global}}s_{\mathrm{block}}x_{\mathrm{E2M1}}$.
E2M1's maximum is six and E4M3's is 448. One-dimensional payload costs
$4+8/16=4.5$ bits per element plus the global scale. Transformer Engine 2.15
uses one-dimensional 16-element activation/gradient blocks, but defaults
weights to $16\times16$ blocks to align rowwise and columnwise representations.
Renaming an MXFP4 checkpoint to NVFP4 cannot change these layouts or scales.

| Format | Element | Scale granularity | Scale type | Finite maximum before scaling |
|---|---|---|---|---:|
| FP8 E4M3FN | 1+4+3 bits | Tensor, channel or block | Recipe-dependent | 448 |
| FP8 E5M2 | 1+5+2 bits | Tensor, channel or block | Recipe-dependent | 57,344 |
| MXFP4 | E2M1, 4 bits | 32 elements | E8M0, 8 bits | 6 |
| NVFP4 | E2M1, 4 bits | 16 elements or $16\times16$ weights | E4M3 block, FP32 tensor | 6 |

<a id="общая-шкала-связывает-соседние-значения"></a>

### A shared scale couples neighboring values

Scaling extends a block's range without adding levels. Consider two
16-element blocks with nonzero values $(0.5,1,2,6)$ and $(8,16,32,96)$;
remaining entries are zero. Effective scales one and sixteen give both
the same E2M1 codes $(0.5,1,2,6)$ and exact reconstruction. This illustrates
block scaling, not the full NVFP4 scale-selection recipe.

Under one shared scale $s=16$, the first block becomes
$(0.03125,0.0625,0.125,0.375)$ before rounding, maps to $(0,0,0,0.5)$,
and reconstructs as $(0,0,0,8)$. MSE over all 32 entries is
$(0.25+1+4+4)/32=0.2890625$ rather than zero. Both scales are exactly
representable: the loss comes from sharing a range, not scale rounding.
Real NVFP4 additionally chooses the tensor scale, rounds block scales,
and follows its kernel's numerical contract.

Low precision can also be used in pretraining. The
[Nemotron 3 Super report, 3 April 2026 revision](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)
describes NVFP4 pretraining. It does not imply that every parameter,
optimizer state and operation uses four bits. The computation recipe and
released checkpoint format are separate facts; success in one run does
not establish stability in another.

<a id="что-меняется-в-одном-линейном-слое"></a>

<a id="worked-execution-weight-only-против-wa"></a>

### What changes in one linear layer

For $W\in\mathbb R^{4096\times4096}$, BF16 storage is 32 MiB.
Four-bit weights with one FP16 scale per 128 elements require 8 MiB
of codes plus 256 KiB of scales, totaling 8.25 MiB.

| Linear-layer data | BF16 | INT8 without metadata | INT4 plus FP16 scale per 128 weights |
|---|---:|---:|---:|
| $4096\times4096$ weights | 32 MiB | 16 MiB | 8.25 MiB |
| One token, 4096 features | 8 KiB | 4 KiB | Not specified by weight format |
| 256-token input | 2 MiB | 1 MiB | Not specified by weight format |

For one new token, weights dwarf the input. A fused W4A16 kernel reads
packed codes and scales, reconstructs small tiles in registers, and
immediately multiplies FP16/BF16 activations. It need not write an entire
FP16 weight copy to HBM every step.

With 256 input rows, each weight tile serves many tokens. GEMM may become
compute-bound. If W4 is unpacked to FP16, multiplication does not
automatically become four-bit. Unpacking and scaling can consume the
benefit of lower traffic.

A supported INT8 W8A8 kernel reduces both operands and uses low-precision
matrix instructions. FP8 W8A8 is a different numerical representation;
INT8 codes cannot be passed unchanged to FP8 multiplication. Measure
prefill and decode separately and inspect unpacking, accumulator dtype,
extra copies and actual kernel selection.

<a id="поворот-координат-перед-квантизацией"></a>

<a id="rotation-methods-где-возникает-и-куда-исчезает-поворот"></a>

### Rotating coordinates before quantization

Scaling changes coordinate lengths. An orthogonal transformation can
spread one large value across coordinates while preserving the product:

$$
XW=(XR)(R^\top W),\qquad R^\top R=I.
$$

For $x=(8,1)$, take a normalized Hadamard matrix:

$$
\begin{aligned}
R&=\frac1{\sqrt2}\begin{pmatrix}1&1\\1&-1\end{pmatrix},\\
xR&=(9/\sqrt2,7/\sqrt2).
\end{aligned}
$$

The largest magnitude falls from eight to approximately 6.36. Direct
symmetric three-bit quantization reconstructs $x$ as $(8,0)$.
Quantizing $xR$ and rotating back gives $(7.5,1.5)$, reducing MSE from
$0.5$ to $0.25$.

<!-- checked-example: quant-rotation -->
```python
rotation = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
rotation_x = np.array([8.0, 1.0])
direct = np.array(symmetric_quantize(rotation_x, 3)[2])
rotated = np.array(symmetric_quantize(rotation_x @ rotation, 3)[2])
recovered = rotated @ rotation.T
np.testing.assert_allclose(rotation.T @ rotation, np.eye(2), atol=1e-15)
np.testing.assert_allclose(recovered, [7.5, 1.5])
np.testing.assert_allclose(np.mean((direct - rotation_x) ** 2), 0.5)
np.testing.assert_allclose(np.mean((recovered - rotation_x) ** 2), 0.25)

```

This isolates activation error: transformed weights were not quantized.
Rounding both operands requires evaluating their joint error.
Orthogonality guarantees pre-quantization equality, not improvement for
every quantized input.

[SpinQuant, §2–3](https://arxiv.org/html/2405.16406v4#S3) learns rotations
on calibration data while original model weights remain frozen.
Cayley SGD preserves orthogonality; the reported experiments initialize
from random Hadamard matrices. Optimization uses the entire quantized
network's loss, such as calibration cross-entropy, unlike GPTQ's
individual-layer output reconstruction.

Without online Hadamard operations, two learned transformations are
absorbed into weights: $R_1$ rotates the residual stream, while $R_2$
coordinates per-head values with the output projection's input.
The lower-precision variant adds online transformations inside attention
and before the MLP down projection. These cannot all disappear through
a static weight change.

An arbitrary rotation cannot be inserted anywhere in a Transformer.
Nonlinearities, residual additions, normalization and RoPE require
consistent coordinates. Coordinated Q/K rotation preserves $QK^\top$;
rotating only Q does not. SpinQuant chooses locations under these constraints.

Static $R^\top W$ can be precomputed. $XR$ remains online unless absorbed
into a neighboring linear layer. Dense rotation costs $O(d^2)$; fast
Hadamard costs $O(d\log d)$. Fusion can reduce overhead, depending on
implementation. Check the profile for an unexpected expensive rotation
at every token.

<a id="совместимость-файла-и-вычислительного-ядра"></a>

### Compatibility between file and kernel

Four-bit storage may still unpack to FP16 before multiplication. INT4
implementations differ in group size, symmetry, scale dtype, packing order
and tensor-parallel shard layouts.

Choose deployment representation from the target device's supported
kernels, then prepare weights for it. An AWQ or GPTQ filename does not
establish a fast path for the installed engine, group size, layer shape
or model. Check the [vLLM table](https://docs.vllm.ai/en/latest/features/quantization/)
for the chosen version and inspect executed code.

<a id="почему-средняя-perplexity-недостаточна"></a>

## Why average perplexity is insufficient

A small mean degradation can conceal severe failures in uncommon regimes:

- numbers, rare tokens and lower-resource languages;
- programming and exact mathematical steps;
- JSON, function calling and constrained decoding;
- long contexts, especially with separate KV quantization;
- close answer alternatives separated by small logit differences.

Calibration must cover expected production input forms. Short English
prose does not establish quality for long Russian tool-using conversations.

<a id="проверка-готового-артефакта"></a>

## Validating a deployable artifact

Compare original and quantized models with identical tokenizer, chat
template and decoding. Check:

1. parameter storage including metadata, and peak HBM use;
2. prefill throughput separately from decode throughput;
3. TTFT and TPOT over multiple batches and lengths;
4. perplexity across domains;
5. code, mathematics, JSON and tool tasks with checkable outcomes;
6. long-context behavior with original and quantized KV;
7. profiler evidence for the intended low-precision kernel;
8. saving and reloading the exact deployment artifact.

A W4 gain at batch-one decode does not establish a gain for large batches
or prefill. A four-times-smaller file does not imply fourfold speed:
scales, KV, activations and inter-device communication remain.

<a id="практика-и-первоисточники"></a>

## Practice and primary sources

- [[05 Источники/Courses/Harvard ML Systems/tinytorch/15_quantization|TinyTorch 15: Quantization]].
- [[05 Источники/Courses/Harvard ML Systems/tinytorch/16_compression|TinyTorch 16: Compression]].
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week09_inference_algorithms/lecture.pdf|Efficient DL Systems: quantization and inference algorithms]].
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week09_inference_algorithms/seminar.ipynb|Efficient DL Systems quantization seminar]].
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week09_inference_algorithms/homework/homework.ipynb|Efficient DL Systems quantization assignment]].
- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol1/model_compression|Harvard CS249r: Model Compression]].
- [[02 Areas/ML & DL/06 Практика/15 Реализовать W8A8 и SmoothQuant|Implement W8A8 and SmoothQuant]].
- Stanford CS336 Spring 2026, [Lecture 10: Inference](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_10.py): quantization, memory-bound decode and hardware efficiency; pinned `8b59b507`.
- Hugging Face, [bitsandbytes integration](https://huggingface.co/blog/hf-bitsandbytes-integration): illustrated codes, scales and outliers.
- Frantar et al., [GPTQ](https://arxiv.org/abs/2210.17323): second-order PTQ.
- Lin et al., [AWQ](https://arxiv.org/abs/2306.00978): activation-aware weight-only scaling.
- Xiao et al., [SmoothQuant](https://arxiv.org/abs/2211.10438): W8A8 activation/weight balancing.
- Dettmers et al., [QLoRA](https://arxiv.org/abs/2305.14314): NF4 and fine-tuning memory.
- Efficient DL Systems, [week 9, pinned lecture](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week09_inference_algorithms/lecture.pdf): PTQ/QAT, GPTQ, SmoothQuant, FP8/FP4 and rotations.
- Liu et al., [SpinQuant](https://arxiv.org/abs/2405.16406): rotation-assisted W4A4KV4.
- NVIDIA CUDA Math API, [E2M1](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-math-api/cuda_math_api/struct____nv__fp4__e2m1.html): exact layout and finite encodings.
- Open Compute Project, [MX v1.0, Table 1 and §5.3](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf): block-32 MXFP4, E2M1 and E8M0.
- NVIDIA Transformer Engine, [NVFP4](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.15/user-guide/features/low_precision_training/nvfp4/nvfp4.html): block/global scaling and one-/two-dimensional weight layout.

← [[56 FlashAttention|FlashAttention]] ·
[[57a KV-cache compression и offload|KV-cache compression and offload]] →
