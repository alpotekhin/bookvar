---
title: "64.1. Connecting a vision encoder to an LLM"
type: textbook-chapter
status: canonical
locale: en
translation_status: current
canonical_revision: "5df467640b95eb59a8e5fdc45d7e72fecd15e703ca52011481ae8c480f43f341"
translation_reviewed: "2026-10-29"
translation_of: "00 Учебник/16 Multimodal Models/64a Connectors и fusion.md"
last_updated: 2026-10-29
last_verified: 2026-09-15
primary_sources:
  - https://arxiv.org/abs/2304.08485
  - https://arxiv.org/abs/2408.03326
  - https://arxiv.org/abs/2301.12597
  - https://arxiv.org/abs/2204.14198
  - https://arxiv.org/abs/2511.21631
  - https://cs336.stanford.edu/
  - https://cs231n.stanford.edu/slides/2025/lecture_16.pdf
---

# Connecting a vision encoder to an LLM

A vision encoder returns a grid $Z_v\in\mathbb R^{N_v\times d_v}$, while the LLM expects hidden states of width $d_l$. The connector must align dimensions, possibly reduce $N_v$, and decide where visual information enters the language network. Those choices control information loss, context usage, and serving cost.

<span id="2-llava-линейная-проекция-и-единая-последовательность"></span>

## LLaVA: projection into one token stream

The original LLaVA takes CLIP ViT patch features and learns

$$
\begin{gathered}
H_v=Z_vW,\\
W\in\mathbb R^{d_v\times d_l},
\\
H_v\in\mathbb R^{N_v\times d_l}.
\end{gathered}
$$

The projected vectors have the LLM embedding width and are inserted next to the language instruction. The decoder architecture remains unchanged; visual and text positions interact through causal self-attention.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/llava-figure1-projector.png]]

*Figure 1 from Liu et al., [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485), PDF p. 4. The projector maps CLIP patch features into the language model’s input stream.*

This design inherits standard decoder kernels and caching. Its cost is equally direct: every image token occupies context and contributes to LLM prefill. A projector does not translate an image into words; it produces continuous vectors that the LLM learns to interpret. Information absent from the vision features cannot be restored by the projector.

### A projector changes width, not sequence length

For a LLaVA-1.5-style example, CLIP ViT-L/14 at $336\times336$ supplies 576 patch vectors of width 1024. Its two-layer MLP maps $[576,1024]$ to $[576,4096]$ for a language backbone of width 4096. The number of positions is unchanged. With 40 instruction tokens, the prefix contains 616 positions before generating the answer; a 60-token answer takes the conceptual sequence length to 676, ignoring template markers.

These numbers explain the bottleneck. A small connector may be cheap to train while its output creates substantial KV state in every LLM layer. Replacing a linear map with an MLP does not by itself lower that cache cost.

The original LLaVA uses a linear projector. LLaVA-1.5's `mlp2x_gelu`, in the [pinned implementation](https://github.com/haotian-liu/LLaVA/tree/c121f0432da27facab705978f83c4ada465e46fd), computes

$$
H_v=\operatorname{GELU}(Z_vW_1+b_1)W_2+b_2.
$$

For widths 1024 and 4096, the linear layer has 4,198,400 parameters including its bias; the two-layer MLP has 20,979,712. Both return 576 positions. Width conversion and sequence compression are separate operations. Cross-attention is another possibility: its own key/value projections can accept visual features without first making every visual vector an LLM input embedding.

### Where the image placeholder goes

The input processor replaces the image placeholder with a sequence of visual embeddings. It does not turn the entire image into one vector. Text positions after the placeholder shift accordingly. In this miniature example, two prefix positions, three visual positions, two question positions, and two answer positions form a nine-position sequence:

```python
import numpy as np

visual = np.array([[1., 0.], [0., 1.], [1., 1.]])
W = np.array([[1., 0., 1.], [0., 2., 1.]])
projected = visual @ W
assert np.array_equal(projected, [[1, 0, 1], [0, 2, 1], [1, 2, 2]])

before = np.zeros((2, 3))
question = np.ones((2, 3))
answer = np.full((2, 3), 2.)
sequence = np.concatenate([before, projected, question, answer])
IGNORE = -100
labels = np.array([IGNORE] * 7 + [7, 8])
assert sequence.shape == (9, 3)

predictor_positions = np.flatnonzero(labels[1:] != IGNORE)
assert predictor_positions.tolist() == [6, 7]
causal = np.tril(np.ones((9, 9), dtype=bool))
assert causal[6, 2:5].all()
assert not causal[2, 3]
```

The output at position 6 predicts the first answer token at position 7; position 7 predicts the next. `-100` excludes a position as a supervised target, not as an attention source. Answer loss can still send gradients through attention into the projected image features. Text before the image cannot attend to later image positions in the causal decoder. The visual features themselves were already contextualized by the vision encoder's non-causal attention.

The 616-position prefill in the earlier example produces the distribution for the first answer token. Selecting that token does not yet put its K/V into the cache. The next forward pass processes it, grows the cache to 617 positions, and predicts the second token. The 676-position sequence describes teacher forcing with all 60 answer tokens, not work that must finish before the first generated token can appear.

For a hypothetical decoder with 32 layers, 32 KV heads, head dimension 128, and two-byte cache elements, each cached position uses $2\times32\times32\times128\times2=524{,}288$ bytes, or 0.5 MiB. Thus 576, 64, and 32 additional positions use 288, 32, and 16 MiB respectively. These are comparisons under one specified MHA configuration, not measured cache sizes for BLIP-2 or Flamingo. GQA, cache precision, and separate cross-attention memory change the calculation.

<span id="от-llava-к-llava-onevision-что-изменилось-кроме-projector"></span>

### From LLaVA to OneVision

[LLaVA-OneVision](https://arxiv.org/abs/2408.03326) retains the vision-encoder → projector → LLM structure but changes the surrounding system: SigLIP supplies visual features, a two-layer MLP connects them to Qwen2, and the data and visual budgets cover images, multiple images, and video.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/multimodal/llava-onevision.png]]

*Stanford CS336, Lecture 17, architecture illustration based on Li et al., [LLaVA-OneVision, Figure 1](https://arxiv.org/abs/2408.03326). Separate feature extraction, dimensional projection, and language processing when tracing the arrows.*

An improvement over early LLaVA cannot be assigned to the connector alone: resolution, vision features, language backbone, and training data also change. Nor are two projectors interchangeable merely because their matrix shapes agree. A trained connector targets the geometry of particular vision and language checkpoints.

## BLIP-2: Q-Former as a learned bottleneck

BLIP-2 keeps both a strong image encoder and a strong LLM frozen. A Querying Transformer places a fixed number of learned query embeddings between them. Queries cross-attend to image features and produce a compact visual representation.

The 32 query embeddings are learned parameters, not a user's question and not 32 selected image patches. Every image starts with the same query parameters; cross-attention makes their outputs image-dependent. Q-Former uses width 768, a shared self-attention module for queries and text, BERT-base initialization, and newly initialized cross-attention inserted every other block.

![[00 Учебник/Assets/Figures/curated/connectors-review-2026-10/blip2-qformer.png]]

*Li et al., [BLIP-2, Figure 2, left panel](https://proceedings.mlr.press/v202/li23q.html), CC BY 4.0. Learned queries read the frozen image encoder's features; the large LLM is not involved in this first training stage.*

### Three objectives, three attention masks

![[00 Учебник/Assets/Figures/curated/connectors-review-2026-10/blip2-attention-masks.png]]

*BLIP-2, Figure 2, right panel, CC BY 4.0. Rows are querying positions and columns are keys. White cells permit attention; gray cells prohibit it.*

| Attention direction | Contrastive (ITC) | Matching (ITM) | Grounded generation (ITG) |
|---|---|---|---|
| Query → query | allowed | allowed | allowed |
| Query → text | blocked | allowed | blocked |
| Text → query | blocked | allowed | allowed |
| Text → text | bidirectional | bidirectional | causal |

**ITC** encodes queries and text independently. If queries could read the paired caption before contrastive comparison, they could copy the information they were meant to retrieve from the image. The image–text similarity is the maximum similarity between the text's CLS representation and any of the 32 query outputs. It is not similarity with their mean: for queries $(1,0),(0,1)$ and text $(1,0)$, the maximum dot product is 1, whereas the mean query gives 0.5.

**ITM** asks whether image and text match after allowing both directions of interaction. A classifier produces two-class logits for each query; the logits are averaged before softmax. Averaging already normalized probabilities is a different operation. For query logits $(-1,2)$ and $(0,1)$, the mean logits are $(-0.5,1.5)$ and the positive probability is approximately 0.880797; averaging the two positive probabilities instead gives about 0.842.

**ITG** predicts a caption. Queries can see each other and image features but not the caption. Text can attend to all queries and preceding text. This lets caption prediction use the image without letting query outputs leak future answer tokens.

```python
import numpy as np

def qformer_mask(mode, queries=2, texts=3):
    size = queries + texts
    allowed = np.zeros((size, size), dtype=bool)
    allowed[:queries, :queries] = True
    if mode == "ITC":
        allowed[queries:, queries:] = True
    elif mode == "ITM":
        allowed[:] = True
    elif mode == "ITG":
        allowed[queries:, :queries] = True
        allowed[queries:, queries:] = np.tril(np.ones((texts, texts), dtype=bool))
    else:
        raise ValueError("unknown objective")
    return allowed

itg = qformer_mask("ITG")
assert itg.astype(int).tolist() == [
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0],
    [1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
]
assert not qformer_mask("ITC")[:2, 2:].any()
assert not qformer_mask("ITC")[2:, :2].any()
assert qformer_mask("ITM").all()
queries = np.eye(2)
text_cls = np.array([1., 0.])
assert (queries @ text_cls).max() == 1
matching_logits = np.array([[-1., 2.], [0., 1.]])
mean_logits = matching_logits.mean(axis=0)
matching_prob = np.exp(mean_logits - mean_logits.max())
matching_prob /= matching_prob.sum()
assert np.allclose(mean_logits, [-.5, 1.5])
assert np.isclose(matching_prob[1], .8807970779)
```

Real batches also need padding masks. The three objectives share parameters but cannot exchange masks without changing what information is available to the prediction.

Thirty-two queries make LLM-side cost predictable regardless of the original patch count. They are also an information bottleneck: the same limited representation must support global captioning and questions about tiny details.

In the second BLIP-2 stage, Q-Former outputs are projected to the frozen LLM's input width and trained for image-conditioned generation. Decoder-only and encoder–decoder backbones use different input arrangements. Frozen parameters still participate in differentiation: the loss must backpropagate through the LLM to train the connector, even though no optimizer updates are applied to LLM weights.

![[00 Учебник/Assets/Figures/curated/connectors-review-2026-10/blip2-llm-connection.png]]

*BLIP-2, [Figure 3](https://proceedings.mlr.press/v202/li23q.html), CC BY 4.0. OPT receives projected queries as a decoder prefix; Flan-T5 receives them on the encoder side.*

For an LLM of width 4096, projection maps $[32,768]$ to $[32,4096]$. With OPT, these positions precede text in causal decoding. With Flan-T5, visual positions and a caption prefix enter the encoder; the decoder predicts the remaining caption. The latter is not a 32-position visual prefix in the decoder's self-attention cache. Q-Former and its projection train while the image encoder and LLM remain frozen. A fixed 32-position output does not make the vision encoder free: its work and Q-Former's cross-attention still depend on the input grid. Nor does it guarantee preservation of small characters. In the original design, the image queries are not conditioned on the eventual user's instruction.

## Flamingo: a separate visual memory

Flamingo targets interleaved image, video, and text sequences. A Perceiver Resampler maps a variable spatiotemporal grid to a fixed set of latents. Newly inserted gated cross-attention layers let text states query that memory between frozen LM blocks.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/flamingo-figure3-architecture.png]]

*Figure 3 from Alayrac et al., [Flamingo](https://arxiv.org/abs/2204.14198), PDF p. 4. Snowflakes mark frozen vision and language components; the resampler and gated cross-attention are trained from scratch.*

A frozen NFNet-F6 supplies spatial features, with temporal information added for video. The Perceiver Resampler produces 64 latents per image or video. Image markers remain in the text stream, but the latents are held in separate memory. With 100 text positions and 64 latents, a text-to-image cross-attention head has a $100\times64$ score matrix, not a $164\times164$ self-attention matrix.

The inserted GATED XATTN-DENSE block has two gated branches:

$$
\begin{gathered}
U=Y+\tanh(\alpha_a)\operatorname{CrossAttn}(Y,R),
\\
Y'=U+\tanh(\alpha_f)\operatorname{FFW}(U).
\end{gathered}
$$

Both gates start at exactly zero. The inserted branches initially act as the identity before the frozen LM block. This does not prevent learning: in $Y+\tanh(\alpha)F_\theta(Y,R)$, the initial gradient to $\theta$ is zero, but the gradient to $\alpha$ includes $F_\theta$ because $\tanh'(0)=1$. After the gate changes, gradients reach the branch parameters.

### Which image can a text token see?

Original Flamingo directly exposes only the most recent preceding image or video, not every preceding image. In “image A — description A — image B — question about B,” the question's cross-attention reads B. Information about A can still reach it through causal text states, because the earlier description already attended to A.

```python
import numpy as np

latest = np.array([-1, 0, 0, 1, 1])
latent_image = np.array([0, 0, 1, 1])
allowed = latest[:, None] == latent_image[None, :]
assert allowed.astype(int).tolist() == [
    [0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0],
    [0, 0, 1, 1], [0, 0, 1, 1],
]
weights = np.zeros((5, 4))
for row in range(5):
    columns = allowed[row]
    if columns.any():
        weights[row, columns] = 1 / columns.sum()
assert np.allclose(weights.sum(axis=1), [0, 1, 1, 1, 1])
assert np.allclose(weights[3], [0, 0, .5, .5])
theta, alpha, target = 2., 0., 1.
value = np.tanh(alpha) * theta
grad_theta = (value - target) * np.tanh(alpha)
grad_alpha = (value - target) * theta * (1 - np.tanh(alpha)**2)
assert grad_theta == 0
assert grad_alpha == -2
```

Here −1 means that no image precedes the text position. Equal allowed logits are used only to illustrate the mask. Real logits depend on text and image content. An all-masked row needs explicit handling: softmax over only $-\infty$ is not a valid distribution.

Visual latents do not occupy 64 positions in the text self-attention cache, but they still consume memory. Each inserted cross-attention layer may cache projected visual K/V once and reuse them during decoding. That cache has its own layer count, heads, and widths. A decoder-only runtime therefore needs support for the added layers, image masks, and visual memory, even if the attention operations themselves use standard efficient kernels.

## Resampling, early fusion, and deep fusion

Q-Former and Perceiver Resampler both use latent queries to read a longer grid, but their training and placement differ. Q-Former is explicitly trained with text-aware objectives before connecting to an LLM. Flamingo’s resampler normalizes image/video features, which are then queried repeatedly by layer-wise cross-attention.

LLaVA is an early-fusion design: vision and text meet before the first LLM layer. Flamingo uses layer-wise fusion. Qwen3-VL adds a DeepStack variation, injecting features from several vision-encoder depths into corresponding early LLM layers. Repeated injection shortens the path from vision to deep language states but complicates implementation.

### Trace a learned resampler

Suppose the encoder returns $Z_v\in\mathbb R^{576\times1024}$ and the interface has a budget of 64 latents. A simplified learned-query attention block computes

$$
\begin{gathered}
A=\operatorname{softmax}\left(\frac{QK(Z_v)^\top}{\sqrt{d_k}}\right),
\\
H=AV(Z_v).
\end{gathered}
$$

$A$ has shape $[64,576]$: every output latent mixes information from the input patches. After further blocks and projection, the LLM may receive $[64,4096]$. These are not 64 selected squares. Attention weights depend on image content and may spread across many patches; a resampler attention map is not automatically a segmentation mask.

This simplified formula is not the full Flamingo resampler. [Appendix A.1.1 and Figure 5](https://arxiv.org/abs/2204.14198v2) also include the current latents in keys and values:

$$
\begin{gathered}
K_{\rm all}=[K(Z_v);K(R)],\\
V_{\rm all}=[V(Z_v);V(R)].
\end{gathered}
$$

The attention matrix is then $[M,N_v+M]$, but the output still has $M$ positions. Residual connections and a feed-forward block update the latents; the next block reads the visual grid again. This small example uses identity projections and omits normalization and the MLP:

```python
import numpy as np

def attention(q, k, v):
    scores = q @ k.T / np.sqrt(q.shape[-1])
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ v, weights

visual = np.array([[1., 0.], [0., 1.], [1., 1.], [-1., 0.]])
latents = np.array([[1., 0.], [0., 1.]])
plain, a_plain = attention(latents, visual, visual)
memory = np.concatenate([visual, latents])
update, a_full = attention(latents, memory, memory)
result = latents + update
assert a_plain.shape == (2, 4)
assert a_full.shape == (2, 6)
assert result.shape == (2, 2)
assert np.allclose(a_full.sum(axis=1), 1)
assert not np.allclose(plain, update)
assert np.all(a_full[:, 4:] > 0)
```

For the first query, unscaled similarities to the four visual vectors are $(1,0,1,-1)$; the two added latents contribute $(1,0)$. All six participate in one softmax. Their weights are approximately $(0.2364,0.1166,0.2364,0.0575,0.2364,0.1166)$, giving $(1.6519,0.4696)$ after the residual addition. This is a calculation for specified vectors, not an output measured on a photograph. After nonlinear blocks, a latent cannot be interpreted as a linear average of pixels. Q-Former shares the use of learned queries but adds a text branch and three pre-LLM training objectives; Flamingo trains its resampler and added cross-attention as part of a generative system.

| Interface | Example shapes | Changes | Does not guarantee |
|---|---|---|---|
| MLP projector | $[576,1024]\to[576,4096]$ | feature width and geometry | token compression or recovered detail |
| Learned resampler | $[576,1024]\to[64,4096]$ | sequence length and aggregated content | preservation of every small character |

A fixed budget is convenient for video but treats a simple object photograph and a dense newspaper page alike. The latter may need dynamic resolution or a larger latent budget.

<span id="6-early-fusion-и-deep-fusion"></span>

### DeepStack is a different route into the same decoder

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/multimodal/qwen3-vl.png]]

*Stanford CS336, Lecture 17, illustration based on the [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631). The main merger supplies a visual prefix; DeepStack additionally injects intermediate vision features into early LLM layers.*

Qwen3-VL's main merger groups neighboring $2\times2$ patches before its MLP, reducing the number of positions fourfold. This grouping, not the MLP alone, causes compression. Separate mergers process features from three intermediate vision layers. In the [Transformers v4.57.1 implementation](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L862), those features are added after each of the first three decoder blocks, at visual positions only. They neither add three new copies of the prefix nor modify every text position:

```python
import numpy as np

hidden = np.arange(15, dtype=float).reshape(5, 3)
visual_mask = np.array([False, True, True, False, False])
features = [
    np.array([[1., 0., 0.], [0., 1., 0.]]),
    np.array([[0., 0., 1.], [1., 0., 0.]]),
    np.array([[0., 1., 0.], [0., 0., 1.]]),
]
before = hidden.copy()
for level in features:
    assert level.shape == hidden[visual_mask].shape
    hidden[visual_mask] += level
assert hidden.shape == (5, 3)
assert np.array_equal(hidden[~visual_mask], before[~visual_mask])
assert np.array_equal(hidden[visual_mask] - before[visual_mask], np.ones((2, 3)))
```

This tests only the masked additions. The real decoder places different blocks between them; adding all three feature sets once at the input would compute something else. Repeated injection requires intermediate features and alignment even though sequence length is unchanged.

“Early fusion” describes where modalities meet, not whether they share a discrete vocabulary. LLaVA supplies continuous vectors that its output head cannot emit as image IDs. Chameleon instead uses discrete image codes in the autoregressive stream. Both can fuse early, but their generation contracts differ.

## Which parameters should move?

Original LLaVA first trains only the projector with CLIP and Vicuna frozen, then trains the projector and LLM while keeping the vision encoder frozen. BLIP-2 keeps both large backbones frozen during its second stage. Flamingo trains the resampler and inserted GATED XATTN-DENSE blocks. These are distinct training choices, not a universal prescription for all VLMs.

A reproducible experiment must state trainable and frozen components at every stage. “Trained on image-text data” is not enough to determine where gradients flowed.

Freezing the LLM removes weight gradients and optimizer state for its parameters, but not necessarily its activation-memory cost: connector training needs derivatives through the downstream computation. Conversely, replacing the vision encoder changes the feature distribution even when output width is identical. Component swaps need retraining and task evaluation, not just a successful shape assertion.

If $h=P_\theta(z)$ and $y=F(h)$, training the projector requires

$$
\frac{\partial\mathcal L}{\partial\theta}
=\frac{\partial h}{\partial\theta}^{\!\top}
\frac{\partial y}{\partial h}^{\!\top}
\frac{\partial\mathcal L}{\partial y}.
$$

The parameters of $F$ need not change, but its derivative with respect to $h$ is still needed. In PyTorch, freezing parameters is therefore different from wrapping the forward pass in `no_grad()`:

```python
import torch

projector = torch.nn.Linear(2, 2, bias=False).double()
frozen = torch.nn.Linear(2, 2, bias=False).double()
with torch.no_grad():
    projector.weight.copy_(torch.eye(2, dtype=torch.float64))
    frozen.weight.copy_(torch.diag(torch.tensor([2., 3.], dtype=torch.float64)))
frozen.requires_grad_(False)
x = torch.tensor([[1., 2.]], dtype=torch.float64)
y = frozen(projector(x))
loss = .5 * y.square().sum()
loss.backward()
assert loss.item() == 20
assert frozen.weight.grad is None
assert torch.equal(projector.weight.grad, torch.tensor([[4., 8.], [18., 36.]], dtype=torch.float64))
with torch.no_grad():
    detached = frozen(projector(x))
assert not detached.requires_grad
```

Putting the frozen module's entire forward inside `no_grad()` would sever the projector's gradient path. Input derivatives and activation storage can remain necessary even when weight gradients and optimizer state are absent.

[VILA, §3.1 and Table 1](https://arxiv.org/abs/2312.07533v2) compares freezing and training the LLM during multimodal pretraining. Under that experiment's conditions, zero-shot scores could remain similar while four-example in-context performance was worse with the frozen LLM. This is an ablation result, not proof that every frozen backbone is unsuitable. Sections 3.2–3.3 separately examine interleaved data and text-only instructions. Visual benchmarks alone do not show whether text-only abilities survived unfreezing; [[en/00 Textbook/16 Multimodal Models/64c Training VLMs — alignment, instruction tuning, and data|the training chapter]] develops these data choices.

## Choosing a connector

A projector is the simplest default for a moderate number of image tokens. A learned bottleneck is attractive when backbone training is expensive or LLM context is scarce. Separate cross-attention memory fits long interleaved inputs and repeated access to visual evidence. Documents may remain limited by resolution before the connector is reached.

Connector comparisons should hold the vision encoder, LLM, data, resolution, and visual-token budget fixed. Otherwise a data or resolution improvement is easily misattributed to fusion architecture.

| Design | LLM-side visual length | Fusion point | Serving consequence |
|---|---|---|---|
| LLaVA projector | usually one position per retained patch | before the decoder | standard causal self-attention with a longer prefix |
| BLIP-2 Q-Former | fixed learned-query count | before the LLM | compact prefix, potentially lossy bottleneck |
| Flamingo resampler and cross-attention | fixed latents per image | inside selected layers | separate visual memory and modified blocks |
| Qwen3-VL merger and DeepStack | merged grid plus aligned intermediate features | input and first three decoder blocks | masked feature additions without extra prefix copies |

Twenty frames at 576 vectors each create 11,520 visual positions before text if no compression is used. Under that workload, a resampler is a substantial cost-control decision, not merely an architectural embellishment. Compare OCR, grounding, spatial relations, TTFT, and KV memory alongside aggregate VQA.

When replacing components, preserve preprocessing, feature-layer selection, grid order, CLS handling, positional representations, and checkpoint versions. Neither a resampler nor an unfrozen LLM can recover an answer removed by the image crop. Comparisons with different token budgets should report those budgets alongside quality, latency, and memory; the count of trainable parameters does not describe all computation through a frozen backbone.

## Sources and next chapter

- Stanford CS231n, [Multimodal Foundation Models](https://cs231n.stanford.edu/slides/2025/lecture_16.pdf), PDF pp. 57–80.
- [LLaVA](https://arxiv.org/abs/2304.08485), [BLIP-2](https://arxiv.org/abs/2301.12597), and [Flamingo](https://arxiv.org/abs/2204.14198).
- [LLaVA-1.5, §3.3](https://arxiv.org/abs/2310.03744v2), [LLaVA-OneVision, §3.1](https://arxiv.org/abs/2408.03326v1), and [Stanford CS336 Lecture 17](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_17.py).
- [BLIP-2, §3.1–3.3](https://proceedings.mlr.press/v202/li23q.html) and its [LAVIS stage-one implementation](https://github.com/salesforce/LAVIS/blob/05cb670f47296522e24e7727e366b4a997d28582/lavis/models/blip2_models/blip2_qformer.py).
- [Flamingo, §2.1–2.3 and Appendix A.1.1–A.1.3](https://arxiv.org/abs/2204.14198v2), [Qwen3-VL, §2–2.2](https://arxiv.org/abs/2511.21631v1), and [VILA, §3.1–3.3](https://arxiv.org/abs/2312.07533v2).
- Previous: [[en/00 Textbook/16 Multimodal Models/64 Multimodal models|ViT, CLIP, and SigLIP]]. Next: [[en/00 Textbook/16 Multimodal Models/64b Resolution, tiling, and spatial positions|Resolution and spatial positions]].
