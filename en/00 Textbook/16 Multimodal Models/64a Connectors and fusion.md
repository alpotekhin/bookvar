---
title: "64.1. Connecting a vision encoder to an LLM"
type: textbook-chapter
status: canonical
locale: en
translation_of: "00 Учебник/16 Multimodal Models/64a Connectors и fusion.md"
last_updated: 2026-09-15
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

## LLaVA: projection into one token stream

The original LLaVA takes CLIP ViT patch features and learns

$$
H_v=Z_vW,\qquad W\in\mathbb R^{d_v\times d_l},
\qquad H_v\in\mathbb R^{N_v\times d_l}.
$$

The projected vectors have the LLM embedding width and are inserted next to the language instruction. The decoder architecture remains unchanged; visual and text positions interact through causal self-attention.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/llava-figure1-projector.png]]

*Figure 1 from Liu et al., [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485), PDF p. 4. The projector maps CLIP patch features into the language model’s input stream.*

This design inherits standard decoder kernels and caching. Its cost is equally direct: every image token occupies context and contributes to LLM prefill. A projector does not translate an image into words; it produces continuous vectors that the LLM learns to interpret. Information absent from the vision features cannot be restored by the projector.

### A projector changes width, not sequence length

For a LLaVA-1.5-style example, CLIP ViT-L/14 at $336\times336$ supplies 576 patch vectors of width 1024. Its two-layer MLP maps $[576,1024]$ to $[576,4096]$ for a language backbone of width 4096. The number of positions is unchanged. With 40 instruction tokens, the prefix contains 616 positions before generating the answer; a 60-token answer takes the conceptual sequence length to 676, ignoring template markers.

These numbers explain the bottleneck. A small connector may be cheap to train while its output creates substantial KV state in every LLM layer. Replacing a linear map with an MLP does not by itself lower that cache cost.

### From LLaVA to OneVision

[LLaVA-OneVision](https://arxiv.org/abs/2408.03326) retains the vision-encoder → projector → LLM structure but changes the surrounding system: SigLIP supplies visual features, a two-layer MLP connects them to Qwen2, and the data and visual budgets cover images, multiple images, and video.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/multimodal/llava-onevision.png]]

*Stanford CS336, Lecture 17, architecture illustration based on Li et al., [LLaVA-OneVision, Figure 1](https://arxiv.org/abs/2408.03326). Separate feature extraction, dimensional projection, and language processing when tracing the arrows.*

An improvement over early LLaVA cannot be assigned to the connector alone: resolution, vision features, language backbone, and training data also change. Nor are two projectors interchangeable merely because their matrix shapes agree. A trained connector targets the geometry of particular vision and language checkpoints.

## BLIP-2: Q-Former as a learned bottleneck

BLIP-2 keeps both a strong image encoder and a strong LLM frozen. A Querying Transformer places a fixed number of learned query embeddings between them. Queries cross-attend to image features and produce a compact visual representation.

Its first stage combines image-text contrastive learning, image-text matching, and image-grounded text generation. Each objective uses a different attention mask, so query-text interaction matches the task.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/blip2-figure2-qformer-objectives.png]]

*Figure 2 from Li et al., [BLIP-2](https://arxiv.org/abs/2301.12597), PDF p. 3. The diagram shows Q-Former and the masks used by its three first-stage objectives.*

Thirty-two queries make LLM-side cost predictable regardless of the original patch count. They are also an information bottleneck: the same limited representation must support global captioning and questions about tiny details.

In the second BLIP-2 stage, Q-Former outputs are projected to the frozen LLM's input width and trained for image-conditioned generation. Decoder-only and encoder–decoder backbones use different input arrangements. Frozen parameters still participate in differentiation: the loss must backpropagate through the LLM to train the connector, even though no optimizer updates are applied to LLM weights.

## Flamingo: a separate visual memory

Flamingo targets interleaved image, video, and text sequences. A Perceiver Resampler maps a variable spatiotemporal grid to a fixed set of latents. Newly inserted gated cross-attention layers let text states query that memory between frozen LM blocks.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/flamingo-figure3-architecture.png]]

*Figure 3 from Alayrac et al., [Flamingo](https://arxiv.org/abs/2204.14198), PDF p. 4. Snowflakes mark frozen vision and language components; the resampler and gated cross-attention are trained from scratch.*

A simplified update is

$$
h'=h+\tanh(\alpha)\operatorname{CrossAttn}(h,Z_v).
$$

The gate starts near zero, preserving the pretrained language function at initialization. Visual states remain separate instead of participating in every textual self-attention edge. The trade-off is a modified architecture and additional attention/caching paths; it does not inherently require inventing a new attention kernel.

## Resampling, early fusion, and deep fusion

Q-Former and Perceiver Resampler both use latent queries to read a longer grid, but their training and placement differ. Q-Former is explicitly trained with text-aware objectives before connecting to an LLM. Flamingo’s resampler normalizes image/video features, which are then queried repeatedly by layer-wise cross-attention.

LLaVA is an early-fusion design: vision and text meet before the first LLM layer. Flamingo uses layer-wise fusion. Qwen3-VL adds a DeepStack variation, injecting features from several vision-encoder depths into corresponding early LLM layers. Repeated injection shortens the path from vision to deep language states but complicates implementation.

### Trace a learned resampler

Suppose the encoder returns $Z_v\in\mathbb R^{576\times1024}$ and the interface has a budget of 64 latents. A simplified learned-query attention block computes

$$
A=\operatorname{softmax}\left(\frac{QK(Z_v)^\top}{\sqrt{d_k}}\right),
\qquad H=AV(Z_v).
$$

$A$ has shape $[64,576]$: every output latent mixes information from the input patches. After further blocks and projection, the LLM may receive $[64,4096]$. These are not 64 selected squares. Attention weights depend on image content and may spread across many patches; a resampler attention map is not automatically a segmentation mask.

| Interface | Example shapes | Changes | Does not guarantee |
|---|---|---|---|
| MLP projector | $[576,1024]\to[576,4096]$ | feature width and geometry | token compression or recovered detail |
| Learned resampler | $[576,1024]\to[64,4096]$ | sequence length and aggregated content | preservation of every small character |

A fixed budget is convenient for video but treats a simple object photograph and a dense newspaper page alike. The latter may need dynamic resolution or a larger latent budget.

### DeepStack is a different route into the same decoder

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/multimodal/qwen3-vl.png]]

*Stanford CS336, Lecture 17, illustration based on the [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631). The main merger supplies a visual prefix; DeepStack additionally injects intermediate vision features into early LLM layers.*

“Early fusion” describes where modalities meet, not whether they share a discrete vocabulary. LLaVA supplies continuous vectors that its output head cannot emit as image IDs. Chameleon instead uses discrete image codes in the autoregressive stream. Both can fuse early, but their generation contracts differ.

## Which parameters should move?

Training only the connector is inexpensive and preserves pretrained skills. Unfreezing the vision encoder helps when its original features are unsuitable for OCR or a specialist domain. Unfreezing the LLM supports richer multimodal interaction and in-context learning but risks degrading language-only behavior. Text-only examples are often retained in the mixture for this reason.

A reproducible experiment must state trainable and frozen components at every stage. “Trained on image-text data” is not enough to determine where gradients flowed.

Freezing the LLM removes weight gradients and optimizer state for its parameters, but not necessarily its activation-memory cost: connector training needs derivatives through the downstream computation. Conversely, replacing the vision encoder changes the feature distribution even when output width is identical. Component swaps need retraining and task evaluation, not just a successful shape assertion.

## Choosing a connector

A projector is the simplest default for a moderate number of image tokens. A learned bottleneck is attractive when backbone training is expensive or LLM context is scarce. Separate cross-attention memory fits long interleaved inputs and repeated access to visual evidence. Documents may remain limited by resolution before the connector is reached.

Connector comparisons should hold the vision encoder, LLM, data, resolution, and visual-token budget fixed. Otherwise a data or resolution improvement is easily misattributed to fusion architecture.

| Design | LLM-side visual length | Fusion point | Serving consequence |
|---|---|---|---|
| LLaVA projector | usually one position per retained patch | before the decoder | standard causal self-attention with a longer prefix |
| BLIP-2 Q-Former | fixed learned-query count | before the LLM | compact prefix, potentially lossy bottleneck |
| Flamingo resampler and cross-attention | fixed latents per image | inside selected layers | separate visual memory and modified blocks |

Twenty frames at 576 vectors each create 11,520 visual positions before text if no compression is used. Under that workload, a resampler is a substantial cost-control decision, not merely an architectural embellishment. Compare OCR, grounding, spatial relations, TTFT, and KV memory alongside aggregate VQA.

## Sources and next chapter

- Stanford CS231n, [Multimodal Foundation Models](https://cs231n.stanford.edu/slides/2025/lecture_16.pdf), PDF pp. 57–80.
- [LLaVA](https://arxiv.org/abs/2304.08485), [BLIP-2](https://arxiv.org/abs/2301.12597), and [Flamingo](https://arxiv.org/abs/2204.14198).
- Previous: [[en/00 Textbook/16 Multimodal Models/64 Multimodal models|64]]. Next: [[en/00 Textbook/16 Multimodal Models/64b Resolution, tiling, and spatial positions|64.2]].
