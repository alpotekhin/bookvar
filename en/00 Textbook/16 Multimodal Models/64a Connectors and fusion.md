---
title: "64.1. Connecting a vision encoder to an LLM"
type: textbook-chapter
status: canonical
locale: en
translation_of: "00 Учебник/16 Multimodal Models/64a Connectors и fusion.md"
last_updated: 2026-07-23
primary_sources:
  - https://arxiv.org/abs/2304.08485
  - https://arxiv.org/abs/2301.12597
  - https://arxiv.org/abs/2204.14198
---

# Connecting a vision encoder to an LLM

A vision encoder returns a grid $Z_v\in\mathbb R^{N_v\times d_v}$, while the LLM expects hidden states of width $d_l$. The connector must align dimensions, possibly reduce $N_v$, and decide where visual information enters the language network. Those choices control information loss, context usage, and serving cost.

## LLaVA: projection into one token stream

The original LLaVA takes CLIP ViT patch features and learns

$$H_v=WZ_v.$$

The projected vectors have the LLM embedding width and are inserted next to the language instruction. The decoder architecture remains unchanged; visual and text positions interact through causal self-attention.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/llava-figure1-projector.png]]

*Figure 1 from Liu et al., [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485), PDF p. 4. The projector maps CLIP patch features into the language model’s input stream.*

This design inherits standard decoder kernels and caching. Its cost is equally direct: every image token occupies context and contributes to LLM prefill. A projector does not translate an image into words; it produces continuous vectors that the LLM learns to interpret. Information absent from the vision features cannot be restored by the projector.

## BLIP-2: Q-Former as a learned bottleneck

BLIP-2 keeps both a strong image encoder and a strong LLM frozen. A Querying Transformer places a fixed number of learned query embeddings between them. Queries cross-attend to image features and produce a compact visual representation.

Its first stage combines image-text contrastive learning, image-text matching, and image-grounded text generation. Each objective uses a different attention mask, so query-text interaction matches the task.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/blip2-figure2-qformer-objectives.png]]

*Figure 2 from Li et al., [BLIP-2](https://arxiv.org/abs/2301.12597), PDF p. 3. The diagram shows Q-Former and the masks used by its three first-stage objectives.*

Thirty-two queries make LLM-side cost predictable regardless of the original patch count. They are also an information bottleneck: the same limited representation must support global captioning and questions about tiny details.

## Flamingo: a separate visual memory

Flamingo targets interleaved image, video, and text sequences. A Perceiver Resampler maps a variable spatiotemporal grid to a fixed set of latents. Newly inserted gated cross-attention layers let text states query that memory between frozen LM blocks.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/flamingo-figure3-architecture.png]]

*Figure 3 from Alayrac et al., [Flamingo](https://arxiv.org/abs/2204.14198), PDF p. 4. Snowflakes mark frozen vision and language components; the resampler and gated cross-attention are trained from scratch.*

A simplified update is

$$h'=h+\tanh(\alpha)\operatorname{CrossAttn}(h,Z_v).$$

The gate starts near zero, preserving the pretrained language function at initialization. Visual states remain separate instead of participating in every textual self-attention edge. The trade-off is a custom architecture, custom kernels, and more complex caching.

## Resampling, early fusion, and deep fusion

Q-Former and Perceiver Resampler both use latent queries to read a longer grid, but their training and placement differ. Q-Former is explicitly trained with text-aware objectives before connecting to an LLM. Flamingo’s resampler normalizes image/video features, which are then queried repeatedly by layer-wise cross-attention.

LLaVA is an early-fusion design: vision and text meet before the first LLM layer. Flamingo uses layer-wise fusion. Qwen3-VL adds a DeepStack variation, injecting features from several vision-encoder depths into corresponding early LLM layers. Repeated injection shortens the path from vision to deep language states but complicates implementation.

## Which parameters should move?

Training only the connector is inexpensive and preserves pretrained skills. Unfreezing the vision encoder helps when its original features are unsuitable for OCR or a specialist domain. Unfreezing the LLM supports richer multimodal interaction and in-context learning but risks degrading language-only behavior. Text-only examples are often retained in the mixture for this reason.

A reproducible experiment must state trainable and frozen components at every stage. “Trained on image-text data” is not enough to determine where gradients flowed.

## Choosing a connector

A projector is the simplest default for a moderate number of image tokens. A learned bottleneck is attractive when backbone training is expensive or LLM context is scarce. Separate cross-attention memory fits long interleaved inputs and repeated access to visual evidence. Documents may remain limited by resolution before the connector is reached.

Connector comparisons should hold the vision encoder, LLM, data, resolution, and visual-token budget fixed. Otherwise a data or resolution improvement is easily misattributed to fusion architecture.

## Sources and next chapter

- Stanford CS231n, [Multimodal Foundation Models](https://cs231n.stanford.edu/slides/2025/lecture_16.pdf), PDF pp. 57–80.
- [LLaVA](https://arxiv.org/abs/2304.08485), [BLIP-2](https://arxiv.org/abs/2301.12597), and [Flamingo](https://arxiv.org/abs/2204.14198).
- Previous: [[en/00 Textbook/16 Multimodal Models/64 Multimodal models|64]]. Next: [[en/00 Textbook/16 Multimodal Models/64b Resolution, tiling, and spatial positions|64.2]].
