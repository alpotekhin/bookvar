---
title: "64.2. Resolution, tiling, and spatial positions"
type: textbook-chapter
status: canonical
locale: en
translation_of: "00 Учебник/16 Multimodal Models/64b Разрешение, tiling и пространственные позиции.md"
last_updated: 2026-07-23
---

# Resolution, tiling, and spatial positions

A photograph can survive aggressive downsampling; a contract may lose the digits that determine its meaning. Fixed classification-era resolutions are therefore a central VLM limitation. The model must allocate enough visual tokens and preserve coordinates through resize, padding, crops, and merging.

For patch size $P$, $N=HW/P^2$. Doubling both dimensions creates four times as many patches. Compute is governed by the post-preprocessing token count, not the JPEG file size.

## Global overview plus detailed crops

Molmo combines a low-resolution view of the full image with overlapping high-resolution crops. The overview preserves composition; crops preserve fine detail; overlap prevents boundary objects from losing context. Special image and row/column tokens record the final layout.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/molmo-figure5-image-tokenization.png]]

*Figure 5 from Deitke et al., [Molmo and PixMo](https://arxiv.org/abs/2409.17146), PDF p. 9. The figure traces an image through low- and high-resolution crops into a token sequence.*

InternVL chooses a grid of $448\times448$ tiles according to aspect ratio. Pixel unshuffle reduces 1,024 patch features per tile to 256 visual tokens. Training and inference impose maximum tile counts; accepting a longer sequence technically does not prove quality beyond the training distribution.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/internvl25-figure2-dynamic-tiles.png]]

*Figure 2 from Wang et al., [InternVL 2.5](https://arxiv.org/abs/2412.05271), PDF p. 3. Dynamic tiling feeds a ViT–MLP–LLM stack for images, multiple images, and video.*

## Native dynamic resolution and compression

Qwen2-VL and Pixtral accept variable image sizes rather than selecting only from a crop grid. Pixtral uses RoPE-2D, row-break tokens, and block-diagonal masks for packed images. Qwen2.5-VL uses mostly windowed attention in the vision encoder with selected global layers. “Native resolution” still includes resizing to allowed multiples and `min_pixels`/`max_pixels` limits.

The LLM need not receive every encoder patch. A $2\times2$ merger quarters sequence length; attention pooling, Q-Former, and Perceiver Resampler compress more aggressively. Evaluation must include OCR and small-object grounding, not only average VQA, because compression first removes fine detail.

## From one-dimensional order to image geometry

Flattening a grid into one sequence makes the end of one row adjacent to the start of the next. Qwen2-VL’s M-RoPE partitions rotary dimensions among time, height, and width. Text uses matching indices; images keep time fixed while height and width vary; video varies all three.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/qwen2vl-figure3-mrope.png]]

*Figure 3 from Wang et al., [Qwen2-VL](https://arxiv.org/abs/2409.12191), PDF p. 5. M-RoPE separates temporal, vertical, and horizontal positions.*

Qwen2.5-VL aligns video positions with physical time; Qwen3-VL adds interleaved M-RoPE and explicit timestamp tokens. These are architectural contracts with frame sampling, not cosmetic embedding variants.

## Returning coordinates to the original image

A box predicted inside a tile must be offset by the crop origin, corrected for overlap and padding, and divided by resize scale. Preprocessing metadata must therefore travel with the tensor. When both overview and crops contain the same object, the training format must unambiguously state which coordinate system the target uses.

Production logs should record original and resized dimensions, crop count, visual-token count, and preprocessing version. Without them, latency and grounding regressions cannot be explained.

## Sources and next chapter

- [Molmo and PixMo](https://arxiv.org/abs/2409.17146), [InternVL 2.5](https://arxiv.org/abs/2412.05271), [Qwen2-VL](https://arxiv.org/abs/2409.12191), and [Pixtral 12B](https://arxiv.org/abs/2410.07073).
- Previous: [[en/00 Textbook/16 Multimodal Models/64a Connectors and fusion|64.1]]. Next: [[en/00 Textbook/16 Multimodal Models/64c Training VLMs — alignment, instruction tuning, and data|64.3]].
