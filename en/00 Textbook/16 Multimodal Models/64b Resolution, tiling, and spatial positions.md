---
title: "64.2. Resolution, tiling, and spatial positions"
type: textbook-chapter
status: canonical
locale: en
translation_of: "00 Учебник/16 Multimodal Models/64b Разрешение, tiling и пространственные позиции.md"
last_updated: 2026-09-15
last_verified: 2026-09-15
primary_sources:
  - https://arxiv.org/abs/2408.03326
  - https://arxiv.org/abs/2409.17146
  - https://arxiv.org/abs/2412.05271
  - https://arxiv.org/abs/2409.12191
  - https://arxiv.org/abs/2511.21631
  - https://cs336.stanford.edu/
---

# Resolution, tiling, and spatial positions

A photograph can survive aggressive downsampling; a contract may lose the digits that determine its meaning. Fixed classification-era resolutions are therefore a central VLM limitation. The model must allocate enough visual tokens and preserve coordinates through resize, padding, crops, and merging.

For patch size $P$, $N=HW/P^2$. Doubling both dimensions creates four times as many patches. Compute is governed by the post-preprocessing token count, not the JPEG file size.

A square resize has two distinct costs. Stretching changes aspect ratio; fitting inside a square introduces padding and may shrink the long side so far that characters disappear. More reasoning cannot recover an unreadable digit.

| Prepared size, patch size 14 | Patch grid | Patch positions | After a $2\times2$ merger |
|---|---:|---:|---:|
| $224\times224$ | $16\times16$ | 256 | 64 |
| $336\times336$ | $24\times24$ | 576 | 144 |
| $672\times672$ | $48\times48$ | 2,304 | 576 |
| $1344\times1344$ | $96\times96$ | 9,216 | 2,304 |

The final row has as many merged positions as the preceding row has unmerged positions. “1344-pixel input” therefore does not determine LLM cost without patch size, merger, crop count, and special-token conventions.

## Global overview plus detailed crops

Molmo combines a low-resolution view of the full image with overlapping high-resolution crops. The overview preserves composition; crops preserve fine detail; overlap prevents boundary objects from losing context. Special image and row/column tokens record the final layout.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/molmo-figure5-image-tokenization.png]]

*Figure 5 from Deitke et al., [Molmo and PixMo](https://arxiv.org/abs/2409.17146), PDF p. 9. The figure traces an image through low- and high-resolution crops into a token sequence.*

InternVL chooses a grid of $448\times448$ tiles according to aspect ratio. Pixel unshuffle reduces 1,024 patch features per tile to 256 visual tokens. Training and inference impose maximum tile counts; accepting a longer sequence technically does not prove quality beyond the training distribution.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/internvl25-figure2-dynamic-tiles.png]]

*Figure 2 from Wang et al., [InternVL 2.5](https://arxiv.org/abs/2412.05271), PDF p. 3. Dynamic tiling feeds a ViT–MLP–LLM stack for images, multiple images, and video.*

### AnyRes: restore detail without discarding the overview

LLaVA-NeXT/OneVision chooses among allowed grids of encoder-sized tiles. The low-resolution whole-image view preserves context; local tiles preserve detail. Encoded tile features are arranged back into their spatial grid before flattening, with feature interpolation when needed to enforce a token budget.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/multimodal/llava-onevision-anyres.png]]

*Stanford CS336, Lecture 17, AnyRes illustration based on [LLaVA-OneVision](https://arxiv.org/abs/2408.03326). Follow grid choice, independent views, and the assembled feature grid; each stage can preserve or lose different information.*

The overview cannot add fine resolution to a crop, but tells the model where that crop belongs. A large letter may be legible locally while its column heading is visible only globally. Increasing tile count helps only if layout markers and positions keep these sources distinguishable and later compression does not remove the regained detail.

## Native dynamic resolution and compression

Qwen2-VL and Pixtral accept variable image sizes rather than selecting only from a crop grid. Pixtral uses RoPE-2D, row-break tokens, and block-diagonal masks for packed images. Qwen2.5-VL uses mostly windowed attention in the vision encoder with selected global layers. “Native resolution” still includes resizing to allowed multiples and `min_pixels`/`max_pixels` limits.

The LLM need not receive every encoder patch. A $2\times2$ merger quarters sequence length; attention pooling, Q-Former, and Perceiver Resampler compress more aggressively. Evaluation must include OCR and small-object grounding, not only average VQA, because compression first removes fine detail.

### Qwen2-VL: count patches, merged vectors, and delimiters separately

With $14\times14$ spatial patches and a $2\times2$ spatial merger, a prepared $224\times224$ image yields

$$
(224/14)^2=256\quad\text{patch positions},
\qquad 256/4=64\quad\text{merged visual vectors}.
$$

A reported length of 66 cannot mean 66 merged patches under this geometry. Two additional positions, if present, must be named—for example image-boundary markers. The actual processor decides the total stream length.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/multimodal/qwen2-vl-architecture.png]]

*Stanford CS336, Lecture 17, based on Wang et al., [Qwen2-VL, Figure 1](https://arxiv.org/abs/2409.12191). The figure shows variable visual-sequence length; the $256\to64$ calculation follows the paper's spatial merger, not an extra operation drawn in this overview.*

“Native” resolution still means a processor-constrained grid. Record the resized dimensions and allowed pixel range rather than assuming that every original pixel reaches the encoder unchanged.

## From one-dimensional order to image geometry

Flattening a grid into one sequence makes the end of one row adjacent to the start of the next. Qwen2-VL’s M-RoPE partitions rotary dimensions among time, height, and width. Text uses matching indices; images keep time fixed while height and width vary; video varies all three.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/qwen2vl-figure3-mrope.png]]

*Figure 3 from Wang et al., [Qwen2-VL](https://arxiv.org/abs/2409.12191), PDF p. 5. M-RoPE separates temporal, vertical, and horizontal positions.*

Qwen2.5-VL scales temporal position IDs with physical time. [Qwen3-VL replaces that absolute-time alignment](https://arxiv.org/html/2511.21631v1#S2.SS3) with explicit textual timestamps before video temporal patches; it does not simply add timestamps to the same time-scaled-ID scheme. Interleaved M-RoPE remains: temporal, height, and width coordinates are distributed across rotary frequency ranges rather than assigned one contiguous band per axis. These are separate decisions about representing physical time and allocating positional frequencies. Neither repairs incorrect timestamps or a wrong patch grid.

## Returning coordinates to the original image

A box predicted inside a tile must be offset by the crop origin, corrected for overlap and padding, and divided by resize scale. Preprocessing metadata must therefore travel with the tensor. Inverting coordinates does not restore pixels lost to downsampling or cropping; it only maps locations within the retained region. When both overview and crops contain the same object, the training format must unambiguously state which coordinate system the target uses.

For example, resize a $1600\times1000$ page by $s=0.5$, then add 12 pixels of left padding and 6 of top padding. A prepared-image box $(112,56,312,156)$ maps back to

$$
\frac{(112-12,\;56-6,\;312-12,\;156-6)}{0.5}
=(200,100,600,300).
$$

For tile-local predictions, first add the tile origin in the padded, resized coordinate system, then remove padding and invert resize. Overlap is already represented by tile origins; it is not a universal extra offset to subtract. Duplicate detections in overlapping tiles need reconciliation. Rotation or nonuniform scaling requires the corresponding inverse transform rather than one scalar.

## Diagnose the stage that lost the evidence

Consider a wrong digit and a systematically shifted box. They need different interventions:

1. Inspect the prepared pixels. If the digit vanished during resize, raise the pixel budget or use a crop.
2. Compare encoder and post-merger representations through downstream probes or controlled ablations. More input pixels are ineffective if later compression discards their useful information.
3. Check row, crop, and source ordering. Correct local content can still be attached to the wrong location.
4. Render the inverse-transformed prediction on the original image. A consistent offset suggests incorrect metadata rather than failed visual recognition.
5. Repeat at several budgets and report quality together with TTFT and memory.

For ordinary photos, begin with a moderate budget and increase it where fine detail matters. Documents and interfaces often need larger budgets, but batching them with short photo requests creates latency trade-offs. A quality–token–latency curve is more useful than one “high-resolution” configuration.

Production logs should record original and resized dimensions, crop count, visual-token count, and preprocessing version. Without them, latency and grounding regressions cannot be explained.

## Sources and next chapter

- [Molmo and PixMo](https://arxiv.org/abs/2409.17146), [InternVL 2.5](https://arxiv.org/abs/2412.05271), [Qwen2-VL](https://arxiv.org/abs/2409.12191), and [Pixtral 12B](https://arxiv.org/abs/2410.07073).
- Previous: [[en/00 Textbook/16 Multimodal Models/64a Connectors and fusion|64.1]]. Next: [[en/00 Textbook/16 Multimodal Models/64c Training VLMs — alignment, instruction tuning, and data|64.3]].
