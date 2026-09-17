---
title: "64. From images to visual tokens: ViT and CLIP"
type: textbook-chapter
status: canonical
locale: en
translation_of: "00 Учебник/16 Multimodal Models/64 Мультимодальные модели.md"
last_updated: 2026-09-15
last_verified: 2026-09-15
primary_sources:
  - https://arxiv.org/abs/2010.11929
  - https://arxiv.org/abs/2103.00020
  - https://arxiv.org/abs/2303.15343
  - https://cs336.stanford.edu/
  - https://cs231n.stanford.edu/slides/2025/lecture_16.pdf
---

# From images to visual tokens: ViT and CLIP

A language model receives a sequence of discrete token indices. An image is a rectangular grid of intensity values. Before a Transformer can process both, the image must become a sequence of vectors. This conversion determines spatial resolution, visual-token count, and which details remain available to every later stage.

## Perception and generation need different representations

Turning an image into “tokens” can mean two different things. For visual question answering, continuous feature vectors are sufficient: they condition a text decoder. For image generation, the model also needs an output representation that can be decoded back into pixels. A retrieval embedding need not preserve every texture or be invertible; a generative representation must retain enough detail to reconstruct an image.

This module first follows the perception path—vision encoder, semantic training, connector, and text generation. Image output is developed separately in [[00 Учебник/16 Multimodal Models/64g Диффузионные и flow-модели изображений|diffusion and flow models]] and [[00 Учебник/16 Multimodal Models/64h Единая мультимодальная последовательность и Chameleon|discrete image codes and Chameleon]]. Those pages may currently use the explicitly labeled Russian fallback. A shared embedding space alone does not supply either generator.

## Images as sequences of patches

Write a color image as $X\in\mathbb R^{H\times W\times C}$. A pixel is too small to carry useful semantics on its own, while a single vector for the whole image discards location. Vision Transformer divides the image into $P\times P$ patches, flattens each patch, and maps it through a learned linear projection. The sequence length is

$$
N=\frac{H}{P}\frac{W}{P}.
$$

A $336\times336$ image with patch size 14 produces $24\times24=576$ patches. This count assumes dimensions divisible by the patch size after preprocessing. Position embeddings are added before the sequence enters a standard Transformer encoder; original ViT also prepends a classification token.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/vit-figure1-patch-sequence.png]]

*Figure 1 from Dosovitskiy et al., [An Image is Worth 16×16 Words](https://arxiv.org/abs/2010.11929), PDF p. 3. Patches are linearly projected and passed to an ordinary Transformer encoder.*

Self-attention makes every output patch contextual: its representation can contain information from the rest of the image. It cannot, however, recover information that preprocessing or a lossy projection has already removed. Patchification itself is a rearrangement and need not discard within-patch detail. If two characters have already merged into one blur, a later LLM can only guess them from context.

### Follow the tensor shapes

For ViT-Base/16, one RGB image has shape `[3,224,224]`. The $14\times14$ grid contains 196 patches, each with $16\cdot16\cdot3=768$ input numbers. A shared linear map produces width 768; adding `[CLS]` gives the encoder input `[197,768]`. The classification token is an extra position, not another image patch.

A smaller $32\times32$ RGB image makes the operation easier to trace. Four $16\times16$ patches become four vectors. For a row-major flattening,

$$
p_{r,c}=\operatorname{vec}(X_{r:r+16,c:c+16,:})W+b,
\qquad W\in\mathbb R^{768\times d}.
$$

The output is $[4,d]$: upper left, upper right, lower left, lower right. No vocabulary lookup assigns a discrete ID to a patch. A visual token here is simply one sequence position containing a continuous vector.

A convolution with `kernel_size=stride=16` and $d$ output channels performs this same projection. Each of its $d$ filters computes one coordinate from one non-overlapping patch. The operation is linear like a patch projection; global mixing happens later in self-attention.

## Resolution is a compute decision

Patch count grows with area. Doubling both sides creates four times as many tokens; full attention inside the vision encoder grows faster still. Large patches are often sufficient for scene classification, but small print, chart marks, and interface controls require much finer sampling. This is why modern VLMs use native resolution, tiling, and token compression rather than treating input size as an incidental preprocessing choice.

With 576 patches, full vision self-attention has $576^2=331\,776$ patch-pair scores per head. Doubling each side produces 2,304 patches and $2\,304^2=5\,308\,416$ pairs: sixteen times as many. These are illustrative pair counts excluding special tokens, not measured runtime or an assertion that every VLM uses global attention.

## CLIP replaces a fixed classifier with a text encoder

A conventional image classifier predicts one of a fixed set of labels. CLIP instead trains an image encoder and a text encoder to produce normalized vectors in one space. For a batch of $B$ matched pairs, it computes

$$
S_{ij}=\frac{v_i^\top t_j}{\tau}.
$$

and applies symmetric cross-entropy over image-to-text and text-to-image directions. Correct pairs lie on the diagonal; all other pairs in the batch act as negatives.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/clip-figure1-training-and-zero-shot.png]]

*Figure 1 from Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), PDF p. 2. The left half shows contrastive batch training; the right half shows text-derived zero-shot classifier weights.*

After training, prompts such as `a photo of a dog` are encoded into class vectors. A new image is assigned to the nearest class vector without fitting a classifier on the target dataset. Prompt wording matters because bare labels can be ambiguous and differ from natural captions. CLIP therefore ensembles several prompt templates.

The same independent encoders support bidirectional retrieval. Image and text vectors can be computed and cached separately, which is valuable at collection scale.

### Two directions of the same loss

The image-to-text half normalizes each row; the text-to-image half normalizes each column:

$$
\mathcal L_{\mathrm{CLIP}}
=-\frac{1}{2B}\left[
\sum_i\log\frac{e^{S_{ii}}}{\sum_j e^{S_{ij}}}
+\sum_j\log\frac{e^{S_{jj}}}{\sum_i e^{S_{ij}}}
\right].
$$

For three matched pairs, consider

$$
S=\begin{pmatrix}
8.1&2.0&-0.4\\
1.7&7.3&0.9\\
-0.2&1.1&6.8
\end{pmatrix}.
$$

Row one compares captions for image one; column one compares images for caption one. Raising $S_{12}$ close to $S_{11}$ makes the first image's choice uncertain even if the diagonal remains largest. Batch composition therefore defines which competing descriptions the model must distinguish, not just how many examples contribute to a gradient.

## What the contrastive objective preserves

The objective asks for matched pairs to be close; it does not require the final global vector to preserve every spatial relation. A model may recognize a horse and grass while confusing “a horse eats grass” with “grass eats a horse.” Strong retrieval is therefore not evidence of grounded relation understanding, OCR, counting, or text generation.

Generative VLMs usually take patch-level features before final pooling. Those retain more local structure than a global CLIP embedding. LLaVA, for example, projects a grid of CLIP ViT features into the language model’s embedding dimension.

## SigLIP: pairwise decisions instead of a shared softmax

[SigLIP](https://arxiv.org/abs/2303.15343) changes how pair similarities become a loss. Let $y_{ij}=+1$ for matched image–text pairs and $-1$ otherwise. With learned scale $a$ and bias $b$, the paper uses

$$
\mathcal L_{\mathrm{SigLIP}}
=-\frac1B\sum_{i=1}^{B}\sum_{j=1}^{B}
\log\sigma\!\left(y_{ij}(a\,v_i^\top t_j+b)\right).
$$

This is normalization by $B$, as in the paper, not the mean over $B^2$ binary decisions. Changing that normalization changes gradient scale as batch size changes. The bias helps accommodate $B$ positive pairs versus $B(B-1)$ negatives.

The missing softmax denominator does not mean missing negatives. Each off-diagonal pair still contributes a negative binary loss, but pair blocks can be evaluated without a shared row denominator. Distributed implementations still communicate whenever remote features are required.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/multimodal/siglip-parallelism.png]]

*Stanford CS336, Lecture 17, illustration of distributed pair blocks based on Zhai et al., [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343). Follow one local block; independent pair losses simplify accumulation, not eliminate all communication.*

This changes the optimization and communication problem, not the meaning of a visual token. Compare losses with the encoder, data, hardware, and training budget controlled; training-day figures on different accelerators are not an isolated speedup.

## Dual encoders and generative VLMs are different systems

Dual encoders produce a shared representation and excel at search, ranking, and zero-shot classification. A generative VLM conditions an autoregressive decoder on visual information. It needs an interface between the vision encoder and the LLM: a projector, Q-Former, Perceiver Resampler, or cross-attention.

The distinction prevents misleading comparisons. Better contrastive retrieval may not improve document OCR. A capable dialogue model may be inefficient for billion-image search. Increasing visual-token count may improve small-text accuracy while making time to first token much worse.

| System | Output | Interaction | Main limitation |
|---|---|---|---|
| Dual encoder | independent image and text vectors | scalar similarity | global pooling can hide fine structure |
| Cross-encoder | pair score or task output | joint image–text processing | pair-specific interaction cannot be cached as two independent final vectors |
| Generative VLM | next-token probabilities | visual memory conditions a decoder | context cost and possible unsupported generation |

## Diagnostic tests

Evaluation should alter the evidence. Occluding the answer region should change the prediction. Swapping object relations should change compositional answers. Reducing font size should produce a measurable degradation curve. For CLIP-like models, retrieval, zero-shot classification, linear probing, and distribution-shift robustness answer different questions and should be reported separately.

## Sources and next chapter

- Stanford CS231n, [Multimodal Foundation Models, Lecture 16](https://cs231n.stanford.edu/slides/2025/lecture_16.pdf), PDF pp. 13–55.
- Dosovitskiy et al., [An Image is Worth 16×16 Words](https://arxiv.org/abs/2010.11929).
- Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020).
- Next: [[en/00 Textbook/16 Multimodal Models/64a Connectors and fusion|64.1. Connectors and fusion]].
