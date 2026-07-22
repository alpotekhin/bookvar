---
title: "64. From images to visual tokens: ViT and CLIP"
type: textbook-chapter
status: canonical
locale: en
translation_of: "00 Учебник/16 Multimodal Models/64 Мультимодальные модели.md"
last_updated: 2026-07-23
primary_sources:
  - https://arxiv.org/abs/2010.11929
  - https://arxiv.org/abs/2103.00020
  - https://cs231n.stanford.edu/slides/2025/lecture_16.pdf
---

# From images to visual tokens: ViT and CLIP

A language model receives a sequence of discrete token indices. An image is a rectangular grid of intensity values. Before a Transformer can process both, the image must become a sequence of vectors. This conversion determines spatial resolution, visual-token count, and which details remain available to every later stage.

## Images as sequences of patches

Write a color image as $X\in\mathbb R^{H\times W\times C}$. A pixel is too small to carry useful semantics on its own, while a single vector for the whole image discards location. Vision Transformer divides the image into $P\times P$ patches, flattens each patch, and maps it through a learned linear projection. The sequence length is

$$N=\frac{H}{P}\frac{W}{P}.$$

A $336\times336$ image with patch size 14 produces $24\times24=576$ patches. Position embeddings are added before the sequence enters a standard Transformer encoder.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/vit-figure1-patch-sequence.png]]

*Figure 1 from Dosovitskiy et al., [An Image is Worth 16×16 Words](https://arxiv.org/abs/2010.11929), PDF p. 3. Patches are linearly projected and passed to an ordinary Transformer encoder.*

Self-attention makes every output patch contextual: its representation can contain information from the rest of the image. It cannot, however, recover information removed by resizing or patchification. If two characters have already merged into one blur, a later LLM can only guess them from context.

## Resolution is a compute decision

Patch count grows with area. Doubling both sides creates four times as many tokens; full attention inside the vision encoder grows faster still. Large patches are often sufficient for scene classification, but small print, chart marks, and interface controls require much finer sampling. This is why modern VLMs use native resolution, tiling, and token compression rather than treating input size as an incidental preprocessing choice.

## CLIP replaces a fixed classifier with a text encoder

A conventional image classifier predicts one of a fixed set of labels. CLIP instead trains an image encoder and a text encoder to produce normalized vectors in one space. For a batch of $B$ matched pairs, it computes

$$S_{ij}=\frac{v_i^\top t_j}{\tau}$$

and applies symmetric cross-entropy over image-to-text and text-to-image directions. Correct pairs lie on the diagonal; all other pairs in the batch act as negatives.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/clip-figure1-training-and-zero-shot.png]]

*Figure 1 from Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), PDF p. 2. The left half shows contrastive batch training; the right half shows text-derived zero-shot classifier weights.*

After training, prompts such as `a photo of a dog` are encoded into class vectors. A new image is assigned to the nearest class vector without fitting a classifier on the target dataset. Prompt wording matters because bare labels can be ambiguous and differ from natural captions. CLIP therefore ensembles several prompt templates.

The same independent encoders support bidirectional retrieval. Image and text vectors can be computed and cached separately, which is valuable at collection scale.

## What the contrastive objective preserves

The objective asks for matched pairs to be close; it does not require the final global vector to preserve every spatial relation. A model may recognize a horse and grass while confusing “a horse eats grass” with “grass eats a horse.” Strong retrieval is therefore not evidence of grounded relation understanding, OCR, counting, or text generation.

Generative VLMs usually take patch-level features before final pooling. Those retain more local structure than a global CLIP embedding. LLaVA, for example, projects a grid of CLIP ViT features into the language model’s embedding dimension.

## Dual encoders and generative VLMs are different systems

Dual encoders produce a shared representation and excel at search, ranking, and zero-shot classification. A generative VLM conditions an autoregressive decoder on visual information. It needs an interface between the vision encoder and the LLM: a projector, Q-Former, Perceiver Resampler, or cross-attention.

The distinction prevents misleading comparisons. Better contrastive retrieval may not improve document OCR. A capable dialogue model may be inefficient for billion-image search. Increasing visual-token count may improve small-text accuracy while making time to first token much worse.

## Diagnostic tests

Evaluation should alter the evidence. Occluding the answer region should change the prediction. Swapping object relations should change compositional answers. Reducing font size should produce a measurable degradation curve. For CLIP-like models, retrieval, zero-shot classification, linear probing, and distribution-shift robustness answer different questions and should be reported separately.

## Sources and next chapter

- Stanford CS231n, [Multimodal Foundation Models, Lecture 16](https://cs231n.stanford.edu/slides/2025/lecture_16.pdf), PDF pp. 13–55.
- Dosovitskiy et al., [An Image is Worth 16×16 Words](https://arxiv.org/abs/2010.11929).
- Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020).
- Next: [[en/00 Textbook/16 Multimodal Models/64a Connectors and fusion|64.1. Connectors and fusion]].
