---
title: "64.3. Training VLMs: alignment, instruction tuning, and data"
type: textbook-chapter
status: canonical
locale: en
translation_of: "00 Учебник/16 Multimodal Models/64c Обучение VLM — alignment, instruction tuning и данные.md"
last_updated: 2026-07-23
---

# Training VLMs: alignment, instruction tuning, and data

Architecture defines the signal path, but stages and data define behavior. Short captions teach object semantics; dense descriptions teach relations; coordinate targets teach grounding; conversations teach response format. A recipe should therefore separate representation learning, modality alignment, instruction tuning, and preference/post-training.

Alignment does not always mean equal embeddings. LLaVA trains a projector with next-token prediction on image-caption pairs. BLIP-2 combines contrastive, matching, and grounded-generation objectives for Q-Former. The relevant questions are the objective, data, and trainable parameters—not the stage name.

## Frozen and trainable components

Connector-only training is cheap but cannot repair a vision encoder that discarded small text. Unfreezing vision adapts perception. Unfreezing the LLM supports deeper multimodal interaction and in-context learning but risks language regression. Text-only examples and smaller backbone learning rates help preserve prior capabilities.

InternVL makes the schedule explicit: warm up the MLP with frozen backbones, optionally adapt the ViT, then perform full-model instruction tuning.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/internvl25-figure4-training-pipeline.png]]

*Figure 4 from Wang et al., [InternVL 2.5](https://arxiv.org/abs/2412.05271), PDF p. 6. Snowflakes and flames show frozen and trainable components at each stage.*

## Data creates different visual behaviors

Web captions name salient objects but omit detail. Dense captions cover background, attributes, and relations. OCR and document data require literal accuracy and layout. Grounding pairs language with points or boxes. Interleaved documents teach correspondence across several images and text spans.

Molmo’s PixMo collection connects separate human-collected datasets to dense captioning, question answering, pointing, and grounded counting.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/molmo-figure1-data-and-capabilities.png]]

*Figure 1 from Deitke et al., [Molmo and PixMo](https://arxiv.org/abs/2409.17146), PDF p. 2. Different data collections support different model capabilities.*

Synthetic data scales strict formats but inherits teacher errors. Provenance matters: a confident wrong OCR label from a proprietary teacher can be distilled into an open-weight student.

## Instruction and preference tuning

LLaVA used GPT-4 over captions and symbolic descriptions to produce conversations, detailed descriptions, and reasoning prompts. GPT-4 did not inspect the original image, so generated instructions could contain only information already present in those inputs.

Instruction loss is normally applied to assistant tokens, with image and user tokens acting as conditions. Packing and masks must prevent future answers from leaking across examples. Multimodal preference pairs should differ in visual correctness, not merely style; otherwise preference optimization rewards fluent language without better grounding.

Verifiable rewards are useful for points, boxes, OCR, and GUI actions. Open descriptions still require human audits because an automatic judge may reward verbosity while missing a visual error.

## Mixtures and reproducibility

Simple concatenation lets the largest web corpus overwhelm rare grounding or document examples. Sampling weights should consider examples, tokens, pixels, and frames. A complete recipe reports data provenance and filtering; component freeze states; resolution and token budget; objectives and masks; mixture weights; optimizer, per-component learning rates, curriculum, and evaluation checkpoints.

## Sources and next chapter

- [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485), [BLIP-2](https://arxiv.org/abs/2301.12597), [InternVL 2.5](https://arxiv.org/abs/2412.05271), and [Molmo and PixMo](https://arxiv.org/abs/2409.17146).
- Previous: [[en/00 Textbook/16 Multimodal Models/64b Resolution, tiling, and spatial positions|64.2]]. Next: [[en/00 Textbook/16 Multimodal Models/64d Documents, OCR, and visual grounding|64.4]].
