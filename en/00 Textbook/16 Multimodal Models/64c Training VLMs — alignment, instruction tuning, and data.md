---
title: "64.3. Training VLMs: alignment, instruction tuning, and data"
type: textbook-chapter
status: canonical
locale: en
translation_of: "00 Учебник/16 Multimodal Models/64c Обучение VLM — alignment, instruction tuning и данные.md"
last_updated: 2026-09-15
last_verified: 2026-09-15
primary_sources:
  - https://arxiv.org/abs/2103.00020
  - https://arxiv.org/abs/2304.08485
  - https://arxiv.org/abs/2408.03326
  - https://arxiv.org/abs/2301.12597
  - https://arxiv.org/abs/2308.12966
  - https://arxiv.org/abs/2511.21631
  - https://arxiv.org/abs/2412.05271
  - https://arxiv.org/abs/2409.17146
  - https://cs336.stanford.edu/
---

# Training VLMs: alignment, instruction tuning, and data

Architecture defines the signal path, but stages and data define behavior. Short captions teach object semantics; dense descriptions teach relations; coordinate targets teach grounding; conversations teach response format. A recipe should therefore separate representation learning, modality alignment, instruction tuning, and preference/post-training.

Alignment does not always mean equal embeddings. LLaVA trains a projector with next-token prediction on image-caption pairs. BLIP-2 combines contrastive, matching, and grounded-generation objectives for Q-Former. The relevant questions are the objective, data, and trainable parameters—not the stage name.

## What changes at each stage?

| Stage | Prediction | Typical data | Parameters that move |
|---|---|---|---|
| Vision pre-training | class, masked content, or paired text | images and image–text pairs | vision encoder |
| Connector alignment | caption or pair relationship | image–caption pairs | connector/Q-Former, according to recipe |
| Visual instruction tuning | answer conditioned on media and a question | VQA, OCR, documents, grounding, dialogue | connector and selected backbones |
| Preference or reward training | preferred response or verifiable result | response pairs, points, boxes, exact answers | policy; reward model only where the method requires one |

This table describes roles, not four mandatory stages. A BLIP-2 model and a LLaVA model can both say “alignment” while using different objectives and gradient paths.

Let $Z_v$ be frozen vision features and $H_v=g_\phi(Z_v)$ the connector output. Caption alignment can use

$$
\mathcal L_{\mathrm{caption}}
=-\sum_{t=1}^{T}\log p_\theta(y_t\mid H_v,y_{<t}).
$$

The LLM parameters $\theta$ stay fixed while $\phi$ changes. Backpropagation still passes through the LLM to $H_v$: otherwise the projector receives no learning signal. Dimension matching is necessary, but the real task is finding continuous inputs that the pretrained language model can use.

## Frozen and trainable components

Connector-only training is cheap but cannot repair a vision encoder that discarded small text. Unfreezing vision adapts perception. Unfreezing the LLM supports deeper multimodal interaction and in-context learning but risks language regression. Text-only examples and smaller backbone learning rates help preserve prior capabilities.

InternVL makes the schedule explicit: warm up the MLP with frozen backbones, optionally adapt the ViT, then perform full-model instruction tuning.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/internvl25-figure4-training-pipeline.png]]

*Figure 4 from Wang et al., [InternVL 2.5](https://arxiv.org/abs/2412.05271), PDF p. 6. Snowflakes and flames show frozen and trainable components at each stage.*

## Loss masks are not attention masks

Suppose the user asks an invoice question and the assistant answers `12 480 RUB`. Token boundaries below are illustrative, not a tokenizer claim.

| Segment | system instruction | image vectors | user question | assistant answer | EOS |
|---|---|---|---|---|---|
| Available as causal context | yes | yes | yes | previous answer tokens | as configured |
| Target loss | no | no | no | yes | yes |

The response loss is

$$
\mathcal L=-\sum_t m_t\log p(y_t\mid y_{<t},I),
\qquad
m_t=\mathbf1[\text{assistant target at }t].
$$

A zero loss mask does not hide an input from attention. Image and question positions must remain usable as conditions. In a multi-turn conversation, several assistant responses can contribute loss, but no response may see its future tokens. When packing independent dialogues, separate their attention regions so an earlier example does not silently become another example's context.

## Data creates different visual behaviors

Web captions name salient objects but omit detail. Dense captions cover background, attributes, and relations. OCR and document data require literal accuracy and layout. Grounding pairs language with points or boxes. Interleaved documents teach correspondence across several images and text spans.

Molmo’s PixMo collection connects separate human-collected datasets to dense captioning, question answering, pointing, and grounded counting.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/molmo-figure1-data-and-capabilities.png]]

*Figure 1 from Deitke et al., [Molmo and PixMo](https://arxiv.org/abs/2409.17146), PDF p. 2. Different data collections support different model capabilities.*

PixMo's dense descriptions were collected by asking people to speak about an image and transcribing that speech. Point annotations provide a different supervision channel: the output location can be checked, and grounded counting can be audited by inspecting the individual points. The annotation interface therefore shapes what the model learns, not only how quickly examples are collected.

Synthetic data scales strict formats but inherits teacher errors. Provenance matters: a confident wrong OCR label from a proprietary teacher can be distilled into an open-weight student.

## Instruction and preference tuning

LLaVA used GPT-4 over captions and symbolic descriptions to produce conversations, detailed descriptions, and reasoning prompts. GPT-4 did not inspect the original image, so generated instructions could contain only information already present in those inputs.

Instruction loss is normally applied to assistant tokens, with image and user tokens acting as conditions. Packing and masks must prevent future answers from leaking across examples. Multimodal preference pairs should differ in visual correctness, not merely style; otherwise preference optimization rewards fluent language without better grounding.

Verifiable rewards are useful for points, boxes, OCR, and GUI actions. Open descriptions still require human audits because an automatic judge may reward verbosity while missing a visual error.

## Sampling mixtures changes both learning and cost

For dataset sizes $n_1,\ldots,n_K$, a common mixture is

$$
p_k=\frac{n_k^\alpha}{\sum_j n_j^\alpha}.
$$

At $\alpha=1$, examples are sampled proportionally to dataset size; at $\alpha=0$, datasets receive equal probability. With 10 million captions, 1 million document questions, and 100,000 grounding examples, proportional sampling gives about 90.1%, 9.0%, and 0.9%. At $\alpha=0.5$, the shares become 70.6%, 22.3%, and 7.1%. Grounding is now encountered almost eight times as frequently, without making the three datasets equally weighted.

Equal example shares still do not imply equal compute. A short caption may use a hundred visual vectors; a video example may require tens of thousands. Report mixture fractions by examples, supervised text tokens, visual tokens, pixels, and frames as appropriate.

## Concrete curricula: LLaVA, OneVision, and Qwen

Original LLaVA first trains the projector with both backbones frozen. It then trains the projector and LLM on visual instructions while retaining the frozen vision encoder. Its synthetic instruction pipeline supplies captions and box descriptions to GPT-4, not image pixels:

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/multimodal/llava-gen.png]]

*Stanford CS336, Lecture 17, based on Liu et al., [Visual Instruction Tuning, Figure 2](https://arxiv.org/abs/2304.08485). The teacher sees symbolic descriptions; this bounds the visual evidence available when it writes a question and answer.*

OneVision extends training from an initial connector stage through high-quality single-image learning to mixtures including multiple images and video. The visual budget and task distribution change along with the trainable components.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/multimodal/llava-onevision-training.png]]

*Stanford CS336, Lecture 17, training illustration based on [LLaVA-OneVision](https://arxiv.org/abs/2408.03326). Read each stage as a combination of data, trainable modules, and supported input structure.*

The intended transfer is compositional: reading one chart can help compare two charts; OCR can help interpret several GUI states. This is an explanation to test, not proof that transfer always occurs.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/multimodal/llava-onevision-transfer-s1.png]]

*Source: Li et al., LLaVA-OneVision; Stanford CS336 Lecture 17, [original course figure 1](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/images/llava-onevision-transfer-s1.png), revision `8b59b507`.*

*Stanford CS336, Lecture 17, [OneVision transfer examples](https://arxiv.org/abs/2408.03326): chart understanding is combined with cross-image comparison.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/multimodal/llava-onevision-transfer-s2.png]]

*Source: Li et al., LLaVA-OneVision; Stanford CS336 Lecture 17, [original course figure 2](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/images/llava-onevision-transfer-s2.png), revision `8b59b507`.*

*Stanford CS336, Lecture 17, [OneVision](https://arxiv.org/abs/2408.03326): distinguish reading an interface from deciding which source image and object the requested action concerns.*

Test the explanation by removing the relevant training category while controlling total updates, visual budget, and backbone. Otherwise direct exposure to similar examples or a higher resolution can masquerade as transfer.

Original [Qwen-VL](https://arxiv.org/abs/2308.12966) uses another schedule. Its initial image–text stage updates vision and adapter while freezing the LLM. A subsequent multitask stage trains all components at higher resolution. During instruction tuning the vision encoder is frozen again, while the adapter and language model learn the dialogue format.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/multimodal/qwen-vl-stages.png]]

*Stanford CS336, Lecture 17, based on Bai et al., [Qwen-VL, Figure 2](https://arxiv.org/abs/2308.12966). Follow resolution, dataset, and freeze-state changes together.*

[Qwen3-VL](https://arxiv.org/abs/2511.21631) extends its pre-training curriculum through connector warm-up and joint stages at 8K, 32K, and 256K context. Its report also describes square-root-normalized per-token loss and separate thinking/non-thinking post-training. The length extension requires a corresponding data mixture; changing a context-limit flag is not equivalent training.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/multimodal/qwen3-vl-pretraining.png]]

*Stanford CS336, Lecture 17, pre-training schedule based on the [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631). The context stages expand the training distribution as well as tensor size.*

This is an external extension, not a fully specified algorithm in this chapter. The [Qwen3-VL introduction](https://arxiv.org/html/2511.21631v1#S1) names square-root-normalized per-token loss but does not give the exact denominator or masking rule. We therefore do not attribute an invented formula to the authors. Reproduction requires the implementation's handling of padding, visual positions, and example boundaries; the loss name alone is insufficient.

## Mixtures and reproducibility

Simple concatenation lets the largest web corpus overwhelm rare grounding or document examples. Sampling weights should consider examples, tokens, pixels, and frames. A complete recipe reports data provenance and filtering; component freeze states; resolution and token budget; objectives and masks; mixture weights; optimizer, per-component learning rates, curriculum, and evaluation checkpoints.

| Record | Required detail |
|---|---|
| Components | encoder, connector, LLM revisions and initialization |
| Gradient path | freeze state at every stage, checkpointing and precision |
| Input | resizing, tiling, frame rate, token budgets |
| Data | origin, licensing, deduplication, filters, synthetic teacher |
| Mixture | formula, caps, example and token shares |
| Objective | target masks, packing boundaries, loss weights |
| Optimization | learning rates by component, batch units, steps |
| Validation | text-only retention, modality-specific tests, checkpoints |

Preference data must also distinguish visual correctness from prose quality. A fluent negative answer naming an absent object is useful for a different reason than a poorly written answer: only the former directly tests whether the policy follows the image.

## Sources and next chapter

- [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485), [BLIP-2](https://arxiv.org/abs/2301.12597), [InternVL 2.5](https://arxiv.org/abs/2412.05271), and [Molmo and PixMo](https://arxiv.org/abs/2409.17146).
- Previous: [[en/00 Textbook/16 Multimodal Models/64b Resolution, tiling, and spatial positions|64.2]]. Next: [[en/00 Textbook/16 Multimodal Models/64d Documents, OCR, and visual grounding|64.4]].
