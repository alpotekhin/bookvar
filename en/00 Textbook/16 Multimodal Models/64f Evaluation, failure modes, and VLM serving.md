---
title: "64.6. Evaluation, failure modes, and VLM serving"
type: textbook-chapter
status: canonical
locale: en
translation_of: "00 Учебник/16 Multimodal Models/64f Оценивание, отказы и serving VLM.md"
last_updated: 2026-09-15
last_verified: 2026-09-15
primary_sources:
  - https://cs231n.stanford.edu/slides/2025/lecture_16.pdf
  - https://arxiv.org/abs/2204.03162
  - https://arxiv.org/abs/2305.10355
  - https://arxiv.org/abs/2409.17146
  - https://arxiv.org/abs/2502.13923
---

# Evaluation, failure modes, and VLM serving

One aggregate multimodal score mixes recognition, OCR, knowledge, reasoning, grounding, and response format. A useful evaluation separates those abilities and then tests whether the answer causally depends on visual evidence.

MMMU covers expert disciplines; MathVista visual mathematics; OCRBench, DocVQA, and ChartQA text and layout; Video-MME and LongVideoBench temporal understanding. None is a complete measure of “multimodal intelligence.”

| Ability | Example test | Useful measurement | Not established by success |
|---|---|---|---|
| Objects and attributes | VQAv2, perception tasks | accuracy, F1 | relation understanding or OCR |
| Composition | Winoground | text/image/group scores | unrestricted scene descriptions |
| Documents | OCRBench, DocVQA, ChartQA | exact match, ANLS, CER/WER | resistance to template shortcuts |
| Grounding | referring expressions, points | IoU, point accuracy | completeness of an answer |
| Visual mathematics | MathVista | answer accuracy | reliable reading of any document |
| Expert reasoning | MMMU | accuracy by discipline | new-domain reliability |
| Video | Video-MME, LongVideoBench | accuracy, localization metrics where annotated | audio–video synchronization |
| Object hallucination | POPE | precision, recall, F1, yes ratio | attribute, count, or OCR correctness |

Metrics apply to particular tasks and annotations; not every listed video benchmark uses temporal IoU. The table is an evaluation map, not a ranking.

## Composition and hallucination

Winoground uses image-caption pairs with the same words and entities but different relations. It tests whether the representation preserves composition rather than a bag of concepts.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/cs231n-lecture16-winoground.png]]

*Stanford CS231n 2025, [Lecture 16](https://cs231n.stanford.edu/slides/2025/lecture_16.pdf), PDF p. 48. The slide introduces Winoground; the primary source is cited on the slide.*

For a Winoground item, let $I_0,I_1$ be images and $T_0,T_1$ their matching captions. Four similarities expose two different directions:

$$
\text{text:}\quad
s(I_0,T_0)>s(I_0,T_1),\quad
s(I_1,T_1)>s(I_1,T_0),
$$

$$
\text{image:}\quad
s(I_0,T_0)>s(I_1,T_0),\quad
s(I_1,T_1)>s(I_0,T_1).
$$

Group success requires all four inequalities. For the illustrative matrix
$S=\begin{pmatrix}0.9&0.8\\0.7&0.6\end{pmatrix}$, the first image chooses its caption correctly but the second does not. Two relatively high diagonal scores do not imply group success. See [Thrush et al., Winoground](https://arxiv.org/abs/2204.03162).

POPE probes object hallucination with absent categories sampled randomly, by frequency, or co-occurrence. It measures one failure mode, not all hallucination: attributes, relations, counts, and OCR can also be false. Newer models can saturate simple negatives.

For 100 binary questions, suppose $TP=40$, $FP=20$, $FN=10$, and $TN=30$. Then precision is $40/60\approx0.667$, recall $40/50=0.8$, F1 $80/110\approx0.727$, accuracy 70%, and the yes ratio 60%. Reporting recall alone hides the false-positive problem. Always include the negative sampling strategy; frequent and co-occurring absent categories test different biases.

Occlusion and counterfactual replacement are stronger tests. Removing the answer region should lower confidence; replacing the image with a matched counterexample should change the factual answer.

## Judges, contamination, and openness

An LLM judge can prefer detail and style while missing a small visual error. Its prompt, version, and access to the image are part of the protocol. Use deterministic metrics for exact text, boxes, numbers, and actions, and audit open-ended scores with humans.

Contamination includes perceptual duplicates, OCR text embedded in images, and published solutions. Post-cutoff or parameterized tests help but trade reproducibility for secrecy.

Open weights do not imply an open training pipeline. Molmo separates the final VLM, LLM backbone, vision encoder, and data/code.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/molmo-figure11-openness.png]]

*Figure 11 from Deitke et al., [Molmo and PixMo](https://arxiv.org/abs/2409.17146), PDF p. 19. The table is a dated comparison of openness across model components and data.*

## Serving path and observability

For a two-page comparison request, limiting the answer to 200 tokens does not limit image cost. Before model execution, the service must validate the two media inputs, decode them, estimate their resized pixel and visual-token counts, and apply an admission policy.

A VLM request passes through media loading, validation, resize/cropping, vision encoding, connector/resampler, LLM prefill, and autoregressive decode. Time to first token includes every preceding stage. Logging only the LLM engine hides media and vision latency.

Admission control should cap pixels, images, frames, and total visual tokens. Variable-length media causes batch imbalance and head-of-line blocking. Image-embedding cache keys must include content hash, preprocessing configuration, encoder, and connector version; video keys also include sampling policy.

### Follow the complete latency clock

For a sequential illustrative path,

$$
T_{\mathrm{TTFT}}=
T_{\mathrm{queue}}+T_{\mathrm{fetch/decode}}
+T_{\mathrm{preprocess}}+T_{\mathrm{vision}}
+T_{\mathrm{connector}}+T_{\mathrm{LLM\ prefill}}
+T_{\mathrm{sample/deliver},1}.
$$

If the seven intervals are 20, 80, 35, 120, 10, 180, and 15 ms, TTFT is 460 ms. Logging only the last two yields 195 ms, missing more than half of user-visible delay. The final term is sampling and first-token delivery after prefill; it need not be another complete decoder pass. With overlapping stages, use the critical path and timestamps rather than adding overlapping durations.

### Avoid head-of-line blocking before the LLM

Ten short photo requests and one 200-page document should not be treated as eleven equal jobs. The document can occupy media decoding, vision execution, and prefill long enough to delay all the photos. Continuous batching inside the LLM cannot fix an upstream media queue on its own.

Estimate cost before admission, cap media counts and pixels, and group compatible workloads by image/document/video or expected cost. Define separate SLOs for these classes. Where supported, packed vision attention prevents images from attending to one another while avoiding padding every image to the largest dimensions.

### Cache identity depends on the cached object

An encoder-feature cache needs the image bytes, preprocessing, and vision-checkpoint identity. A cache of **projected** visual embeddings additionally depends on connector weights. A full LLM KV cache also depends on text prefix, LLM revision, adapters, positions, and multimodal ordering. Identical visible text does not justify reusing KV from another image.

For video, retain selected frame timestamps and sampling configuration in identity. A preprocessing change is a representation change even when the media file is unchanged. Logs should distinguish encoder-cache hits from LLM prefix-cache hits.

System metrics include media preprocessing, vision latency, LLM prefill, TTFT, inter-token latency, peak memory, and throughput per pixel/frame/token. Workload tests need short photos, large documents, multi-image prompts, and video—not one average request.

Images can contain prompt injection, tiny hidden text, QR codes, and personal data. OCR output must not become a trusted system instruction. Media parsers require format and size limits; remote URLs require network isolation; visual logs need explicit retention policy.

An evaluator should retain the judge prompt, checkpoint/version, media access, response ordering, and rubric. Deterministic subtasks and blinded human audits provide checks against a judge's preference for verbose answers. A model's own confidence after occlusion need not be calibrated; assess whether the answer or abstention follows changed evidence, not only whether it expresses uncertainty.

Reliable interfaces expose evidence: crop and coordinates for documents, timestamp for video, segment for audio. Inspectable provenance is more valuable than a fluent unsupported explanation.

Quality and efficiency should be inspected together on photos, dense documents, multiple-image prompts, and videos. The useful result is a quality–budget–latency curve with failure slices, not one average score or a gallery of successful responses.

The [[06 Практика/19 Найти автомобильный номер в видеопотоке|video license-plate practical]] connects detection, tracking, OCR, and direct VLM calls to measurable frame-rate and latency requirements. The next conceptual branch, [[00 Учебник/16 Multimodal Models/64g Диффузионные и flow-модели изображений|diffusion and flow matching]], changes the output from text to generated images; these linked pages can use the marked Russian fallback until separately localized.

## Sources

- Stanford CS231n, [Multimodal Foundation Models](https://cs231n.stanford.edu/slides/2025/lecture_16.pdf).
- Li et al., [POPE](https://arxiv.org/abs/2305.10355).
- Deitke et al., [Molmo and PixMo](https://arxiv.org/abs/2409.17146).
- Previous: [[en/00 Textbook/16 Multimodal Models/64e Video, audio, and omni models|64.5]].
