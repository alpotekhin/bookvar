---
title: "64.6. Evaluation, failure modes, and VLM serving"
type: textbook-chapter
status: canonical
locale: en
translation_of: "00 Учебник/16 Multimodal Models/64f Оценивание, отказы и serving VLM.md"
last_updated: 2026-07-23
---

# Evaluation, failure modes, and VLM serving

One aggregate multimodal score mixes recognition, OCR, knowledge, reasoning, grounding, and response format. A useful evaluation separates those abilities and then tests whether the answer causally depends on visual evidence.

MMMU covers expert disciplines; MathVista visual mathematics; OCRBench, DocVQA, and ChartQA text and layout; Video-MME and LongVideoBench temporal understanding. None is a complete measure of “multimodal intelligence.”

## Composition and hallucination

Winoground uses image-caption pairs with the same words and entities but different relations. It tests whether the representation preserves composition rather than a bag of concepts.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/cs231n-lecture16-winoground.png]]

*Stanford CS231n 2025, [Lecture 16](https://cs231n.stanford.edu/slides/2025/lecture_16.pdf), PDF p. 48. The slide introduces Winoground; the primary source is cited on the slide.*

POPE probes object hallucination with absent categories sampled randomly, by frequency, or co-occurrence. It measures one failure mode, not all hallucination: attributes, relations, counts, and OCR can also be false. Newer models can saturate simple negatives.

Occlusion and counterfactual replacement are stronger tests. Removing the answer region should lower confidence; replacing the image with a matched counterexample should change the factual answer.

## Judges, contamination, and openness

An LLM judge can prefer detail and style while missing a small visual error. Its prompt, version, and access to the image are part of the protocol. Use deterministic metrics for exact text, boxes, numbers, and actions, and audit open-ended scores with humans.

Contamination includes perceptual duplicates, OCR text embedded in images, and published solutions. Post-cutoff or parameterized tests help but trade reproducibility for secrecy.

Open weights do not imply an open training pipeline. Molmo separates the final VLM, LLM backbone, vision encoder, and data/code.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/molmo-figure11-openness.png]]

*Figure 11 from Deitke et al., [Molmo and PixMo](https://arxiv.org/abs/2409.17146), PDF p. 19. The table is a dated comparison of openness across model components and data.*

## Serving path and observability

A VLM request passes through media loading, validation, resize/cropping, vision encoding, connector/resampler, LLM prefill, and autoregressive decode. Time to first token includes every preceding stage. Logging only the LLM engine hides media and vision latency.

Admission control should cap pixels, images, frames, and total visual tokens. Variable-length media causes batch imbalance and head-of-line blocking. Image-embedding cache keys must include content hash, preprocessing configuration, encoder, and connector version; video keys also include sampling policy.

System metrics include media preprocessing, vision latency, LLM prefill, TTFT, inter-token latency, peak memory, and throughput per pixel/frame/token. Workload tests need short photos, large documents, multi-image prompts, and video—not one average request.

Images can contain prompt injection, tiny hidden text, QR codes, and personal data. OCR output must not become a trusted system instruction. Media parsers require format and size limits; remote URLs require network isolation; visual logs need explicit retention policy.

Reliable interfaces expose evidence: crop and coordinates for documents, timestamp for video, segment for audio. Inspectable provenance is more valuable than a fluent unsupported explanation.

## Sources

- Stanford CS231n, [Multimodal Foundation Models](https://cs231n.stanford.edu/slides/2025/lecture_16.pdf).
- Li et al., [POPE](https://arxiv.org/abs/2305.10355).
- Deitke et al., [Molmo and PixMo](https://arxiv.org/abs/2409.17146).
- Previous: [[en/00 Textbook/16 Multimodal Models/64e Video, audio, and omni models|64.5]].
