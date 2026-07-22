---
title: "64.4. Documents, OCR, and visual grounding"
type: textbook-chapter
status: canonical
locale: en
translation_of: "00 Учебник/16 Multimodal Models/64d Документы, OCR и visual grounding.md"
last_updated: 2026-07-23
---

# Documents, OCR, and visual grounding

An approximate photo caption may still be useful; one wrong digit can invalidate an invoice or contract. Document understanding therefore combines character recognition, reading order, layout semantics, and reasoning. These levels require separate evaluation: a correct answer may come from a template prior, while exact transcription may still lose table structure.

High resolution is essential because characters disappear during downsampling. Qwen2-VL’s dynamic resolution and Qwen2.5-VL’s document-oriented training treat dense OCR as an explicit capability.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/qwen2vl-figure11-dense-ocr.png]]

*Excerpt from Figure 11 in Wang et al., [Qwen2-VL](https://arxiv.org/abs/2409.12191), PDF p. 29. It illustrates a long OCR response over a dense page, not a general accuracy guarantee.*

## Cascaded and end-to-end systems

A cascade runs specialist OCR first and gives text plus coordinates to an LLM. It is auditable and modular, but segmentation and reading-order errors become fixed upstream. An end-to-end VLM can combine text with color, arrows, and layout, yet its fluent answer is harder to verify and may fill missing characters from language priors.

For critical documents, a hybrid is often preferable: OCR provides complete text and provenance; the VLM reasons over both image and OCR; every critical field links back to a region.

## Tables, charts, and grounding

Tables are two-dimensional relations, not linear strings. A useful output preserves row and column headers, merged cells, and hierarchy in JSON or HTML. Charts add axes, units, legends, scales, and marks; identifying a trend is easier than extracting the value at a particular coordinate.

Visual grounding connects a phrase to a point, box, or mask. Pointing can expose evidence and enable grounded counting: the model indicates every object it counted. Structured JSON guarantees syntax but not geometric correctness.

Coordinates must be transformed back through crop origin, overlap, padding, and resize. If a tile coordinate is $(u,v)$, crop origin $(o_x,o_y)$, padding $(p_x,p_y)$, and scale $s$, then

$$x=(u+o_x-p_x)/s,\qquad y=(v+o_y-p_y)/s.$$

The exact metadata must travel with the tensor.

## Data, metrics, and interventions

Grounding data comes from detection datasets, referring expressions, document layouts, GUI screenshots, and human points. Negative examples teach the model to say that an object is absent. Ambiguous scenes require relational descriptions such as “the second cup from the left.”

OCR uses character/word error and exact match for critical fields; extraction uses field-level F1; boxes use IoU; points use containment or normalized distance. Report slices by font size, blur, rotation, language, aspect ratio, object size, and number of distractors.

Occluding the answer region, swapping table rows, or replacing a number with a synthetic value tests whether the output follows visual evidence rather than a familiar document template.

## Sources and next chapter

- [Qwen2-VL](https://arxiv.org/abs/2409.12191), [Qwen2.5-VL](https://arxiv.org/abs/2502.13923), and [Molmo and PixMo](https://arxiv.org/abs/2409.17146).
- Previous: [[en/00 Textbook/16 Multimodal Models/64c Training VLMs — alignment, instruction tuning, and data|64.3]]. Next: [[en/00 Textbook/16 Multimodal Models/64e Video, audio, and omni models|64.5]].
