---
title: "64.4. Documents, OCR, and visual grounding"
type: textbook-chapter
status: canonical
locale: en
translation_of: "00 Учебник/16 Multimodal Models/64d Документы, OCR и visual grounding.md"
last_updated: 2026-09-15
last_verified: 2026-09-15
primary_sources:
  - https://arxiv.org/abs/2409.12191
  - https://arxiv.org/abs/2502.13923
  - https://arxiv.org/abs/2210.03347
  - https://arxiv.org/abs/2111.15664
  - https://arxiv.org/abs/2204.08387
  - https://arxiv.org/abs/2409.17146
---

# Documents, OCR, and visual grounding

An approximate photo caption may still be useful; one wrong digit can invalidate an invoice or contract. Document understanding therefore combines character recognition, reading order, layout semantics, and reasoning. These levels require separate evaluation: a correct answer may come from a template prior, while exact transcription may still lose table structure.

Consider an invoice question: “What is the total, and where is it shown?” A useful answer gives the value, page, and region—not only a plausible amount. The system must distinguish the total from a subtotal, preserve the label–value relationship, and show evidence that can be inspected independently.

High resolution is essential because characters disappear during downsampling. Qwen2-VL’s dynamic resolution and Qwen2.5-VL’s document-oriented training treat dense OCR as an explicit capability.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/qwen2vl-figure11-dense-ocr.png]]

*Excerpt from Figure 11 in Wang et al., [Qwen2-VL](https://arxiv.org/abs/2409.12191), PDF p. 29. It illustrates a long OCR response over a dense page, not a general accuracy guarantee.*

## Cascaded and end-to-end systems

A cascade runs specialist OCR first and gives text plus coordinates to an LLM. It is auditable and modular, but segmentation and reading-order errors become fixed upstream. An end-to-end VLM can combine text with color, arrows, and layout, yet its fluent answer is harder to verify and may fill missing characters from language priors.

For critical documents, a hybrid is often preferable: OCR provides complete text and provenance; the VLM reasons over both image and OCR; every critical field links back to a region.

### Two specialist alternatives

A layout-aware model makes geometry explicit. [LayoutLMv3](https://arxiv.org/abs/2204.08387) combines OCR text, two-dimensional positions, and image patches. An OCR-free model instead predicts structure directly from pixels: [Donut](https://arxiv.org/abs/2111.15664) avoids a separate OCR stage, while [Pix2Struct](https://arxiv.org/abs/2210.03347) learns screenshot-to-structure conversion through simplified HTML targets.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/pix2struct-figure1-tasks.png]]

*Lee et al., [Pix2Struct, Figure 1](https://arxiv.org/abs/2210.03347). Screenshot parsing supplies a pixel-to-structure objective that transfers to diagrams, interfaces, and documents.*

| Approach | Reasoning input | Audit advantage | Main failure boundary |
|---|---|---|---|
| OCR-first | text and coordinates | text can be checked separately | upstream segmentation and reading order |
| Layout-aware | text, positions, patches | spatial relationships are explicit | still depends on OCR |
| OCR-free specialist | pixels | no fixed OCR boundary | character and structural errors can be intertwined |
| General VLM | media and instruction | broad interface | language priors can conceal perception errors |

### Trace the invoice request

First rasterize the document page at a known resolution, retaining the transform between PDF coordinates and image pixels. Tile the image if small print would otherwise disappear; keep scale, padding, and crop origins. Locate candidates such as “Subtotal: 11,000 RUB” and “Total: 12,480 RUB.” Use layout and labels to select the requested field, not merely the closest number.

An illustrative result is:

```json
{
  "value": "12 480 RUB",
  "page": 1,
  "box": [0.68, 0.82, 0.91, 0.87],
  "coordinate_system": "normalized_original_page"
}
```

The backend can render the box, compare transcription with OCR, and recompute line-item arithmetic if that was required. A correct sum does not prove that the correct lines were read; exact transcription does not prove that the number belongs to the requested header.

## Tables, charts, and grounding

Tables are two-dimensional relations, not linear strings. A useful output preserves row and column headers, merged cells, and hierarchy in JSON or HTML. Charts add axes, units, legends, scales, and marks; identifying a trend is easier than extracting the value at a particular coordinate.

Visual grounding connects a phrase to a point, box, or mask. Pointing can expose evidence and enable grounded counting: the model indicates every object it counted. Structured JSON guarantees syntax but not geometric correctness.

Suppose OCR emits “Coffee 2 900; Tea 1 300.” Without column headers and coordinates, 900 could be a unit price or a row total; the system must not infer that convention from the number alone. Preserve the schema and, where necessary, ask for clarification. With explicit headers `quantity`, `unit_price_RUB`, and `line_total_RUB`, the coffee row can be `{quantity: 2, unit_price_RUB: 900, line_total_RUB: 1800}`: the arithmetic check is $2\times900=1800$, not an assumption that 900 already includes both units. For a chart, also preserve whether the scale is linear or logarithmic before calculating a difference.

Grounded counting exposes another useful check. If the answer says four cups, there should be four distinct valid points, not three points and a duplicate. For an absent object the correct output is an empty set. Relational descriptions such as “the cup to the right of the teapot” reduce ambiguity among similar objects.

Coordinates must be transformed back through crop origin, overlap, padding, and resize. If a tile coordinate is $(u,v)$, crop origin $(o_x,o_y)$, padding $(p_x,p_y)$, and scale $s$, then

$$
x=(u+o_x-p_x)/s,\qquad y=(v+o_y-p_y)/s.
$$

Here the tile origin is measured in the **padded, resized** coordinate system. For $s=0.5$, padding $(20,0)$, tile origin $(500,800)$, and local point $(120,60)$, the original point is

$$
x=(120+500-20)/0.5=1200,\qquad
y=(60+800)/0.5=1720.
$$

The exact metadata must travel with the tensor. Overlap is encoded by crop placement, not a universal subtraction. Rotated pages or unequal horizontal and vertical scaling need the corresponding inverse transform.

## Data, metrics, and interventions

Grounding data comes from detection datasets, referring expressions, document layouts, GUI screenshots, and human points. Negative examples teach the model to say that an object is absent. Ambiguous scenes require relational descriptions such as “the second cup from the left.”

OCR uses character/word error and exact match for critical fields; extraction uses field-level F1; boxes use IoU; points use containment or normalized distance. Report slices by font size, blur, rotation, language, aspect ratio, object size, and number of distractors.

Occluding the answer region, swapping table rows, or replacing a number with a synthetic value tests whether the output follows visual evidence rather than a familiar document template.

### Check metrics on a single error

Let the reference be `total 12480` and the transcription `total 1248O`, with letter O replacing zero. Ignoring spaces, one substitution among ten reference characters gives

$$
\mathrm{CER}=\frac{S+D+I}{N}=\frac1{10}=10\%.
$$

One of two words is wrong, so WER is 50%; the critical field's exact match is zero. These metrics tell different truths about the same output. State punctuation, whitespace, case, and Unicode normalization before reporting them.

For a reference box of area 100, predicted area 120, and intersection 80,

$$
\mathrm{IoU}=\frac{80}{100+120-80}\approx0.571.
$$

This passes an IoU threshold of 0.5 but might still include a neighboring number. Field extraction therefore needs semantic correctness and evidence-region quality, not just a detection threshold.

## A document-system contract

For each critical field, retain the source page, preprocessing version, coordinate convention, recognized text, model revision, and prompt under an explicit retention policy. Evidence should be inspectable in the interface. A confidence number is useful only with a stated calibration method; self-reported certainty is not calibrated probability.

If OCR and VLM disagree, a crop is missing, or the source is unreadable, return an abstention or review request. Do not silently “repair” a document from a familiar template. Counterfactual tests should replace the total, swap rows, hide the relevant region, and remove one tile. The predicted value and its evidence must follow the intervention.

## Sources and next chapter

- [Qwen2-VL](https://arxiv.org/abs/2409.12191), [Qwen2.5-VL](https://arxiv.org/abs/2502.13923), and [Molmo and PixMo](https://arxiv.org/abs/2409.17146).
- Previous: [[en/00 Textbook/16 Multimodal Models/64c Training VLMs — alignment, instruction tuning, and data|64.3]]. Next: [[en/00 Textbook/16 Multimodal Models/64e Video, audio, and omni models|64.5]].
