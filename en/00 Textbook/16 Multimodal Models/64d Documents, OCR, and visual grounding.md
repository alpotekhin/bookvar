---
title: "64.4. Documents, OCR, and visual grounding"
type: textbook-chapter
status: canonical
locale: en
translation_status: current
canonical_revision: "b2e3e923347babfafb3344252e55952c999287fe9fc51a19ad11965d4933f259"
translation_reviewed: "2026-10-29"
translation_of: "00 Учебник/16 Multimodal Models/64d Документы, OCR и visual grounding.md"
last_updated: 2026-10-29
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

*Figure 11 in Wang et al., [Qwen2-VL](https://arxiv.org/abs/2409.12191v1), PDF p. 29. The requested output translates dense Chinese text into English. It is a qualitative reading-and-translation example, not literal transcription or a CER measurement.*

A fluent translation may omit a word or number. It does not establish reliable extraction of account details or monetary amounts; those require reference source characters and field-value checks. In the running invoice example, subtotal 11,000 plus tax 1,480 equals total 12,480. Arithmetic consistency helps detect reading mistakes but does not by itself establish which row the user requested: another invoice may show a deposit below the total.

## Cascaded and end-to-end systems

A cascade runs specialist OCR first and gives text plus coordinates to an LLM. It is auditable and modular, but segmentation and reading-order errors become fixed upstream. An end-to-end VLM can combine text with color, arrows, and layout, yet its fluent answer is harder to verify and may fill missing characters from language priors.

A hybrid can retain OCR candidates and their provenance while letting a VLM inspect both text and pixels. Neither component guarantees complete transcription. A detector can omit a line entirely; rerunning recognition on the detected crops will not recover an absent crop. Keep detection, character recognition, reading order, field selection, localization, and arithmetic as separately inspectable outputs.

Sorting words by vertical position mixes the two columns of a page. A table, caption, and footnote can each follow a different reading order. Preserve block identity and layout before ordering lines within each block. When OCR words split into subword tokens, keep each token linked to the original region; a text-only string such as `Total 12480` loses that evidence link. Geometry can suggest a label–value pair but cannot establish semantics universally: values may wrap to the next line or follow a right-to-left layout.

### Two specialist alternatives

A layout-aware model makes geometry explicit. [LayoutLMv3](https://arxiv.org/abs/2204.08387) combines OCR text, two-dimensional positions, and image patches. An OCR-free model instead predicts structure directly from pixels: [Donut](https://arxiv.org/abs/2111.15664) avoids a separate OCR stage, while [Pix2Struct](https://arxiv.org/abs/2210.03347) learns screenshot-to-structure conversion through simplified HTML targets.

### LayoutLMv3: three pretraining targets

LayoutLMv3 combines token embeddings, one-dimensional text positions, and two-dimensional segment coordinates with projected page patches in a shared bidirectional Transformer. Words in the same segment share segment coordinates. The base $224\times224$ image with $16\times16$ patches provides 196 visual positions: relatively coarse for small text, but recognized characters also arrive through OCR. The model is not OCR-free.

![[00 Учебник/Assets/Figures/curated/documents-review-2026-10/layoutlmv3-objectives.png]]

*Huang et al., [LayoutLMv3, Figure 3](https://arxiv.org/pdf/2204.08387v3#page=3), CC BY-NC-SA 4.0. The shared Transformer supports masked language modeling, masked image modeling, and word–patch alignment.*

MLM reconstructs masked text while retaining its coordinates, allowing both textual and visual evidence. MIM predicts discrete codes from a separate image tokenizer for masked patches, not RGB pixels; that target tokenizer is different from the input patch projection. WPA predicts whether the patches corresponding to a visible word were masked. Visible text with a visible corresponding image region is aligned; visible text whose region was masked is unaligned. Masked text tokens are excluded from WPA.

In the figure, $T_1,T_2$ are MLM targets and $V_2,V_3$ are MIM targets. Visible $T_3$ corresponds to a masked patch and is unaligned; $T_4$ and visible $V_4$ are aligned. WPA is not selecting a correct bounding box from candidates. For one selected target per task with correct probabilities 0.8, 0.5, and 0.9, the sum of cross-entropies is $-\log0.8-\log0.5-\log0.9\approx1.02165$:

```python
from math import log, isclose

probabilities = {"MLM": 0.8, "MIM": 0.5, "WPA": 0.9}
losses = {name: -log(p) for name, p in probabilities.items()}
assert isclose(sum(losses.values()), 1.0216512475319812)
assert 224 // 16 * (224 // 16) == 196
```

These are illustrative probabilities, not measured LayoutLMv3 predictions. Real batches have their own selected positions and reductions. Downstream heads can label text tokens or choose answer-span endpoints. An extractive head cannot select a sum absent from the OCR text, unlike an unconstrained decoder that could generate a new string.

### Donut: serialize fields and then parse them

Donut uses a Swin image encoder and an autoregressive BART-based decoder initialized with multilingual weights. Cross-attention reads visual states; a task prefix selects classification, VQA, or field parsing.

![[00 Учебник/Assets/Figures/curated/documents-review-2026-10/donut-overview.png]]

*Kim et al., [official Donut overview](https://github.com/clovaai/donut/blob/4cfcf972560e1a0f26eb3e294c8fc88a0d336626/misc/overview.png). Original image unchanged. © 2022–present NAVER Corp.; [MIT license](/licenses/donut-4cfcf972-MIT.txt).*

A target such as `<s_total>12480</s_total><s_currency>RUB</s_currency>` becomes `{"total":"12480","currency":"RUB"}` through the published [`json2token`/`token2json`](https://github.com/clovaai/donut/blob/4cfcf972560e1a0f26eb3e294c8fc88a0d336626/donut/model.py#L499) conversion. `<sep/>` separates list entries. Boundaries must survive for both fields and items. Parsing validates structure, not whether the digits are correct.

Training uses the correct previous prefix; inference uses generated tokens. A missing closing token can break field extraction, while well-formed tags around a wrong amount produce valid but incorrect JSON. “OCR-free” describes inference inputs: Donut's pretraining still used external-OCR pseudo-labels for IIT-CDIP scans and known text from SynthDoG synthetic documents. Initial training reads page text in order, followed by task-specific structural targets. A target containing only a total does not teach its box; localization needs separate supervision or a later lookup.

### Pix2Struct: reconstruct a selected subtree

The target is simplified visible HTML structure, not full HTML with styles and scripts. Invisible elements and unnecessary nesting are removed while text, image filenames, and alt-text remain. If a whole page exceeds output length, a subtree is selected and highlighted in the screenshot. Half its text is masked, but the decoder reconstructs the entire selected subtree, including unmasked content.

![[00 Учебник/Assets/Figures/curated/documents-review-2026-10/pix2struct-screenshot-parsing.png]]

*Lee et al., [Pix2Struct, Figure 3](https://proceedings.mlr.press/v202/lee23g/lee23g.pdf#page=4), CC BY 4.0. Original form, highlighted masked region, and target serialization.*

In the example, Python is hidden, C++ remains visible, and Java is partly hidden. The model reconstructs all three entries and their nesting, not merely missing letters. Reading visible C++, inferring hidden Python from remaining evidence, and producing `img_alt` descriptions contribute distinct signals to one output sequence. A preceding reading curriculum on synthetic text images gives the decoder character-reading practice before screenshot parsing.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/pix2struct-figure1-tasks.png]]

*Lee et al., [Pix2Struct, Figure 1](https://arxiv.org/abs/2210.03347). Screenshot parsing supplies a pixel-to-structure objective that transfers to diagrams, interfaces, and documents.*

For VQA, Pix2Struct renders the question above the image, so question and document both enter visually. That is a Pix2Struct interface choice, not a requirement of all OCR-free models. For UI candidate selection, a candidate box is drawn on the image and candidates are scored; the task need not directly generate coordinates.

General VLMs need corresponding document supervision. Qwen2.5-VL §2.2 uses HTML targets retaining text, tables, formulas, order, and regions. Qwen3-VL §3.2.3 prepares real-document labels with a layout model followed by Qwen2.5-VL-72B region recognition and layout-preserving assembly. QwenVL-HTML includes element boxes; QwenVL-Markdown localizes images and tables, expressing tables in LaTeX. It does not establish a box for every text line in that second format.

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

A PDF's text layer can supply candidates but may itself be wrong or stale relative to the rendered image. Save the visible source. For a $1000\times1400$ page, consider four OCR fragments returned out of order: total value at $[700,1140,910,1180]$, subtotal label at $[100,1060,500,1100]$, total label at $[100,1140,250,1180]$, and subtotal value at $[700,1060,910,1100]$. Grouping these known rows by vertical center and sorting within rows produces the right label–value association:

```python
items = [
    ("v_total", "12 480", (700, 1140, 910, 1180)),
    ("l_subtotal", "Промежуточный итог", (100, 1060, 500, 1100)),
    ("l_total", "Итого", (100, 1140, 250, 1180)),
    ("v_subtotal", "11 000", (700, 1060, 910, 1100)),
]
groups = {}
for ident, text, box in items:
    cy = (box[1] + box[3]) / 2
    row = 0 if cy < 1120 else 1
    groups.setdefault(row, []).append((box[0], ident, text))
rows = [
    [entry[1] for entry in sorted(groups[row])]
    for row in sorted(groups)
]
assert rows == [["l_subtotal", "v_subtotal"], ["l_total", "v_total"]]
answer_id = rows[1][1]
answer_box = next(box for ident, _, box in items if ident == answer_id)
assert answer_box == (700, 1140, 910, 1180)
```

The row boundary 1120 is specific to this example, not a general layout algorithm. Keep the selected `v_total` ID and use its saved box for highlighting rather than asking a text model to invent coordinates afterward. If a VLM generates a box directly, evaluate that as a separate prediction. In a multipage file, “Total” on page 7 may concern last year while page 12 concerns this year; retrieval must preserve page, date, and table identity. If page 12 never reaches the answer model, it cannot extract the value from that missing context.

## Resolution and layout must be preserved together

A 20-pixel character on a page 2800 pixels high shrinks to 3.2 pixels when page height becomes 448. Interpolation can erase a distinguishing stroke. Tiles retain larger characters but can clip boundary lines or duplicate them in overlap. Reconcile detections after mapping to common page coordinates, using geometry and text: identical amounts can legitimately occur in two places.

![[00 Учебник/Assets/Figures/curated/documents-review-2026-10/pix2struct-variable-resolution.png]]

*Lee et al., [Pix2Struct, Figure 2](https://proceedings.mlr.press/v202/lee23g/lee23g.pdf#page=3), CC BY 4.0. Both panels use 36 patches, but only the left preserves aspect ratio.*

A $3\times12$ grid preserves a page ratio of 1:4, while $6\times6$ distorts it to a square despite the same 36-patch budget. Pix2Struct combines aspect-preserving resizing with two-dimensional patch positions; exact rounding and supported limits still belong to its processor.

```python
from math import isclose

original_height, letter_height = 2800, 20
reduced_letter = letter_height * 448 / original_height
assert isclose(reduced_letter, 3.2)
assert 3 * 12 == 6 * 6 == 36
assert isclose(3 / 12, 0.25)
```

## Tables, charts, and grounding

Tables are two-dimensional relations, not linear strings. A useful output preserves row and column headers, merged cells, and hierarchy in JSON or HTML. Charts add axes, units, legends, scales, and marks; identifying a trend is easier than extracting the value at a particular coordinate.

Visual grounding connects a phrase to a point, box, or mask. Pointing can expose evidence and enable grounded counting: the model indicates every object it counted. Structured JSON guarantees syntax but not geometric correctness.

Suppose OCR emits “Coffee 2 900; Tea 1 300.” Without column headers and coordinates, 900 could be a unit price or a row total; the system must not infer that convention from the number alone. Preserve the schema and, where necessary, ask for clarification. With explicit headers `quantity`, `unit_price_RUB`, and `line_total_RUB`, the coffee row can be `{quantity: 2, unit_price_RUB: 900, line_total_RUB: 1800}`: the arithmetic check is $2\times900=1800$, not an assumption that 900 already includes both units. For a chart, also preserve whether the scale is linear or logarithmic before calculating a difference.

Adding tea at one unit for 300 gives a total of 2100. Decimal strings and decimal arithmetic avoid confusing binary rounding with a document discrepancy:

```python
from decimal import Decimal

positions = [
    {"quantity": 2, "unit_price": "900.00", "line_total": "1800.00"},
    {"quantity": 1, "unit_price": "300.00", "line_total": "300.00"},
]
for position in positions:
    expected = position["quantity"] * Decimal(position["unit_price"])
    assert expected == Decimal(position["line_total"])
total = sum((Decimal(p["line_total"]) for p in positions), Decimal("0"))
assert total == Decimal("2100.00")
```

Taxes, discounts, delivery, and rounding rules need separate fields when present. Arithmetic consistency cannot reconstruct a lost header, and jointly wrong quantities and totals may coincidentally agree. Preserve raw values and regions; mark a calculated value as calculated rather than printed. A multilevel table header such as “2025” over “Revenue” and “Expenses” needs both levels attached to each cell, including merged-cell structure.

For a chart, let values 10 and 100 correspond to vertical pixels 300 and 100, and a mark lie at pixel 200. Its fraction upward is $t=(y_{low}-y)/(y_{low}-y_{high})=0.5$. Linear interpolation gives $a+t(b-a)=55$; a logarithmic axis gives $\exp((1-t)\log a+t\log b)=\sqrt{1000}\approx31.62$:

```python
from math import exp, log, isclose

a, b = 10.0, 100.0
y_low, y_high, y = 300.0, 100.0, 200.0
t = (y_low - y) / (y_low - y_high)
linear = a + t * (b - a)
logarithmic = exp((1 - t) * log(a) + t * log(b))
assert linear == 55.0
assert isclose(logarithmic, 31.622776601683793)
```

The example assumes exact pixel positions. Line thickness, compression, and tick-center selection create uncertainty on real charts; reporting many decimal places is unjustified when the line spans several pixels.

Grounded counting exposes another useful check. If the answer says four cups, there should be four distinct valid points, not three points and a duplicate. For an absent object the correct output is an empty set. Relational descriptions such as “the cup to the right of the teapot” reduce ambiguity among similar objects.

A point identifies an instance but not its extent. A box can include background or neighboring objects; a mask specifies pixels and supports shape/area checks. A precise mask cannot be recovered from a box alone without another prediction. Molmo trains on all-instance point annotations, including absent-object requests, and orders multiple points top-to-bottom and left-to-right. Four points can hit only three instances:

```python
regions = [(0, 0, 10, 10), (20, 0, 30, 10), (40, 0, 50, 10)]
points = [(5, 5), (25, 5), (26, 5), (45, 5)]
def instances_hit(regions, points):
    hits = []
    for x, y in points:
        candidates = [
            i for i, (x1, y1, x2, y2) in enumerate(regions)
            if x1 <= x < x2 and y1 <= y < y2
        ]
        assert len(candidates) == 1
        hits.append(candidates[0])
    return hits

hits = instances_hit(regions, points)
assert hits == [0, 1, 1, 2]
assert len(points) == 4 and len(set(hits)) == 3
assert instances_hit(regions, []) == []
```

These non-overlapping reference rectangles make instance membership explicit; proximity alone would not establish that two points hit the same object. A plausible box next to a correct amount also does not prove causal use of that region; image interventions test that separately.

| Pinned model/report | Coordinate convention |
|---|---|
| Molmo v2, §4 | normalized 0–100 |
| Qwen2.5-VL v1, §2.1.2 and grounding data | absolute image coordinates |
| Qwen3-VL v1, §3.2.4 | normalized 0–1000 |

The processor determines which image the absolute coordinates refer to. An interface that independently resizes an image must preserve that transform.

Coordinates must be transformed back through crop origin, overlap, padding, and resize. If a tile coordinate is $(u,v)$, crop origin $(o_x,o_y)$, padding $(p_x,p_y)$, and scale $s$, then

$$
\begin{gathered}
x=(u+o_x-p_x)/s,\\
y=(v+o_y-p_y)/s.
\end{gathered}
$$

Here the tile origin is measured in the **padded, resized** coordinate system. For $s=0.5$, padding $(20,0)$, tile origin $(500,800)$, and local point $(120,60)$, the original point is

$$
\begin{gathered}
x=(120+500-20)/0.5=1200,\\
y=(60+800)/0.5=1720.
\end{gathered}
$$

The exact metadata must travel with the tensor. Overlap is encoded by crop placement, not a universal subtraction. Rotated pages or unequal horizontal and vertical scaling need the corresponding inverse transform.

On a $600\times400$ tile, Molmo coordinate $(20,15)$ and Qwen3-VL coordinate $(200,150)$ both mean pixel point $(120,60)$. Convert normalization before reversing the spatial transform:

```python
from math import isclose

tile_width, tile_height = 600, 400
molmo, qwen3 = (20, 15), (200, 150)
molmo_px = (molmo[0] / 100 * tile_width, molmo[1] / 100 * tile_height)
qwen3_px = (qwen3[0] / 1000 * tile_width, qwen3[1] / 1000 * tile_height)
assert molmo_px == qwen3_px == (120, 60)

s, px, py, ox, oy = 0.5, 20, 0, 500, 800
u, v = molmo_px
x, y = (u + ox - px) / s, (v + oy - py) / s
assert isclose(x, 1200) and isclose(y, 1720)
```

Dividing a 0–1000 coordinate by 100 produces a tenfold error despite valid syntax. Clipping the resulting point to the page boundary hides the symptom without fixing the transform.

### Rotations, PDF coordinates, and box corners

Represent resize, padding, and crop in homogeneous coordinates:

$$
\begin{gathered}
A=\begin{pmatrix}s&0&p_x-o_x\\0&s&p_y-o_y\\0&0&1\end{pmatrix},
\\
(u,v,1)^\top=A(x,y,1)^\top.
\end{gathered}
$$

Use $A^{-1}$ for the inverse. A clockwise 90° rotation of a page of width $W$ and height $H$ maps continuous boundary coordinates as $x'=H-y$, $y'=x$. Pixel-center indexing uses different edge conventions. The box $[700,1140,910,1180]$ on a $1000\times1400$ page becomes $[220,700,260,910]$. Transform all four corners before taking coordinate extrema:

```python
import numpy as np

W, H = 1000, 1400
rotation = np.array([[0, -1, H], [1, 0, 0], [0, 0, 1]], dtype=float)
corners = np.array([
    [700, 1140, 1], [910, 1140, 1],
    [910, 1180, 1], [700, 1180, 1],
], dtype=float).T
rotated = rotation @ corners
rotated_box = (
    rotated[0].min(), rotated[1].min(),
    rotated[0].max(), rotated[1].max(),
)
assert rotated_box == (220, 700, 260, 910)
assert np.allclose(np.linalg.inv(rotation) @ rotated, corners)

scale_crop = np.array([[0.5, 0, -480], [0, 0.5, -800], [0, 0, 1]])
original = np.array([1200, 1720, 1])
assert np.allclose(scale_crop @ original, [120, 60, 1])
assert np.allclose(np.linalg.inv(scale_crop) @ [120, 60, 1], original)
```

At arbitrary angles, the axis-aligned envelope contains extra area; keeping the quadrilateral is more precise. PDF and raster origins may differ too. For an unrotated 700-point-high PDF with bottom-left origin, two pixels per point, and no crop-box offset, PDF point $(100,100)$ maps to raster $(200,1200)$. Obtain page rotation and crop-box transforms from the rasterizer, not guessed DPI scaling.

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

Levenshtein distance counts the minimum insertions, deletions, and substitutions. CER divides character edits by reference characters; WER divides word edits by reference words. The following uses the Russian source fixture, where Cyrillic `О` replaces digit `0`. Keeping the space gives length 11 and CER $1/11$; removing it gives 10%. Insertions can make CER exceed 100%, so it is not a bounded error probability:

```python
from math import isclose

def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(
                previous[j] + 1, current[j - 1] + 1,
                previous[j - 1] + (left != right),
            ))
        previous = current
    return previous[-1]

reference, prediction = "итого 12480", "итого 1248О"
cer_spaces = edit_distance(reference, prediction) / len(reference)
cer_no_spaces = (
    edit_distance(reference.replace(" ", ""), prediction.replace(" ", ""))
    / len(reference.replace(" ", ""))
)
wer = edit_distance(reference.split(), prediction.split()) / len(reference.split())
assert isclose(cer_spaces, 1 / 11)
assert isclose(cer_no_spaces, 0.1) and isclose(wer, 0.5)
assert edit_distance("12", "12345") / len("12") == 1.5
```

### ANLS and exact field values

[DocVQA 2020](https://www.docvqa.org/challenges/2020) uses Average Normalized Levenshtein Similarity. Normalize edit distance by the larger of reference and prediction lengths. If the normalized distance is strictly below 0.5, score $1-d$; otherwise score zero. Choose the highest score among accepted references, then average over questions. `12480` versus `1248О` scores 0.8 but remains an incorrect monetary amount:

```python
from math import isclose

def levenshtein(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        new = [i]
        for j, y in enumerate(b, 1):
            new.append(min(row[j] + 1, new[j - 1] + 1, row[j - 1] + (x != y)))
        row = new
    return row[-1]

def anls_one(reference, prediction):
    a, b = reference.lower(), prediction.lower()
    length = max(len(a), len(b))
    distance = levenshtein(a, b) / length if length else 0.0
    return 1 - distance if distance < 0.5 else 0.0

assert isclose(anls_one("12480", "1248О"), 0.8)
assert anls_one("ab", "ac") == 0.0
assert anls_one("RUB", "rub") == 1.0
assert max(anls_one(a, "12480") for a in ["12 480", "12480"]) == 1.0
```

This example ignores case and preserves spaces, with an explicit empty-string rule. Reproducing a benchmark table requires its official evaluator because versions can differ in answer aggregation and normalization. A distance exactly 0.5 fails this threshold.

For a reference box of area 100, predicted area 120, and intersection 80,

$$
\mathrm{IoU}=\frac{80}{100+120-80}\approx0.571.
$$

This passes an IoU threshold of 0.5 but might still include a neighboring number. Field extraction therefore needs semantic correctness and evidence-region quality, not just a detection threshold.

For boxes $[0,0,10,10]$ and $[2,0,14,10]$, the calculation is executable:

```python
from math import isclose

def box_iou(a, b):
    intersection = (
        max(0, min(a[2], b[2]) - max(a[0], b[0]))
        * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    )
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0

assert isclose(box_iou((0, 0, 10, 10), (2, 0, 14, 10)), 4 / 7)
assert box_iou((0, 0, 10, 10), (20, 0, 30, 10)) == 0.0
assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
```

Define a field match as a name plus exact value, optionally also page and a localization threshold. With five reference fields, four predicted fields, and three exact matches, precision is 0.75, recall 0.6, and F1 $2\cdot3/(4+5)=2/3$. Matching names alone would miss wrong values. Evaluate table text separately from cell structure and merged headers; for multipage QA, evaluate page selection, answer text, and location separately. Slice by language, small print, columns, rotation, scan quality, and handwriting.

### Interventions need the right expected behavior

Hold question, model, decoding, and preprocessing fixed, changing only a documented image region. Hide the printed total and its PDF text layer to test missing-field handling, but do not require refusal if visible line items still permit arithmetic recovery. In that case the system should identify its inputs and label the answer as calculated. Replacing the total and consistently updating its summands should change the extracted value. Moving complete label–value rows changes location but should preserve the answer; swapping values between labels should change the extracted field and may expose arithmetic inconsistency. Removing the relevant tile should prevent a claim to a precise location in an unavailable region.

Repeat interventions over documents, recording expected changes, stability under irrelevant edits, and false certainty separately. One successful edited invoice does not establish general reliability.

## A document-system contract

For each critical field, retain the source page, preprocessing version, coordinate convention, recognized text, model revision, and prompt under an explicit retention policy. Evidence should be inspectable in the interface. A confidence number is useful only with a stated calibration method; self-reported certainty is not calibrated probability.

If OCR and VLM disagree, a crop is missing, or the source is unreadable, return an abstention or review request. Do not silently “repair” a document from a familiar template. Counterfactual tests should replace the total, swap rows, hide the relevant region, and remove one tile. The predicted value and its evidence must follow the intervention.

Keep raw extraction, semantic normalization, and calculation distinct. The source string `12 480 ₽` can normalize to decimal string `12480.00` and currency RUB; an ambiguous Cyrillic `О` should not silently turn into zero. A reproducible record can retain:

```json
{
  "field": "total", "status": "extracted",
  "raw_text": "12 480", "value": "12480.00", "currency": "RUB",
  "evidence": {
    "page": 1, "coordinate_space": "page_pixels",
    "page_size": [1000, 1400], "box_xyxy": [700, 1140, 910, 1180],
    "region_id": "v_total"
  },
  "checks": {"arithmetic": "passed", "requires_review": false}
}
```

This example uses one-based pages and `x1,y1,x2,y2`, not a universal Donut/Qwen schema. Keep source-file identity, rasterizer version, page transforms, OCR output, model revision, and prompt. The UI must use the same rotation and scale when highlighting evidence.

A generated `confidence: 0.95`, or a sequence token probability, is not a validated probability that a field is correct. Calibration needs held-out labeled documents and a coverage–risk evaluation. Scores 0.99, 0.95, 0.80, 0.60 with correctness true, false, true, false yield 50% coverage and 50% accepted-error risk at threshold 0.90:

```python
from math import isclose

scores = [0.99, 0.95, 0.80, 0.60]
correct = [True, False, True, False]
accepted = [ok for score, ok in zip(scores, correct) if score >= 0.90]
coverage = len(accepted) / len(scores)
risk = sum(not ok for ok in accepted) / len(accepted)
assert isclose(coverage, 0.5) and isclose(risk, 0.5)

tp, predicted, expected = 3, 4, 5
assert isclose(tp / predicted, 0.75)
assert isclose(tp / expected, 0.6)
assert isclose(2 * tp / (predicted + expected), 2 / 3)
```

For ambiguous headers, missing pages, or recognizer disagreement, return the specific limitation and readable source crops for review. Repeating a confident model sentence adds no evidence about the disputed digit.

## Sources and next chapter

- [Qwen2-VL](https://arxiv.org/abs/2409.12191), [Qwen2.5-VL](https://arxiv.org/abs/2502.13923), and [Molmo and PixMo](https://arxiv.org/abs/2409.17146).
- [LayoutLMv3 §2.1–2.2, §3.1](https://arxiv.org/abs/2204.08387v3); [Donut §2](https://arxiv.org/abs/2111.15664v5); [Pix2Struct §2.2–2.5](https://proceedings.mlr.press/v202/lee23g.html); [Qwen3-VL document and grounding data, §3.2.3–3.2.4](https://arxiv.org/abs/2511.21631v1).
- [DocVQA 2020 evaluation](https://www.docvqa.org/challenges/2020) and [Scene Text VQA §3.4](https://openaccess.thecvf.com/content_ICCV_2019/papers/Biten_Scene_Text_Visual_Question_Answering_ICCV_2019_paper.pdf) specify the ANLS basis.
- Previous: [[en/00 Textbook/16 Multimodal Models/64c Training VLMs — alignment, instruction tuning, and data|Training VLMs]]. Next: [[en/00 Textbook/16 Multimodal Models/64e Video, audio, and omni models|Video, audio, and omni models]].
