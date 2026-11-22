---
title: "Resolution, tiling, and spatial positions"
type: textbook-chapter
status: canonical
locale: en
translation_status: current
canonical_revision: "863735716405d1b65fde69cf066ea5a0a5dfee819d12f0c2cfb7e5f97ec198fe"
translation_reviewed: "2026-11-22"
translation_of: "00 Учебник/16 Multimodal Models/64b Разрешение, tiling и пространственные позиции.md"
last_updated: 2026-11-22
last_verified: 2026-11-22
previous: "[[en/00 Textbook/16 Multimodal Models/64a Connectors and fusion]]"
next:
  - "[[en/00 Textbook/16 Multimodal Models/64c Training VLMs — alignment, instruction tuning, and data]]"
primary_sources:
  - https://arxiv.org/abs/2408.03326
  - https://arxiv.org/abs/2409.17146
  - https://arxiv.org/abs/2412.05271
  - https://arxiv.org/abs/2508.18265v1
  - https://arxiv.org/abs/2409.12191
  - https://arxiv.org/abs/2511.21631
  - https://cs336.stanford.edu/
---

# Resolution, tiling, and spatial positions

In the [[en/00 Textbook/16 Multimodal Models/64a Connectors and fusion|previous chapter]],
we compared ways to pass a scan's feature grid to an LLM. Take a
1600×1000 scan of the same regulations. The number of days allowed
for review is in a cell near the right edge, between columns 1475
and 1525. Fitting the page into a 336×336 square keeps the cell,
but shrinks its digits. Resizing the short side to 336 and cropping
the center square drops the cell altogether.

Tiling lets the ViT process more pixels per cell. The tile features
then have to be put back in order and fit into the LLM's context.
For the same scan, we will count pixels after preprocessing, patches
at the ViT input, and positions passed to the LLM.

<a id="vlm-pixel-budget"></a>

## Visual token budget

If the prepared height $H$ and width $W$ are both multiples of patch
size $P$, the grid has $HW/P^2$ patch positions, excluding CLS.
Merging each $m\times m$ group divides the position count by $m^2$
when the grid dimensions are multiples of $m$. Image boundaries,
row separators, and text add positions of their own.

With 14-pixel patches and a $2\times2$ merger:

| Prepared size | ViT | LLM |
|---|---:|---:|
| $224\times224$ | 256 | 64 |
| $336\times336$ | 576 | 144 |
| $672\times672$ | 2,304 | 576 |
| $1344\times1344$ | 9,216 | 2,304 |

The final row has as many merged positions as the preceding row has unmerged positions. “1344-pixel input” therefore does not determine LLM cost without patch size, merger, crop count, and special-token conventions.

Doubling both sides multiplies the number of query–key pairs in full attention by 16. This is not a 16-fold estimate for the entire network: projections and FFNs scale linearly with positions, and FlashAttention changes memory traffic. The ViT works on the grid before merging; the LLM works on the shorter grid afterward. Merging cannot undo computation already performed by the ViT.

### Stretching, fitting, and center-cropping

Stretching the 1600×1000 scan to $336\times336$ scales its width by
0.21 and its height by 0.336, distorting the letters. Fitting the
page without distortion gives $336\times210$, with 63 rows of
padding above and below. A 50-pixel-wide cell becomes about 10.5
pixels wide. A 16-pixel-tall digit becomes about 3.4 pixels tall;
whether it is readable depends on the font, contrast, and resampling.

Resizing the short side to 336 gives a width of about 538.
Center-cropping removes about 101 pixels per side in the resized
image, leaving approximately original columns 301 through 1301.
The cell at 1475–1525 is outside that crop. All three routes still
give the ViT 576 patches and the LLM 144 visual vectors after a
2×2 merger. The position count alone cannot tell us if the cell survived.

## Global overview plus detailed crops

<a id="vlm-molmo-overlap"></a>

### Molmo: overlapping crops and grid assembly

Molmo combines a low-resolution view of the full image with overlapping high-resolution crops. The overview preserves composition; crops preserve fine detail; overlap prevents boundary objects from losing context. Special image and row/column tokens record the final layout.

In [Molmo, Figure 5, p. 9](https://arxiv.org/pdf/2409.17146v2#page=9),
the left side shows the overview and four overlapping crops. On the
right, the first token block represents the overview and the second
represents the assembled high-resolution grid. The same part of the
gate appears at both scales. Those two representations are distinct
from duplicated margins within the high-resolution grid.

The drawn $6\times6$ grids are schematic. Molmo's CLIP-L/14 variant uses $336\times336$ crops, or $24\times24$ patches, combines features from two ViT layers along channels, then attention-pools each $2\times2$ group to a $12\times12$ grid. The mean of the four features supplies the query; the four features supply keys and values. The result is an attention-weighted sum, not arithmetic averaging. Learned padding embeddings distinguish fully visible, partly padded, and entirely padded patches.

Overlap supplies context for features near crop edges. Duplicate internal margins are removed after encoding, and the retained features are assembled row by row into one high-resolution grid. The [pinned processor](https://github.com/allenai/molmo/blob/793fa387edfd6fd0f5b21eb8e0a7620a1f3799e1/olmo/data/model_preprocessor.py#L386-L553) uses `overlap_margins=(4,4)`: four patches on each side. With 24-patch crops, the stride is $24-4-4=16$ patches, or 224 pixels. Neighboring 336-pixel inputs therefore overlap by 112 pixels. The paper's 56-pixel margin refers to one margin, not that whole overlap.

Horizontally, the first crop spans patches 0–23 and the second spans
16–39. Both encoder passes see patches 16–23. When assembling the
grid, keep features for 0–19 from the first crop and 20–39 from the
second. For example, the ViT can see patches 20–23 when it encodes
patch 19 in the left crop. Only the left crop contributes patch 19
to the assembled grid.

For a $2\times2$ crop layout, retaining external edges and removing duplicated internal margins leaves $10\times10$ pooled vectors per crop. Together these form a $20\times20$ grid. Add the $12\times12$ overview: there are $400+144=544$ visual vectors. A `<im_col>` marker follows each row, despite its name; 20 high-resolution rows, 12 overview rows, and two pairs of boundary markers make $544+32+4=580$ LLM positions.

<details id="resolution-code-molmo">
<summary>Counting Molmo positions</summary>

```python
def molmo_axis(tiles, patches=24, margin=4, pool=2):
    if tiles == 1:
        return patches // pool
    step = patches - 2 * margin
    first = (step + margin) // pool
    middle = step // pool
    last = (step + margin) // pool
    return first + (tiles - 2) * middle + last

high_h = high_w = molmo_axis(2)
overview_h = overview_w = 12
features = high_h * high_w + overview_h * overview_w
row_markers = high_h + overview_h
positions = features + row_markers + 4
assert (high_h, high_w, features, positions) == (20, 20, 544, 580)
```

</details>

The overview comes first, followed by rows of the assembled grid, not all tokens of crop 1 followed by all tokens of crop 2. That permutation preserves horizontal adjacency across crop boundaries.

<span id="anyres-глобальный-кадр-и-локальные-плитки"></span>

<a id="vlm-anyres"></a>

<span id="anyres-restore-detail-without-discarding-the-overview"></span>

## AnyRes: local tiles and a global view

LLaVA-NeXT and OneVision choose among allowed grids of encoder-sized
tiles. A separate overview shows where the cell belongs on the page.
The local tiles can preserve the small print. OneVision's higher
resolution path can shrink the feature grid after the ViT has read
those tiles.

![[00 Учебник/Assets/Figures/curated/resolution-2026-11-09/llava-onevision-anyres.png|The top panel encodes detailed tiles before feature interpolation. The bottom resizes the image before splitting it. Each panel includes a separate global view.]]

*Bo Li et al., [LLaVA-OneVision v1, Figure 2](https://arxiv.org/html/2408.03326v1#S3.F2),
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).
In (a), the ViT encodes smaller, more detailed tiles before interpolation
reduces the feature grid. In (b), the original AnyRes resizes the image
before splitting it. The purple block in each panel is the whole-page view.
Both panels are reproduced unchanged.*

The overview cannot add fine resolution to a crop, but tells the model where that crop belongs. A large letter may be legible locally while its column heading is visible only globally. Increasing tile count helps only if layout markers and positions keep these sources distinguishable and later compression does not remove the regained detail.

### Selecting a canvas and reassembling its grid

[LLaVA-NeXT's `select_best_resolution`](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/bce12e479bc4dfee2b9c50c88137b01ff51bd483/llava/mm_utils.py#L119-L215) maximizes effective image area after fitting inside each candidate canvas. Effective area is capped by original image area: upscaling creates no new source pixels. Ties favor less padding. For a $768\times384$ source, candidate canvases $384\times384$, $768\times384$, and $768\times768$ preserve 73,728, 294,912, and 294,912 source-equivalent pixels. The second wins because it avoids the third canvas's padding, yielding two tiles in a row.

OneVision's SigLIP produces $27\times27=729$ features for a $384\times384$ view. Two tiles must be reassembled as a $27\times54$ spatial grid. Concatenating their flattened vectors would incorrectly place the entire left tile before the right one. The `unpad` branch removes padded features, optionally interpolates the grid, and adds an `image_newline` vector after each row. The overview is prepended separately.

For the exact-ratio example, 1458 local vectors, 27 row separators, and 729 overview vectors total 2214 positions. The paper's simplified $(ab+1)T$ expression gives 2187 because it omits separators. In the following index example, the first row must continue from patch 26 to 729, not 27:

<details id="resolution-code-anyres">
<summary>Patch order in AnyRes</summary>

```python
import numpy as np

side = 27
tiles = np.arange(2 * side**2).reshape(1, 2, side, side)
grid = tiles.transpose(0, 2, 1, 3).reshape(side, 2 * side)
assert grid.shape == (27, 54)
assert grid[0, :28].tolist() == list(range(27)) + [729]
newline = np.full((side, 1), -1)
local = np.concatenate([grid, newline], axis=1).ravel()
assert (local == -1).sum() == 27
assert side**2 + len(local) == 2214
```

</details>

The pinned [`anyres_max`/`unpad` branch](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/bce12e479bc4dfee2b9c50c88137b01ff51bd483/llava/model/llava_arch.py#L350-L408) computes $r=\sqrt{hw/(K\cdot27^2)}$, interpolates only if $r>1.1$, and rounds the new sides down. The overview and separators lie outside this area budget, so `K=9` is not a strict limit of $9\cdot729$ total positions. For the $27\times54$ grid and $K=1$, $r=\sqrt2$ gives a $19\times38$ grid: 722 local features plus 19 separators and 729 overview positions, or 1470 total. Both detailed tiles have already passed through the ViT. This reduces LLM work, not the number of images encoded. Other branches, such as `maxpool2x2`, implement different operations.

## How InternVL chooses the tile count

InternVL chooses a grid of $448\times448$ tiles according to
aspect ratio. Pixel unshuffle rearranges 1,024 patch features per
tile into 256 positions. A processor accepting more tiles does not
show how well the model reads them.

In [Figure 2 of Zhe Chen et al.](https://arxiv.org/pdf/2412.05271v5#page=3), an
$800\times1300$ page is resized to $896\times1344$ and cut into two
columns by three rows of $448\times448$ tiles. A seventh square shows
the whole page: small letters are harder to read, but their locations
on the page remain visible. After the ViT, pixel unshuffle packs every
four neighboring positions into one wider vector. The MLP projects
those vectors to the LLM width. Seven squares contribute
$7\cdot256=1792$ visual vectors before special tokens.

### Aspect ratio does not determine tile count by itself

With `max_num=12`, [InternVL's processor](https://github.com/OpenGVLab/InternVL/blob/2410d1dbf208f0e799459aff9376e5747dbf41a2/internvl_chat/README.md#L279-L337) considers grids $(i,j)$ with $1\le ij\le12$, comparing $|W/H-i/j|$. On a tie it can prefer a larger grid if the original area exceeds half that grid's area. It resizes directly to the chosen rectangular canvas; this is not letterboxing, and an imperfect ratio match changes proportions slightly.

Both $1600\times800$ and $800\times400$ have aspect ratio 2:1. The first has area 1,280,000, exceeding half the area of eight 448-pixel tiles, 802,816, and chooses $4\times2$. The smaller image chooses $2\times1$. If enabled, a whole-image thumbnail is added only when there is more than one tile. These inputs therefore produce nine and three ViT squares, or 2304 and 768 visual vectors at 256 per square.

<details id="resolution-code-internvl">
<summary>Choosing the InternVL grid</summary>

```python
def internvl_grid(width, height, maximum=12, size=448):
    candidates = sorted(
        [(i, j) for i in range(1, maximum + 1)
         for j in range(1, maximum + 1) if i * j <= maximum],
        key=lambda ij: (ij[0] * ij[1], ij),
    )
    best, error = (1, 1), float("inf")
    for i, j in candidates:
        diff = abs(width / height - i / j)
        if diff < error:
            best, error = (i, j), diff
        elif diff == error and width * height > .5 * size**2 * i * j:
            best = (i, j)
    return best

assert internvl_grid(1600, 800) == (4, 2)
assert internvl_grid(800, 400) == (2, 1)
assert (8 + 1) * 256 == 2304
assert (2 + 1) * 256 == 768
```

</details>

The report's §4.1 gives different limits for the training datasets:
24 or 36 tiles for high-resolution documents and infographics, 6 or
12 for many other images, and one for a video frame. These are
dataset-specific settings. Section 3.1 describes how images in one
training sample share the tile budget.

### Pixel unshuffle is not averaging

A 448-pixel tile with patch size 14 yields $32\times32=1024$ patch features after removing CLS. Grouping neighboring $2\times2$ positions into channels transforms

$$
32\times32\times d_v\longrightarrow16\times16\times4d_v.
$$

The scalar count is unchanged and the permutation is reversible. An MLP then maps width $4d_v$ to the LLM width, leaving 256 positions. That learned projection may discard information, and the LLM no longer assigns four independent attention weights to the four original positions. InternVL calls the operation [`pixel_shuffle`](https://github.com/OpenGVLab/InternVL/blob/2410d1dbf208f0e799459aff9376e5747dbf41a2/internvl_chat/internvl/model/internvl_chat/modeling_internvl_chat.py#L257-L294) with factor 0.5. Its `v1` and `v2` variants differ in whether spatial axes are transposed back; channel and axis order are part of the checkpoint contract.

<details id="resolution-code-unshuffle">
<summary>Reversing the patch rearrangement</summary>

```python
import numpy as np

x = np.arange(16).reshape(1, 4, 4, 1)
y = x.reshape(1, 4, 2, 2).transpose(0, 2, 1, 3)
y = y.copy().reshape(1, 2, 2, 4).transpose(0, 2, 1, 3)
assert y[0, 0, 0].tolist() == [0, 1, 4, 5]
assert y.shape == (1, 2, 2, 4)
restored = y.transpose(0, 2, 1, 3).copy()
restored = restored.reshape(1, 2, 4, 2).transpose(0, 2, 1, 3)
restored = restored.copy().reshape(x.shape)
assert np.array_equal(restored, x)
```

</details>

<a id="vlm-internvl-resolution-router"></a>

### InternVL3.5-Flash: how many vectors to keep per tile

[InternVL 3.5, §2.1](https://arxiv.org/html/2508.18265v1#S2.SS1)
retains tiling and the overview. The upper diagram follows one tile
after the ViT: the feature rearrangement reduces 1,024 positions
to 256, then the MLP changes the width of each vector. The lower
diagram shows Flash. Its router receives features from the same tile
and chooses one of two paths: `High Resolution` keeps 256 vectors,
and `Low Resolution` keeps 64. Both paths lead to the projector.
Image dimensions determine the tile grid earlier; here the router
determines how many features to pass on.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-connectors-2026-11-13/internvl35-figure2c.png|Panel c: InternVL 3.5 uses fixed compression before the MLP. The Flash router selects a high- or low-resolution path.]]

*Weiyun Wang et al., [InternVL3.5, Figure 2](https://arxiv.org/html/2508.18265v1#S2.F2),
[original PNG](https://arxiv.org/html/2508.18265v1/architecture.png),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Complete panel (c) cropped from the source; its pixels and labels are unchanged.*

Take the six detailed tiles from the page above, excluding the overview.
The standard path produces $6\cdot256=1536$ vectors. If Flash keeps 256
vectors for four tiles and 64 for the other two, the LLM receives
$4\cdot256+2\cdot64=1152$ vectors. That is one quarter fewer with
the same ViT input. The split is illustrative; the report does not
give routing decisions for this page.

For router training, [InternVL3.5 compares both paths](https://arxiv.org/html/2508.18265v1#S2.SS3)
with one frozen reference model, which always uses 256 vectors per
tile. Both paths get a ViCO loss based on the KL divergence of the
answer-token distributions. If the losses are 0.06 with 64 vectors
and 0.02 with 256, their ratio is 3. With an illustrative threshold
of 2, the training label says “keep 256.” The paper sets the
threshold from recent ratios, rather than fixing it at 2. At
inference the trained classifier chooses a path without computing
both losses.

<span id="native-dynamic-resolution-and-compression"></span>

## A rectangular grid without tiling

Qwen2-VL and Pixtral can encode the scan as one rectangular
grid. Their preprocessors can still resize the page and round
its dimensions to match the patch size and pixel budget.
The encoder uses grid coordinates rather than a table for
one fixed number of rows and columns.

<span id="qwen2-vl-native-resolution-merger-и-mrope"></span>

### Qwen2-VL: count patches, merged vectors, and delimiters separately

With $14\times14$ spatial patches and a $2\times2$ spatial merger, a prepared $224\times224$ image yields

$$
\begin{gathered}
(224/14)^2=256\\
\text{patch positions},
\\
256/4=64\\
\text{merged visual vectors}.
\end{gathered}
$$

The Qwen2-VL paper reports 66 tokens because it also counts `<|vision_start|>` and `<|vision_end|>`: 64 visual vectors plus two boundaries. Its ViT uses RoPE-2D, while MRoPE supplies positions to the multimodal decoder sequence. These are different attention systems.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/multimodal/qwen2-vl-architecture.png|Qwen2-VL turns images of different dimensions and videos into visual sequences of different lengths.]]

*Peng Wang et al., [Qwen2-VL, Figure 2, p. 4](https://arxiv.org/pdf/2409.12191v1#page=4),
as used in Stanford CS336 Lecture 17. The 224×28 strip gives
$(224/28)\cdot(28/28)=8$ visual vectors, and the 1260×700 photo
gives $(1260/28)\cdot(700/28)=1125$. Those counts follow the merger
and exclude the boundary tokens.*

“Native” resolution still means a processor-constrained grid. Record the resized dimensions and allowed pixel range rather than assuming that every original pixel reaches the encoder unchanged.

For a concrete [`smart_resize` example](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L54-L85), set `factor=28`, `min_pixels=56*56`, and `max_pixels=1280*28*28`. An image of height 1000 and width 1600 first rounds to $1008\times1596$, exceeding the 1,003,520-pixel cap. The processor divides original sides by $\beta=\sqrt{1000\cdot1600/1{,}003{,}520}$ and rounds each down to a multiple of 28, producing $784\times1260$. The ViT sees $56\times90=5040$ patches and the LLM receives $28\times45=1260$ merged vectors. Discrete allowed dimensions mean the nominal 1280-position budget need not be filled exactly, and aspect ratio is preserved only approximately.

<details id="resolution-code-resize">
<summary>Rounding image dimensions in Qwen2-VL</summary>

```python
import math

def resize_example(height, width, factor=28,
                   min_pixels=56**2, max_pixels=1280 * 28**2):
    if max(height, width) / min(height, width) > 200:
        raise ValueError("unsupported aspect ratio")
    h = round(height / factor) * factor
    w = round(width / factor) * factor
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h = max(factor, math.floor(height / beta / factor) * factor)
        w = max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    return h, w

h, w = resize_example(1000, 1600)
assert (h, w) == (784, 1260)
assert (h // 14) * (w // 14) == 5040
assert (h // 28) * (w // 28) == 1260
```

</details>

At horizontal scale $1260/1600$, the cell spans roughly columns
1162 to 1201. With a stride of 28 pixels after the 2×2 merger,
it lies in grid columns 41 and 42, counted from zero. A 16-pixel-tall
digit spans about 12.5 pixels vertically, up from 3.4 in the fitted
square. Adding the two boundaries gives $1260+2=1262$ LLM positions.
This compares input geometry, not OCR accuracy.

These are explicitly chosen processor parameters, not universal defaults. Qwen2-VL's patch embedding also spans two temporal samples. For a still image, the processor repeats the frame, giving `grid_t=1`, not twice as many visual tokens.

### Windowed attention in Qwen2.5-VL

In [Qwen2.5-VL, §2.1.1 and Table 1](https://arxiv.org/abs/2502.13923v1), most vision layers use $112\times112$-pixel windows, or $8\times8$ patches; layers 7, 15, 23, and 31 retain full attention. A $32\times32$ grid has 16 windows of 64 patches, giving $16\cdot64^2=65{,}536$ pairs per windowed head versus $1024^2=1{,}048{,}576$ for full attention, excluding padding, special positions, projections, and FFNs. Global layers still connect distant image regions, so the complete ViT is not uniformly linear in grid size.

<a id="vlm-pixtral-packing"></a>

### Pixtral: packing images without cross-image leakage

![[00 Учебник/Assets/Figures/curated/resolution-review-2026-10/pixtral-vision-packing.png|Two images occupy separate vision-attention blocks. The processor adds row and image delimiters after projection.]]

*Agrawal et al., [Pixtral 12B, Figure 2](https://arxiv.org/abs/2410.07073v1), CC BY 4.0. The original figure shows separate image-attention regions and the subsequent row/image delimiters.*

Pixtral uses $16\times16$ patches and RoPE-2D inside its vision encoder. Images with 12 and four patches can be packed into one 16-position tensor, but a block-diagonal mask must isolate their attention. Otherwise an image's encoding would depend on unrelated batch neighbors. The two-layer projector changes feature width without reducing position count. [`PixtralProcessor` v4.57.1](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/models/pixtral/processing_pixtral.py#L166-L207) appends `[IMG_BREAK]` after each row, replacing the last one with `[IMG_END]`:

```text
[IMG] [IMG] [IMG] [IMG_BREAK]
[IMG] [IMG] [IMG] [IMG_END]
```

The $2\times3$ grid uses six visual vectors and two separators. A $3\times2$ grid has the same visual count but three separators. Total patch count alone does not specify row geometry.

<details id="resolution-code-pixtral">
<summary>Pixtral row markers and attention mask</summary>

```python
import numpy as np

def pixtral_rows(height, width):
    result = []
    for _ in range(height):
        result.extend(["[IMG]"] * width + ["[IMG_BREAK]"])
    result[-1] = "[IMG_END]"
    return result

assert len(pixtral_rows(2, 3)) == 8
assert len(pixtral_rows(3, 2)) == 9
image_id = np.repeat([0, 1], [12, 4])
allowed = image_id[:, None] == image_id[None, :]
assert allowed[:12, :12].all() and allowed[12:, 12:].all()
assert not allowed[:12, 12:].any()
```

</details>

This block-diagonal isolation belongs to the vision encoder's batch, not the multimodal conversation. Pixtral's language decoder uses ordinary one-dimensional RoPE and causal attention; later text can read preceding images.

### Equal output lengths can hide different compression

Averaging scalar groups $(1,0,0,1)$ and $(0,1,1,0)$ gives 0.5 in both cases, losing the distinction between their diagonal patterns. Channel concatenation retains all four values and their order until a subsequent projection; attention pooling learns content-dependent weighting. Molmo's mean query does not make its output an arithmetic mean. InternVL's permutation and Qwen's grouping before an MLP should not be treated as the same operation as averaging.

Q-Former and Perceiver Resampler instead read the grid with a fixed number of learned queries. Increasing image size can leave output length unchanged, forcing each latent to represent more content. A fixed output budget is not evidence that every small character survives. Compare OCR, small objects, spatial relations, and latency at several budgets with the same inputs. Measure vision-encoder work as well as LLM sequence length.

<span id="6-почему-одной-позиции-в-последовательности-недостаточно"></span>

## From one-dimensional order to image geometry

Flattening a grid into one sequence makes the end of one row adjacent to the start of the next. Qwen2-VL’s M-RoPE partitions rotary dimensions among time, height, and width. Text uses matching indices; images keep time fixed while height and width vary; video varies all three.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/qwen2vl-figure3-mrope.png|In Qwen2-VL MRoPE, text has equal indices on all three axes; image positions vary by height and width, and video also varies in time.]]

*Figure 3 from Wang et al., [Qwen2-VL](https://arxiv.org/abs/2409.12191), PDF p. 5. M-RoPE separates temporal, vertical, and horizontal positions.*

### Position IDs are not cache addresses

Suppose three text positions 0, 1, and 2 precede a merged image grid with two rows and three columns; position 2 is `<|vision_start|>`. Offset the image's coordinates by 3:

```text
top row:    (3,3,3)  (3,3,4)  (3,3,5)
bottom row: (3,4,3)  (3,4,4)  (3,4,5)
```

Following text begins at the largest coordinate plus one, 6. The next two positions, `<|vision_end|>` and a question token, receive $(6,6,6)$ and $(7,7,7)$. There are 11 sequence positions, but the next rotary index is 8. The KV cache still contains 11 positions; geometric coordinates do not replace cache length.

<details id="resolution-code-positions">
<summary>MRoPE indices and cache length</summary>

```python
import numpy as np

prefix = np.repeat(np.arange(3)[None, :], 3, axis=0)
h, w = np.indices((2, 3))
visual = np.stack([
    np.zeros(6, dtype=int), h.ravel(), w.ravel()
]) + 3
suffix_start = visual.max() + 1
suffix = np.repeat(
    np.arange(suffix_start, suffix_start + 2)[None, :], 3, axis=0
)
position_ids = np.concatenate([prefix, visual, suffix], axis=1)
assert position_ids.shape == (3, 11)
assert position_ids[:, 3].tolist() == [3, 3, 3]
assert position_ids[:, 8].tolist() == [3, 4, 5]
assert position_ids[:, 9].tolist() == [6, 6, 6]
rope_delta = int(position_ids.max() + 1 - position_ids.shape[1])
assert rope_delta == -3
```

</details>

The difference −3 is what [`get_rope_index`](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L925-L1082) records as `mrope_position_deltas`. Subsequent generated text continues with this offset. Changing grid shape can change the offset without changing image count.

<a id="vlm-mrope-frequencies"></a>

### Which frequencies each axis uses

For one rotary pair and frequency $\omega$,

$$
R(p\omega)=\begin{pmatrix}\cos(p\omega)&-\sin(p\omega)\\\sin(p\omega)&\cos(p\omega)\end{pmatrix}.
$$

The dot product after both rotations is

$$
q^\top R((p_k-p_q)\omega)k.
$$

MRoPE assigns different pairs to differences in time, height, and width.
Horizontal neighbors differ only in width; vertical neighbors differ in
height. A shared shift leaves this dot product unchanged for fixed $q$
and $k$. The full image network need not be translation-invariant:

<details id="resolution-code-rotary">
<summary>The dot product after rotation</summary>

```python
import numpy as np

def rotate_three(x, position, omega=.7):
    pairs = np.asarray(x, dtype=float).reshape(3, 2)
    angles = np.asarray(position) * omega
    c, s = np.cos(angles), np.sin(angles)
    return np.stack([
        c * pairs[:, 0] - s * pairs[:, 1],
        s * pairs[:, 0] + c * pairs[:, 1],
    ], axis=1).ravel()

q = np.array([1., 2., 3., 4., 5., 6.])
k = np.array([2., -1., 1., 0., -2., 3.])
p, r = np.array([3, 3, 3]), np.array([3, 3, 4])
offset = np.array([7, 11, 13])
a = rotate_three(q, p) @ rotate_three(k, r)
b = rotate_three(q, p + offset) @ rotate_three(k, r + offset)
assert np.isclose(a, b)
```

</details>

Qwen2.5-VL scales temporal position IDs with physical time. [Qwen3-VL replaces that absolute-time alignment](https://arxiv.org/html/2511.21631v1#S2.SS3) with explicit textual timestamps before video temporal patches; it does not simply add timestamps to the same time-scaled-ID scheme. Interleaved M-RoPE remains: temporal, height, and width coordinates are distributed across rotary frequency ranges rather than assigned one contiguous band per axis. These are separate decisions about representing physical time and allocating positional frequencies. Neither repairs incorrect timestamps or a wrong patch grid.

Interleaving does not mean repeating `t,h,w` uniformly over every pair. For example, `mrope_section=[24,20,20]` in [Qwen3-VL's pinned implementation](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L297-L314) assigns 24 frequency pairs to time and 20 each to height and width. Pixtral's RoPE-2D instead alternates height and width within vision attention; it does not imply multidimensional rotary positions in that model's language decoder.

Consider a smaller head with eight frequencies, numbered 0 through 7
in descending order, and an allocation of four, two, and two pairs.
Contiguous blocks give `T T T T H H W W`. The interleaving code starts
with all `T`, sets indices `1, 4` to `H`, and sets `2, 5` to `W`.
The result is `T H W T H W T T`. The counts are unchanged, but width
uses $\omega_2$ and $\omega_5$ instead of $\omega_6$ and $\omega_7$.

Horizontal neighbors differ by one in width. With $q=k=(1,0)$
in one width pair, that pair contributes $\cos\omega_j$ to the dot
product. The contribution is about 0.8776 for $\omega_2=0.5$ and
0.9988 for $\omega_6=0.05$. These are selected frequencies for one
pair, not a prediction of the trained model's full attention logit.

## Returning coordinates to the original image

A box predicted inside a tile must be offset by the crop origin,
corrected for padding, and divided by resize scale. Preprocessing
metadata must travel with the tensor. Inverting coordinates maps
locations within the retained region; it does not restore lost pixels.
When the overview and crops contain the same object, the target
format must specify which coordinate system it uses.

For example, resize a $1600\times1000$ page by $s=0.5$, then add 12 pixels of left padding and 6 of top padding. A prepared-image box $(112,56,312,156)$ maps back to

$$
\begin{gathered}
\frac{(112-12,\;56-6,\;312-12,\;156-6)}{0.5}\\
=(200,100,600,300).
\end{gathered}
$$

For tile-local predictions, first add the tile origin in the padded, resized coordinate system, then remove padding and invert resize. Overlap is already represented by tile origins; it is not a universal extra offset to subtract. Duplicate detections in overlapping tiles need reconciliation. Rotation or nonuniform scaling requires the corresponding inverse transform rather than one scalar.

First verify the model's output contract. A model already trained to return original-image coordinates must not be inverse-transformed again. For the order resize → padding → crop, with scales $(s_x,s_y)$, padding $(p_x,p_y)$, and crop origin $(o_x,o_y)$, a point obeys

$$
u=s_xx+p_x-o_x,\quad v=s_yy+p_y-o_y,
$$
$$
\begin{gathered}
x=(u+o_x-p_x)/s_x,\\
y=(v+o_y-p_y)/s_y.
\end{gathered}
$$

A crop beginning at $(80,40)$ turns the earlier prepared box into $(32,16,232,116)$. Both coordinate paths must recover the same original box:

<details id="resolution-code-inverse">
<summary>Mapping the box to the original coordinates</summary>

```python
import numpy as np

def to_original(box, scale, padding, origin=(0, 0)):
    box = np.asarray(box, dtype=float).reshape(2, 2)
    return ((box + origin - np.asarray(padding)) / scale).ravel()

expected = np.array([200, 100, 600, 300])
a = to_original([112, 56, 312, 156], [.5, .5], [12, 6])
b = to_original([32, 16, 232, 116], [.5, .5], [12, 6], [80, 40])
assert np.allclose(a, expected) and np.allclose(b, expected)
```

</details>

Coordinates here are continuous rectangle boundaries, without the `+1` used by some discrete formats. If a crop clipped the object, inversion returns only the visible box, not the missing part.

<a id="vlm-tile-boxes"></a>

### Boxes from overlapping tiles

Keep tile A at origin $(80,40)$ and add tile B at $(96,48)$,
with the same scale and padding. When the full object fits in both,
the local boxes are $(32,16,232,116)$ and $(16,8,216,108)$.
Both map to $(200,100,600,300)$, so their IoU is 1.
Comparing the local numbers directly gives about 0.7337 even
though both predictions are exact. No separate overlap correction
is needed; the tile origins already account for the difference.

Now let tile B end at local $u=128$. Its visible box ends at 128
rather than 216; that edge maps to $x=(128+96-12)/0.5=424$.
With unchanged height, the intersection with the full box covers
224 of its 400 pixels in width, for an IoU of 0.56. A threshold
of 0.7 would leave two detections for the same object. Lowering
the threshold can also merge adjacent objects of the same class.

[SAHI, §3](https://arxiv.org/html/2202.06934v5#S3) applies this
sequence to an object detector: predict on each tile, return the
boxes to the image coordinate system, then match them.
That is post-processing of tile predictions, not the assembly of
an input feature grid before an LLM as in AnyRes.

<span id="где-теряется-информация-crop-tiling-merge-и-фиксированный-budget"></span>

<span id="diagnose-the-stage-that-lost-the-evidence"></span>

## Tracing a reading or coordinate error

If the cell at the right edge never reached ViT, changing rotary
positions cannot restore it. If the digit is readable on the
prepared scan but the model still gives a wrong answer, compare
the same pixels at different merger budgets and check row and
tile order. A human reading the crop shows only that its pixels
contain the digit, not that the encoder's features retained it.

If the digit is right but the box is shifted, map the box back to
the original scan. A constant offset suggests checking padding
and tile origins; a growing error away from the corner suggests
checking the scales. Both can be tested with a known box and the
same metadata used to prepare the scan.

To compare budgets, measure reading accuracy, ViT and prefill
times, and memory. Include scans, photos and unusually shaped
images. Log the prepared dimensions, tile origins and padding,
grid shape, positions after merging, and full LLM input length.
A per-image cap does not ensure that ten images, the question,
and the expected answer fit together.

The same visible cell can support a text answer, “14 days,” or
a bounding box. Those outputs require different supervision.
[[en/00 Textbook/16 Multimodal Models/64c Training VLMs — alignment, instruction tuning, and data|The next chapter]]
looks at how such examples are used in training.

## Sources and next chapter

- [Molmo and PixMo](https://arxiv.org/abs/2409.17146), [InternVL 2.5](https://arxiv.org/abs/2412.05271), [Qwen2-VL](https://arxiv.org/abs/2409.12191), and [Pixtral 12B](https://arxiv.org/abs/2410.07073).
- Version-specific details: [Molmo Appendix A.1](https://arxiv.org/abs/2409.17146v2), [LLaVA-OneVision §3.2](https://arxiv.org/abs/2408.03326v1), [InternVL 2.5 §2.1, §3.1, §4.1](https://arxiv.org/abs/2412.05271v1), [Qwen2-VL §2.1](https://arxiv.org/abs/2409.12191v1), [Qwen2.5-VL §2.1](https://arxiv.org/abs/2502.13923v1), [Qwen3-VL §2.1–2.3](https://arxiv.org/abs/2511.21631v1), [Pixtral §2.1–2.3](https://arxiv.org/abs/2410.07073v1).
- Previous: [[en/00 Textbook/16 Multimodal Models/64a Connectors and fusion|Connecting a vision encoder to an LLM]]. Next: [[en/00 Textbook/16 Multimodal Models/64c Training VLMs — alignment, instruction tuning, and data|Training VLMs]].
