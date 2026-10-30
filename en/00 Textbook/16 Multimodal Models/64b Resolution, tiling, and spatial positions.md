---
title: "64.2. Resolution, tiling, and spatial positions"
type: textbook-chapter
status: canonical
locale: en
translation_status: current
canonical_revision: "aa23f1104f3a83d91bc759f207ab809ebf018b9946f44a022c761fc3e9e81258"
translation_reviewed: "2026-10-29"
translation_of: "00 Учебник/16 Multimodal Models/64b Разрешение, tiling и пространственные позиции.md"
last_updated: 2026-10-29
last_verified: 2026-09-15
primary_sources:
  - https://arxiv.org/abs/2408.03326
  - https://arxiv.org/abs/2409.17146
  - https://arxiv.org/abs/2412.05271
  - https://arxiv.org/abs/2409.12191
  - https://arxiv.org/abs/2511.21631
  - https://cs336.stanford.edu/
---

# Resolution, tiling, and spatial positions

A photograph can survive aggressive downsampling; a contract may lose the digits that determine its meaning. Fixed classification-era resolutions are therefore a central VLM limitation. The model must allocate enough visual tokens and preserve coordinates through resize, padding, crops, and merging.

For patch size $P$, $N=HW/P^2$. Doubling both dimensions creates four times as many patches. Compute is governed by the post-preprocessing token count, not the JPEG file size.

A square resize has two distinct costs. Stretching changes aspect ratio; fitting inside a square introduces padding and may shrink the long side so far that characters disappear. More reasoning cannot recover an unreadable digit.

| Prepared size, patch size 14 | Patch grid | Patch positions | After a $2\times2$ merger |
|---|---:|---:|---:|
| $224\times224$ | $16\times16$ | 256 | 64 |
| $336\times336$ | $24\times24$ | 576 | 144 |
| $672\times672$ | $48\times48$ | 2,304 | 576 |
| $1344\times1344$ | $96\times96$ | 9,216 | 2,304 |

The final row has as many merged positions as the preceding row has unmerged positions. “1344-pixel input” therefore does not determine LLM cost without patch size, merger, crop count, and special-token conventions.

Doubling both sides multiplies the number of query–key pairs in full attention by 16. This is not a 16-fold estimate for the entire network: projections and FFNs scale linearly with positions, and FlashAttention changes memory traffic. The ViT works on the grid before merging; the LLM works on the shorter grid afterward. Merging cannot undo computation already performed by the ViT.

Consider a page 1600 pixels wide and 1000 high. Stretching it to $336\times336$ uses horizontal scale 0.21 and vertical scale 0.336, distorting letters. Fitting it without distortion produces $336\times210$, with 63 padding rows above and below. Resizing the short side to 336 produces a width of about 538; center-cropping then removes about 101 prepared pixels from each side. These operations retain different evidence. `resize` alone is not a reproducible preprocessing specification, even with an identical checkpoint.

## Global overview plus detailed crops

Molmo combines a low-resolution view of the full image with overlapping high-resolution crops. The overview preserves composition; crops preserve fine detail; overlap prevents boundary objects from losing context. Special image and row/column tokens record the final layout.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/molmo-figure5-image-tokenization.png]]

*Figure 5 from Deitke et al., [Molmo and PixMo](https://arxiv.org/abs/2409.17146), PDF p. 9. The figure traces an image through low- and high-resolution crops into a token sequence.*

The drawn $6\times6$ grids are schematic. Molmo's CLIP-L/14 variant uses $336\times336$ crops, or $24\times24$ patches, combines features from two ViT layers along channels, then attention-pools each $2\times2$ group to a $12\times12$ grid. The mean of the four features supplies the query; the four features supply keys and values. The result is an attention-weighted sum, not arithmetic averaging. Learned padding embeddings distinguish fully visible, partly padded, and entirely padded patches.

Overlap supplies context for features near crop edges. Duplicate internal margins are removed after encoding, and the retained features are assembled row by row into one high-resolution grid. The [pinned processor](https://github.com/allenai/molmo/blob/793fa387edfd6fd0f5b21eb8e0a7620a1f3799e1/olmo/data/model_preprocessor.py#L386-L553) uses `overlap_margins=(4,4)`: four patches on each side. With 24-patch crops, the stride is $24-4-4=16$ patches, or 224 pixels. Neighboring 336-pixel inputs therefore overlap by 112 pixels. The paper's 56-pixel margin refers to one margin, not that whole overlap.

For a $2\times2$ crop layout, retaining external edges and removing duplicated internal margins leaves $10\times10$ pooled vectors per crop. Together these form a $20\times20$ grid. Add the $12\times12$ overview: there are $400+144=544$ visual vectors. A `<im_col>` marker follows each row, despite its name; 20 high-resolution rows, 12 overview rows, and two pairs of boundary markers make $544+32+4=580$ LLM positions.

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

The overview comes first, followed by rows of the assembled grid, not all tokens of crop 1 followed by all tokens of crop 2. That permutation preserves horizontal adjacency across crop boundaries.

InternVL chooses a grid of $448\times448$ tiles according to aspect ratio. Pixel unshuffle reduces 1,024 patch features per tile to 256 visual tokens. Training and inference impose maximum tile counts; accepting a longer sequence technically does not prove quality beyond the training distribution.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/internvl25-figure2-dynamic-tiles.png]]

*Figure 2 from Zhe Chen et al., [InternVL 2.5](https://arxiv.org/abs/2412.05271), PDF p. 3. Dynamic tiling feeds a ViT–MLP–LLM stack for images, multiple images, and video.*

### Aspect ratio does not determine tile count by itself

With `max_num=12`, [InternVL's processor](https://github.com/OpenGVLab/InternVL/blob/2410d1dbf208f0e799459aff9376e5747dbf41a2/internvl_chat/README.md#L279-L337) considers grids $(i,j)$ with $1\le ij\le12$, comparing $|W/H-i/j|$. On a tie it can prefer a larger grid if the original area exceeds half that grid's area. It resizes directly to the chosen rectangular canvas; this is not letterboxing, and an imperfect ratio match changes proportions slightly.

Both $1600\times800$ and $800\times400$ have aspect ratio 2:1. The first has area 1,280,000, exceeding half the area of eight 448-pixel tiles, 802,816, and chooses $4\times2$. The smaller image chooses $2\times1$. If enabled, a whole-image thumbnail is added only when there is more than one tile. These inputs therefore produce nine and three ViT squares, or 2304 and 768 visual vectors at 256 per square.

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

The report's limits vary by evaluation data: 24 or 36 tiles for difficult documents and infographics, 6 or 12 for many other images, and one for a video frame. The number 36 is not a universal training configuration. Multiple images must also share a request-level budget.

### Pixel unshuffle is not averaging

A 448-pixel tile with patch size 14 yields $32\times32=1024$ patch features after removing CLS. Grouping neighboring $2\times2$ positions into channels transforms

$$
32\times32\times d_v\longrightarrow16\times16\times4d_v.
$$

The scalar count is unchanged and the permutation is reversible. An MLP then maps width $4d_v$ to the LLM width, leaving 256 positions. That learned projection may discard information, and the LLM no longer assigns four independent attention weights to the four original positions. InternVL calls the operation [`pixel_shuffle`](https://github.com/OpenGVLab/InternVL/blob/2410d1dbf208f0e799459aff9376e5747dbf41a2/internvl_chat/internvl/model/internvl_chat/modeling_internvl_chat.py#L257-L294) with factor 0.5. Its `v1` and `v2` variants differ in whether spatial axes are transposed back; channel and axis order are part of the checkpoint contract.

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

<span id="anyres-глобальный-кадр-и-локальные-плитки"></span>

### AnyRes: restore detail without discarding the overview

LLaVA-NeXT/OneVision chooses among allowed grids of encoder-sized tiles. The low-resolution whole-image view preserves context; local tiles preserve detail. Encoded tile features are arranged back into their spatial grid before flattening, with feature interpolation when needed to enforce a token budget.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/multimodal/llava-onevision-anyres.png]]

*Stanford CS336, Lecture 17, AnyRes illustration based on [LLaVA-OneVision](https://arxiv.org/abs/2408.03326). Follow grid choice, independent views, and the assembled feature grid; each stage can preserve or lose different information.*

The overview cannot add fine resolution to a crop, but tells the model where that crop belongs. A large letter may be legible locally while its column heading is visible only globally. Increasing tile count helps only if layout markers and positions keep these sources distinguishable and later compression does not remove the regained detail.

### Selecting a canvas and reassembling its grid

[LLaVA-NeXT's `select_best_resolution`](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/bce12e479bc4dfee2b9c50c88137b01ff51bd483/llava/mm_utils.py#L119-L215) maximizes effective image area after fitting inside each candidate canvas. Effective area is capped by original image area: upscaling creates no new source pixels. Ties favor less padding. For a $768\times384$ source, candidate canvases $384\times384$, $768\times384$, and $768\times768$ preserve 73,728, 294,912, and 294,912 source-equivalent pixels. The second wins because it avoids the third canvas's padding, yielding two tiles in a row.

OneVision's SigLIP produces $27\times27=729$ features for a $384\times384$ view. Two tiles must be reassembled as a $27\times54$ spatial grid. Concatenating their flattened vectors would incorrectly place the entire left tile before the right one. The `unpad` branch removes padded features, optionally interpolates the grid, and adds an `image_newline` vector after each row. The overview is prepended separately.

For the exact-ratio example, 1458 local vectors, 27 row separators, and 729 overview vectors total 2214 positions. The paper's simplified $(ab+1)T$ expression gives 2187 because it omits separators. In the following index example, the first row must continue from patch 26 to 729, not 27:

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

The pinned [`anyres_max`/`unpad` branch](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/bce12e479bc4dfee2b9c50c88137b01ff51bd483/llava/model/llava_arch.py#L350-L408) computes $r=\sqrt{hw/(K\cdot27^2)}$, interpolates only if $r>1.1$, and rounds the new sides down. The overview and separators lie outside this area budget, so `K=9` is not a strict limit of $9\cdot729$ total positions. For the $27\times54$ grid and $K=1$, $r=\sqrt2$ gives a $19\times38$ grid: 722 local features plus 19 separators and 729 overview positions, or 1470 total. Both detailed tiles have already passed through the ViT. This reduces LLM work, not the number of images encoded. Other branches, such as `maxpool2x2`, implement different operations.

## Native dynamic resolution and compression

Qwen2-VL and Pixtral accept variable image sizes rather than selecting only from a crop grid. Pixtral uses RoPE-2D, row-break tokens, and block-diagonal masks for packed images. Qwen2.5-VL uses mostly windowed attention in the vision encoder with selected global layers. “Native resolution” still includes resizing to allowed multiples and `min_pixels`/`max_pixels` limits.

The LLM need not receive every encoder patch. A $2\times2$ merger quarters sequence length; attention pooling, Q-Former, and Perceiver Resampler offer other compression budgets. Evaluation must include OCR and small-object grounding, not only average VQA, because a useful average score can conceal the loss of fine detail.

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

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/multimodal/qwen2-vl-architecture.png]]

*Stanford CS336, Lecture 17, based on Wang et al., [Qwen2-VL, Figure 1](https://arxiv.org/abs/2409.12191). The figure shows variable visual-sequence length; the $256\to64$ calculation follows the paper's spatial merger, not an extra operation drawn in this overview.*

“Native” resolution still means a processor-constrained grid. Record the resized dimensions and allowed pixel range rather than assuming that every original pixel reaches the encoder unchanged.

For a concrete [`smart_resize` example](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L54-L85), set `factor=28`, `min_pixels=56*56`, and `max_pixels=1280*28*28`. An image of height 1000 and width 1600 first rounds to $1008\times1596$, exceeding the 1,003,520-pixel cap. The processor divides original sides by $\beta=\sqrt{1000\cdot1600/1{,}003{,}520}$ and rounds each down to a multiple of 28, producing $784\times1260$. The ViT sees $56\times90=5040$ patches and the LLM receives $28\times45=1260$ merged vectors. Discrete allowed dimensions mean the nominal 1280-position budget need not be filled exactly, and aspect ratio is preserved only approximately.

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

These are explicitly chosen processor parameters, not universal defaults. Qwen2-VL's patch embedding also spans two temporal samples. For a still image, the processor repeats the frame, giving `grid_t=1`, not twice as many visual tokens.

### Pixtral: packing images without cross-image leakage

![[00 Учебник/Assets/Figures/curated/resolution-review-2026-10/pixtral-vision-packing.png]]

*Agrawal et al., [Pixtral 12B, Figure 2](https://arxiv.org/abs/2410.07073v1), CC BY 4.0. The original figure shows separate image-attention regions and the subsequent row/image delimiters.*

Pixtral uses $16\times16$ patches and RoPE-2D inside its vision encoder. Images with 12 and four patches can be packed into one 16-position tensor, but a block-diagonal mask must isolate their attention. Otherwise an image's encoding would depend on unrelated batch neighbors. The two-layer projector changes feature width without reducing position count. [`PixtralProcessor` v4.57.1](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/models/pixtral/processing_pixtral.py#L166-L207) appends `[IMG_BREAK]` after each row, replacing the last one with `[IMG_END]`:

```text
[IMG] [IMG] [IMG] [IMG_BREAK]
[IMG] [IMG] [IMG] [IMG_END]
```

The $2\times3$ grid uses six visual vectors and two separators. A $3\times2$ grid has the same visual count but three separators. Total patch count alone does not specify row geometry.

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

This block-diagonal isolation belongs to the vision encoder's batch, not the multimodal conversation. Pixtral's language decoder uses ordinary one-dimensional RoPE and causal attention; later text can read preceding images.

### Equal output lengths can hide different compression

Averaging scalar groups $(1,0,0,1)$ and $(0,1,1,0)$ gives 0.5 in both cases, losing the distinction between their diagonal patterns. Channel concatenation retains all four values and their order until a subsequent projection; attention pooling learns content-dependent weighting. Molmo's mean query does not make its output an arithmetic mean. InternVL's permutation and Qwen's grouping before an MLP should not be treated as the same operation as averaging.

Q-Former and Perceiver Resampler instead read the grid with a fixed number of learned queries. Increasing image size can leave output length unchanged, forcing each latent to represent more content. A fixed output budget is not evidence that every small character survives. Compare OCR, small objects, spatial relations, and latency at several budgets with the same inputs. Measure vision-encoder work as well as LLM sequence length.

<span id="6-почему-одной-позиции-в-последовательности-недостаточно"></span>

## From one-dimensional order to image geometry

Flattening a grid into one sequence makes the end of one row adjacent to the start of the next. Qwen2-VL’s M-RoPE partitions rotary dimensions among time, height, and width. Text uses matching indices; images keep time fixed while height and width vary; video varies all three.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/qwen2vl-figure3-mrope.png]]

*Figure 3 from Wang et al., [Qwen2-VL](https://arxiv.org/abs/2409.12191), PDF p. 5. M-RoPE separates temporal, vertical, and horizontal positions.*

### Position IDs are not cache addresses

Suppose three text positions 0, 1, and 2 precede a merged image grid with two rows and three columns; position 2 is `<|vision_start|>`. Offset the image's coordinates by 3:

```text
top row:    (3,3,3)  (3,3,4)  (3,3,5)
bottom row: (3,4,3)  (3,4,4)  (3,4,5)
```

Following text begins at the largest coordinate plus one, 6. The next two positions, `<|vision_end|>` and a question token, receive $(6,6,6)$ and $(7,7,7)$. There are 11 sequence positions, but the next rotary index is 8. The KV cache still contains 11 positions; geometric coordinates do not replace cache length.

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

The difference −3 is what [`get_rope_index`](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L925-L1082) records as `mrope_position_deltas`. Subsequent generated text continues with this offset. Changing grid shape can change the offset without changing image count.

For one rotary pair and frequency $\omega$,

$$
R(p\omega)=\begin{pmatrix}\cos(p\omega)&-\sin(p\omega)\\\sin(p\omega)&\cos(p\omega)\end{pmatrix}.
$$

The rotated dot product is $q^\top R((p_k-p_q)\omega)k$. MRoPE assigns different pairs to differences in time, height, and width. Horizontal neighbors differ only in width coordinates; vertical neighbors differ in height. A shared coordinate shift leaves this dot product unchanged for fixed $q$ and $k$. That algebraic property does not make the entire image network translation-invariant:

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

Qwen2.5-VL scales temporal position IDs with physical time. [Qwen3-VL replaces that absolute-time alignment](https://arxiv.org/html/2511.21631v1#S2.SS3) with explicit textual timestamps before video temporal patches; it does not simply add timestamps to the same time-scaled-ID scheme. Interleaved M-RoPE remains: temporal, height, and width coordinates are distributed across rotary frequency ranges rather than assigned one contiguous band per axis. These are separate decisions about representing physical time and allocating positional frequencies. Neither repairs incorrect timestamps or a wrong patch grid.

Interleaving does not mean repeating `t,h,w` uniformly over every pair. For example, `mrope_section=[24,20,20]` in [Qwen3-VL's pinned implementation](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L297-L314) assigns 24 frequency pairs to time and 20 each to height and width. Pixtral's RoPE-2D instead alternates height and width within vision attention; it does not imply multidimensional rotary positions in that model's language decoder.

## Returning coordinates to the original image

A box predicted inside a tile must be offset by the crop origin, corrected for overlap and padding, and divided by resize scale. Preprocessing metadata must therefore travel with the tensor. Inverting coordinates does not restore pixels lost to downsampling or cropping; it only maps locations within the retained region. When both overview and crops contain the same object, the training format must unambiguously state which coordinate system the target uses.

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

Coordinates here are continuous rectangle boundaries, without the `+1` used by some discrete formats. If a crop clipped the object, inversion returns only the visible box, not the missing part. Map detections from different tiles into a common coordinate system before comparing IoU; equal tile-local numbers can refer to different places. Confidence alone does not establish that two boxes describe one object.

<span id="где-теряется-информация-crop-tiling-merge-и-фиксированный-budget"></span>

## Diagnose the stage that lost the evidence

Consider a wrong digit and a systematically shifted box. They need different interventions:

1. Inspect the prepared pixels. If the digit vanished during resize, raise the pixel budget or use a crop.
2. Compare encoder and post-merger representations through downstream probes or controlled ablations. More input pixels are ineffective if later compression discards their useful information.
3. Check row, crop, and source ordering. Correct local content can still be attached to the wrong location.
4. Render the inverse-transformed prediction on the original image. A consistent offset suggests incorrect metadata rather than failed visual recognition.
5. Repeat at several budgets and report quality together with TTFT and memory.

For ordinary photos, begin with a moderate budget and increase it where fine detail matters. Documents and interfaces often need larger budgets, but batching them with short photo requests creates latency trade-offs. A quality–token–latency curve is more useful than one “high-resolution” configuration.

Production logs should record original and resized dimensions, crop count, visual-token count, and preprocessing version. Without them, latency and grounding regressions cannot be explained.

Also retain padding, crop origins, ViT grid dimensions, post-merger length, full LLM input length, and the output coordinate system. A constant box offset suggests checking padding and crop origins; an error growing with distance from the corner suggests checking scale. These are hypotheses to verify against saved inputs and intermediate coordinates. A human being able to read a digit in the crop does not prove the encoder retained it; identifying a merger failure requires a controlled comparison with the same pixels and different compression.

Budget experiments should include photos, small text, tables, diagrams, and unusually wide or tall images, measuring each group's accuracy plus vision time, prefill, and response latency. A per-image cap is insufficient for ten-image requests: the combined images, text, and answer budget must fit together.

In [Qwen2.5-VL, §2.1.1 and Table 1](https://arxiv.org/abs/2502.13923v1), most vision layers use $112\times112$-pixel windows, or $8\times8$ patches; layers 7, 15, 23, and 31 retain full attention. A $32\times32$ grid has 16 windows of 64 patches, giving $16\cdot64^2=65{,}536$ pairs per windowed head versus $1024^2=1{,}048{,}576$ for full attention, excluding padding, special positions, projections, and FFNs. Global layers still connect distant image regions, so the complete ViT is not uniformly linear in grid size.

## Sources and next chapter

- [Molmo and PixMo](https://arxiv.org/abs/2409.17146), [InternVL 2.5](https://arxiv.org/abs/2412.05271), [Qwen2-VL](https://arxiv.org/abs/2409.12191), and [Pixtral 12B](https://arxiv.org/abs/2410.07073).
- Version-specific details: [Molmo Appendix A.1](https://arxiv.org/abs/2409.17146v2), [LLaVA-OneVision §3.2](https://arxiv.org/abs/2408.03326v1), [InternVL 2.5 §2.1, §3.1, §4.1](https://arxiv.org/abs/2412.05271v1), [Qwen2-VL §2.1](https://arxiv.org/abs/2409.12191v1), [Qwen2.5-VL §2.1](https://arxiv.org/abs/2502.13923v1), [Qwen3-VL §2.1–2.3](https://arxiv.org/abs/2511.21631v1), [Pixtral §2.1–2.3](https://arxiv.org/abs/2410.07073v1).
- Previous: [[en/00 Textbook/16 Multimodal Models/64a Connectors and fusion|Connectors and fusion]]. Next: [[en/00 Textbook/16 Multimodal Models/64c Training VLMs — alignment, instruction tuning, and data|Training VLMs]].
