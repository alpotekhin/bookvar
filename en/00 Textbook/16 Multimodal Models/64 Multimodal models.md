---
title: "64. From images to visual tokens: ViT and CLIP"
type: textbook-chapter
status: canonical
locale: en
translation_status: current
canonical_revision: "26464f4cb33c2b281c72e63e1870157e0a41a23807c4eca7c2b79f542000608d"
translation_reviewed: "2026-10-29"
translation_of: "00 Учебник/16 Multimodal Models/64 Мультимодальные модели.md"
last_updated: 2026-10-29
last_verified: 2026-09-15
primary_sources:
  - https://arxiv.org/abs/2010.11929
  - https://arxiv.org/abs/2103.00020
  - https://arxiv.org/abs/2303.15343
  - https://cs336.stanford.edu/
  - https://cs231n.stanford.edu/slides/2025/lecture_16.pdf
---

# From images to visual tokens: ViT and CLIP

A language model receives a sequence of discrete token indices. An image is a rectangular grid of intensity values. Before a Transformer can process both, the image must become a sequence of vectors. This conversion determines spatial resolution, visual-token count, and which details remain available to every later stage.

<span id="две-задачи-понять-чужую-модальность-и-породить-её"></span>

## Perception and generation need different representations

Turning an image into “tokens” can mean two different things. For visual question answering, continuous feature vectors are sufficient: they condition a text decoder. For image generation, the model also needs an output representation that can be decoded back into pixels. A retrieval embedding need not preserve every texture or be invertible; a generative representation must retain enough detail to reconstruct an image.

This module first follows the perception path—vision encoder, semantic training, connector, and text generation. Image output is developed separately in [[00 Учебник/16 Multimodal Models/64g Диффузионные и flow-модели изображений|diffusion and flow models]] and [[00 Учебник/16 Multimodal Models/64h Единая мультимодальная последовательность и Chameleon|discrete image codes and Chameleon]]. Those pages may currently use the explicitly labeled Russian fallback. A shared embedding space alone does not supply either generator.

<span id="1-изображение-как-тензор"></span>

## Images as sequences of patches

Write a color image as $X\in\mathbb R^{H\times W\times C}$. A pixel is too small to carry useful semantics on its own, while a single vector for the whole image discards location. Vision Transformer divides the image into $P\times P$ patches, flattens each patch, and maps it through a learned linear projection. The sequence length is

$$
N=\frac{H}{P}\frac{W}{P}.
$$

A $336\times336$ image with patch size 14 produces $24\times24=576$ patches. This count assumes dimensions divisible by the patch size after preprocessing. Position embeddings are added before the sequence enters a standard Transformer encoder; original ViT also prepends a classification token.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/vit-figure1-patch-sequence.png]]

*Figure 1 from Dosovitskiy et al., [An Image is Worth 16×16 Words](https://arxiv.org/abs/2010.11929), PDF p. 3. Patches are linearly projected and passed to an ordinary Transformer encoder.*

Self-attention makes every output patch contextual: its representation can contain information from the rest of the image. It cannot, however, recover information that preprocessing or a lossy projection has already removed. Patchification itself is a rearrangement and need not discard within-patch detail. If two characters have already merged into one blur, a later LLM can only guess them from context.

### Follow the tensor shapes

For ViT-Base/16, one RGB image has shape `[3,224,224]`. The $14\times14$ grid contains 196 patches, each with $16\cdot16\cdot3=768$ input numbers. A shared linear map produces width 768; adding `[CLS]` gives the encoder input `[197,768]`. The classification token is an extra position, not another image patch.

A smaller $32\times32$ RGB image makes the operation easier to trace. Four $16\times16$ patches become four vectors. For a row-major flattening,

$$
\begin{gathered}
p_{r,c}=\operatorname{vec}(X_{r:r+16,c:c+16,:})W+b,
\\
W\in\mathbb R^{768\times d}.
\end{gathered}
$$

The output is $[4,d]$: upper left, upper right, lower left, lower right. No vocabulary lookup assigns a discrete ID to a patch. A visual token here is simply one sequence position containing a continuous vector.

A convolution with `kernel_size=stride=16` and $d$ output channels performs this same projection. Each of its $d$ filters computes one coordinate from one non-overlapping patch. The operation is linear like a patch projection; global mixing happens later in self-attention.

### Patch indexing and the convolution equivalent

Consider an HWC tensor of shape `[4,4,3]` whose entries are numbered from 0
to 47. With $2\times2$ patches, the upper-left patch contains
`[0,1,2,3,4,5,12,13,14,15,16,17]`. Reshaping the entire image directly into
four rows would mix rectangular patch boundaries. We must expose the grid
axes, move them next to each other, then flatten each patch.

```python
import numpy as np

image = np.arange(48, dtype=float).reshape(4, 4, 3)
P, D = 2, 2
patches = (
    image.reshape(2, P, 2, P, 3)
    .transpose(0, 2, 1, 3, 4)
    .reshape(4, P * P * 3)
)
assert patches[0].tolist() == [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17]
W = np.arange(24, dtype=float).reshape(12, D) / 100
bias = np.array([-1., 1.])
projected = patches @ W + bias

kernel = W.reshape(P, P, 3, D).transpose(3, 2, 0, 1)
chw = image.transpose(2, 0, 1)
conv = []
for row in range(0, 4, P):
    for col in range(0, 4, P):
        local = chw[:, row:row + P, col:col + P]
        conv.append(np.einsum("ocij,cij->o", kernel, local) + bias)
assert np.allclose(projected, conv)
assert np.allclose(projected[0], [15.24, 18.26])
```

The result has shape `[4,2]`. Matching the C/H/W order and the filter
coefficient order makes the two calculations equal. The original
[CLIP VisionTransformer](https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/clip/model.py#L206)
uses an input `Conv2d` without bias; this example includes the general affine
case. A linear map from 12 coordinates to two has a nontrivial nullspace:
some changes in the pixels leave its output unchanged. The rearrangement
into patches is invertible; that low-dimensional projection is not.

### Positions, CLS and the encoder block

The learned CLS vector is an extra sequence position, not a patch of zero
pixels. Without position information, full self-attention with dropout
disabled is permutation-equivariant: $f(\Pi Z)=\Pi f(Z)$. If a permutation
leaves CLS in place, its output cannot distinguish the original arrangement
from the reordered patches. The following small attention operation checks
this property and then fixes positional vectors to locations.

```python
import numpy as np

def attend(z):
    scores = z @ z.T / np.sqrt(z.shape[1])
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ z

z = np.array([[.2, .1], [1, 0], [0, 1], [.5, .5], [-.5, .2]])
positions = np.array([[0, 0], [.2, 0], [0, .3], [-.2, 0], [0, -.3]])
permutation = [0, 2, 1, 3, 4]
assert np.allclose(attend(z[permutation]), attend(z)[permutation])
assert np.allclose(attend(z)[0], [.27125546, .36137529])
assert np.allclose(attend(z)[0], attend(z[permutation])[0])
original = attend(z + positions)[0]
swapped = attend(z[permutation] + positions)[0]
assert not np.allclose(original, swapped)
```

Original ViT stores learned positions in a one-dimensional table whose
entries correspond to the two-dimensional patch grid. For a new resolution,
grid positions are interpolated spatially and CLS is handled separately
([ViT §3.2](https://arxiv.org/abs/2010.11929v2)). The table's storage layout
does not restrict the model to one-dimensional visual relationships.

The image encoder uses full attention, not a causal mask: the whole image
is available at once. Each pre-normalized block computes

$$
\begin{gathered}
U=Z+\operatorname{MSA}(\operatorname{LN}(Z)),\\
Z'=U+\operatorname{MLP}(\operatorname{LN}(U)).
\end{gathered}
$$

MSA mixes positions. The MLP expands each vector, applies GELU and projects
back to width $D$, using the same weights at every position. Both branches
return `[N+1,D]`, so the residual additions preserve shape. After $L$ blocks,
normalized CLS goes to the ViT classifier or, in CLIP's ViT variant, to the
image–text comparison projection. Local patch states are already contextual
after the first attention layer.

## Resolution is a compute decision

Patch count grows with area. Doubling both sides creates four times as many tokens; full attention inside the vision encoder grows faster still. Large patches are often sufficient for scene classification, but small print, chart marks, and interface controls require much finer sampling. This is why modern VLMs use native resolution, tiling, and token compression rather than treating input size as an incidental preprocessing choice.

With 576 patches, full vision self-attention has $576^2=331\,776$ patch-pair scores per head. Doubling each side produces 2,304 patches and $2\,304^2=5\,308\,416$ pairs: sixteen times as many. These are illustrative pair counts excluding special tokens, not measured runtime or an assertion that every VLM uses global attention.

For $224\times224$ and $448\times448$ with $P=16$, the patch counts are
196 and 784, or 197 and 785 positions including CLS. Projection and MLP
work grows linearly with sequence length at fixed width. Efficient attention
kernels need not materialize every score, although the full-attention pair
arithmetic still grows quadratically. Increasing token count after small
print has disappeared during preprocessing cannot reconstruct the lost print.

Original [CLIP preprocessing](https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/clip/clip.py#L79)
resizes the short side, takes a center crop, converts to RGB and normalizes
channels. The target comes from `model.input_resolution`; 336 belongs to
ViT-L/14@336px, not every CLIP model. A 640-wide, 480-high image resized to
a short side of 336 becomes $448\times336$. The central crop removes 56
columns at each side, about 80 original pixels per side. A label near the
edge may vanish before encoding. Resizing reduces sampling density, whereas
cropping removes complete regions. Resolution, tiling and token compression
must therefore be evaluated at each stage, not inferred from the input file.

<span id="3-от-классификации-к-общему-пространству-изображения-и-текста"></span>

## CLIP replaces a fixed classifier with a text encoder

A conventional image classifier predicts one of a fixed set of labels. CLIP instead trains an image encoder and a text encoder to produce normalized vectors in one space. For a batch of $B$ matched pairs, it computes

$$
S_{ij}=\frac{v_i^\top t_j}{\tau}.
$$

and applies symmetric cross-entropy over image-to-text and text-to-image directions. Correct pairs lie on the diagonal; all other pairs in the batch act as negatives.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/vlm-2026/clip-figure1-training-and-zero-shot.png]]

*Figure 1 from Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), PDF p. 2. The left half shows contrastive batch training; the right half shows text-derived zero-shot classifier weights.*

After training, prompts such as `a photo of a dog` are encoded into class vectors. A new image is assigned to the nearest class vector without fitting a classifier on the target dataset. Prompt wording matters because bare labels can be ambiguous and differ from natural captions. CLIP therefore ensembles several prompt templates.

The same independent encoders support bidirectional retrieval. Image and text vectors can be computed and cached separately, which is valuable at collection scale.

The image and text encoders may have different native widths; separate
learned projections map them to the common comparison dimension. Equal
dimensions alone do not make two spaces comparable. Original CLIP trained
on WIT, 400 million web image–text pairs gathered using 500,000 queries,
with per-query limits to reduce frequency imbalance
([CLIP §2.1](https://proceedings.mlr.press/v139/radford21a.html)). A caption
may name a dog without specifying its collar or position: the training
signal is limited by what the text describes.

CLIP does not require a ViT: the paper also uses modified ResNets with
attention pooling. Its text Transformer uses a causal mask, and the final
EOT state supplies the text representation after normalization and
projection. It is neither mean pooling nor BERT-style CLS pooling. Released
original models use a 77-token text window including special positions.
This causal text encoder learns the contrastive task, not next-token loss.
The functions `encode_image` and `encode_text` return features that
[`forward` subsequently normalizes](https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/clip/model.py#L340);
callers using those functions independently must normalize explicitly.

### Two directions of the same loss

The image-to-text half normalizes each row; the text-to-image half normalizes each column:

$$
\begin{aligned}
\mathcal L_{\mathrm{CLIP}}=-\frac{1}{2B}\Bigg[&
\sum_i\log\frac{e^{S_{ii}}}{\sum_j e^{S_{ij}}}\\
&+\sum_j\log\frac{e^{S_{jj}}}{\sum_i e^{S_{ij}}}
\Bigg].
\end{aligned}
$$

For three matched pairs, consider

$$
S=\begin{pmatrix}
8.1&2.0&-0.4\\
1.7&7.3&0.9\\
-0.2&1.1&6.8
\end{pmatrix}.
$$

Row one compares captions for image one; column one compares images for caption one. Raising $S_{12}$ close to $S_{11}$ makes the first image's choice uncertain even if the diagonal remains largest. Batch composition therefore defines which competing descriptions the model must distinguish, not just how many examples contribute to a gradient.

The scale is learned as $a=\exp(\gamma)=1/\tau$. Unit-vector dot products
alone lie between −1 and 1, limiting the concentration of a softmax over
many captions. For the displayed matrix, the first row loss is approximately
0.002443 and the first column loss 0.001908: their competitors differ.

```python
import numpy as np

def logsumexp(x, axis):
    maximum = x.max(axis=axis, keepdims=True)
    return np.squeeze(
        maximum + np.log(np.exp(x - maximum).sum(axis=axis, keepdims=True)),
        axis=axis,
    )

def clip_loss(scores):
    scores = np.asarray(scores, dtype=float)
    if (scores.ndim != 2 or scores.shape[0] != scores.shape[1]
            or scores.shape[0] == 0 or not np.isfinite(scores).all()):
        raise ValueError("finite nonempty square matrix required")
    row = logsumexp(scores, 1) - np.diag(scores)
    col = logsumexp(scores, 0) - np.diag(scores)
    return (row.mean() + col.mean()) / 2

scores = np.array([[8.1, 2., -.4], [1.7, 7.3, .9], [-.2, 1.1, 6.8]])
loss = clip_loss(scores)
assert np.isclose(loss, .004070330028012345)
wrong = scores.copy()
wrong[0, 1] = 8.0
assert clip_loss(wrong) > loss
assert np.isclose(clip_loss(scores + 1000), loss)
assert np.isclose(clip_loss(np.zeros((2, 2))), np.log(2))
```

An off-diagonal pair is labeled negative because it was not an observed
pair, not because someone proved its caption false. Two dog photographs
with the same caption give identical text vectors while the diagonal target
asks the model to distinguish their indices. With just those two captions,
each image-to-text probability is $1/2$, so this directional loss cannot
reach zero. Larger batches supply both difficult alternatives and false
negatives. Adding a common constant to logits changes neither softmax nor
loss; changing an individual competitor affects both directions.

An image may also have several correct captions describing different
details. Caption generation predicts one particular word sequence, whereas
contrastive training distinguishes the observed caption from batch rivals.

![[00 Учебник/Assets/Figures/curated/vision-encoders-review-2026-10/clip-objectives.png]]

*Radford et al., [CLIP Figure 2](https://proceedings.mlr.press/v139/radford21a.html),
CC BY 4.0. The horizontal axis counts processed images and the vertical axis
measures zero-shot ImageNet accuracy in the 2021 experiment. Contrastive
training reaches the measured classification accuracy with fewer examples;
this does not establish superior caption quality or the same ratio in
FLOPs, training time or a new architecture.*

### Constructing class vectors and interpreting probabilities

The prompt `crane` may describe a bird or a machine; `boxer` may describe a
dog or a person. CLIP §3.1.4 uses context and prompt ensembles. Normalize
each template embedding, average them for one class and normalize that
average again. The resulting class vector can be cached. This differs from
averaging several classifiers' softmax probabilities.

```python
import numpy as np

def unit(x):
    x = np.asarray(x, dtype=float)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if not np.isfinite(x).all() or np.any(norm == 0):
        raise ValueError("finite nonzero vectors required")
    return x / norm

def class_probabilities(image, labels, scale):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("positive finite scale required")
    logits = scale * (unit(labels) @ unit(image))
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()

image = np.array([1., 0.])
labels = np.array([[.8, .6], [.6, .8], [-.8, .6]])
before = class_probabilities(image, labels, 5)
assert np.isclose(before[0], .730879336)
extra = np.array([[.9, np.sqrt(.19)]])
after = class_probabilities(image, np.concatenate([labels, extra]), 5)
assert after.argmax() == 3
assert np.isclose(after[0], .331462100)
templates = unit([[.8, .6], [1., 0.]])
prototype = unit(templates.mean(axis=0))
assert np.allclose(prototype, [.948683298, .316227766])
```

The three initial cosine scores are 0.8, 0.6 and −0.8. Adding a fourth class
reduces the first class's probability from 0.731 to 0.331 even though neither
the image nor any old cosine score changed. These probabilities depend on
the candidates, scale and prompts; 0.731 is not a universally calibrated
probability that the picture depicts a cat.

Zero-shot constructs weights from text without training on labeled examples
of the target task. A linear probe trains a new head using such examples
while freezing the encoder. Fine-tuning also changes encoder weights. These
protocols have different data access and should not share an unlabeled
“CLIP accuracy” result.

![[00 Учебник/Assets/Figures/curated/vision-encoders-review-2026-10/clip-zero-shot-linear-probe.png]]

*Radford et al., [CLIP Figure 5](https://proceedings.mlr.press/v139/radford21a.html),
CC BY 4.0. The star is zero-shot CLIP; curves train classifiers with different
numbers of examples per class. Averaged over the experiment's 20 datasets,
zero-shot resembles a four-shot linear classifier on the same features.
More labels eventually help the trained head; the plot does not establish
that avoiding labels is generally better.*

Save templates, class names/order, preprocessing, model/checkpoint and
normalization with the result. Selecting prompts on the test set uses test
information even if no gradient is computed there.

## What the contrastive objective preserves

The objective asks for matched pairs to be close; it does not require the final global vector to preserve every spatial relation. A model may recognize a horse and grass while confusing “a horse eats grass” with “grass eats a horse.” Strong retrieval is therefore not evidence of grounded relation understanding, OCR, counting, or text generation.

Generative VLMs usually take patch-level features before final pooling. Those retain more local structure than a global CLIP embedding. LLaVA, for example, projects a grid of CLIP ViT features into the language model’s embedding dimension.

Patch features have shape $[N,D_v]$, whereas the pooled comparison vector
has shape $[d_e]$. Multiple positions permit location-specific access but
do not guarantee exact counting or OCR. Layer selection, normalization,
CLS removal and projection belong to the particular model interface.
There is no universal rule that the penultimate CLIP layer is best, or that
encoders of the same width can be substituted without training. If captions
mostly distinguish object categories, the loss need not strongly reward
precise object counts or spatial relations.

<span id="clip-и-siglip-две-формы-контрастивного-обучения"></span>

## SigLIP: pairwise decisions instead of a shared softmax

[SigLIP](https://arxiv.org/abs/2303.15343) changes how pair similarities become a loss. Let $y_{ij}=+1$ for matched image–text pairs and $-1$ otherwise. With learned scale $a$ and bias $b$, the paper uses

$$
\begin{aligned}
\mathcal L_{\mathrm{SigLIP}}={}&-\frac1B\sum_{i=1}^{B}\sum_{j=1}^{B}\\
&\log\sigma\!\left(y_{ij}(a\,v_i^\top t_j+b)\right).
\end{aligned}
$$

This is normalization by $B$, as in the paper, not the mean over $B^2$ binary decisions. Changing that normalization changes gradient scale as batch size changes. The bias helps accommodate $B$ positive pairs versus $B(B-1)$ negatives.

The missing softmax denominator does not mean missing negatives. Each off-diagonal pair still contributes a negative binary loss, but pair blocks can be evaluated without a shared row denominator. Distributed implementations still communicate whenever remote features are required.

The original Algorithm 1 retains this sign and normalization:

```text
logits = dot(zimg, ztxt.T) * t + b
labels = 2 * eye(n) - ones(n) # -1 with diagonal 1
l = -sum(log_sigmoid(labels * logits)) / n
```

*Zhai et al., [SigLIP Algorithm 1](https://arxiv.org/abs/2303.15343v4),
CC BY 4.0; original English pseudocode. The
[pinned reference implementation](https://github.com/google-research/big_vision/blob/0127fb6b337ee2a27bf4e54dea79cff176527356/big_vision/trainers/proj/image_text/siglip.py#L287)
also adds the bias before the sigmoid.*

For logits $Z=\left(\begin{smallmatrix}2&-1\\-2&1\end{smallmatrix}\right)$,
the positive pair with logit 2 contributes $\log(1+e^{-2})=0.126928$;
the negative pair with logit −1 contributes $\log(1+e^{-1})=0.313262$.
The other terms repeat those values. Their sum divided by two is 0.440190.

```python
import numpy as np

def siglip_loss(logits):
    logits = np.asarray(logits, dtype=float)
    if (logits.ndim != 2 or logits.shape[0] != logits.shape[1]
            or logits.shape[0] == 0 or not np.isfinite(logits).all()):
        raise ValueError("finite nonempty square matrix required")
    B = len(logits)
    labels = 2 * np.eye(B) - 1
    return np.logaddexp(0, -labels * logits).sum() / B

logits = np.array([[2., -1.], [-2., 1.]])
assert np.isclose(siglip_loss(logits), .4401896985611954)
B = 3
neutral = np.zeros((B, B))
prior_bias = -np.log(B - 1)
assert siglip_loss(neutral + prior_bias) < siglip_loss(neutral)
labels = 2 * np.eye(B) - 1
terms = np.logaddexp(0, -labels * (neutral + prior_bias))
blocks = [np.arange(2), np.arange(2, B)]
chunked = sum(terms[np.ix_(r, c)].sum() for r in blocks for c in blocks) / B
assert np.isclose(chunked, siglip_loss(neutral + prior_bias))
```

There are $B$ positives and $B(B-1)$ negatives. If dot products are initially
zero and only the bias varies, its optimum satisfies $\sigma(b)=1/B$,
giving $b=-\log(B-1)$ for $B>1$. This is about −0.693 for $B=3$ and −10.4
for approximately 32,000 pairs. The paper initializes scale at 10 and bias
at −10. This adjusts the initial training-class prior; it does not calibrate
arbitrary future image–caption pairs. Dividing by $B^2$ instead of $B$
would shrink every gradient by $B$, requiring a corresponding optimization
choice. Comparing the raw CLIP and SigLIP loss values is not a comparison
of model quality.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/multimodal/siglip-parallelism.png]]

*Stanford CS336, Lecture 17, illustration of distributed pair blocks based on Zhai et al., [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343). Follow one local block; independent pair losses simplify accumulation, not eliminate all communication.*

This changes the optimization and communication problem, not the meaning of a visual token. Compare losses with the encoder, data, hardware, and training budget controlled; training-day figures on different accelerators are not an isolated speedup.

The diagram has 12 matched pairs across three devices, four per device:
144 pair comparisons. The first local $4\times4$ blocks cover 48 comparisons;
two exchanges of text features supply the remaining 96. Blocking reduces
simultaneously retained logits, not the number of comparisons. The §3.3
schedule avoids a full all-gather by exchanging feature blocks; other
implementations may still use all-gather. For CLIP, independent block-wise
cross-entropies cannot replace the global loss: row/column normalization
sums and gradients must be accumulated consistently across all candidates.

![[00 Учебник/Assets/Figures/curated/vision-encoders-review-2026-10/siglip-batch-size.png]]

*Zhai et al., [SigLIP Figure 2, middle panel](https://arxiv.org/abs/2303.15343v4),
CC BY 4.0. Batch size in thousands versus zero-shot ImageNet accuracy after
9 billion processed examples. This is the SigLIP experiment, not the adjacent
SigLiT or multilingual variants. Very large batches fail to improve either
loss variant here. Batch size still changes negative comparisons and the
number of updates for a fixed example budget. SigLIP also retains false
negatives when interchangeable captions receive opposing labels.*

## Dual encoders and generative VLMs are different systems

Dual encoders produce a shared representation and excel at search, ranking, and zero-shot classification. A generative VLM conditions an autoregressive decoder on visual information. It needs an interface between the vision encoder and the LLM: a projector, Q-Former, Perceiver Resampler, or cross-attention.

The distinction prevents misleading comparisons. Better contrastive retrieval may not improve document OCR. A capable dialogue model may be inefficient for billion-image search. Increasing visual-token count may improve small-text accuracy while making time to first token much worse.

| System | Output | Interaction | Main limitation |
|---|---|---|---|
| Dual encoder | independent image and text vectors | scalar similarity | global pooling can hide fine structure |
| Cross-encoder | pair score or task output | joint image–text processing | pair-specific interaction cannot be cached as two independent final vectors |
| Generative VLM | next-token probabilities | visual memory conditions a decoder | context cost and possible unsupported generation |

## Diagnostic tests

Evaluation should alter the evidence. Occluding the answer region should change the prediction. Swapping object relations should change compositional answers. Reducing font size should produce a measurable degradation curve. For CLIP-like models, retrieval, zero-shot classification, linear probing, and distribution-shift robustness answer different questions and should be reported separately.

### Winoground checks both matching directions

[Winoground §3.2](https://openaccess.thecvf.com/content/CVPR2022/html/Thrush_Winoground_Probing_Vision_and_Language_Models_for_Visio-Linguistic_Compositionality_CVPR_2022_paper.html)
uses two images and two captions containing the same words in different
relations, such as a person leading a horse versus a horse leading a person.
All four pairings are scored. Text success requires the correct caption
for both images; image success requires the correct image for both captions;
group success requires both. Strict inequalities make ties failures.

```python
import numpy as np

def paired_scores(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (2, 2) or not np.isfinite(scores).all():
        raise ValueError("finite 2-by-2 scores required")
    text = scores[0, 0] > scores[0, 1] and scores[1, 1] > scores[1, 0]
    image = scores[0, 0] > scores[1, 0] and scores[1, 1] > scores[0, 1]
    return bool(text), bool(image), bool(text and image)

assert paired_scores([[3, 2], [1, 2.5]]) == (True, True, True)
assert paired_scores([[3, 2], [4, 5]]) == (True, False, False)
assert paired_scores(np.ones((2, 2))) == (False, False, False)
```

In the second matrix both rows prefer the correct caption, yet caption one
scores higher for image two. Reporting only text success conceals that
failure. These numbers illustrate the metric, not measured CLIP performance.
Likewise, masking a document cell does not test reading if surrounding
context reveals its value. Choose examples whose answer depends on the
hidden region, and inspect preprocessing, encoder grid, selected layer and
connector input before attributing the error to the language decoder.

## Sources and next chapter

- Stanford CS231n, [Multimodal Foundation Models, Lecture 16](https://cs231n.stanford.edu/slides/2025/lecture_16.pdf), PDF pp. 13–55.
- Dosovitskiy et al., [An Image is Worth 16×16 Words](https://arxiv.org/abs/2010.11929).
- Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020).
- Next: [[en/00 Textbook/16 Multimodal Models/64a Connectors and fusion|Connectors and fusion]].
