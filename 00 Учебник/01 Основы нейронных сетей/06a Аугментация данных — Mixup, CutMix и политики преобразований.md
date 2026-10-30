---
title: Аугментация данных — Mixup, CutMix и политики преобразований
type: textbook-chapter
status: canonical
source_language: en
last_updated: 2026-10-28
last_verified: 2026-10-28
primary_sources:
  - https://d2l.ai/chapter_computer-vision/image-augmentation.html
  - https://arxiv.org/html/1710.09412v2
  - https://arxiv.org/html/1905.04899v1
  - https://github.com/clovaai/CutMix-PyTorch/tree/2d8eb68faff7fe4962776ad51d175c3b01a25734
  - https://arxiv.org/html/1805.09501v3
  - https://arxiv.org/html/1909.13719v2
  - https://arxiv.org/html/1511.06709v4
  - https://github.com/harvard-edge/cs249r_book/blob/2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd/book/quarto/contents/vol1/data_selection/data_selection.qmd
---

# Аугментация данных — Mixup, CutMix и политики преобразований

A photograph of a cat can be cropped or reflected without necessarily changing
the answer “cat.” A left-turn arrow cannot be reflected while keeping the
answer “left.” Data augmentation constructs training examples by transforming
existing ones. Choosing a transformation includes deciding what should happen
to the target.

Some transformations keep a class label, some transform an annotation along
with the input, and some construct a mixture of targets. Mixup and CutMix belong
to the last group. They do not require a pasted or blended image to have the
unchanged label of its first parent.

The examples use small tensors so that coordinates, target weights, losses,
and gradients can be inspected. They do not measure recognition quality.
The image galleries and comparison below are the authors' original figures,
not new augmentation results generated for this book.

<a id="augmentation-target"></a>
## Transform the input and its target together

For classification, a label-preserving transformation asks for
$f(T(x))$ to predict the same class as $f(x)$. In object detection, the target
contains coordinates. Reflecting an image must also reflect its boxes.
For segmentation, the class mask must undergo the same spatial transformation.
Image and annotation cannot draw their geometric random choices independently.

Use half-open pixel coordinates: a box $(x_1,y_1,x_2,y_2)$ covers columns
$x_1,\ldots,x_2-1$ and rows $y_1,\ldots,y_2-1$. In an image of width $W$,
horizontal reflection gives

$$
\begin{aligned}
x'_1&=W-x_2, &x'_2&=W-x_1,\\
y'_1&=y_1,   &y'_2&=y_2.
\end{aligned}
$$

For $W=6$, the box $(1,1,4,3)$ becomes $(2,1,5,3)$. Its width remains three
pixels. Using $W-1-x_2$ would mix the half-open box convention with the
formula for reflecting a single pixel index.

```python
import torch

image = torch.arange(24).reshape(4, 6)
height, width = image.shape
box = (1, 1, 4, 3)
x1, y1, x2, y2 = box
mask = torch.zeros_like(image, dtype=torch.bool)
mask[y1:y2, x1:x2] = True

reflected = image.flip(-1)
new_box = (width - x2, y1, width - x1, y2)
nx1, ny1, nx2, ny2 = new_box
new_mask = torch.zeros_like(mask)
new_mask[ny1:ny2, nx1:nx2] = True
assert new_box == (2, 1, 5, 3)
assert torch.equal(new_mask, mask.flip(-1))
assert torch.equal(reflected.flip(-1), image)
assert new_mask.sum().item() == mask.sum().item() == 6
print("box", box, "->", new_box)
```

After cropping, boxes must also be translated into crop coordinates and clipped
to the retained region. Some objects disappear entirely. A classification crop
can similarly remove the object that justified the original label. A policy
must decide which crops are admissible, rather than treating every valid array
shape as a valid labelled example.

### Crop size, scale, and interpolation

> Most image augmentation methods have a certain degree of randomness. To make it easier for us to observe the effect of image augmentation, next we define an auxiliary function `apply`. This function runs the image augmentation method `aug` multiple times on the input image `img` and shows all the results.

[D2L, Common Image Augmentation Methods](https://d2l.ai/chapter_computer-vision/image-augmentation.html#common-image-augmentation-methods),
CC BY-SA 4.0. Original explanation and the following PyTorch excerpts;
`img` is the book's input photograph and `d2l` its plotting helper.

```python title="D2L source excerpt: repeated sampling and random resized crops"
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)

shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/data-augmentation-2026-10-28/d2l-random-crops.svg|Original D2L gallery of random resized crops]]

*Aston Zhang, Zachary C. Lipton, Mu Li and Alexander J. Smola,
[D2L, Image Augmentation: Flipping and Cropping](https://d2l.ai/chapter_computer-vision/image-augmentation.html#flipping-and-cropping),
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Complete original gallery, unchanged. The displayed crop variants are saved
book outputs, not an execution of the local tensor examples.*

D2L samples a crop and resizes it to a fixed output shape. Its demonstration
uses output $200\times200$, source-area fractions from 0.1 to 1, and aspect
ratios from 0.5 to 2. Area fraction is not side-length fraction. A square crop
covering one quarter of the source area has half its side length.
Resizing then changes the object's scale relative to the output image.

The same output shape can therefore contain a whole object, a close-up,
or background. Sampling parameters, rejection/fallback rules, and interpolation
belong to the transformation definition. For a categorical segmentation mask,
ordinary bilinear interpolation can invent intermediate class indices;
the annotation needs a label-appropriate interpolation rule.

Colour changes also depend on the task. Changing brightness may simulate
illumination variation; replacing the red of a traffic signal with green can
change its meaning. Normalization is a deterministic change of representation,
not automatically another source of useful task variation.

<a id="augmentation-objective"></a>
## The training objective averages transformed examples

Let $q(\tau)$ describe the random choices of an augmentation. A transformation
can act on the whole labelled pair, giving
$(\widetilde x_i,\widetilde y_i)=A_\tau(x_i,y_i)$. The augmented objective is

$$
\begin{aligned}
L_{\mathrm{aug}}(\theta)
&=\frac1n\sum_{i=1}^n
  \mathbb E_{\tau\sim q}\big[
     \ell(f_\theta(\widetilde x_i),\widetilde y_i)
  \big].
\end{aligned}
$$

Sampling a new transformation when an example is visited estimates this
expectation over training. Saving ten variants once instead fixes a finite
set of variants. Both reuse the original observation; neither collects ten
independent objects, patients, speakers, or documents.

The loss of an averaged prediction generally differs from the average loss
used during augmentation. For a linear predictor with weights $(1,0)$,
the input $(1,3)$ and its reflection $(3,1)$ produce predictions 1 and 3.
If the target is 2, each squared error is 1. Averaging the predictions first
would give 2 and zero error, but that is a different computation.

```python
import torch

views = torch.tensor([[1., 3.], [3., 1.]], dtype=torch.float64)
weights = torch.tensor([1., 0.], dtype=torch.float64, requires_grad=True)
target = 2.
predictions = views @ weights
mean_loss = (predictions - target).square().mean()
loss_of_mean = (predictions.mean() - target).square()
gradient, = torch.autograd.grad(mean_loss, weights)
assert mean_loss.item() == 1. and loss_of_mean.item() == 0.
torch.testing.assert_close(gradient, torch.tensor([2., -2.], dtype=weights.dtype))
balanced_weights = torch.tensor([.5, .5], dtype=weights.dtype)
assert torch.equal(views @ balanced_weights, torch.tensor([2., 2.], dtype=weights.dtype))
print("average loss", mean_loss.item(), "loss of average", loss_of_mean.item())
print("gradient", gradient)
```

Here training on both views can favour equal treatment of the two input
positions. The example establishes the objective and its gradient, not that
this symmetry is appropriate for every task. If reflection changes the answer,
forcing the same target introduces contradictory supervision.

<a id="augmentation-views"></a>
## Occlusion and photometric changes

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/simclr-2026-10-28/augmentations.png|Original SimCLR gallery of crop, colour, blur and masking operations]]

*Ting Chen, Simon Kornblith, Mohammad Norouzi and Geoffrey Hinton,
[SimCLR, Figure 4](https://proceedings.mlr.press/v119/chen20j.html),
CC BY 4.0. The original photograph is credited to Von.grzanka under CC-BY.
The complete figure is retained. It lists operations considered in the paper's
ablation, not a requirement to apply every operation to each training image.*

The figure separates changes to image geometry from changes to colour,
sharpness, and visible regions. Each operation changes which cues remain
available. A random colour perturbation can make colour histograms less
reliable as a shortcut. It does not prove that every remaining cue describes
the object; backgrounds and acquisition artefacts can still be shared.

Cutout removes a contiguous input region while retaining the original class
target. It asks a classifier to use evidence outside the removed region.
This is different from independently dropping hidden activations: the location,
correlation, and stage of the mask differ.

The fill value has units. For a normalized channel
$z=(x-\mu)/\sigma$, setting $z$ to zero inserts the raw-space mean $\mu$.
Setting raw $x$ to zero instead gives $z=-\mu/\sigma$.
“Fill with zero” does not always mean a black patch.

```python
import torch

raw = torch.ones(1, 4, 4, dtype=torch.float64)
mask = torch.ones_like(raw)
mask[:, 1:3, 1:3] = 0
masked_raw = raw * mask
assert (masked_raw == 0).sum().item() == 4

mean, std = .4, .2
normalized = (raw - mean) / std
fill_after_normalization = normalized * mask
fill_before_normalization = (masked_raw - mean) / std
hole = mask == 0
assert torch.equal(fill_after_normalization[hole], torch.zeros(4, dtype=raw.dtype))
assert torch.equal(fill_before_normalization[hole], torch.full((4,), -2., dtype=raw.dtype))
assert mean + std * fill_after_normalization[hole][0].item() == mean
print("normalized values in hole:", fill_after_normalization[hole],
      fill_before_normalization[hole])
```

An occlusion can remove all evidence for the class. Its size and placement
must be evaluated, not justified by the fact that the target tensor still
contains a valid class index. A larger erased fraction need not improve
generalization.

<a id="mixup"></a>
## Mixup interpolates inputs and targets

Mixup draws two labelled examples and a coefficient $\lambda\in[0,1]$:

$$
\begin{aligned}
\widetilde x&=\lambda x_A+(1-\lambda)x_B,\\
\widetilde y&=\lambda y_A+(1-\lambda)y_B.
\end{aligned}
$$

For classification, $y_A$ and $y_B$ are one-hot vectors. Mixing class 0 and
class 1 with $\lambda=0.25$ gives target $(0.25,0.75)$.
The target expresses the training rule for the interpolated input. It is
not a verified measurement that a physical scene contains exactly those
object proportions.

Zhang et al., [mixup, §2](https://arxiv.org/html/1710.09412v2#S2), formulate
this as a vicinal training distribution. Instead of evaluating loss only
at stored examples, training includes points interpolated between examples.
The rule encourages corresponding interpolation of predictions between
parents. It does not make every mixed image a natural photograph.

For probabilities $p=\operatorname{softmax}(z)$ and a probability target $q$,
cross-entropy is $-\sum_c q_c\log p_c$. It is linear in the target:

$$
\begin{aligned}
\ell(z,\widetilde y)
&=\lambda\ell(z,y_A)\\
&\quad+(1-\lambda)\ell(z,y_B).
\end{aligned}
$$

Both losses use the same logits for the mixed input. Running the network
separately on $x_A$ and $x_B$ and combining those losses does not implement
the same update.

Suppose the mixed input produces probabilities $(0.8,0.2)$. With target
$(0.25,0.75)$, the loss is about 1.2629 and its logit gradient is
$(0.8,0.2)-(0.25,0.75)=(0.55,-0.55)$.
Taking the target's argmax would replace it with class 1, change the loss
to about 1.6094, and discard the mixture weight.

```python
import torch
import torch.nn.functional as F

lam = .25
x_a = torch.tensor([0., 2.], dtype=torch.float64)
x_b = torch.tensor([4., 6.], dtype=torch.float64)
mixed_input = lam * x_a + (1 - lam) * x_b
q = torch.tensor([lam, 1 - lam], dtype=torch.float64)
logits = torch.tensor([.8, .2], dtype=torch.float64).log().requires_grad_()
soft_loss = -(q * logits.log_softmax(-1)).sum()
weighted_hard_loss = (lam * F.cross_entropy(logits[None], torch.tensor([0]))
                      + (1 - lam) * F.cross_entropy(logits[None], torch.tensor([1])))
gradient, = torch.autograd.grad(soft_loss, logits)
torch.testing.assert_close(soft_loss, weighted_hard_loss)
torch.testing.assert_close(gradient, torch.tensor([.55, -.55], dtype=logits.dtype))
torch.testing.assert_close(mixed_input, torch.tensor([3., 5.], dtype=x_a.dtype))
hard_loss = F.cross_entropy(logits[None], q.argmax()[None])
assert hard_loss > soft_loss
print("input", mixed_input, "target", q)
print("soft loss", soft_loss.item(), "argmax loss", hard_loss.item())
```

The paper samples $\lambda\sim\operatorname{Beta}(\alpha,\alpha)$.
At $\alpha=1$ the distribution is uniform. Smaller positive $\alpha$ places
more mass near zero and one; larger $\alpha$ concentrates nearer one half.
The coefficient's mean is always one half, while its variance is
$1/[4(2\alpha+1)]$. Increasing $\alpha$ changes how often strongly blended
examples are presented; it does not change the average coefficient.

A random permutation of a minibatch supplies partners without a second
data loader. A permutation can contain fixed points, so some examples may
be paired with themselves. Whether coefficients are shared by a batch or
sampled per example is another implementation choice.
Inputs must already share compatible shapes and numerical units.

<a id="cutmix"></a>
## CutMix uses a patch and measures its actual area

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/data-augmentation-2026-10-28/cutmix-comparison.png|Original NAVER comparison of Mixup, Cutout and CutMix inputs, labels and reported results]]

*Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe and
Youngjoon Yoo, [official CutMix repository, original overview](https://github.com/clovaai/CutMix-PyTorch/blob/2d8eb68faff7fe4962776ad51d175c3b01a25734/README.md),
[MIT license](/licenses/cutmix-2d8eb68-MIT.txt).
Complete released PNG, unchanged. The first two rows compare inputs and labels;
the remaining rows report different historical evaluation tasks and metrics.
They are authors' results, not local measurements or a common unit of quality.*

CutMix preserves the pixels outside a rectangle from image A and copies
the corresponding rectangle from B. Let $M$ equal one where A is retained:

$$
\begin{aligned}
\widetilde x&=M\odot x_A+(1-M)\odot x_B,\\
\widetilde y&=\lambda_{\mathrm{actual}}y_A
                 +(1-\lambda_{\mathrm{actual}})y_B.
\end{aligned}
$$

The coefficient is the retained fraction of image A. It depends on the
rectangle actually written, after integer rounding and boundary clipping:

$$
\lambda_{\mathrm{actual}}
=1-\frac{(x_2-x_1)(y_2-y_1)}{WH}.
$$

In the continuous sampling rule, a nominal $\lambda$ gives rectangle dimensions
$W\sqrt{1-\lambda}$ and $H\sqrt{1-\lambda}$. In an array, those dimensions
must become integers, and part of a centred rectangle may fall outside the image.
The nominal coefficient can therefore differ from the target weight.

For a $4\times4$ image and nominal $\lambda=0.5$, flooring each side gives
a $2\times2$ rectangle. At the centre it replaces four of sixteen pixels,
so the actual retained weight is 0.75. At the corner, clipping leaves only
$1\times1$ and the retained weight becomes $15/16=0.9375$.

The released training loop performs that correction before calculating either
cross-entropy term:

```python title="Original CutMix train.py, lines 230–240; archival CUDA excerpt"
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            # compute output
            output = model(input)
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
```

Yun et al., [CutMix reference implementation, pinned `train.py`](https://github.com/clovaai/CutMix-PyTorch/blob/2d8eb68faff7fe4962776ad51d175c3b01a25734/train.py#L230-L240),
MIT. The excerpt is unchanged and depends on the surrounding script.
Its `rand_bbox` names dimension 2 `W` and dimension 3 `H`, although their
roles in BCHW are height and width. It also uses the obsolete `np.int` alias.

The next helper uses explicit row/column conventions for tensors
`[B,C,H,W]`. It follows the released code's integer half-width rule and
recalculates the weight. It is a CPU teaching implementation, not a copy
of the full historical CUDA training script.

```python
import math
import torch
import torch.nn.functional as F

def cutmix_batch(images, labels, partners, nominal_lam, center, classes):
    if images.ndim != 4 or not 0 <= nominal_lam <= 1:
        raise ValueError("expected BCHW images and lambda in [0, 1]")
    if len(partners) != len(images) or sorted(partners.tolist()) != list(range(len(images))):
        raise ValueError("partners must be a batch permutation")
    height, width = images.shape[-2:]
    cy, cx = center
    cut_h = int(height * math.sqrt(1 - nominal_lam))
    cut_w = int(width * math.sqrt(1 - nominal_lam))
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, height)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, width)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[partners, :, y1:y2, x1:x2]
    actual_lam = 1 - (y2 - y1) * (x2 - x1) / (height * width)
    targets = F.one_hot(labels, classes).to(images.dtype)
    targets = actual_lam * targets + (1 - actual_lam) * targets[partners]
    return mixed, targets, actual_lam, (x1, y1, x2, y2)

images = torch.stack([torch.zeros(1, 4, 4), torch.ones(1, 4, 4)]).double()
labels = torch.tensor([0, 1])
partners = torch.tensor([1, 0])
middle = cutmix_batch(images, labels, partners, .5, (2, 2), 2)
corner = cutmix_batch(images, labels, partners, .5, (0, 0), 2)
assert middle[2] == .75 and corner[2] == 15 / 16
assert (middle[0][0] == 1).sum().item() == 4
assert (corner[0][0] == 1).sum().item() == 1
torch.testing.assert_close(middle[1][0], torch.tensor([.75, .25], dtype=images.dtype))
assert torch.equal(images[0], torch.zeros_like(images[0]))
print("centre:", middle[2:], "corner:", corner[2:])
```

The helper expects valid class indices, a positive image size, and a centre
inside the image. It is not a general validator for arbitrary annotation
formats. Its box clipping and coefficient calculation work for nonsquare
inputs as well; the dimensions are kept separate.

Area is a chosen label-assignment rule, not a measurement of visible object
mass. A small pasted patch can contain an entire second object, while a large
patch can contain only background. CutMix's empirical usefulness does not
establish an exact physical correspondence between area and semantic evidence.
For detection or segmentation, copying patches requires rules for those
annotations rather than mechanically mixing image-level one-hot labels.

### One optimizer step with the mixed target

Using `cutmix_batch` from the preceding program, the following complete
forward/backward/update sequence trains a small linear classifier on four
fixed arrays. The transform is held fixed for the before/after loss comparison.

```python
import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(53)
inputs = torch.randn(4, 1, 4, 4, dtype=torch.float64)
labels = torch.tensor([0, 1, 0, 2])
partners = torch.tensor([1, 0, 3, 2])
mixed, targets, actual_lam, box = cutmix_batch(
    inputs, labels, partners, .5, (2, 2), classes=3)
model = nn.Sequential(nn.Flatten(), nn.Linear(16, 3)).double()
optimizer = torch.optim.SGD(model.parameters(), lr=.01)

optimizer.zero_grad()
logits = model(mixed)
loss = -(targets * logits.log_softmax(-1)).sum(-1).mean()
weighted = (actual_lam * F.cross_entropy(logits, labels)
            + (1 - actual_lam) * F.cross_entropy(logits, labels[partners]))
torch.testing.assert_close(loss, weighted)
loss.backward()
assert model[1].weight.grad.norm().item() > 0
assert torch.isfinite(model[1].weight.grad).all()
optimizer.step()
with torch.no_grad():
    after = -(targets * model(mixed).log_softmax(-1)).sum(-1).mean()
assert after < loss.detach()
torch.testing.assert_close(targets.sum(-1), torch.ones(4, dtype=inputs.dtype))
print("fixed transformed batch loss", loss.item(), "->", after.item())
```

Only one model forward is required for the mixed batch. The probability targets
have the same class dimension as the logits and sum to one. The update reaches
the model parameters; it does not train the data transformation.
The lower loss on these same arrays is a check of implementation, not evidence
of improved validation accuracy.

Training accuracy against the original hard labels also needs interpretation:
the input and target have been changed. It is not directly comparable with
accuracy on unmodified images. Evaluate the intended recognition task using
its declared held-out protocol.

<a id="augmentation-policies"></a>
## AutoAugment searches policies; RandAugment reduces the search parameters

A list of permitted operations does not specify a policy. A policy also
determines their order, probability, strength, and whether choices are shared
across a batch. Rotating then cropping need not produce the same image as
cropping then rotating.

In [AutoAugment, §3](https://arxiv.org/html/1805.09501v3#S3), a candidate
policy contains five sub-policies. Each sub-policy contains two ordered
operations, with an application probability and magnitude for each.
For each training image, one sub-policy is sampled, then the two probability
decisions are made. The search trains child classifiers and uses their
validation accuracy to update a controller. The original search used PPO;
its role is policy selection, not a new loss for the final image classifier.

The paper's searched policy uses 16 operation types, 10 magnitude values,
and 11 probability values. There are ten operation slots across its five
sub-policies. After search, the authors concatenate sub-policies from the
best five policies into a 25-sub-policy policy for final training.
The search candidate and the final concatenated policy are different objects.

For a small probability example, suppose two equally likely sub-policies
have application probabilities $(0.8,0.5)$ and $(0.3,0.1)$.
The probability that neither operation is called is
$0.5(0.2\cdot0.5)+0.5(0.7\cdot0.9)=0.365$.
Not calling an operation is different from calling an operation whose chosen
magnitude happens to leave an image unchanged.

[RandAugment, §3](https://arxiv.org/html/1909.13719v2#S3) uses a uniform
sampler over a pool of 14 operations and two main hyperparameters:
the number of sequential operations $N$ and a shared magnitude $M$.
Sampling operations for each image remains random. The reduction concerns
the parameters searched to choose the policy, not a requirement to use the
same two transformations for every image.

```python
from itertools import product
from math import isclose
import random

no_call_probability = .5 * (1 - .8) * (1 - .5) + .5 * (1 - .3) * (1 - .1)
assert isclose(no_call_probability, .365)

operations = ["Identity", "AutoContrast", "Equalize", "Rotate", "Solarize",
              "Color", "Posterize", "Contrast", "Brightness", "Sharpness",
              "ShearX", "ShearY", "TranslateX", "TranslateY"]
N, M = 2, 7
rng = random.Random(31)
sampled_policy = [(operation, M) for operation in rng.choices(operations, k=N)]
possible_sequences = list(product(operations, repeat=N))
assert len(operations) == 14 and len(possible_sequences) == 196
assert ("Rotate", "Rotate") in possible_sequences
assert all(magnitude == M for _, magnitude in sampled_policy)
autoaugment_search_space = (16 * 10 * 11) ** 10
assert autoaugment_search_space > 10**32
print("sampled RandAugment operation sequence", sampled_policy)
print("AutoAugment candidate count", autoaugment_search_space)
```

This program checks policy sampling and counts; it does not implement the
fourteen image operators or search for a good policy. Repeated operation
types are possible because sampling is with replacement. The 196 possible
two-operation sequences are not 196 hyperparameter trials that must each be
trained: the sampler draws from them while a model trains.

A magnitude index must be mapped separately to rotation angle, translation,
colour adjustment, or another operation's units. The same index does not
mean the same numerical change in those units. Magnitude scales and available
operations can also differ between implementations.

Policy strength needs validation on the intended model and dataset.
RandAugment's motivation includes the failure of transferring an optimum
from a small proxy task without adjustment. Neither paper establishes that
every smaller model should receive stronger augmentation. A model can
underfit when transformations or target mixtures are too strong.

<a id="text-audio-augmentation"></a>
## Text and speech need task-specific transformations

Deleting “not” from “the film was not good” can reverse sentiment. Replacing
a word with a dictionary synonym can change its sense in context.
For entity extraction, changing a name requires updating its spans; for
question answering, a paraphrase must preserve the answer relation.
String validity and grammatical fluency are not enough to preserve a target.

Round-trip translation can produce a paraphrase by translating a sentence
to another language and back. This is different from the synthetic parallel
data construction in Sennrich, Haddow and Birch's
[Back-translation, §3.2](https://arxiv.org/html/1511.06709v4#S3.SS2).
To train English-to-German translation, the method takes an existing German
target sentence and uses a German-to-English system to construct a source.
The added pair has a synthetic English source and the original German target.
It does not require a second translation back into German.

For illustration, a German target “Das Haus ist klein.” can be paired with
an automatically obtained English source such as “The house is small.”
These sentences specify the sides of the training pair, not a new local
translation-model result. Translation errors on the synthetic side and
overlap with evaluation material still need checking.

### Count original recordings, not only augmented files

[Harvard ML Systems, Data Selection](https://github.com/harvard-edge/cs249r_book/blob/2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd/book/quarto/contents/vol1/data_selection/data_selection.qmd)
uses a keyword-spotting scenario with 50 speakers and ten recordings
per speaker: 500 originals. With ten transform settings and two noise settings,
there are $500\times10\times2=10{,}000$ derived records. The factor of two
makes the noise variants explicit; the records still represent only 50 speakers.
Pitch, time stretch, gain, background noise and microphone variation must be
chosen for the wake-word task; some settings can make speech unintelligible.

Splitting the 10,000 files randomly can put variants of one recording or
speaker on both sides. If the evaluation question concerns unseen speakers,
speaker membership must be decided before generating derived records.

```python
recordings = [(speaker, utterance) for speaker in range(50)
              for utterance in range(10)]
derived = [(speaker, utterance, transform, noise)
           for speaker, utterance in recordings
           for transform in range(10) for noise in range(2)]
train = [r for r in derived if r[0] < 40]
test = [r for r in derived if r[0] >= 40]
assert len(recordings) == 500 and len(derived) == 10_000
assert len({r[0] for r in derived}) == 50
assert len({r[:2] for r in derived}) == 500
assert len(train) == 8000 and len(test) == 2000
assert {r[0] for r in train}.isdisjoint({r[0] for r in test})
print("original recordings", len(recordings), "derived records", len(derived))
```

The program checks identity and partitioning; it does not synthesize audio.
The same parent-ID rule applies to document paraphrases and image crops.
More independent source material and more views of existing material are
different ways to spend the data budget.

<a id="augmentation-evaluation"></a>
## Training transforms, validation, and test-time augmentation

Training usually samples new transformations. A validation procedure needs
its own declared input processing, often deterministic. In D2L's example,
the training loader applies random horizontal reflection and the evaluation
loader does not. A gallery showing many transformations should not be read
as the exact training recipe used by a later code cell.

Calling `model.eval()` changes modules such as dropout and BatchNorm.
It does not disable a random crop in a dataset or data loader.
If stochastic evaluation is intended, specify its seeds, number of views,
aggregation rule, and uncertainty.

Test-time augmentation runs the model on several transformed views and
combines their predictions. For image classification, the views must refer
to the same class space. Spatial predictions such as masks or keypoints
must be transformed back into a common coordinate system before aggregation.

Averaging probabilities and averaging logits give different predictors.
Take two views with class probabilities $(0.9,0.1)$ and $(0.6,0.4)$.
Probability averaging gives $(0.75,0.25)$. Averaging their log-probabilities
and applying softmax instead normalizes their geometric means.

```python
import torch

view_probabilities = torch.tensor([[.9, .1], [.6, .4]], dtype=torch.float64)
probability_average = view_probabilities.mean(0)
logit_average = view_probabilities.log().mean(0).softmax(0)
torch.testing.assert_close(probability_average,
                           torch.tensor([.75, .25], dtype=torch.float64))
geometric = view_probabilities.prod(0).sqrt()
torch.testing.assert_close(logit_average, geometric / geometric.sum())
assert not torch.allclose(probability_average, logit_average)
print("mean probabilities", probability_average)
print("softmax of mean logits", logit_average)
```

Choose that rule on validation data rather than selecting whichever version
scores best on the final test set. Test-time augmentation also requires
additional forwards; training-only augmentation need not change inference
cost. Report which procedure produced the metric.

To test an augmentation recipe, hold out data by the original acquisition
unit before constructing variants. Compare an unchanged baseline with the
new recipe under declared training steps, optimizer settings, and selection
rules. Include unaugmented evaluation examples and task-relevant shifts.
A high score on generated variants alone can reward the generator's pattern.

Heavy transformations can move the training bottleneck into CPU processing
or data transfer. Their end-to-end cost and repetition rate are developed in
[[02 Areas/ML & DL/00 Учебник/10 ML Systems/06 Data pipeline, padding и packing|Data pipeline, padding and packing]]
and [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41i Отбор данных — стоимость, шарды и проверка качества|selection engineering and data echoing]].
Learning from two transformed views is developed further in
[[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41e Обучение с частичной разметкой — pseudo-labeling и FixMatch|FixMatch]]
and [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41f Контрастивное предобучение — SimCLR и проверка представлений|SimCLR]].
Generating a new document or a simulator scene introduces additional
correctness and distribution questions beyond a transform of a stored input;
the [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/53 Синтетические данные и учебные программы|synthetic-data chapter]]
develops generation, filtering, mixtures and recursive training.

---

**Предыдущая глава:** [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/06 Dropout и регуляризация|Dropout и регуляризация]]

**Следующая глава:** [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/07 CNN — от свёртки до ResNet|CNN — от свёртки до ResNet]]

**Источники:** Zhang et al.,
[D2L, Image Augmentation](https://d2l.ai/chapter_computer-vision/image-augmentation.html),
CC BY-SA 4.0; Zhang et al.,
[mixup, §2](https://arxiv.org/html/1710.09412v2#S2); Yun et al.,
[CutMix, §§3.1–3.2](https://arxiv.org/html/1905.04899v1#S3) and the
[MIT-licensed reference implementation](https://github.com/clovaai/CutMix-PyTorch/blob/2d8eb68faff7fe4962776ad51d175c3b01a25734/train.py);
Cubuk et al., [AutoAugment, §3](https://arxiv.org/html/1805.09501v3#S3)
and [RandAugment, §3](https://arxiv.org/html/1909.13719v2#S3);
Sennrich, Haddow and Birch,
[Improving Neural Machine Translation Models with Monolingual Data, §3](https://arxiv.org/html/1511.06709v4#S3).
Run the nine CPU examples in reading order; the full-step block uses the
preceding CutMix helper. The two labelled source excerpts retain their
authors' code and require the dependencies of the original books or scripts.
