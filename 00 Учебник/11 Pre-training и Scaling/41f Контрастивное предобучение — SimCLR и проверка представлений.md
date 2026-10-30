---
title: Контрастивное предобучение — SimCLR и проверка представлений
type: textbook-chapter
status: canonical
source_language: en
last_updated: 2026-10-28
last_verified: 2026-10-28
primary_sources:
  - https://proceedings.mlr.press/v119/chen20j.html
  - https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html
  - https://github.com/phlippe/uvadlc_notebooks/blob/c7d9e18f42fee53bc8925f08ec2358371b1e9200/docs/tutorial_notebooks/tutorial17/SimCLR.ipynb
---

# 41f. Контрастивное предобучение — SimCLR и проверка представлений

Two crops of a photograph can show different parts of the same dog. Their
pixels differ, and neither crop comes with a human class label. Their common
origin nevertheless supplies a training relation: both were made from the
same stored image. SimCLR trains an encoder to distinguish that pair from
views of other images in the batch.

The target is the identity of the matching view, not a predicted class such
as “dog”. This differs from the classifier-generated pseudo-labels in
[[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41e Обучение с частичной разметкой — pseudo-labeling и FixMatch|FixMatch]].
Only after pretraining do human class labels enter a downstream classifier
or fine-tuning task. The choice of views determines which input differences
the representation is encouraged to ignore.

<a id="two-view-encoder"></a>
## Two views, one encoder and a projection head

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/simclr-2026-10-28/framework.png|Original SimCLR encoder and projection-head framework]]

*Ting Chen, Simon Kornblith, Mohammad Norouzi and Geoffrey Hinton,
[SimCLR, Figure 2, PMLR 119](https://proceedings.mlr.press/v119/chen20j.html),
[PDF p. 2](https://proceedings.mlr.press/v119/chen20j/chen20j.pdf#page=2),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The complete diagram is extracted from the original PDF; labels are unchanged.*

Sample two transformations independently from an augmentation family
$\mathcal T$. Applying them to an image $x$ gives views $\tilde x_i=t(x)$
and $\tilde x_j=t'(x)$. The two branches use the same parameters:

$$
\begin{aligned}
 h_i&=f_\theta(\tilde x_i), & h_j&=f_\theta(\tilde x_j),\\
 z_i&=g_\phi(h_i), & z_j&=g_\phi(h_j).
\end{aligned}
$$

The base encoder $f_\theta$ extracts a representation $h$. A small nonlinear
network $g_\phi$ maps it to $z$, where the contrastive loss is computed.
In the original study the base model is a ResNet and the projection head
is a two-layer MLP with a ReLU between its linear layers. Both networks
receive gradients during pretraining. Neither branch is an EMA teacher,
and neither target embedding is detached as in FixMatch's weak branch.

For downstream linear evaluation, the original SimCLR protocol removes
$g_\phi$, freezes $f_\theta$ and trains a classifier on $h$. Keeping a
projection head during pretraining and removing it afterward are compatible:
the head changes the gradient that trained the encoder, even when the head
is absent at evaluation.

<a id="augmentation-task"></a>
## Augmentation defines the prediction task

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/simclr-2026-10-28/crop-views.png|Original global-local and adjacent crop relations]]

*Chen et al., [SimCLR, Figure 3](https://proceedings.mlr.press/v119/chen20j.html),
[PDF p. 3](https://proceedings.mlr.press/v119/chen20j/chen20j.pdf#page=3),
CC BY 4.0. Dashed rectangles are crops; solid rectangles are the original
image boundaries. Both panels and their labels are retained.*

In the left panel, crop A lies inside crop B. Matching them requires some
tolerance to scale and missing surrounding context. In the right panel,
C and D contain neighbouring regions. A model may need to relate different
parts of the scene, rather than match the same pixels. Neither construction
guarantees that both crops retain the object required by a later task.

Phillip Lippe's
[UvA tutorial, Data Augmentation for Contrastive Learning](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html#Data-Augmentation-for-Contrastive-Learning)
explains a shortcut that remains if cropping is used alone:

> While situation (a) requires the model to learn some sort of scale
> invariance to make crops A and B similar in latent space, situation (b)
> is more challenging since the model needs to recognize an object beyond
> its limited view. However, without color distortion, there is a loophole
> that the model can exploit, namely that different crops of the same image
> usually look very similar in color space.

*Phillip Lippe, UvA Deep Learning Tutorials, notebook cell 9;
[MIT license copy](/licenses/uva-c7d9e18-MIT.txt). Original English excerpt.*

A dog crop and a grass crop can share the photograph's characteristic
colour distribution. Matching colour histograms need not produce a useful
object representation. Independently distorting colours weakens that cue.
It does not prove that all remaining cues are semantic: backgrounds,
watermarks and acquisition artefacts may still identify the source image.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/simclr-2026-10-28/augmentations.png|Original ten-panel comparison of SimCLR augmentation operators]]

*Chen et al., [SimCLR, Figure 4](https://proceedings.mlr.press/v119/chen20j.html),
[PDF p. 4](https://proceedings.mlr.press/v119/chen20j/chen20j.pdf#page=4),
CC BY 4.0; original photograph credited in the paper to Von.grzanka, CC-BY.
All ten panels and labels are retained. These are the operators studied
in ablations, not a list of ten operations applied in every training example.*

The original training policy combines random crop with resize and flip,
colour distortion and Gaussian blur. Figure 4 also shows rotation, Cutout,
Gaussian noise and Sobel filtering because the authors studied them as
alternatives. Confusing an ablation catalogue with the final training
policy changes the algorithm.

The UvA implementation makes two independent calls to the same stochastic
pipeline. Its wrapper is preserved below without changing the class body:

```python
class ContrastiveTransformations(object):
    
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views
        
    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]

# Check the two calls without downloading an image dataset.
calls = []
def record_view(x):
    calls.append(x)
    return (x, len(calls))
views = ContrastiveTransformations(record_view)("image-A")
assert views == [("image-A", 1), ("image-A", 2)]
assert calls == ["image-A", "image-A"]
```

*Phillip Lippe, original wrapper from notebook cell 8, MIT. The small
call-recording check after the class is an editorial addition.*

Calling the transform twice permits different random draws; it does not
require different outputs on every call. Two flips can both choose “do
nothing”. Producing one augmented tensor and copying it instead would
remove the independent draw.

The UvA experiment uses 96×96 STL10 images and five transformations:
horizontal flip, random resized crop, colour jitter, random grayscale
and Gaussian blur. Its jitter strengths are 0.5 for brightness, contrast
and saturation and 0.1 for hue, weaker than the cited original settings.
Those are the tutorial's experimental choices, not additional requirements
of NT-Xent. In another task, colour itself may determine the answer;
forcing invariance to it can discard useful evidence. The same caution
applies to deleting negation or changing an answer-bearing number in text.

<a id="positive-index"></a>
## A batch contains views, not class labels

Let a batch contain $N$ stored images. Two views per image give $2N$
embeddings. For each anchor there is one known matching view, $2N-2$
views of other images and one self-comparison that must be excluded.
The positive is still included in the denominator.

The UvA code concatenates the first views and then the second views.
With three images A, B and C the order is:

| Row | View | Positive row | Views treated as negatives |
| ---: | --- | ---: | --- |
| 0 | A, first | 3 | 1, 2, 4, 5 |
| 1 | B, first | 4 | 0, 2, 3, 5 |
| 2 | C, first | 5 | 0, 1, 3, 4 |
| 3 | A, second | 0 | 1, 2, 4, 5 |
| 4 | B, second | 1 | 0, 2, 3, 5 |
| 5 | C, second | 2 | 0, 1, 3, 4 |

For this layout the positive index is $p(i)=(i+N)\bmod2N$. The paper's
Algorithm 1 instead interleaves pairs, giving order A1, A2, B1, B2, C1, C2.
Both layouts represent the same task if the positive indices are changed
consistently. Applying the offset-$N$ rule to the interleaved layout would
train against the wrong targets.

Two different dog photographs are negatives under this instance-based
objective, even though a later classifier may assign both to “dog”.
The method has no class label that would reveal that relationship.
Near-duplicate photographs can create another false-negative case.
The representation's usefulness for class recognition must therefore be
measured, rather than inferred from the pairing rule.

<a id="nt-xent"></a>
## NT-Xent: choosing the matching view

For nonzero embeddings, normalize $u_i=z_i/\|z_i\|_2$ and form scaled
similarities $S_{ik}=u_i^\top u_k/\tau$, where $\tau>0$. For anchor $i$:

$$
\begin{aligned}
 q_{ik}&=\frac{\exp S_{ik}}{\sum_{r\ne i}\exp S_{ir}},\qquad k\ne i,\\
 \ell_i&=-\log q_{i,p(i)},\\
 \mathcal L&=\frac1{2N}\sum_{i=0}^{2N-1}\ell_i.
\end{aligned}
$$

This is cross-entropy over the other views in the current batch. Its
probability is conditional on that candidate set, not a probability that
two photographs have the same semantic class. Both directions of every
positive pair contribute to the mean. There are $2N$ anchors, not $N$.

For four unit vectors, use
$u_0=(1,0)$, $u_1=(0,1)$, $u_2=(0.8,0.6)$ and $u_3=(-0.6,0.8)$.
The layout pairs rows 0↔2 and 1↔3. At temperature 0.5, anchor 0 has
three candidate logits:

| Candidate | Relation to row 0 | Cosine | Scaled logit |
| ---: | --- | ---: | ---: |
| 1 | negative | 0 | 0 |
| 2 | positive | 0.8 | 1.6 |
| 3 | negative | −0.6 | −1.2 |

The normalizer is $1+e^{1.6}+e^{-1.2}$. Its positive probability is
$e^{1.6}/(1+e^{1.6}+e^{-1.2})$, so the loss is
$\log(1+e^{1.6}+e^{-1.2})-1.6$. Including row 0 would add $e^2$
from the self-cosine 1 and turn “match the other view” into a different
task. Removing the positive from the denominator would also change
the objective; the expression would no longer be this categorical CE.

For one row, differentiating with respect to a scaled candidate logit
gives $q_{ik}-\mathbf1[k=p(i)]$. Gradient descent raises the positive
score and lowers negative scores, with more weight on high-scoring
negatives. Differentiating an unscaled cosine introduces $1/\tau$.
The encoder gradients combine all rows: an embedding appears as an
anchor and as a candidate in other anchors' denominators.

The following implementation uses the UvA positive-index and diagonal-mask
construction, replacing its broadcast cosine operation and logging with a
normalized matrix multiplication. Each row has its own positive target.

```python
import math
import torch
import torch.nn.functional as F

def nt_xent(features, temperature):
    if features.ndim != 2 or features.shape[0] < 2 or features.shape[0] % 2:
        raise ValueError("need two equally sized view blocks")
    if not features.is_floating_point() or not torch.isfinite(features).all():
        raise ValueError("need finite floating-point features")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if torch.any(features.norm(dim=1) == 0):
        raise ValueError("cosine is undefined for a zero vector")
    n = features.shape[0] // 2
    unit = features / features.norm(dim=1, keepdim=True)
    logits = (unit @ unit.T) / temperature
    diagonal = torch.eye(2 * n, dtype=torch.bool, device=features.device)
    logits = logits.masked_fill(diagonal, -torch.inf)
    positive = (torch.arange(2 * n, device=features.device) + n) % (2 * n)
    return F.cross_entropy(logits, positive), logits, positive

features = torch.tensor([[1., 0.], [0., 1.], [.8, .6], [-.6, .8]],
                        dtype=torch.float64, requires_grad=True)
loss, logits, positive = nt_xent(features, .5)
per_anchor = F.cross_entropy(logits, positive, reduction="none")
expected_first = math.log(1 + math.exp(1.6) + math.exp(-1.2)) - 1.6
assert positive.tolist() == [2, 3, 0, 1]
assert abs(per_anchor[0].item() - expected_first) < 1e-12
assert torch.equal(logits.softmax(dim=1).diagonal(), torch.zeros(4, dtype=torch.float64))
loss.backward()
assert torch.isfinite(features.grad).all()
assert (features.grad.norm(dim=1) > 0).all()
print("per-anchor loss", per_anchor.detach().tolist(), "mean", loss.item())
```

The four losses are approximately 0.233257, 0.627123, 0.627123 and
0.233257; their mean is 0.430190. The two directions of a pair need
not have equal losses because their other candidate similarities differ.

The original UvA method is in cell 17 of the
[pinned notebook](https://github.com/phlippe/uvadlc_notebooks/blob/c7d9e18f42fee53bc8925f08ec2358371b1e9200/docs/tutorial_notebooks/tutorial17/SimCLR.ipynb).
Its `self_mask.roll(...//2)` encodes the same positive mapping. The
standalone function above is an adaptation, not an unchanged copy of the
Lightning training class.

<a id="shared-training-step"></a>
## One shared-parameter training step

The preceding arrays isolate the loss. A training step must generate
embeddings with the same encoder and head, accumulate gradients from both
views, and update those parameters once. This small CPU example uses three
two-dimensional inputs and fixed paired perturbations. It checks that
computation without claiming to learn visual semantics or reproduce an
ImageNet run. Run it after the `nt_xent` definition above.

```python
import torch

torch.manual_seed(11)
encoder = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh()).double()
projector = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Tanh(),
                                torch.nn.Linear(8, 4)).double()
first = torch.tensor([[1., 0.], [0., 1.], [-1., -1.]], dtype=torch.float64)
second = first + torch.tensor([[.02, -.03], [-.03, .02], [.01, .03]], dtype=torch.float64)
parameters = list(encoder.parameters()) + list(projector.parameters())
optimizer = torch.optim.SGD(parameters, lr=.01)
before = [p.detach().clone() for p in parameters]

optimizer.zero_grad()
encoded = encoder(torch.cat([first, second]))
encoded.retain_grad()
projected = projector(encoded)
loss, _, _ = nt_xent(projected, .5)
loss.backward()
assert encoded.grad[:3].norm() > 0 and encoded.grad[3:].norm() > 0
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
optimizer.step()
assert any(not torch.equal(a, b) for a, b in zip(before, parameters))
with torch.no_grad():
    after, _, _ = nt_xent(projector(encoder(torch.cat([first, second]))), .5)
assert after < loss.detach()
print("small-step loss", loss.item(), "->", after.item())
```

With the stated seed and inputs, this step reduces the loss from
approximately 1.258085 to 1.156215. Both view blocks send gradients
through the same encoder; the labels of a downstream task are not used.

The full experiments also have an image pipeline, a ResNet, normalization
layers, a schedule and an optimizer configuration. The original paper's
default ablations use ResNet-50, a 128-dimensional projection, batch size
4096, LARS and 100 epochs, with ten warmup epochs. The UvA exercise uses
ResNet-18, AdamW, batch size 256, temperature 0.07 and 500 epochs.
Those are different experiments. Neither the small step nor the notebook's
loss implementation reproduces the original paper's reported accuracy.

In distributed ResNet training, the paper aggregates batch-normalization
statistics across devices to reduce a local-statistics shortcut. The
present linear example has no batch normalization or distributed exchange.
Correct positive indices and a correct CE are necessary but do not cover
these additional training-system decisions.

<a id="collapse-and-negatives"></a>
## Equal embeddings and the role of negatives

If all $2N$ embeddings are the same nonzero vector, all candidate logits
within a row are equal. The positive probability is $1/(2N-1)$ and each
anchor's loss is $\log(2N-1)$. Increasing the positive relative to other
images can lower the objective, so constant embeddings are not a good
solution to the discrimination task.

That statement does not guarantee escape from every collapsed state.
For exact identical normalized vectors, cosine derivatives can vanish:
the normalized dot product is already at its stationary maximum. A
nonzero scalar loss alone does not prove a nonzero parameter gradient.

```python
import math
import torch

collapsed = torch.ones(6, 3, dtype=torch.float64, requires_grad=True)
loss, _, _ = nt_xent(collapsed, .5)
loss.backward()
assert abs(loss.item() - math.log(5)) < 1e-12
assert collapsed.grad.abs().max().item() < 1e-12

# One original image has no negatives: its other view is the sole candidate.
single = torch.tensor([[1., 0.], [.8, .6]], dtype=torch.float64, requires_grad=True)
single_loss, _, _ = nt_xent(single, .5)
single_loss.backward()
assert single_loss.item() == 0 and single.grad.abs().max().item() == 0
print("collapsed loss", loss.item(), "single-pair loss", single_loss.item())
```

The negative candidates supply a relative discrimination objective;
they do not certify the geometry learned by an optimizer. More candidates
can also introduce more false negatives. At image batch size $N=4096$
there are 8192 views and 8190 negatives per anchor, not 4096 negatives.
The full float32 similarity matrix alone has $8192^2$ entries, requiring
268,435,456 bytes before gradients, activations and temporary buffers.

Accumulating gradients over separate small batches does not create
comparisons between their embeddings. In the extreme case above, each
one-image microbatch gives zero loss and zero gradient; summing those
gradients still gives zero. A joint two-image batch supplies negatives
and a different objective. Larger candidate sets need an explicit
mechanism such as a joint batch, an embedding exchange or a suitable
cache; optimizer batch size is not a substitute.

<a id="projection-versus-representation"></a>
## Why evaluate the representation before the head?

The projection is trained to make selected augmentations agree. A
downstream task may require information those augmentations change.
A nonlinear $g$ can suppress colour in $z$ while $h$ still carries it.
The training objective alone does not prove that this separation happens;
the paper tests representations before and after the head.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/simclr-2026-10-28/projection-head.png|Original comparison of linear, nonlinear and absent SimCLR projection heads]]

*Chen et al., [SimCLR, Figure 8](https://proceedings.mlr.press/v119/chen20j.html),
[PDF p. 6](https://proceedings.mlr.press/v119/chen20j/chen20j.pdf#page=6),
CC BY 4.0. The full chart retains both axes and the legend. The measured
representation $h$ is 2048-dimensional; the horizontal axis changes
the projection output dimension, not the dimension of the evaluated $h$.*

In this experiment, a nonlinear head during pretraining improves linear
evaluation of $h$ compared with a linear head or no head. It is not a plot
of classification accuracy at progressively shorter $h$ embeddings.
That would be a different experiment, such as the
[[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/60 Embeddings и metric learning|Matryoshka evaluation]].
Section 4.2 also compares $h$ and $g(h)$ directly and probes information
about applied transformations. The authors offer information removal as
an explanation of their results, not a theorem that every head necessarily
discards exactly the nuisance factors of a new task.

<a id="linear-evaluation"></a>
## Linear evaluation uses labels and a frozen encoder

The UvA tutorial describes the evaluation stage as follows:

> A common setup, which also verifies whether the model has learned
> generalized representations, is to perform Logistic Regression on the
> features. In other words, we learn a single, linear layer that maps
> the representations to a class prediction.

*Phillip Lippe, UvA Tutorial 17, Logistic Regression, cell 26; MIT.
The excerpt is preserved in English.*

For $C$ classes, a classifier maps $h$ to logits $Wh+b$ and is trained
with labelled cross-entropy. Its parameters change; the encoder's do not.
This asks whether the desired classes can be separated by a linear
function in the fixed feature space. It does not test arbitrary future
tasks and does not make the evaluation label-free.

The original feature-preparation function removes the entire ResNet
`fc` projection, switches to evaluation mode and computes features under
`torch.no_grad()`. Freezing weights and choosing evaluation mode solve
different problems: the latter also controls dropout and batch-norm
behaviour. Cached deterministic features make training a small probe
cheap, but do not account for the preceding encoder-pretraining cost.

The following program isolates the frozen-feature step with a deliberately
specified encoder, rather than claiming that these features were learned
from photographs. It fits and checks the probe on four toy inputs only;
there is no generalization estimate from these same four examples.

```python
import torch
import torch.nn.functional as F

encoder = torch.nn.Linear(2, 3, bias=False, dtype=torch.float64)
with torch.no_grad():
    encoder.weight.copy_(torch.tensor([[1., 0.], [-1., 0.], [0., 1.]], dtype=torch.float64))
encoder.requires_grad_(False)
encoder.eval()
saved = encoder.weight.detach().clone()
x = torch.tensor([[2., 1.], [1., 2.], [-2., 1.], [-1., 2.]], dtype=torch.float64)
y = torch.tensor([0, 0, 1, 1])
with torch.no_grad():
    cached = encoder(x)
probe = torch.nn.Linear(3, 2, dtype=torch.float64)
with torch.no_grad():
    probe.weight.zero_()
    probe.bias.zero_()
optimizer = torch.optim.SGD(probe.parameters(), lr=.2)
start = F.cross_entropy(probe(cached), y).item()
for _ in range(30):
    optimizer.zero_grad()
    objective = F.cross_entropy(probe(cached), y)
    objective.backward()
    optimizer.step()
finish = F.cross_entropy(probe(cached), y).item()
assert finish < start
assert torch.equal(encoder.weight, saved) and encoder.weight.grad is None
assert torch.equal(probe(cached).argmax(dim=1), y)
print("probe fit loss", start, "->", finish, "encoder unchanged")
```

### A protocol correction when using the tutorial's results

The archived UvA code passes STL10's test loader as Lightning's validation
loader and selects checkpoints by that accuracy. Cell 34 explicitly
describes test-set evaluation for early stopping; cell 35 implements it.
Checking every ten epochs reduces the number of looks, but does not make
checkpoint selection independent of the test set. The source's reported
curve is therefore not an untouched-test estimate under that protocol.

For a new experiment, split labelled training data into probe-training
and validation parts before selection or augmentation. Choose the epoch,
regularization and other settings using validation only, then evaluate
the frozen chosen procedure on the test split. Keep related photographs
and near-duplicates on the same side of any split. Report the label count
per class, the unlabelled pretraining corpus, encoder, view policy and
selection rule. Fine-tuning the encoder is a different protocol from a
linear probe and needs its own result.

The original SimCLR paper studies image representations. Applying the same
softmax construction to text, retrieval or image–text pairs changes the
meaning of positives and negatives. Those choices, along with SimCSE and
retrieval metrics, are developed in the
[[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/60 Embeddings и metric learning|embeddings chapter]].
Generating caption targets through a connector and language decoder is
covered in
[[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64c Обучение VLM — alignment, instruction tuning и данные|VLM training]].

---

К модулям: [[02 Areas/ML & DL/00 Учебник/_index#Углублённые модули|Выбрать углублённый модуль]]

**Следующая глава:** [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41g Momentum contrast — MoCo, очередь и key encoder|Momentum contrast — MoCo, очередь и key encoder]]

**Источники:** Ting Chen, Simon Kornblith, Mohammad Norouzi and Geoffrey Hinton,
[A Simple Framework for Contrastive Learning of Visual Representations](https://proceedings.mlr.press/v119/chen20j.html),
PMLR 119:1597–1607, 2020, §§1–5, Figures 2–4 and 8;
[license copy](/licenses/simclr-PMLR-CC-BY-4.0.txt).
Phillip Lippe,
[UvA Tutorial 17](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html),
[pinned notebook](https://github.com/phlippe/uvadlc_notebooks/blob/c7d9e18f42fee53bc8925f08ec2358371b1e9200/docs/tutorial_notebooks/tutorial17/SimCLR.ipynb),
MIT. Quoted prose and the wrapper retain their original language; the
standalone loss, numerical checks and small training programs are adaptations
and editorial examples. No STL10 dataset, model weights or historical training
run is needed to execute these five small programs in order.
