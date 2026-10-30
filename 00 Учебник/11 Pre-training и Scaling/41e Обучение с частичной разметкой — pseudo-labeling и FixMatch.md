---
title: Обучение с частичной разметкой — pseudo-labeling и FixMatch
type: textbook-chapter
status: canonical
source_language: en
last_updated: 2026-10-29
last_verified: 2026-10-29
primary_sources:
  - https://arxiv.org/abs/2001.07685v2
  - https://github.com/google-research/fixmatch/tree/d4985a158065947dba803e626ee9a6721709c570
  - https://scikit-learn.org/stable/modules/semi_supervised.html
  - https://scikit-learn.org/stable/auto_examples/semi_supervised/plot_label_propagation_structure.html
  - https://arxiv.org/abs/1704.03976v2
  - https://arxiv.org/abs/1605.07725v4
  - https://arxiv.org/abs/1904.12848v6
  - https://github.com/google-research/uda/tree/960684e363251772a5938451d4d2bc0f1da9e24b
  - https://aclanthology.org/2022.sdp-1.5/
  - https://arxiv.org/abs/1911.04252v4
  - https://aclanthology.org/2020.acl-main.194/
  - https://arxiv.org/abs/1511.01432v1
  - https://research.google/blog/from-vision-to-language-semi-supervised-learning-in-action-at-scale/
---

# 41e. Обучение с частичной разметкой — pseudo-labeling и FixMatch

A classifier can have access to many more inputs than human-verified labels.
Training only on the labelled examples ignores the remaining inputs. Asking
an annotator to label some of them is
[[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41c Active learning — запрос меток и выбор примеров|active learning]].
Semi-supervised learning instead uses both the labelled and unlabelled
examples during training, without obtaining a human target for each new input.

The unlabelled inputs reveal their distribution, but that distribution does
not specify the intended classes. Consider four points at the corners of a
square. The bottom-left corner is labelled 0 and the top-right corner 1.
Both the rule “classify by the horizontal coordinate” and the rule “classify
by the vertical coordinate” agree with those labels:

| Point | Available label | Class from horizontal coordinate | Class from vertical coordinate |
| --- | --- | ---: | ---: |
| $(-1,-1)$ | 0 | 0 | 0 |
| $(-1,1)$ | unknown | 0 | 1 |
| $(1,-1)$ | unknown | 1 | 0 |
| $(1,1)$ | 1 | 1 | 1 |

Observing the other two inputs cannot resolve the disagreement. A
semi-supervised method therefore needs assumptions linking input structure
to labels: nearby points should often share a class, or an augmentation
should preserve the intended class. Whether these assumptions hold depends
on the task and representation.

<a id="pseudo-labels"></a>
## Self-training and the origin of a pseudo-label

A supervised classifier can supply a provisional target for an unlabelled
example. If its predicted class probabilities are $q=(0.97,0.03)$, the hard
pseudo-label is class 0. With a confidence threshold $\tau=0.95$, this
example is accepted; an example with $q=(0.75,0.25)$ is not. The threshold
uses the model's probability, not a verified probability that its answer
is correct.

The scikit-learn
[self-training documentation](https://scikit-learn.org/stable/modules/semi_supervised.html#self-training)
describes the iterative procedure in the original English:

> In each iteration, the `estimator` predicts labels for the unlabeled
> samples and adds a subset of these labels to the labeled dataset.
>
> The choice of this subset is determined by the selection criterion.
> This selection can be done using a `threshold` on the prediction
> probabilities, or by choosing the `k_best` samples according to the
> prediction probabilities.

*The scikit-learn developers, BSD-3-Clause;
[license copy](/licenses/semi-supervised-sklearn-BSD-3-Clause.txt).
The excerpt preserves the source language and identifiers.*

After refitting, the classifier may become confident on further examples.
The loop stops when its iteration budget is exhausted, no new examples
pass the criterion, or all examples have provisional labels. The source
implementation records when each label was assigned. Those records should
not turn pseudo-labels into human annotations: their origin remains a
model prediction, and their errors can be correlated with the classifier's
existing errors.

A higher threshold reduces the number of accepted examples for a fixed
set of predictions. It does not necessarily improve their actual accuracy:
a shifted or poorly calibrated classifier can be confidently wrong. A
lower threshold may add more examples while reinforcing incorrect targets.
The useful threshold must be evaluated with true held-out labels rather
than chosen by maximizing the number of generated labels.

<!-- BEGIN REVIEW: Noisy Student 2026-10-29 -->
<a id="noisy-student"></a>
## Noisy Student: clean targets, a noisy learner

The iterative self-training procedure above can use two separate models.
A teacher first learns from human-labelled examples. It then supplies
fixed targets for an unlabelled pool. A student learns from both sources
of supervision. After training, that student can become the next teacher,
which relabels the pool for another round.

[Noisy Student, §2 and Algorithm 1](https://arxiv.org/html/1911.04252v4#S2)
specifies where noise enters this procedure. The teacher generates
pseudo-labels in inference mode, on clean inputs, without dropout or
stochastic depth. The student learns with input augmentation and model
noise. Training the original teacher may itself use regularization;
“clean teacher” describes pseudo-label generation, not an assertion that
the teacher was trained without noise.

The authors' [original Figure 1](https://arxiv.org/html/1911.04252v4#S2.F1)
shows the four stages and puts data augmentation, dropout and stochastic
depth beside the student-training stage. The student is equal in size
to or larger than the teacher; being smaller is not the method's purpose.
A larger architecture provides capacity, not a guarantee that the student
will improve. The new teacher is obtained after a training round, rather
than by updating an EMA teacher at every minibatch.

### The target remains fixed while the student changes

For an unlabelled input $u_i$, let the clean teacher supply the
probability vector $q_i$. Let $p_i$ be the student's prediction under
its sampled augmentation and model noise. Soft-target cross-entropy is

$$
 \ell_i=-\sum_c q_{ic}\log p_{ic}.
$$

The teacher is held fixed during student training. For student logits
$z_i$, the derivative is $p_i-q_i$. If an identical, unnoised student
already predicts $p_i=q_i$, this gradient is zero. The cross-entropy
value is still $H(q_i)$, not zero unless the target is one-hot. KL
divergence would be zero. Noise can make the student's prediction differ
from the clean target and create a consistency gradient; it does not
make a wrong target correct.

The following program compares those cases, then performs one joint
student update. Its two-dimensional input changes and dropout isolate
the mechanism. They are not RandAugment, stochastic depth, an EfficientNet
training run or evidence of improved ImageNet accuracy.

```python
import copy
import torch
import torch.nn.functional as F

torch.manual_seed(31)
teacher = torch.nn.Sequential(
    torch.nn.Linear(2, 4), torch.nn.Tanh(),
    torch.nn.Dropout(.5), torch.nn.Linear(4, 2)
).double()
teacher.eval()
student = copy.deepcopy(teacher)
u = torch.tensor([[1., .2], [-.8, .4]], dtype=torch.float64)
with torch.no_grad():
    q = teacher(u).softmax(-1)
teacher_before = [p.detach().clone() for p in teacher.parameters()]

student.eval()
clean_ce = -(q * student(u).log_softmax(-1)).sum(-1).mean()
entropy = -(q * q.log()).sum(-1).mean()
torch.testing.assert_close(clean_ce.detach(), entropy)
assert clean_ce.item() > 0
clean_grad = torch.autograd.grad(clean_ce, tuple(student.parameters()))
assert max(g.abs().max().item() for g in clean_grad) < 1e-12

student.train()
augmented = u + u.new_tensor([[.4, -.2], [-.3, .1]])
human_x = u.new_tensor([[.6, -.5]])
human_y = torch.tensor([1])
optimizer = torch.optim.SGD(student.parameters(), lr=.1)
optimizer.zero_grad()
lab = F.cross_entropy(student(human_x), human_y)
unl = -(q * student(augmented).log_softmax(-1)).sum(-1).mean()
ratio = 2.
loss = lab + ratio * unl
combined_mean = (lab + 2 * unl) / 3
torch.testing.assert_close(loss, 3 * combined_mean)
loss.backward()
assert not q.requires_grad
assert all(p.grad is None for p in teacher.parameters())
assert any(p.grad is not None and p.grad.norm() > 0 for p in student.parameters())
before = [p.detach().clone() for p in student.parameters()]
optimizer.step()
assert any(not torch.equal(a, b) for a, b in zip(before, student.parameters()))
assert all(torch.equal(a, b) for a, b in zip(teacher_before, teacher.parameters()))
print("clean CE / target entropy", clean_ce.item(), entropy.item())
print("labelled / pseudo-labelled losses", lab.item(), unl.item())
```

Algorithm 1 writes a sum of two dataset means. The released
[loss assembly](https://github.com/google-research/noisystudent/blob/aac52bf02448f6e45aee6c418f3e588803e4fa43/main.py)
weights the unlabelled mean by its batch-size ratio $r$, giving
$\mathcal L_{\mathrm{lab}}+r\mathcal L_{\mathrm{unl}}$ before weight
decay. At $r=2$, this is three times the mean over one labelled and two
unlabelled examples. The relative supervision weight and overall gradient
scale must both be specified. The program uses that reduction and omits
the implementation's label smoothing and weight decay.

### Filtering, balancing and another training round

The paper's ImageNet protocol starts with roughly 300 million JFT images,
whose existing labels are ignored. In
[§3.1](https://arxiv.org/html/1911.04252v4#S3.SS1),
an EfficientNet-B0 filters by confidence above 0.3. At most 130,000
high-confidence images are retained per predicted class; underfilled
classes are duplicated to reach the same count. The resulting 130 million
training entries contain about 81 million unique images. Pool size,
retained entries and unique examples are three different counts.
Duplicating a confidently wrong pseudo-label increases its training
weight without adding independent evidence.

The final iterative run uses B7 as teacher, then L2 as student and as
subsequent teacher. [Table 7](https://arxiv.org/html/1911.04252v4#S4)
reports 87.6%, 88.1% and 88.4% ImageNet top-1 accuracy across three
rounds. The last round also changes the unlabelled-to-labelled batch ratio
from 14:1 to 28:1, so its gain cannot be attributed only to replacing the
teacher. These are historical vision results, not an NLP benchmark or a
universal reason to choose those ratios.

For language applications, Luong and Hu's
[2021 account of Semi-Supervised Distillation](https://research.google/blog/from-vision-to-language-semi-supervised-learning-in-action-at-scale/)
describes a related two-stage procedure: self-train an equal-or-larger
teacher, then distil it into a smaller production student. Their SSD
variant removes the Noisy Student input-noise requirement for the described
Search application. This is a distinct adaptation; it does not establish
that the original image augmentations transfer directly to arbitrary NLP
tasks. More generally, knowledge distillation can itself use augmentation
and regularization, so the presence of noise does not define every
distillation method.

<!-- END REVIEW: Noisy Student 2026-10-29 -->

<a id="consistency"></a>
## Consistency under an augmentation

An image and a mildly shifted version of it often have the same class.
For such an augmentation, the model can be trained to make compatible
predictions on the two views even if no class label has been supplied.
One consistency objective compares predicted distributions:

$$
 \mathcal L_{\mathrm{cons}}
 =\frac1M\sum_{b=1}^{M}
   D\bigl(q_b, p_\theta(y\mid A(u_b))\bigr).
$$

Here $u_b$ is an unlabelled input, $A$ constructs another view, $q_b$ is a
target distribution obtained from a first view, and $D$ measures their
disagreement. Depending on the method, the target may come from the same
model, an earlier prediction or an averaged teacher. These are choices
about the target, not interchangeable implementations of one algorithm.

Predicting the same class for every image also achieves perfect
consistency. The labelled loss anchors the classes, while the augmentation
assumption determines where those labels may be extended. A transformation
that changes the answer supplies a wrong invariance constraint. Removing
negation from a text or replacing an answer-bearing number is not
automatically a valid augmentation for the original classification task.

In FixMatch, a weak view supplies a confident hard target and a strong view
must predict that target. The authors use horizontal flip and translation
for weak image augmentation, with no horizontal flips for SVHN digits.
Strong augmentation uses RandAugment or CTAugment followed by Cutout.
The names “weak” and “strong” describe transformations in this image
protocol, not a universally valid pair of text transformations.

<!-- BEGIN REVIEW: VAT and UDA 2026-10-29 -->
<a id="virtual-adversarial-training"></a>
## VAT: consistency in the most sensitive local direction

For an unlabelled input, there is no human target with which to compute an
adversarial cross-entropy gradient. Virtual adversarial training uses the
model's current distribution as a fixed target:
$q=p_{\hat\theta}(\cdot\mid s)$. Define

$$
\begin{aligned}
 D(r)&=D_{\mathrm{KL}}
     \bigl(q\|p_{\hat\theta}(\cdot\mid s+r)\bigr),\\
 r_{\mathrm{vadv}}&\approx
      \arg\max_{\|r\|_2\le\varepsilon} D(r),\\
 \mathcal L_{\mathrm{VAT}}(\theta)&=
      D_{\mathrm{KL}}\bigl(q\|p_\theta(\cdot\mid s+r_{\mathrm{vadv}})\bigr).
\end{aligned}
$$

Both $q$ and the selected $r_{\mathrm{vadv}}$ are treated as constants
when differentiating the outer loss with respect to $\theta$.
The labelled loss still anchors class identities. The consistency
regularizer can be averaged over labelled and unlabelled inputs. Agreement
with an incorrect clean prediction does not establish that either answer
is correct. This construction and the approximation below follow
[Miyato et al., VAT, §III](https://arxiv.org/html/1704.03976v2#S3);
the embedding-space application is specified in
[Miyato, Dai and Goodfellow, §§2–3](https://arxiv.org/html/1605.07725v4#S2).

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/consistency-training-2026-10-29/vat.gif|Original author animation of VAT predictions and local distributional smoothness]]

*Takeru Miyato, original `vat.gif` from
[vat_tf, revision c5125d2](https://github.com/takerum/vat_tf/blob/c5125d267531ce0f10b2238cf95604d287de63c8/vat.gif),
MIT; [license copy](/licenses/vat-tf-c5125d2-MIT.txt).
The complete 14-frame, 320×180 animation is unchanged and is not upscaled.
The left panel shows predictions and the right panel LDS, the local
distributional-smoothness penalty; blue marks larger local changes than
grey. The coloured diamonds supply labelled
anchors; the surrounding points also affect consistency without receiving
human labels. This synthetic two-dimensional illustration does not measure
the accuracy of a text classifier.*

### Why the gradient at the clean input cannot select a direction

At $r=0$, the two distributions agree, so $D(0)=0$. KL is nonnegative.
For a differentiable model, zero is therefore a local minimum and
$\nabla_rD(0)=0$. An ordinary normalized first-gradient step has no direction
to normalize. The local quadratic approximation instead uses the Hessian:

$$
 D(r)\approx\frac12r^\top Hr,\qquad
 H=\left.\nabla_r^2D(r)\right|_{r=0}.
$$

Within an $\ell_2$ ball, this quadratic is largest along a dominant
eigenvector of the positive-semidefinite $H$. Constructing a full
$TD\times TD$ matrix for $T$ token vectors of dimension $D$ is unnecessary.
For a unit vector $d$ and a small positive $\xi$,

$$
 Hd\approx\frac{\nabla_rD(\xi d)-\nabla_rD(0)}{\xi}
          =\frac{\nabla_rD(\xi d)}{\xi}.
$$

After normalization, the positive factor $1/\xi$ cancels. Start from a
random unit $d$, compute the gradient at $r=\xi d$, normalize it, and
repeat for the chosen power-iteration budget. The final perturbation
has magnitude $\varepsilon$. One iteration is an approximation, not an
exact solution of the nonlinear inner maximization. A zero Hessian,
a degenerate leading eigenspace or an uninformative initial direction
needs explicit handling.

A binary classifier makes the calculation visible. Let
$p(1\mid s)=\sigma(w^\top s)$, with $w=(2,1)$ and $s=(0,0)$.
The clean target is $(1/2,1/2)$, and

$$
 H=\frac14ww^\top=
 \begin{pmatrix}1&1/2\\1/2&1/4\end{pmatrix}.
$$

Its eigenvalues are $0$ and $1.25$. A unit direction parallel to $(2,1)$
changes the logit; the perpendicular direction $(1,-2)/\sqrt5$ does not.
For $\varepsilon=0.2$, the first direction produces logit
$0.2\sqrt5$ and divergence
$\log\cosh(0.1\sqrt5)$. The perpendicular perturbation gives zero
divergence in this linear model. A random direction need not expose the
largest change even when its norm is identical.

```python
import math
import torch

w = torch.tensor([2., 1.], dtype=torch.float64)
q = torch.tensor([.5, .5], dtype=torch.float64)
def divergence(r):
    logits = torch.stack([r.new_zeros(()), w @ r])
    return (q * (q.log() - logits.log_softmax(0))).sum()

zero = torch.zeros(2, dtype=torch.float64, requires_grad=True)
gradient, = torch.autograd.grad(divergence(zero), zero)
torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-14, rtol=0)
hessian = torch.autograd.functional.hessian(divergence, zero)
torch.testing.assert_close(hessian, .25 * torch.outer(w, w))
eigenvalues = torch.linalg.eigvalsh(hessian)
torch.testing.assert_close(eigenvalues, torch.tensor([0., 1.25], dtype=w.dtype))

d = torch.tensor([1., -1.], dtype=w.dtype)
d /= d.norm()
probe = (.001 * d).requires_grad_()
g, = torch.autograd.grad(divergence(probe), probe)
direction = (g / g.norm()).detach()
torch.testing.assert_close(direction.abs(), (w / w.norm()).abs())
radius = .2
sensitive = radius * direction
perpendicular = radius * torch.tensor([1., -2.], dtype=w.dtype) / math.sqrt(5)
expected = math.log(math.cosh(.1 * math.sqrt(5)))
assert abs(divergence(sensitive).item() - expected) < 1e-12
assert abs(divergence(perpendicular).item()) < 1e-14
print("Hessian", hessian.tolist(), "eigenvalues", eigenvalues.tolist())
print("sensitive / perpendicular KL",
      divergence(sensitive).item(), divergence(perpendicular).item())
```

The sign of a quadratic eigenvector is arbitrary; finite-radius nonlinear
losses need not be symmetric. Increasing the number of power iterations
improves a direction estimate under the local spectral assumptions, but
does not certify a global adversary for an arbitrary neural network.
The probe radius $\xi$, perturbation radius $\varepsilon$,
regularizer weight and number of iterations have different roles.
An experiment that fixes some of them has not eliminated them from the
algorithm.

### A complete embedding-space VAT update

For text, the continuous variable $s$ is obtained after embedding lookup.
The normalization and sign convention of labelled adversarial training
are developed in
[[02 Areas/ML & DL/00 Учебник/19 Deployment, Reliability и MLOps/05 Устойчивость ML-систем#embedding-adversarial-training|embedding-space adversarial regularization]].
The original text implementation masks both padding and EOS while
constructing its perturbation. A budget over all coordinates of one
sequence is different from an independent budget for every token.

The following small classifier uses masked mean pooling and has no EOS,
dropout or batch normalization. Its perturbations exclude padding and
have one $\ell_2$ budget per sequence. Vocabulary-frequency normalization
is omitted in this fixture. These choices isolate gradient flow; they
do not reproduce the paper's LSTM training protocol.

```python
import torch
import torch.nn.functional as F

torch.manual_seed(19)
dtype = torch.float64
embedding = torch.nn.Embedding(6, 2, padding_idx=0).to(dtype)
head = torch.nn.Linear(2, 2).to(dtype)
parameters = list(embedding.parameters()) + list(head.parameters())
optimizer = torch.optim.SGD(parameters, lr=.1)
unlabelled = torch.tensor([[1, 2, 0], [3, 0, 0], [4, 5, 0]])
labelled = torch.tensor([[1, 2, 0], [3, 4, 0]])
human_targets = torch.tensor([0, 1])
mask = (unlabelled != 0)[..., None]

def classify(vectors, valid):
    count = valid.sum(dim=1)
    assert (count > 0).all()
    return head((vectors * valid).sum(dim=1) / count)

def unit(vector):
    vector = vector * mask
    norm = vector.flatten(1).norm(dim=1)[:, None, None]
    return vector / norm.clamp_min(1e-12)

def kl(target, logits):
    return (target * (target.log() - logits.log_softmax(dim=-1))).sum(-1).mean()

optimizer.zero_grad()
search_vectors = embedding(unlabelled).detach()
q = classify(search_vectors, mask).softmax(-1).detach()
probe = (.001 * unit(torch.randn_like(search_vectors))).detach().requires_grad_()
search_loss = kl(q, classify(search_vectors + probe, mask))
g, = torch.autograd.grad(search_loss, probe)
r = (.3 * unit(g)).detach()
assert all(p.grad is None for p in parameters)
assert torch.equal(r[~mask.expand_as(r)], torch.zeros_like(r[~mask.expand_as(r)]))
torch.testing.assert_close(r.flatten(1).norm(dim=1), r.new_full((3,), .3))

# Only this outer graph updates the embedding table and classifier.
clean_labelled = classify(embedding(labelled), (labelled != 0)[..., None])
supervised = F.cross_entropy(clean_labelled, human_targets)
vat = kl(q, classify(embedding(unlabelled) + r, mask))
loss = supervised + vat
before = [p.detach().clone() for p in parameters]
loss.backward()
assert not q.requires_grad and not r.requires_grad
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
assert embedding.weight.grad[0].count_nonzero() == 0
optimizer.step()
assert any(not torch.equal(a, b) for a, b in zip(before, parameters))
print("supervised / VAT", supervised.item(), vat.item())
print("perturbation norms", r.flatten(1).norm(dim=1).tolist())
```

The inner derivative is taken only with respect to the probe; it does not
accumulate an optimizer gradient in the parameter arrays. The outer pass
recomputes embeddings, so the shared table remains trainable. The clean
distribution and selected perturbation are held fixed for this outer
derivative. The original
[VAT implementation](https://github.com/takerum/vat_tf/blob/c5125d267531ce0f10b2238cf95604d287de63c8/vat.py)
uses the same stop-gradient separation.

For $K$ search iterations, the VAT paper counts $K$ search
backpropagations, one regularizer backpropagation and one supervised
backpropagation. Shared or batched execution can change actual runtime.
With independent dropout samples or changing batch-normalization state,
two predictions at the same unperturbed input may already differ;
a search must specify how that stochastic state is controlled. A small
loss at the selected perturbation is not a certificate against all
embedding perturbations, much less against arbitrary changes of text.

<a id="uda"></a>
## UDA: consistency across augmented text

Embedding perturbations need not produce readable text. UDA instead
constructs another input, such as a back-translated sentence, and trains
the prediction on that input to match a fixed distribution from the
original. The label-preserving assumption is about the augmentation:
changing “the treatment did not help” to “the treatment helped” would
supply a wrong consistency target even if both strings are fluent.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/consistency-training-2026-10-29/uda-fixmatch.png|Original UDA and FixMatch text-training comparison with complete caption]]

*Po-Wei Huang and co-authors,
[Lightweight Contextual Logical Structure Recovery, Figure 2](https://aclanthology.org/2022.sdp-1.5.pdf#page=5),
[CC BY 4.0](https://aclanthology.org/faq/copyright/).
Complete original figure and caption rendered at 400 dpi. This is the
authors' text-domain adaptation: EDA on the right is their weak text
augmentation, not part of FixMatch's original image protocol.
Within each method the repeated backbone boxes share parameters.
The left branch compares distributions; the right creates hard
pseudo-labels. Gradient stops, confidence gates and the KL direction
are specified by the equations and original code below, not by the
arrows alone.*

The original UDA
[method, §§2 and A](https://arxiv.org/html/1904.12848v6#S2)
adds a consistency term to supervised cross-entropy. For $M$ unlabelled
inputs, obtain clean probabilities $q_i$, augmented predictions $p_i$
and, optionally, sharpened clean targets:

$$
\begin{aligned}
 q_i&=p_{\hat\theta}(\cdot\mid u_i),\\
 \tilde q_{ic}
   &=\frac{\exp(z_{ic}/T)}
           {\sum_j\exp(z_{ij}/T)},\\
 m_i&=\mathbf1[\max_c q_{ic}>\beta],\\
 p_i&=p_\theta(\cdot\mid A(u_i)),\\
 \mathcal L_u&=\frac1M\sum_i m_i
       D_{\mathrm{KL}}(\tilde q_i\|p_i).
\end{aligned}
$$

The target $\tilde q_i$ and gate $m_i$ are held fixed in the outer
derivative; the augmented prediction $p_i$ supplies the parameter gradient.
Here $z_i$ contains the clean logits, $T$ is the target-distribution
temperature and $\beta$ the confidence threshold. The gate uses
**unsharpened** probabilities. Otherwise,
sharpening could make an uncertain prediction pass the same gate that
was meant to reject it. The original text
[implementation](https://github.com/google-research/uda/blob/960684e363251772a5938451d4d2bc0f1da9e24b/text/uda.py)
uses strict `>` for this confidence test and divides by the entire
unlabelled batch, including rejected examples.

The paper writes cross-entropy for the soft-target objective; the
released text implementation computes KL. With the target detached,
$\mathrm{CE}(\tilde q,p)=H(\tilde q)+D_{\mathrm{KL}}(\tilde q\|p)$.
The two losses have the same gradient with respect to the prediction
logits, but different numeric values. Neither is identical to replacing
$\tilde q$ by its most likely class.

For clean predictions $(0.9,0.1)$, $(0.8,0.2)$ and $(0.6,0.4)$, let
$\beta=0.8$ and $T=0.5$. Only the first example passes. Squaring and
renormalizing its probabilities gives $(81/82,1/82)$. If the augmented
prediction is $(0.6,0.4)$, its logit gradient is
$\bigl((0.6,0.4)-(81/82,1/82)\bigr)/3$.
The other two examples have zero masked gradients.

```python
import torch
import torch.nn.functional as F

clean_logits = torch.tensor([[.9, .1], [.8, .2], [.6, .4]],
                            dtype=torch.float64).log().requires_grad_()
aug_logits = torch.tensor([[.6, .4], [.3, .7], [.2, .8]],
                          dtype=torch.float64).log().requires_grad_()
q = clean_logits.detach().softmax(-1)
target = (clean_logits.detach() / .5).softmax(-1)
mask = q.max(-1).values > .8
per_example = (target * (target.log() - aug_logits.log_softmax(-1))).sum(-1)
loss = (mask * per_example).mean()
loss.backward()
assert mask.tolist() == [True, False, False]
torch.testing.assert_close(target[0], target.new_tensor([81/82, 1/82]))
expected = (aug_logits.detach().softmax(-1) - target) * mask[:, None] / 3
torch.testing.assert_close(aug_logits.grad, expected)
assert clean_logits.grad is None
cross_entropy = -(target * aug_logits.detach().log_softmax(-1)).sum(-1)
entropy = -(target * target.log()).sum(-1)
torch.testing.assert_close(cross_entropy, entropy + per_example.detach())
print("accepted", mask.tolist(), "KL", loss.item(), "gradient", aug_logits.grad.tolist())
```

### How the text augmentation is constructed

UDA uses English–French and French–English translation models to obtain
paraphrases. It samples translations with a decoding temperature instead
of always choosing the best beam. Generating the same paraphrase repeatedly
adds little diversity; sampling too aggressively can change the meaning.
The decoding temperature controlling this transformation is distinct
from the target-softmax temperature $T$ above. The translation systems
were trained on parallel data: “unlabelled” here means no target
classification labels for each augmented review, not an absence of
supervision anywhere in the pipeline.

For topic classification, the paper also develops TF-IDF-based word
replacement. Its original recipe selects word positions before WordPiece
segmentation, rather than perturbing a subword embedding vector.
Let $a_i$ be a word's TF-IDF score in the sentence,
$C=\max_i a_i$, and $Z=\frac1n\sum_i(C-a_i)$. The replacement probability is

$$
 \pi_i=\min\left(\rho\frac{C-a_i}{Z},1\right).
$$

Low-scoring positions are replaced more often. Replacement words are
sampled with weights proportional to $S_{\max}-S(w)$, where
$S(w)=\operatorname{freq}(w)\operatorname{IDF}(w)$ is computed over the
corpus. The two distributions answer different questions: which
position to change and which vocabulary word to insert.

For sentence scores $(3,1,0)$ and $\rho=0.7$, $Z=5/3$ and the probabilities
are $(0,0.84,1)$. Clipping changes the expected replacement count from
$\rho n=2.1$ to $1.84$. With vocabulary scores $(30,10,0)$, the insertion
distribution is $(0,0.4,0.6)$. These are constructed scores for checking
the formulas, not scores of particular English words.

```python
import numpy as np

score = np.array([3., 1., 0.])
difference = score.max() - score
probability = np.minimum(.7 * difference / difference.mean(), 1.)
np.testing.assert_allclose(probability, [0., .84, 1.])
assert np.isclose(probability.sum(), 1.84)
vocabulary_score = np.array([30., 10., 0.])
weight = vocabulary_score.max() - vocabulary_score
np.testing.assert_allclose(weight / weight.sum(), [0., .4, .6])
print("position probabilities", probability, "expected changes", probability.sum())
```

Equal TF-IDF scores give $Z=0$; equal vocabulary scores give a zero
sampling normalizer. A production implementation needs a stated fallback
for these cases. More generally, low TF-IDF does not imply that a word is
irrelevant to the target: negation can reverse sentiment. Validate the
augmentation on examples with known labels and inspect meaning-changing
failures. Fluent output and a retained keyword do not prove invariance.

<a id="training-signal-annealing"></a>
### Training Signal Annealing acts on labelled examples

A small labelled set can be revisited many times while the much larger
unlabelled pool is still being processed. UDA's Training Signal Annealing
(TSA) suppresses the labelled loss for examples already predicted
confidently **in their correct class**. With $C$ classes and training
progress $u=t/T_{\mathrm{steps}}$, its threshold is

$$
 \eta_t=\frac1C+\left(1-\frac1C\right)a(u).
$$

The three schedules in the paper and code use
$a(u)=u$, $a(u)=e^{5(u-1)}$ and $a(u)=1-e^{-5u}$.
The exponential schedule releases most labelled examples late; the
logarithmic schedule raises the ceiling earlier. The formulas approach
the endpoints differently: the exponential threshold starts slightly
above $1/C$, and the logarithmic threshold ends slightly below 1.

The text implementation keeps an example when
$p_\theta(y_i\mid x_i)\le\eta_t$ and normalizes over the **kept labelled
examples**:

$$
 b_i=\mathbf1[p_\theta(y_i\mid x_i)\le\eta_t],\qquad
 \mathcal L_s=
 \frac{\sum_i b_i[-\log p_\theta(y_i\mid x_i)]}
      {\max(1,\sum_i b_i)}.
$$

This is the opposite selection pressure from unlabelled confidence
masking: TSA withholds already-easy human-labelled examples, whereas
the unlabelled gate accepts only confident model-generated targets.
The original implementation uses different denominators for the two
terms. Changing both to an average over accepted examples changes
the unlabelled regularizer's effective weight.

For three classes and halfway through a linear schedule, the ceiling
is $2/3$. Suppose the probabilities assigned to the human targets are
$0.9$, $0.2$ and $0.5$. TSA retains the second and third examples,
giving $(-\log0.2-\log0.5)/2\approx1.151293$.
The second prediction can have maximum probability $0.7$ in a wrong
class. Testing that maximum instead of the human-target probability
would discard a confidently wrong example that needs labelled correction.

```python
import math
import torch
import torch.nn.functional as F

probability = torch.tensor([[.9, .06, .04], [.7, .2, .1], [.2, .3, .5]],
                           dtype=torch.float64)
labels = torch.tensor([0, 1, 2])
logits = probability.log().requires_grad_()
ce = F.cross_entropy(logits, labels, reduction="none")
threshold = 1/3 + (1 - 1/3) * .5
correct = logits.detach().softmax(-1)[torch.arange(3), labels]
keep = correct <= threshold
loss = (ce * keep).sum() / keep.sum().clamp_min(1)
loss.backward()
assert keep.tolist() == [False, True, True]
assert abs(loss.item() - (-math.log(.2) - math.log(.5))/2) < 1e-12
assert (probability.max(-1).values <= threshold).tolist() == [False, False, True]
torch.testing.assert_close(logits.grad[0], torch.zeros(3, dtype=logits.dtype))

# No retained labelled example gives zero loss and zero gradient.
all_easy = torch.tensor([[.9, .1], [.95, .05]], dtype=torch.float64).log()
all_easy.requires_grad_()
gate = all_easy.detach().softmax(-1)[:, 0] <= .5
empty = (F.cross_entropy(all_easy, torch.zeros(2, dtype=torch.long),
                         reduction="none") * gate).sum() / gate.sum().clamp_min(1)
empty.backward()
assert empty.item() == 0 and all_easy.grad.count_nonzero() == 0
print("TSA mask", keep.tolist(), "labelled loss", loss.item())
```

The original
[`get_tsa_threshold` and `create_model`](https://github.com/google-research/uda/blob/960684e363251772a5938451d4d2bc0f1da9e24b/text/uda.py)
provide the exact gate, temperature and reduction semantics. The examples
above execute these computations on specified arrays; they do not run
a translation system or reproduce IMDb performance.

The initialization also changes the result. The IMDb entries in
[UDA, Table 4](https://arxiv.org/html/1904.12848v6#S4.T4) are test-error
percentages; both rows below use 20 labelled reviews and sequence
length 512:

| Initialization | Without UDA | With UDA |
| --- | ---: | ---: |
| BERT LARGE | 11.72 | 4.78 |
| BERT LARGE additionally adapted on in-domain unlabelled text | 6.50 | 4.20 |

Thus 4.20% includes both in-domain adaptation and UDA. The fully labelled
BERT LARGE reference uses 25,000 labelled reviews and reports 4.51%.
Comparing 4.20 with 4.51 mixes different supervision and initialization
protocols; comparing 6.50 with 4.20 isolates the UDA addition within
the paper's in-domain-adapted, 20-label setting. These published numbers
are not results of the small programs here. FixMatch, described below,
uses a thresholded **hard** target from a weak view rather than UDA's
soft distribution target.

<!-- END REVIEW: VAT and UDA 2026-10-29 -->

<!-- BEGIN REVIEW: MixText 2026-10-29 -->
<a id="mixtext"></a>
## MixText: interpolate hidden states and their targets

A back-translation produces another sentence. A convex combination of
token identifiers does not: averaging identifiers 10 and 20 gives
identifier 15, whose meaning depends on an arbitrary vocabulary order.
[Chen, Yang and Yang, §§3–4](https://aclanthology.org/2020.acl-main.194/)
instead mix continuous encoder states. Their supervised augmentation is
called **TMix**. **MixText** adds unlabelled examples, guessed soft labels
and further regularization around that operation.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/semi-supervised-extensions-2026-10-29/tmix-original.png|Original TMix hidden-state interpolation and complete caption]]

*Jiaao Chen, Zichao Yang and Diyi Yang,
[MixText, Figure 1](https://aclanthology.org/2020.acl-main.194.pdf#page=1),
[CC BY 4.0](https://aclanthology.org/faq/copyright/).
Both lower branches use the shared encoder; after mixing, only one
branch continues through the upper layers. Complete original figure
and caption, rendered at 240 dpi. The PDF's embedded diagram is not
enlarged beyond its native pixel dimensions.*

### Where the two encoder branches meet

Run both inputs through the same first $m$ layers, obtaining
$h_m^a$ and $h_m^b$. Mix their states, then run the mixed state through
the remaining layers and classifier:

$$
\begin{aligned}
 \tilde h_m&=\lambda h_m^a+(1-\lambda)h_m^b,\\
 \tilde y&=\lambda y_a+(1-\lambda)y_b.
\end{aligned}
$$

The same coefficient must be used for states and targets. With
$h_m^a=(2,0)$, $h_m^b=(0,2)$, $\lambda=0.75$ and opposite one-hot
targets, the mixed state is $(1.5,0.5)$ and the target is $(0.75,0.25)$.
The upper network predicts a probability distribution and is trained
against that soft target. A nonlinear upper network need not already
predict the target, or even the weighted average of its two endpoint
predictions. The loss encourages the desired interpolation behaviour.

The paper samples one coefficient per batch from
$\operatorname{Beta}(\alpha,\alpha)$ and folds it with
$\lambda\leftarrow\max(\lambda,1-\lambda)$, so the first branch has at
least half the weight. Large $\alpha$ concentrates the draw near
$1/2$; small $\alpha$ puts more draws near the endpoints, and folding
maps either endpoint towards 1. The chosen BERT-base mixing-layer set is
$\{7,9,12\}$, sampled per batch. Those layer numbers are an experimental
choice for this encoder, not a universal taxonomy of syntax and meaning.

The program below uses one state vector per input, a shared trainable
lower projection and a nonlinear upper network. If $g$ is the gradient
at the mixed state, the two incoming gradients must be $\lambda g$ and
$(1-\lambda)g$. Both contribute to the shared lower parameters.

```python
import torch
import torch.nn.functional as F

dtype = torch.float64
lower = torch.eye(2, dtype=dtype, requires_grad=True)
upper = torch.tensor([[1., -.5], [-.7, 1.2]], dtype=dtype, requires_grad=True)
a, b = torch.tensor([[2., 0.], [0., 2.]], dtype=dtype)
ha, hb = lower @ a, lower @ b
ha.retain_grad()
hb.retain_grad()
lam = .75
mixed = lam * ha + (1 - lam) * hb
mixed.retain_grad()
target = torch.tensor([.75, .25], dtype=dtype)
logits = upper @ mixed.tanh()
loss = -(target * logits.log_softmax(0)).sum()
loss.backward()
torch.testing.assert_close(mixed.detach(), mixed.new_tensor([1.5, .5]))
torch.testing.assert_close(ha.grad, lam * mixed.grad)
torch.testing.assert_close(hb.grad, (1 - lam) * mixed.grad)
torch.testing.assert_close(lower.grad,
                          torch.outer(ha.grad, a) + torch.outer(hb.grad, b))
endpoint_probability = (
    lam * (upper.detach() @ ha.detach().tanh()).softmax(0)
    + (1 - lam) * (upper.detach() @ hb.detach().tanh()).softmax(0))
assert not torch.allclose(logits.detach().softmax(0), endpoint_probability)
assert lower.grad.norm() > 0 and upper.grad.norm() > 0
print("mixed state / target", mixed.detach().tolist(), target.tolist())
print("mixed prediction / endpoint average",
      logits.detach().softmax(0).tolist(), endpoint_probability.tolist())
```

This illustrates an interpolation gradient, not a trained BERT.
In a Transformer the mixed tensors contain all token positions, not
just the pooled classification vector. Shapes, padding and attention
masks must be specified. The released
[encoder](https://github.com/SALT-NLP/MixText/blob/f17198d98e1bbb012d8e33cd26e228dd43bb4673/code/mixtext.py)
mixes token-state tensors at the selected layer and mean-pools the final
sequence. It creates all-ones attention masks when none are supplied.
That implementation choice should not be mistaken for a proof that
padding can be ignored, or that positions in two unrelated sentences
have a token-by-token semantic alignment.

### Guessing a shared target for several views

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/semi-supervised-extensions-2026-10-29/mixtext-original.png|Original complete MixText training framework and caption]]

*Chen, Yang and Yang,
[MixText, Figure 2](https://aclanthology.org/2020.acl-main.194.pdf#page=5),
[CC BY 4.0](https://aclanthology.org/faq/copyright/).
The red classifier boxes share parameters. Their predictions supply
the weighted average and sharpened target; the upper path mixes the
resulting labelled and pseudo-labelled pool. Complete original figure
and caption, rendered at 240 dpi without redrawing.*

For an unlabelled sentence, the paper constructs German and Russian
back-translations. Let $p_k$ be predictions on the original and those
views, with nonnegative quality weights $w_k$ and positive total weight.
The guessed distribution is their weighted mean, followed by sharpening:

$$
\begin{aligned}
 q&=\frac{\sum_k w_kp_k}{\sum_k w_k},\\
 \hat q_c&=\frac{q_c^{1/T}}{\sum_jq_j^{1/T}}.
\end{aligned}
$$

The original and its augmentations receive the same guessed target.
They join the human-labelled pool, so TMix can pair labelled inputs
with each other, labelled with unlabelled, or two unlabelled inputs.
Pseudo-label predictions are detached when used as targets; gradients
still train the encoder through the mixed-input branches.

For predictions $(0.9,0.1)$, $(0.5,0.5)$ and $(0.1,0.9)$, weights
$(1,1,0)$ discard the third view and give $q=(0.7,0.3)$. At $T=0.5$,
the sharpened target is $(49/58,9/58)$. Sharpening before averaging
would generally give a different distribution. As $T$ tends to zero,
probability concentrates on the largest components; a tie does not
select a unique one-hot label without an additional tie rule.
Lower entropy measures concentration, not correctness.

The paper's extra margin term uses the squared probability norm:

$$
 \mathcal L_{\mathrm{margin}}
 =\mathbb E_u\max\left(0,\gamma-\sum_c p_c(u)^2\right).
$$

For two classes and $\gamma=0.7$, uniform prediction has norm squared
$0.5$ and penalty $0.2$; $(0.9,0.1)$ has norm squared $0.82$ and
zero penalty. This is not Shannon entropy, even though both can be used
to encourage concentration. The probabilities in a trainable penalty
need a gradient path. Applying the penalty only to a detached guessed
target would not update the learner.

```python
import torch

dtype = torch.float64
views = torch.tensor([[.9, .1], [.5, .5], [.1, .9]], dtype=dtype,
                     requires_grad=True)
weight = torch.tensor([1., 1., 0.], dtype=dtype)
q = (weight[:, None] * views.detach()).sum(0) / weight.sum()
target = q.square() / q.square().sum()
torch.testing.assert_close(q, q.new_tensor([.7, .3]))
torch.testing.assert_close(target, q.new_tensor([49/58, 9/58]))
assert not target.requires_grad

logits = torch.tensor([[.5, .5], [.6, .4], [.9, .1]], dtype=dtype).log()
logits.requires_grad_()
p = logits.softmax(-1)
penalties = (.7 - p.square().sum(-1)).clamp_min(0)
penalties.mean().backward()
torch.testing.assert_close(penalties.detach(), p.new_tensor([.2, .18, 0.]))
torch.testing.assert_close(logits.grad[0], torch.zeros(2, dtype=dtype), atol=1e-14, rtol=0)
assert logits.grad[1].norm() > 0 and logits.grad[2].count_nonzero() == 0
assert views.grad is None
print("guessed / sharpened", q.tolist(), target.tolist())
print("margin penalties / gradients", penalties.detach().tolist(), logits.grad.tolist())
```

Uniform logits are stationary by symmetry even when this penalty is
positive; confidence regularization alone does not resolve which class
is correct. Human labels and the remaining objective supply information
missing from that symmetric case.

### Paper and released training code are different specifications

The pinned
[training code](https://github.com/SALT-NLP/MixText/blob/f17198d98e1bbb012d8e33cd26e228dd43bb4673/code/train.py)
contains configurable variants. Its current label-guessing expression
sets the two translation weights to zero and the original weight to one.
Comments give alternative dataset-specific weights; they are not active
just because translations were loaded. The usual hidden-mixing branch
uses a soft-target cross-entropy on labelled-majority samples and KL
on unlabelled-majority samples, with a ramped weight. Cross-entropy
and KL have the same prediction gradient for a fixed target but differ
by its entropy.

The same branch computes a Shannon-entropy hinge,
$\max(0,H(p)-\gamma)$, rather than the paper's squared-norm hinge.
Moreover, with the default mixing path, the training loop does not add
that computed extra term to its loss. Other flags activate different
branches. Reproducing a particular run therefore requires its commit,
arguments, target weights and actual loss assembly; importing a class
named SemiLoss does not determine them.

The label-guessing passes run inside no-grad while the model remains in
training mode. No-grad stops autograd recording; it does not disable
dropout. This is different from Noisy Student's clean inference-mode
teacher and should be preserved when comparing implementations.

[Table 2 of the ACL paper](https://aclanthology.org/2020.acl-main.194.pdf#page=7)
reports these AG News test accuracies with **10 labels per class**:
BERT 69.5%, supervised TMix 74.1%, the authors' UDA implementation 84.4%,
and MixText 88.4%. AG News has four classes, so that setting uses
40 human labels, not ten in total. The unlabelled pool has 5,000 examples
per class. Results are averaged over three runs; the UDA comparison uses
the authors' matched PyTorch reimplementation, not UDA's original
TPU/BERT-LARGE IMDb protocol. These results do not establish a universal
“works below 100 labels” rule across datasets or modern encoders.

<!-- END REVIEW: MixText 2026-10-29 -->

<a id="fixmatch"></a>
## FixMatch: weak target, strong prediction

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/semi-supervised-2026-10-28/fixmatch-diagram.png|Original FixMatch weak and strong augmentation diagram]]

*Kihyuk Sohn and co-authors,
[FixMatch, Figure 1](https://arxiv.org/pdf/2001.07685v2#page=2).
Complete original diagram extracted at 400 dpi, without redrawing or
upscaling the repository bitmap. The same illustration is released in the
[authors' repository](https://github.com/google-research/fixmatch/blob/d4985a158065947dba803e626ee9a6721709c570/media/FixMatch%20diagram%20white.png)
under Apache-2.0;
[license copy](/licenses/fixmatch-d4985a1-Apache-2.0.txt).
Both model boxes use the same current classifier. The upper prediction
becomes a hard target only when its confidence passes the threshold; the
lower prediction is trained against it. The dashed line in the upper
histogram is the threshold, not a second model or an accuracy measurement.*

Let a minibatch contain $B$ labelled examples $(x_b,y_b)$ and $M=\mu B$
unlabelled examples $u_b$. Denote weak augmentation by $\alpha$ and strong
augmentation by $A$. FixMatch obtains

$$
\begin{aligned}
 q_b&=p_\theta(y\mid\alpha(u_b)),\\
 \hat y_b&=\arg\max_c q_{b,c},\\
 m_b&=\mathbf1[\max_c q_{b,c}\ge\tau].
\end{aligned}
$$

The model is trained with the labelled and unlabelled losses

$$
\begin{aligned}
 \mathcal L_s&=-\frac1B\sum_{b=1}^B
       \log p_\theta(y_b\mid\alpha(x_b)),\\
 \mathcal L_u&=-\frac1M\sum_{b=1}^M m_b
       \log p_\theta(\hat y_b\mid A(u_b)),\\
 \mathcal L&=\mathcal L_s+\lambda_u\mathcal L_u.
\end{aligned}
$$

The target and gate are treated as fixed when differentiating the
unlabelled loss. In the authors'
[implementation](https://github.com/google-research/fixmatch/blob/d4985a158065947dba803e626ee9a6721709c570/fixmatch.py),
`tf.stop_gradient` is applied to weak-view probabilities; the hard labels
and mask then supervise the strong-view logits. The weak and strong
branches share model parameters. Stopping the target gradient does not
freeze those parameters: the labelled and strong-view losses still update
the classifier.

The denominator is the full unlabelled batch size $M$, including examples
whose mask is zero. If $K$ examples are accepted, then, for $K>0$,

$$
 \mathcal L_u=\frac KM\overline\ell_{\mathrm{accepted}}.
$$

Averaging over accepted examples instead would remove the $K/M$ factor
and increase their effective weight when few predictions pass the gate.
When none passes, the stated objective is zero and its strong-view
gradient is zero. No division by $K$ is required.

For four weak predictions, take $\tau=0.95$ and the following strong-view
probability of class 0:

| Example | Weak probabilities | Pseudo-label | Accepted? | Strong $p(0)$ | Contribution before division by 4 |
| --- | --- | ---: | --- | ---: | ---: |
| 0 | $(0.97,0.03)$ | 0 | yes | 0.6 | $-\log0.6$ |
| 1 | $(0.75,0.25)$ | 0 | no | 0.4 | 0 |
| 2 | $(0.08,0.92)$ | 1 | no | 0.8 | 0 |
| 3 | $(0.95,0.05)$ | 0 | yes | 0.2 | $-\log0.2$ |

Equality at the threshold is accepted. The unlabelled loss is
$(-\log0.6-\log0.2)/4\approx0.530066$; averaging over the two accepted
examples would give approximately $1.060132$ instead. The following
program checks both the value and the gradient paths.

```python
import torch
import torch.nn.functional as F

weak = torch.tensor([[.97, .03], [.75, .25], [.08, .92], [.95, .05]],
                    dtype=torch.float64, requires_grad=True)
strong = torch.tensor([[.6, .4], [.4, .6], [.8, .2], [.2, .8]],
                      dtype=torch.float64).log().requires_grad_()
target = weak.detach().argmax(dim=1)
mask = weak.detach().max(dim=1).values >= .95
per_example = F.cross_entropy(strong, target, reduction="none")
loss = (per_example * mask).mean()
loss.backward()
assert mask.tolist() == [True, False, False, True]
assert torch.isclose(loss.detach(), torch.tensor(.5300658841, dtype=torch.float64))
assert weak.grad is None
assert torch.equal(strong.grad[~mask], torch.zeros_like(strong.grad[~mask]))
expected = strong.detach().softmax(dim=1)
expected[torch.arange(4), target] -= 1.
expected *= mask[:, None] / 4.
torch.testing.assert_close(strong.grad, expected)
print("accepted", mask.tolist(), "loss", loss.item())
print("strong gradient", strong.grad)
```

This program starts from specified probability arrays to isolate the
objective. A complete training step must also obtain the two views from
the same input and compute their logits with the shared model.

<a id="training-step"></a>
## One complete, small training step

The next program trains a two-class linear classifier. The paired vectors
are fixed synthetic views for checking the computation; they are not
image augmentations or a reproduction of the paper's CNN experiment.
Human targets enter only through the two labelled examples.

```python
import torch
import torch.nn.functional as F

torch.manual_seed(7)
model = torch.nn.Linear(2, 2, dtype=torch.float64)
with torch.no_grad():
    model.weight.copy_(torch.tensor([[2., 0.], [-2., 0.]], dtype=torch.float64))
    model.bias.zero_()
labelled_x = torch.tensor([[1., .2], [-1., -.2]], dtype=torch.float64)
human_y = torch.tensor([0, 1])
weak_x = torch.tensor([[2., 0.], [-2., 0.], [.1, .2], [-.1, -.2]], dtype=torch.float64)
strong_x = torch.tensor([[1.4, .1], [-1.3, -.1], [.15, .1], [-.15, -.1]], dtype=torch.float64)
optimizer = torch.optim.SGD(model.parameters(), lr=.1)
before = [p.detach().clone() for p in model.parameters()]

optimizer.zero_grad()
with torch.no_grad():
    weak_probability = model(weak_x).softmax(dim=1)
    confidence, target = weak_probability.max(dim=1)
    mask = confidence >= .8
labelled_loss = F.cross_entropy(model(labelled_x), human_y)
strong_logits = model(strong_x)
unlabelled_loss = (F.cross_entropy(strong_logits, target, reduction="none") * mask).mean()
loss = labelled_loss + unlabelled_loss
loss.backward()
assert mask.tolist() == [True, True, False, False]
assert not weak_probability.requires_grad
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
optimizer.step()
assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
print("targets", target.tolist(), "mask", mask.tolist())
print("labelled loss", labelled_loss.item(), "unlabelled loss", unlabelled_loss.item())
```

FixMatch refreshes the pseudo-label from the current weak-view prediction
on each step, rather than permanently assigning a human-like label to
every example in the pool. The paper also permits labelled inputs to
appear in the unlabelled batch with their labels ignored. Counts of
labelled and unlabelled *views* therefore need not describe disjoint sets
of stored images.

The original implementation combines labelled, weak and strong views for
forward computation and handles batch normalization through interleaving.
It also uses weight decay, momentum SGD, a learning-rate schedule and an
exponential moving average for evaluation. An EMA evaluation model does
not mean an EMA teacher generated the pseudo-labels in the weak branch.
With unlabelled ratio $\mu=7$, one labelled batch of size $B$ produces
$B+2\mu B=15B$ input views: labelled, weak unlabelled and strong unlabelled.
At $B=64$ this is 960 views. Rejected pseudo-labels give zero masked loss,
but the original code still computes their strong-view predictions.
View counts are not a measured GPU-cost ratio; batching, backward computation
and batch normalization also affect the cost. The linear example has none
of these batch-normalization or image-pipeline details, so its successful
execution verifies only the stated small step.

<a id="confirmation-bias"></a>
## Confidence can reinforce a wrong class

If the model confidently assigns class 0 to a true class-1 example, the
pseudo-label loss pushes the strong-view prediction toward class 0.
The mask tests confidence, so it cannot detect this error without
independent information. The following calculation makes the feedback
visible by comparing the true-class probability before and after one
pseudo-label update.

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([[4., 0.]], dtype=torch.float64, requires_grad=True)
true_class = 1  # Known here only for checking the failure case.
before = logits.detach().softmax(dim=1)[0, true_class].item()
with torch.no_grad():
    confidence, pseudo = logits.softmax(dim=1).max(dim=1)
assert confidence.item() > .95 and pseudo.item() == 0
F.cross_entropy(logits, pseudo).backward()
with torch.no_grad():
    logits -= .5 * logits.grad
after = logits.detach().softmax(dim=1)[0, true_class].item()
assert after < before
print("true-class probability", before, "->", after)
```

Class imbalance can make the accepted pseudo-labelled batch more
imbalanced than the input pool: confidence may be higher on majority
classes. Report accepted counts and, where verified labels are available,
precision by class and relevant data group. A single overall acceptance
rate can hide a class that never receives useful pseudo-labels.

Changing a threshold is only one part of validation. The unlabelled pool
may contain classes absent from the labelled set, or inputs from another
distribution. Strong augmentation may remove evidence needed to classify
an input. Check these cases on a separate labelled evaluation set and
compare against a supervised run with the same architecture and training
budget. More accepted pseudo-labels do not by themselves establish a gain.

<a id="label-propagation"></a>
## Propagating labels through a similarity graph

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/semi-supervised-2026-10-28/circles-raw.png|Original two-circle pool with only two known labels]]

*The scikit-learn developers,
[Label Propagation circles: Learning a complex structure](https://scikit-learn.org/stable/auto_examples/semi_supervised/plot_label_propagation_structure.html),
BSD-3-Clause. All 200 points are observed, but only two have labels: one
on each circle. The remaining orange points are not a third class;
their labels are unknown. Complete original figure, unchanged.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/semi-supervised-2026-10-28/circles-spread.png|Original label-spreading assignments on the two circles]]

*The scikit-learn developers, the same
[circles example](https://scikit-learn.org/stable/auto_examples/semi_supervised/plot_label_propagation_structure.html),
BSD-3-Clause. The source code uses `LabelSpreading(kernel="knn", alpha=0.8)`,
despite the broader “Label Propagation” page title. The result illustrates
this favourable geometry, not a general guarantee of recovery from two labels.
Complete original figure, unchanged;
[license copy](/licenses/semi-supervised-sklearn-BSD-3-Clause.txt).*

A graph method connects similar examples and prefers assignments that vary
little across strong edges. Unlabelled points participate in constructing
paths through the data: two points on the same curved structure can be
connected by many short edges even when their direct Euclidean distance
is large. If graph edges join different classes, the same propagation
mechanism can spread incorrect labels.

<!-- BEGIN REVIEW: SSL structural assumptions 2026-10-29 -->
<a id="ssl-structural-assumptions"></a>
### What the graph assumes about labels

Semi-supervised smoothness connects nearby outputs along well-supported,
high-density regions. The cluster assumption expects one such connected
region to contain mostly one class; it does not require every class to
form a single cluster. Placing a boundary in a low-density gap avoids
splitting that region. These are assumptions about the relation between
input density and labels, not consequences of observing unlabelled points.

The manifold assumption concerns geometry: high-dimensional inputs may
concentrate near a lower-dimensional structure. In the circles example,
each curve is one-dimensional even though the coordinates are
two-dimensional. Short graph edges can approximate paths along a curve;
straight-line distance through empty space need not describe that path.
[Chapelle, Schölkopf and Zien, §1.2.1–1.2.3](https://www.cs.cmu.edu/~zhuxj/tmp/book.pdf#page=12)
distinguish these assumptions.

Low-dimensional geometry alone does not supply a class rule. If the
intended label changes halfway around one circle, smoothing the same
label around the whole curve is wrong. The representation, edge weights
and available labels must support the chosen smoothness constraint.

<!-- END REVIEW: SSL structural assumptions 2026-10-29 -->

### Harmonic label propagation

A hard-clamped harmonic version makes this assumption explicit. Let
$W=W^\top$ contain nonnegative edge weights, with zero diagonal, and let
$d_i=\sum_j W_{ij}$. For a binary problem, $f_i$ is the score of class 1.
Keep $f_i=y_i$ on labelled nodes and require every unlabelled score with
$d_i>0$ to be the weighted average of its neighbours:

$$
 f_i=\frac{\sum_j W_{ij}f_j}{d_i}.
$$

These equations minimize graph variation with the boundary labels fixed:

$$
\begin{aligned}
 E(f)&=\frac12\sum_{i<j}W_{ij}(f_i-f_j)^2,\\
 \frac{\partial E}{\partial f_i}
     &=d_if_i-\sum_jW_{ij}f_j.
\end{aligned}
$$

Only the unlabelled coordinates may vary. Setting their derivatives to
zero gives the neighbour-average equations, while the verified endpoint
values stay fixed. With graph Laplacian $L=D-W$, partition the rows and columns into
unlabelled $U$ and labelled $A$. The unknown scores solve

$$
 L_{UU}f_U=-L_{UA}y_A.
$$

For a chain of five nodes with unit edge weights and endpoint labels
$0$ and $1$, the interior equations are

$$
\begin{aligned}
 2f_1-f_2&=0,\\
 -f_1+2f_2-f_3&=0,\\
 -f_2+2f_3&=1.
\end{aligned}
$$

The solution is $(f_1,f_2,f_3)=(1/4,1/2,3/4)$. Each interior value is
the average of its two neighbours; the middle node is equally supported
by the two endpoint classes. Hard classification of its score needs an
explicit tie rule. Iterative averaging with endpoints restored after
each update converges to the same scores in this example.

```python
import numpy as np

W = np.zeros((5, 5))
for i in range(4):
    W[i, i+1] = W[i+1, i] = 1.
L = np.diag(W.sum(axis=1)) - W
U, A = [1, 2, 3], [0, 4]
y = np.array([0., 1.])
solution = np.linalg.solve(L[np.ix_(U, U)], -L[np.ix_(U, A)] @ y)
np.testing.assert_allclose(solution, [.25, .5, .75])

f = np.array([0., 0., 0., 0., 1.])
for _ in range(200):
    f = (W @ f) / W.sum(axis=1)
    f[A] = y  # Hard clamping of verified labels.
np.testing.assert_allclose(f[U], solution, atol=1e-12)
print("harmonic scores", solution.tolist())
```

Changing an edge weight changes how strongly neighbouring scores are
tied together. For the same chain with last-edge weight 10 instead of 1,
the interior scores become $10/31,20/31,30/31$. The larger weight makes
the last interior node closer in score to the endpoint labelled 1; the
remaining changes are distributed along the earlier edges. Similarity
construction therefore affects the prediction, not only runtime.

The linear system needs a labelled anchor in each connected component
of this undirected graph. A component containing only unlabelled nodes
has no boundary class information; its Laplacian block is singular.
Silently assigning such a component to a class does not recover a label
from the observed structure.

`LabelSpreading` in the original circles example is a related algorithm,
not this exact hard-clamped solver. It uses a normalized graph operator
and repeatedly combines neighbour information with initial label
information. Its parameter `alpha` controls that mixture. It may change
the distributions at initially labelled points, whereas hard clamping
restores those values exactly. The distinction matters when comparing
algorithms or handling noisy initial labels.

RBF similarities usually create a dense graph. Storing one float32
weight for every ordered pair of one million points alone requires
$4\times10^{12}$ bytes, before indices, features or solver state.
A k-nearest-neighbour graph can be sparse, but finding the neighbours
and updating the graph still has a cost. A label-propagation API does
not make a dense graph practical for a web-scale embedding corpus.

<a id="published-results"></a>
## Reading a label-efficiency result

FixMatch's paper reports the following CIFAR-10 **test error percentages**
in Table 2 of [v2](https://arxiv.org/abs/2001.07685v2). The two rows differ
in the strong-augmentation method; they must not be combined into one
unspecified configuration.

| Strong augmentation | 40 labels | 250 labels | 4,000 labels |
| --- | ---: | ---: | ---: |
| RandAugment | $13.81\pm3.37$ | $5.07\pm0.65$ | $4.26\pm0.05$ |
| CTAugment | $11.39\pm3.35$ | $5.07\pm0.33$ | $4.31\pm0.15$ |

The entries are retained as reported across five labelled-data folds;
the $\pm$ values are not relabelled here as confidence intervals. The
94.93% accuracy quoted for 250 labels is $100-5.07$, and 88.61% for 40
labels is the CTAugment result $100-11.39$. Forty labels means four
labelled examples per class in that experimental split, not forty
arbitrary examples from an imbalanced pool.

The experiment uses Wide ResNet-28-2 for CIFAR-10, a labelled batch of
64, unlabelled ratio 7, threshold 0.95, unlabelled weight 1, and
$2^{20}$ training steps. The labelled examples are reused many times.
An unlabelled pool is still available. “250 labels” does not mean that
only 250 input images were processed or stored.

The broad statements “80–95% accuracy with 10–20% labels” or “5–10 times
fewer labels at the same accuracy” do not follow for arbitrary tasks.
They require a specified dataset, split, model and metric. Likewise,
the fixed 5× compute cost in the Harvard worked calculation is a
scenario assumption, not a multiplier established by the FixMatch
objective or Table 2.

<a id="cost-accounting"></a>
## Label costs, training costs and reuse

Let $n_l$ be the number of purchased labels, $c_l$ their unit cost, and
$C_{\mathrm{train}}$ the measured training cost. A basic comparison is

$$
 C=n_lc_l+C_{\mathrm{train}}+C_{\mathrm{data}}+C_{\mathrm{evaluation}}.
$$

The Harvard example assigns USD 1 per label, USD 50 of supervised training
and USD 250 of semi-supervised training. With 4,000 versus 250 purchased
labels, those assigned totals are USD 4,050 and USD 500, a ratio of 8.1.
These two-term totals omit data acquisition and evaluation costs.
The label-count ratio for this comparison is 16, while comparing 250
labels against all 50,000 CIFAR-10 labels gives 200. Using the latter
ratio to describe the former cost baseline would mix two comparisons.
No accuracy equivalence or compute multiplier is established by this
arithmetic.

<a id="pretraining-amortization-figure"></a>
![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/harvard-data-selection-2026-10-28/pretraining-amortization.png|Original Harvard comparison of scratch training and reusable pretraining costs]]

*Harvard University and the ML Systems contributors,
[Data Selection, Figure 9: Cost Amortization in Foundation Models](https://mlsysbook.ai/vol1/09_data_selection/09_data_selection.html#fig-amortization-comparison),
web edition retrieved 28 October 2026,
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).
Complete original image, unchanged. The assigned ten-task example totals
10,000 GPU-hours from scratch and 10,500 with pretraining. The smaller
adaptation bars do not yet repay the initial pretraining cost; the calculation
below checks the first profitable integer task count. These are scenario
values, not measured costs of a particular foundation model.*

The same accounting applies to a reusable pretrained model. With an
upfront pretraining cost $P$, adaptation cost $F$ per task and scratch
cost $S$ per task, pretraining has lower *total compute cost* only when
$P+TF<TS$. If $S>F$, this requires $T>P/(S-F)$. The value of $F$ alone
does not account for the initial investment.

```python
label_cost = 1.
supervised = 4000 * label_cost + 50.
semi_supervised = 250 * label_cost + 250.
assert supervised / semi_supervised == 8.1
assert 4000 / 250 == 16 and 50000 / 250 == 200

# Separate assigned scenario from the Harvard amortization example.
pretrain, scratch_per_task, adapt_per_task = 10000., 1000., 50.
for tasks in (10, 11):
    scratch = tasks * scratch_per_task
    reused = pretrain + tasks * adapt_per_task
    print(tasks, "tasks:", scratch, "vs", reused, "GPU-hours")
assert pretrain + 10 * adapt_per_task > 10 * scratch_per_task
assert pretrain + 11 * adapt_per_task < 11 * scratch_per_task
print("break-even task count", pretrain / (scratch_per_task - adapt_per_task))
```

These are assigned costs, not measurements of the medical, fraud or
language-model tasks used as motivating applications. Annotation quality,
evaluation, data handling and transfer performance must be checked for
the actual application before interpreting the calculation as savings.

<a id="self-supervised-boundary"></a>
## Where self-supervised pretraining differs

In pseudo-labeling, a model predicts the target class for an unlabelled
example. In next-token prediction, the target is the token actually
present at the next position in the stored sequence. In masked language
modeling, it is the token deliberately hidden from the input. These
targets have different origins and different failure modes.

The language-model objectives are developed in
[[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/42 Next-token prediction|next-token prediction]]
and [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03 BERT, RoBERTa и DeBERTa|BERT and masked-token prediction]].
They do not require a person to annotate every prediction position,
but their training texts may still contain human writing, mistakes,
preferences and personally identifying information. The absence of
task-specific annotation does not make the corpus neutral or guarantee
that the learned representation solves a downstream task.

Evaluate a pretrained representation with a frozen-encoder probe,
fine-tuning or another stated downstream protocol. Keep that result
separate from the pretext loss and the number of pseudo-labels accepted
by a semi-supervised classifier.

<!-- BEGIN REVIEW: Dai–Le sequence pretraining 2026-10-29 -->
<a id="dai-le-sequence-pretraining"></a>
### Dai–Le: initialize a sequence classifier before using class labels

[Dai and Le, 2015, §§2–4](https://arxiv.org/html/1511.01432v1#S2)
compare two ways to initialize an LSTM classifier from unlabelled
sequences. A recurrent language model predicts the next token from its
prefix. A sequence autoencoder first reads the complete document, then
reconstructs that document. Both provide token targets from the stored
text, rather than predicting its unknown class label.

In the authors'
[original Figure 1](https://arxiv.org/html/1511.01432v1#S2.F1),
the input is W, X, Y, Z, followed by an end marker. The recurrent state
after that marker starts reconstruction: predict W, consume W and
predict X, then Y, Z and the end marker. Encoder and decoder use
**the same recurrent weights**. The full document influences the initial
decoding state, so this is not the same conditional prediction problem
as a language model that has seen only the prefix.
The reconstruction must use the finite-dimensional recurrent state;
this architecture has no attention path that can retrieve every encoder
position directly.

| Objective | Inputs available for predicting W | Target supplied by |
| --- | --- | --- |
| Constructed next-token LM with a start marker | Start marker only | First stored token W |
| Sequence reconstruction | W, X, Y, Z and the end marker | First stored token W |
| Document classification | The document representation | Human class annotation |

For a sequence $x$, the two unsupervised losses can be written as

$$
\begin{aligned}
 \mathcal L_{\mathrm{LM}}
   &=-\sum_t\log p(x_t\mid x_{<t}),\\
 \mathcal L_{\mathrm{SA}}
   &=-\sum_t\log p(x_t\mid z(x),x_{<t}).
\end{aligned}
$$

Here $z(x)$ is the encoder's final recurrent state, including the cell
state. Teacher forcing supplies the preceding true token while training
the reconstruction. There is no human class target in either loss.

After pretraining, the embedding and recurrent parameters initialize a
classifier. Its class-output layer is new. The classifier is then trained
with human labels, and the transferred embedding and LSTM remain
trainable. This is different from freezing an encoder and training a
linear probe, and from using only pretrained word vectors while
randomly initializing the recurrent body.

The program follows the WXYZ reconstruction alignment and transfers the
result of one small update. It also constructs the LM alignment to show
the different conditioning. The numbers are not an experiment on
generalization or a reproduction of the paper's long-document training.

```python
import copy
import torch
import torch.nn.functional as F

torch.manual_seed(37)
dtype = torch.float64
embedding = torch.nn.Embedding(6, 4).to(dtype)  # 0=start, 1=end, 2..5=W..Z
body = torch.nn.LSTM(4, 5, batch_first=True).to(dtype)
token_head = torch.nn.Linear(5, 6).to(dtype)
pretrain_parameters = list(embedding.parameters()) + list(body.parameters())
pretrain_parameters += list(token_head.parameters())
optimizer = torch.optim.SGD(pretrain_parameters, lr=.1)

targets = torch.tensor([[2, 3, 4, 5, 1]])
lm_inputs = torch.tensor([[0, 2, 3, 4, 5]])
lm_states, _ = body(embedding(lm_inputs))
lm_loss = F.cross_entropy(token_head(lm_states).flatten(0, 1), targets.flatten())

# The first reconstruction prediction is made after reading the end marker.
sa_inputs = torch.tensor([[2, 3, 4, 5, 1, 2, 3, 4, 5]])
sa_states, _ = body(embedding(sa_inputs))
reconstruction = token_head(sa_states[:, 4:])
assert reconstruction.shape == (1, 5, 6)
assert sa_inputs[0, 4:].tolist() == [1, 2, 3, 4, 5]
sa_loss = F.cross_entropy(reconstruction.flatten(0, 1), targets.flatten())
optimizer.zero_grad()
sa_loss.backward()
assert embedding.weight.grad.norm() > 0
assert body.weight_hh_l0.grad.norm() > 0
optimizer.step()
optimizer.zero_grad(set_to_none=True)

# Transfer the trained body, not the vocabulary-output head.
class_embedding = copy.deepcopy(embedding)
class_body = copy.deepcopy(body)
class_head = torch.nn.Linear(5, 2).to(dtype)
assert class_body.weight_hh_l0.data_ptr() != body.weight_hh_l0.data_ptr()
torch.testing.assert_close(class_body.weight_hh_l0, body.weight_hh_l0)
pretrained_snapshot = body.weight_hh_l0.detach().clone()
documents = torch.tensor([[2, 3, 4, 5, 1], [5, 4, 3, 2, 1]])
labels = torch.tensor([0, 1])
class_parameters = (list(class_embedding.parameters()) + list(class_body.parameters())
                    + list(class_head.parameters()))
class_optimizer = torch.optim.SGD(class_parameters, lr=.1)
class_optimizer.zero_grad()
states, _ = class_body(class_embedding(documents))
classification_loss = F.cross_entropy(class_head(states[:, -1]), labels)
classification_loss.backward()
assert class_embedding.weight.grad.norm() > 0
assert class_body.weight_hh_l0.grad.norm() > 0
assert all(p.grad is None for p in token_head.parameters())
class_optimizer.step()
assert not torch.equal(class_body.weight_hh_l0, pretrained_snapshot)
assert torch.equal(body.weight_hh_l0, pretrained_snapshot)
print("LM / reconstruction losses", lm_loss.item(), sa_loss.item())
print("classification loss", classification_loss.item())
```

The paper reads complete documents for the reconstruction objective,
but uses truncated backpropagation over up to 400 timesteps from the end
in the described experiments. Reading the whole sequence does not imply
that gradients traverse every earlier step. The nine-step program above
does not truncate its graph. The paper's roughly 500,000 pretraining
updates, dropout tuning and validation-based early stopping are also
absent from this fixture.

The experiments distinguish LM-LSTM, SA-LSTM, joint reconstruction plus
classification, and linear label gain, which attaches weighted class
losses to intermediate timesteps. None is a guarantee that every
pretraining objective improves every dataset. For example, their
20 Newsgroups table reports LM-LSTM at 15.3% error and SA-LSTM at 15.6%.
The v1 IMDb prose and table disagree on the LM-LSTM number; that
internally inconsistent value is not used here as a comparative result.

This work demonstrates recurrent parameter initialization followed by
supervised fine-tuning. BERT later uses a bidirectional Transformer and
masked-token pretraining, whose information access and objective are
different. Calling the earlier model “BERT with an LSTM” would hide those
differences. The detailed objectives are explained in the linked
language-model and BERT chapters above.

<!-- END REVIEW: Dai–Le sequence pretraining 2026-10-29 -->

---

**Предыдущая глава:** [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41d Геометрический отбор — k-center, coreset и herding|Геометрический отбор — k-center, coreset и herding]]

**Следующая глава:** [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41i Отбор данных — стоимость, шарды и проверка качества|Отбор данных: стоимость, шарды и проверка качества]]

**Источники:** Kihyuk Sohn and co-authors,
[FixMatch, v2](https://arxiv.org/abs/2001.07685v2), §§1–4.1 and Table 2;
[original implementation](https://github.com/google-research/fixmatch/blob/d4985a158065947dba803e626ee9a6721709c570/fixmatch.py),
Apache-2.0. The scikit-learn developers,
[Semi-supervised learning](https://scikit-learn.org/stable/modules/semi_supervised.html)
and the attributed circles example, BSD-3-Clause. Harvard ML Systems,
[Data Selection](https://mlsysbook.ai/vol1/09_data_selection/09_data_selection.html),
semi-supervised discussion and the explicitly assigned cost/amortization scenarios.
