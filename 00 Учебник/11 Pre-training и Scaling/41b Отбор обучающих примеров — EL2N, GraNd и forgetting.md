---
title: Отбор обучающих примеров — EL2N, GraNd и forgetting
type: textbook-chapter
status: canonical
source_language: en
last_updated: 2026-10-29
last_verified: 2026-10-29
primary_sources:
  - https://arxiv.org/abs/2107.07075v2
  - https://arxiv.org/abs/2303.14753v1
  - https://github.com/mansheej/data_diet/tree/aee462eab93aa0fb14c1ddfeba45d645b85a5fbb
  - https://github.com/BlackHC/pytorch_datadiet/tree/fda604dc34eade5918a2c98568c2dc0f7861ced3
  - https://mlsysbook.ai/vol1/09_data_selection/09_data_selection.html
---

# 41b. Отбор обучающих примеров — EL2N, GraNd и forgetting

A labelled training set contains examples the model already predicts correctly,
examples it has not learned, and examples whose labels may be wrong. Repeating
all of them equally spends training steps on different kinds of errors. Data
pruning selects a subset before the final training run, using measurements from
a preliminary run to decide which examples to retain.

Removing examples changes the training distribution. Keeping only difficult
examples may discard an entire easy class; keeping the largest errors may retain
incorrect labels. The selection problem therefore includes a target metric,
a retained-data budget, and constraints on the populations that must remain
represented. A score provides an ordering of examples, not a guarantee that any
prefix of that ordering is a suitable training set.

The methods below use labelled classification examples. Their definitions come
from Paul, Ganguli and Dziugaite's
[Data Diet, v2, §§2–4](https://arxiv.org/abs/2107.07075v2), with the reproduction
and correction examined by
[Kirsch, §§2–3](https://arxiv.org/abs/2303.14753v1). The examples use small arrays
so that each score and indexing decision can be checked without training a model.
Corpus cleaning belongs to the
[[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41a Дедупликация, PII и контроль качества корпуса|preceding chapter]];
these scores do not replace deduplication, privacy rules, or a label audit.

<a id="selection-problem"></a>
## Selecting from an already labelled pool

Let the pool be $D=\{(x_i,y_i)\}_{i=1}^{N}$, and suppose the budget permits
retaining $m<N$ examples. A preliminary model assigns a score $s_i$ to each
stable example identifier. A selection rule then produces a set of identifiers
$S$ with $|S|=m$. The final model is trained on this subset, usually from a fresh
initialization when evaluating data pruning as in Data Diet.

The score, the selection rule, and the training schedule are separate choices.
For example, sorting by EL2N does not specify whether to keep the highest scores,
remove a noisy high-score tail, or impose a quota for each class. Nor does it
specify whether the subset is trained for the same number of epochs or the same
number of optimizer steps as the full pool. At equal epochs, a smaller pool
typically requires fewer steps; at equal steps, the subset is repeated more often.
Those experiments answer different questions.

For an unlabelled candidate $x$, a model can calculate predictive entropy before
asking a person for its label. That is an acquisition decision in active learning.
EL2N, in contrast, compares predictions with an existing label $y$. An acquisition
score and a labelled-data pruning score cannot be substituted for one another
without changing the available information.

<a id="el2n"></a>
## EL2N: distance between a probability vector and its label

Consider a classifier with logits $z_\theta(x)$, class probabilities
$p_\theta(x)=\operatorname{softmax}(z_\theta(x))$, and a one-hot label $y$.
Its cross-entropy loss on one example is

$$
\ell(\theta;x,y)=-\sum_{c=1}^{C}y_c\log p_{\theta,c}(x).
$$

For a one-hot label, this is the negative log probability assigned to the labelled
class. Differentiating through softmax gives a vector with one entry per logit:

$$
\frac{\partial\ell}{\partial z_c}=p_{\theta,c}(x)-y_c.
$$

The error vector $p-y$ therefore also gives the loss gradient with respect to the
logits. A correct, confident prediction has a small error vector. A prediction
that assigns nearly all probability to a different class has a large one.
At a single checkpoint, its Euclidean length is

$$
e_i=\|p_\theta(x_i)-y_i\|_2.
$$

In the paper's definition, the EL2N score is the expectation of this length over
training randomness at a chosen early time $t$. A finite set of $R$ independent
preliminary runs estimates it as

$$
\widehat{s}^{\mathrm{EL2N}}_{i,t}
=\frac{1}{R}\sum_{r=1}^{R}
\|p_{\theta^{(r)}_t}(x_i)-y_i\|_2.
$$

The norm is taken inside the average. Averaging probability vectors first can
cancel differences between runs, producing a different statistic. For example,
take $y=(1,0,0)$ and predictions $(0.1,0.8,0.1)$ and $(0.1,0.1,0.8)$.
Each run has error norm $\sqrt{1.46}\approx1.2083$. Their average prediction is
$(0.1,0.45,0.45)$, whose error norm is $\sqrt{1.215}\approx1.1023$.
The latter value does not estimate the paper's EL2N definition.

For a one-hot label, $e_i$ lies between zero and $\sqrt{2}$. The upper endpoint
occurs when all probability is placed on one wrong class. Ordinary finite logits
give probabilities strictly inside the simplex, so these endpoints are limits.

### A confident mistake and an uncertain prediction

For a binary problem with label $y=(1,0)$, the score reduces to
$\sqrt{2}(1-p_0)$. Compare three examples at the same checkpoint. Predictive
entropy $H(p)=-\sum_c p_c\log p_c$ uses natural logarithms, so its unit is nats.
The table rounds scores to three decimal places.

| Case | $p$ | EL2N | $H(p)$ |
| --- | --- | --- | --- |
| A | $(0.99,0.01)$ | $0.014$ | $0.056$ |
| B | $(0.51,0.49)$ | $0.693$ | $0.693$ |
| C | $(0.01,0.99)$ | $1.400$ | $0.056$ |

Entropy is largest for B, whose class probabilities are nearly equal. EL2N is
largest for C, which confidently disagrees with the supplied label. A and C have
the same entropy because entropy does not use the label. C might be a difficult
example, an example outside the model's experience, or a labelling mistake; its
score alone does not distinguish these explanations.

The following program calculates the table and the two ensemble statistics.
It uses fixed probabilities, not a trained preliminary model.

```python
import numpy as np

p = np.array([[.99, .01], [.51, .49], [.01, .99]])
y = np.array([1., 0.])
el2n = np.linalg.norm(p - y, axis=1)
entropy = -(p * np.log(p)).sum(axis=1)
assert np.allclose(el2n, np.sqrt(2) * np.array([.01, .49, .99]))
assert np.isclose(entropy[0], entropy[2])
assert entropy.argmax() == 1 and el2n.argmax() == 2
for name, score, h in zip("ABC", el2n, entropy):
    print(name, f"EL2N={score:.6f}", f"entropy={h:.6f}")

predictions = np.array([[.1, .8, .1], [.1, .1, .8]])
errors = predictions - np.array([1., 0., 0.])
mean_norm = np.linalg.norm(errors, axis=1).mean()
norm_of_mean = np.linalg.norm(errors.mean(axis=0))
assert np.isclose(mean_norm, np.sqrt(1.46))
assert np.isclose(norm_of_mean, np.sqrt(1.215))
assert mean_norm > norm_of_mean
print("mean norm", mean_norm, "norm of mean", norm_of_mean)
```

The original implementation calculates `softmax(logits) - Y`, then
`linalg.norm(..., ord=2, axis=-1)` in
[`scores.py`](https://github.com/mansheej/data_diet/blob/aee462eab93aa0fb14c1ddfeba45d645b85a5fbb/data_diet/scores.py).
[`get_mean_score.py`](https://github.com/mansheej/data_diet/blob/aee462eab93aa0fb14c1ddfeba45d645b85a5fbb/scripts/get_mean_score.py)
averages the resulting per-example scores across runs. Squaring the norms before
averaging defines another statistic; it can change the ranking across runs even
though squaring preserves a ranking of nonnegative scores from one run.

<a id="grand"></a>
## GraNd: the parameter gradient contains more than prediction error

GraNd averages the norm of the per-example parameter gradient:

$$
\widehat{s}^{\mathrm{GraNd}}_{i,t}
=\frac{1}{R}\sum_{r=1}^{R}
\left\|\nabla_\theta\ell(\theta^{(r)}_t;x_i,y_i)\right\|_2.
$$

This gradient includes the model's response to a parameter change, not only the
probability error. If $J_i=\partial z_\theta(x_i)/\partial\theta$ is the logit
Jacobian, the chain rule gives

$$
\nabla_\theta\ell_i=J_i^\top(p_i-y_i).
$$

EL2N omits $J_i$. Two examples can have identical probabilities and EL2N scores
while having very different parameter-gradient norms. For a linear classifier
$z=Wh$ without a bias, the gradient is the outer product
$(p-y)h^\top$, so its Frobenius norm is

$$
\|\nabla_W\ell\|_F=\|p-y\|_2\,\|h\|_2.
$$

Set $W=0$, label the first class, and compare features $h_A=(1,0)$ and
$h_B=(10,0)$. Both examples have probabilities $(0.5,0.5)$ and EL2N
$1/\sqrt{2}$. The gradient norms are $1/\sqrt{2}$ and $10/\sqrt{2}$.
The larger norm arises entirely from feature magnitude.

```python
import torch
import torch.nn.functional as F

features = torch.tensor([[1., 0.], [10., 0.]], dtype=torch.float64)
error_norms, parameter_norms = [], []
for h in features:
    W = torch.zeros(2, 2, dtype=torch.float64, requires_grad=True)
    logits = W @ h
    y = torch.tensor([0])
    loss = F.cross_entropy(logits[None], y)
    gradient, = torch.autograd.grad(loss, W)
    error = logits.softmax(0) - torch.tensor([1., 0.])
    expected_gradient = error[:, None] * h[None, :]
    torch.testing.assert_close(gradient, expected_gradient)
    torch.testing.assert_close(gradient.norm(), error.norm() * h.norm())
    error_norms.append(error.norm().item())
    parameter_norms.append(gradient.norm().item())
assert error_norms[0] == error_norms[1]
assert abs(parameter_norms[1] / parameter_norms[0] - 10.) < 1e-12
print("EL2N", error_norms, "gradient norms", parameter_norms)

target_gradient = torch.tensor([1., 0.])
candidate_gradients = torch.tensor([[10., 0.], [-10., 0.]])
eta = .01
first_order_changes = -eta * (candidate_gradients @ target_gradient)
torch.testing.assert_close(first_order_changes, torch.tensor([-.1, .1]))
torch.testing.assert_close(candidate_gradients.sum(0), torch.zeros(2))
print("target loss changes from individual updates", first_order_changes)
```

For a deep network, the feature extractor also contributes gradients. The
linear-head identity above cannot be applied to the entire parameter vector.
Data Diet §2.2 motivates EL2N as a proxy when the gradients of individual logits
are roughly orthogonal and have comparable magnitudes across classes and
examples. Under those assumptions, variation in $\|p-y\|$ can approximate
variation in the full gradient norm. The assumptions need checking in the model
and training regime where the score will be used.

To see why the paper uses the norm, consider a summed SGD update at fixed
parameters, with $g_j=\nabla_\theta\ell_j$:

$$
\Delta\theta=-\eta\sum_j g_j.
$$

For a separate target example with gradient $g^*$, the first-order change in
loss is $\langle g^*,\Delta\theta\rangle$. Removing one summand $g_i$ changes
this first-order quantity by $d_i$, while holding all other summands fixed:

$$
\begin{aligned}
d_i&=\eta\langle g^*,g_i\rangle,\\
|d_i|&\le\eta\|g^*\|_2\|g_i\|_2.
\end{aligned}
$$

The inequality bounds the magnitude of influence using the gradient norm. Its
sign still depends on the angle between gradients. For $g^*=(1,0)$, candidate
gradients $(10,0)$ and $(-10,0)$ both have norm 10. With $\eta=0.01$, their
individual updates predict target-loss changes of $-0.1$ and $+0.1$; one helps
the target and the other hurts it. Together their summed update is zero, as
checked in the program. A large norm alone cannot establish improvement.

This calculation describes one local, first-order step, not the effect of
removing an example from an entire training trajectory. For a mean-reduced
batch loss, changing the denominator after removal also rescales the other
gradients; the calculation above deliberately keeps their weights fixed.
Computing a single gradient of the averaged batch loss can hide cancellation.
GraNd requires per-example gradients before taking their norms.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/data-selection-2026-10-28/joost_train_input_norm_vs_grad_norm_sorted.png]]

*Input norm and GraNd at initialization in the minimal CIFAR-10 reproduction.
The original repository plot reports rank correlation 0.97; the vertical axis
contains the two raw scores, not equalized units. A similar ordering can arise
from input magnitude before the model has learned task structure. This plot
does not measure the quality of a pruned training set. Andreas Kirsch,
[original file at commit fda604dc](https://github.com/BlackHC/pytorch_datadiet/blob/fda604dc34eade5918a2c98568c2dc0f7861ced3/sampled_grad_norms/joost_train_input_norm_vs_grad_norm_sorted.png),
[Apache-2.0](/licenses/data-diet-reproduction-fda604dc-Apache-2.0.txt).
Complete original image, unchanged.*

<a id="forgetting"></a>
## Forgetting records transitions during training

A forgetting event occurs when an example classified correctly at one
observation is classified incorrectly at the next. This uses a history of
predictions, whereas EL2N and GraNd can be measured at a chosen early checkpoint.
Let $a_i^{(t)}$ be one if example $i$ is classified correctly at observation $t$.
A forgetting count is

$$
F_i=\sum_{t=2}^{T}
\mathbf{1}\{a_i^{(t-1)}=1,\ a_i^{(t)}=0\}.
$$

The observations must follow the same example. In a minibatch implementation,
successive observations may occur when the example is revisited rather than at
every optimizer step. Evaluation frequency and training length affect what is
recorded. A zero count means no recorded correct-to-incorrect transition; it does
not imply that the example was ever learned.

For three examples, consider four consecutive observations:

| $t$ | Example 0 | Example 1 | Example 2 |
| --- | --- | --- | --- |
| 1 | wrong | wrong | correct |
| 2 | correct | wrong | correct |
| 3 | wrong | wrong | wrong |
| 4 | correct | wrong | correct |

Examples 0 and 2 each have one forgetting event, at observation 3. Example 1 has
zero events but was never correct. Treating it as an easy, unforgettable example
would reverse the meaning of the record. The Data Diet implementation keeps a
separate `never_correct` set and exports those examples with infinite forgetting
scores, so they are distinguishable from learned examples with zero events.

The two helper functions below are reproduced unchanged from
[`forgetting.py`](https://github.com/mansheej/data_diet/blob/aee462eab93aa0fb14c1ddfeba45d645b85a5fbb/data_diet/forgetting.py),
Mansheej Paul and the Data Diet contributors,
[Apache-2.0](/licenses/data-diet-aee462ea-Apache-2.0.txt).
The history and assertions after the helpers are a teaching example. The export
convention is calculated in memory; the program writes no files.

```python
import numpy as np
from types import SimpleNamespace


def init_forget_stats(args):
  forget_stats = SimpleNamespace()
  forget_stats.prev_accs = np.zeros(args.num_train_examples, dtype=np.int32)
  forget_stats.num_forgets = np.zeros(args.num_train_examples, dtype=float)
  forget_stats.never_correct = np.arange(args.num_train_examples, dtype=np.int32)
  return forget_stats


def update_forget_stats(forget_stats, idxs, accs):
  forget_stats.num_forgets[idxs[forget_stats.prev_accs[idxs] > accs]] += 1
  forget_stats.prev_accs[idxs] = accs
  forget_stats.never_correct = np.setdiff1d(forget_stats.never_correct, idxs[accs.astype(bool)], True)
  return forget_stats


stats = init_forget_stats(SimpleNamespace(num_train_examples=3))
ids = np.arange(3)
history = [[0, 0, 1], [1, 0, 1], [0, 0, 0], [1, 0, 1]]
for correct in history:
    update_forget_stats(stats, ids, np.array(correct))
exported = stats.num_forgets.copy()
exported[stats.never_correct] = np.inf
assert stats.num_forgets.tolist() == [1., 0., 1.]
assert stats.never_correct.tolist() == [1]
assert exported.tolist() == [1., np.inf, 1.]
print("counts", stats.num_forgets, "exported", exported)
```

These helpers expect unique integer dataset positions within each update and
binary correctness values. They are not a validator for an arbitrary data stream.
A long history is more expensive to obtain than one early forward pass. Whether
a cheaper preliminary model preserves a useful forgetting ordering for another
model remains an experimental question.

<a id="stable-identifiers"></a>
## Scores must follow example identifiers, not loader positions

Suppose examples A, B and C have identifiers 101, 7 and 55. The first program
gives C the largest EL2N. A scoring loader may return them in the order C, A, B.
Appending its scores produces approximately `[1.400071, 0.014142, 0.692965]`.
If those values are then zipped with the original identifier order `[101, 7, 55]`,
the largest score is incorrectly assigned to A.

Shuffling does not make a score invalid; losing its identifier does. Each batch
must carry the IDs alongside the model inputs and labels. The result should be
stored as ID–score pairs and checked against the exact pool to be selected.
An ensemble must align those pairs by ID before averaging runs.

This complete CPU example uses an identity model whose inputs are fixed logits.
It verifies the join, not the usefulness of EL2N for real data pruning. The three
dataset IDs are deliberately different from their positional indices.

```python
import torch
from torch.utils.data import DataLoader, Dataset, Subset


class LabelledPool(Dataset):
    def __init__(self):
        self.ids = [101, 7, 55]
        probabilities = torch.tensor(
            [[.99, .01], [.51, .49], [.01, .99]], dtype=torch.float64)
        self.logits = probabilities.log()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, position):
        return self.ids[position], self.logits[position], 0


def score_by_id(model, loader, expected_ids):
    expected = set(expected_ids)
    if len(expected) != len(expected_ids):
        raise ValueError("pool IDs must be unique")
    scores = {}
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for ids, inputs, labels in loader:
                probabilities = model(inputs).softmax(-1)
                target = torch.nn.functional.one_hot(
                    labels, num_classes=probabilities.shape[-1])
                batch_scores = (probabilities - target).norm(dim=-1)
                for example_id, score in zip(ids.tolist(), batch_scores.tolist()):
                    if example_id in scores:
                        raise ValueError("duplicate ID in scoring pass")
                    if not torch.isfinite(torch.tensor(score)):
                        raise ValueError("non-finite score")
                    scores[example_id] = score
    finally:
        model.train(was_training)
    if scores.keys() != expected:
        raise ValueError("scored IDs do not match the selected pool")
    return scores


pool = LabelledPool()
model = torch.nn.Identity()
ordered = DataLoader(Subset(pool, [0, 1, 2]), batch_size=2)
shuffled = DataLoader(Subset(pool, [2, 0, 1]), batch_size=2)
first = score_by_id(model, ordered, pool.ids)
second = score_by_id(model, shuffled, pool.ids)
assert first == second and model.training
selected_id = max(second, key=second.get)
wrong_join = dict(zip(pool.ids, second.values()))
assert selected_id == 55
assert max(wrong_join, key=wrong_join.get) == 101
print("correct ID", selected_id, "incorrect positional join", wrong_join)
```

For a GPU model, inputs and labels also need the appropriate device transfer.
For stochastic augmentation, one ID may intentionally have multiple measurements;
those views need an explicit aggregation rule rather than being mistaken for
different examples. Distributed scoring can repeat IDs when a sampler pads its
shards. The example rejects duplicate observations, so such a sampler needs a
different, documented reduction policy or a non-padding partition.

`model.eval()` prevents training-mode dropout and changes batch-normalization
behaviour; `inference_mode()` prevents recording a gradient graph. Both fit this
EL2N forward pass. GraNd needs gradients, so its scoring pass cannot be wrapped
in `inference_mode()`.

<a id="reproduction"></a>
## What the reproduction establishes

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/data-selection-2026-10-28/figure11_repro_cifar10_hlb.svg]]

*Test accuracy after pruning CIFAR-10 in Andreas Kirsch's HLB reproduction.
Each curve uses a different ordering of the same training pool. The horizontal
axis is the fraction removed, so 0.5 means retaining half the examples.
“At Epoch 1” refers to score measurement after one preliminary training epoch;
“at Init” refers to measurement before training. The full author-supplied SVG
and its shaded bands are retained unchanged. This repository plot is not a crop
of Figure 2(b) in the earlier arXiv PDF, whose plotted range differs.
[Original file, commit fda604dc](https://github.com/BlackHC/pytorch_datadiet/blob/fda604dc34eade5918a2c98568c2dc0f7861ced3/figure11_repro_cifar10_hlb.svg),
[Apache-2.0](/licenses/data-diet-reproduction-fda604dc-Apache-2.0.txt).*

EL2N and GraNd measured after training perform better than their initialization
counterparts in this experiment. At larger pruning fractions, the gap to random
selection becomes visible. A rank correlation between input norm and GraNd at
initialization, as in the earlier figure, does not establish that either ranking
preserves test accuracy after pruning.

In the pinned reproduction script, scores at epoch 1 are averaged over ten
preliminary runs. Each retained fraction is then trained with ten target seeds.
The plotting call uses Seaborn's default aggregation and does not explicitly
set the uncertainty interval or pin the library version. The shaded area is
therefore reproduced as published; it is not relabelled here as a standard
deviation, percentile range, or independently verified confidence interval.
The original paper's own Figure 1 uses a different protocol and reports
16th–84th percentiles across four target-training seeds.

Kirsch's reproduction investigated why GraNd at initialization did not match
the original result. The relevant historical checkpoint loader treated step
zero as a false condition and loaded the latest checkpoint instead of the
initial one. Measuring that checkpoint as “initialization” attached the wrong
training time to the scores. The bug was fixed in Flax in April 2021; it is not
a claim about current Flax releases. Data Diet v2, posted in March 2023, corrects
the initialization result.

<details>
<summary>Kirsch's explanation of the checkpoint error, in the original English</summary>

> On further investigation, the author of this reproduction found a bug in
> flax.training.restore_checkpoint that was fixed in April 2021: passing a 0
> step (i.e. initialization) would trigger loading the latest checkpoint instead
> of the zero-th checkpoint because the internal implementation was checking
> if step: instead of if step is not None: when deciding whether to fallback to
> loading the latest checkpoint.

Andreas Kirsch, [§3, Conclusion](https://arxiv.org/abs/2303.14753v1),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The footnote marker is omitted; the linked
[historical Flax fix](https://github.com/google/flax/commit/28fbd95500f4bf2f9924d2560062fa50e919b1a5)
identifies the change. The quoted explanation is not translated or rewritten.

</details>

The reproduction supports trained scores in the reported CIFAR-10 settings.
It does not establish a universal pruning fraction, an optimal measurement
epoch, or unchanged quality for a language model. For an autoregressive model,
one must additionally define the unit being selected: document, sequence,
instruction–response pair, or token. A sequence-level reduction changes how
length and loss masks affect the score. The classifier results alone do not
choose that reduction.

<a id="selection-controls"></a>
## High scores, label noise, and the selection experiment

A preliminary model that has learned common patterns may give low scores to
redundant examples and high scores to examples it still needs. It may also give
high scores to wrong labels, rare legitimate cases, and inputs outside its
experience. These cases have different consequences when retained or removed.
Data Diet §4 studies how scores behave with label noise, including selection
windows that omit the highest-scoring tail. It does not justify removing all
hard examples from an arbitrary dataset.

Suppose a minority class has ten difficult examples and a majority class has
ten thousand easy ones. Removing the ten largest errors could delete the whole
minority class while barely changing average validation accuracy. A class-wise
or group-wise evaluation reveals the failure; a global average alone may not.
Whether to impose class quotas depends on the intended task distribution and
constraints, rather than on the EL2N definition.

Harvard's Data Selection chapter states the representation constraint explicitly:

> The decision also needs a guardrail before any score is applied: classes,
> demographic groups, time windows, and rare failure modes that matter at
> deployment require minimum representation, because the highest-average-ICR
> subset can still remove the examples that define production risk.

Harvard ML Systems, [Coreset selection algorithms](https://mlsysbook.ai/vol1/09_data_selection/09_data_selection.html),
[source at 2bd97c5](https://github.com/harvard-edge/cs249r_book/blob/2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd/book/quarto/contents/vol1/data_selection/data_selection.qmd),
CC BY-NC-SA 4.0. Original English excerpt, unchanged.

A pruning experiment needs a full-pool baseline and random subsets of the same
sizes as the scored subsets. Comparing only a small scored subset with the full
pool cannot separate selection quality from the effect of the smaller training
budget. Target training should use fresh seeds and a declared schedule. The
score-measurement epoch, number of preliminary runs, retained fraction, tie rule,
and treatment of each class must be fixed or selected on validation data. The
test set is reserved for the resulting comparison, not for choosing the window.

Report both final quality and total work. If the chosen schedule repeats the
subset until it has consumed the original number of training tokens, pruning
has not saved those token computations. It may still improve the distribution,
but that is a quality result rather than an automatic compute saving.

<!-- confident-learning-addition: start -->
<a id="confident-learning"></a>
## Confident Learning: finding labels to review

Suppose a classifier assigns probability 0.80 to class B for an example labelled
A. Its large EL2N tells us that the prediction and label disagree. It does not
tell us which one is wrong. Confident Learning uses predictions across the
dataset to estimate how labels are confused between classes, then identifies
individual records for review or pruning. A classifier that consistently misses
a valid subgroup may flag that subgroup's correct labels as errors.

Let $\tilde y_i$ be the label stored with example $i$, and $y_i^*$ its unknown
correct label. The distinction matters even if most stored labels are correct.
The target is a joint distribution

$$
Q_{ab}=P(\tilde y=a,\ y^*=b).
$$

A row describes records carrying the same stored label; a column describes their
latent correct class. An off-diagonal cell $Q_{AB}$ is the fraction of the entire
dataset labelled A but belonging to B. It is not the fraction of A-labelled
records that belong to B: that conditional probability requires division by the
row sum.

The formulation in Northcutt, Jiang and Chuang's
[Confident Learning, §§2–3](https://jair.org/index.php/jair/article/view/12125)
assumes class-conditional label noise: given the correct class, the probability
of a label change does not depend on the particular input. Thus
$P(\tilde y\mid y^*,x)=P(\tilde y\mid y^*)$.
This permits asymmetric errors, such as more dogs labelled fox than foxes
labelled dog. It does not describe every annotation process. If annotators
mislabel only blurred photographs or sentences with sarcasm, the error also
depends on the input within a class. The method can still produce candidates,
but its theoretical guarantees do not automatically extend to that situation.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/confident-learning-2026-10-29/figure1-process.png]]

*Original Figure 1: predictions produce counts, the counts estimate a joint
distribution, and a pruning rule chooses records. The matrix columns marked
$y^*$ are estimates, not newly observed ground truth. In this particular figure
the counts sum to 400, so 40 becomes 0.10. The example below also includes
unassigned records, where simply dividing the raw counts by their total would
give the wrong observed class proportions. Northcutt, Jiang and Chuang,
“Confident Learning: Estimating Uncertainty in Dataset Labels,” Journal of
Artificial Intelligence Research 70 (2021), 1373–1411,
[doi:10.1613/jair.1.12125](https://doi.org/10.1613/jair.1.12125), p. 1378.
© 2021 AI Access Foundation. Complete figure panel rendered from the publisher's
PDF; surrounding prose excluded, labels and geometry unchanged.
[JAIR License Version 1.0](/licenses/confident-learning-JAIR-v1.html).*

### Each prediction must come from a model that did not train on that record

A sufficiently flexible classifier can memorize an incorrect label. If we fit
it on all records and inspect its predictions on those same records, agreement
may show memorization rather than annotation quality. An out-of-sample
prediction avoids this direct circularity. It does not make the prediction
correct; it removes the record's own training label from that model's fit.

To obtain predictions for every training record, divide the pool into folds.
With three folds A, B and C, fit a fresh model on B+C and predict A; fit on A+C
and predict B; fit on A+B and predict C. Put each prediction back at its original
record position. Do not average in predictions from the two models that trained
on that record. This is the procedure in Cleanlab's
[out-of-sample prediction guide](https://docs.cleanlab.ai/stable/tutorials/pred_probs_cross_val.html).

Preprocessing must respect the same split. A scaler or feature selector fitted
on the whole pool leaks held-out information. Multiple images of one object,
near-duplicate texts, or records from the same patient may require group-based
folds. Random row separation does not separate those underlying examples.
Stratification helps retain classes in each fold; it cannot solve a class with
too few independent examples. Record the split, model version, class-column
order, and stable IDs together with the probabilities.

This small CPU program demonstrates the fold bookkeeping with a classifier and
scaler fitted separately for each fold. Its synthetic features are not a label
noise benchmark. The models see the stored labels, not a flag identifying which
three were changed. The program makes no claim that all three will be detected.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(17)
centres = np.array([[-2., 0.], [2., 0.], [0., 3.]])
X = np.vstack([rng.normal(c, .8, size=(20, 2)) for c in centres])
labels = np.repeat(np.arange(3), 20)
labels[[0, 20, 40]] = [1, 2, 0]
oof = np.full((60, 3), np.nan)
visits = np.zeros(60, dtype=int)
folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=9)
for train, held_out in folds.split(X, labels):
    assert not np.intersect1d(train, held_out).size
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X[train], labels[train])
    assert model.classes_.tolist() == [0, 1, 2]
    oof[held_out] = model.predict_proba(X[held_out])
    visits[held_out] += 1
assert np.all(visits == 1) and np.isfinite(oof).all()
assert np.allclose(oof.sum(axis=1), 1.)
print("out-of-fold rows", len(oof), "class columns", oof.shape[1])
```

### Class thresholds and the records that pass them

Write $p_{ij}$ for the predicted probability of class $j$ on record $i$.
The threshold for class $j$ is the average probability of that class among
records whose **stored label** is $j$:

$$
t_j=\frac{1}{n_j}\sum_{i:\tilde y_i=j}p_{ij},
\qquad n_j=|\{i:\tilde y_i=j\}|.
$$

This average is computable without knowing the correct labels. It includes
incorrectly labelled records if they are present. It is not an externally
validated probability that a label is correct, and 0.5 is not a universal
replacement for it. Classes can have different prediction scales. The paper
explains the intended adjustment in its original English:

> For example, if examples labeled i tend to have higher probabilities
> because the model is over-confident about class i, then ti will be
> proportionally larger; if some other class j tends toward low probabilities,
> tj will be smaller.

Northcutt, Jiang and Chuang, §3.1, p. 1379; same article and
[JAIR terms](/licenses/confident-learning-JAIR-v1.html) as Figure 1.
© 2021 AI Access Foundation. Original wording; mathematical symbols are set
as plain text.

Consider the following assigned probabilities. They are a separate arithmetic
example, not outputs claimed from the preceding logistic-regression run. Each
class has four stored labels, so the thresholds are

$$
\begin{aligned}
t_A&=(.80+.70+.10+.36)/4=.49,\\
t_B&=(.80+.70+.10+.30)/4=.475,\\
t_C&=(.80+.70+.15+.30)/4=.4875.
\end{aligned}
$$

| ID | Stored label | $p_A$ | $p_B$ | $p_C$ | Class passing its threshold |
| --- | --- | --- | --- | --- | --- |
| 1 | A | .80 | .15 | .05 | A |
| 2 | A | .70 | .20 | .10 | A |
| 3 | A | .10 | .80 | .10 | B |
| 4 | A | .36 | .34 | .30 | none |
| 5 | B | .10 | .80 | .10 | B |
| 6 | B | .20 | .70 | .10 | B |
| 7 | B | .80 | .10 | .10 | A |
| 8 | B | .10 | .30 | .60 | C |
| 9 | C | .10 | .10 | .80 | C |
| 10 | C | .20 | .10 | .70 | C |
| 11 | C | .10 | .75 | .15 | B |
| 12 | C | .40 | .30 | .30 | none |

ID 3 passes the B threshold because $.80\ge .475$. Its stored label A chooses
the matrix row, and the threshold-qualified B chooses the column. We add one
to $C_{AB}$. ID 4 passes no threshold, so it contributes no count. An ordinary
argmax confusion matrix would count it in column A anyway. ID 12 would enter
column A too, becoming an additional disagreement with its stored C label.
The confident joint leaves both unassigned. Unassigned does not mean clean,
wrong, or out-of-distribution: none of those conclusions follows from failing
the thresholds alone.

In this example every assigned record passes exactly one threshold. In general,
several classes can pass. Counting the record once for each would inflate the
counts. Equation 1 restricts argmax to threshold-qualified classes; the paper's
following collision paragraph and Appendix Algorithm 1 use global argmax.
The pinned Cleanlab implementation follows the latter rule. For records with
one passing class, both rules choose that class.

### From ten counted records to an estimate for twelve

The raw table has rows for stored A, B and C labels, and columns for inferred
A, B and C labels:

$$
C=\begin{pmatrix}2&1&0\\1&2&1\\0&1&2\end{pmatrix}.
$$

The row sums are 3, 4 and 3. Dividing by the total count 10 would assign row
probabilities .3, .4 and .3, even though the observed dataset contains four
records of each stored class. Equation 3 first rescales each row to its known
number of records and then normalizes by the full dataset size:

$$
\begin{aligned}
\widetilde C_{ab}&=\frac{n_a C_{ab}}{\sum_j C_{aj}},\\
\widehat Q_{ab}&=\frac{\widetilde C_{ab}}{N}.
\end{aligned}
$$

Here the A and C rows are multiplied by $4/3$, while the B row stays unchanged.
The resulting estimated counts are

$$
\widetilde C=
\begin{pmatrix}8/3&4/3&0\\1&2&1\\0&4/3&8/3\end{pmatrix}.
$$

Fractional entries are valid estimates; they are not literal records. For
example, $\widehat Q_{AB}=(4/3)/12=1/9$ estimates the fraction of all records
labelled A but belonging to B. Dividing it by the A row probability $1/3$
gives an estimated one-third of A-labelled records belonging to B. The B column
sums to $7/18$, an estimate of the true B-class frequency. Dividing $1/9$ by
$7/18$ instead gives $2/7$: the estimated fraction of true-B records carrying
label A. These are different conditioning directions.

Rescaling a row cannot determine the correct label of its unassigned records.
It extrapolates from the counted records. If the omitted records have a different
error pattern, that extrapolation may be wrong. A nonempty stored class with
zero counted records cannot be normalized by this formula at all; it requires
an explicit fallback or more evidence. Filling a diagonal cell, as software
may do, is a policy for this case, not an observed correctly labelled record.

### Counts, candidates, and a review order

Four records were directly counted off the diagonal: IDs 3, 7, 8 and 11.
The sum of the estimated off-diagonal counts is instead $14/3$, corresponding
to an estimated noise fraction $7/18\approx .389$. The count estimate does not
give us an extra two-thirds of a record to inspect. It can inform a selection
budget, which needs rounding and a ranking rule.

Section 3.2 separates these choices. One rule takes the actual off-diagonal
records. Prune by Class estimates a number to remove from each stored class
and ranks that class's records by $p_{i,\tilde y_i}$, called self-confidence.
Prune by Noise Rate allocates counts to particular off-diagonal cells and ranks
by the alternative-class margin. Their intersection is another method, not an
equivalent way to produce the same subset. The worked program uses only the
direct off-diagonal rule, so every candidate has an identifiable counted record.

For a review order, use the paper's margin
$p_{i,\tilde y_i}-\max_{j\ne\tilde y_i}p_{ij}$, smallest first. IDs 3 and 7
have margin $-.70$, ID 11 has $-.60$, and ID 8 has $-.30$. Break the first tie
by stable ID. This yields 3, 7, 11, 8. Adding one and dividing by two would
place the score in $[0,1]$ without changing the order; the raw margin here is
not a calibrated probability of annotation error.

```python
import numpy as np

ids = np.arange(1, 13)
labels = np.repeat(np.arange(3), 4)
p = np.array([
    [.80, .15, .05], [.70, .20, .10], [.10, .80, .10], [.36, .34, .30],
    [.10, .80, .10], [.20, .70, .10], [.80, .10, .10], [.10, .30, .60],
    [.10, .10, .80], [.20, .10, .70], [.10, .75, .15], [.40, .30, .30],
])
t = np.array([p[labels == j, j].mean() for j in range(3)])
np.testing.assert_allclose(t, [.49, .475, .4875])
qualified = p >= t
assert np.all(qualified.sum(axis=1) <= 1)  # no collisions in this example
assigned = qualified.any(axis=1)
guessed = qualified.argmax(axis=1)  # use only on assigned rows
C = np.zeros((3, 3), dtype=int)
np.add.at(C, (labels[assigned], guessed[assigned]), 1)
np.testing.assert_array_equal(C, [[2, 1, 0], [1, 2, 1], [0, 1, 2]])
counts = np.bincount(labels, minlength=3)
assert np.all(C.sum(axis=1) > 0)
estimated_counts = C * (counts / C.sum(axis=1))[:, None]
Q = estimated_counts / len(labels)
np.testing.assert_allclose(Q.sum(axis=1), counts / len(labels))
assert np.isclose(Q.sum(), 1.)
assert np.isclose(1. - np.trace(Q), 7/18)
assert np.isclose(Q[0, 1] / Q[0].sum(), 1/3)
assert np.isclose(Q[0, 1] / Q[:, 1].sum(), 2/7)

candidates = assigned & (guessed != labels)
alternatives = p.copy()
alternatives[np.arange(len(labels)), labels] = -np.inf
margin = p[np.arange(len(labels)), labels] - alternatives.max(axis=1)
positions = np.flatnonzero(candidates)
order = positions[np.lexsort((ids[positions], margin[positions]))]
assert ids[order].tolist() == [3, 7, 11, 8]
assert ids[~assigned].tolist() == [4, 12]
print("thresholds", t, "raw joint", C, "estimated joint", Q, sep="\n")
print("review IDs", ids[order], "estimated noise fraction", 1. - np.trace(Q))
```

### Where the paper and the software differ

The implementation inspected here is
[`cleanlab/count.py`, commit 7506257](https://github.com/cleanlab/cleanlab/blob/750625747de1b26d8530954f51f0530bd0b51d3c/cleanlab/count.py).
Its `get_confident_thresholds` uses the class means above, with additional
policies for missing classes and a lower bound. `compute_confident_joint`
uses a $10^{-6}$ comparison tolerance, resolves multiple passing classes with
global argmax, and sets every diagonal entry to at least one even when
`calibrate=False`. `calibrate_confident_joint` rescales the rows and rounds
counts while preserving their totals. Thus its default integer output need
not equal the fractional $\widetilde C$ above.

Take probabilities $(.40,.35,.25)$ and thresholds $(.50,.30,.20)$.
B and C pass; A does not. Restricted argmax chooses B. Global argmax chooses A,
despite A failing its threshold. This is a counterexample to treating Equation 1
and Algorithm 1 as interchangeable on every input. It does not establish how
often the difference occurs in a trained model.

The following program checks that example and the diagonal policy separately.
It illustrates those rules; it is not a reimplementation of the Cleanlab API.

```python
import numpy as np

p = np.array([.40, .35, .25])
t = np.array([.50, .30, .20])
passing = np.flatnonzero(p >= t)
restricted = passing[p[passing].argmax()]
global_choice = p.argmax()
assert passing.tolist() == [1, 2]
assert restricted == 1 and global_choice == 0

raw = np.array([[0, 2], [2, 0]])
with_diagonal_floor = raw.copy()
np.fill_diagonal(with_diagonal_floor, raw.diagonal().clip(min=1))
assert raw.sum() == 4 and with_diagonal_floor.sum() == 6
print("restricted/global", restricted, global_choice)
print("four counted records; after diagonal policy", with_diagonal_floor)
```

Keep the returned off-diagonal record indices distinct from a calibrated matrix:
scaling or rounding counts does not create additional record identities. Before
using another release, check its counting and filtering functions rather than
assuming that a parameter named `confident_joint` always contains raw counts.

### What a reviewer must decide

Suppose ID 11 is a valid but rare C example. A model trained mostly on common
cases may still assign it $.75$ probability of B. It passes exactly the same
thresholds as an actual mislabelled B example. Neither the margin nor the joint
table can resolve this ambiguity from these probabilities alone. A reviewer
needs the original input, annotation instructions, and enough domain context
to adjudicate the label.

For each candidate retain its ID, original label, proposed alternative,
probability vector, score, split/model provenance, and review decision. A
confirmed error can receive a corrected label. A valid hard example stays.
An ambiguous input may need multiple labels, a revised annotation policy, or
quarantine for further review. Changing every candidate to the model's argmax
would turn the classifier's mistakes into new training targets.

The paper makes this distinction explicitly in its ImageNet experiment:

> We use the term issue versus error because examples found by CL consist of
> a mixture of multi-label images, ontological issues, and actual label errors.

Northcutt, Jiang and Chuang, §5.2, pp. 1387–1388; same article and
[JAIR terms](/licenses/confident-learning-JAIR-v1.html) as Figure 1.
© 2021 AI Access Foundation. Original English, joined across the page break.

In that experiment, human inspection of a random sample of 500 flagged records
found that 58% had one of these issues. That is not a finding that 58% were
simple wrong-class labels, and it is not a precision guarantee for a different
dataset. A review of only the highest-scoring examples likewise cannot estimate
precision across the whole candidate list. Random sampling within declared
score/class groups can measure it; estimating missed errors also requires
reviewing records outside the list.

### What the theory and experiments establish

Theorem 1 assumes ideal predicted probabilities and a noise transition matrix
whose diagonal entries maximize their rows and columns. Theorem 2 relaxes the
prediction assumption but still imposes bounded errors, no label collisions,
and row-diagonal conditions. The intuition is that errors in probabilities can
be tolerated if they do not move records across the relevant class thresholds.
These are sufficient conditions for the result, not properties guaranteed by
softmax or by cross-validation. On an arbitrary dataset we cannot certify them
merely by observing the model's own probabilities.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/confident-learning-2026-10-29/figure2-joint-estimation.png]]

*Original Figure 2, all three panels. Left: the joint distribution known from
the synthetic corruption experiment. Middle: the CL estimate. Right: absolute
cell-wise differences. Values are multiplied by 100, so 4.0 means probability
0.04, not four records. The setting is CIFAR-10 with 40% label noise and 60%
sparsity among off-diagonal noise entries; sparsity is not the fraction of clean
labels. Northcutt, Jiang and Chuang, “Confident Learning: Estimating Uncertainty
in Dataset Labels,” Journal of Artificial Intelligence Research 70 (2021),
1373–1411, [doi:10.1613/jair.1.12125](https://doi.org/10.1613/jair.1.12125),
p. 1386. © 2021 AI Access Foundation.
[JAIR License Version 1.0](/licenses/confident-learning-JAIR-v1.html).
Complete figure panel rendered from the publisher's PDF without rearranging
panels or changing labels. Open the image to inspect individual cells.*

For instance, the true joint's plane/plane entry is 4.0, the estimate is 1.3,
and the absolute difference is 2.7 in the displayed units. This shows an error
in estimated probability mass, not the identity of mislabelled records. An
accurate joint distribution does not by itself demonstrate that the individual
candidate list is accurate. The paper evaluates the latter separately with
precision and recall, and final training quality separately again.

On a new dataset, compare the resulting training run with the unchanged pool
and with equal-size random removal under the same training budget. Check
per-class and relevant subgroup metrics as well as the average, especially
when rare examples receive high suspicion scores. Keep a held-out evaluation
set out of the cleaning decisions. If its labels also need an audit, version
that audit separately and report which evaluation labels each result uses.

Counting after predictions costs $O(NK+K^2)$ for $N$ records and $K$ classes,
including a $K\times K$ table. Producing the out-of-fold predictions can dominate
that cost: it requires several fits, not just a pass through the probabilities.
Human review and final retraining also belong in the selection ledger below.
The worked examples validate bookkeeping and arithmetic, not a CIFAR/ImageNet
reproduction or a measured saving on language-model training.

<!-- confident-learning-addition: end -->

<a id="selection-cost"></a>
## Amortizing the scoring cost

Let $C_F$ be the cost of one full-pool target run, $C_S$ the cost of one retained
subset run, and $C_P$ the cost of preliminary training, score computation, and
selection. Reusing the selected IDs for $K$ target runs costs

$$
\begin{aligned}
C_{\mathrm{selected}}&=C_P+K C_S,\\
C_{\mathrm{full}}&=K C_F.
\end{aligned}
$$

For $C_F>C_S$, the selection pays for itself when
$K(C_F-C_S)>C_P$. This calculation is conditional on acceptable target quality
and on the selected IDs remaining useful for all $K$ runs. A new label set,
training distribution, or model family may require new scoring and a new
quality comparison.

For an illustrative ledger, let one full run cost 100 units, a subset run
40 units, and the complete selection procedure 180 units. One target run then
costs 220 units with selection, versus 100 without. Three runs break even at
300; four cost 340 instead of 400. These are assigned accounting values, not
measurements or an empirical claim about an algorithm's speed.

```python
full_run, subset_run, preliminary = 100, 40, 180
ledger = []
for runs in range(1, 5):
    full = runs * full_run
    selected = preliminary + runs * subset_run
    ledger.append((runs, full, selected))
assert ledger == [(1, 100, 220), (2, 200, 260),
                  (3, 300, 300), (4, 400, 340)]
print("runs, full-pool cost, selection plus subset cost")
for row in ledger:
    print(*row)
```

The ledger must include all preliminary seeds, full-pool scoring passes, data
movement, and the cost of producing or loading representations. Per-example
gradients can also require more memory and computation than a forward-only
EL2N pass. Reusing scores makes their fixed cost smaller per target run; it
does not prove that a ranking transfers to a new target model.

---

К модулям: [[02 Areas/ML & DL/00 Учебник/_index#Углублённые модули|Выбрать углублённый модуль]]

**Следующая глава:** [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41c Active learning — запрос меток и выбор примеров|Active learning — запрос меток и выбор примеров]]

**Связанная тема:** [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/53 Синтетические данные и учебные программы|Синтетические данные, смеси и учебные программы]]

**Источники и код:** Paul, Ganguli and Dziugaite,
[Data Diet v2](https://arxiv.org/abs/2107.07075v2), §§2–4 and
[reference implementation](https://github.com/mansheej/data_diet/tree/aee462eab93aa0fb14c1ddfeba45d645b85a5fbb);
Kirsch, [reproduction](https://arxiv.org/abs/2303.14753v1), §§2–3 and
[HLB implementation and figures](https://github.com/BlackHC/pytorch_datadiet/tree/fda604dc34eade5918a2c98568c2dc0f7861ced3).
