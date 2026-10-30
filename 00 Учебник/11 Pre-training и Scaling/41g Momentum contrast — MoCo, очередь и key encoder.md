---
title: Momentum contrast — MoCo, очередь и key encoder
type: textbook-chapter
status: canonical
source_language: en
last_updated: 2026-10-28
last_verified: 2026-10-28
primary_sources:
  - https://arxiv.org/abs/1911.05722v3
  - https://arxiv.org/abs/2003.04297v1
  - https://github.com/facebookresearch/moco/tree/132aea9da9bb995a411c915c0420be3db62052f3
  - https://cs231n.stanford.edu/slides/2025/lecture_12.pdf
---

# 41g. Momentum contrast — MoCo, очередь и key encoder

An image batch can be small even when the contrastive objective needs many
negative examples. Encoding every negative again on every step would repeat
the expensive image-network computation. MoCo stores previously computed
key vectors in a queue and compares each new query with those vectors.
The current batch supplies the queries and their positive keys; the queue
supplies negatives from earlier batches.

Stored keys have an age. Their encoder may have changed since they were
computed. MoCo therefore updates the key encoder gradually from the query
encoder's parameters. The queue and this momentum update work together:
one preserves negative vectors across steps, while the other reduces changes
in the function that produces new keys. Old vectors are not recomputed when
the key encoder changes.

<a id="dictionary"></a>
## The dictionary is a queue of vectors

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/moco-2026-10-28/moco-dictionary.png|Original MoCo query encoder, momentum encoder and key queue]]

*Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie and Ross Girshick,
[Momentum Contrast, Figure 1](https://arxiv.org/abs/1911.05722v3).
The same diagram is released in the
[authors' project documentation](https://github.com/facebookresearch/moco/blob/aea97d4b6eb091846590506784f6809f4f4d79bd/README.md)
under CC BY-NC 4.0; [license copy](/licenses/moco-aea97d4-CC-BY-NC-4.0.txt).
The complete native-PDF diagram is retained, with unchanged labels.*

Two independent augmentations of one image form a query input $x^q$ and a
positive-key input $x^k$. The query encoder $f_q$ and key encoder $f_k$ return
vectors that are L2-normalized before comparison. The positive relation
comes from the two inputs' shared image identity, not a human class label.

For a batch of $B$ queries, collect query vectors in
$Q\in\mathbb R^{B\times d}$ and positive vectors in
$K^+\in\mathbb R^{B\times d}$. Let
$V\in\mathbb R^{d\times K}$ be the stored queue, with one negative key per
column. Each query has one matching current key and $K$ queued candidates.
The loss matrix has shape $B\times(K+1)$, even if $B$ is only two.

The word “dictionary” here refers to these candidate vectors. It is not
a tokenizer vocabulary, and the columns are not stable word or class IDs.
Nor is this a per-image memory bank containing one persistent entry for
every image in the dataset. MoCo retains a bounded window of recently
computed keys and replaces old entries as training proceeds.

### What changes relative to SimCLR

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/simclr-2026-10-28/framework.png|Original SimCLR shared-encoder framework for comparison with MoCo]]

*Ting Chen, Simon Kornblith, Mohammad Norouzi and Geoffrey Hinton,
[SimCLR, Figure 2, PMLR 119](https://proceedings.mlr.press/v119/chen20j.html),
CC BY 4.0. Both branches in this original diagram are differentiated during
SimCLR training. MoCo instead differentiates through the query branch and
updates its key branch by the parameter rule below.*

In [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41f Контрастивное предобучение — SimCLR и проверка представлений|SimCLR]],
the current $2N$ views provide both positive and negative comparisons.
Their embeddings are produced by current shared parameters, and gradients
pass through both views. Retaining more negatives normally requires a larger
joint candidate set, not merely more gradient-accumulation steps.

MoCo separates current image computation from the stored candidate count.
It computes both current query and key views, but stores old keys as tensors
without their original activation graphs. Increasing $K$ adds vector storage
and dot products, without asking the CNN to re-encode all $K$ images on
that step. The old keys are fixed targets for that step's gradient.

| Property | Current-batch SimCLR | MoCo v1/v2 implementation discussed here |
| --- | --- | --- |
| Negative candidates | Other views in the current joint batch | Stored keys from preceding batches |
| Key-branch parameter update | Backpropagation through shared encoder/head | Moving average of query parameters |
| Stored old image activations | No historical queue | Not retained with the queued vectors |
| Positive class in CE | Index of the matching view | Column 0, where the current positive is placed |
| Current dictionary size | Coupled to the joint set of views | Queue capacity configured separately from image batch size |

These mechanisms impose the same kind of instance discrimination, but their
training computations differ. A larger queue does not mean that old keys
have become up-to-date, or that the entire computation is independent of
batch size.

<a id="momentum-encoder"></a>
## A moving average of parameters

Write $\theta^q_t$ for the query parameters available at the beginning of
iteration $t$. In the pinned implementation, the key parameters are updated
inside the forward call, before computing current keys:

$$
\begin{aligned}
 \theta^k_t&=m\theta^k_{t-1}+(1-m)\theta^q_t,\\
 &0\le m<1.
\end{aligned}
$$

Only the query encoder receives the loss gradient. Its optimizer update
produces the next query state $\theta^q_{t+1}$. The key update averages
parameters, not predicted class probabilities and not the stored vectors.
It is also separate from SGD's momentum buffer: the source uses both a
key-encoder coefficient $m=0.999$ and an SGD momentum of 0.9.

Repeated substitution gives a weighted history:

$$
 \theta^k_t=m^t\theta^k_0
 +(1-m)\sum_{r=0}^{t-1}m^r\theta^q_{t-r}.
$$

This identity treats $t$ as the number of EMA updates starting from the
initial key parameters. Recent query states have larger coefficients.
The parameters initially match the query encoder, so the first EMA call
does not change them if the query has not yet been optimized.

For a single parameter, take $m=0.9$, initial key value 0 and manually
specified query values 2, 4, 4. The key values become 0.2, 0.58 and 0.922.
The following class retains the original implementation's momentum method;
its small constructor and parameter trace are added for this example.

```python
import torch
import torch.nn as nn

class MomentumPair(nn.Module):
    def __init__(self, momentum):
        super().__init__()
        self.encoder_q = nn.Linear(1, 1, bias=False, dtype=torch.float64)
        self.encoder_k = nn.Linear(1, 1, bias=False, dtype=torch.float64)
        self.encoder_k.requires_grad_(False)
        self.m = momentum
        with torch.no_grad():
            self.encoder_k.weight.zero_()

    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

pair = MomentumPair(.9)
history = []
for value in [2., 4., 4.]:
    with torch.no_grad():
        pair.encoder_q.weight.fill_(value)
    pair._momentum_update_key_encoder()
    history.append(pair.encoder_k.weight.item())
torch.testing.assert_close(torch.tensor(history), torch.tensor([.2, .58, .922]))
assert pair.encoder_k.weight.grad is None
print("key parameter history", history)
```

*Original `_momentum_update_key_encoder` method: Meta Platforms and
contributors, [pinned builder.py](https://github.com/facebookresearch/moco/blob/132aea9da9bb995a411c915c0420be3db62052f3/moco/builder.py),
MIT; [license copy](/licenses/moco-132aea9-MIT.txt). The original method
body is unchanged; the constructor and trace are not the full source model.*

A coefficient of 0.999 gives an e-folding scale of about 1000 updates and
a half-life of about 693 updates for an old parameter contribution. These
describe EMA weights, not the age of an image or a measured training delay.
With a queue of 65,536 keys and 256 newly inserted global keys per step,
one turnover takes 256 insertion steps. Over that many EMA updates, the
coefficient on an old state is $0.999^{256}\approx0.774$.

The stored vectors still came from different key-encoder states. Averaging
parameters reduces abrupt changes; for a nonlinear network it does not prove
that its outputs change by a fixed bound. A coefficient too close to one can
also make the teacher follow query updates slowly. The useful choice must
be checked with the queue, optimization and evaluation protocol.

<a id="fifo-queue"></a>
## Queue storage and chronological order

The source stores a $d\times K$ tensor and a pointer to the next columns
to replace. The pointer is part of the queue state. Physical column order
does not always equal chronological order, but that permutation does not
change a softmax denominator over the same set of negative vectors.

For a readable trace, use scalar IDs instead of embeddings. The capacity
is five and the pointer starts at column 4:

| Operation | Physical columns 0…4 | Next replacement | Chronological contents, oldest first |
| --- | --- | ---: | --- |
| Initial state | 10, 11, 12, 13, 14 | 4 | 14, 10, 11, 12, 13 |
| Insert 20, 21 | 21, 11, 12, 13, 20 | 1 | 11, 12, 13, 20, 21 |
| Insert 22, 23, 24 | 21, 22, 23, 24, 20 | 4 | 20, 21, 22, 23, 24 |

These IDs show replacement only; they are not vectors on which a contrastive
loss is computed. The generalized ring operation below permits insertion
across the last column. The original helper instead requires the global
batch size to divide $K$ and uses a contiguous slice, with a pointer initially
zero. That simpler source convention must not be presented as supporting
arbitrary batch sizes without modification.

```python
import torch

@torch.no_grad()
def enqueue(queue, pointer, keys):
    # queue: d x K; keys: B x d. This version supports wrap-around.
    d, capacity = queue.shape
    batch = keys.shape[0]
    if keys.ndim != 2 or keys.shape[1] != d or not 0 < batch <= capacity:
        raise ValueError("need 1..K new keys with matching feature dimension")
    if not isinstance(pointer, int) or not 0 <= pointer < capacity:
        raise ValueError("pointer must identify a queue column")
    slots = (torch.arange(batch, device=queue.device) + pointer) % capacity
    queue[:, slots] = keys.detach().T
    return (pointer + batch) % capacity

queue_ids = torch.tensor([[10., 11., 12., 13., 14.]])
pointer = enqueue(queue_ids, 4, torch.tensor([[20.], [21.]]))
assert pointer == 1 and queue_ids.tolist() == [[21., 11., 12., 13., 20.]]
pointer = enqueue(queue_ids, pointer, torch.tensor([[22.], [23.], [24.]]))
order = (torch.arange(5) + pointer) % 5
assert pointer == 4 and queue_ids[:, order].tolist() == [[20., 21., 22., 23., 24.]]
print("physical queue", queue_ids.tolist(), "pointer", pointer)
```

The source initializes all queue columns as random normalized vectors.
It does not first wait for $K$ real examples and mask an unfilled prefix.
During startup, the denominator therefore contains initialized vectors
until they have been replaced. After one turnover, all entries have come
from encoded inputs, assuming the queue was populated continuously.

A queue is not guaranteed to contain $K$ distinct images or $K$ different
semantic classes. Repeated sampling can insert the same image again.
The source stores vectors without image IDs, so it does not automatically
exclude a queued vector when the current image reappears. Same-class and
duplicate-image false negatives require separate consideration.

<a id="moco-loss"></a>
## One positive plus queued negatives

For a normalized query $q_b$, its current positive $k_b^+$ and queued
columns $v_j$, construct

$$
\begin{aligned}
 S_{b,0}&=q_b^\top k_b^+/\tau,\\
 S_{b,j+1}&=q_b^\top v_j/\tau,\quad j=0,\ldots,K-1,\\
 \mathcal L&=-\frac1B\sum_b
 \log\frac{e^{S_{b,0}}}{\sum_{r=0}^{K}e^{S_{b,r}}}.
\end{aligned}
$$

All CE targets are zero because the positive is placed in column 0.
Zero is a candidate index, not an image-class annotation. There is no
query self-comparison to mask on this $B\times(K+1)$ matrix.
The implementation does not add all other current positive keys as extra
in-batch negatives: the negative columns come from the old queue.

Take $q=(1,0)$, $k^+=(0.8,0.6)$ and three negative columns
$(0,1)$, $(-1,0)$ and $(0.6,0.8)$. At $\tau=0.2$, the logits are
$(4,0,-5,3)$. The positive probability is
$e^4/(e^4+1+e^{-5}+e^3)$. The negative with cosine 0.6 competes much
more strongly than the one with cosine −1.

The program differentiates with respect to the supplied normalized query
vector. A real encoder also differentiates through its L2 normalization;
this isolated check is not the full raw-feature gradient.

```python
import torch
import torch.nn.functional as F

def moco_logits(query, positive, queue, temperature):
    # All supplied rows/columns are already normalized.
    positive_logit = (query * positive.detach()).sum(dim=1, keepdim=True)
    negative_logits = query @ queue.detach().clone()
    return torch.cat([positive_logit, negative_logits], dim=1) / temperature

query = torch.tensor([[1., 0.]], dtype=torch.float64, requires_grad=True)
positive = torch.tensor([[.8, .6]], dtype=torch.float64, requires_grad=True)
negative_queue = torch.tensor([[0., -1., .6], [1., 0., .8]],
                              dtype=torch.float64, requires_grad=True)
old_queue = negative_queue.detach().clone()
logits = moco_logits(query, positive, negative_queue, .2)
torch.testing.assert_close(logits, torch.tensor([[4., 0., -5., 3.]], dtype=torch.float64))
loss = F.cross_entropy(logits, torch.zeros(1, dtype=torch.long))
probability = logits.detach().softmax(dim=1)
expected_gradient = ((probability[0, 0] - 1) * positive.detach()[0]
                     + old_queue @ probability[0, 1:]) / .2
with torch.no_grad():
    negative_queue.zero_()  # Reuse the buffer after the old snapshot was taken.
loss.backward()
torch.testing.assert_close(query.grad[0], expected_gradient)
assert positive.grad is None and negative_queue.grad is None
print("positive probability", probability[0, 0].item(), "loss", loss.item())
print("query gradient", query.grad.tolist())
```

Here the positive probability is approximately 0.721335 and the loss is
0.326652. The query-vector gradient is approximately $(-0.319012,0.291521)$.
A single current query can have a nonzero contrastive loss because its
alternatives already exist in the queue; a one-image SimCLR batch has no
other-image negatives.

Only the query path receives gradients here. The key encoder remains a
parameterized network, but `torch.no_grad()` avoids constructing its
current activation graph and the old queue has no image graph to revisit.
This is one reason a large stored dictionary has a different memory cost
from differentiating a large end-to-end image batch.

<a id="queue-snapshot"></a>
## Why the old queue is cloned

The pinned forward call updates the queue before returning logits to the
training loop. Backward happens afterward. Although the queue needs no
gradient of its own, its *old values* are needed to compute the query
gradient of $Q V$. Detaching a tensor removes its gradient path but still
shares its storage. It does not create an immutable copy.

The original negative-logit expression is preserved here:

> `l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])`

*Meta Platforms and contributors, pinned `builder.py`, MIT.*

The following failure case uses the same storage relationship without a
CNN or queue class. An in-place update invalidates the values saved for
backward. Taking a clone before the update preserves the needed snapshot.

```python
import torch

query = torch.tensor([[1., 2.]], dtype=torch.float64, requires_grad=True)
buffer = torch.eye(2, dtype=torch.float64)
unsafe = query @ buffer.detach()
buffer.add_(1.)
try:
    unsafe.sum().backward()
except RuntimeError as error:
    assert "modified by an inplace operation" in str(error)
else:
    raise AssertionError("expected a saved-tensor version error")

safe_query = torch.tensor([[1., 2.]], dtype=torch.float64, requires_grad=True)
buffer = torch.eye(2, dtype=torch.float64)
safe = safe_query @ buffer.detach().clone()
buffer.add_(1.)
safe.sum().backward()
torch.testing.assert_close(safe_query.grad, torch.ones_like(safe_query))
print("old queue snapshot preserved for backward")
```

Changing an update from in-place to another storage convention can change
the observed failure, but it does not change which dictionary the stated
loss is supposed to use. An alternative implementation can delay insertion
until after backward; it must retain the intended negative set and update
ordering rather than silently changing the training objective.

<a id="complete-step"></a>
## Two complete small training steps

The next program follows the pinned implementation's sequence: query
forward, key EMA and forward without gradients, logits against the old
queue, insertion of current keys, then backward and a query optimizer step.
It reuses `enqueue` and `moco_logits` from the preceding programs. The
networks are linear and the paired inputs are fixed vectors, not a
reproduction of ResNet training or image augmentations.

```python
import copy
import torch
import torch.nn.functional as F

torch.manual_seed(19)
query_encoder = torch.nn.Linear(2, 3, bias=False, dtype=torch.float64)
key_encoder = copy.deepcopy(query_encoder).requires_grad_(False)
queue = F.normalize(torch.randn(3, 6, dtype=torch.float64), dim=0)
pointer, momentum = 0, .9
optimizer = torch.optim.SGD(query_encoder.parameters(), lr=.03)
query_input = torch.tensor([[1., .2], [-.3, 1.]], dtype=torch.float64)
key_input = torch.tensor([[.9, .25], [-.25, .95]], dtype=torch.float64)

for step in range(2):
    query_before = query_encoder.weight.detach().clone()
    key_before = key_encoder.weight.detach().clone()
    optimizer.zero_grad()
    q = F.normalize(query_encoder(query_input), dim=1)
    with torch.no_grad():
        key_encoder.weight.mul_(momentum).add_(query_encoder.weight, alpha=1-momentum)
        k = F.normalize(key_encoder(key_input), dim=1)
    expected_key = momentum * key_before + (1-momentum) * query_before
    logits = moco_logits(q, k, queue, .2)
    loss = F.cross_entropy(logits, torch.zeros(2, dtype=torch.long))
    pointer = enqueue(queue, pointer, k)  # Before backward, as in source forward.
    loss.backward()
    assert torch.isfinite(query_encoder.weight.grad).all()
    assert key_encoder.weight.grad is None and not k.requires_grad
    optimizer.step()
    torch.testing.assert_close(key_encoder.weight, expected_key)
    assert not torch.equal(query_encoder.weight, query_before)
    if step == 1:
        assert not torch.equal(key_encoder.weight, key_before)
    print("step", step, "loss", loss.item(), "queue pointer", pointer)
assert pointer == 4
```

On the second iteration, the EMA incorporates the query parameters changed
by the first optimizer step. Each loss is computed against a different
queue state, so comparing the two scalar losses is not a controlled
before/after comparison on one fixed objective.

### Paper pseudocode and implementation timing

Algorithm 1 in the paper shows key forward, query SGD, then a key-parameter
EMA for the next iteration. In the implementation, EMA happens at the
beginning of the current key forward. With the initial query and key
parameters equal, these are compatible descriptions shifted by one step:
the first implementation EMA is a no-op; the next incorporates the preceding
query SGD update. EMA must not be applied twice simply because both
descriptions are present in an implementation plan.

The key vectors inserted on a step were computed before that step's query
optimizer update. Updating key parameters afterward would not retroactively
change those stored vectors. A queue checkpoint therefore needs its contents
and pointer as well as both encoders. Exact continuation also depends on
optimizer, step/schedule and data/random-state handling; saving only the
query weights does not restore a pretraining trajectory.

<a id="distributed-shuffle"></a>
## Global insertion and batch-normalization shuffle

In the original distributed implementation, the enqueue helper first
gathers keys from all ranks. Its effective insertion size is the global
number of keys, not one rank's local image batch. Replicated queues need
matching initial contents and pointer, and every rank must insert the same
gathered keys in the same order. Otherwise identical queries could be trained
against different negative dictionaries.

The source's gather is deliberately non-differentiable. The gathered keys
do not need gradients. That conclusion is specific to this stopped key path;
it should not be carried over to a SimCLR exchange where candidate embeddings
participate in end-to-end differentiation.

The image encoder also contains batch normalization. Its statistics couple
examples within a local batch, which can supply a shortcut when query and
positive-key batches have the same grouping. MoCo shuffles key inputs across
ranks, encodes them with these new groups, gathers the results and undoes the
permutation so that each query is paired with the right positive again.

Consider two ranks, each initially holding three examples. Rank 0's queries
are IDs 0,1,2 and rank 1's are 3,4,5. A global permutation changes key groups
to 4,1,5 and 0,3,2. Inverting the permutation after encoding restores the
original pairing. The markers below track ordering only; they do not simulate
batch-normalization outputs or a distributed collective.

```python
import torch

ids = torch.arange(6)
permutation = torch.tensor([4, 1, 5, 0, 3, 2])
inverse = permutation.argsort()
shuffled = ids[permutation]
rank_groups = shuffled.reshape(2, 3)
assert rank_groups.tolist() == [[4, 1, 5], [0, 3, 2]]
encoded_markers = shuffled * 10  # Mark each input's identity, not its BN feature.
restored = encoded_markers[inverse]
assert restored.tolist() == [0, 10, 20, 30, 40, 50]
assert restored[:3].tolist() == [0, 10, 20]
print("key rank groups", rank_groups.tolist(), "restored", restored.tolist())
```

A random permutation can still produce overlapping or even unchanged
groups; it is not a proof that every positive sees entirely different
companions. The operation breaks systematic grouping correspondence.
Failing to unshuffle, by contrast, changes positive identities directly.

Disabling parameter gradients does not put a network into evaluation mode.
The source trains both encoder modules in training mode. Its EMA helper
loops over parameters, not all `state_dict` entries: batch-norm running
statistics are buffers and are not averaged by that parameter loop.
Replacing the key branch with an `eval()` model therefore changes more
than gradient storage and needs separate justification.

<a id="cost-and-versions"></a>
## Cost, historical results and MoCo v2

With $d=128$, $K=65,536$ and float32 storage, one queue occupies
$128\cdot65,536\cdot4=33,554,432$ bytes, or 32 MiB. For a batch of 256,
the $256\times65,537$ float32 logits alone occupy 67,109,888 bytes,
slightly more than 64 MiB. Encoder weights, activations, gradients,
optimizer state, temporary buffers and replicated queues are additional.
Thus decoupling queue capacity from image batch size does not make large
dictionaries free.

Larger $K$ both increases comparisons and keeps keys longer at fixed
insertion rate. Those are different effects. More old negatives can also
mean more stale or false-negative candidates. The original momentum
ablation uses $K=4096$, not the default $65,536$:

| Key momentum $m$ | 0 | 0.9 | 0.99 | 0.999 | 0.9999 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reported ImageNet linear top-1, % | failed to converge | 55.2 | 57.8 | 59.0 | 58.9 |

These are the paper's ResNet-50 results under its specified pretraining
and linear-evaluation protocol. They do not prove that $m=0$ fails for
every possible queue, encoder or optimizer, nor that ever-larger $m$
always improves quality. “60.6% MoCo” refers to a different reported
configuration with the larger default queue.

[MoCo v2](https://arxiv.org/abs/2003.04297v1) retains the momentum/queue
mechanism and changes the training recipe. A two-layer MLP replaces the
single output projection, Gaussian blur is added, the temperature is
retuned for the MLP, and a cosine schedule is included in the final recipe.
The v2 note's ablations separate these changes:

| Source configuration | Pretraining epochs | ImageNet linear top-1, % |
| --- | ---: | ---: |
| Original MoCo v1 | 200 | 60.6 |
| MLP and its retuned temperature | 200 | 66.2 |
| Extra augmentation without MLP | 200 | 63.4 |
| MLP plus extra augmentation | 200 | 67.3 |
| MLP, extra augmentation and cosine schedule | 200 | 67.5 |
| Same final recipe, longer training | 800 | 71.1 |

The MLP result includes temperature retuning: in the note, adding an MLP
while keeping temperature 0.07 gives 62.9%; the 66.2% entry uses 0.2.
It would be incorrect to attribute that entire increase to architecture
while holding all other settings fixed. Nor is 71.1% the 200-epoch result.
The note also finds that linear classification and transferred detection
do not rank every ablation identically; one scalar probe does not establish
universal representation quality.

The repository documents five-trial 200-epoch means of 60.8±0.2 and
67.5±0.1 for v1 and v2, while its downloadable individual checkpoints have
their own scores. Historical time and memory measurements refer to the
authors' V100 setup, not current hardware or the small CPU programs above.
No dataset, checkpoint or full vision-training run is needed for those
programs. The frozen-probe procedure and its split-selection pitfalls are
developed in the preceding SimCLR chapter.

---

**Предыдущая глава:** [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41f Контрастивное предобучение — SimCLR и проверка представлений|Контрастивное предобучение — SimCLR и проверка представлений]]

**Следующая глава:** [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41h Masked autoencoders — MAE и восстановление патчей|Masked autoencoders — MAE и восстановление патчей]]

**Источники:** He et al.,
[Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722v3),
§§1–3, Algorithm 1 and §4.1; Chen et al.,
[Improved Baselines with Momentum Contrastive Learning](https://arxiv.org/abs/2003.04297v1),
the complete three-page note. Meta Platforms and contributors,
[builder.py](https://github.com/facebookresearch/moco/blob/132aea9da9bb995a411c915c0420be3db62052f3/moco/builder.py),
`loader.py`, and the training/configuration sections of `main_moco.py`, MIT.
Ehsan Adeli, [Stanford CS231n 2025, Lecture 12](https://cs231n.stanford.edu/slides/2025/lecture_12.pdf#page=87),
slides 87–93, for the course's MoCo/v2 sequence and references.
