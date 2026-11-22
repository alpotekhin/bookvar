---
title: Fine-tuning and PEFT — adapters, LoRA, QLoRA and soft prompts
type: textbook-chapter
status: canonical
source_language: en
last_updated: 2026-11-22
last_verified: 2026-11-22
aliases: [Fine-tuning, PEFT, LoRA, QLoRA, Prompt tuning, Prefix tuning]
previous: "[[00 Учебник/12 Post-training и Alignment/01 SFT и instruction data]]"
next: "[[00 Учебник/12 Post-training и Alignment/02 Preference data]]"
reference_terms:
  - label: Полное и частичное дообучение
    label_en: Full and partial fine-tuning
    anchor: choosing-trainable-parameters
  - label: Bottleneck adapters
    label_en: Bottleneck adapters
    anchor: bottleneck-adapters
  - label: LoRA — инициализация, градиенты и ранг
    label_en: LoRA — initialization, gradients and rank
    anchor: lora-update
  - label: Слияние LoRA и обслуживание нескольких адаптеров
    label_en: LoRA merging and multi-adapter serving
    anchor: merging-and-serving
  - label: QLoRA — обучение с квантованной базой
    label_en: QLoRA — training through a quantized base
    anchor: qlora-training
  - label: QLoRA — градиент через замороженную квантованную матрицу
    label_en: QLoRA — gradient through a frozen quantized projection
    anchor: qlora-frozen-gradient-example
  - label: Слияние LoRA и повторная квантизация
    label_en: LoRA merging and requantization
    anchor: qlora-requantized-merge
  - label: Память при полном дообучении, LoRA и QLoRA
    label_en: Training memory for full fine-tuning, LoRA and QLoRA
    anchor: peft-memory-budget
  - label: Soft prompts и prompt tuning
    label_en: Soft prompts and prompt tuning
    anchor: soft-prompts
  - label: Prefix tuning — обучаемые K/V
    label_en: Prefix tuning — learned keys and values
    anchor: prefix-keys-values
  - label: Дообучение энкодера на малой выборке
    label_en: Fine-tuning encoders on small datasets
    anchor: small-data-fine-tuning
  - label: Bias correction при дообучении BERT
    label_en: Adam bias correction during BERT fine-tuning
    anchor: fine-tuning-bias-correction
  - label: Переинициализация верхних слоёв
    label_en: Reinitializing upper layers
    anchor: fine-tuning-reinitialization
  - label: Число обновлений и выбор модели
    label_en: Update budgets and model selection
    anchor: fine-tuning-update-budget
  - label: Перенос через промежуточную задачу
    label_en: Intermediate-task transfer
    anchor: intermediate-task-transfer
  - label: Сохранение навыков при дообучении
    label_en: Retaining capabilities during fine-tuning
    anchor: retaining-capabilities
  - label: Replay при дообучении
    label_en: Replay during fine-tuning
    anchor: fine-tuning-replay
  - label: Регуляризация L2-SP
    label_en: L2-SP regularization
    anchor: l2-sp
  - label: Elastic Weight Consolidation
    label_en: Elastic Weight Consolidation
    anchor: elastic-weight-consolidation
  - label: Fisher и empirical Fisher в EWC
    label_en: Fisher and empirical Fisher in EWC
    anchor: fisher-for-ewc
primary_sources:
  - https://proceedings.mlr.press/v97/houlsby19a.html
  - https://arxiv.org/html/2106.09685v2
  - https://arxiv.org/html/2305.14314v1
  - https://arxiv.org/html/2407.21783v3#S3.T3
  - https://aclanthology.org/2021.emnlp-main.243/
  - https://aclanthology.org/2021.acl-long.353/
  - https://github.com/huggingface/peft/tree/fee923eb6e0382236478e4259bc69f4aeaab6813
  - https://arxiv.org/html/2006.05987v3
  - https://arxiv.org/html/2006.04884v3
  - https://proceedings.mlr.press/v80/li18a.html
  - https://arxiv.org/html/1612.00796v2
  - https://arxiv.org/html/1905.12558v3
  - https://arxiv.org/html/1712.03847v1
---

# Fine-tuning and PEFT

A 7-billion-parameter model stored at two bytes per weight takes 14 GB.
If invoice extraction, support and summarization each get a fully
fine-tuned checkpoint, storing the three copies takes 42 GB. With a
shared base and three BF16 LoRA adapters of about 4.2 million parameters
each, the stored weights instead take 14 GB plus about 25 MB. This
counts weight files, not training memory or serving caches. Every request
still runs the base network.

The [[00 Учебник/12 Post-training и Alignment/01 SFT и instruction data|SFT chapter]]
described which tokens contribute to the loss. Here we keep that choice
fixed and vary which tensors the optimizer can update. Full fine-tuning
updates the original weights. Parameter-efficient fine-tuning (PEFT)
updates a smaller set, such as selected layers, added LoRA factors or
prefix vectors. Those same parameters can be trained with another loss,
such as DPO.

The optional CPU listings run in order in one Python session. They check
the calculations on small tensors, without pretrained weights.

<a id="choosing-trainable-parameters"></a>
<span id="choosing-the-trainable-parameters"></span>

## Trainable parameters

For classification with an encoder, write the prediction as
$p(y\mid x)=\operatorname{softmax}(C f_{\theta}(x)+b)$.
The pretrained encoder computes a representation $f_\theta(x)$; the new
classifier maps it to the labels of the downstream task. A sentiment
classifier might have two rows in $C$, even though the original pretraining
head predicted a vocabulary of thousands of tokens.

In **feature extraction**, keep $\theta$ fixed and train $C,b$. With a linear
classifier this is often called a linear probe. The classifier can only
separate examples using information already accessible in the fixed
representation. When the encoder and preprocessing are deterministic, its
outputs can be computed once and cached for subsequent classifier updates.
If random augmentations or dropout are part of feature extraction, cached
features instead define one particular realization of that procedure.

In **full fine-tuning**, update the encoder and the classifier together.
The gradient can reshape the representation so that the classes become
easier to separate. **Partial fine-tuning** updates a specified subset,
such as the last two Transformer blocks and the classifier. Selective
methods can instead train biases or normalization parameters. Freezing the
whole base is common in PEFT, but is not the definition of every PEFT method.

For a causal LM, the output can remain a distribution over vocabulary
tokens. A classification target may be the text `positive`, whose tokens
are trained using the same shifted cross-entropy as another completion.
Changing the output head and changing the set of trainable parameters are
independent decisions. The distinction between feature extraction,
fine-tuning and inserted adapters is developed in
[Houlsby et al., Introduction and §2](https://proceedings.mlr.press/v97/houlsby19a.html).

For a binary classifier, take a representation $f=(1,2)$, a classifier
weight $c=(1,-1)$ and a positive target. Its logit is $z=c^Tf=-1$,
so $p=\sigma(-1)\approx0.26894$. Binary cross-entropy gives
$\partial\mathcal L/\partial z=p-1\approx-0.73106$. The classifier
gradient is $(p-1)f\approx(-0.73106,-1.46212)$. With full fine-tuning,
the representation also receives $(p-1)c\approx(-0.73106,0.73106)$,
which is propagated into the encoder. A linear probe changes the classifier
but keeps that representation fixed; it does not use the second gradient
to update the encoder.

<details id="peft-code-probe">
<summary>Code: a linear probe and encoder gradients</summary>

```python
import math
import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)
dtype = torch.float64
features = torch.tensor([1., 2.], dtype=dtype, requires_grad=True)
classifier = nn.Parameter(torch.tensor([1., -1.], dtype=dtype))
positive = torch.tensor(1., dtype=dtype)
F.binary_cross_entropy_with_logits(classifier @ features, positive).backward()
residual = torch.sigmoid(torch.tensor(-1., dtype=dtype)) - 1
torch.testing.assert_close(classifier.grad, residual * features.detach())
torch.testing.assert_close(features.grad, residual * classifier.detach())

probe_features = features.detach()
probe_classifier = nn.Parameter(classifier.detach().clone())
F.binary_cross_entropy_with_logits(probe_classifier @ probe_features, positive).backward()
assert probe_features.grad is None and probe_classifier.grad is not None
```

</details>

| Method | Parameters changed for a task | Where the change acts |
|---|---|---|
| Linear probe | New classifier | After a fixed representation |
| Partial/full fine-tuning | Selected/all original weights, plus any new head | Inside the existing network |
| Bottleneck adapter | New down/up projections, optionally other named parameters | Between existing sublayers |
| LoRA | New low-rank factors, optionally named heads or biases | Added to selected weight projections |
| Prompt tuning | New input embedding vectors | Before the first Transformer layer |
| Prefix tuning | New per-layer prefix states, often K/V | Inside attention at multiple layers |

“Frozen” means that an optimizer must not update a parameter. It does not
mean that every operation using it can be removed from backpropagation.
If a trainable adapter sits below a frozen layer, the loss still has to be
differentiated through that frozen layer to reach the adapter. This is
different from a linear probe sitting above an entirely frozen encoder.

<a id="bottleneck-adapters"></a>
## Bottleneck adapter modules

![[00 Учебник/Assets/Figures/curated/peft-2026-10-29/houlsby-adapter.png|Down-projection, nonlinearity and up-projection with a residual; two adapters in a post-norm Transformer block]]

*Neil Houlsby et al., [Parameter-Efficient Transfer Learning for NLP,
Figure 2, PDF p. 3](https://proceedings.mlr.press/v97/houlsby19a/houlsby19a.pdf#page=3),
CC BY 4.0. Both original diagram panels are retained. The caption is outside
the crop: green modules include the adapters and task-specific LayerNorm;
the task classifier is also trained but is not drawn. This is the paper's
post-norm Transformer, not a diagram of every modern decoder.*

The paper describes the bottleneck as follows:

> The adapters first project the original d-dimensional features into a
> smaller dimension, m, apply a nonlinearity, then project back to d
> dimensions. The total number of parameters added per layer, including
> biases, is 2md + d + m.

— [Houlsby et al., §2.1](https://proceedings.mlr.press/v97/houlsby19a.html),
CC BY 4.0; mathematical symbols are typeset as plain text in this quotation.

For a column vector $h\in\mathbb R^d$, one such module computes

$$
\begin{aligned}
z&=W_{\mathrm{down}}h+b_{\mathrm{down}},\\
u&=\sigma(z),\\
\operatorname{Adapter}(h)&=h+W_{\mathrm{up}}u+b_{\mathrm{up}}.
\end{aligned}
$$

Here $W_{\mathrm{down}}\in\mathbb R^{m\times d}$ and
$W_{\mathrm{up}}\in\mathbb R^{d\times m}$. The bottleneck width $m$
controls the number of task-specific parameters. For $d=768,m=64$,
one module has $2\cdot768\cdot64+768+64=99\,136$ parameters.
The original placement uses two modules in each Transformer layer; a
whole-model count must also include the trained normalization parameters
and task head. The residual connection inside the adapter is separate
from the residual connection around the attention or FFN sublayer.

A small numerical case makes the transformation explicit. Let
$h=(2,-1,0,3)$, let the down projection select the first coordinate, use
ReLU, and let the up projection add $0.1u$ to the second coordinate.
The bottleneck receives 2, ReLU returns 2, and the result is
$(2,-0.8,0,3)$. The module changes the representation while preserving
the original four-dimensional interface.

The skip connection permits an initialization close to the identity,
so inserting adapters need not immediately replace the pretrained
representation with unrelated activations. As training proceeds, the new
branch learns task-specific changes. In general the nonlinearity prevents
this branch from being folded into one constant update to a neighbouring
weight matrix: its effective linear map changes with the input. The extra
operations remain in the inference graph. Their latency depends on shapes,
batching and implementation; “20–30% overhead” is not a universal property.

<details id="peft-code-adapter">
<summary>Code: the bottleneck example</summary>

```python
h = torch.tensor([2., -1., 0., 3.], dtype=dtype)
down = torch.tensor([[1., 0., 0., 0.]], dtype=dtype)
up = torch.tensor([[0.], [.1], [0.], [0.]], dtype=dtype)
adapted = h + up @ torch.relu(down @ h)
torch.testing.assert_close(adapted, torch.tensor([2., -.8, 0., 3.], dtype=dtype))
assert 2 * 768 * 64 + 768 + 64 == 99136
```

</details>

<a id="lora-update"></a>
<span id="lora-changes-a-projection-through-two-small-factors"></span>

## LoRA

> LoRA represents the weight updates $\Delta W$ with two smaller matrices
> (called *update matrices*) through low-rank decomposition. These new
> matrices can be trained to adapt to the new data while keeping the
> overall number of parameters low. The original weight matrix remains
> frozen and doesn't receive any further updates.

— Hugging Face Team, [PEFT: LoRA](https://github.com/huggingface/peft/blob/fee923eb6e0382236478e4259bc69f4aeaab6813/docs/source/package_reference/lora.md),
Apache-2.0, excerpt retained in English.

![[00 Учебник/Assets/Figures/curated/peft-2026-10-29/lora-animated.gif|The same input passes through a frozen projection and a trainable A-then-B branch, whose outputs are added]]

*Hugging Face Team, [PEFT LoRA documentation](https://huggingface.co/docs/peft/main/en/package_reference/lora),
[original animation](https://huggingface.co/datasets/huggingface/documentation-images/blob/861b6c920fe27492d35256c39f266592a518d50c/peft/lora_animated.gif),
CC BY-NC-SA 4.0. The complete animation is copied unchanged.*

Take a pretrained projection $W_0\in\mathbb R^{d\times k}$ and an input
$x\in\mathbb R^k$. LoRA parameterizes the adapted output as

$$
\begin{aligned}
h&=W_0x+sBAx,\\
A&\in\mathbb R^{r\times k},\quad B\in\mathbb R^{d\times r},\quad
s=\alpha/r.
\end{aligned}
$$

The original matrix can still be full rank. The added matrix $BA$ has rank
at most $r$, and only $r(k+d)$ factor parameters are trained instead of
$dk$ entries of $W_0$. This is a constraint on the update, not a claim
that the pretrained matrix itself is low rank. Nor does it prove that the
update learned by unconstrained full fine-tuning must have low rank.

For a $4096\times4096$ projection and $r=8$, the factors contain
$8(4096+4096)=65\,536$ parameters, versus $16\,777\,216$ original weights:
one 256th as many. Applying this choice to two projections in each of 32
layers gives $4\,194\,304$ trainable factor parameters if both projections
are square.

<a id="lora-gqa-budget"></a>

<span id="compare-target-modules-at-the-same-parameter-budget"></span>

### LoRA placement at a fixed budget

[Llama 3 8B](https://arxiv.org/html/2407.21783v3#S3.T3) also has 32
layers and a hidden width of 4,096, but it has 32 query heads and
only eight K/V heads. Each head has width $4096/32=128$.
A value projection therefore returns $8\cdot128=1024$ coordinates,
while a query projection returns all 4,096. At rank 8, their LoRA
branches have $8(4096+1024)=40\,960$ and $8(4096+4096)=65\,536$
parameters respectively. The model-wide total for query and value is
$(65\,536+40\,960)\cdot32=3\,407\,872$.

| Projections per block | Rank | Total parameters |
| --- | ---: | ---: |
| Query and value | 8 | 3,407,872 |
| All attention projections | 4 | 3,407,872 |
| Attention and FFN | 8 | 20,971,520 |

The first two rows have the same budget: adding keys costs as much as
adding values, and the attention output projection costs as much as queries.
Halving the rank offsets doubling the sum of the projection dimensions.
The third row includes three SwiGLU matrices that connect widths 4,096 and
14,336. They add $3\cdot8(4096+14336)$ parameters per block beyond
adapting all four attention projections at rank 8.
These counts exclude embeddings, the output head, biases and any other
trainable tensors. At BF16 the first two checkpoints would each hold
about 6.82 MB of factors before metadata. To compare their quality, keep
the same SFT targets and
[[00 Учебник/12 Post-training и Alignment/01 SFT и instruction data#sft-accumulation-denominator|loss denominator]]
as well as the same data and evaluation.

The original LoRA paper investigates which attention projections to adapt
and how rank affects particular tasks. Its low-rank results on GPT-3 and
WikiSQL/MultiNLI are experimental observations, not a universal instruction
to use rank 8. A larger rank changes both the parameter count and, at fixed
$\alpha$, the scale $\alpha/r$. Compare target modules, scaling, data and
training budget together. See [Hu et al., §§4 and 7](https://arxiv.org/html/2106.09685v2).

In the paper's Table 5, the base is GPT-3 175B with 96 layers and the
adapter budget is fixed at about 18 million parameters. Targeting more
projections therefore requires a smaller rank for each one:

| Targets | Rank | WikiSQL | MultiNLI |
|---|---:|---:|---:|
| $W_q$ | 8 | 70.4 | 91.0 |
| $W_v$ | 8 | 73.0 | 91.0 |
| $W_q$, $W_v$ | 4 | 73.7 | 91.3 |
| $W_q$, $W_k$, $W_v$, $W_o$ | 2 | 73.7 | 91.7 |

Both task columns report validation accuracy in percent. The authors found typical
seed standard deviations of about 0.5 and
0.1 percentage points for the two tasks. A larger rank concentrated in
one projection did not outperform all lower-rank, broader placements.
The budget and target matrices must accompany a rank comparison.

<span id="reported-quality-and-the-parameter-budget"></span>

### GPT-3 results at different budgets

The main GPT-3 comparison in [LoRA, §5.5 and Table 4](https://arxiv.org/html/2106.09685v2)
uses different adapter budgets from the 18M diagnostic comparison above:

| GPT-3 175B | Full tuning | LoRA 4.7M | LoRA 37.7M |
|---|---:|---:|---:|
| Trainable | 175,255.8M | 4.7M | 37.7M |
| WikiSQL, % | 73.8 | 73.4 | 74.0 |
| MNLI-m, % | 89.5 | 91.7 | 91.6 |
| SAMSum R-1 | 52.0 | 53.8 | 53.4 |
| SAMSum R-2 | 28.0 | 29.8 | 29.2 |
| SAMSum R-L | 44.5 | 45.9 | 45.1 |

The two accuracy rows report validation scores. SAMSum uses
ROUGE-1, ROUGE-2 and ROUGE-L.

The authors report typical seed fluctuations of about 0.5 percentage
points on WikiSQL and 0.1 on MNLI-m. The 4.7M run is not better than the
full-weight baseline on every displayed number. Increasing the adapter
budget also does not improve every task. These results do not define a
universal “95–100% of full fine-tuning” quality range.

<a id="lora-gpt3-budget-plot"></a>

![[00 Учебник/Assets/Figures/curated/lora-primary-2026-11-10/lora-gpt3-figure2.png|Original GPT-3 175B plot: trainable parameter count versus validation accuracy on WikiSQL and MultiNLI-matched, comparing LoRA, full fine-tuning and two prefix methods.]]

*Edward J. Hu et al., [LoRA, Figure 2](https://arxiv.org/html/2106.09685v2#S5.F2).
[Microsoft's repository original](https://github.com/microsoft/LoRA/blob/c4593f060e6a368d7bb5af5273b8e42810cdef90/examples/NLG/figures/LoRA_GPT3.PNG),
[MIT](/licenses/microsoft-lora-c4593f06-MIT.txt). Image and original caption reproduced unchanged.*

On the horizontal axis, 7 means $10^7$ trainable parameters, not 7 million.
The full fine-tuning point at the far right still uses the same 175B
base as the LoRA triangles. The leftmost LoRA point, at about 4.7 million
parameters, is slightly below full fine-tuning on WikiSQL (left) and
above it on MultiNLI-matched (right). The crosses and stars show the
two prefix methods; several of their larger settings score lower than
smaller ones. The plot compares choices for this base and these two tasks,
not the effect of making the base model larger.

Budget labels also change checkpoint arithmetic: 4.7 million FP16 values
occupy about 9.4 decimal MB, whereas 18 million occupy about 36 MB, before
metadata or separately trained tensors. A 35 MB example must not be
attached to every 4.7M adapter. Likewise, rank 256 is not inherently
“almost full fine-tuning”: on one $4096\times4096$ projection its two
factors have $2\cdot4096\cdot256=2\,097\,152$ entries, one eighth
of the original matrix's parameter count.

<span id="the-first-update-including-both-factor-gradients"></span>

### Initialization and the first update

Initialize $B=0$ and $A$ to nonzero random values. Then $BA=0$, so the
adapted layer initially gives the base output. For one example, let
$g=\partial\mathcal L/\partial h$. The chain rule gives

$$
\begin{aligned}
\nabla_B\mathcal L&=s\,g(Ax)^T,\\
\nabla_A\mathcal L&=s\,B^Tg\,x^T,\\
\nabla_x\mathcal L&=W_0^Tg+sA^TB^Tg.
\end{aligned}
$$

At initialization the gradient of $A$ is zero, but the gradient of $B$
need not be. After $B$ changes, gradients can update $A$ as well. If both
factors were initialized to zero, both gradients would be zero and this
bilinear branch could not start learning through ordinary gradient descent.
The usual PEFT initialization uses Kaiming-uniform for $A$ and zero for $B$;
the paper and the animation describe a Gaussian choice for $A$. Both keep
the initial added matrix zero. The pinned documentation distinguishes
these choices in its *Initialization* section.

For an exact calculation, choose $W_0=I_2$, $A=(1,2)$, $B=(0,0)^T$,
$x=(1,-1)^T$, $s=1$, target $y=(0,0)^T$ and
$\mathcal L=\tfrac12\|h-y\|^2$. Then $Ax=-1$,
$h=(1,-1)^T$, and $\mathcal L=1$. The factor gradients are
$\nabla_B=(-1,1)^T$ and $\nabla_A=(0,0)$.
An SGD step of size 0.1 produces $B=(0.1,-0.1)^T$ while $A$ stays
unchanged. The new output is $(0.9,-0.9)^T$ and the loss is 0.81.

<details id="peft-code-lora-step">
<summary>Code: the first LoRA update</summary>

```python
W0 = torch.eye(2, dtype=dtype)  # frozen, no parameter gradient
A = nn.Parameter(torch.tensor([[1., 2.]], dtype=dtype))
B = nn.Parameter(torch.zeros(2, 1, dtype=dtype))
x = torch.tensor([1., -1.], dtype=dtype)
optimizer = torch.optim.SGD([A, B], lr=.1)

def lora_output(x):
    return W0 @ x + B @ (A @ x)

loss = .5 * lora_output(x).square().sum()
loss.backward()
torch.testing.assert_close(B.grad, torch.tensor([[-1.], [1.]], dtype=dtype))
torch.testing.assert_close(A.grad, torch.zeros_like(A))
assert W0.grad is None
optimizer.step()
torch.testing.assert_close(lora_output(x), torch.tensor([.9, -.9], dtype=dtype))
assert math.isclose((.5 * lora_output(x).square().sum()).item(), .81)

optimizer.zero_grad(set_to_none=True)
(.5 * lora_output(x).square().sum()).backward()
assert A.grad.norm().item() > 0

# A zero-zero factorization is a negative control, not an initialization recipe.
Az = nn.Parameter(torch.zeros(1, 2, dtype=dtype))
Bz = nn.Parameter(torch.zeros(2, 1, dtype=dtype))
(.5 * (W0 @ x + Bz @ Az @ x).square().sum()).backward()
assert Az.grad.abs().sum().item() == Bz.grad.abs().sum().item() == 0
```

</details>

<span id="a-frozen-layer-can-still-carry-the-gradient"></span>

### Backpropagation through frozen layers

Put a frozen projection $U$ above the adapted layer. Backpropagation needs
$U^T\nabla_y\mathcal L$ to reach $B$ and $A$ even though no gradient
for the entries of $U$ is accumulated. Wrapping that projection in
`torch.no_grad()` would detach its output and break this path.

<details id="peft-code-frozen-layer">
<summary>Code: input gradients through a frozen layer</summary>

```python
U = torch.tensor([[2., 0.], [0., 3.]], dtype=dtype)
optimizer.zero_grad(set_to_none=True)
(.5 * (U @ lora_output(x)).square().sum()).backward()
assert U.grad is None and B.grad.norm().item() > 0
with torch.no_grad():
    detached = U @ lora_output(x)
assert not detached.requires_grad
```

</details>

Freezing removes gradients and optimizer state for the base weights, but
does not remove the forward pass, all activation storage, or all backward
matrix multiplications. Gradient checkpointing can trade saved activations
for recomputation. The memory saving therefore need not equal the ratio
between the numbers of trainable parameters.

<a id="merging-and-serving"></a>
<span id="merging-checkpoints-and-serving-different-adapters"></span>

## Merging and serving LoRA adapters

![[00 Учебник/Assets/Figures/curated/peft-2026-10-29/lora-merge.png|The low-rank update is added to the base matrix before inference so that one projection replaces two branches]]

*Hugging Face Team, [PEFT: Merge LoRA weights into the base model](https://github.com/huggingface/peft/blob/fee923eb6e0382236478e4259bc69f4aeaab6813/docs/source/package_reference/lora.md#merge-lora-weights-into-the-base-model),
[original PNG](https://huggingface.co/datasets/huggingface/documentation-images/blob/861b6c920fe27492d35256c39f266592a518d50c/peft/lora_diagram.png),
CC BY-NC-SA 4.0. The drawing omits the scale $s$; in the equations below it
must be included unless absorbed into a factor.*

For a plain linear layer in deterministic inference, compute
$W_{\mathrm{merged}}=W_0+sBA$ once. Then
$W_{\mathrm{merged}}x=W_0x+sBAx$, and the extra branch no longer needs
to execute. This algebraic equality explains the “no additional inference
latency” claim for a merged LoRA projection. Training-time dropout in the
LoRA branch must be disabled for this comparison. Quantized weights add a
rounding step, so a quantized merge is not the same exact arithmetic identity.

<details id="peft-code-merge">
<summary>Code: the floating-point merge</summary>

```python
torch.manual_seed(29)
test_inputs = torch.randn(7, 2, dtype=dtype)
with torch.no_grad():
    unmerged = test_inputs @ W0.T + (test_inputs @ A.T) @ B.T
    merged_weight = W0 + B @ A
    merged = test_inputs @ merged_weight.T
torch.testing.assert_close(merged, unmerged, rtol=1e-12, atol=1e-12)
```

</details>

The PEFT documentation describes the deployment operation:

> To eliminate latency, use the `merge_and_unload` function to
> merge the adapter weights with the base model. This allows you to use the
> newly merged model as a standalone model. The `merge_and_unload`
> function doesn't keep the adapter weights in memory.

— Hugging Face Team, pinned *LoRA / Post-Training* documentation, Apache-2.0.
The documentation's API links are rendered as method names in this quotation.
In Python the method is `model.merge_and_unload()` and its return value is
the merged model.

An **adapter checkpoint** normally stores the factors and configuration,
not the entire frozen base. Loading it requires the correct base revision,
module names, tensor shapes, rank/scaling settings and any separately
trained head or embedding rows. A compatible shape alone does not establish
that two base checkpoints have compatible representations. A **merged
checkpoint** stores the resulting model weights and is correspondingly
larger.

For the three tasks at the start, separate merged checkpoints again
store three full weight files. With unmerged adapters, a server can
keep one base and select a branch per request. A batch with several
adapters then needs LoRA factors selected by request; a single shared
merged matrix cannot represent the batch. Implementations differ in
how they batch those extra projections.

Adding two adapters also requires choosing what to add:
a sum of completed weight updates does not include the cross terms
that appear when the factors themselves are added:

$$
\begin{aligned}
(B_1+B_2)(A_1+A_2)&=B_1A_1+B_2A_2\\
&\quad+B_1A_2+B_2A_1.
\end{aligned}
$$

The two cross terms are absent from the desired sum of independent updates.
Adapter composition must specify its operation and then be evaluated.

<a id="qlora-training"></a>
<span id="qlora-training-through-a-quantized-frozen-base"></span>

## QLoRA

![[00 Учебник/Assets/Figures/curated/quantization-review-2026-10/qlora-training-original.png|Full tuning stores optimizer state for the base; LoRA trains added factors; QLoRA also stores the base in four bits]]

*Tim Dettmers, Artidoro Pagnoni, Ari Holtzman and Luke Zettlemoyer,
[QLoRA, Figure 1](https://arxiv.org/html/2305.14314v1), CC BY 4.0.
Complete original figure, rendered from its standalone vector PDF. Read
the three columns as changes in stored weights, optimizer state and
gradient paths, not as three guarantees of peak GPU memory.*

QLoRA keeps a compressed representation of the frozen base weights while
training floating-point LoRA factors. The authors distinguish storage
from computation explicitly:

> QLoRA has one low-precision storage data type, in our case usually 4-bit,
> and one computation data type that is usually BFloat16. In practice,
> this means whenever a QLoRA weight tensor is used, we dequantize the
> tensor to BFloat16, and then perform a matrix multiplication in 16-bit.

— [Dettmers et al., §3](https://arxiv.org/html/2305.14314v1), CC BY 4.0;
the paper's method macro is expanded to “QLoRA”.

If $\widetilde W_0=\operatorname{dequant}(Q,c)$ is reconstructed from
packed codes $Q$ and scales $c$, the adapted projection is

$$
h=\widetilde W_0x+sBAx.
$$

The training loop updates $A,B$, not the packed base codes. The input
gradient still contains $\widetilde W_0^Tg$. A lower trainable adapter
therefore receives a gradient through a frozen, dequantized upper layer.
It is not necessary to differentiate a rounding operation to update the
factors. Quantization changes the fixed function through which those
gradients pass; it does not remove that computation.

<a id="qlora-frozen-gradient-example"></a>
<span id="one-update-through-a-frozen-quantized-projection"></span>

### An update through a quantized projection

Take a floating-point base $W_0=\operatorname{diag}(1.1,1.9)$.
For this constructed example, use uniform signed four-bit values with
scale $0.5$, rather than NF4. Rounding produces codes $(2,4)$ on the
diagonal, reconstructed as $\widetilde W_0=\operatorname{diag}(1,2)$.
The code below keeps those codes in ordinary integer tensors; it does
not pack two codes into a byte or implement a GPU quantization kernel.

Set $x=(1,-1)^T$, $A=(1,2)$, $B=(0,0)^T$, $s=1$, and
$\mathcal L=\tfrac12\|h\|^2$. The reconstructed base gives
$h=(1,-2)^T$ and loss $2.5$. Since $Ax=-1$, the LoRA gradients are
$\nabla_B=(-1,2)^T$ and $\nabla_A=(0,0)$. The input receives
$\nabla_x=\widetilde W_0^Th=(1,-4)^T$. An earlier trainable layer can
use this input gradient even though the base codes and scale stay fixed.

<details id="peft-code-quantized-step">
<summary>Code: the frozen quantized projection</summary>

```python
q_original = torch.diag(torch.tensor([1.1, 1.9], dtype=dtype))
q_scale = torch.tensor(.5, dtype=dtype)  # fixed, not an optimizer parameter
q_codes = torch.round(q_original / q_scale).clamp(-8, 7).to(torch.int8)
q_codes_before = q_codes.clone()
q_base = q_codes.to(dtype) * q_scale
q_x = torch.tensor([1., -1.], dtype=dtype, requires_grad=True)
q_a = nn.Parameter(torch.tensor([[1., 2.]], dtype=dtype))
q_b = nn.Parameter(torch.zeros(2, 1, dtype=dtype))
q_optimizer = torch.optim.SGD([q_a, q_b], lr=.1)

def quantized_adapter_output():
    return q_base @ q_x + q_b @ (q_a @ q_x)

q_initial = quantized_adapter_output()
torch.testing.assert_close(q_initial, torch.tensor([1., -2.], dtype=dtype))
assert not torch.allclose(q_initial, q_original @ q_x)
q_loss = .5 * q_initial.square().sum()
assert math.isclose(q_loss.item(), 2.5)
q_loss.backward()
torch.testing.assert_close(q_b.grad, torch.tensor([[-1.], [2.]], dtype=dtype))
torch.testing.assert_close(q_a.grad, torch.zeros_like(q_a))
torch.testing.assert_close(q_x.grad, torch.tensor([1., -4.], dtype=dtype))
assert q_base.grad is None and q_scale.grad is None and q_codes.grad is None

q_optimizer.step()
torch.testing.assert_close(
    quantized_adapter_output(), torch.tensor([.9, -1.8], dtype=dtype)
)
assert math.isclose((.5 * quantized_adapter_output().square().sum()).item(), 2.025)
torch.testing.assert_close(q_codes, q_codes_before)
```

</details>

Only $B$ changes on this first step; neither the rounding operation nor
the integer codes need a weight gradient. Zero-initialized $B$ preserves
the **reconstructed** base output, not the output of the original
floating-point weights. This example follows the frozen-base gradient
path in [QLoRA, §3, equations 5–6](https://arxiv.org/html/2305.14314v1#S3);
its uniform codebook and FP64 arithmetic are chosen for inspection,
not to reproduce NF4, double quantization or BF16 rounding.

<a id="qlora-requantized-merge"></a>

#### Merging and rounding

In the same uniform quantizer, take one reconstructed weight of 1
and a learned update of 0.2. For an input of 1, the unmerged layer
returns $1+0.2=1.2$. Storing the merged weight in floating point keeps
that sum. To store it again on the original grid with scale 0.5,
we round $1.2/0.5=2.4$ to code 2 and reconstruct 1. The update disappears
in this example. A different scale or codebook changes the error; the
floating-point merge identity alone cannot predict it.

### NF4, double quantization and paged optimizers

Three parts of the original recipe address different memory costs.
**NF4** stores block-normalized weights using a 16-value codebook designed
around a normal distribution. **Double quantization** compresses the
block scales themselves. **Paged optimizers** use unified memory for
optimizer-state pages to handle memory pressure. These pages are not
the inference KV-cache blocks from
[[00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention|PagedAttention]].
The codebook, normalization and quantization-error examples are developed
in the [[00 Учебник/14 Inference и оптимизация/57 Квантизация языковых моделей|quantization chapter]].

For a 64-weight block, one FP32 scale costs $32/64=0.5$ bits per weight.
In the paper's double-quantization accounting, 8-bit first-level scale
values and one FP32 second-level scale per 256 first-level scales cost

$$
\frac8{64}+\frac{32}{64\cdot256}=0.126953125
\quad\text{bits per weight}.
$$

The saving is $0.373046875$ bits per weight, about 3.03 decimal GB for
65 billion weights. This calculation concerns the scale representation;
it does not count every tensor, alignment byte or temporary allocation
of a particular library. Even before scale overhead, 175 billion 4-bit
weights require $175\cdot10^9/2=87.5$ decimal GB. Such a base cannot fit
in 48 GB merely because its trainable adapter is small.

<a id="qlora-target-placement"></a>

<span id="what-changes-when-adapters-are-moved-to-other-projections"></span>

### QLoRA placement

![[00 Учебник/Assets/Figures/curated/peft-qlora-2026-11-14/qlora-figure2-target-modules.svg|Five groups of runs. QLoRA-All is around 64.5 ROUGE-L, QLoRA-FFN around 64.0 and QLoRA-Attention around 63.4. The authors' 16-bit full fine-tuning baseline is around 64.3.]]

*Tim Dettmers, Artidoro Pagnoni, Ari Holtzman and Luke Zettlemoyer,
[QLoRA, Figure 2](https://arxiv.org/html/2305.14314v1#S4.F2),
[original SVG](https://arxiv.org/html/2305.14314v1/hyper.svg),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Unmodified figure.*

The authors fine-tuned the original LLaMA 7B on Alpaca. Each dot is
a run with a different seed; height is ROUGE-L. In the first three
groups the researchers stored the base in four bits. Adapting
attention alone reaches about 63.4; FFN alone reaches about 64.0.
Adapting both reaches about 64.5. The authors' tuned 16-bit full
fine-tuning baseline, `Alpaca (ours)`, is about 64.3. The original
Stanford-Alpaca recipe scores lower in this experiment; using it as
the only baseline would overstate the gain.

Moving from one group to another changes the placement and can also
change the number of trainable parameters. Unlike
[LoRA Table 5](https://arxiv.org/html/2106.09685v2#S7.T5),
the figure does not hold that number fixed. Nor do the dots measure
variation across datasets. In PEFT, `target_modules="all-linear"`
selects a broad set of modules. List the actual matches before applying
the seven-projection count from Llama 3 8B to another model.

<span id="account-for-base-weights-trainable-state-and-activations-separately"></span>

### Training memory

<a id="peft-memory-budget"></a>

Count frozen weights separately from trainable weights and their
associated state. Add activations and temporary buffers to estimate
the training footprint:

$$
\begin{aligned}
M\approx{}&M_{\mathrm{frozen\ weights}}+M_{\mathrm{trainable\ state}}\\
&+M_{\mathrm{activations}}+M_{\mathrm{temporary}}.
\end{aligned}
$$

For an illustrative mixed-precision Adam setup, a trainable parameter
could require two bytes for its weight, two for its gradient and eight
for FP32 first/second moments: 12 bytes. An additional FP32 master copy
would bring this to 16. Another implementation can keep parameters and
gradients in FP32 directly. State the actual dtypes and sharding scheme;
neither number is a universal PyTorch default.

Use the 7B model from the opening and a LoRA adapter with
$4\,194\,304$ parameters. At 12 bytes per trainable parameter the
adapter state occupies 50.3 MB. If every frozen weight is stored as a
four-bit code with the scale overhead derived above, the base costs
$7\cdot10^9(4+0.126953125)/8$ bytes, about 3.61 GB.

| Setup | Frozen weights | Trainable state | Subtotal |
|---|---:|---:|---:|
| Full tuning | 0 | 84 GB | 84 GB |
| LoRA, BF16 base | 14 GB | 50.3 MB | 14.05 GB |
| LoRA, 4-bit base | 3.61 GB | 50.3 MB | 3.66 GB |

The subtotal excludes activations, temporary buffers and any tensors
kept outside the quantized format. Halving the adapter rank halves
the factor state at fixed placement and dtype; it does not halve the
base or guarantee a similar decrease in activations. The paper's
breakdown below includes more of these terms.

<a id="qlora-memory-breakdown"></a>

![[00 Учебник/Assets/Figures/curated/peft-qlora-2026-11-14/qlora-figure6-training-memory.svg|Four normalized stacks for 7B, 13B, 33B and 65B. Total estimates are 6.9, 11.3, 24.7 and 45.0 GB. The quantized base is the largest block in every stack.]]

*Dettmers et al., [QLoRA, Appendix G, Figure 6](https://arxiv.org/html/2305.14314v1#A7.F6),
[original SVG](https://arxiv.org/html/2305.14314v1/mem.svg),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Unmodified figure.*

All four stacks end at 100%, so read absolute memory from the labels.
The 65B column has a total estimate of 45.0 GB. Its base alone is
37,074 MB and the optimizer contribution is 5,760 MB. The base accounts
for roughly 82% of the total. The estimate uses a batch of one,
512 tokens and gradient checkpointing; the input-gradient term omits
attention. Longer inputs can enlarge the part of memory that the chart
only estimates. The separate “65B on a 48 GB GPU” result used the
authors' training configuration with paged optimizer state. This
stack is not a peak-memory measurement or a promise that 3 GB will
be free on a different run.

The paper also reports 99.3% for Guanaco 65B relative to ChatGPT on
[the Vicuna evaluation](https://arxiv.org/html/2305.14314v1#S5.T6).
That number is a ratio of scores assigned by GPT-4, not an estimate
of memory, a win probability or a comparison with full fine-tuning.

<details id="peft-code-cost">
<summary>Code: factor and scale accounting</summary>

```python
rank, width, layers = 8, 4096, 32
per_projection = rank * (width + width)
adapter_parameters = 2 * layers * per_projection
assert per_projection == 65536
assert width * width // per_projection == 256
assert adapter_parameters == 4194304
assert 7_000_000_000 * 12 / 1e9 == 84
scale_bits = 8 / 64 + 32 / (64 * 256)
saved_gb_65b = (.5 - scale_bits) * 65_000_000_000 / 8 / 1e9
assert math.isclose(saved_gb_65b, 3.031005859375)
assert 175_000_000_000 * 4 / 8 / 1e9 == 87.5
```

</details>

<a id="soft-prompts"></a>
<span id="soft-prompts-change-the-input-embeddings"></span>

## Prompt tuning

![[00 Учебник/Assets/Figures/curated/peft-2026-10-29/prompt-tuning.png|Full tuning uses separate model weights for each task; prompt tuning reuses the model and learns separate input vectors]]

*Brian Lester, Rami Al-Rfou and Noah Constant,
[The Power of Scale for Parameter-Efficient Prompt Tuning, Figure 2](https://aclanthology.org/2021.emnlp-main.243/),
as reproduced in [PEFT](https://huggingface.co/docs/peft/main/en/package_reference/prompt_tuning).
Original PNG from the pinned Hugging Face documentation-image repository,
CC BY-NC-SA 4.0. The 11B/20K counts describe the illustrated T5 example;
they are not dimensions required by prompt tuning.*

The PEFT documentation gives the parameter distinction directly:

> Prompt tuning adds a task-specific, virtual prompt to the input that
> consists of trainable vectors in the embedding space. The virtual token
> parameters are updated independently of the pretrained model parameters
> which are frozen.

— Hugging Face Team, *Prompt tuning*, Apache-2.0; the linked method name is
rendered as ordinary text here.

A hard prompt such as `Classify the review:` selects existing token
embeddings. A learned soft prompt instead stores a matrix
$P\in\mathbb R^{p\times d}$ of continuous vectors. They need not equal
any rows of the vocabulary embedding table. With embedded text
$X\in\mathbb R^{n\times d}$, the first layer receives
$[P;X]\in\mathbb R^{(p+n)\times d}$. Training keeps the base weights
fixed and updates $P$ from the target loss.

For $p=20,d=4096$, this is 81,920 trainable values. The prompt travels
through the ordinary network, so later representations depend on it.
In a causal decoder it occupies attention positions and contributes K/V
to later tokens. Its positions are context, not extra target words to
be reproduced. Labels, attention masks and position IDs must agree with
the way the prompt is inserted. Reserving more virtual positions may
leave less room for ordinary text under a fixed sequence-length limit.

Consider one attention head with identity Q/K/V projections. There is
one real token $x=(1,0)$ and one soft-prompt vector $p=(0,0)$. The query
for the real position attends to the prefix and itself. Its two scores
are $0$ and $1/\sqrt2$, giving attention weights approximately
$(0.33024,0.66976)$. The output is $(0.66976,0)$ rather than $(1,0)$.
Even a zero prompt changes the denominator of the softmax; it is not
automatically an identity initialization.

<details id="peft-code-prompt">
<summary>Code: attention to a soft prompt</summary>

```python
def attention_row(query, keys, values):
    weights = torch.softmax(keys @ query / math.sqrt(query.numel()), dim=0)
    return weights @ values

real = torch.tensor([1., 0.], dtype=dtype)
prompt = nn.Parameter(torch.zeros(1, 2, dtype=dtype))
tokens = torch.cat([prompt, real[None]], dim=0)
soft_output = attention_row(real, tokens, tokens)
assert math.isclose(soft_output[0].item(), .6697615493266569)
assert soft_output[1].item() == 0
target = torch.tensor([0., 1.], dtype=dtype)
(.5 * (soft_output - target).square().sum()).backward()
assert prompt.grad.norm().item() > 0
```

</details>

The loss updates the prompt through its effects on both keys and values,
despite the projections being fixed identities. In a real Transformer,
backpropagation passes through the frozen layers in the same way.
Initializing a prompt from embeddings of readable words can supply a
starting point, but optimization is then continuous and need not preserve
those word identities.

The original study used T5 and examined how prompt-tuning performance
changed with model size. It also tested a one-time LM-adaptation stage
before freezing T5, because span-corruption pretraining encouraged
sentinel-prefixed targets. Thus its comparison is not evidence that
an arbitrary frozen base and twenty zero vectors will match full
fine-tuning. See [Lester et al., §§2–3](https://aclanthology.org/2021.emnlp-main.243/).

<a id="prefix-keys-values"></a>
<span id="prefix-tuning-learns-states-at-multiple-layers"></span>

## Prefix tuning

![[00 Учебник/Assets/Figures/curated/peft-2026-10-29/prefix-tuning.png|Fine-tuning changes the network at all sequence positions; prefix tuning learns states at prefix positions while the network stays frozen]]

*Xiang Lisa Li and Percy Liang,
[Prefix-Tuning, Figure 1](https://aclanthology.org/2021.acl-long.353/),
as reproduced in [PEFT](https://huggingface.co/docs/peft/main/en/package_reference/prefix_tuning).
Original PNG from the pinned Hugging Face documentation-image repository,
CC BY-NC-SA 4.0. Red denotes trainable parameters/states, grey the shared
frozen model. Each vertical block denotes activations at a time step,
not a new Transformer layer.*

> Prefix tuning prefixes a series of task-specific vectors to the input
> sequence that can be learned while keeping the pretrained model frozen.
> The prefix parameters are inserted in all of the model layers.

— Hugging Face Team, *Prefix tuning*, Apache-2.0; linked method names are
rendered as ordinary text.

An input soft prompt induces all of its deeper states by executing the
frozen network. Prefix tuning can instead provide trainable states
directly at multiple layers. For a decoder attention head, represent
these as learned keys $K_P$ and values $V_P$. For real-token states
$Q,K_X,V_X$, compute

$$
\operatorname{Attention}(Q,[K_P;K_X],[V_P;V_X]).
$$

The real queries can use the prefix positions and the allowed ordinary
positions. The prefix need not be a vocabulary sequence, or even the
K/V produced by applying one common input embedding matrix. Its keys
control which queries select it; its values control what information
is returned. Causal masking of ordinary tokens still applies.

In the preceding one-head example, keep the prefix key at $(0,0)$ but
give its value $(0,2)$. The scores do not change. The output becomes
$(0.66976,0.66048)$: the prefix contributes a second-coordinate value
without also changing its key. This freedom was absent from our
identity-projection soft prompt, where the same vector supplied both.

<details id="peft-code-prefix">
<summary>Code: independent prefix keys and values</summary>

```python
prefix_k = nn.Parameter(torch.zeros(1, 2, dtype=dtype))
prefix_v = nn.Parameter(torch.tensor([[0., 2.]], dtype=dtype))
prefix_output = attention_row(
    real, torch.cat([prefix_k, real[None]]),
    torch.cat([prefix_v, real[None]])
)
torch.testing.assert_close(prefix_output, torch.tensor([
    .6697615493266569, .6604769013466862
], dtype=dtype))
(.5 * (prefix_output - target).square().sum()).backward()
assert prefix_k.grad.norm().item() > 0 and prefix_v.grad.norm().item() > 0
```

</details>

For $L$ decoder layers, $p$ positions, $H_{KV}$ KV heads and head size
$d_h$, direct per-layer K/V prefixes contain
$2LpH_{KV}d_h$ values. With $L=32,p=20,H_{KV}=8,d_h=128$, that is
1,310,720 values, about 2.62 decimal MB at BF16. It exceeds the 81,920
input-prompt values above because this parameterization independently
stores K/V at every layer. A prefix encoder MLP can introduce additional
training parameters; the direct K/V count describes the materialized
prefix, not every possible training parameterization.

Li and Liang found direct optimization sensitive to initialization and
instead parameterized the prefix through an MLP. Once training finished,
they materialized the prefix and discarded that training MLP. This does
not merge the prefix into the model weights: attention still reads it
during generation. In a causal decoder with a fixed prefix and fixed
positions, its learned K/V can be prepared for reuse, but request-specific
states must still be computed. The original paper also treated
encoder-decoder prefixes; a library's decoder-only implementation is
not automatically the same algorithm. See
[Li and Liang, §§3–4.3](https://aclanthology.org/2021.acl-long.353/).

<a id="small-data-fine-tuning"></a>
## Fine-tuning encoders on small datasets

Consider a sentence-pair classifier with 1,000 labelled examples. The
pretrained encoder already represents the two texts, but the classification
head is new. Training must connect those representations to the task's
labels. Changing the head initialization or the order of the first batches
can change that process, even with the same learning rate and number of
epochs. This setting updates model parameters; it is different from putting
a few demonstrations into a frozen model's context.

Zhang et al. describe the scope of their study in the original repository:

> In this paper, we study the problem of few-sample BERT fine-tuning and identify three sub-optimal practices.
> First, we observe that the omission of the gradient bias correction in the BERTAdam makes fine-tuning unstable.
> We also find that the top layers of BERT provide a detrimental initialization and simply re-initializing these layers improves convergence and performance.
> Finally, we observe that commonly used recipes often do not allocate sufficient time for training.

— Tianyi Zhang, Felix Wu, Arzoo Katiyar, Kilian Q. Weinberger and Yoav Artzi,
[original README, Overview](https://github.com/asappresearch/revisit-bert-finetuning/blob/0aa4f4e117ee4422f7cb9355158203e01d6730db/README.md),
Copyright 2020 ASAPP Inc., MIT. “Bias correction” below refers to the
first- and second-moment estimates in Adam.

![[00 Учебник/Assets/Figures/curated/small-data-finetuning-2026-10-29/asapp-overview.png|Original comparison of bias correction, reinitialization and training duration]]

*Zhang et al., [original repository illustration](https://github.com/asappresearch/revisit-bert-finetuning/blob/0aa4f4e117ee4422f7cb9355158203e01d6730db/repo_illustration.png),
MIT. All three panels are retained without cropping or redrawing. The left
panel reports a distribution of test scores, the middle panel changes the
initial weights of selected layers, and the right panel varies the update
budget on STS-B with 1,000 training examples. These are separate experiments,
not three settings measured on one common axis.*

<span id="a-failed-optimization-run-and-a-generalization-difference"></span>

### Optimization failure and generalization

A balanced binary classifier that predicts 0.5 for every training example
has cross-entropy $-\log 0.5\approx0.6931$. If its loss stays near this
value, a poor development score does not show that the model memorized the
training set: it has not even fitted the training labels. By contrast,
two runs can both fit the training set and disagree on unseen examples.
Those runs have different generalization, not the same optimization failure.

The following prescribed predictions illustrate the distinction. They
are not results of training BERT.

<details id="peft-code-failure">
<summary>Code: failed runs and held-out accuracy</summary>

```python
balanced_targets = torch.tensor([0., 0., 1., 1.], dtype=dtype)
failed_probs = torch.full((4,), .5, dtype=dtype)
failed_loss = F.binary_cross_entropy(failed_probs, balanced_targets)
assert math.isclose(failed_loss.item(), math.log(2))

train_probs = torch.tensor([.01, .01, .99, .99], dtype=dtype)
fitted_loss = F.binary_cross_entropy(train_probs, balanced_targets)
assert math.isclose(fitted_loss.item(), -math.log(.99))
dev_a = torch.tensor([.1, .2, .8, .9], dtype=dtype)
dev_b = torch.tensor([.1, .8, .2, .9], dtype=dtype)
assert ((dev_a >= .5) == balanced_targets.bool()).sum().item() == 4
assert ((dev_b >= .5) == balanced_targets.bool()).sum().item() == 2

# A constructed gradient-path example, not a model of all BERT failures.
lower_weight = nn.Parameter(torch.tensor(10., dtype=dtype))
head_weight = nn.Parameter(torch.tensor(1., dtype=dtype))
hidden = torch.tanh(lower_weight)
F.binary_cross_entropy_with_logits(
    head_weight * hidden, torch.tensor(0., dtype=dtype)
).backward()
assert head_weight.grad.item() > .7
assert 0 < lower_weight.grad.item() < 1e-7
```

</details>

In the last calculation, the head receives a gradient of about 0.731,
while the lower parameter receives about $6.0\cdot10^{-9}$.
The derivative of $\tanh(10)$ supplies the small multiplier. An upper
head can therefore keep changing while almost no gradient reaches a lower
parameter. This constructed example explains a gradient path; it does not
establish that one saturated tanh causes every failed Transformer run.

Mosbach et al. measured a related separation in actual fine-tuning:
failed runs had very small gradients in lower BERT layers and a nearly
constant training loss. Successful runs could also begin with small
gradients, then develop larger gradients as training continued. Their
comparison of runs with similar final training loss further showed that
removing failed optimization runs does not eliminate all variation in
development performance. See [§§4–5 and Figures 2, 4 and 8](https://arxiv.org/html/2006.04884v3).

These observations also limit the explanation “fine-tuning failed because
it forgot pretraining.” Forgetting and failure can occur together, but a
classifier still near its trivial training loss has not successfully
learned the new task. Preserving MLM perplexity alone does not establish
that downstream optimization is working.

<a id="fine-tuning-bias-correction"></a>
<span id="adam-bias-correction-changes-the-early-updates"></span>

### Adam bias correction

Take a scalar parameter with first gradient $g_1=2$, learning rate
$\alpha=0.001$, $\beta_1=0.9$ and $\beta_2=0.999$. For this calculation
only, set epsilon and weight decay to zero. Starting the moment estimates
at zero gives

$$
\begin{aligned}
m_1&=(1-0.9)2=0.2,\\
v_1&=(1-0.999)2^2=0.004.
\end{aligned}
$$

Using these raw moments changes the parameter by
$0.001\cdot0.2/\sqrt{0.004}\approx0.0031623$.
Correcting the moments gives
$\hat m_1=0.2/0.1=2$ and $\hat v_1=0.004/0.001=4$,
so the change is $0.001$. Although both raw moments are smaller than
their corrected values, their ratio produces a larger update here.

For step $t$, Adam computes

$$
\begin{aligned}
m_t&=\beta_1m_{t-1}+(1-\beta_1)g_t,\\
v_t&=\beta_2v_{t-1}+(1-\beta_2)g_t^2,\\
\hat m_t&=m_t/(1-\beta_1^t),\qquad
\hat v_t=v_t/(1-\beta_2^t).
\end{aligned}
$$

Holding the current moment estimates fixed and ignoring epsilon, the
magnitude of the update without correction divided by the corrected
magnitude is

$$
R_t=\frac{1-\beta_1^t}{\sqrt{1-\beta_2^t}}.
$$

For these beta values, $R_1\approx3.162$, $R_{10}\approx6.528$,
$R_{100}\approx3.241$ and $R_{1000}\approx1.258$.
The correction has a deterministic dependence on the update count; it is
not an unpredictable jump in the learning rate. Nor is its early shape
the same as a monotone linear warmup. Epsilon placement and weight decay
must be retained when comparing actual optimizer implementations.

<details id="peft-code-adam">
<summary>Code: the first Adam update</summary>

```python
beta1, beta2, step_size, first_gradient = .9, .999, .001, 2.
first_m = (1 - beta1) * first_gradient
first_v = (1 - beta2) * first_gradient**2
raw_update = step_size * first_m / math.sqrt(first_v)
corrected_update = step_size * (first_m / (1 - beta1)) / math.sqrt(
    first_v / (1 - beta2)
)
assert math.isclose(raw_update / corrected_update, math.sqrt(10))
assert math.isclose(corrected_update, .001)
ratio_10 = (1 - beta1**10) / math.sqrt(1 - beta2**10)
assert 6.52 < ratio_10 < 6.54

scalar_parameter = nn.Parameter(torch.tensor(1., dtype=dtype))
scalar_parameter.grad = torch.tensor(first_gradient, dtype=dtype)
scalar_adam = torch.optim.AdamW(
    [scalar_parameter], lr=step_size, betas=(beta1, beta2),
    eps=0., weight_decay=0.
)
scalar_adam.step()
assert math.isclose(scalar_parameter.item(), 1 - corrected_update)
```

</details>

The historical BERTAdam omitted this correction. Zhang et al. compared
the two optimizers over 50 seeds and reported that 48% of their uncorrected
RTE runs scored below 55% accuracy. That percentage belongs to their
BERT-Large setup and data split. It is not a failure probability for
every small dataset or a promised improvement from changing one import.
Their [§4 and Algorithm 1](https://arxiv.org/html/2006.05987v3) describe
the comparison; the published code also exposes the optimizer choice.

Mosbach et al. found benefits from combining bias correction with explicit
warmup. Training longer with a smaller learning rate was another route
to more stable optimization in their experiments. The update formula
explains a changed step size; the empirical studies supply the evidence
about failed BERT runs. The formula alone does not prove that a particular
step size will cause vanishing gradients.

<a id="fine-tuning-reinitialization"></a>
### Reinitializing selected upper layers

A pretrained encoder is an initialization for the downstream optimization,
not a requirement to retain every initial tensor. Zhang et al. tested
resetting the pooler and the upper $L$ Transformer blocks before
fine-tuning. The architecture and depth remained unchanged. The lower
blocks retained pretrained weights, and subsequent full fine-tuning could
update both the retained and reset blocks.

The original implementation resets linear and embedding weights from a
normal distribution with the configured BERT initializer scale, zeroes
linear biases, and resets LayerNorm scale/bias to one/zero. The pooler
is reset separately. Resetting every parameter to random values would
discard the transfer from pretraining; resetting only the classifier is
the usual baseline, not this upper-layer intervention.

This reduced example preserves those parameter-type distinctions:

<details id="peft-code-reset">
<summary>Code: resetting upper layers</summary>

```python
def reset_top_blocks(blocks, pooler, count, std=.02):
    if not 0 <= count <= len(blocks):
        raise ValueError("count must be between zero and the block count")
    chosen = [] if count == 0 else list(blocks)[-count:]
    with torch.no_grad():
        for part in [pooler, *chosen]:
            for module in part.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.normal_(0., std)
                elif isinstance(module, nn.LayerNorm):
                    module.weight.fill_(1.)
                    module.bias.zero_()
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.zero_()

torch.manual_seed(59)
toy_blocks = nn.ModuleList([
    nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.LayerNorm(4)).to(dtype)
    for _ in range(3)
])
toy_pooler = nn.Linear(4, 4).to(dtype)
before_blocks = [
    {name: value.detach().clone() for name, value in block.state_dict().items()}
    for block in toy_blocks
]
reset_top_blocks(toy_blocks, toy_pooler, count=0)
for block, before in zip(toy_blocks, before_blocks):
    for name, value in block.state_dict().items():
        torch.testing.assert_close(value, before[name])

reset_top_blocks(toy_blocks, toy_pooler, count=1)
for block, before in zip(toy_blocks[:2], before_blocks[:2]):
    for name, value in block.state_dict().items():
        torch.testing.assert_close(value, before[name])
assert not torch.equal(toy_blocks[-1][0].weight, before_blocks[-1]["0.weight"])
torch.testing.assert_close(toy_blocks[-1][2].weight, torch.ones(4, dtype=dtype))
torch.testing.assert_close(toy_blocks[-1][2].bias, torch.zeros(4, dtype=dtype))
assert all(p.requires_grad for p in toy_blocks.parameters())
```

</details>

The zero-block case still resets the pooler. It must not accidentally
select every block through Python's `[-0:]` slice. Reinitialization happens
before constructing a new fine-tuning optimizer; otherwise optimizer
moments from a prior run would not correspond to the reset tensors.

This example adapts the BERT branch of the authors'
[`run_glue.py`, lines 736–764](https://github.com/asappresearch/revisit-bert-finetuning/blob/0aa4f4e117ee4422f7cb9355158203e01d6730db/run_glue.py#L736-L764),
Copyright 2020 ASAPP Inc., MIT. It uses current PyTorch module types and
small synthetic blocks, not the original Transformers 2.8 training stack.

In the paper, $L$ was selected from 1–6 using mean validation performance.
Resetting more layers did not give an unlimited improvement: performance
could plateau or decrease as useful transferred features were removed.
The results support testing a different initialization; they do not prove
that layers 1–4 always encode grammar and layers 9–12 are disposable.
The original pooler also participated in NSP, so calling it an
“MLM-specific head” would misdescribe its pretraining role.

<a id="fine-tuning-update-budget"></a>
### Epochs, optimizer updates and the selection budget

With 1,000 examples and batch size 32, one epoch has 32 updates if the
last partial batch is retained. Three epochs give 96 updates. With 2,400
examples, there are 75 updates per epoch and 225 in three epochs.
The same number of epochs therefore does not specify the same optimization
budget. Dropping the partial batch, accumulating gradients or distributing
examples changes the count again.

<details id="peft-code-budget">
<summary>Code: counting optimizer updates</summary>

```python
def single_device_updates(examples, batch, epochs, drop_last=False):
    per_epoch = examples // batch if drop_last else math.ceil(examples / batch)
    return epochs * per_epoch

assert single_device_updates(1000, 32, 3) == 96
assert single_device_updates(1000, 32, 3, drop_last=True) == 93
assert single_device_updates(2400, 32, 3) == 225
assert single_device_updates(1000, 32, 20) == 640
```

</details>

A linear schedule also depends on this count. With a 10% warmup, a
96-update run has roughly ten warmup updates, whereas a 3,200-update run
has 320. Extending training changes the learning-rate trajectory, not
only the number of times the labels are revisited.

Zhang et al. searched update budgets from 200 through 3,200 for their
downsampled datasets. They evaluated ten checkpoints per run in both the
short and long settings, selected the budget using eight seeds, then
reported test results over 20 seeds. This controls the number of
checkpoint candidates while changing the training budget. See
[§6](https://arxiv.org/html/2006.05987v3).

Mosbach et al. used 20 epochs, bias-corrected Adam, a peak learning rate
of $2\cdot10^{-5}$ and 10% linear warmup in their proposed baseline.
Their evidence concerns small GLUE classification datasets. It does not
establish that every task benefits from driving training loss to zero,
or that 20 epochs should replace three as another universal default.

<a id="intermediate-task-transfer"></a>
### Transferring through an intermediate task

Instead of adapting directly from pretraining to a small labelled task,
one can first fine-tune on a larger labelled task, then transfer that
encoder to the target task. For example, MNLI and RTE both classify
relationships between texts. The intermediate MNLI stage can adapt
representations for inference between sentence pairs before the smaller
RTE stage.

The two label spaces still differ: MNLI uses three classes, while RTE
uses two. Transferring the encoder does not require reusing a
three-output classifier as a two-output classifier. Start from the
intermediate encoder checkpoint, initialize the target head with the
correct output shape, and create the target-stage optimizer for those
parameters. The tokenizer and input representation must remain compatible.

The next block starts from a supplied toy intermediate checkpoint. It
checks this transfer contract and one target update, not learning MNLI.

<details id="peft-code-transfer">
<summary>Code: replacing the classifier during transfer</summary>

```python
intermediate_encoder = nn.Linear(4, 4, bias=False).to(dtype)
with torch.no_grad():
    intermediate_encoder.weight.copy_(torch.eye(4, dtype=dtype))
intermediate_state = {
    key: value.detach().clone()
    for key, value in intermediate_encoder.state_dict().items()
}
intermediate_head = nn.Linear(4, 3).to(dtype)
target_encoder = nn.Linear(4, 4, bias=False).to(dtype)
target_encoder.load_state_dict(intermediate_state)
target_head = nn.Linear(4, 2).to(dtype)
torch.testing.assert_close(target_encoder.weight, intermediate_encoder.weight)
assert intermediate_head.weight.shape == (3, 4)
assert target_head.weight.shape == (2, 4)

target_optimizer = torch.optim.SGD(
    [*target_encoder.parameters(), *target_head.parameters()], lr=.1
)
target_input = torch.tensor([[1., -1., .5, 0.]], dtype=dtype)
target_label = torch.tensor([1])
target_logits = target_head(torch.tanh(target_encoder(target_input)))
F.cross_entropy(target_logits, target_label).backward()
assert target_encoder.weight.grad.norm().item() > 0
target_optimizer.step()
assert not torch.equal(target_encoder.weight, intermediate_state["weight"])
torch.testing.assert_close(intermediate_encoder.weight, intermediate_state["weight"])
```

</details>

In [Zhang et al., §7 and Table 2](https://arxiv.org/html/2006.05987v3),
intermediate-task training uses additional annotated MNLI data. Its cost
and supervision are therefore different from an optimizer correction on
the target dataset alone. Transfer can help when tasks share useful
representations, but related names do not guarantee positive transfer.

The following values are the authors' **reported test means and standard
deviations**, not new Bookvar measurements. They come from Table 2:
BERT-Large, bias-corrected Adam, 20 seeds, and three target-task epochs
except for the Longer setting.

| Setup | RTE accuracy, % | MRPC F1 | CoLA MCC |
|---|---:|---:|---:|
| Standard | 69.5 ± 2.5 | 90.8 ± 1.3 | 63.0 ± 1.5 |
| Re-init | 72.6 ± 1.6 | 91.4 ± 0.8 | 64.2 ± 1.6 |
| Longer | 72.3 ± 1.9 | 91.0 ± 1.3 | 62.4 ± 1.7 |
| MNLI transfer | 81.8 ± 1.7 | 91.8 ± 1.0 | 63.9 ± 1.8 |

The CoLA Re-init entry differs between Tables 1 and 2 of the source
version; the row above follows Table 2 rather than silently mixing them.
Longer training also does not improve every entry. These task-specific
results do not imply that LLRD or Mixout never help another model or task.

<span id="report-failed-runs-and-keep-selection-separate-from-testing"></span>

### Failed runs and model selection

Suppose four seeds score 0.50, 0.51, 0.70 and 0.72 on a development set.
Their mean is 0.6075. Removing the first two as “degenerate” changes the
reported mean to 0.71 and hides a 50% failure rate. It describes a
conditional subset of the runs, not the reliability of the full procedure.

<details id="peft-code-seeds">
<summary>Code: including failed seeds</summary>

```python
from statistics import mean, stdev

seed_scores = [.50, .51, .70, .72]
retained_scores = [score for score in seed_scores if score >= .55]
assert math.isclose(mean(seed_scores), .6075)
assert math.isclose(mean(retained_scores), .71)
assert sum(score < .55 for score in seed_scores) / len(seed_scores) == .5
assert stdev(retained_scores) < stdev(seed_scores)
```

</details>

A validation-based early-stopping or run-allocation rule can be a legitimate
algorithm. State that rule, its total compute budget and the fate of the
terminated runs. Do not report only the surviving runs as if they were all
the independent restarts. A low standard deviation is also insufficient
if every seed predicts the majority class.

![[00 Учебник/Assets/Figures/curated/small-data-finetuning-2026-10-29/mosbach-stability.png|Original development-score distributions across 25 fine-tuning seeds]]

*Marius Mosbach, Maksym Andriushchenko and Dietrich Klakow,
[Figure 1, original repository raster](https://github.com/uds-lsv/bert-stable-fine-tuning/blob/3cf27e8667aee9d1c822d747c63ce331b231f283/images/fig1.png),
Apache-2.0. Complete original figure and caption retained. RTE, MRPC and
CoLA use different task metrics; the three axes must not be pooled into
one accuracy measure.*

The cross marks and outliers show individual runs, while the boxes,
means and medians summarize distributions. The proposed baseline narrows
the distributions without the reader having to discard bad seeds.
This study reports development-set results on its own protocol. Its
numbers should not be directly ranked against Zhang et al.'s held-out
test halves: the data splits, selection rules and experiment budgets differ.

Keep the final test set outside the choice of layer-reset count, learning
rate, update budget and random seed. Report those choices, the number of
trials, aggregate scores and failure criteria together. Neither a
small-data BERT experiment nor a PEFT parameter count establishes a
universal ranking against instruction-tuned LLMs or in-context learning.

<a id="retaining-capabilities"></a>
## Retaining earlier capabilities during adaptation

A model adapted to invoice extraction can improve on invoices while becoming
worse at answering ordinary questions. The new training loss evaluates invoice
targets; it contains no term measuring those earlier answers. A small learning
rate can reduce the size of each update, but it does not tell the optimizer which
earlier behaviour to preserve. Continued evaluation on fixed old tasks is needed
to observe forgetting.

There are two different ways to add such a constraint. Replay evaluates the
current model on examples from earlier tasks. Parameter regularization instead
penalizes movement away from a saved checkpoint. These methods require different
information and need not preserve the same behaviour. They can also be combined.

<a id="fine-tuning-replay"></a>
<span id="replay-adds-old-examples-to-the-objective"></span>

### Replay

Write $\mathcal L_{\mathrm{new}}$ and $\mathcal L_{\mathrm{old}}$ as mean losses
on new and replay data. With replay probability $0\leq\rho\leq1$,
the objective is

$$
\mathcal L_{\mathrm{mix}}(\theta)
=(1-\rho)\mathcal L_{\mathrm{new}}(\theta)
+\rho\mathcal L_{\mathrm{old}}(\theta).
$$

Sampling an old example with probability $\rho$ and a new one otherwise gives an
unbiased gradient estimate for this objective, provided each source is sampled
according to the distribution defining its mean. Alternatively, form both losses
in each step and weight them explicitly. The old examples contribute gradients
that are missing from new-task-only training.

A scalar example exposes a conflict that replay cannot eliminate. Suppose an
old task wants $\theta=1$, while the new task wants $\theta=-1$:

$$
\mathcal L_{\mathrm{old}}=\tfrac12(\theta-1)^2,\qquad
\mathcal L_{\mathrm{new}}=\tfrac12(\theta+1)^2.
$$

The mixture gradient is $\theta+1-2\rho$, so its minimum is
$\theta^*=2\rho-1$. New-only training gives $-1$, with old loss 2.
A 25% replay weight gives $-0.5$, with old loss 1.125 and new loss 0.125.
Equal weights give 0, with loss 0.5 on each task. The compromise follows from
the chosen weights; no value of this one parameter solves both targets exactly.

<details id="peft-code-replay">
<summary>Code: the replay compromise</summary>

```python
def replay_losses(theta):
    return .5 * (theta - 1)**2, .5 * (theta + 1)**2

for replay_weight, expected in [(0., -1.), (.25, -.5), (.5, 0.)]:
    replay_solution = torch.tensor(2 * replay_weight - 1, dtype=dtype,
                                   requires_grad=True)
    old_loss, new_loss = replay_losses(replay_solution)
    mixture = (1 - replay_weight) * new_loss + replay_weight * old_loss
    mixture.backward()
    assert abs(replay_solution.grad.item()) < 1e-12
    assert math.isclose(replay_solution.item(), expected)
old_at_quarter, new_at_quarter = replay_losses(-.5)
assert math.isclose(old_at_quarter, 1.125)
assert math.isclose(new_at_quarter, .125)
```

</details>

In language-model training, $\rho$ must specify its unit. Selecting 25% of
documents from an old corpus does not imply that 25% of loss-bearing tokens
come from it. SFT masks can further separate input tokens from supervised
tokens. The chapter on
[[00 Учебник/11 Pre-training и Scaling/53 Синтетические данные и учебные программы#curricula-replay|data mixtures and replay]]
works through token shares, repeated use of small corpora and fixed-budget
comparisons.

Replay requires usable old data, or a retained sample of it, and adds old-task
computation during adaptation. A buffer can miss a rare language or skill;
uniform sampling from the buffer cannot recover examples that were never kept.
An available pretraining corpus is also only a proxy for the particular
capabilities one wants to retain. Measure those capabilities on held-out data.
The interleaving argument is described in
[Kirkpatrick et al., Introduction](https://arxiv.org/html/1612.00796v2#S1);
the scalar calculation above is an added example.

<a id="l2-sp"></a>
<span id="l2-sp-uses-the-pretrained-checkpoint-as-the-reference"></span>

### L2-SP regularization

Ordinary zero-centred L2 regularization adds
$\tfrac{\lambda}{2}\|\theta\|^2$, whose gradient is $\lambda\theta$.
It pulls weights toward zero. Starting optimization at a pretrained checkpoint
does not change the centre of that penalty. L2-SP instead saves the starting
weights $\theta_A$ and uses

$$
\mathcal L(\theta)=\mathcal L_{\mathrm{new}}(\theta)
+\frac{\lambda}{2}\|\theta-\theta_A\|^2.
$$

Its penalty gradient is $\lambda(\theta-\theta_A)$, which is zero at the saved
checkpoint and grows as the parameters move away. Li, Grandvalet and Davoine
explain the reference choice in their original introduction:

> We thus advocate for a coherent parameter regularization approach, where the
> pre-trained model is both used as the starting point of the optimization process
> and as the reference in the penalty that encodes an explicit inductive bias.

— [Explicit Inductive Bias for Transfer Learning with Convolutional Networks,
§1](https://proceedings.mlr.press/v80/li18a/li18a.pdf#page=1),
ICML 2018, PMLR 80:2825–2834, Copyright 2018 the authors, CC BY 4.0.
The original passage is retained in English.

For a new quadratic loss $\tfrac12\|\theta-b\|^2$, setting the total gradient to
zero gives $\theta^*=(b+\lambda\theta_A)/(1+\lambda)$.
With $\theta_A=(2,2)$, $b=(0,0)$ and $\lambda=1$, L2-SP gives $(1,1)$.
A zero-centred penalty gives $(0,0)$ in this example: both its terms prefer zero.

<details id="peft-code-l2sp">
<summary>Code: a penalty around the starting weights</summary>

```python
anchor = torch.tensor([2., 2.], dtype=dtype).detach().clone()
new_target = torch.zeros(2, dtype=dtype)
strength = 1.
sp_solution = ((new_target + strength * anchor) / (1 + strength)).requires_grad_()
sp_objective = (.5 * (sp_solution - new_target).square().sum()
                + strength / 2 * (sp_solution - anchor).square().sum())
sp_objective.backward()
torch.testing.assert_close(sp_solution, torch.ones(2, dtype=dtype))
torch.testing.assert_close(sp_solution.grad, torch.zeros(2, dtype=dtype))
assert not anchor.requires_grad
zero_centred_solution = new_target / (1 + strength)
torch.testing.assert_close(zero_centred_solution, torch.zeros(2, dtype=dtype))
```

</details>

The anchor must be a detached snapshot, not a view of live parameters that changes
with every optimizer step. When fine-tuning replaces a classification head, the
new head has no corresponding pretrained tensor. Li et al.'s Eq. 3 applies the
checkpoint-centred penalty to shared parameters and a separate zero-centred
penalty to new parameters. Do not align a two-label head with an unrelated
three-label head merely because both are called `classifier`.

A squared-loss penalty and decoupled AdamW weight decay are different update
rules with adaptive optimizers. Setting AdamW's `weight_decay` does not implement
L2-SP. An explicit checkpoint-centred loss must use the intended parameter set
and normalization.

<a id="elastic-weight-consolidation"></a>
<span id="ewc-assigns-different-penalties-to-different-parameters"></span>

### EWC penalties

L2-SP treats equal displacements in every parameter equally. Elastic weight
consolidation stores an importance coefficient $F_i$ for each parameter after
task A, then minimizes

$$
\mathcal L_{\mathrm{EWC}}(\theta)
=\mathcal L_{\mathrm{new}}(\theta)
+\frac{\lambda}{2}\sum_iF_i(\theta_i-\theta_{A,i})^2.
$$

The penalty gradient for coordinate $i$ is
$\lambda F_i(\theta_i-\theta_{A,i})$. A large $F_i$ permits less movement under
the same opposing new-task gradient. A zero coefficient imposes no constraint
on that coordinate. The coefficient does not freeze a weight absolutely.

For the preceding two-parameter example, assign $F=(9,0.1)$. These are supplied
toy coefficients, not estimates from a trained network. The stationary point is

$$
\theta_i^*=\frac{b_i+\lambda F_i\theta_{A,i}}{1+\lambda F_i}.
$$

At $\lambda=1$, it is $(1.8,\,2/11)\approx(1.8,0.1818)$.
The first coordinate stays near 2; the second moves nearly to the new target 0.
Using the same coefficients to define the protected old quadratic
$\tfrac12\sum_iF_i(\theta_i-\theta_{A,i})^2$ gives these exact trade-offs:

| Solution | New quadratic loss | Protected old quadratic |
|---|---:|---:|
| New-only, $(0,0)$ | 0 | 18.2 |
| Isotropic L2-SP, $(1,1)$ | 1 | 4.55 |
| EWC, $(1.8,2/11)$ | about 1.637 | about 0.345 |

These numbers compare objectives in a constructed example. They are not accuracy
scores, and the protected quadratic is not an evaluation of a real old task.

<details id="peft-code-ewc">
<summary>Code: a coordinate-weighted penalty</summary>

```python
importance = torch.tensor([9., .1], dtype=dtype)
ewc_solution = ((new_target + strength * importance * anchor)
                / (1 + strength * importance)).requires_grad_()
new_quadratic = .5 * (ewc_solution - new_target).square().sum()
old_quadratic = .5 * (importance * (ewc_solution - anchor).square()).sum()
(new_quadratic + strength * old_quadratic).backward()
torch.testing.assert_close(ewc_solution,
                          torch.tensor([1.8, 2 / 11], dtype=dtype))
torch.testing.assert_close(ewc_solution.grad, torch.zeros(2, dtype=dtype),
                          rtol=0., atol=1e-12)
assert math.isclose(new_quadratic.item(), 1.6365289256198348)
assert math.isclose(old_quadratic.item(), .3452892561983471)

# A frozen base can still give a different function when its adapter is enabled.
frozen_base = torch.tensor([[1.]], dtype=dtype)
adapter_a = torch.tensor([[1.]], dtype=dtype)
adapter_b = torch.tensor([[-2.]], dtype=dtype)
old_input = torch.tensor([1.], dtype=dtype)
assert (frozen_base @ old_input).item() == 1.
assert ((frozen_base + adapter_b @ adapter_a) @ old_input).item() == -1.
```

</details>

EWC's original motivation approximates the parameter posterior after task A by
a Gaussian around $\theta_A$, then uses that approximation when learning B.
The diagonal Fisher supplies the per-coordinate precision approximation.
This is a local, diagonal approximation; it discards interactions between
parameters and does not prove that a shared good solution exists.
See [Kirkpatrick et al., §2 and Eq. 3](https://arxiv.org/html/1612.00796v2#S2).

The last calculation also shows why frozen LoRA base weights do not guarantee
retention. The base weight is still 1, but the enabled update changes its output
on the old input from 1 to -1. A penalty on the frozen base alone is identically
zero. To constrain an adapter-enabled model, choose a penalty on its actual
trainable parameterization or effective update, or evaluate old behaviour through
replay. These choices are not interchangeable with applying a Fisher estimated
for the original base directly to LoRA factor coordinates.

<a id="fisher-for-ewc"></a>
<span id="estimating-the-fisher-square-before-averaging"></span>

### Estimating the Fisher diagonal

For a conditional model, one diagonal Fisher convention is

$$
F_i=\mathbb E_{x\sim D_A}\,
\mathbb E_{y\sim p_{\theta_A}(\cdot\mid x)}
\left[
\left(\frac{\partial\log p_\theta(y\mid x)}{\partial\theta_i}
\bigg|_{\theta=\theta_A}\right)^2
\right].
$$

Average over old inputs and, for each input, over labels drawn from the saved
model's predictive distribution. For a small classifier the inner expectation
can be summed over all classes; for large output spaces it is often estimated
by sampling. Li et al. write the categorical sum explicitly below their Eq. 4.
The saved model is used for this estimation before new-task updates.

Using observed dataset labels instead of model-distributed labels gives the
quantity often called the **empirical Fisher** in machine-learning papers.
It is not generally the same matrix. Nor is either quantity obtained by
squaring the gradient of a mean batch loss.

For a scalar logistic model with input 1 and parameter 0, the probability of
class 1 is 0.5. Two old examples with labels 1 and 0 have loss gradients
$-0.5$ and $+0.5$. Their mean gradient is zero, but the mean of their squared
gradients is 0.25. Averaging first would incorrectly assign zero importance.

<details id="peft-code-fisher">
<summary>Code: per-example squared gradients</summary>

```python
logistic_weight = torch.tensor(0., dtype=dtype, requires_grad=True)
old_labels = torch.tensor([1., 0.], dtype=dtype)
per_example_loss = F.binary_cross_entropy_with_logits(
    logistic_weight.expand_as(old_labels), old_labels, reduction="none"
)
per_example_grad = torch.stack([
    torch.autograd.grad(loss, logistic_weight, retain_graph=True)[0]
    for loss in per_example_loss
])
empirical_diagonal = per_example_grad.square().mean()
squared_mean_gradient = per_example_grad.mean().square()
assert math.isclose(empirical_diagonal.item(), .25)
assert squared_mean_gradient.item() == 0.

# At p=.9, observed positive labels and model-distributed labels differ.
p = .9
observed_label_square = (p - 1)**2
model_fisher = p * (1 - p)
assert math.isclose(observed_label_square, .01)
assert math.isclose(model_fisher, .09)
```

</details>

The first example happens to have balanced labels matching the model
probabilities. In the second, replacing the model expectation with observed
positive labels changes the result by a factor of nine. Neither a converged
mean gradient nor confident training predictions justify calling all weights
unimportant. [Kunstner, Balles and Hennig, §§1 and 4](https://arxiv.org/html/1905.12558v3)
derive the distinction and explain why the empirical Fisher is not generally
a curvature approximation.

A sum over examples and a mean differ by the sample count. Changing that
convention, the new-loss reduction or the number of estimated tokens changes
the effective penalty strength. Report them with $\lambda$. The diagonal alone
needs one stored scalar per protected parameter, in addition to the checkpoint
anchor. EWC avoids replaying old data during each new-task step, but it still
needs data or an equivalent source distribution to estimate importance.

For more than two tasks, also specify how the anchor and precision are updated.
Keeping a separate penalty at every historical anchor and updating one
consolidated posterior approximation are different algorithms.
[Huszár's derivation](https://arxiv.org/html/1712.03847v1) shows a double-counting
problem with treating the former as exact sequential Bayesian inference.
The two-task equation above does not silently define a complete multi-task
implementation.

<span id="what-the-original-transfer-experiment-measures"></span>

### L2-SP transfer results

![[00 Учебник/Assets/Figures/curated/retention-2026-10-29/l2sp-target-comparison.png|Original L2-SP comparison on four target datasets]]

*Xuhong Li, Yves Grandvalet and Franck Davoine,
[Figure 2, ICML 2018, PDF p. 6](https://proceedings.mlr.press/v80/li18a/li18a.pdf#page=6),
CC BY 4.0. All four panels and the original caption are retained without
redrawing or relabelling. Source tasks are identified by both the caption and
the explanation below, not by colour alone.*

The panels report **target-task accuracy** for ResNet transfer experiments.
Each dot is a run; ImageNet and Places 365 are different source datasets.
The scene-classification target MIT Indoor 67 transfers better from Places,
while the object targets benefit more from ImageNet. The L2-SP and
L2-SP-Fisher columns have similar target accuracy in these experiments.
The L1 and group-lasso columns are other penalties tested in the source;
they are not variants of LoRA.

![[00 Учебник/Assets/Figures/curated/retention-2026-10-29/l2sp-retention.png|Original comparison of accuracy lost on source tasks after fine-tuning]]

*Li, Grandvalet and Davoine,
[Table 3, PDF p. 6](https://proceedings.mlr.press/v80/li18a/li18a.pdf#page=6),
CC BY 4.0. Complete original table and caption, cropped from the same page.
Its negative entries describe changes on the old source task, not target
accuracies and not relative percentage reductions.*

For Stanford Dogs transfer, the original ImageNet accuracy is 76.7%.
The reported drops of 14.1, 4.7 and 4.2 percentage points give retained
accuracies of 62.6%, 72.0% and 72.5% for L2, L2-SP and L2-SP-Fisher.
The old classification head is restored for this measurement; no old-task
retraining is performed. Even the Fisher-weighted version loses accuracy.
Thus similar new-task scores can coexist with different retention, and the
penalty does not guarantee preservation. These are the authors' reported
vision experiments, not measurements on LLMs or a claim that EWC should
replace replay in every adaptation.

The new task and the retained tasks therefore need separate held-out metrics.
Tune the mixture or penalty against the stated objective and report both.
Keep unchanged evaluation data and the original base checkpoint so that a
later adaptation can be compared with the same reference.

<span id="training-and-evaluating-an-adaptation"></span>

## Comparing adaptations

Keep the task and data fixed when comparing parameterizations. For a
small classification dataset, compare a fixed-representation head,
selected-layer tuning and full fine-tuning under an explicit model-
selection budget. For instruction tuning, preserve the chat template,
target mask and generation evaluation while varying the trainable modules.
An adapter does not repair incorrect targets or test contamination.

Record the actual names and shapes of trainable tensors before training.
A fused QKV projection, rectangular GQA projections and a separate
classifier cannot be treated as interchangeable module names. Save the
base revision alongside the adapter configuration. If vocabulary rows
or a new output head are trained, those changes must also be saved.

Learning rate, rank, scaling, dropout, training duration and data mixture
are experimental choices. Count **optimizer updates** as well as epochs:
2,400 examples with batch 32 and three epochs give about 225 updates
before any special sampling or dropped final batch. A claim about
“three epochs” without the dataset, batching and validation procedure
does not specify an equivalent optimization budget.

Inspect both the downstream task and retained capabilities on held-out
data. Restoring the untouched base by disabling an adapter restores its
weights, but does not prove that the adapter-enabled model preserved
every earlier ability. Report failed or degenerate runs rather than
silently dropping them from the mean. A small trainable parameter count
does not establish generalization, freedom from forgetting or superiority
over full fine-tuning.

<span id="sources-and-reproduced-material"></span>

## References

- Houlsby et al., [Parameter-Efficient Transfer Learning for NLP](https://proceedings.mlr.press/v97/houlsby19a.html),
  Introduction and §2.1. The quoted bottleneck description and original
  Figure 2 retain the source's English; the numerical example is added.
- Hu et al., [LoRA](https://arxiv.org/html/2106.09685v2), §§4 and 7.
  Consult the original task-specific experiments when comparing ranks
  or selected projections; their results are not universal recipes.
- Dettmers et al., [QLoRA](https://arxiv.org/html/2305.14314v1), §§2–4,
  CC BY 4.0. Figure 1 and the storage/computation paragraph are reproduced;
  the memory examples specify their own assumptions.
- Lester, Al-Rfou and Constant, [Prompt Tuning](https://aclanthology.org/2021.emnlp-main.243/),
  §§2–3, and Li and Liang, [Prefix-Tuning](https://aclanthology.org/2021.acl-long.353/),
  §§3–4.3. The original figures are reproduced through the licensed HF
  documentation-image archive.
- Hugging Face Team, [PEFT documentation at fee923eb](https://github.com/huggingface/peft/tree/fee923eb6e0382236478e4259bc69f4aeaab6813),
  Copyright 2023–present The HuggingFace Team, Apache-2.0.
  Exact excerpts from *LoRA*, *Prompt tuning* and *Prefix tuning* are
  marked above. Surrounding derivations, examples and navigation were
  added for this chapter; this is not an unchanged copy of the manual.
  [Full license](/licenses/peft-fee923eb-Apache-2.0.txt).
- The [Hugging Face documentation images](https://huggingface.co/datasets/huggingface/documentation-images/blob/861b6c920fe27492d35256c39f266592a518d50c/README.md)
  have a separate CC BY-NC-SA 4.0 license. Their original pixels and
  animation are preserved, with attribution beside each figure.

- Zhang et al., [Revisiting Few-sample BERT Fine-tuning](https://arxiv.org/html/2006.05987v3),
  §§3–7; Mosbach et al., [On the Stability of Fine-tuning BERT](https://arxiv.org/html/2006.04884v3),
  §§4–6. The small-data section distinguishes their reported experiments
  from added scalar, gradient, transfer-contract and selection examples.
  The original ASAPP README excerpt, illustration and adapted reset code
  retain attribution to Copyright 2020 ASAPP Inc., MIT; the upstream
  Transformers code remains Apache-2.0. [Full repository licenses](/licenses/asapp-finetuning-0aa4f4e-LICENSE.txt).
  The Mosbach repository figure is copied unchanged under Apache-2.0;
  [full license](/licenses/mosbach-finetuning-3cf27e8-Apache-2.0.txt).

- Li, Grandvalet and Davoine, [Explicit Inductive Bias for Transfer Learning
  with Convolutional Networks](https://proceedings.mlr.press/v80/li18a.html),
  §§1, 3 and 4.2–4.3.2, Figure 2 and Table 3. Original English passage,
  plots and table reproduced under CC BY 4.0. The replay, quadratic and
  logistic examples are added, not reported training runs.
- Kirkpatrick et al., [Overcoming catastrophic forgetting](https://arxiv.org/html/1612.00796v2),
  Introduction and §2; Kunstner, Balles and Hennig,
  [Limitations of the Empirical Fisher Approximation](https://arxiv.org/html/1905.12558v3),
  §§1 and 4; Huszár, [On Quadratic Penalties in EWC](https://arxiv.org/html/1712.03847v1).
  These sources support the mechanism and its limits; their full text and
  figures are linked rather than reproduced.

Previous: [[00 Учебник/12 Post-training и Alignment/01 SFT и instruction data|SFT: learning from demonstrations]].

Next: [[00 Учебник/12 Post-training и Alignment/02 Preference data|Preference data]].
A pair of responses supplies a different training signal regardless of
whether all weights or only an adapter will change.
