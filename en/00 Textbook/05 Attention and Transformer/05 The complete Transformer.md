---
title: Building the Transformer — from the original block to BERT, GPT, and LLaMA
type: textbook-chapter
status: canonical
locale: en
translation_status: current
canonical_revision: "1ffb968f9bc3b562b3065f1b059c12e9e9c0b41f78176f2e6d6e5c303f6e52b5"
translation_reviewed: "2026-10-29"
translation_of: "00 Учебник/05 Attention и Transformer/03 Полный Transformer.md"
last_updated: 2026-10-30
source_unit_id:
  - lecture-01-section-175-model-architecture
  - lecture-03-architecture-framing
  - lecture-03-recap
  - assignment-01-task-transformer-block
  - assignment-01-task-transformer-lm
primary_sources:
  - https://arxiv.org/abs/1706.03762
  - https://arxiv.org/abs/1810.04805
  - https://arxiv.org/abs/2302.13971
  - https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_03.pdf
  - https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf
previous: "[[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/04 Позиционная информация]]"
next: "[[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна]]"
---

# Building the Transformer: from the original block to BERT, GPT, and LLaMA

A translator receives `кот спит` and must produce `the cat sleeps`.
In this teaching example, each word is one token; an actual tokenizer may
split the words differently. The input and output need not have the same
number of tokens. The encoder builds a representation of each source
position. The decoder predicts the translation using both these
representations and the beginning of the translation already available.

In the 2017 Transformer, both parts consist of attention and feed-forward
blocks with residual connections and normalization. Unlike a recurrent
translator, the encoder processes all known input positions in parallel.
During training, the decoder positions are also processed in parallel;
during generation, the model must first produce the next token.

BERT, GPT, and LLaMA use different inputs, connections, and training
objectives; Section 8 compares them with the translator. Q/K/V computation
and shapes are assumed from the
[[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V|self-attention chapter]];
causal masking and head splitting are covered in the
[[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/Masking, multi-head и формы тензоров|chapter on attention masks and heads]].

<a id="architecture-lineage"></a>

## 1. The central decomposition

Suppose that after the first attention layer, the vector for `кот` already
depends on `спит`. This vector contains a mixture of features extracted
from the phrase. A feed-forward network, or FFN, transforms its coordinates:
it expands the representation, applies a nonlinearity, and returns to the
original width. It applies the same network with the same weights to the
state of `спит`. A different Transformer layer has its own FFN.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/Transformer_encoder.png]]

*The two encoder sublayers. Self-attention connects positions; the FFN
processes each resulting vector independently. This simplified diagram does
not show residual connections or normalization.
Source: Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

Given input `[B,T,d]`, both sublayers return `[B,T,d]`. Their internal
computations differ: the attention-weight matrix has two position axes,
while the FFN's hidden representation has one position axis and an
expanded feature axis. The FFN does not directly read neighboring rows.
It receives a vector that already contains context, and its result can
affect neighbors through the next layer's attention.

Softmax makes attention nonlinear. The claim that “we need the FFN because
attention is linear” is therefore incorrect. The FFN adds a learned
nonlinear feature transformation at each position. Without the activation
between its two linear layers, the FFN could be collapsed into one affine
layer, as the example below shows.

## 2. The residual stream

For a row $x$, attention or the FFN computes an update $F(x)$ of the same
width. A residual connection adds it to the input. For example,
$x=(1,2,3,4)$ and $F(x)=(0,1,0,-1)$ give $(1,3,3,3)$. A sublayer can change
some features while preserving others through direct addition. It does
not have to reproduce the entire input vector.

In the original paper, LayerNorm follows the addition. Normalization acts
on the coordinates of **each row separately**, without mixing tokens or
examples:

$$
\begin{aligned}
\mu&=\frac1d\sum_{r=1}^d x_r,\\
\sigma^2&=\frac1d\sum_{r=1}^d(x_r-\mu)^2,\\
\operatorname{LN}(x)&=
\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta.
\end{aligned}
$$

With $\gamma=(1,1,1,1)$, $\beta=0$, and negligibly small $\epsilon$, the
vector $(1,2,3,4)$ has mean $2.5$ and variance $1.25$, and normalizes to
approximately $(-1.3416,-0.4472,0.4472,1.3416)$. The parameters $\gamma,\beta$
are learned: normalization does not permanently fix the features' scale
and offset.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/transformer_resideual_layer_norm_2.png]]

*A post-norm encoder for two positions. The dashed path carries the input
to the addition with the sublayer output; LayerNorm follows the sum.
The green and pink vectors inside Add & Normalize are the two summands,
not two separate normalizations.
Source: Jay Alammar, [The Illustrated Transformer, The Residuals](https://jalammar.github.io/illustrated-transformer/#the-residuals),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

The order of operations distinguishes post-norm from pre-norm:

$$
\begin{aligned}
\text{post-norm:}\quad&y=\operatorname{LN}(x+F(x)),\\
\text{pre-norm:}\quad&y=x+F(\operatorname{LN}(x)).
\end{aligned}
$$

These are different functions even with identical weights. For $F(x)=2x$,
post-norm approximately returns $\operatorname{LN}(x)$, because
normalization removes the common positive scale factor $3$. Pre-norm
returns $x+2\operatorname{LN}(x)$: approximately
$(-1.6833,1.1056,3.8944,6.6833)$ for the same vector. The direct residual
path preserves the original scale.

In backward, $x+F(\operatorname{LN}(x))$ has an identity-derivative term
with respect to $x$. In post-norm, the entire sum passes through the
normalization derivative. This explains the difference between gradient
paths, but does not prove that any pre-norm stack will train stably:
terms can cancel, and update scale depends on initialization and depth.
Pre-norm models usually add a final normalization before the output projection.

[Vaswani et al., §3.1](https://arxiv.org/html/1706.03762v7#S3.SS1) uses
post-norm. In [The Annotated Transformer, Encoder and Decoder Stacks](https://nlp.seas.harvard.edu/annotated-transformer/#encoder-and-decoder-stacks),
the explanatory text reproduces that formula, but `SublayerConnection`
computes pre-norm; the authors note this in its docstring. The code has
another difference: `std` with its default correction and $\epsilon$
outside the square root do not match the LayerNorm formula above.
The implementation below therefore uses `nn.LayerNorm`; the paper's
diagram and the tutorial's code are not treated as identical.

During training, the original Transformer applies dropout to the sublayer
output **before residual addition**. The sum of token and position
embeddings also passes through dropout. Dropout is disabled in `eval()`.
Normalization and gradient propagation are discussed further in the
[[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/05 Инициализация, нормализация и residual connections|chapter on initialization and normalization]].

<!-- BEGIN REVIEW: Transformer dropout 2026-10-29 -->

<a id="transformer-dropout"></a>

### Dropout on attention weights, branch outputs and embeddings

An attention mask and attention dropout act at different steps. The causal
or padding mask removes forbidden scores before softmax. Dropout then
randomly zeros some of the resulting weights during training and scales
the survivors by $1/q$, where $q$ is the keep probability:

$$
A=\operatorname{softmax}(S_{\mathrm{masked}}),\qquad
\widetilde A=\frac{M\odot A}{q},\qquad
O=\widetilde A V.
$$

Each row of $A$ sums to one. A particular row of $\widetilde A$ need not.
For $A=(0.8,0.2,0)$ and $q=0.5$, enumerate the two allowed positions:

| Retained positions | Dropped weights | Row sum | Output for $V=(2,6,99)^T$ |
|---|---|---:|---:|
| Neither | $(0,0,0)$ | $0$ | $0$ |
| Second | $(0,0.4,0)$ | $0.4$ | $2.4$ |
| First | $(1.6,0,0)$ | $1.6$ | $3.2$ |
| Both | $(1.6,0.4,0)$ | $2$ | $5.6$ |

The masks are equiprobable. Their mean weights are $(0.8,0.2,0)$, and
the mean output is $2.8=0.8\cdot2+0.2\cdot6$.
The third value, $99$, never contributes: its attention weight was zero
before dropout. A surviving weight can exceed one, and there is no
second softmax or division by the row sum.

```python
from itertools import product
import torch

ad_weights = torch.tensor([.8, .2, 0.], dtype=torch.float64)
ad_values = torch.tensor([2., 6., 99.], dtype=ad_weights.dtype)
ad_masks = torch.tensor([(*m, 0.) for m in product([0., 1.], repeat=2)],
                       dtype=ad_weights.dtype)
ad_dropped = ad_masks * ad_weights / .5
ad_outputs = ad_dropped @ ad_values
torch.testing.assert_close(ad_dropped.sum(1),
                           torch.tensor([0., .4, 1.6, 2.], dtype=ad_weights.dtype))
torch.testing.assert_close(ad_outputs,
                           torch.tensor([0., 2.4, 3.2, 5.6], dtype=ad_weights.dtype))
torch.testing.assert_close(ad_dropped.mean(0), ad_weights)
torch.testing.assert_close(ad_outputs.mean(), ad_weights @ ad_values)
assert torch.count_nonzero(ad_dropped[:, 2]) == 0
assert torch.count_nonzero(ad_dropped[0]) == 0
print(ad_dropped, ad_outputs, ad_outputs.mean())
```

A fully dropped row gives a zero value mixture. This is a valid multiplication,
unlike softmax of a row containing only $-\infty$, which has no defined
normalization. The output projection's bias or the residual input can still
make the block output nonzero. The equality of means above holds for fixed
$A,V$; nonlinear layers and the loss do not in general commute with averaging
over masks. Dropout therefore does not guarantee a more uniform attention
distribution or an unchanged mean training loss.

[The Annotated Transformer, Attention](https://nlp.seas.harvard.edu/annotated-transformer/#attention)
implements this order explicitly: softmax, dropout, multiplication by values.
The `MHA` class below does the same and passes `self.training` to
`F.dropout`. Calling `eval()` disables that functional dropout because
the flag is passed explicitly, not because the function knows its parent.

Residual dropout has a different shape and location: it masks coordinates
of the sublayer output after the attention output projection or the FFN,
before adding the unmodified skip input:

$$
y=\operatorname{LN}\!\left(x+\frac{m\odot F(x)}q\right)
\quad\text{(original post-norm block).}
$$

Here $m$ has the branch output's shape $[B,T,d]$. It is not an attention
mask over $[B,h,T_q,T_k]$. A stochastic-depth gate would instead be shared
across the whole branch for a sample or mini-batch, as explained in
[[00 Учебник/01 Основы нейронных сетей/06 Dropout и регуляризация#stochastic-depth|the regularization chapter]].

Embedding dropout is applied after the scaled token embedding and position
vector have been added:
$\widetilde x=m\odot(\sqrt d\,E+\mathrm{PE})/q$.
If one coordinate of $\sqrt d\,E$ is $2$ and the corresponding position
coordinate is $1$, retaining their sum at $q=0.5$ gives $6$.
Dropping only the token coordinate before adding the unchanged position
would give $5$; that is a different operation.
[Vaswani et al., §5.4](https://arxiv.org/html/1706.03762v7#S5.SS4)
specify dropout before residual addition and on the embedding–position sum.
Their $0.1$ probability is a setting of those experiments, not a universal
requirement for all later Transformer models.

<!-- END REVIEW: Transformer dropout 2026-10-29 -->

## 3. Feed-forward network

The original FFN consists of two affine transformations with ReLU between them:

$$
\begin{aligned}
u&=xW_1+b_1,& W_1&\in\mathbb R^{d\times d_{ff}},\\
a&=\max(0,u),\\
\operatorname{FFN}(x)&=aW_2+b_2,
& W_2&\in\mathbb R^{d_{ff}\times d}.
\end{aligned}
$$

Transformer Base uses $d=512$ and $d_{ff}=2048$. Expansion produces 2048
intermediate features; ReLU retains positive coordinates, and $W_2$
combines them into 512 output coordinates. Different tokens activate
different intermediate coordinates even though the matrices are shared.

Take $d=2$, $d_{ff}=3$, and row vectors:

$$
\begin{aligned}
W_1&=\begin{pmatrix}1&-1&0\\0&1&1\end{pmatrix},
\quad b_1=(0,1,0),\\
W_2&=\begin{pmatrix}1&2\\3&4\\5&6\end{pmatrix},
\quad b_2=0.
\end{aligned}
$$

For $x=(1,-2)$, the first step gives $u=(1,-2,-2)$. ReLU leaves $(1,0,0)$,
and the output is $(1,2)$. For $x'=(-1,2)$, the same FFN produces
$u'=(-1,4,2)$, then $(0,4,2)$, and finally $(22,28)$. The two positions
activate different combinations of intermediate features. Changing only
the second input row does not change the FFN's first output row.

Without ReLU, the expression would expand to
$x(W_1W_2)+(b_1W_2+b_2)$, an ordinary affine map. With ReLU, the effective
matrix depends on which coordinates of $u$ are positive. In the first
example, the derivative does not pass through the second and third
intermediate coordinates. The sum of the two outputs depends on the
active first coordinate as $1u_1+2u_1=3u_1$, and $u_1=x_1$. Its gradient
with respect to the input is therefore $(3,0)$. Gradients with respect to
$W_1$ and $W_2$ also pass through the first intermediate coordinate; the
input numbers themselves are not updated in this example.

```python
import torch
import torch.nn.functional as F

x = torch.tensor([[1., -2.], [-1., 2.]], requires_grad=True)
w1 = torch.tensor([[1., -1., 0.], [0., 1., 1.]])
b1 = torch.tensor([0., 1., 0.])
w2 = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
u = x @ w1 + b1
y = u.relu() @ w2
torch.testing.assert_close(y, torch.tensor([[1., 2.], [22., 28.]]))
y[0].sum().backward()
torch.testing.assert_close(x.grad, torch.tensor([[3., 0.], [0., 0.]]))
print(u.detach(), y.detach(), x.grad)
```

Ignoring biases, the FFN contains $2dd_{ff}$ parameters. At $d_{ff}=4d$,
that is $8d^2$, compared with $4d^2$ for ordinary MHA whose heads have
combined width $d$. In one encoder block, the FFN accounts for two thirds
of these projection weights. A decoder block also contains cross-attention,
so its FFN accounts for one half, not two thirds. Embedding tables are not
included in this count.

Gated FFNs change the mapping itself. In SwiGLU, one projection supplies
a multiplier for another:

$$
\begin{aligned}
g&=\operatorname{SiLU}(xW_g),\\
u&=xW_u,\\
\operatorname{SwiGLU}(x)&=(g\odot u)W_d.
\end{aligned}
$$

Here $\operatorname{SiLU}(z)=z\,\sigma(z)$. There are three matrices instead
of two. Matching the parameter count of a ReLU FFN of width $4d$ gives
$3dm\approx8d^2$, or $m\approx8d/3$, before implementation-specific
rounding. The SiLU multiplier is not bounded to $[0,1]$ and can be
negative: it is not the probability that a gate is open. This variant is
introduced here for comparison with LLaMA; the 2017 model uses ReLU.
Position-wise FFNs and weight sharing across positions are explained in
[Vaswani et al., §3.3](https://arxiv.org/html/1706.03762v7#S3.SS3) and
[D2L, Transformer](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html#positionwise-feed-forward-networks).

## 4. Positional information

The encoder and decoder begin with integer IDs. For a shared vocabulary
$V$, the table $E\in\mathbb R^{|V|\times d}$ stores a learned vector for
each token. Looking up rows for input `[B,T]` yields `[B,T,d]`. The
2017 Transformer adds a fixed sinusoidal vector to the scaled embedding:

$$
h_i^{(0)}=\operatorname{Dropout}\big(\sqrt d\,E[t_i]+PE_i\big).
$$

At $d=4$, embedding $(1,0,0,0)$ at position 0 becomes $(2,1,0,1)$ before
dropout: the scaled token is $(2,0,0,0)$, and the position-0 sinusoidal
encoding is $(0,1,0,1)$. Source and target sequences have their own
position numbering. PE is added **once at the bottom of each stack**,
not again after every block.

In full, unmasked self-attention without position features, permuting
input rows permutes output rows in the same way. A causal mask already
breaks this symmetry: positions have different prefixes. It is therefore
incorrect to say that *every* model without separate PE cannot distinguish
order at all. The symmetry check, learned positions, Shaw, T5, RoPE, and
ALiBi are covered in the
[[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/04 Позиционная информация|previous chapter]].

Early GPT and BERT also add learned absolute positions at the bottom.
LLaMA's RoPE instead rotates Q and K inside each attention layer.
Adding the original Transformer's sinusoidal sum to such a block would
introduce another positional mechanism and change the model.

## 5. The original encoder block

Let $H^{(0)}$ contain the source embeddings plus positions. At layer
$\ell$, the encoder first performs full multi-head self-attention, then an FFN:

$$
\begin{aligned}
H&=H^{(\ell-1)},\\
A&=\operatorname{MHA}_{\ell}(H,H,H),\\
U&=\operatorname{LN}_{\ell,1}(H+\operatorname{Dropout}(A)),\\
F&=\operatorname{FFN}_{\ell}(U),\\
H^{(\ell)}&=\operatorname{LN}_{\ell,2}(U+\operatorname{Dropout}(F)).
\end{aligned}
$$

MHA's arguments identify the sources of Q, K, and V before learned
projections. “Full” means that every real position may read every real
position in the source phrase. PAD keys are excluded by a mask. The
translator's encoder needs no causal mask: the entire source is already known.

After the first attention operation, the row for `кот` may depend on
`спит`. The following FFN transforms this contextual row. Attention in
the second layer receives the transformed representations of both
positions, not the original vocabulary embeddings. Depth thus combines
the outputs of successive sublayers.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/The_transformer_encoder_decoder_stack.png]]

*Encoder and decoder stacks. Position states, not token IDs, pass between
blocks. Arrows from the top encoder block lead to cross-attention in every
decoder block.
Source: Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

Transformer Base has six encoder blocks and six decoder blocks, $d=512$,
eight heads of width 64, and FFN width 2048. Blocks have the same structure
but **different parameters**. Repeatedly applying one `Block` object
would share weights across depth; an ordinary stack creates separate instances.

The result $M=H^{(N)}$ has shape `[B,T_src,d]`. Source length is preserved:
the encoder does not compress the phrase into a single vector like the
fixed context of early Seq2Seq. All rows of $M$ remain available to
decoder attention.

## 6. The original decoder block

The first target state is built from `<BOS>`. It does not yet contain
the selected token `the`. Masked self-attention processes the known
target prefix, then cross-attention reads memory $M$. The FFN transforms
each target row afterward.

For state $D$, one post-norm block makes three successive updates:

$$
\begin{aligned}
A&=\operatorname{MHA}_{self}(D,D,D),\\
U&=\operatorname{LN}_1(D+\operatorname{Dropout}(A)),\\
B&=\operatorname{MHA}_{cross}(U,M,M),\\
C&=\operatorname{LN}_2(U+\operatorname{Dropout}(B)),\\
F&=\operatorname{FFN}(C),\\
D'&=\operatorname{LN}_3(C+\operatorname{Dropout}(F)).
\end{aligned}
$$

Self-attention uses a causal mask over target positions. Cross-attention
may read the entire real source text. A triangular mask would be
inappropriate there: translation position 1 may need the last source
word. Source and target lengths need not even match.

For one cross-attention head:

$$
\begin{aligned}
Q&=UW_Q\in\mathbb R^{T_{tgt}\times d_h},\\
K&=MW_K\in\mathbb R^{T_{src}\times d_h},\\
V&=MW_V\in\mathbb R^{T_{src}\times d_h}.
\end{aligned}
$$

The score matrix has shape $T_{tgt}\times T_{src}$. It determines which
source positions contribute to each target row. Every decoder block has
its own $W_K,W_V$: memory $M$ is shared, but its projections need not be
the same across layers.

In a numerical example, the projections produce two queries and three
keys, with the last key corresponding to PAD:

$$
\begin{aligned}
Q&=\begin{pmatrix}1&0\\0&\sqrt2\ln3\end{pmatrix},\\
K&=\begin{pmatrix}\sqrt2\ln2&0\\0&1\\100&100\end{pmatrix},\\
V&=\begin{pmatrix}2&0\\0&3\\100&100\end{pmatrix}.
\end{aligned}
$$

After division by $\sqrt2$ and exclusion of PAD, the first row's scores
are $(\ln2,0,-\infty)$. Their exponentials are proportional to $(2,1,0)$,
the weights are $(2/3,1/3,0)$, and the context is $(4/3,1)$.
For the second row, scores $(0,\ln3,-\infty)$ give weights $(1/4,3/4,0)$
and context $(1/2,9/4)$. The two queries use the same source values in
different proportions. Even very large numbers in the PAD row do not
affect the context.

```python
import math
import torch

q = torch.tensor([[1., 0.], [0., math.sqrt(2)*math.log(3)]],
                 dtype=torch.float64)
k = torch.tensor([[math.sqrt(2)*math.log(2), 0.],
                  [0., 1.], [100., 100.]], dtype=q.dtype)
v = torch.tensor([[2., 0.], [0., 3.], [100., 100.]], dtype=q.dtype)
scores = (q @ k.T / math.sqrt(2)).masked_fill(
    torch.tensor([[False, False, True]]), -torch.inf)
a = scores.softmax(-1)
context = a @ v
torch.testing.assert_close(a, torch.tensor(
    [[2/3, 1/3, 0.], [1/4, 3/4, 0.]], dtype=q.dtype))
torch.testing.assert_close(context, torch.tensor(
    [[4/3, 1.], [1/2, 9/4]], dtype=q.dtype))
print(a, context)
```

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/full-transformer-2026-10-21/transformer_resideual_layer_norm_3.png]]

*Each decoder block has three sublayers and three residual connections;
an encoder block has two. The left side shows two source positions, and
the right side shows target-position representations. Decoder
self-attention must be causal, although the diagram labels it simply
Self-Attention.
Source: Jay Alammar, [The Illustrated Transformer, The Residuals](https://jalammar.github.io/illustrated-transformer/#the-residuals),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

The last decoder block still returns vectors of width $d$. A separate
output projection is needed to obtain a vocabulary distribution.

## 7. Training and generation

### From a decoder state to a distribution

Let $h_i\in\mathbb R^d$ be the output of the last block. Projection
$W_{out}\in\mathbb R^{d\times|V|}$ computes
$z_i=h_iW_{out}+b_{out}$. Coordinate $z_{iv}$ is the logit for token $v$.
Softmax over the vocabulary gives $p(v\mid x,y_{<i})$. This is not yet a
selected token: argmax, sampling, and beam search use the same distribution
in different ways.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/transformer_decoder_output_softmax.png]]

*The output vector is expanded to vocabulary size. The diagram selects
the index with maximum probability: greedy decoding. The label log_probs
corresponds to log-softmax, while the box is named Softmax. Both give the
same argmax index, but probabilities and log-probabilities are different
values. The code below passes raw logits to cross-entropy.
Source: Jay Alammar, [The Illustrated Transformer, The Final Linear and Softmax Layer](https://jalammar.github.io/illustrated-transformer/#the-final-linear-and-softmax-layer),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

For an example, restrict the output vocabulary to three tokens: `cat`,
`sleeps`, and `<EOS>`. With state $h=(1,2)$ and columns of $W_{out}$
equal to $(1,0)$, $(0,1)$, and $(-1,0)$, the logits are $(1,2,-1)$.
The normalizing sum is $e^1+e^2+e^{-1}$; the probabilities are approximately
$(0.259496,0.705385,0.035119)$. For the correct token `sleeps`, loss
$-\ln p$ is $0.349012$.

The derivative with respect to logits is $p-\mathrm{onehot}(\text{target})$:
$(0.259496,-0.294615,0.035119)$. A gradient-descent step increases the
correct logit and decreases the others if these three numbers are
optimized directly. A network optimizes weights: the gradient passes
through $W_{out}$ into the last block, cross-attention, encoder memory,
and embedding tables.

In the original paper, one matrix is shared between the source table,
target table, and output projection when using a shared vocabulary.
Then $W_{out}=E^\top$, and gradients from every use add together.
Equal matrix dimensions do not themselves establish sharing: the code
must use one parameter object, not two copies of its numbers.

### Shifting the target sequence and applying masks

During training, the correct translation is known:

```text
position:         0       1       2       3
decoder input:  <BOS>    the     cat    sleeps
target:          the    cat    sleeps   <EOS>
```

Row 2 receives `cat` and must predict `sleeps`. Its self-attention may
read positions 0, 1, and 2. The token `sleeps` is already present at input
row 3, but the causal mask prevents reading it. Without shifting, the
target token would occupy the same row; a triangular mask alone would
not prevent that leakage.

All decoder layers preserve causal connections. The FFN and LayerNorm
act on one position and add no future connections. Cross-attention reads
only the source, so it does not expose future translation tokens either.
All four distributions can be computed in parallel and trained using
the sum or mean of their losses.

When examples are padded to a common length, three separate operations
are required: excluding PAD keys in the encoder, excluding target-prefix
PAD keys in decoder self-attention, and excluding PAD targets from the
loss. Cross-attention reuses the **source** PAD-key mask. A key mask need
not zero PAD-query states: these rows may be nonzero, but must not affect
real positions or the loss.

For logits $z_{bt}$ and targets $y_{bt}$, use the mean loss over $n$
real target tokens:

$$
\mathcal L=-\frac1n\sum_{(b,t):\,y_{bt}\ne PAD}
\log\operatorname{softmax}(z_{bt})_{y_{bt}}.
$$

EOS contributes to this sum: the model learns to stop. BOS starts the
prefix, but is not an additional target in this layout.

The original Transformer used label smoothing with $\epsilon_{ls}=0.1$.
For three classes, a target with confidence $0.9$ on the correct class
and $0.05$ on each other class gives loss $0.549012$ in our example.
The objective fits a soft target distribution instead of driving one
probability to one. [The Annotated Transformer, Label Smoothing](https://nlp.seas.harvard.edu/annotated-transformer/#label-smoothing)
also excludes the PAD class, and its loss function returns KL divergence.
This differs from cross-entropy by the target distribution's entropy,
but has the same gradient with respect to logits when the target is fixed.
PyTorch's `label_smoothing` mixes the target with a uniform distribution
**over all** classes; for three classes, that gives different values.
The complete example below disables smoothing to check the shift, masks,
and ordinary cross-entropy separately.

```python
import torch
import torch.nn.functional as F

z = torch.tensor([1., 2., -1.], dtype=torch.float64, requires_grad=True)
logp = z.log_softmax(-1)
p = logp.exp()
loss = -logp[1]
loss.backward()
expected_grad = p.detach() - torch.tensor([0., 1., 0.])
torch.testing.assert_close(z.grad, expected_grad)
soft_target = torch.tensor([0.05, 0.9, 0.05], dtype=z.dtype)
smooth_ce = -(soft_target * logp).sum()
smooth_kl = (soft_target * (soft_target.log() - logp)).sum()
torch.testing.assert_close(
    F.cross_entropy(z[None], torch.tensor([1]), label_smoothing=0.1),
    -(torch.tensor([1/30, 14/15, 1/30], dtype=z.dtype) * logp).sum())
print(p.detach(), float(loss.detach()), float(smooth_ce.detach()))
print("KL", float(smooth_kl.detach()), "gradient", z.grad)
```

### A complete forward pass and one training step

The implementation below retains post-norm, three decoder sublayers,
embedding scaling, sinusoidal positions, and a shared vocabulary. It
shows the computation graph but does not reproduce Transformer Base
training: width, layer count, initialization, data, optimizer, and loss
are chosen for teaching. Masks use `True = allowed key`.

This implementation uses right padding: each source has at least one
real token, and the target input begins with BOS. A query with no allowed
keys is explicitly rejected. Left padding would need the empty-row
handling explained in the masking chapter.

```python
import math
import torch
from torch import nn
import torch.nn.functional as F

PAD, BOS, EOS = 0, 1, 2

class MHA(nn.Module):
    def __init__(self, d, heads, dropout=0.0):
        super().__init__()
        if d % heads:
            raise ValueError("d must be divisible by heads")
        self.heads, self.dh = heads, d // heads
        self.q, self.k, self.v, self.o = [nn.Linear(d, d) for _ in range(4)]
        self.dropout = dropout

    def forward(self, query, key, value, allowed):
        b, tq, d = query.shape
        def split(x):
            return x.reshape(b, -1, self.heads, self.dh).transpose(1, 2)
        q, k, v = split(self.q(query)), split(self.k(key)), split(self.v(value))
        if allowed.dtype != torch.bool or not allowed.any(-1).all():
            raise ValueError("each query needs an allowed key; use boolean masks")
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        weights = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
        self.last_weights = weights
        weights = F.dropout(weights, self.dropout, self.training)
        out = (weights @ v).transpose(1, 2).contiguous().reshape(b, tq, d)
        return self.o(out)

class EncoderBlock(nn.Module):
    def __init__(self, d, heads, ff, dropout):
        super().__init__()
        self.attn = MHA(d, heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Linear(ff, d))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, allowed):
        x = self.n1(x + self.drop(self.attn(x, x, x, allowed)))
        return self.n2(x + self.drop(self.ff(x)))

class DecoderBlock(nn.Module):
    def __init__(self, d, heads, ff, dropout):
        super().__init__()
        self.self_attn, self.cross_attn = MHA(d, heads, dropout), MHA(d, heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Linear(ff, d))
        self.n1, self.n2, self.n3 = [nn.LayerNorm(d) for _ in range(3)]
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, self_allowed, source_allowed):
        x = self.n1(x + self.drop(self.self_attn(x, x, x, self_allowed)))
        x = self.n2(x + self.drop(self.cross_attn(x, memory, memory, source_allowed)))
        return self.n3(x + self.drop(self.ff(x)))

class TinyTranslation(nn.Module):
    def __init__(self, vocab=10, d=8, heads=2, ff=16, layers=2, dropout=0.0):
        super().__init__()
        if d % 2 or layers < 1:
            raise ValueError("use even d and a nonempty stack")
        self.emb = nn.Embedding(vocab, d)
        self.enc = nn.ModuleList([EncoderBlock(d, heads, ff, dropout)
                                  for _ in range(layers)])
        self.dec = nn.ModuleList([DecoderBlock(d, heads, ff, dropout)
                                  for _ in range(layers)])
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight
        self.drop = nn.Dropout(dropout)

    def embed(self, ids):
        d = self.emb.embedding_dim
        pos = torch.arange(ids.shape[1], device=ids.device, dtype=self.emb.weight.dtype)
        freq = 10000.0 ** (-torch.arange(
            0, d, 2, device=ids.device, dtype=self.emb.weight.dtype) / d)
        angles = pos[:, None] * freq[None, :]
        pe = torch.stack((angles.sin(), angles.cos()), -1).flatten(-2)
        return self.drop(math.sqrt(d) * self.emb(ids) + pe)

    def encode(self, src):
        valid = src.ne(PAD)
        memory = self.embed(src)
        for layer in self.enc:
            memory = layer(memory, valid[:, None, None, :])
        return memory, valid

    def decode(self, ids, memory, source_valid):
        t = ids.shape[1]
        causal = torch.ones(t, t, dtype=torch.bool, device=ids.device).tril()
        allowed = causal[None, None] & ids.ne(PAD)[:, None, None, :]
        x = self.embed(ids)
        for layer in self.dec:
            x = layer(x, memory, allowed, source_valid[:, None, None, :])
        return self.head(x)

    def forward(self, src, decoder_ids):
        memory, valid = self.encode(src)
        return self.decode(decoder_ids, memory, valid)

torch.manual_seed(7)
model = TinyTranslation().double()
# Shared vocabulary: 3=кот, 4=спит, 5=the, 6=cat, 7=sleeps.
src = torch.tensor([[3, 4, EOS, PAD]])
target = torch.tensor([[BOS, 5, 6, 7, EOS, PAD]])
decoder_ids, labels = target[:, :-1], target[:, 1:]
logits = model(src, decoder_ids)
assert logits.shape == (1, 5, 10)
loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=PAD)
token_losses = F.cross_entropy(
    logits.flatten(0, 1), labels.flatten(), ignore_index=PAD, reduction="none")
torch.testing.assert_close(loss, token_losses.sum() / labels.ne(PAD).sum())
target_probabilities = logits.detach().softmax(-1).gather(-1, labels[..., None]).squeeze(-1)
print("per-target loss:", token_losses.detach().tolist())
print("target probabilities:", target_probabilities.tolist())
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer.zero_grad()
loss.backward()
assert model.enc[0].attn.q.weight.grad is not None
assert model.head.weight is model.emb.weight
print("shared PAD row gradient norm:", model.emb.weight.grad[PAD].norm().item())
optimizer.step()
new_loss = F.cross_entropy(
    model(src, decoder_ids).flatten(0, 1), labels.flatten(), ignore_index=PAD)
print("before/after", float(loss.detach()), float(new_loss.detach()))
print("cross-attention", tuple(model.dec[-1].cross_attn.last_weights.shape))
```

Here memory has shape `[1,4,8]`, decoder input `[1,5]`, cross-attention
`[1,2,5,4]`, and logits `[1,5,10]`. The loss includes four targets,
`the cat sleeps <EOS>`, but not the final PAD. Separate block instances
do not share weights across depth; the embedding table and output layer,
by contrast, intentionally share them.

With the specified seed on CPU in float64, the correct-token probabilities
and losses before the training step are:

| Target | Target probability | Loss $-\ln p$ |
| --- | ---: | ---: |
| `the` | $5.619617\cdot10^{-5}$ | 9.786662 |
| `cat` | $9.737951\cdot10^{-3}$ | 4.631725 |
| `sleeps` | $1.880076\cdot10^{-5}$ | 10.881613 |
| `<EOS>` | $5.502324\cdot10^{-6}$ | 12.110340 |

The four losses sum to approximately 37.410340, with mean 9.352585.
The fifth element of `token_losses` is zero: its PAD target is excluded.
Dividing by all five elements would give 7.482068, so an ordinary
`token_losses.mean()` is not the same as cross-entropy with
`ignore_index`. The denominator is the number of real targets, not the
padded tensor's length. After one SGD step, the same example's mean loss
falls to 8.497538.

PAD nevertheless remains one of the ten output softmax classes.
`ignore_index=PAD` excludes a target position; it does not remove the PAD
logit from the denominator at other positions. These are distinct
operations in the [CrossEntropyLoss formula](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html).
The output projection can therefore send a gradient to the PAD row of
the shared table E, even though source PAD keys are not read. This is
not information leaking from padding: an output-class weight is changing.

A gradient for the first encoder block's query projection demonstrates
the path from translation error through the decoder and cross-attention
into the encoder. Checking only `head.weight.grad` is insufficient:
the output layer could train even if memory had accidentally been detached.
One loss-reducing step does not mean that the model has learned to
translate or generalize.

### Autoregressive inference

At generation time, the correct `the cat sleeps` is not yet available
as input. The decoder first receives only BOS and produces a distribution
for the first token. The chosen token is appended to the prefix, and the
computation repeats until EOS or a length limit.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/transformer_decoding_2.gif]]

*Repeated decoder steps with a growing target prefix. The source phrase
has already been encoded; each newly selected token returns as input at
the bottom. The author's animation is reproduced without changing frames.
Source: Jay Alammar, [The Illustrated Transformer, The Decoder Side](https://jalammar.github.io/illustrated-transformer/#the-decoder-side),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

An implementation can retain encoder memory instead of rerunning the
encoder. Cross-attention K/V also remain fixed during one translation,
although they differ across decoder layers. The decoder's own KV cache
retains K/V for processed target positions. In a post-norm Transformer
with no future connections, these old states do not change when another
token is appended. Equivalence to full recomputation assumes eval mode
and the same positions and masks.

Caching reduces repeated computation; the next unknown token still
depends on the token just selected. Neither masking nor parallel
training makes ordinary autoregressive generation a single pass. The
implementation above has no cache: for clarity, `decode` recomputes
the entire prefix.

## 8. Three architectural branches

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-30/gpt-2-transformer-xl-bert-3.png]]

*GPT-2 on the left has causal blocks without cross-attention; BERT in
the center has bidirectional blocks; Transformer-XL on the right carries
memory across segments. The figure does not compare GPT-1 with LLaMA;
their differences are described below.
Source: Jay Alammar, [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

In the encoder–decoder just examined, the source becomes separate memory
and the decoder generates a conditional output. Two other variants retain
only the encoder or only a causal decoder. LLaMA develops the latter,
changing normalization, the FFN, and the positional mechanism.

### Encoder-only: BERT

BERT retains the encoder stack:

- full bidirectional self-attention;
- token, segment, and learned position representations;
- masked language modeling;
- next sentence prediction in the original model.

MLM selects 15% of WordPieces; 80% of the selected pieces become `[MASK]`,
10% become random tokens, and 10% stay unchanged. The loss is computed
at selected positions. BERT is not an autoregressive generator:
pre-training reconstructs hidden pieces rather than continuing a prefix
from left to right ([Devlin et al., §3–3.1](https://arxiv.org/pdf/1810.04805v2#page=3)).

### Decoder-only: GPT

GPT uses a causal stack without an encoder or cross-attention:

- masked self-attention;
- an FFN;
- learned positional representations in early GPT;
- next-token prediction.

GPT-1 uses causal language pre-training to initialize learning on labeled
NLP tasks. GPT-2 then scaled model and data and emphasized transfer without
additional gradient training on the target task. GPT-2's question-answering
experiments placed example pairs in the context without changing weights
([GPT-2 report, §3.8](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf#page=7)).

### Modern decoder-only: LLaMA

LLaMA retains a causal decoder while changing its details:

- pre-normalization with RMSNorm;
- RoPE instead of absolute position representations;
- SwiGLU instead of a ReLU FFN;
- no bias in several linear layers;
- grouped-query attention in later Llama releases.

LLaMA 1 used multi-head attention. GQA in later releases is not a property
of the first version. The changes are described in
[Touvron et al., §2.2–2.4](https://arxiv.org/pdf/2302.13971v1#page=3);
the absence of biases and equal Q/K/V head counts are visible in the
[first release's code](https://github.com/meta-llama/llama/blob/1076b9c51c77ad06e9d7ba8a4c6df775741732bd/llama/model.py).

## 9. Architectural diff

| Component | Transformer 2017 | BERT Base | GPT-1 | LLaMA 1 |
|---|---|---|---|---|
| architecture | encoder–decoder | encoder | causal decoder | causal decoder |
| norm layout | Post-LN | Post-LN | Post-LN | Pre-RMSNorm |
| activation / FFN | ReLU | GELU | GELU | SwiGLU |
| positions | sinusoidal | learned absolute | learned absolute | RoPE |
| cross-attention | decoder only | no | no | no |
| training objective | translation | MLM + NSP | next token + fine-tuning | next token |

In GPT-1, normalization follows the sum in both sublayers, as shown by
`block` in the original [OpenAI train.py](https://github.com/openai/finetune-transformer-lm/blob/a69b5c43b0452462890bca8ff92fb75dee9290cf/train.py#L140-L147).
GPT-2 moves normalization before the sublayers and adds a final LayerNorm
([GPT-2 report, §2.3](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)).
The GPT-1 row therefore does not describe all later GPT models.
The label LLaMA 1 is likewise intentionally restricted to the first release.

<a id="architecture-evidence-boundary"></a>

## 10. An architectural choice is not an empirical recipe

Normalization placement can be established from code. Whether that
placement improves quality requires a training comparison. Replacing
LayerNorm with RMSNorm changes centering, parameter count, and computation
at once; comparing two released models does not separate these causes
from differences in data, tokenization, or training-token count.

[CS336 Lecture 3, slides 2–9](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_03.pdf#page=2)
compares the original Transformer with a modern teaching architecture,
then with a collection of released models. The collection shows common
choices, but does not replace a controlled ablation.

For example, a three-matrix SwiGLU with the same hidden width as a
two-matrix ReLU FFN has roughly one and a half times as many projection
weights. Comparing activations at a similar parameter count requires
adjusting width. Holding width fixed addresses a different question,
but then two properties change in the experiment.

Matching parameter counts does not guarantee equal execution time either:
matrix-kernel efficiency depends on dimensions and implementation.
A quality claim needs comparable data and training; a speed claim needs
measurements on specified hardware in a specified regime.

## 11. Why the Transformer displaced the recurrent backbone

The original paper compares:

| Layer | Complexity per layer | Sequential operations | Maximum path length |
|---|---:|---:|---:|
| self-attention | $O(T^2d)$ | $O(1)$ | $O(1)$ |
| recurrent | $O(Td^2)$ | $O(T)$ | $O(T)$ |
| dilated convolution | $O(kTd^2)$ | $O(1)$ | $O(\log_k T)$ |

The logarithmic path in the last row assumes a stack of increasingly
dilated kernels of width $k>1$. For ordinary contiguous kernels without
dilation, the paper gives $O(T/k)$: each layer expands the accessible
region only linearly. Cost is per layer; path length traverses the depth
needed to connect distant positions.

The table compares interactions between positions inside a layer. A
complete Transformer also includes Q/K/V projections of order $Td^2$
and an FFN of order $Tdd_{ff}$. The row $O(T^2d)$ alone therefore neither
describes the entire block's cost nor proves a speedup at every length.

When processing a known sequence, all queries can be computed in
parallel; two distant tokens are connected by one full attention layer.
Dense attention's pairwise arithmetic grows quadratically with length.
FlashAttention changes memory traffic without removing that arithmetic;
sparse and linear attention change the structure or computation of
the interactions.

> [!warning] The table depends on the regime
> $O(1)$ sequential operations refers to processing an already known sequence
> inside a layer, not generating unknown future tokens. Self-attention is
> not always cheaper than a recurrent network: the comparison depends on
> $T$, $d$, hardware, and implementation.

## 12. A minimal modern causal block

[CS336 Assignment 1, §3.5, pages 26–27](https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf#page=26)
describes the complete block as follows, in the original:

> A Transformer block contains two ‘sub-layers’, one for the multihead self attention,
> and another for the SwiGLU feed-forward network. In each sub-layer, we first
> perform RMSNorm, then the main operation (MHA/FF), finally adding in the residual connection.

*Stanford CS336, Assignment 1, pinned revision a158843b.
License: [MIT](/licenses/stanford-cs336-assignment1-MIT.txt).*

Instead of the translator decoder's three sublayers, two remain: causal
self-attention and an FFN. The source text is now the same stream that
the model continues. An embedding table, block stack, final RMSNorm, and
output projection together form a language model.

RMSNorm divides a row by the root mean square of its coordinates without
subtracting the mean:

$$
\operatorname{RMSNorm}(x)=
\gamma\odot\frac{x}{\sqrt{\frac1d\sum_r x_r^2+\epsilon}}.
$$

For $(1,2,3,4)$, the denominator without $\epsilon$ is $\sqrt{7.5}$,
and with $\gamma=1$ the output coordinates are approximately 0.365148,
0.730297, 1.095445, and 1.460593. This is not the LayerNorm result from
Section 2: the row's positive offset has not been removed. In this block,
RMSNorm precedes attention and the FFN; RoPE acts on Q/K after their
projections; V is not rotated.

The following code is self-contained. It uses ordinary MHA, without
GQA, KV caching, dropout, MoE, or mechanisms specific to individual
releases. It is a small modern decoder following this architecture,
not a LLaMA weight loader or a complete solution to the CS336 tests.

```python
import math
import torch
from torch import nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * (x.square().mean(-1, keepdim=True) + self.eps).rsqrt() * self.weight

class RotarySelfAttention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        if d % heads or (d // heads) % 2:
            raise ValueError("use even head width and d divisible by heads")
        self.heads, self.dh = heads, d // heads
        self.q, self.k, self.v, self.o = [
            nn.Linear(d, d, bias=False) for _ in range(4)]

    def rotate(self, x):
        t = x.shape[-2]
        positions = torch.arange(t, device=x.device, dtype=x.dtype)
        freq = 10000.0 ** (-torch.arange(0, self.dh, 2,
                                        device=x.device, dtype=x.dtype) / self.dh)
        angles = positions[:, None] * freq[None, :]
        pairs = x.reshape(*x.shape[:-1], -1, 2)
        a, b = pairs.unbind(-1)
        return torch.stack((a*angles.cos() - b*angles.sin(),
                            a*angles.sin() + b*angles.cos()), -1).flatten(-2)

    def forward(self, x):
        b, t, d = x.shape
        def split(y):
            return y.reshape(b, t, self.heads, self.dh).transpose(1, 2)
        q, k, v = self.rotate(split(self.q(x))), self.rotate(split(self.k(x))), split(self.v(x))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
        allowed = torch.ones(t, t, device=x.device, dtype=torch.bool).tril()
        weights = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
        y = (weights @ v).transpose(1, 2).contiguous().reshape(b, t, d)
        return self.o(y)

class SwiGLU(nn.Module):
    def __init__(self, d, ff):
        super().__init__()
        self.gate, self.up = nn.Linear(d, ff, bias=False), nn.Linear(d, ff, bias=False)
        self.down = nn.Linear(ff, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class Block(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.attn_norm, self.ffn_norm = RMSNorm(d), RMSNorm(d)
        self.attn, self.ffn = RotarySelfAttention(d, heads), SwiGLU(d, ff)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))

class TinyCausalLM(nn.Module):
    def __init__(self, vocab=10, d=8, heads=2, ff=24, layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, heads, ff) for _ in range(layers)])
        self.norm, self.head = RMSNorm(d), nn.Linear(d, vocab, bias=False)

    def forward(self, ids):
        x = self.emb(ids)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

torch.manual_seed(11)
lm = TinyCausalLM().double()
tokens = torch.tensor([[1, 5, 6, 7, 2]])
logits = lm(tokens[:, :-1])
loss = F.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten())
loss.backward()
print("logits", tuple(logits.shape), "loss", float(loss.detach()))
```

Layers run sequentially in depth; positions within a layer run in
parallel. Input `[1,4]` produces logits `[1,4,10]` for four next-token
targets. There is no padding in this example. A variable-length batch
would need key masks and PAD-target exclusion, as in the translator
above. The output matrix is **not tied** to the embedding table here.

RMSNorm, SwiGLU, and operation order are examined further in
[[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/03 Pre-norm, RMSNorm, SwiGLU и residual|the anatomy of a modern block]].
The official CS336 adapters and tests remain in the
[course repository](https://github.com/stanford-cs336/assignment1-basics/tree/a158843b20107949f1a8d7df1b05cd33b9166712/tests);
running this example does not count as passing those tests.

## 13. How to read real code

Three levels of difficulty:

1. [Karpathy's `ng-video-lecture`](https://github.com/karpathy/ng-video-lecture)
   builds a causal decoder incrementally in PyTorch.
2. [Karpathy's `build-nanogpt`](https://github.com/karpathy/build-nanogpt)
   reproduces the GPT-2 124M architecture, weight loading, data pipeline,
   initialization, and training.
3. [Meta Llama 3 reference code](https://github.com/meta-llama/llama3/blob/main/llama/model.py)
   implements RoPE, RMSNorm, SwiGLU, and GQA in a modern model.

These implementations have different purposes and scales. Short teaching
code shows how operations connect; a GPT-2 reproduction adds weight
compatibility and a training procedure; Llama reference code specifies
a concrete parameter format and model computation. None should
automatically be treated as an exact implementation of the 2017 Transformer.

## 14. Tensor-shape check

For a causal LLM with ordinary multi-head attention (MHA):

```text
token ids                 [B, T]
token embeddings          [B, T, d]
Q, K, V                   [B, h, T, dh]
attention scores          [B, h, T, T]
attention result          [B, T, d]
residual after attention  [B, T, d]
FFN hidden                [B, T, dff]
residual after FFN        [B, T, d]
vocabulary logits         [B, T, |Vocab|]
```

With GQA, Q has $h_q$ heads while K/V have $h_{kv}<h_q$, giving shapes
`[B,h_q,T,dh]` and `[B,h_kv,T,dh]` respectively. The table is therefore
not a universal KV-head layout.

For $x+F(x)$, both summands must have matching shapes. Shape agreement
does not by itself check element order after head splitting, mask
correctness, or the normalization used.

Parameter counts, matrix products, and backward-memory requirements
are covered in the
[[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/05 Transformer с нуля — формы, параметры и стоимость|chapter on shapes, parameters, and cost]],
after which the same computation can be implemented in
[[02 Areas/ML & DL/06 Практика/20 Собрать языковую модель с нуля|Capstone 1]].

## Summary

- The original Transformer is an encoder–decoder, not a GPT-like tower.
- An encoder block contains full self-attention and an FFN.
- The original decoder contains causal self-attention, cross-attention, and an FFN.
- Attention transfers information between positions; the FFN transforms each
  position nonlinearly and independently.
- The residual stream preserves shape `B × T × d_model`.
- BERT, GPT, and LLaMA differ in training objective, positional mechanism,
  normalization, and FFN, not just masking.
- A modern LLM diagram must not be presented as the exact 2017 architecture.

## Sources and interactive explanations

### Primary sources

- [Stanford CS336 Spring 2026, Lecture 3 — pinned source](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_03.pdf)
- [Stanford CS336 Spring 2026, Assignment 1 — pinned handout](https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf)
- [Vaswani et al., Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Radford et al., Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [Devlin et al., BERT](https://arxiv.org/abs/1810.04805)
- [Touvron et al., LLaMA](https://arxiv.org/abs/2302.13971)

### Code and explanations

- [[05 Источники/Courses/Harvard ML Systems/tinytorch/13_transformers|TinyTorch 13 — Transformers]] — executable construction of `TransformerBlock` from attention, MLP, and layer normalization, with autoregressive generation.
- [Harvard, The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- [Karpathy, Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Karpathy, Let's reproduce GPT-2](https://www.youtube.com/watch?v=l8pRSuU81PU)
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
- [Jay Alammar, The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [[02 Areas/ML & DL/01 Справочник/Архитектурные паттерны/Encoder-Decoder|Encoder–Decoder reference]]

**Previous:** [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/04 Позиционная информация|Positional information]] ·
**Next:** [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна|Three Transformer architectural patterns]]
