---
title: Self-Attention — Q, K, V
type: textbook-chapter
status: canonical
locale: en
translation_status: current
canonical_revision: "75fd6a6a528fc69025b84f44acb10af1a6ed20d7ec1e271ab90f4b48fc73c02c"
translation_reviewed: "2026-11-21"
translation_of: "00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V.md"
last_updated: 2026-11-21
primary_sources:
  - https://arxiv.org/abs/1706.03762
---

# Self-Attention — Q, K, V

<a id="why-a-word-must-look-at-other-words"></a>

## Token representations in context

In `American shrew mole`, `one mole of carbon dioxide`, and
`a biopsy of the mole`, the word `mole` names an animal, a unit, and a mark
on the skin. If the tokenizer assigns it the same token ID in all three
cases, the embedding table returns the same vector. Position features can
tell the network where the token occurs, but not which meaning the sentence
calls for. The example comes from [3Blue1Brown's lesson on attention](https://www.3blue1brown.com/lessons/attention/).
A real tokenizer can also split a word into several tokens.

In the preceding chapter, the decoder read the source sentence. In
self-attention, each query reads from the sequence that supplies it.
An encoder can use `cat` to represent `was tired` in “The cat that had been
running all day was tired.” A model predicting the next token `tired` uses
the preceding position instead. Letting that position read `tired` would
give it the target.

<a id="1-the-simplest-channel-average-the-past"></a>

## Averaging the current and earlier positions

Let $x_i$ be the vector at position $i$. Define its context as the average of
all available tokens:

$$
\bar{x}_i=\frac{1}{i}\sum_{j\le i}x_j.
$$

For four tokens this is a matrix multiplication:

$$
\begin{bmatrix}
1&0&0&0\\
\frac12&\frac12&0&0\\
\frac13&\frac13&\frac13&0\\
\frac14&\frac14&\frac14&\frac14
\end{bmatrix}
\begin{bmatrix}x_1\\x_2\\x_3\\x_4\end{bmatrix}.
$$

The lower-triangular matrix defines access: row $i$ uses only the current and
earlier positions $j\le i$.

For input rows $(1,0)$, $(0,2)$, $(3,1)$, and $(-1,3)$, the outputs are
$(1,0)$, $(0.5,1)$, $(4/3,1)$, and $(0.75,1.5)$. The third row adds the first
three input vectors and divides by three. It does not wait for the second
output row, so all four rows can be evaluated together.

```python
import torch

X = torch.tensor([[1., 0.], [0., 2.], [3., 1.], [-1., 3.]],
                 dtype=torch.float64)
allowed = torch.ones(4, 4, dtype=torch.bool).tril()
A_mean = allowed.to(X.dtype)
A_mean /= A_mean.sum(-1, keepdim=True)
loop = torch.stack([X[:i+1].mean(0) for i in range(len(X))])
torch.testing.assert_close(A_mean @ X, loop)
scores = torch.zeros(4, 4, dtype=X.dtype).masked_fill(~allowed, -torch.inf)
torch.testing.assert_close(scores.softmax(-1), A_mean)
print(A_mean @ X)
```

The third row gives each of its three inputs a weight of $1/3$, whatever
their vectors contain. [Karpathy's GPT lecture](https://www.youtube.com/watch?v=kCc8FmEb1nY)
starts with that fixed average, then lets the input determine the weights.

<a id="2-make-weights-depend-on-content"></a>

## Queries, keys, and values

Keep the weighted sum, but let the input determine both what goes into
it and how much weight each term gets. Three learned matrices produce
a query, key, and value from every input row:

$$
\begin{gathered}
q_i=x_iW_Q,\qquad k_i=x_iW_K,\\
v_i=x_iW_V.
\end{gathered}
$$

To compute output $i$, compare its query $q_i$ with the keys $k_j$ of
the available positions. The resulting scores determine the weights
on those positions' values $v_j$. The same three matrices act on every
row, so a longer sequence does not add projection parameters.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/attention/transformer_self_attention_vectors.png|Thinking passes through three separate matrices to produce q₁, k₁, and v₁]]

*Jay Alammar, [The Illustrated
Transformer](https://jalammar.github.io/illustrated-transformer/): every token
vector is projected into Q, K, and V. CC BY-NC-SA 4.0.*

<a id="4-a-numerical-example-by-hand"></a>

## The output at the third token

Use the three tokens `a cat sleeps`, with all three positions visible.
Set $X=I_3$. Each projection matrix has shape $[3,2]$, and $W_Q$
and $W_K$ have the same entries in this example. The projected rows are:

$$
Q=K=
\begin{bmatrix}
1&0\\
0&1\\
1&1
\end{bmatrix},
\qquad
V=
\begin{bmatrix}
1&0\\
0&2\\
3&1
\end{bmatrix}.
$$

Multiplying by $x_i$ selects row $i$ of each projection matrix.
For instance, $x_3=(0,0,1)$ gives $q_3=k_3=(1,1)$ and $v_3=(3,1)$.
The matrices $W_Q$ and $W_K$ are still separate parameters. Using
independent rows of $I_3$ lets us set all three values independently:
if $x_3=x_1+x_2$, linearity would force $v_3=v_1+v_2$.

<a id="step-a-dot-products"></a>

### Comparing the query with the keys

The score $s_{ij}=q_i k_j^\top$ compares query $i$ with key $j$.
Both vectors therefore need the same number of entries, $d_k$.
For the third query:

$$
q_3K^\top=(1,1,2).
$$

The first two dot products are $1\cdot1+1\cdot0$ and $1\cdot0+1\cdot1$.
The third is $1\cdot1+1\cdot1$. With the query fixed, changing one key
changes only its score before softmax. Changing the query can change
the whole row.

<a id="step-b-scaling"></a>

### Scaling and softmax

Divide the scores by $\sqrt{d_k}=\sqrt2$:

$$
\frac{q_3K^\top}{\sqrt2}\approx(0.707107,0.707107,1.414214).
$$

<a id="step-c-softmax"></a>

To apply softmax, exponentiate the scores and divide each exponential by
their sum. Subtracting the same number from all three scores leaves their
ratios unchanged. Subtract $1/\sqrt2$ to get exponentials
$(1,1,e^{1/\sqrt2})$. Their sum is $2+e^{1/\sqrt2}$, giving the weights

$$
a_3\approx(0.248255,0.248255,0.503490).
$$

For a forbidden key, the mask replaces its score with $-\infty$ before
softmax. Its exponential is zero, so it contributes nothing to the
denominator, provided at least one key is allowed.

<a id="step-d-weighted-sum-of-values"></a>

### The weighted sum of values

$$
\begin{aligned}
z_3={}&0.248255[1,0]+0.248255[0,2]\\
&+0.503490[3,1]\\
\approx{}&[1.758725,1].
\end{aligned}
$$

The third value receives about half the weight. The other two also
contribute: the second coordinate is $2(0.248255)+0.503490=1$
to the displayed precision.

<a id="3-why-value-is-separate"></a>

Now change only $v_3$, from $(3,1)$ to $(3,3)$. The queries, keys, and
weights stay fixed. The second coordinate of $z_3$ increases by
$2a_{33}\approx1.006980$, giving $(1.758725,2.006980)$. With $X=I_3$,
this takes a change to just one entry of $W_V$.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/self-attention-2026-10-21/self-attention-output.png|Scores for Thinking and Machines are scaled and normalized, and the two resulting weights multiply their value vectors]]

*Alammar's example compares the query for `Thinking` with the keys for
`Thinking` and `Machines`. The key width is 64, so the scores are
divided by 8. The resulting weights of 0.88 and 0.12 multiply the first
and second values. Jay Alammar,
[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/#self-attention-in-detail),
CC BY-NC-SA 4.0.*

Before attention dropout, every nonempty row of weights is nonnegative and sums
to one. The output is therefore in the convex hull of its available values.
Residual addition and the FFN are later operations, not part of this statement.
Attention softmax normalizes source positions; the language model's output
softmax normalizes vocabulary entries. An attention output is a feature vector,
not a next-token distribution.

Each row lists one query's weights. The column labels name the value
vectors multiplied by those weights. The table rounds to four decimal places.

| Query | $v_1$ | $v_2$ | $v_3$ |
|---|---:|---:|---:|
| $q_1$ | 0.4011 | 0.1978 | 0.4011 |
| $q_2$ | 0.1978 | 0.4011 | 0.4011 |
| $q_3$ | 0.2483 | 0.2483 | 0.5035 |

Even with $Q=K$, $a_{13}=0.401112$ and $a_{31}=0.248255$ differ.
Their raw scores are equal, but row 1 contains $(1,0,1)$ and row 3
contains $(1,1,2)$. The softmax denominators differ.

The query for `cat` assigns `sleeps` a weight of $0.401112$.
An encoder can use that position. A model predicting `sleeps`
from `a cat` must [[en/00 Textbook/05 Attention and Transformer/03 Masking, multi-head attention, and tensor shapes#Causal masking and shifted targets|mask that key]].

<a id="the-whole-computation-in-two-figures"></a>

## Matrix form and shapes

Stack the $T$ input vectors as rows of $X$ to compute $Q=XW_Q$,
$K=XW_K$, and $V=XW_V$ together. The query and key matrices share
the width $d_k$. A scalar weight can multiply a value of a different
width, $d_v$.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/attention/self-attention-matrix-calculation-2.png|The query and key matrices produce scores; row-wise softmax gives the weights that multiply V]]

*Jay Alammar, [The Illustrated
Transformer](https://jalammar.github.io/illustrated-transformer/): the matrix
form of the full computation. Read it from right to left: Values determine the
content being transmitted, while `softmax(QKᵀ/√dₖ)` supplies mixing weights.
CC BY-NC-SA 4.0.*

Without a mask, the complete computation is

$$
\operatorname{Attention}(Q,K,V)=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

For one sequence, the matrix shapes are:

| Tensor | Shape | Meaning of a row |
|---|---|---|
| $X$ | $[T,d_{\mathrm{model}}]$ | input features at a position |
| $Q,K$ | $[T,d_k]$ | query or key |
| $V$ | $[T,d_v]$ | transmitted features |
| $S=QK^\top/\sqrt{d_k}$ | $[T,T]$ | one query's scores against all keys |
| $A=\operatorname{softmax}(S)$ | $[T,T]$ | weights normalized across keys |
| $Z=AV$ | $[T,d_v]$ | result for one query |

Each row is normalized independently. Column sums need not equal one, and
output row $i$ still belongs to input query $i$.

<a id="5-why-divide-by-dksqrtd_kdk"></a>

## Score scale and the softmax derivative

If the components of $q$ and $k$ are independent with mean zero and
variance one, then $\mathbb E[q_a^2k_a^2]=1$ and $\mathbb E[q_ak_a]=0$.
Each product has variance one. Those products are independent, so

$$
\operatorname{Var}(q k^\top)=d_k,\qquad
\operatorname{Var}(q k^\top/\sqrt{d_k})=1.
$$

Typical differences between dot-product scores grow with dimension under these
assumptions. Softmax of widely separated scores
becomes nearly one-hot: changing one score slightly barely changes most output
weights, making optimization harder. Division by $\sqrt{d_k}$ restores a
typical scale near one.

For scores $(0,1)$ the second weight is about 0.731059; for $(0,8)$ it is
0.999665. Its derivative with respect to its own score is $p(1-p)$, which
falls from about 0.196612 to 0.000335. A large common offset does not have
this effect: softmax of $(100,100)$ is still $(0.5,0.5)$.

The independence calculation describes a scale model, not a guarantee about
trained features. It does not apply literally to the previous deterministic
example where $Q=K$. Correlations and learned projection norms can still
produce sharp attention after scaling.

The divisor is $\sqrt{d_k}$, the key dimension of one head, not automatically
$\sqrt{d_{model}}$.

<a id="6-minimal-implementation-of-one-head"></a>

## Training the projections

```python
import math
import torch

def attention(q, k, v):
    # q: [..., Tq, dk], k: [..., Tk, dk], v: [..., Tk, dv]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    weights = scores.softmax(dim=-1)  # normalize across keys
    output = weights @ v
    return output, weights
```

The function accepts projected tensors. The shared input and projection
matrices must be constructed before calling it.

### Backpropagation through queries, keys, and values

Use the same numerical example and define
$L=\tfrac12\|z_3-u\|^2$ with target $u=(0,2)$. This auxiliary loss tests the
mechanism; it is not a language-model objective. Its upstream gradient
$G=\partial L/\partial Z$ is zero in the first two rows and
$G_3\approx(1.758725,-1)$.

The matrix product $Z=AV$ gives

$$
D_V=A^\top G,\qquad D_A=GV^\top.
$$

Every value used by the third query receives a gradient. For example,
$(D_V)_1=a_{31}G_3\approx(0.436612,-0.248255)$. The first two queries have
no direct loss, but their positions still supply keys and values to $z_3$.

The derivatives of the loss with respect to the three weights in row 3 are
approximately $(1.758725,-2,4.276174)$. They cannot be used directly as
score derivatives: changing one score changes every softmax weight in the row.
For $S=QK^\top/\sqrt{d_k}$, the row-wise softmax derivative is

$$
\begin{gathered}
b_i=\sum_\ell a_{i\ell}(D_A)_{i\ell},\\
(D_S)_{ij}=a_{ij}\big((D_A)_{ij}-b_i\big).
\end{gathered}
$$

Here $b_3\approx2.093112$, so
$(D_S)_{32}\approx0.248255(-2-2.093112)=-1.016136$.
The full third row is approximately $(-0.083013,-1.016136,1.099149)$ and
sums to zero, consistently with softmax's invariance to a common score shift.

The chain rule then gives

$$
D_Q=D_SK/\sqrt{d_k},\qquad
D_K=D_S^\top Q/\sqrt{d_k}.
$$

Only $q_3$ gets a nonzero derivative here, approximately $(0.718517,0.058699)$;
all three keys get derivatives because all were read by that query.
For example, $(D_K)_1\approx(-0.058699,-0.058699)$.
Finally,

$$
\begin{gathered}
\nabla_{W_Q}L=X^\top D_Q,\\
\nabla_{W_K}L=X^\top D_K,\\
\nabla_{W_V}L=X^\top D_V,\\
\nabla_XL=D_QW_Q^\top+D_KW_K^\top+D_VW_V^\top .
\end{gathered}
$$

The three contributions to the input gradient add because the same $X$
enters all three projections. With $X=I_3$, the parameter derivatives equal
the corresponding projected-tensor derivatives; general inputs combine
contributions from several positions.

<a id="check-the-forward-pass-and-every-gradient"></a>

### Forward and backward passes in PyTorch

```python
import math
import torch

torch.set_num_threads(1)
dtype = torch.float64

def attention(q, k, v):
    # q: [..., Tq, dk], k: [..., Tk, dk], v: [..., Tk, dv]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    weights = scores.softmax(dim=-1)
    return weights @ v, weights

X = torch.eye(3, dtype=dtype, requires_grad=True)
WQ = torch.tensor([[1., 0.], [0., 1.], [1., 1.]],
                  dtype=dtype, requires_grad=True)
WK = WQ.detach().clone().requires_grad_()
WV = torch.tensor([[1., 0.], [0., 2.], [3., 1.]],
                  dtype=dtype, requires_grad=True)
Q, K, V = X @ WQ, X @ WK, X @ WV
Z, A = attention(Q, K, V)
target = torch.tensor([0., 2.], dtype=dtype)
loss = .5 * (Z[-1] - target).square().sum()
loss.backward()

with torch.no_grad():
    G = torch.zeros_like(Z)
    G[-1] = Z[-1] - target
    dV = A.T @ G
    dA = G @ V.T
    dS = A * (dA - (A * dA).sum(-1, keepdim=True))
    dQ = dS @ K / math.sqrt(Q.size(-1))
    dK = dS.T @ Q / math.sqrt(Q.size(-1))
    torch.testing.assert_close(WQ.grad, X.T @ dQ)
    torch.testing.assert_close(WK.grad, X.T @ dK)
    torch.testing.assert_close(WV.grad, X.T @ dV)
    torch.testing.assert_close(
        X.grad, dQ @ WQ.T + dK @ WK.T + dV @ WV.T)
    torch.testing.assert_close(A.sum(-1), torch.ones(3, dtype=dtype))

print("weights:", A.detach().tolist())
print("outputs:", Z.detach().tolist())
before = loss.item()
with torch.no_grad():
    for parameter in [WQ, WK, WV]:
        parameter -= .05 * parameter.grad
    updated, _ = attention(X @ WQ, X @ WK, X @ WV)
    after = (.5 * (updated[-1] - target).square().sum()).item()
print("one SGD step:", before, "->", after)
assert after < before
```

The assertions compare manual derivatives with autograd for all three projection
matrices and the shared input. Updating only the matrices with step size 0.05
reduces this particular loss from approximately 2.046556 to 1.838140.
Masks, attention dropout, multiple heads, and output projection are omitted.
Since $d_v=2$ and $d_{\mathrm{model}}=3$, residual addition would require a
projection back to the model width.

## Parallel execution and position order

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/self-attention-2026-10-21/cnn-rnn-self-attention.svg|Convolution reads nearby inputs; recurrence passes states along the sequence; full attention reads every input directly]]

*Zhang, Lipton, Li, and Smola, [Dive into Deep Learning, §11.6.2](https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html),
CC BY-SA 4.0. RNN states depend on earlier states; self-attention output rows
read input rows directly. The illustration compares connections, not measured
GPU speed.*

Every query is constructed from the current layer's input. Computing $z_3$
does not require the already computed $z_2$ of that layer. Known sequence
positions can therefore run in parallel, although generation still waits for
the token chosen at the preceding autoregressive step.

Dense attention costs $O(T^2d_k)$ for query–key scores and $O(T^2d_v)$ for
value mixing. Projections add $O(Td_{\mathrm{model}}(2d_k+d_v))$ work.
The direct implementation stores $T^2$ attention weights per head;
FlashAttention later computes the same operation without materializing that
matrix in HBM.

Full attention with no positional features is permutation equivariant.
For a permutation matrix $P$,

$$
\begin{gathered}
A(PX)=PA(X)P^\top,\\
Z(PX)=PZ(X).
\end{gathered}
$$

Permuting our input rows into order $(3,1,2)$ gives the old outputs in order
$(z_3,z_1,z_2)$. The output attached to a particular token does not acquire
new content merely because its row moved. This is insufficient to distinguish
the order-dependent roles in 'Alice follows Bob' and 'Bob follows Alice'.
The statement requires full attention without position features; a fixed
causal mask changes which predecessors a token can access. Positional
information and masking are developed in the following chapters.

<a id="common-confusions"></a>

## What an attention map shows

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/transformer_self-attention_visualization.png|Lines connect it to other positions in the encoder, including the animal]]

*Attention to the position `it`: lines connect it to source positions, with
line width showing attention weights. This is encoder self-attention, so
positions to the right are available too. Jay Alammar,
[The Illustrated Transformer, Self-Attention at a High Level](https://jalammar.github.io/illustrated-transformer/),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

The lines show the coefficients on the value vectors. A thick line
alone does not establish that removing that token would change the
model's prediction. Even for fixed values, different weights can
produce the same head output.

### Different weights, the same output

Let the values be $v_1=(1,0)$, $v_2=(0,1)$ and
$v_3=(0.5,0.5)$. The third value is the midpoint of the first two.
For scalar keys $K=(0,0,1)^{\top}$ and a scalar query $q$, the scores
are $(0,0,q)$, so

$$
a(q)=\frac{(1,1,e^q)}{2+e^q}.
$$

| Query | Attention weights | Output $a(q)V$ |
|---|---|---|
| $\ln2$ | $(0.25,0.25,0.50)$ | $(0.5,0.5)$ |
| $-\ln2$ | $(0.40,0.40,0.20)$ | $(0.5,0.5)$ |

Both rows are strictly positive softmax distributions from finite Q/K
scores. Moving weight from $v_3$ equally to $v_1$ and $v_2$ preserves
the mixture. Their difference $\delta=(0.15,0.15,-0.30)$ satisfies

$$
\delta V=0,\qquad \sum_j\delta_j=0.
$$

The code constructs both cases with shared projections from $X=I_3$;
only the third query changes. Here $d_k=1$, $d_v=2$.

```python
import math
import torch

dtype = torch.float64
X = torch.eye(3, dtype=dtype)
WK = torch.tensor([[0.], [0.], [1.]], dtype=dtype)
WV = torch.tensor([[1., 0.], [0., 1.], [.5, .5]], dtype=dtype)
K, V = X @ WK, X @ WV
rows, outputs = [], []
for query in [math.log(2), -math.log(2)]:
    WQ = torch.tensor([[0.], [0.], [query]], dtype=dtype)
    Q = X @ WQ
    A = (Q @ K.T).softmax(-1)  # sqrt(dk) = 1
    rows.append(A[-1])
    outputs.append((A @ V)[-1])

torch.testing.assert_close(torch.stack(rows), torch.tensor(
    [[.25, .25, .50], [.40, .40, .20]], dtype=dtype))
torch.testing.assert_close(torch.stack(outputs),
                           torch.full((2, 2), .5, dtype=dtype))
delta = rows[1] - rows[0]
torch.testing.assert_close(delta @ V, torch.zeros(2, dtype=dtype),
                           atol=1e-15, rtol=0)
print("different weights:", torch.stack(rows).tolist())
print("same output:", torch.stack(outputs).tolist())
```

The output alone cannot distinguish these weights. For fixed inputs and
parameters the head remains deterministic. This example does not establish
that every head admits arbitrary replacements of its attention matrix.
Brunner et al. analyse the corresponding null-space question in
[On Identifiability in Transformers, §3.1–3.2](https://arxiv.org/html/1908.04211v3#S3).
An attention map records mixing coefficients; interpreting it as causal
token importance requires additional evidence.

<details>
<summary>The attention head in Karpathy's lecture code</summary>

<a id="reading-the-lecture-implementation"></a>

The [lecture implementation of Head](https://github.com/karpathy/ng-video-lecture/blob/52201428ed7b46804849dea0b3ccf0de9df1a5c3/gpt.py#L61-L90)
adds a causal mask before softmax and applies dropout afterwards.
The lecture reaches that implementation in this order:

1. [bigram baseline](https://github.com/karpathy/ng-video-lecture/blob/master/bigram.py);
2. uniform averaging of earlier tokens with a lower-triangular matrix;
3. learned Q/K affinities;
4. Values and one head;
5. masking, multiple heads, and output projection;
6. [a complete minimal GPT](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py).

[Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
shows these operations within a GPT-2 block.

</details>

## Sources

- [[05 Источники/Courses/Harvard ML Systems/tinytorch/12_attention|TinyTorch 12 — Attention]] — scaled dot-product and multi-head attention, causal masking, and a check of quadratic context-length cost.
- [Vaswani et al., Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Karpathy, Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Karpathy, lecture code](https://github.com/karpathy/ng-video-lecture)
- [PyTorch, scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention)
- [3Blue1Brown, Attention in transformers](https://www.3blue1brown.com/lessons/attention)
- [Jay Alammar, The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — projection and matrix-operation diagrams, CC BY-NC-SA 4.0.
- [Dive into Deep Learning, Self-Attention and Positional Encoding](https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html) — computation paths and permutation equivariance, CC BY-SA 4.0.
- [Brunner et al., On Identifiability in Transformers](https://arxiv.org/html/1908.04211v3#S3) — limits of recovering attention weights from their outputs.
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
- [BertViz](https://github.com/jessevig/bertviz)

**Previous:** [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/01 От Seq2Seq к Transformer|Attention for translation]] ·
**Next:** [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/Masking, multi-head и формы тензоров|Masking, multi-head attention, and tensor shapes]]
