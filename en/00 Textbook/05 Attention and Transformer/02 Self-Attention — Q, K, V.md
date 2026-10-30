---
title: Self-attention from the inside — Q, K, V
type: textbook-chapter
status: canonical
locale: en
translation_status: current
canonical_revision: "270146832ba9d5f349939f802663643786563c05fd88bb0271633fcfce0baba7"
translation_reviewed: "2026-10-29"
translation_of: "00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V.md"
last_updated: 2026-10-29
primary_sources:
  - https://arxiv.org/abs/1706.03762
---

# Self-attention from the inside: Q, K, V

## Why a word must look at other words

An embedding table gives the same token ID the same initial vector in every sentence.
Before positional information is added, `mole` therefore has the same embedding in a
discussion of chemical amount, an animal, and a skin mark. Contextual meaning
must arise inside the network: positions need a way to receive information from
other positions and decide which ones matter now.

The word labels come from [3Blue1Brown's attention
lesson](https://www.3blue1brown.com/lessons/attention). A real tokenizer can split
a word into several tokens; the comparison assumes the same token ID at the
position being examined. Positional features can distinguish locations, but do
not by themselves tell the network which sense of `mole` the sentence uses.

Compare three phrases:

- `American shrew mole` — the animal;
- `one mole of carbon dioxide` — amount of substance;
- `a biopsy of the mole` — a skin mark.

The embedding table initially supplies the same vector for `mole`. It knows the
token identity, but not which sentence contains it. Attention refines that
vector with context.

For a long-range dependency, consider:

> **The cat that had been running all day was tired.**

To represent `was tired`, the model should connect the predicate to `cat`
despite the intervening phrase. If each word is processed independently, that
connection does not yet exist in the computation.

This describes representations of known input. When an autoregressive model
predicts `tired`, that token is not yet available: the preceding input position
must supply its prediction. Reading an already supplied target would change the
task. Self-attention means that queries, keys, and values originate from the
same sequence; whether future positions are visible depends on the mask.

Karpathy's [Let's build GPT from
scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) constructs this connection
step by step.

## 1. The simplest channel: average the past

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

This already creates context, but its weights are fixed. For `tired`, `cat` and
`running` should matter more than `the`, yet uniform averaging cannot know that.

## 2. Make weights depend on content

Each position needs two different roles:

- describe **what it currently needs**;
- announce **the features by which it can be selected**.

The model obtains two learned projections from every input vector:

$$
q_i=x_iW_Q,\qquad k_i=x_iW_K.
$$

Here $x_i$ is a $1\times d_{\mathrm{model}}$ row, $W_Q,W_K$ have shape
$d_{\mathrm{model}}\times d_k$, and $q_i,k_i$ are $1\times d_k$ rows.
Compatibility between the query at $i$ and key at $j$ is a scalar:

$$
s_{ij}=q_i k_j^\top=\sum_{a=1}^{d_k}q_{i,a}k_{j,a}.
$$

The same projection matrices act on every row. More tokens therefore do not
add projection parameters. For example, if the rows of $W_Q$ are $(1,0)$,
$(0,1)$, and $(1,1)$, input $x=(2,1,0)$ produces $q=(2,1)$ by weighting and
adding those rows. Since $W_Q$ and $W_K$ are separate, reading from position
$i$ to $j$ generally has a different score from reading in the opposite direction.

A large $s_{ij}$ does not mean that two words are generally similar. It means
that, in this layer and this head, position $j$ matches what position $i$ is
seeking.

:::note[Analogy and exact meaning]
`Query = question` and `Key = address` are useful mnemonics. Q and K do not
contain predetermined human roles; they are learned linear projections whose
meaning is determined by the loss and the rest of the network.
:::

## 3. Why Value is separate

The key participates in selection, but the vector transmitted to the recipient
is different:

$$
v_j=x_jW_V.
$$

The features used to find a position need not be the information that should be
copied from it. A database analogy is helpful: lookup uses an index, while the
query returns the record's contents.

$W_V$ has shape $[d_{\mathrm{model}},d_v]$; $d_v$ need not equal $d_k$.
Doubling $W_V$ while keeping $X,W_Q,W_K$ fixed doubles the values and attention
outputs, without changing any attention weight. The choice of where to read
and the content read from those positions have separate parameter paths.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/self-attention-2026-10-21/self-attention-output.png]]

*Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/),
CC BY-NC-SA 4.0. Follow projection, dot products, scaling, softmax, multiplication
by values, and summation. The illustration uses $d_k=64$, hence division by 8;
its numbers are separate from the worked example below.*

Before attention dropout, every nonempty row of weights is nonnegative and sums
to one. The output is therefore in the convex hull of its available values.
Residual addition and the FFN are later operations, not part of this statement.
Attention softmax normalizes source positions; the language model's output
softmax normalizes vocabulary entries. An attention output is a feature vector,
not a next-token distribution.

## The whole computation in two figures

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/attention/transformer_self_attention_vectors.png]]

*Jay Alammar, [The Illustrated
Transformer](https://jalammar.github.io/illustrated-transformer/): every token
vector is projected into Q, K, and V. CC BY-NC-SA 4.0.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/attention/self-attention-matrix-calculation-2.png]]

*Jay Alammar, [The Illustrated
Transformer](https://jalammar.github.io/illustrated-transformer/): the matrix
form of the full computation. Read it from right to left: Values determine the
content being transmitted, while `softmax(QKᵀ/√dₖ)` supplies mixing weights.
CC BY-NC-SA 4.0.*

The complete formula is

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

## 4. A numerical example by hand

Take three positions with $d_{\mathrm{model}}=3$ and $d_k=d_v=2$.
Set $X=I_3$ and choose separate projection matrices $W_Q,W_K,W_V$ whose rows
are the matrices displayed below. Then $Q=XW_Q$, $K=XW_K$, and $V=XW_V$:

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

Consider the third-position query $q_3=[1,1]$.

The input dimension matters. If we had instead set $X=Q=K$ above, then
$x_3=x_1+x_2$, so every shared linear projection would require
$v_3=v_1+v_2$. The displayed values violate that constraint. Choosing the
independent input rows of $I_3$ makes all three projections realizable. Equal
initial values for $W_Q$ and $W_K$ do not make them one shared parameter.

### Step A: dot products

$$
q_3K^\top=[1,1,2].
$$

### Step B: scaling

$$
\frac{q_3K^\top}{\sqrt2}\approx[0.707,0.707,1.414].
$$

### Step C: softmax

$$
\alpha_3\approx[0.248255,0.248255,0.503490].
$$

The weights are normalized across the three keys; the displayed rounding sums to one.
Subtracting the common score $1/\sqrt2$ before exponentiation gives weights
proportional to $(1,1,e^{1/\sqrt2})$. Dividing by $2+e^{1/\sqrt2}$ yields
the displayed row.

### Step D: weighted sum of Values

$$
\begin{aligned}
z_3={}&0.248255[1,0]+0.248255[0,2]\\
&+0.503490[3,1]\\
\approx{}&[1.758725,1].
\end{aligned}
$$

The third position now has a contextual representation. It combines content
from all three Values, with roughly half the weight assigned to the third.

| Query | Weight of $v_1$ | Weight of $v_2$ | Weight of $v_3$ |
|---|---:|---:|---:|
| $q_1$ | 0.401112 | 0.197776 | 0.401112 |
| $q_2$ | 0.197776 | 0.401112 | 0.401112 |
| $q_3$ | 0.248255 | 0.248255 | 0.503490 |

Even with $Q=K$, $a_{13}$ and $a_{31}$ differ: symmetric dot products are
normalized by different row denominators.

## 5. Why divide by $\sqrt{d_k}$

If components of $q$ and $k$ are independent with mean zero and variance one,
then

$$
\operatorname{Var}(q k^\top)=d_k.
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

## 6. Minimal implementation of one head

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

### Check the forward pass and every gradient

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

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/self-attention-2026-10-21/cnn-rnn-self-attention.svg]]

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

## Common confusions

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/transformer_self-attention_visualization.png]]

*Attention to the position `it`: lines connect it to source positions, with
line width showing attention weights. This is encoder self-attention, so
positions to the right are available too. Jay Alammar,
[The Illustrated Transformer, Self-Attention at a High Level](https://jalammar.github.io/illustrated-transformer/),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

- Softmax runs over keys separately for each query.
- Before attention dropout, attention weights sum to one across a row, but are not calibrated
  probabilities of “true importance.”
- Q, K, and V are learned projections, not renamed input embeddings.
- A position's output is a weighted sum of rows of $V$, not $Q$ or $K$.

## Different weights, the same output

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

## Reading the lecture implementation

The most useful practical sequence is:

1. [bigram baseline](https://github.com/karpathy/ng-video-lecture/blob/master/bigram.py);
2. uniform averaging of earlier tokens with a lower-triangular matrix;
3. learned Q/K affinities;
4. Values and one head;
5. masking, multiple heads, and output projection;
6. [a complete minimal GPT](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py).

Then open [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
and identify the same stages inside a real GPT-2.

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

**Previous:** [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/01 От Seq2Seq к Transformer|From seq2seq to Transformer]] ·
**Next:** [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/Masking, multi-head и формы тензоров|Masking, multi-head attention, and tensor shapes]]
