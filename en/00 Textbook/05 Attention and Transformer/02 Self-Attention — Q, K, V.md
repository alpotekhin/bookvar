---
title: Self-attention from the inside — Q, K, V
type: textbook-chapter
status: canonical
last_updated: 2026-08-03
primary_sources:
  - https://arxiv.org/abs/1706.03762
---

# Self-attention from the inside: Q, K, V

## Why a word must look at other words

An embedding table gives a token the same initial vector in every sentence.
Before the first layer, `mole` therefore has the same representation in a
discussion of chemical amount, an animal, and a skin mark. Contextual meaning
must arise inside the network: positions need a way to receive information from
other positions and decide which ones matter now.

This example is developed visually in [3Blue1Brown's attention
lesson](https://www.3blue1brown.com/lessons/attention). The purpose of the
operation is easiest to see by starting with simple averaging, then replacing
fixed weights with learned queries, keys, and values.

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

Compatibility between the query at $i$ and key at $j$ is

$$
s_{ij}=q_i^\top k_j.
$$

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

## The whole computation in two figures

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/attention/transformer_self_attention_vectors.png]]

*Jay Alammar, [The Illustrated
Transformer](https://jalammar.github.io/illustrated-transformer/): every token
vector is projected into Q, K, and V.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/attention/self-attention-matrix-calculation-2.png]]

*Jay Alammar, [The Illustrated
Transformer](https://jalammar.github.io/illustrated-transformer/): the matrix
form of the full computation. Read it from right to left: Values determine the
content being transmitted, while `softmax(QKᵀ/√dₖ)` supplies mixing weights.*

The complete formula is

$$
\operatorname{Attention}(Q,K,V)=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

Each symbol now has a concrete origin.

## 4. A numerical example by hand

Take three positions and one head with $d_k=2$. To expose the mechanics rather
than hide them behind projection matrices, set $Q=K=X$:

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
\alpha_3\approx[0.248,0.248,0.503].
$$

Rounding explains why the displayed values sum to 0.999.

### Step D: weighted sum of Values

$$
z_3=0.248[1,0]+0.248[0,2]+0.503[3,1]
\approx[1.758,0.999].
$$

The third position now has a contextual representation. It combines content
from all three Values, with roughly half the weight assigned to the third.

## 5. Why divide by $\sqrt{d_k}$

If components of $q$ and $k$ are independent with mean zero and variance one,
then

$$
\operatorname{Var}(q^\top k)=d_k.
$$

Dot-product magnitude grows with dimension. Softmax of large-magnitude scores
becomes nearly one-hot: changing one score slightly barely changes most output
weights, making optimization harder. Division by $\sqrt{d_k}$ restores a
typical scale near one.

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

The code deliberately omits masking and multiple heads. Queries select Values
through normalized compatibility with Keys. The next chapter adds access
constraints and a fourth dimension: head index.

## Common confusions

- Softmax runs over keys separately for each query.
- Attention weights sum to one across a row, but are not calibrated
  probabilities of “true importance.”
- Q, K, and V are learned projections, not renamed input embeddings.
- A position's output is a weighted sum of rows of $V$, not $Q$ or $K$.

## Karpathy laboratory route

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
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
- [BertViz](https://github.com/jessevig/bertviz)

**Previous:** [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/01 От Seq2Seq к Transformer|From seq2seq to Transformer]] ·
**Next:** [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/Masking, multi-head и формы тензоров|Masking, multi-head attention, and tensor shapes]]
