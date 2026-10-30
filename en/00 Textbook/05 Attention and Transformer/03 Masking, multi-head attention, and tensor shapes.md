---
title: Masking, multi-head attention, and tensor shapes
type: textbook-chapter
status: canonical
locale: en
translation_status: current
canonical_revision: "9306be89c604c905489000ffcebffb32e16161f66e07540005d0f394097c0fd5"
translation_reviewed: "2026-10-29"
translation_of: "00 Учебник/05 Attention и Transformer/Masking, multi-head и формы тензоров.md"
last_updated: 2026-10-29
source_unit_id:
  - lecture-02-einops-named-axes
  - assignment-01-task-scaled-dot-product-attention
  - assignment-01-task-multihead-self-attention
previous: "[[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V]]"
next: "[[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/04 Позиционная информация]]"
primary_sources:
  - https://arxiv.org/abs/1706.03762
  - https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html
  - https://jalammar.github.io/illustrated-transformer/
  - https://d2l.ai/chapter_attention-mechanisms-and-transformers/multihead-attention.html
  - https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_02.py
  - https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf
---

# Masks, multiple heads, and tensor shapes

Suppose a batch contains a two-token sentence and a four-token sentence.
The shorter sentence is padded to the common length, but its real tokens must
not read the fillers. Language-model training must also prevent access to
future positions, even though the whole training sequence is already in memory.

A mask specifies which query–key pairs are allowed. Multiple heads assign
different scores to these pairs and read different projections of the values.
The tensor therefore needs separate batch, head, query, and key axes.
Their order determines which rows are multiplied.

## A mask changes the set of accessible keys

Let $S=QK^\top/\sqrt{D_h}$ be the score matrix. Add the mask to $S$ before softmax:

$$
A=\operatorname{softmax}(S+M),\qquad Z=AV.
$$

An allowed cell has $M_{ij}=0$; a forbidden cell has $M_{ij}=-\infty$.
Exponentiation sends a forbidden cell to zero, excluding it from both the
weighted sum and the normalization of the remaining weights. Zeroing an
already normalized matrix $A$ is insufficient: the allowed weights no longer
sum to one unless they are normalized again.

For example, one query has scores $(2,1,0)$ and cannot access the third key.
Ordinary softmax gives approximately $(0.665241,0.244728,0.090031)$.
Removing the third weight leaves a sum of $0.909969$.
Removing the third key before softmax instead gives a denominator of
$e^2+e^1$ and weights $(0.731059,0.268941,0)$.

For values $(10,20,100)$, the correct output is approximately $12.689414$.
The zeroed but unnormalized row produces $11.546979$: the forbidden key
has already affected the shares assigned to the allowed values.

```python
import torch

scores = torch.tensor([2., 1., 0.], dtype=torch.float64)
allowed = torch.tensor([True, True, False])
values = torch.tensor([10., 20., 100.], dtype=torch.float64)
late = scores.softmax(0) * allowed
early = scores.masked_fill(~allowed, -torch.inf).softmax(0)
print("after softmax:", late.tolist(), "sum:", late.sum().item())
print("before softmax:", early.tolist(), "sum:", early.sum().item())
print("outputs:", (late @ values).item(), (early @ values).item())

# A large forbidden score can underflow allowed weights to zero.
extreme = torch.tensor([-1000., -1001., 1000.], dtype=torch.float64)
late_extreme = extreme.softmax(0) * allowed
early_extreme = extreme.masked_fill(~allowed, -torch.inf).softmax(0)
assert late_extreme.sum() == 0
torch.testing.assert_close(early_extreme, early)
```

In exact arithmetic, renormalizing after removing the weights would recover
the answer whenever their sum is nonzero. In the `extreme` example,
underflow has already erased the allowed exponentials. Dividing these zeros
by their zero sum cannot recover the result.

### Causal masking and shifted targets

During language-model training, the complete sequence is known and all
positions are processed in parallel. But the state at position $i$, from which
the model predicts the next token, may depend only on positions $j\le i$.
Otherwise the network can read the correct continuation from its input.
A lower-triangular mask enforces the required dependency:

$$
M^{\text{causal}}_{ij}=
\begin{cases}
0,&j\le i,\\
-\infty,&j>i.
\end{cases}
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-15-24/causal-mask-detailed.png]]

*The diagram follows the second token's query along the red path. It is
compared with the first and second keys; future keys do not participate.
Image by dvgodoy, [original and CC BY 4.0 license](https://commons.wikimedia.org/wiki/File:Decoder_self-attention_with_causal_masking,_detailed_diagram.png).*

Lena Voita shows the same operation in an
[animation of masked self-attention](https://lena-voita.github.io/resources/lectures/seq2seq/transformer/masked_self_attn.mp4):
moving along the target sequence adds one position to the accessible
lower-triangular region.

For the sequence A, B, EOS, the shifted inputs and prediction targets are:

| Query position | Known input | Prediction target | Accessible inputs |
|---|---|---|---|
| 0 | BOS | A | BOS |
| 1 | A | B | BOS, A |
| 2 | B | EOS | BOS, A, B |

The diagonal is allowed: A at input position 1 is already known, whereas
that position's target is the unread B. Masking and target shifting work
together. Feeding the targets themselves as inputs would leak the answer
along the diagonal even with a lower-triangular mask.

Masking does not make training sequential. All rows of $S$ can still be
computed in one matrix multiplication; forbidden connections are removed
before softmax. Generation remains sequential because the next input token
is not yet known.

### Padding masks: exclude filler keys

Sentences of different lengths are commonly padded with a special token
to a shared length $T$. These positions provide a rectangular tensor, not
additional text. If only five of seven positions are real, the key mask is
`[1,1,1,1,1,0,0]`: every real query may read the first five keys but not
the two fillers.

A causal mask depends on the pair `(query position, key position)` and
has shape `[T,T]`. A padding mask depends on the example and key position,
so its natural shape is `[B,T]`. A decoder-only model needs both:

$$
M_{b,i,j}=M^{\text{causal}}_{i,j}+M^{\text{padding}}_{b,j}.
$$

The key positions must be masked. An output from a padding query can also
be zeroed later, but that separate operation does not prevent real tokens
from reading filler keys.

## Why one position needs multiple heads

Let $X_Q,X_K,X_V$ denote the inputs before projection. In self-attention
all three are the same $X$. In cross-attention, $X_Q$ comes from the
decoder and $X_K=X_V$ from the encoder's final output. The projected
$Q,K,V$ below retain the notation of the preceding chapter.

One softmax produces one distribution over keys. A word may need several
relationships at once: a nearby word, the subject with which it agrees,
and a modifier that disambiguates its meaning. Multi-head attention provides
separate projection sets and separate attention distributions. These roles
are not assigned to heads in advance; several heads can learn similar
distributions. All head parameters are trained jointly under the same loss:

$$
\begin{aligned}
Q_r&=X_QW_r^Q,\\
K_r&=X_KW_r^K,\\
V_r&=X_VW_r^V,\\
\operatorname{head}_r&=\operatorname{Attention}(Q_r,K_r,V_r).
\end{aligned}
$$

$$
\begin{gathered}
\operatorname{MHA}(X_Q,X_K,X_V)\\
=\operatorname{Concat}(\operatorname{head}_1,\ldots,
\operatorname{head}_H)W^O.
\end{gathered}
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-15-24/qkv-for-heads.png]]

*Each head receives its own query, key, and value projections. Source:
Lena Voita, [Multi-Head Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html#multi_head_attention).
The original page also includes an animation of the complete computation.*

The classic Transformer chooses $d_k=d_v=D_h=D_{model}/H$.
Each head's projection is therefore narrower than the model, and
concatenation has width $HD_h=D_{model}$. This is a dimensionality choice,
not a requirement of every multi-head attention layer. In the general case,
$W^O$ has shape $[Hd_v,D_{model}]$.

The output projection mixes features from different heads at one position.
It neither mixes rows belonging to different queries nor replaces attention
between them. These formulas use row vectors. PyTorch `Linear` stores
weights for the operation $xW^\top$; matrices written in our notation must
be transposed when loaded into the layer.

<a id="named-attention-shapes"></a>

## Split and merge: one operation, two notations

The input axes are `batch`, `sequence`, and `model_dim`.
After projection, the last axis contains consecutive features from all heads.
Split it into `heads` and `head_dim`, then move the head axis before the
sequence axis so that the final two axes form a matrix.

[CS336 Lecture 2](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_02.py#L202-L276)
uses `rearrange` to split and merge heads and `einsum` for contractions
over named axes. We first write the transformation with `reshape` and
`transpose`, then compare it with the named-axis notation.

Check $D_{model}=HD_h$ and the lengths of matching axes before multiplication.
Broadcasting can repeat a length-one axis, but it cannot determine whether
an axis represents examples, heads, or queries.

To compute all heads' queries in one multiplication, concatenate their
$W_r^Q$ matrices along the output axis into $W^Q$. The result has shape
`[B,T,Dmodel]`, with the first head's features before the second head's
features, and so on. Compute keys and values similarly using $W^K$ and
$W^V$, then split each projected tensor's final axis into head index and
within-head coordinates.

Self-attention can combine the three projection types as well: one
multiplication produces `[B,T,3*Dmodel]`, which is split into Q, K, and V.
The implementation below keeps three separate `Linear` layers so their
parameters and computation paths remain visible.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-15-24/transformer_attention_heads_qkv.png]]

*Jay Alammar illustrates the separate projections producing each head's
queries, keys, and values. Source: [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/),
CC BY-NC-SA 4.0.*

Take $B=2,T=3,D_{model}=4,H=2,D_h=2$. The first example has projected
rows $(0,1,2,3)$, $(4,5,6,7)$, and $(8,9,10,11)$.
Head 0 must receive $(0,1),(4,5),(8,9)$, while head 1 receives
$(2,3),(6,7),(10,11)$.

```python
import torch

B, T, Dmodel, H = 2, 3, 4, 2
Dh = Dmodel // H
q = torch.arange(B*T*Dmodel).reshape(B, T, Dmodel)
split = q.reshape(B, T, H, Dh).transpose(1, 2)
wrong = q.reshape(B, H, T, Dh)
print("head 0:", split[0, 0].tolist())
print("wrong head 0:", wrong[0, 0].tolist())
assert split.shape == wrong.shape
assert not torch.equal(split, wrong)

merged = split.transpose(1, 2).contiguous().view(B, T, Dmodel)
torch.testing.assert_close(merged, q)
```

Directly calling `reshape(B,H,T,Dh)` gives the right dimensions but
the wrong pairs: its first head would contain $(0,1),(2,3),(4,5)$.
A shape-only assertion cannot detect this error.

`transpose` changes memory strides rather than moving elements into a
new contiguous allocation. The reverse permutation is therefore followed
by `contiguous().view(...)`. A `reshape(...)` can make the copy itself
when needed. Some special dimensions permit `view` without a copy,
but general code should not rely on that coincidence.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-15-24/transformer_attention_heads_z.png]]

*Head results stay separate until concatenation. Source: Jay Alammar,
[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/),
CC BY-NC-SA 4.0.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-15-24/transformer_attention_heads_weight_matrix_o.png]]

*After concatenation, $W^O$ mixes head outputs and returns width
$D_{model}$. Source: Jay Alammar,
[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/),
CC BY-NC-SA 4.0.*

<a id="einops-attention-shapes"></a>

### Named axes: rearrange, einsum, and reduce

In `q @ k.transpose(-2,-1)`, the meaning of each axis must be recovered
from surrounding code. The `einops` library puts that meaning into the
operation. Splitting heads uses
`"batch token (head coord) -> batch head token coord"`.
Parentheses on the left say that the final axis combines head index and
within-head coordinate. With `head=2`, the first axis has length two
and the second has length `Dmodel / 2`. The right side specifies the
new axis order. One `rearrange` thus expresses both steps above:
splitting the last axis and exchanging the token and head axes.

The order inside parentheses determines the coordinate order.
For `(head coord)`, the coordinate changes first, then the head index,
so each head's features are consecutive. Using `(coord head)` would
select interleaved features instead, even if the output shape were identical.
The inverse pattern
`"batch head token coord -> batch token (head coord)"`
merges the heads in their original order. Like `reshape`, `rearrange`
may copy data when necessary; the notation promises neither a separate
GPU kernel nor a speedup. The [rearrange documentation](https://einops.rocks/api/rearrange/)
describes ordering and view semantics.

Score calculation preserves the example and head axes, enumerates all
query–key pairs, and sums products over coordinates:

$$
S_{b,h,i,j}=\sum_{c=0}^{D_h-1}Q_{b,h,i,c}K_{b,h,j,c}.
$$

In `einops.einsum`, this is
`"batch head query coord, batch head key coord -> batch head query key"`.
The names on either side of the comma describe the two inputs.
The shared `coord` axis is absent from the output, so it is summed out.
`batch` and `head` remain: they select a pair of matrices rather than
being reduced. Different `query` and `key` names create all position
pairs, and their lengths may differ. The order of names after the arrow
determines the output axis order. In the [einops.einsum API](https://einops.rocks/api/einsum/),
the tensors come first and the pattern comes last.

For one input, `reduce` specifies the reduction explicitly.
The pattern `"batch head token coord -> batch head token"` with
`"sum"` adds each row's coordinates, retaining a result for every
example, head, and token. Applied to squared coordinates, it returns
squared Euclidean lengths. Rows $(0,1)$, $(4,5)$, and $(8,9)$ yield
$1$, $41$, and $145$. This sum mixes neither heads nor positions
and is not attention normalization. Other operations supported by
[reduce](https://einops.rocks/api/reduce/), such as `"mean"` or `"max"`,
produce different results.

The following block is self-contained. It requires PyTorch and `einops`;
the canonical example was checked with versions 2.14.1 and 0.8.1.
Values 0–23 make it possible to track individual elements rather than
only checking dimensions:

```python
import torch
from einops import rearrange, einsum, reduce

projected = torch.arange(24., dtype=torch.float64).reshape(2, 3, 4)
q_heads = rearrange(
    projected, "batch token (head coord) -> batch head token coord", head=2)
k_heads = rearrange(
    projected.flip(1),
    "batch token (head coord) -> batch head token coord", head=2)
torch.testing.assert_close(
    q_heads, projected.reshape(2, 3, 2, 2).transpose(1, 2))

named_scores = einsum(
    q_heads, k_heads,
    "batch head query coord, batch head key coord -> batch head query key")
torch.testing.assert_close(
    named_scores, q_heads @ k_heads.transpose(-2, -1))
assert named_scores[0, 0, 0].tolist() == [9., 5., 1.]

norm_sq = reduce(
    q_heads.square(), "batch head token coord -> batch head token", "sum")
torch.testing.assert_close(norm_sq, q_heads.square().sum(-1))
assert norm_sq[0, 0].tolist() == [1., 41., 145.]

restored = rearrange(
    q_heads, "batch head token coord -> batch token (head coord)")
torch.testing.assert_close(restored, projected)
print("head 0:", q_heads[0, 0].tolist())
print("scores of first query:", named_scores[0, 0, 0].tolist())
print("squared row norms:", norm_sq[0, 0].tolist())
```

The key positions are reversed: their first head has rows $(8,9)$,
$(4,5)$, then $(0,1)$. Query $(0,1)$ therefore receives scores
$0\cdot8+1\cdot9=9$, followed by $5$ and $1$. These are not yet
attention weights: scaling, masking, and softmax follow this product.

Axis names exist in the pattern, not on the tensor itself.
If equal-length axes have already been swapped, the library cannot know
that the data have changed meaning. With $B=H=2$, swapping examples
and heads preserves the shape. Checks on individual values remain necessary.
An ellipsis may stand for several leading axes, as in
`"... query coord, ... key coord -> ... query key"`, but the caller
still determines the meaning of those hidden axes.

[CS336 Lecture 2, einops_einsum, einops_reduce, and einops_rearrange](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_02.py#L202-L276)
works through this sequence of `einsum`, `reduce`, and `rearrange`.

### A numerical example with two heads

Let the first head's values be $(0,8)$ and the second head's values be
$(4,12)$. For one query, assign weights $(0.75,0.25)$ to head 1 and
$(0.25,0.75)$ to head 2. Then:

$$
\begin{gathered}
z_1=0.75\cdot0+0.25\cdot8=2,\\
z_2=0.25\cdot4+0.75\cdot12=10.
\end{gathered}
$$

Scalar queries equal to one, with scores $(\log3,0)$ and $(0,\log3)$,
produce these weights. The complete implementation below supplies
projection matrices producing precisely these queries, keys, and values.

Concatenation gives the row $(2,10)$. Choose the output matrix:

$$
W^O=\begin{bmatrix}1&2\\3&-1\end{bmatrix}.
$$

Then $(2,10)W^O=(32,-6)$. Averaging the heads would produce a single
number, $6$, losing the distinction between $(2,10)$ and $(6,6)$.
Yet $(6,6)W^O=(24,6)$. That one average cannot recover both projected
outputs.

## Complete shape trace

Let the batch contain $B$ sequences of length $T$, with model width
$D_{model}$, $H$ heads, and head width $D_h=D_{model}/H$.

| Step | General shape | For $B=2,T=3,H=2,D_h=2$ |
|---|---|---|
| input $X$ | `[B,T,Dmodel]` | `[2,3,4]` |
| three projections $Q,K,V$ | each `[B,T,Dmodel]` | each `[2,3,4]` |
| split + transpose | each `[B,H,T,Dh]` | each `[2,2,3,2]` |
| $K^\top$ | `[B,H,Dh,T]` | `[2,2,2,3]` |
| scores $S$ and weights $A$ | each `[B,H,T,T]` | each `[2,2,3,3]` |
| values $Z=AV$ | `[B,H,T,Dh]` | `[2,2,3,2]` |
| merge and output projection | `[B,T,Dmodel]` | `[2,3,4]` |

This example's score matrix contains 36 numbers: two examples, two heads,
and nine position pairs. Keeping $D_{model}=4$ but using four heads of
width one gives shape `[2,4,3,3]`, with 72 numbers. Each head stores
its own distribution over the same position pairs even when its queries
and keys have fewer features.

The four projections still contain $4D_{model}^2=64$ weights without bias.
The $QK^\top$ product requires
$BHT^2D_h=BT^2D_{model}=72$ multiply-accumulate operations with either
two or four heads: head count is offset by head width. The $AV$ product
requires the same number under our choice $d_v=D_h$. But softmax now
processes 72 elements rather than 36. Equal parameter counts and matrix
operation counts therefore do not imply equal execution time.

The counts of 36 and 72 apply to each explicitly stored $S$ or $A$
matrix. Fused algorithms such as FlashAttention can compute the result
in blocks without storing the whole matrix. Their memory use cannot
be inferred from the table alone.

Softmax runs along the last axis: for every `(b,h,i)`, weights sum
to one over keys $j$ when the allowed row is nonempty. Empty rows follow
the separate policy below. Normalizing over queries instead would make
columns sum to one, allocating shares among different queries for one
key. A query row would no longer define a normalized mixture of values.

## Broadcasting masks

After splitting heads, scores have shape `[B,H,Tq,Tk]`.
A mask need not physically store all these elements: length-one axes
broadcast.

| Purpose | Convenient shape | Broadcast axes |
|---|---|---|
| a common causal mask | `[1,1,Tq,Tk]` | batch and heads |
| each example's key padding | `[B,1,1,Tk]` | heads and queries |
| exclude padding queries | `[B,1,Tq,1]` | heads and keys |
| each example's full mask | `[B,1,Tq,Tk]` | heads |

For example, query position 3 of the second example may read key 1 only
if the causal rule permits it, key 1 is real, and the query itself is
not padding. In a full-sequence self-attention pass, query and key lengths are equal;
cross-attention requires them to be specified separately.

### Empty rows and Boolean-mask polarity

With left padding `[0,0,1,1]`, the first two queries may have no
allowed keys. Softmax of a row containing only $-\infty$ is undefined.
Replacing a resulting `NaN` with zero does not guarantee a correct
backward pass through the preceding softmax.

The implementation below substitutes finite placeholder scores for
empty rows before softmax, then replaces their weights with zeros.
The resulting $AV$ is zero and forbidden paths carry no gradient.
Nonempty rows retain ordinary masked softmax. This assumes finite
Q/K/V inputs: a mask does not repair arbitrary `NaN` values inside V.

For `[0,0,1,1]`, row sums are `[0,0,1,1]`. The first real token
reads itself; the second reads both real tokens. An output-projection
bias or residual connection can make padding states nonzero again.
Masking prediction targets in the loss is a separate operation.

The same Boolean `True` has opposite meanings in different interfaces:

| PyTorch 2.14 interface | Meaning of `True` |
|---|---|
| `F.scaled_dot_product_attention(attn_mask=...)` | the pair participates in attention |
| `nn.MultiheadAttention(attn_mask=...)` | the pair is forbidden |
| `nn.MultiheadAttention(key_padding_mask=...)` | the key is padding |

These are current interfaces, not a distinction between “new” and “old”
PyTorch. Their contracts are specified in the versioned
[SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
and [MultiheadAttention](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MultiheadAttention.html)
documentation. Unlike a Boolean mask, a floating-point SDPA mask is
added to the scores.

### Rectangular masks and KV-cache positions

Two queries and five source keys produce scores of shape `[B,H,2,5]`.
This is common in cross-attention, where a causal mask comparing target
and source indices is usually unnecessary: the entire source is known.

A rectangular shape has a different meaning during cached generation.
Suppose one query represents absolute position 3, with keys at positions
0, 1, 2, and 3. All four keys are allowed. An ordinary `tril` of shape
`[1,4]` would allow only the first key because it compares local row
and column indices.

Build the access rule from absolute positions instead. In PyTorch 2.14,
rectangular SDPA with `is_causal=True` uses upper-left alignment.
It does not tell the function the absolute position of a single query.
Equal scores and values $(1,2,3,4)$ expose the difference:

```python
import torch
import torch.nn.functional as F

q = torch.zeros(1, 1, 1, 2, dtype=torch.float64)
k = torch.zeros(1, 1, 4, 2, dtype=torch.float64)
v = torch.arange(1., 5., dtype=torch.float64).reshape(1, 1, 4, 1)
query_positions = torch.tensor([3])
key_positions = torch.arange(4)
allowed = key_positions[None, :] <= query_positions[:, None]

cached = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed,
                                        dropout_p=0.)
upper_left = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                            dropout_p=0.)
print("absolute-position mask:", cached.item())
print("upper-left causal mask:", upper_left.item())
torch.testing.assert_close(cached, torch.full_like(cached, 2.5))
torch.testing.assert_close(upper_left, torch.ones_like(upper_left))
```

Here $2.5$ averages all four allowed values; $1.0$ reads only the first.

Now suppose positions 0–2 are already cached and one call processes
two new tokens at positions 3 and 4. Their keys are appended to the
three old keys. The first query must not read position 4; the second
may read all five positions. The Boolean mask must be:

$$
\operatorname{allowed}=\begin{bmatrix}
1&1&1&1&0\\
1&1&1&1&1
\end{bmatrix}.
$$

With zero Q and K, all allowed scores are equal. For values $(1,2,3,4,5)$,
the first row gives $(1+2+3+4)/4=2.5$ and the second gives
$(1+2+3+4+5)/5=3$. An upper-left triangle would allow only key 0
in the first row and keys 0 and 1 in the second, producing $1$ and $1.5$.
Removing the mask entirely would produce $3$ in both rows; the first
query would read the future value $5$. The following example checks
all three cases:

```python
import torch
import torch.nn.functional as F

q_block = torch.zeros(1, 1, 2, 2, dtype=torch.float64)
k_block = torch.zeros(1, 1, 5, 2, dtype=torch.float64)
v_block = torch.arange(1., 6., dtype=torch.float64).reshape(1, 1, 5, 1)
query_positions = torch.tensor([3, 4])
key_positions = torch.arange(5)
block_allowed = key_positions[None, :] <= query_positions[:, None]
block_cached = F.scaled_dot_product_attention(
    q_block, k_block, v_block, attn_mask=block_allowed, dropout_p=0.)
block_upper_left = F.scaled_dot_product_attention(
    q_block, k_block, v_block, is_causal=True, dropout_p=0.)
block_unmasked = F.scaled_dot_product_attention(
    q_block, k_block, v_block, dropout_p=0.)
print("two-query absolute mask:", block_allowed.int().tolist())
print("two-query outputs:", block_cached.flatten().tolist())
torch.testing.assert_close(
    block_cached.flatten(), torch.tensor([2.5, 3.], dtype=torch.float64))
torch.testing.assert_close(
    block_upper_left.flatten(), torch.tensor([1., 1.5], dtype=torch.float64))
torch.testing.assert_close(
    block_unmasked.flatten(), torch.tensor([3., 3.], dtype=torch.float64))
```

Here the queries occupy the final two positions of a contiguous set
of keys. The mask can also be formed with a lower triangle offset
by $T_k-T_q=3$, as provided by
[causal_lower_right](https://docs.pytorch.org/docs/2.14/generated/torch.nn.attention.bias.causal_lower_right.html).
For arbitrary positions, determine access from their actual values.
Padding and packed independent texts also require token-validity and
same-document constraints. The inference section covers KV-cache storage.

SDPA applies dropout according to `dropout_p` even when the calling
module is in `eval()` mode. Evaluation must explicitly pass zero.
All numerical checks below disable dropout.

## Plausible-looking implementation errors

- **Softmax along the wrong axis.** `weights.sum(-1)` should contain
  ones for rows with at least one allowed key.
- **Masking after softmax.** Forbidden weights become zero, but the
  remaining weights are not renormalized.
- **Confusing $T_q$ with $T_k$.** They may differ in cross-attention;
  scores always have shape `[B,H,Tq,Tk]`.
- **Incorrect broadcasting that succeeds.** A `[B,T]` mask aligns
  with the final two axes of `[B,H,T,T]`. When $B=T$, this may run
  without error but does not represent each example's key mask.
  That requires `[B,1,1,T]`.
- **An entirely masked row.** Direct `torch.softmax` over only
  $-\infty$ produces `NaN`. Such rows require an explicit policy,
  particularly with left padding.
- **Forgetting to merge heads.** Averaging heads is not concatenation
  followed by a learned $W^O$.
- **Using the wrong scale.** Divide by $\sqrt{D_h}$, not
  $\sqrt{D_{model}}$.

<a id="shape-evidence"></a>

## A complete implementation check

```python
import math
import torch
from torch import nn

def masked_attention(q, k, v, allowed):
    # Final axes: [Tq,dk], [Tk,dk], [Tk,dv].
    if allowed.dtype != torch.bool:
        raise ValueError("allowed must be Boolean; True means allowed")
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    allowed = torch.broadcast_to(allowed, scores.shape)
    nonempty = allowed.any(-1, keepdim=True)
    masked = scores.masked_fill(~allowed, -torch.inf)
    safe = torch.where(nonempty, masked, torch.zeros_like(masked))
    weights = safe.softmax(-1)
    weights = torch.where(nonempty, weights, torch.zeros_like(weights))
    return weights @ v, weights

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads:
            raise ValueError("positive d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x):
        B, T, _ = x.shape
        return x.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, valid=None, causal=True):
        if x.ndim != 3 or x.size(-1) != self.d_model:
            raise ValueError("x must have shape [B,T,d_model]")
        B, T, _ = x.shape
        if valid is None:
            valid = torch.ones(B, T, dtype=torch.bool, device=x.device)
        if valid.dtype != torch.bool or valid.shape != x.shape[:2]:
            raise ValueError("valid must be a Boolean [B,T] mask")
        valid = valid.to(x.device)
        q = self.split_heads(self.q_proj(x))
        k = self.split_heads(self.k_proj(x))
        v = self.split_heads(self.v_proj(x))
        allowed = valid[:, None, None, :] & valid[:, None, :, None]
        if causal:
            positions = torch.arange(T, device=x.device)
            causal_mask = positions[None, :] <= positions[:, None]
            allowed = allowed & causal_mask[None, None, :, :]
        z, weights = masked_attention(q, k, v, allowed)
        merged = z.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(merged), weights

torch.set_num_threads(1)
dtype = torch.float64
toy = MultiHeadSelfAttention(2, 2).to(dtype)
WQ = torch.ones(2, 2, dtype=dtype)
WK = torch.eye(2, dtype=dtype) * math.log(3)
WV = torch.tensor([[0., 4.], [8., 12.]], dtype=dtype)
WO = torch.tensor([[1., 2.], [3., -1.]], dtype=dtype)
with torch.no_grad():
    for layer, matrix in [(toy.q_proj, WQ), (toy.k_proj, WK),
                          (toy.v_proj, WV), (toy.out_proj, WO)]:
        layer.weight.copy_(matrix.T)

toy_out, toy_weights = toy(torch.eye(2, dtype=dtype)[None], causal=False)
print("two head rows:", toy_weights[0, :, 0].detach().tolist())
print("projected output:", toy_out[0, 0].detach().tolist())
torch.testing.assert_close(
    toy_weights[0, :, 0],
    torch.tensor([[.75, .25], [.25, .75]], dtype=dtype))
torch.testing.assert_close(
    toy_out, torch.tensor([[[32., -6.], [32., -6.]]], dtype=dtype))

torch.manual_seed(11)
model = MultiHeadSelfAttention(4, 2).to(dtype)
x = torch.randn(3, 4, 4, dtype=dtype, requires_grad=True)
valid = torch.tensor([[False, False, True, True],
                      [True, True, True, True],
                      [False, False, False, False]])
out, weights = model(x, valid)
assert out.shape == (3, 4, 4)
assert weights.shape == (3, 2, 4, 4)
expected_sums = valid[:, None, :].expand_as(weights.sum(-1)).to(dtype)
torch.testing.assert_close(weights.sum(-1), expected_sums)
assert torch.count_nonzero(weights.triu(diagonal=1)) == 0
assert torch.count_nonzero(out[~valid]) == 0
assert torch.isfinite(out).all()
print("left-padding row sums:", weights[0, 0].sum(-1).detach().tolist())

loss = out[valid].square().mean()
loss.backward()
assert torch.isfinite(x.grad).all()
assert torch.count_nonzero(x.grad[~valid]) == 0
for parameter in model.parameters():
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()
```

The toy example's two weight rows are $(0.75,0.25)$ and $(0.25,0.75)$,
and the first position's output is $(32,-6)$. Causal masking is disabled
for this run, so both heads can read both known positions.

The second run checks causal attention in a batch containing a
left-padded example, ordinary text, and an entirely empty example.
The first example's row sums are $(0,0,1,1)$. The quadratic function
of real outputs checks backward propagation here; it is not a
language-model training objective.

The module has no bias, dropout, RoPE, residual connections, or normalization.
A padding query's zero $AV$ therefore remains zero after $W^O$.
A complete Transformer must account for those other operations separately.
The masked softmax, three projections, and merge follow the construction
in [D2L](https://d2l.ai/chapter_attention-mechanisms-and-transformers/multihead-attention.html)
and [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/);
the empty-row policy is an explicit addition here.

Checking shapes alone does not establish data order: `wrong` has
the correct dimensions. Comparing implementations also requires numerical
outputs and derivatives, a separate per-head reference, rectangular Q/K,
and independence from forbidden tokens. Appropriate tolerances depend
on dtype and computational backend.

[CS336 Assignment 1](https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf),
§§3.4.4–3.4.5, specifies contracts for three- and four-dimensional attention
and complete MHA. The assignment also applies RoPE; rotations of queries
and keys are covered in the next chapter on positional information.
Completing that assignment requires adding RoPE to this module.

A complete Transformer adds an FFN, residual connections, and normalization.
The connection between shapes, parameters, FLOPs, and memory continues in
[[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/05 Transformer с нуля — формы, параметры и стоимость#Сначала имена осей, затем операции|the Transformer cost chapter]].
Implementation continues in
[[02 Areas/ML & DL/06 Практика/20 Собрать языковую модель с нуля#Этап 2. Модель от базовых операций к логитам|Capstone 1]].

## Sources and visual continuations

- [Stanford CS336 Spring 2026, Lecture 2 — pinned source](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_02.py): named axes, `einsum`, and the relationship between shapes and resource counts.
- [Stanford CS336 Spring 2026, Assignment 1 — pinned assignment](https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf): testable attention and Transformer interfaces.
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), §§3.2.2–3.2.3: original multi-head and masked-attention definitions.
- Lena Voita, [Seq2seq and Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html): causal-mask and individual-head animations.
- Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/): visual split, concatenation, and $W^O$.
- Dive into Deep Learning, [Multi-Head Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/multihead-attention.html): implementation with explicit shape transformations.
- Harvard NLP, [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/): subsequent-position masking and multi-head implementation alongside the equations.
- PyTorch, [scaled_dot_product_attention](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html): versioned semantics of `attn_mask`, `is_causal`, and dropout.

**Previous:** [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V|Self-attention]] ·
**Next:** [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/04 Позиционная информация|Positional information]]
