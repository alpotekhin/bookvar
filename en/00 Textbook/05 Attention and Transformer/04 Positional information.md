---
title: Positional information in Transformers
type: textbook-chapter
status: canonical
locale: en
translation_status: current
canonical_revision: "897a19c428e9ef20a6c651e981eb32d905d6f72a616907c2ea2e20fa1842451a"
translation_reviewed: "2026-10-29"
translation_of: "00 Учебник/05 Attention и Transformer/04 Позиционная информация.md"
last_updated: 2026-10-29
source_unit_id:
  - lecture-03-rope-derivation
previous: "[[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/Masking, multi-head и формы тензоров]]"
next: "[[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/03 Полный Transformer]]"
primary_sources:
  - https://arxiv.org/abs/1706.03762
  - https://arxiv.org/abs/1803.02155
  - https://jmlr.org/papers/v21/20-074.html
  - https://arxiv.org/abs/2104.09864
  - https://arxiv.org/abs/2108.12409
  - https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_03.pdf
---

# Positional information in Transformers

If a sentence is split into words, `John loves Mary` and `Mary loves John`
contain the same tokens but describe different relationships.
For full, unmasked self-attention without positional features, permuting
input rows permutes Q/K/V and output rows in the same way. Formally,
$F(PX)=PF(X)$ for a permutation matrix $P$. This is equivariance, not
an unchanged output matrix: each row's content moves with its token.

A causal mask prevents position $i$ from reading keys $j>i$ and creates
an asymmetry itself. If tokens are permuted while the triangular mask stays
fixed, their accessible predecessors change; the equality need not hold.
The mask adds no explicit metric encoding of distance, but can provide
an implicit positional signal. Causal NoPE studies investigate learned
positional features ([Haviv et al.](https://aclanthology.org/2022.findings-emnlp.99/),
[Kazemnejad et al.](https://arxiv.org/abs/2305.19466)).
For an arbitrary mask, equivariance requires permuting its rows and columns
consistently: $M\mapsto PMP^\top$. A padding mask excludes invalid positions
but does not replace the chosen positional mechanism.

The unmasked equality can be checked step by step.
Permuting rows gives $Q'=PQ$, $K'=PK$, and $V'=PV$:

$$
\begin{aligned}
S'&=PSP^\top,\\
A'&=PAP^\top,\\
Z'&=A'V'=PAV=PZ.
\end{aligned}
$$

Softmax normalizes each row over keys. Permuting queries therefore moves
weight rows, while permuting keys moves their columns. The model has no
separate “first” or “third” feature: a token retains its representation
when surrounded by the same set of neighbors.

A causal counterexample needs no learned scores. Let the values be
$(1,3,7)$, with each query averaging its accessible prefix.
The outputs are $(1,2,11/3)$. Reversing the input gives $(7,5,11/3)$,
whereas reversing the original outputs gives $(11/3,2,1)$.
Different predecessor sets already distinguish positions. This does not
show that prefix averaging recovers exact distances or solves arbitrary
word-order tasks.

Position methods can be compared by where they change the Transformer:

1. Before the first layer, add an absolute vector to the token embedding.
2. In the score, add a term depending on relative offset $j-i$ to
   $q_i k_j^\top$.
3. Before calculating the score, transform queries and keys so that their
   dot product depends on $j-i$.

## Starting point: where position enters attention

For one head before positional information is added, use row vectors
$x_i\in\mathbb R^{1\times d_{model}}$ and
$q_i,k_i\in\mathbb R^{1\times d_h}$.
$W_Q,W_K$ have shape $d_{model}\times d_h$; Value width is also $d_h$ here:

$$
\begin{aligned}
q_i&=x_iW_Q,\\
k_j&=x_jW_K,\\
v_j&=x_jW_V,
\end{aligned}
$$

$$
\begin{aligned}
s_{ij}&=\frac{q_i k_j^\top}{\sqrt{d_h}}+M_{ij},\\
a_{ij}&=\operatorname{softmax}_j(s_{ij}),\\
z_i&=\sum_j a_{ij}v_j.
\end{aligned}
$$

$M_{ij}$ denotes a causal or padding mask. Each method below changes
$x_i$, $s_{ij}$, or $q_i,k_j$. In an implementation, these operations
may belong to different modules: vector addition near the embedding table,
relative bias near the score matrix, and RoPE rotation after Q/K projection.

## Absolute positions: an address for each row

### A learned table

The most direct method stores
$P\in\mathbb R^{T_{max}\times d_{model}}$. Token $t_i$ enters the
first layer as

$$
h_i^{(0)}=e(t_i)+P_i.
$$

BERT and early GPT models use this design. Each position $i$ receives
a learned vector $P_i$ with the same width as a token embedding.
For a batch `[B,T,d]` whose examples all use positions 0 through $T-1$,
the first `T` rows of `[T_max,d]` are added to each example.
Examples with different numbering instead select rows by explicit position IDs.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/transformer_positional_encoding_vectors.png]]

*Position vectors are added to the embeddings of Je, suis, and étudiant.
The first block and its $W_Q$, $W_K$, and $W_V$ projections receive
the sum of content and position. Source: Jay Alammar,
[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

The table contains $T_{max}d_{model}$ parameters. The gradient from each
use of $P_i$ is added to gradients from the same position in other examples.
An allocated row that is never used during training has numerical values,
but has learned no useful positional geometry.

Positions outside the table are a different problem. With $T_{max}=2048$,
$P_{4096}$ does not exist. Enlarging the table, interpolating it, and
training added rows change the model. Allocating a larger tensor does not
by itself train new positions.

Take token embeddings $e_A=(2,0)$ and $e_B=(0,1)$, with
$P_0=(0,1)$ and $P_1=(1,0)$. Attention receives $h_0=(2,1)$
and $h_1=(1,1)$. With identity Q/K projections, query 0 and key 1
have dot product $3$. It expands into four terms:

$$
\begin{aligned}
h_0h_1^\top
&=e_Ae_B^\top+e_AP_1^\top\\
&\quad+P_0e_B^\top+P_0P_1^\top\\
&=0+2+1+0=3.
\end{aligned}
$$

Position changes more than the comparison between two position vectors:
cross-terms compare one token's content with the other's position.
After division by $\sqrt2$, query 0 has scores
$(5/\sqrt2,3/\sqrt2)$ and weights approximately
$(0.804430,0.195570)$. If the V projection is also the identity,
the output is $(1.804430,1)$.

```python
import math
import torch
from torch import nn

dtype = torch.float64
e = torch.tensor([[2., 0.], [0., 1.]], dtype=dtype, requires_grad=True)
P = nn.Parameter(torch.tensor([[0., 1.], [1., 0.],
                               [0., 0.], [0., 0.]], dtype=dtype))
h = e + P[:2]
# Identity Q/K/V projections isolate the effect of adding positions.
scores = h @ h.T / math.sqrt(2)
weights = scores.softmax(-1)
z = weights @ h
print("positioned inputs:", h.detach().tolist())
print("scaled scores:", scores.detach().tolist())
print("first output:", z[0].detach().tolist())
torch.testing.assert_close(scores[0, 1],
                           torch.tensor(3 / math.sqrt(2), dtype=dtype))

loss = z[0].square().mean()
loss.backward()
assert torch.isfinite(e.grad).all() and torch.isfinite(P.grad).all()
assert torch.count_nonzero(P.grad[2:]) == 0
assert torch.count_nonzero(P.grad[:2]) > 0
```

The example's table has four rows, but uses only the first two.
Backward propagation gives zero gradient for the two unused rows.
This teaching example uses identity projections without dropout or
normalization; later chapters cover each architecture's additional operations.

Relative distance creates a further learning problem. The relationship
“the key is three positions to the left” must be learned from a pair of
absolute vectors $P_i,P_j$. Shifting the same text span to the right
presents a different address pair even though the relative structure is unchanged.

### Sinusoidal encoding

The original Transformer used fixed sine and cosine functions instead of
a learned table:

$$
\begin{aligned}
PE_{i,2k}&=\sin\left(i/10000^{2k/d}\right),\\
PE_{i,2k+1}&=\cos\left(i/10000^{2k/d}\right).
\end{aligned}
$$

Like a learned position vector, $PE_i$ is added to the token embedding.
The formula introduces no parameters and can produce a code for any
integer $i$. In §3.5, Vaswani et al. report similar results for fixed and
learned encodings on their translation task. That is not a comparison
across all models or arbitrary sequence lengths.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-25-31/d2l-sinusoidal-curves.svg]]

*Coordinates 6–9 of a width-32 encoding over 60 positions. Coordinates 6
and 7 oscillate faster than 8 and 9; each sine/cosine pair has a phase
offset. Source: Zhang et al., [Dive into Deep Learning, §11.6](https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html),
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-25-31/d2l-positional-heatmap.svg]]

*The same signal as a matrix: 60 position rows and 32 coordinate columns.
Phase changes rapidly between rows on the left; the same interval covers
only a small part of a slow oscillation on the right. Source: Zhang et al.,
[Dive into Deep Learning, §11.6](https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html),
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

For $d=4$, the frequencies are 1 and 0.01 radians per position.
Position 0 has code $(0,1,0,1)$. At position 1, the first pair is
approximately $(0.841471,0.540302)$ and the second is
$(0.010000,0.999950)$. One step noticeably rotates the fast pair
while barely changing the slow pair. The same index is represented
at several scales.

For one pair, let $c_i$ be the column containing its sine and cosine.
Angle-addition identities give

$$
c_i=\begin{bmatrix}\sin(i\omega)\\\cos(i\omega)\end{bmatrix},
\qquad
c_{i+r}=U_r c_i,
$$

$$
U_r=\begin{bmatrix}
\cos(r\omega)&\sin(r\omega)\\
-\sin(r\omega)&\cos(r\omega)
\end{bmatrix}.
$$

$U_r$ depends on the shift $r$, not the original position $i$.
Its sine signs differ from a standard rotation of an $(x,y)$ column
because these coordinates are ordered as $(\sin,\cos)$.

```python
import math
import torch

def sinusoidal(positions, d, base=10000.):
    if d <= 0 or d % 2 or base <= 1:
        raise ValueError("positive even d and base > 1 required")
    positions = torch.as_tensor(positions, dtype=torch.float64)
    frequency = base ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)
    angles = positions[..., None] * frequency
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)

P = sinusoidal(torch.arange(3), 4)
print("sinusoidal rows:", P.tolist())
torch.testing.assert_close(P[0], torch.tensor([0., 1., 0., 1.],
                                              dtype=torch.float64))
torch.testing.assert_close(P.square().sum(-1),
                           torch.full((3,), 2., dtype=torch.float64))
# For the [sin, cos] coordinate order, a shift acts by this matrix on columns.
delta, frequency = 3., 1.
c, s = math.cos(delta * frequency), math.sin(delta * frequency)
U = torch.tensor([[c, s], [-s, c]], dtype=torch.float64)
torch.testing.assert_close(U @ P[1, :2], sinusoidal([4], 4)[0, :2])
print("one-position PE dot product:", (P[0] @ P[1]).item())
```

Each pair has squared norm one, so $\|PE_i\|^2=d/2$.
The dot product of two pure position codes is

$$
PE_iPE_j^\top=
\sum_{k=0}^{d/2-1}\cos\bigl((i-j)\omega_k\bigr).
$$

It depends only on the difference between their indices.
Attention, however, receives sums of content and position and applies
learned projections. The four terms from the preceding example remain:
the property of pure PE vectors does not make the whole attention
operation invariant to a common shift.

The formula can be evaluated beyond training length, but this does not
guarantee quality at new distances. The model encounters new combinations
of phases and new sets of keys. This limitation differs from the
physical boundary of a learned table.

## Shaw relative representations: distance becomes part of an edge

Shaw, Uszkoreit, and Vaswani encode directed offset $j-i$, rather than
the two addresses $i$ and $j$ independently. One score variant is

$$
s_{ij}=\frac{q_i(k_j+a^K_{ij})^\top}{\sqrt{d_h}},
$$

with output

$$
z_i=\sum_j a_{ij}(v_j+a^V_{ij}).
$$

The vectors $a^K_{ij}$ and $a^V_{ij}$ are selected from learned tables
using an offset clipped at distance $k$:

$$
\operatorname{clip}(j-i,k)=\max(-k,\min(k,j-i)).
$$

Distances beyond $k$ share a class. The first additional term changes
the edge's weight: the query compares against both key content and
relative-position type. The second changes the message transmitted:
the same value may contribute differently depending on direction and distance.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/positions-2026-10-21/shaw-relative-figure1.png]]

*Edges $x_2\to x_1$, $x_2\to x_4$, and $x_4\to x_n$ receive
representations for offsets $-1$, $2$, and the limiting class $k$.
Source: Shaw et al., [Self-Attention with Relative Position Representations,
Figure 1](https://aclanthology.org/N18-2074.pdf#page=3), CC BY 4.0.
The full figure and original caption are retained.*

With $k=2$, a query at position 4 distinguishes keys at positions 3
and 4 as offsets $-1$ and $0$, while keys 0, 1, and 2 all receive
the far-left class $-2$. Shifting both indices by ten leaves those
classes unchanged. With the same incoming Q/K/V and attention mask,
the edge computation is preserved.

For a numerical unmasked example, place the query at position 2,
keys at 1, 2, and 4, and set $k=1$. Offset classes are $(-1,0,1)$;
the last key receives the limiting right-hand class. Let the scalar
query equal 1, all keys equal 2, and the relative key additions be
$(0,1,-1)$. Scores are $(2,3,1)$, giving weights approximately
$(0.244728,0.665241,0.090031)$.

Values $(10,20,30)$ and relative-value additions $(1,0,-2)$ produce
messages $(11,20,28)$. Their weighted output is approximately
$18.517688$. Changing a relative key changes its weight;
changing a relative value changes the number transmitted along that edge.

```python
import torch

dtype = torch.float64
query_position = 2
key_positions = torch.tensor([1, 2, 4])
clip_distance = 1
relative = (key_positions - query_position).clamp(-clip_distance, clip_distance)
# Table order is [-1, 0, +1]; scalar head dimension is 1.
key_table = torch.tensor([0., 1., -1.], dtype=dtype, requires_grad=True)
value_table = torch.tensor([1., 0., -2.], dtype=dtype, requires_grad=True)
indices = relative + clip_distance
q = torch.tensor(1., dtype=dtype)
k = torch.tensor([2., 2., 2.], dtype=dtype)
v = torch.tensor([10., 20., 30.], dtype=dtype)
scores = q * (k + key_table[indices])
weights = scores.softmax(-1)
messages = v + value_table[indices]
z = weights @ messages
print("clipped offsets:", relative.tolist())
print("Shaw weights:", weights.detach().tolist())
print("Shaw messages:", messages.detach().tolist(), "output:", z.item())
torch.testing.assert_close(scores, torch.tensor([2., 3., 1.], dtype=dtype))
z.backward()
# For this scalar head: dz/d(relative value) = attention weight.
torch.testing.assert_close(value_table.grad, weights.detach())
expected_key_grad = q * weights.detach() * (messages.detach() - z.detach())
torch.testing.assert_close(key_table.grad, expected_key_grad)
```

In this scalar example, the derivative with respect to a relative value
equals its attention weight. The derivative with respect to a relative
key is $q\,a_{ij}(v_j+a^V_{ij}-z_i)$. Both identities are checked
against backward propagation.

A naive implementation creates a separate vector for each position pair,
forming a $T_qT_kd_h$ intermediate tensor. In §3.3, Shaw et al.
split the score into the ordinary Q/K product and a relative-representation
term, reusing relative representations across examples and, with the chosen
sharing scheme, across heads. Each table contains $2k+1$ vectors,
not a separate parameter for every pair. Parameter sharing and
intermediate-tensor materialization are distinct implementation choices.

## T5 relative position bias: one scalar per bucket and head

T5's encoder and decoder self-attention receive a learned scalar
position bias for each head:

$$
\begin{aligned}
s_{ij}^{(h)}&=q_i^{(h)}(k_j^{(h)})^\top\\
&\quad+b_h\bigl(\operatorname{bucket}(j-i)\bigr)+M_{ij}.
\end{aligned}
$$

The [pinned T5 implementation](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/models/t5/modeling_t5.py#L529-L562)
multiplies Q/K without an additional division by $\sqrt{d_h}$;
projection initialization accounts for this choice. The equation above
uses T5's convention. Relative bias can also be added to ordinary scaled
attention, but that configuration has a different numerical scale.

Unlike Shaw's vectors, each relative representation is a single scalar.
It changes softmax, not V. Different heads have different numbers for
the same distance class. The [T5 paper, §2.1](https://jmlr.org/papers/volume21/20-074/20-074.pdf#page=5)
shares parameters across layers. In the pinned implementation, encoder
and decoder have separate tables; within each stack, the computed bias
is passed to later layers. Cross-attention does not receive such a
learned distance table.

With 32 buckets and maximum distance 128, the boundaries differ:

- The causal decoder assigns all 32 classes to current and past positions.
  Distances 0–15 are exact; larger distances use logarithmic intervals.
- The bidirectional encoder assigns 16 classes to each direction.
  Distances 0–7 are exact within each direction. Positive $j-i$
  adds 16 to the bucket number.

Offset zero belongs to bucket 0. The causal bucket function also
returns 0 for future positions; a separate causal mask forbids them.

| Absolute distance | Decoder | Encoder past | Encoder future at the same distance |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 1 | 1 | 1 | 17 |
| 8 | 8 | 8 | 24 |
| 16 | 16 | 10 | 26 |
| 32 | 21 | 12 | 28 |
| 64 | 26 | 14 | 30 |
| 128 | 31 | 15 | 31 |
| 512 | 31 | 15 | 31 |

Large-distance intervals come from the logarithm of distance divided
by the boundary of the exact region. The resulting bucket is clipped
to the final class. The code adapts the
[Transformers v4.57.1 function](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/models/t5/modeling_t5.py#L400-L446)
under Apache-2.0. Small distances are temporarily clamped in the unused
logarithmic branch to avoid evaluating $\log0$; bucket numbers are unchanged.

```python
import math
import torch
from torch import nn

# Bucket mapping adapted from Mesh TensorFlow/T5/Hugging Face (Apache-2.0).
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
def t5_bucket(relative_position, bidirectional=True,
              num_buckets=32, max_distance=128):
    if num_buckets < 4 or (bidirectional and num_buckets % 2):
        raise ValueError("invalid bucket count")
    relative_position = torch.as_tensor(relative_position, dtype=torch.long)
    buckets_per_direction = num_buckets // 2 if bidirectional else num_buckets
    exact = buckets_per_direction // 2
    if max_distance <= exact:
        raise ValueError("max_distance must exceed the exact range")
    if bidirectional:
        direction = (relative_position > 0).long() * buckets_per_direction
        distance = relative_position.abs()
    else:
        direction = torch.zeros_like(relative_position)
        distance = (-relative_position).clamp_min(0)
    # Clamp only the unused logarithmic branch at small distances.
    large = exact + (
        torch.log(distance.clamp_min(exact).float() / exact)
        / math.log(max_distance / exact) * (buckets_per_direction - exact)
    ).long()
    large = large.clamp_max(buckets_per_direction - 1)
    return direction + torch.where(distance < exact, distance, large)

distance = torch.tensor([0, 1, 7, 8, 15, 16, 32, 64, 128, 512])
print("causal buckets:", t5_bucket(-distance, False).tolist())
print("encoder past:", t5_bucket(-distance, True).tolist())
print("encoder future:", t5_bucket(distance, True).tolist())

relative = -torch.tensor([1, 128, 512])
bucket_ids = t5_bucket(relative, False)
table = nn.Embedding(32, 2, dtype=torch.float64)
with torch.no_grad():
    table.weight.zero_()
    table.weight[31] = torch.tensor([-1., 1.], dtype=torch.float64)
# Shape [Tk,H] -> [H,Tk]; all content logits are zero.
bias = table(bucket_ids).T
weights = bias.softmax(-1)
values = torch.tensor([10., 20., 100.], dtype=torch.float64)
z = weights @ values
print("T5 two-head weights:", weights.detach().tolist())
print("T5 two-head outputs:", z.detach().tolist())
z.sum().backward()
assert torch.isfinite(table.weight.grad).all()
expected_far = (weights[:, 1:] * (values[1:] - z[:, None])).sum(-1)
torch.testing.assert_close(table.weight.grad[31], expected_far)
```

The final example has keys at distances 1, 128, and 512, with zero
content scores. The first head assigns bias $-1$ to bucket 31,
giving weights approximately $(0.576117,0.211942,0.211942)$.
The second head assigns $+1$ to the same bucket and gives both
distant keys more weight.

Distances 128 and 512 receive the same position bias. If their content
scores differ, their final weights can still differ. The coarse grouping
limits one bias term, not the entire multilayer network's ability to
use distant information.

A bucket parameter receives the sum of gradients from all pairs assigned
to it. For one query and a scalar output, this sum is
$\sum_{j\in b}a_{ij}(v_j-z_i)$; the example checks it.

One table has $32H$ parameters. Encoder and decoder have two separate
tables when both stacks use the mechanism with the same number of heads.

<a id="rope-relative-position-derivation"></a>

## RoPE: position as a rotation of query and key

The preceding methods add position to either embeddings or scores.
RoPE transforms already projected Q/K: each coordinate pair rotates
by an angle proportional to position. V is not rotated.

For unchanged content, the edge score should remain unchanged under
a common shift of both indices.
[CS336 Lecture 3, slides 30–35](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_03.pdf#page=30)
develops this requirement from the mixed terms of absolute encodings
to the geometry of rotations. Rotating both vectors by the same
additional angle preserves their dot product.

The standard rotation of a two-coordinate column is

$$
R_i(\theta)=
\begin{bmatrix}
\cos(i\theta)&-\sin(i\theta)\\
\sin(i\theta)& \cos(i\theta)
\end{bmatrix}.
$$

Our states are rows, so

$$
q_i'=q_iR_i^\top,\qquad k_j'=k_jR_j^\top.
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/positions-2026-10-21/roformer-figure1.png]]

*The upper diagram rotates one coordinate pair by $m\theta_1$.
The lower diagram applies the same operation to multiple pairs with
different frequencies. Source: Su et al.,
[RoFormer v5, Figure 1](https://arxiv.org/pdf/2104.09864v5#page=5),
CC BY-NC-ND 4.0. The figure and its original caption are reproduced
without changing their content.*

Substitute the transformed rows into the dot product:

$$
\begin{aligned}
q_i'(k_j')^\top
&=q_iR_i^\top R_jk_j^\top\\
&=q_iR_{j-i}k_j^\top.
\end{aligned}
$$

The second line uses $R_i^\top=R_{-i}$ and angle composition
$R_{-i}R_j=R_{j-i}$. After adding $c$ to both indices,
$(j+c)-(i+c)$ is unchanged.

This property assumes fixed incoming Q/K and a consistently transformed
mask. Adding new predecessors to the text changes context and token
representations; that is more than relabeling two indices. Nor does the
identity show that one score recovers an arbitrary distance.

### A numerical RoPE example

Take $q=(1,2)$ and $k=(3,4)$, with illustrative frequency
$\theta=\pi/2$. Put the query at $i=2$ and the key at $j=1$:

1. Rotating the query by $180^\circ$ gives $q'=(-1,-2)$.
2. Rotating the key by $90^\circ$ gives $k'=(-4,3)$.
3. The dot product is $4-6=-2$.
4. In relative form, $R_{j-i}=R_{-1}$ and
   $R_{-1}k^\top=(4,-3)^\top$, so $qR_{-1}k^\top=4-6=-2$.

Before rotation the dot product was 11. Squared norms are preserved:
$\|q'\|^2=5$ and $\|k'\|^2=25$. Reversing the rotation direction
by mistake produces $+2$ rather than $-2$. A zero result or identical
Q/K vectors could conceal this sign error.

A head of width $d_h$ uses $d_h/2$ pairs. For
$r=0,\ldots,d_h/2-1$, their frequencies are

$$
\theta_r=\operatorname{base}^{-2r/d_h}.
$$

With $d_h=4$ and base 10000, the frequencies are 1 and 0.01 radians
per position. The dot product sums all pairs' contributions.
Rotation preserves each pair's norm and the complete vector's norm;
that norm is not a distance-to-query measure.

There is no need to construct a large block-diagonal matrix.
For pair $(u,v)$ and angle $\varphi$, compute

$$
\begin{aligned}
u'&=u\cos\varphi-v\sin\varphi,\\
v'&=u\sin\varphi+v\cos\varphi.
\end{aligned}
$$

This is the implementation in RoFormer §3.4.2.
For the row-vector expression $q_i'=q_iR_i^\top$, backward gives
$G_{q_i}=G_{q_i'}R_i$, applying the inverse of the forward rotation.
RoPE itself adds no trainable parameters; the Q/K projections remain trainable.

### RoPE and KV-cache: a key retains its position

A full pass over known text rotates all queries and keys in one call.
During incremental generation, previously rotated keys can be cached.
The next step computes and rotates only new Q/K, reading old K and V
from the cache. Each layer has its own projections and cached keys.

Suppose four positions all project to $q_i=k_i=(1,0)$, with scalar
values $v_i=i+1$. This head has one pair at frequency one radian
per position. After RoPE, $q_i'=k_i'=(\cos i,\sin i)$, so an
allowed score is $\cos(i-j)/\sqrt2$. Step 0 can read only value 1.
At step 1, weights are approximately $(0.419444,0.580556)$,
giving $0.419444\cdot1+0.580556\cdot2=1.580556$.

| Query position | New key | Cached key positions after insertion | Output |
|---:|---|---|---:|
| 0 | $(1,0)$ | 0 | 1.000000 |
| 1 | $(\cos1,\sin1)$ | 0, 1 | 1.580556 |
| 2 | $(\cos2,\sin2)$ | 0, 1, 2 | 2.302710 |
| 3 | $(\cos3,\sin3)$ | 0, 1, 2, 3 | 3.061223 |

At the final step, keys 0, 1, 2, and 3 have scores
$-0.700030$, $-0.294260$, $0.382051$, and $0.707107$.
Softmax gives $0.104871$, $0.157355$, $0.309455$, and $0.428319$,
which weight values $(1,2,3,4)$. A full causal pass must reproduce
all four outputs even though it computes them simultaneously.

Rotating a cached $k_j'$ by $j$ again would replace its angle $j$
with $2j$. The final query would then have scores
$\cos(3-2j)/\sqrt2$ and output $2.5$, rather than $3.061223$.
Resetting a new query's position to zero is a different error: cached
keys retain their angles, but the query compares with them as position 0.

The following code rotates the entire head using adjacent coordinate pairs.
Positions may be shared as `[T]` or match every leading input axis.
For `[B,H,T,D]`, example-specific positions are supplied as
`[B,H,T]`, for instance with
`position_ids[:,None,:].expand(-1,H,-1)`.
The implementation forbids implicitly aligning a `[B,T]` tensor
with the head axis.

```python
import math
import torch
import torch.nn.functional as F

def apply_rope(x, positions, base=10000.):
    # x: [...,T,D]. Positions: [T] or exactly x.shape[:-1].
    if not x.is_floating_point() or x.ndim < 2:
        raise ValueError("x must be floating point [...,T,D]")
    d = x.shape[-1]
    if d <= 0 or d % 2 or base <= 1:
        raise ValueError("positive even rotary width and base > 1 required")
    positions = torch.as_tensor(positions, device=x.device)
    if positions.shape not in [(x.shape[-2],), x.shape[:-1]]:
        raise ValueError("positions must be [T] or match every leading axis")
    angle_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    frequency = base ** (
        -torch.arange(0, d, 2, device=x.device, dtype=angle_dtype) / d)
    angles = positions.to(angle_dtype)[..., None] * frequency
    c, s = angles.cos(), angles.sin()
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((even*c - odd*s, even*s + odd*c), dim=-1)
    return rotated.flatten(-2).to(x.dtype)

torch.set_num_threads(1)
torch.manual_seed(31)
B, H, T, D = 2, 3, 4, 4
q, k, v = [torch.randn(B, H, T, D, dtype=torch.float64, requires_grad=True)
           for _ in range(3)]
positions = torch.arange(T)
qr = apply_rope(q, positions)
kr = apply_rope(k, positions)
torch.testing.assert_close(qr.square().sum(-1), q.square().sum(-1))
torch.testing.assert_close(
    qr @ kr.transpose(-2, -1),
    apply_rope(q, positions + 5) @ apply_rope(k, positions + 5).transpose(-2, -1))

allowed = positions[None, :] <= positions[:, None]
out = F.scaled_dot_product_attention(qr, kr, v, attn_mask=allowed, dropout_p=0.)
# Cache contains keys already rotated at positions 0,1,2,3.
last_allowed = positions[None, :] <= positions[-1:]
cached = F.scaled_dot_product_attention(
    apply_rope(q[..., -1:, :], positions[-1:]), kr, v,
    attn_mask=last_allowed, dropout_p=0.)
torch.testing.assert_close(cached, out[..., -1:, :])
wrong = F.scaled_dot_product_attention(
    apply_rope(q[..., -1:, :], torch.tensor([0])), kr, v,
    attn_mask=last_allowed, dropout_p=0.)
print("full-versus-cache error:", (cached-out[..., -1:, :]).abs().max().item())
print("reset-position error:", (wrong-cached).abs().max().item())
assert not torch.allclose(wrong, cached)

out.square().mean().backward()
for tensor in (q, k, v):
    assert tensor.grad is not None and torch.isfinite(tensor.grad).all()

# Four scalar outputs from the table: one head, dk=2 and dv=1.
raw = torch.tensor([1., 0.], dtype=torch.float64).expand(1, 1, 4, 2)
values = torch.arange(1., 5., dtype=torch.float64).reshape(1, 1, 4, 1)
p = torch.arange(4)
rotated = apply_rope(raw, p)
full_demo = F.scaled_dot_product_attention(
    rotated, rotated, values, attn_mask=p[None, :] <= p[:, None], dropout_p=0.)
key_cache = value_cache = None
step_outputs = []
for i in range(4):
    # Only the new key is rotated; earlier cache entries are not changed.
    current = apply_rope(raw[..., i:i+1, :], p[i:i+1])
    new_value = values[..., i:i+1, :]
    key_cache = current if key_cache is None else torch.cat((key_cache, current), dim=-2)
    value_cache = new_value if value_cache is None else torch.cat((value_cache, new_value), dim=-2)
    step_outputs.append(F.scaled_dot_product_attention(
        current, key_cache, value_cache, dropout_p=0.))
incremental = torch.cat(step_outputs, dim=-2)
torch.testing.assert_close(incremental, full_demo)
print("incremental RoPE outputs:", incremental.flatten().tolist())

# Two queries processed together still need the causal rule within the block.
block_allowed = p[None, :] <= p[2:, None]
block_output = F.scaled_dot_product_attention(
    rotated[..., 2:, :], key_cache, value_cache,
    attn_mask=block_allowed, dropout_p=0.)
torch.testing.assert_close(block_output, full_demo[..., 2:, :])
double_rotated = F.scaled_dot_product_attention(
    rotated[..., -1:, :], apply_rope(key_cache, p), value_cache, dropout_p=0.)
torch.testing.assert_close(double_rotated, torch.full_like(double_rotated, 2.5))
print("double-rotated cache output:", double_rotated.item())
```

The full pass and cached evaluation of its last query give the same
output. Assigning the query position 0 again changes its relative angles
to the stored keys. Those keys are already rotated and must not be
rotated twice. Position semantics and causal access remain separate conditions.

In the stepwise loop, each query reads the entire accumulated cache;
there are no future keys yet. That single query row therefore needs
no additional causal mask. If positions 2 and 3 are processed together,
the first may read keys 0–2 and the second keys 0–3. Their mask rows
are `1110` and `1111`, as in the preceding chapter's rectangular
example. RoPE positions and the mask are specified separately.
The code checks one attention layer with given projections and no dropout,
not a complete language-model generation implementation.

Two further implementation choices are independent:

- **Pair order.** This code uses adjacent pairs $(0,1),(2,3)$.
  A `rotate_half` layout at width four may pair $(0,2),(1,3)$.
  Both layouts can express the same mechanism if coordinates and Q/K
  projection outputs are permuted consistently. Replacing only the
  rotation function in an existing checkpoint is insufficient.
- **Rotary width.** An even-sized portion of the head can be rotated
  while the other coordinates remain unchanged. For $(a,b,c,d)$,
  one might rotate $(a,b)$ and retain $(c,d)$. That is a separate
  configuration; the code above rotates all D coordinates.

RoPE also supplies no monotonic penalty for distant keys.
The standard first pair has $\theta_0=1$. With $q=k=(1,0)$,
dot products at distances 0, 3, and 6 are
$1$, $\cos3\approx-0.989992$, and $\cos6\approx0.960170$.
A farther key can score higher than an intermediate one.
RoFormer §3.4.3 studies sums of different-frequency contributions
and a bound on their long-distance behavior; it does not guarantee
that every individual attention score decreases at every step.

## ALiBi: a linear distance penalty

Attention with Linear Biases adds a fixed linear penalty to each head's
scores based on distance to a past key. It adds no position vectors
to the input and performs no Q/K rotation:

$$
\begin{gathered}
s_{ij}^{(h)}
=\frac{q_i^{(h)}(k_j^{(h)})^\top}{\sqrt{d_h}}-m_h(i-j),\\
j\le i.
\end{gathered}
$$

$m_h>0$ is a predetermined slope, not a trainable parameter.
With equal content, a farther key gets a lower score. Different slopes
impose different strengths of preference for nearby keys; they do not
guarantee predetermined head specializations.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/positions-2026-10-21/alibi-original.jpeg]]

*The content-score matrix receives negative distances multiplied by the
head's slope. A causal mask blocks future positions; the linear bias
alone does not forbid them. Source: Press, Smith, and Lewis,
[ALiBi repository](https://github.com/ofirpress/attention_with_linear_biases/blob/4b92f28a005ead2567abe2359f633e73e08f3833/README.md),
[MIT license](https://github.com/ofirpress/attention_with_linear_biases/blob/4b92f28a005ead2567abe2359f633e73e08f3833/LICENSE).
The original JPEG is unchanged.*

For a query at position 5 and identical content scores `2.0`
for keys 4, 3, and 1, take $m_h=0.25$:

| Key $j$ | Distance $5-j$ | Bias | Final logit |
|---:|---:|---:|---:|
| 4 | 1 | $-0.25$ | 1.75 |
| 3 | 2 | $-0.50$ | 1.50 |
| 1 | 4 | $-1.00$ | 1.00 |

Softmax gives approximately $(0.444214,0.345954,0.209832)$.
For values $(10,20,100)$, the output is $32.344406$.
The nearest-to-farthest weight ratio is
$\exp(0.25\cdot(4-1))=\exp(0.75)$: the shared content score
cancels from the ratio of exponentials.

In the [original implementation](https://github.com/ofirpress/attention_with_linear_biases/blob/4b92f28a005ead2567abe2359f633e73e08f3833/fairseq/models/transformer.py#L742-L762),
eight heads receive slopes $2^{-1},2^{-2},\ldots,2^{-8}$.
This scale is geometric rather than evenly spaced. If the head count
is not a power of two, the code extends the nearest smaller scale
with part of a denser one.

```python
import math
import torch

def alibi_slopes(n):
    if not isinstance(n, int) or n <= 0:
        raise ValueError("positive integer head count required")
    def power_of_two(count):
        start = 2 ** (-2 ** -(math.log2(count) - 3))
        return [start * start**i for i in range(count)]
    if math.log2(n).is_integer():
        return power_of_two(n)
    lower = 2 ** math.floor(math.log2(n))
    return power_of_two(lower) + alibi_slopes(2*lower)[0::2][:n-lower]

slopes = torch.tensor(alibi_slopes(8), dtype=torch.float64)
m = slopes[1]  # 0.25
query_position = 5
key_positions = torch.tensor([4, 3, 1])
bias = -m * (query_position - key_positions)
content = torch.full((3,), 2., dtype=torch.float64)
weights = (content + bias).softmax(-1)
values = torch.tensor([10., 20., 100.], dtype=torch.float64)
print("eight ALiBi slopes:", slopes.tolist())
print("ALiBi weights:", weights.tolist(), "output:", (weights @ values).item())
torch.testing.assert_close(weights[0] / weights[2], torch.exp(3*m))

# Remove a row-wise constant, keeping exactly the same causal softmax.
positions = torch.arange(5)
distance = positions[:, None] - positions[None, :]
allowed = distance >= 0
relative_bias = -slopes[:, None, None] * distance
column_bias = slopes[:, None, None] * positions[None, None, :]
content = torch.arange(25, dtype=torch.float64).reshape(1, 5, 5) / 7
full = (content + relative_bias).masked_fill(~allowed, -torch.inf)
compact = (content + column_bias).masked_fill(~allowed, -torch.inf)
torch.testing.assert_close(full.softmax(-1), compact.softmax(-1))
print("relative-bias and column-bias attention agree")
```

The second check explains why the authors' implementation can store
a term depending only on key index. Expanding
$-m_h(i-j)=m_hj-m_hi$ isolates the row constant $-m_hi$.
It cancels in softmax, so the same causal mask permits using only
$m_hj$. Attention weights are preserved; this does not turn ALiBi
into an absolute positional table.

In [§3 of the paper](https://ofir.io/train_short_test_long.pdf#page=5),
the bias is added after scaling the dot product. Do not divide the
linear penalty by $\sqrt{d_h}$ a second time.

The linear function is defined at every distance. ALiBi experiments
test training on short contexts and evaluating on longer ones;
their results apply to the studied configurations, not all tasks.
At a large distance, the penalty can suppress a needed key.
Being able to extend the matrix is therefore insufficient.

## One example, six interventions

Place the query at $i=5$ and the key at $j=2$, giving $j-i=-3$.

- **Learned absolute:** the first layer receives $e(t_5)+P_5$ and
  $e(t_2)+P_2$; distance must emerge through the learned projections.
- **Sinusoidal:** it receives $e(t_5)+PE_5$ and $e(t_2)+PE_2$,
  with vectors supplied by a fixed formula.
- **Shaw relative:** the offset-class vector
  $\operatorname{clip}(-3,k)$ is added to $k_2$ for the edge.
  The full variant also adds a positional vector to $v_2$.
- **T5 bias:** a scalar for the bucket containing $-3$ is added
  to the head's logit.
- **RoPE:** $q_5$ and $k_2$ rotate separately; their dot product
  contains relative rotation $R_{-3}$.
- **ALiBi:** the causal decoder subtracts $3m_h$ from the logit.

## Comparing the methods

| Method | Where computation changes | Position parameters | Relative distance | Beyond training length |
|---|---|---:|---|---|
| Learned absolute | $x_i\leftarrow e(t_i)+P_i$ before layer one | $T_{max}d$ | implicit through $P_i,P_j$ | unused rows are untrained; positions outside the table have no row |
| Sinusoidal | $x_i\leftarrow e(t_i)+PE_i$ before layer one | 0 | extractable from phase pairs | formula exists; quality is not guaranteed |
| Shaw relative | vector $a^K_{ij}$ in the score, sometimes $a^V_{ij}$ in the message | relative-vector tables | explicit, usually clipped | far distances share a limiting class |
| T5 bias | scalar bucket bias in each head's logits | bucket count × $H$ per stack | explicit and bucketed | new distances enter coarse far buckets |
| RoPE | Q/K rotation in each attention layer | 0 | appears in the dot product | formula exists; extension quality needs separate evaluation |
| ALiBi | fixed linear logit bias | 0 | explicit by distance | linear continuation may impose a strong far penalty |

The positional method is part of a trained architecture.
Replacing a learned table with RoPE changes projection inputs and scores
even if all tensor shapes match. Replacing one code class does not
turn existing weights into the weights of another positional model.

## Extrapolation: what needs to be evaluated

“Supports 128K” may mean only that the code does not crash.
Evaluation needs at least four independent properties:

1. Local quality: perplexity and target-task metrics at training lengths.
2. Use of distance: fact retrieval and synthetic tasks whose answers
   occupy different context positions.
3. Generation stability: no degradation, repetition, or loss of
   instructions during a long answer.
4. Computational feasibility: KV-cache memory and attention cost,
   neither of which a position method reduces by itself.

For RoPE, scaling indices $i\mapsto i/s$ differs from changing the base.
Index scaling divides all frequencies by $s$. Changing the base affects
frequencies differently: with $d_h=4$, changing base 10000 to 40000
changes $(1,0.01)$ to $(1,0.005)$, whereas dividing indices by four
gives effective frequencies $(0.25,0.0025)$. These are not interchangeable.

Compressing indices also reduces the phase separation between neighbors.
Checkpoint compatibility and training on longer data require separate checks.
[[02 Areas/ML & DL/00 Учебник/08 Эффективный Attention и длинный контекст/03 Длинный контекст — расширение, разреженность и оценивание|The long-context chapter]]
continues the discussion of context-extension methods.

Accepting 128K tokens does not establish equally good retrieval at 8K
and 120K. One successful needle-in-a-haystack example is weaker evidence
than quality curves across positions, lengths, and multiple task types.

## Practical errors that shapes do not reveal

- During cached generation, a new token needs an absolute `position_id`
  continuing after the cached keys, not zero again.
- In left-padded batches, real-token positions must follow the training
  convention. A tensor row index is not always its `position_id`.
- RoPE implementations differ in coordinate pairing (`rotate_half`
  or interleaved), base, and scaling. A checkpoint mismatch can preserve
  shapes while giving incorrect results.
- T5 bucket modes differ for bidirectional encoders and causal decoders.
  Their class numbers encode directions and exact distances differently.
- Padding masks, causal masks, and positional biases can all enter the
  logits while serving different purposes. No position method hides padding.

## Where the operations sit in a complete block

In the original encoder–decoder Transformer, sinusoidal vectors are
added to embeddings before the first layer. Later layers carry this
position signal through attention, FFNs, and residual connections with
the other features; it is not added again in every block.

RoPE instead rotates each layer's newly projected Q/K.
T5 and ALiBi add position terms to each layer's scores without changing
V through that operation. These choices determine where the operations
belong in the complete block assembled next.

## Sources and further reading

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762): sinusoidal positional encoding.
- Shaw, Uszkoreit, and Vaswani, [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155): vector-valued relative key/value representations.
- Raffel et al., [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://jmlr.org/papers/v21/20-074.html): T5 relative buckets and architecture.
- Su et al., [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864): RoPE derivation and experiments.
- Press, Smith, and Lewis, [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409): ALiBi.
- Stanford CS224N, [Self-Attention and Transformers lecture notes](https://web.stanford.edu/class/cs224n/readings/cs224n-self-attention-transformers-2023_draft.pdf): permutation equivariance and relative positions.
- Zhang et al., [Dive into Deep Learning: Self-Attention and Positional Encoding](https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html): executable sinusoidal examples.
- Stanford CS336, [Lecture 3, slides 30–35](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_03.pdf#page=30): absolute-encoding cross-terms, rotations, and RoPE's position in a layer.
- Hugging Face, [T5Attention in Transformers v4.57.1](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/models/t5/modeling_t5.py): exact buckets, sharing, and score scale.
- Press et al., [pinned ALiBi implementation](https://github.com/ofirpress/attention_with_linear_biases/blob/4b92f28a005ead2567abe2359f633e73e08f3833/fairseq/models/transformer.py#L742-L762): head slopes and cancellation of row constants.
- Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/): adding token and position vectors.

**Previous:** [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/Masking, multi-head и формы тензоров|Masks, heads, and shapes]] ·
**Next:** [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/03 Полный Transformer|The complete Transformer]].
