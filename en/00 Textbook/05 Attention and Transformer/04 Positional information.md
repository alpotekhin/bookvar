---
title: Positional information in Transformers
type: textbook-chapter
status: canonical
last_updated: 2026-08-03
primary_sources:
  - https://arxiv.org/abs/1706.03762
  - https://arxiv.org/abs/1803.02155
  - https://jmlr.org/papers/v21/20-074.html
  - https://arxiv.org/abs/2104.09864
  - https://arxiv.org/abs/2108.12409
---

# Positional information in Transformers

`The dog bit the man` and `the man bit the dog` contain the same words but
describe different events. Self-attention does not discover this difference by
itself. Permuting the rows of the input matrix permutes query, key, value, and
output rows in the same way. Without another signal, the operation knows
**which** tokens are present but not **where** they occur.

A causal mask solves a different problem. It prevents position $i$ from reading
future keys $j>i$, but does not say whether an allowed key is one or one hundred
tokens to the left. A padding mask merely hides empty slots. Order and distance
must be introduced separately.

Position methods are easiest to compare by where they intervene:

1. **Before the first layer:** add an absolute vector to the token embedding.
2. **Inside the score:** add a term depending on relative offset $j-i$ to
   $q_i^\top k_j$.
3. **Before the score:** transform query and key so their dot product depends
   on $j-i$.

These are not equivalent notations for one technique. They use parameters
differently, interact with heads differently, and extrapolate differently
beyond training length.

## Starting point: where position can enter attention

For one head without a position signal,

$$
q_i=x_iW_Q,\qquad k_j=x_jW_K,\qquad v_j=x_jW_V,
$$

$$
s_{ij}=\frac{q_i^\top k_j}{\sqrt{d_h}}+M_{ij},\qquad
a_{ij}=\operatorname{softmax}_j(s_{ij}),\qquad
z_i=\sum_j a_{ij}v_j.
$$

$M_{ij}$ is a causal or padding mask. Every method below modifies $x_i$,
$s_{ij}$, or $q_i,k_j$. When reading code, locate that intervention rather
than merely searching for a class named `PositionEmbedding`.

## Absolute positions: an address for each row

### Learned table

The most direct method stores
$P\in\mathbb R^{T_{max}\times d_{model}}$ and feeds the first layer

$$
h_i^{(0)}=e(x_i)+P_i.
$$

BERT and early GPT models use this design. Position $i$ receives its own
learned vector $P_i$ with the same width as a token embedding. For a batch
`[B,T,d]`, the first `T` rows of `[T_max,d]` are added to every example.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/transformer_positional_encoding_vectors.png]]

*A position vector is added to a token embedding before the first Transformer
block. Follow the signal path rather than individual coordinate values: all
subsequent $W_Q$, $W_K$, and $W_V$ receive the sum of content and position.
Source: Jay Alammar, [The Illustrated
Transformer](https://jalammar.github.io/illustrated-transformer/), [CC
BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

A learned table is simple and lets the model choose useful position geometry.
Rows unseen in training, however, are undefined. A checkpoint trained with
$T_{max}=2048$ has no $P_{4096}$. Interpolation or further training can extend
the table, but that changes the model rather than providing free extrapolation.

There is another limitation. A relation such as “the key is three places to
the left” must be learned through a particular pair $P_i,P_j$. Shift the same
text span to the right and the model sees another address pair even though its
relative structure is unchanged.

### Sinusoidal encoding

The original Transformer used fixed sines and cosines:

$$
PE_{i,2k}=\sin\left(i/10000^{2k/d}\right),\qquad
PE_{i,2k+1}=\cos\left(i/10000^{2k/d}\right).
$$

Like a learned vector, $PE_i$ is added to the token embedding. Unlike a table,
the formula adds no parameters and can compute a code for any integer $i$.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-25-31/d2l-sinusoidal-curves.svg]]

*Several sinusoidal coordinates along a sequence. Fast oscillations distinguish
neighboring positions; slow ones supply a larger-scale signal. Source: Zhang et
al., [Dive into Deep Learning,
§11.6](https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html),
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-25-31/d2l-positional-heatmap.svg]]

*The same signal as a matrix: rows are positions and columns coordinates.
Different periods coexist, describing every position at several scales. Source:
Zhang et al., [Dive into Deep Learning,
§11.6](https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html),
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*

For a fixed shift $r$, $PE_{i+r}$ is a linear transformation of $PE_i$ by the
angle-addition identities. Attention projections can therefore in principle
extract relative offset. *In principle* does not mean that a model trained at
2K retains quality at 32K. Beyond training it encounters new phase
combinations, and neither data nor attention optimization taught it to use
those distances.

## Shaw relative representations: distance becomes part of an edge

Shaw, Uszkoreit, and Vaswani encode directed offset $j-i$ rather than addresses
$i$ and $j$ separately. One variant uses

$$
s_{ij}=\frac{q_i^\top(k_j+a^K_{ij})}{\sqrt{d_h}},\qquad
z_i=\sum_j a_{ij}(v_j+a^V_{ij}).
$$

$a^K_{ij}$ and $a^V_{ij}$ are selected from learned tables by clipped distance:

$$
\operatorname{clip}(j-i,k)=\max(-k,\min(k,j-i)).
$$

Distances beyond $k$ share one class. The score term changes the **weight of an
edge**: a query compares with both key content and relative-position type. The
value term changes the **message**: the same value may contribute differently
by direction and distance.

With $k=2$, a query at position 4 distinguishes keys 3 and 4 as offsets $-1$
and $0$, while positions 0, 1, and 2 all enter the far-left class $-2$. Shift
the whole fragment ten positions and the distance classes remain unchanged.
This is the translation invariance of a relative scheme.

The cost is more complex computation and memory for pairwise representations.
Later work simplifies how the relative signal is added.

## T5 relative position bias: one scalar per bucket and head

T5 adds no positional embedding to the residual stream. Each attention head
instead receives a learned logit bias:

$$
s_{ij}^{(h)}=\frac{(q_i^{(h)})^\top k_j^{(h)}}{\sqrt{d_h}}
+b_h(\operatorname{bucket}(j-i))+M_{ij}.
$$

Unlike Shaw's vector, $b_h$ is a scalar. It changes softmax but not the Value.
Heads can learn different preferences: one may amplify the previous token,
another may barely penalize long-range edges.

Buckets preserve exact short distances and group longer intervals
logarithmically. An illustrative set is

$$
0,1,2,3,4,\quad 5\!:\!7,\quad 8\!:\!15,\quad 16+.
$$

These are not exact T5 configuration boundaries; they illustrate the principle.
Local syntax benefits from precise short distance, while long dependencies can
use a coarser scale. A bidirectional encoder distinguishes negative and
positive offsets. A causal decoder contains only the present and past.

The bucket function can be evaluated for longer input, but every distance past
the final boundary receives the same bias: the model knows “very far,” not how
far. This is controlled generalization, not an exact coordinate system.

## RoPE: position as a rotation of query and key

Rotary Position Embedding neither adds a vector to $x_i$ nor a separate bias to
the score. It rotates coordinate pairs **after** the $W_Q,W_K$ projections:

$$
q_i'=R_iq_i,\qquad k_j'=R_jk_j,
$$

with one coordinate pair using

$$
R_i(\theta)=
\begin{bmatrix}
\cos(i\theta)&-\sin(i\theta)\\
\sin(i\theta)&\cos(i\theta)
\end{bmatrix}.
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-first-32-34/roformer-rope-figure1.png]]

*Two query or key coordinates form a point in a plane and rotate by a
position-dependent angle. Compare one source vector under different
$m\theta$: length stays fixed while phase changes. Source: Su et al.,
[RoFormer, Figure 1](https://arxiv.org/pdf/2104.09864#page=5), 2021.*

The key derivation is

$$
(R_iq_i)^\top(R_jk_j)
=q_i^\top R_i^\top R_jk_j
=q_i^\top R_{j-i}k_j.
$$

Each vector rotates according to absolute position, but their dot product
depends on relative offset $j-i$. Values are not rotated: RoPE changes the
selection path, not the information being transmitted.

In a real head, each coordinate pair has its own frequency

$$
\theta_k=\text{base}^{-2k/d_h}.
$$

High frequencies change phase quickly and distinguish nearby positions; low
frequencies form a slow scale for long distances. This resembles sinusoidal
encoding, but its insertion point is different: sinusoids enter the residual
stream once; RoPE transforms Q and K in every attention layer.

### A small numerical RoPE example

Take one coordinate pair $q=(1,0)$, $k=(1,0)$, and $\theta=\pi/2$. Query is at
$i=1$ and key at $j=2$.

1. $R_1q=(0,1)$: query rotates by $90^\circ$.
2. $R_2k=(-1,0)$: key rotates by $180^\circ$.
3. Their score is $(0,1)^\top(-1,0)=0$.
4. The relative form agrees: $q^\top R_{2-1}k=(1,0)^\top(0,1)=0$.
5. Shift both positions by five; relative distance remains one and this pair's
   score is unchanged.

A full head sums all two-dimensional pairs at different frequencies. The
example demonstrates invariance to a common shift, not the entire head.

RoPE can be computed beyond the training window, but new positions produce
unseen phases and distances. Long-context models therefore change `base`,
interpolate position indices, or continue training on longer sequences. “RoPE
scaling” covers several distinct recipes; compatibility must be checked in the
checkpoint configuration.

## ALiBi: a linear distance penalty

Attention with Linear Biases creates neither embeddings nor rotations. In
causal attention, each head receives a fixed linear penalty:

$$
s_{ij}^{(h)}=\frac{(q_i^{(h)})^\top k_j^{(h)}}{\sqrt{d_h}}
-m_h(i-j),\qquad j\le i,
$$

where $m_h>0$ is a predetermined slope. Farther keys receive smaller logits.
Different heads have different slopes: some specialize locally, while flatter
ones can retain long-range connections.

For a query at position 5, identical content logits `2.0`, and $m_h=0.25$:

| key $j$ | distance $5-j$ | bias | final logit |
|---:|---:|---:|---:|
| 4 | 1 | $-0.25$ | 1.75 |
| 3 | 2 | $-0.50$ | 1.50 |
| 1 | 4 | $-1.00$ | 1.00 |

Softmax favors the nearest key despite equal content similarity. Shifting all
positions preserves the bias because only distance matters.

ALiBi was designed for train-short, test-long: a linear function is defined at
any distance and introduces no unseen absolute rows. Its paper reports markedly
better length extrapolation than the compared absolute baselines. This is not a
universal guarantee. The bias systematically favors nearby keys, outcomes
depend on task and scale, and many modern long-context models instead combine
RoPE with a separate scaling recipe.

## One pair, six interventions

Let query be at $i=5$, key at $j=2$, so $j-i=-3$.

- **Learned absolute:** the first layer receives $e(x_5)+P_5$ and
  $e(x_2)+P_2$; distance must emerge through learned projections.
- **Sinusoidal:** it receives $e(x_5)+PE_5$ and $e(x_2)+PE_2$, with vectors
  fixed by formula.
- **Shaw relative:** the offset-class vector $-3$ is added to $k_2$ for the
  edge; the full variant also adds a position vector to $v_2$.
- **T5 bias:** one scalar for the bucket containing $-3$ is added to the head
  logit.
- **RoPE:** $q_5$ and $k_2$ rotate separately; their dot product contains
  relative rotation $R_{-3}$.
- **ALiBi:** a causal decoder subtracts $3m_h$ from the logit.

The model “knows order” in all six cases, but the data path is different.

## Method comparison

| Method | Where computation changes | Position parameters | Relative distance | Beyond training length |
|---|---|---:|---|---|
| Learned absolute | $x_i\leftarrow e(x_i)+P_i$ before layer one | $T_{max}d$ | implicit via $P_i,P_j$ | no new table rows; interpolate or train further |
| Sinusoidal | $x_i\leftarrow e(x_i)+PE_i$ before layer one | 0 | extractable from phase pairs | formula exists; quality is unproven |
| Shaw relative | vector $a^K_{ij}$ in score, sometimes $a^V_{ij}$ in message | relative-vector tables | explicit, usually clipped | far distances merge into the edge class |
| T5 bias | scalar bucket bias in each head's logits | $H\times B$ with sharing | explicit and bucketed | new distances enter coarse far buckets |
| RoPE | rotate Q and K in each layer | 0 | emerges in the dot product | computable, but requires validated scaling/training |
| ALiBi | fixed linear bias in logits | 0 | explicit by distance | linear continuation may impose a strong far penalty |

There is no winner from the table alone. Absolute embeddings work well for
fixed-window encoders. T5 bias fits encoder–decoder models and can distinguish
directions. RoPE is standard in many decoder-only LLMs and works with KV cache
when `position_ids` are correct. ALiBi is attractive when length extrapolation
is part of the experiment from the outset.

## Extrapolation: what must actually be tested

“Supports 128K” may mean only that the code does not crash. At least four
independent properties matter:

1. **Local quality:** perplexity or task quality inside the training window.
2. **Distance use:** retrieval and synthetic tasks with answers at varied
   context positions.
3. **Generation stability:** no collapse, repetition, or instruction loss over
   long decoding.
4. **Computational feasibility:** attention and KV-cache cost, which a position
   method does not itself reduce.

A RoPE model accepting 128K tokens has not thereby shown equal retrieval at 8K
and 120K. One successful needle-in-a-haystack example is weaker evidence than
quality curves over positions, lengths, and several task types.

## Practical errors invisible from shapes

- During cached decoding, a new token needs an absolute `position_id`
  continuing after cached keys, not zero again.
- In a left-padded batch, real-token positions must follow the training
  convention; tensor row index is not always the `position_id`.
- RoPE implementations differ in coordinate pairing (`rotate_half` versus
  interleaving), `base`, and scaling. A checkpoint mismatch preserves shapes
  while destroying quality.
- T5 bucket functions differ between bidirectional encoder and causal decoder;
  direction handling cannot be reused blindly.
- Padding masks, causal masks, and position bias all add to logits but carry
  different meanings. No position method hides padding.

## Summary

Positional information is not necessarily “an embedding next to the token
embedding.” It is any mechanism that breaks attention's permutation symmetry.
Learned absolute and sinusoidal methods alter the layer input. Shaw and T5
modify an edge between positions. RoPE changes query–key geometry. ALiBi gives
each head a distance prior. When reading a model, ask: **where does position
enter the equations, and what does the model encounter beyond its training
range?**

## Sources and further reading

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — sinusoidal encoding.
- Shaw, Uszkoreit, Vaswani, [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155) — relative key/value vectors.
- Raffel et al., [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://jmlr.org/papers/v21/20-074.html) — T5 buckets.
- Su et al., [RoFormer](https://arxiv.org/abs/2104.09864) — RoPE derivation and experiments.
- Press, Smith, Lewis, [Train Short, Test Long](https://arxiv.org/abs/2108.12409) — ALiBi.
- Stanford CS224N, [Self-Attention and Transformers lecture notes](https://web.stanford.edu/class/cs224n/readings/cs224n-self-attention-transformers-2023_draft.pdf) — permutation equivariance and relative positions.
- Zhang et al., [Dive into Deep Learning: Self-Attention and Positional Encoding](https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html) — executable sinusoidal examples.
- Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — adding token and position vectors.

**Next:** [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/03 Полный Transformer|assemble attention, positional signal, FFN, and the residual path into a complete Transformer]].
