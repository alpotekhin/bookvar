---
title: Masking, multi-head attention, and tensor shapes
type: textbook-chapter
status: canonical
last_updated: 2026-08-03
primary_sources:
  - https://arxiv.org/abs/1706.03762
  - https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html
  - https://jalammar.github.io/illustrated-transformer/
  - https://d2l.ai/chapter_attention-mechanisms-and-transformers/multihead-attention.html
---

# Masks, multiple heads, and tensor shapes

One attention head consumes $Q$, $K$, and $V$ matrices and returns a new matrix
of representations. A real model needs more detail. One batch contains
sentences of different lengths; an autoregressive model must not read future
tokens; and several heads must run together without Python loops. The formula
becomes a reliable implementation only after three questions are answered:
which position pairs are allowed, where the head index lives in the tensor, and
along which axis softmax is normalized.

## A mask changes the set of accessible keys

Let $S=QK^\top/\sqrt{D_h}$ be the score matrix. The mask is added to $S$
**before** softmax:

$$
A=\operatorname{softmax}(S+M),\qquad Z=AV.
$$

For an allowed cell $M_{ij}=0$; for a forbidden cell it is $-\infty$.
Exponentiation maps a forbidden cell to zero, so it contributes neither to the
weighted sum nor to normalization of the remaining weights. Zeroing an already
normalized $A$ is insufficient: allowed weights no longer sum to one unless
they are renormalized.

### Causal mask: do not peek at the answer

During language-model training, the complete sequence is known and all
positions run in parallel. Yet the state at position $i$, from which the model
predicts the next token, may depend only on positions $j\le i$. Otherwise the
network sees the correct continuation in its input and the objective becomes
meaningless. A lower-triangular mask enforces this constraint:

$$
M^{\text{causal}}_{ij}=
\begin{cases}
0,&j\le i,\\
-\infty,&j>i.
\end{cases}
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-15-24/causal-mask-detailed.png]]

*A causal mask in decoder self-attention: each row retains only the current and
earlier positions. Image by Daniel Voigt Godoy, [original and CC BY 4.0
license](https://commons.wikimedia.org/wiki/File:Decoder_self-attention_with_causal_masking,_detailed_diagram.png).*

Lena Voita presents the same transition as an [animation of masked
self-attention](https://lena-voita.github.io/resources/lectures/seq2seq/transformer/masked_self_attn.mp4): as the target position advances, the accessible
lower-triangular region grows by one cell.

The mask does not make training sequential. Every row of $S$ is still computed
with one matrix multiplication; forbidden edges are removed before softmax.
Generation remains sequential because the next input token is not yet known.

### Padding mask: do not read filler tokens

Sequences of different lengths are usually extended with a special token to a
common length $T$. These positions are necessary for tensor shape but contain
no text. If only the first five tokens of the second example are real, its key
mask is `[1, 1, 1, 1, 1, 0, 0]`: every real query can read the first five keys
but not the two fillers.

A causal mask depends on `(query position, key position)` and has shape
`[T,T]`. A padding mask depends on example and key position, so its natural
shape is `[B,T]`. A decoder-only model needs both:

$$
M_{b,i,j}=M^{\text{causal}}_{i,j}+M^{\text{padding}}_{b,j}.
$$

It is the **key** positions that must be masked. An output originating from a
padding query can also be zeroed later, but that is a different operation and
does not stop real tokens from reading filler keys.

## Why one position needs several heads

One softmax creates one distribution over keys. A word may simultaneously need
different relationships: its nearest neighbor, a governing subject, and a
modifier that disambiguates its meaning. Multi-head attention supplies several
independently learned projection sets and several attention distributions:

$$
\operatorname{head}_r=
\operatorname{Attention}(QW_r^Q,KW_r^K,VW_r^V),
$$

$$
\operatorname{MHA}(Q,K,V)=
\operatorname{Concat}(\operatorname{head}_1,\ldots,
\operatorname{head}_H)W^O.
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-15-24/qkv-for-heads.png]]

*Each head receives its own projections of queries, keys, and values. Source:
Lena Voita, [Multi-Head Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html#multi_head_attention). The original page also contains an animation of the complete operation.*

Several heads are not several full copies of a $D_{model}$-wide vector. In the
classic Transformer, total width is divided across $H$ heads:
$D_h=D_{model}/H$. Concatenation therefore restores width
$HD_h=D_{model}$, and $W^O$ mixes features across heads and returns them to the
residual stream.

## Split and merge: one operation, two notations

In the mathematical definition, every head owns separate
$W_r^Q,W_r^K,W_r^V$. Libraries usually concatenate those matrices. One
projection gives $Q,K,V\in\mathbb{R}^{B\times T\times D_{model}}$, after
which the last axis is split into head index and within-head features.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-15-24/transformer_attention_heads_qkv.png]]

*Jay Alammar shows the individual projections that produce each head's
queries, keys, and values. Source: [The Illustrated
Transformer](https://jalammar.github.io/illustrated-transformer/), CC
BY-NC-SA 4.0.*

For input $X$ of shape `[B,T,Dmodel]`, one projected tensor follows this path:

```python
q = q_proj(x)                    # [B, T, Dmodel]
q = q.view(B, T, H, Dh)          # [B, T, H, Dh]
q = q.transpose(1, 2)            # [B, H, T, Dh]
```

The axis permutation is not new model mathematics. It prepares batched matrix
multiplication by putting `[T,Dh]` in the final two axes. After attention, the
steps are reversed:

```python
z = weights @ v                  # [B, H, T, Dh]
z = z.transpose(1, 2)            # [B, T, H, Dh]
z = z.contiguous().view(B, T, Dmodel)
z = out_proj(z)                  # [B, T, Dmodel]
```

In PyTorch, `contiguous()` before `view()` matters: `transpose` changes memory
strides without rearranging the underlying data. `reshape()` may make the copy
automatically, but the axis permutation still must be understood.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-15-24/transformer_attention_heads_z.png]]

*Head results remain separate until concatenation. Source: Jay Alammar, [The
Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/),
CC BY-NC-SA 4.0.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-15-24/transformer_attention_heads_weight_matrix_o.png]]

*After concatenation, $W^O$ mixes head outputs and restores width
$D_{model}$. Source: Jay Alammar, [The Illustrated
Transformer](https://jalammar.github.io/illustrated-transformer/), CC
BY-NC-SA 4.0.*

## Complete shape trace

Let a batch contain $B$ sequences of length $T$, model width $D_{model}$,
$H$ heads, and head width $D_h=D_{model}/H$.

| Step | Tensor | Shape |
|---|---|---|
| input | $X$ | `[B, T, Dmodel]` |
| three linear projections | $Q,K,V$ | each `[B, T, Dmodel]` |
| split + transpose | $Q,K,V$ | each `[B, H, T, Dh]` |
| transpose keys | $K^\top$ | `[B, H, Dh, T]` |
| all pair scores | $S=QK^\top/\sqrt{D_h}$ | `[B, H, T, T]` |
| mask and softmax over keys | $A$ | `[B, H, T, T]` |
| weighted Values | $Z=AV$ | `[B, H, T, Dh]` |
| transpose + concatenate | $Z_{cat}$ | `[B, T, Dmodel]` |
| output projection | $Z_{cat}W^O$ | `[B, T, Dmodel]` |

Softmax runs over the **last** axis: for every `(b,h,i)`, the sum over keys $j$
is one. Normalizing over queries would answer a different question and change
the mechanism.

## Broadcasting masks

After splitting heads, scores have shape `[B,H,Tq,Tk]`. A mask need not
physically store all those elements: axes of length one broadcast.

| Purpose | Convenient shape | Broadcast axes |
|---|---|---|
| one causal mask for all examples | `[1, 1, T, T]` | batch and heads |
| key padding for each example | `[B, 1, 1, T]` | heads and queries |
| one arbitrary mask per example | `[B, 1, T, T]` | heads |

```python
scores = q @ k.transpose(-2, -1) / math.sqrt(Dh)  # [B,H,T,T]

causal = torch.ones(T, T, dtype=torch.bool).tril()
causal = causal[None, None, :, :]                   # [1,1,T,T]

key_ok = attention_mask[:, None, None, :].bool()   # [B,1,1,T]
allowed = causal & key_ok
scores = scores.masked_fill(~allowed, float("-inf"))
weights = scores.softmax(dim=-1)
```

Some APIs interpret boolean `True` as “allow”; others interpret it as “hide.”
The argument name is not enough—check the exact function's documentation. In
`torch.nn.functional.scaled_dot_product_attention`, `True` means an allowed
element, while several older PyTorch interfaces use the opposite convention.

## Plausible-looking implementation errors

- **Softmax along the wrong axis.** `weights.sum(-1)` should be all ones for
  rows with at least one allowed key.
- **Masking after softmax.** Forbidden weights become zero, but the remainder
  is not renormalized.
- **Confusing $T_q$ with $T_k$.** They may differ in cross-attention; scores
  always have shape `[B,H,Tq,Tk]`.
- **Silently incorrect broadcasting.** `[B,T]` does not align from the right
  with `[B,H,T,T]`; insert two singleton axes.
- **A fully masked row.** Softmax of all $-\infty$ yields `NaN`. Such a row
  needs an explicit policy, especially with left padding.
- **Forgetting to merge heads.** Averaging heads is not concatenation followed
  by a learned $W^O$.
- **Wrong scale.** Divide by $\sqrt{D_h}$, not $\sqrt{D_{model}}$.

## Minimal implementation checks

```python
assert Dmodel == H * Dh
assert q.shape == (B, H, T, Dh)
assert scores.shape == (B, H, T, T)
assert weights.shape == (B, H, T, T)
assert torch.allclose(
    weights.sum(dim=-1),
    torch.ones_like(weights.sum(dim=-1)),
    atol=1e-5,
)

# No causal head may assign weight to a future position.
future = torch.ones(T, T, dtype=torch.bool).triu(diagonal=1)
assert torch.count_nonzero(weights[..., future]) == 0
```

These checks connect the formula to actual axes. With them in place, a full
Transformer can be assembled from attention, feed-forward layers, residual
connections, and normalization without leaving masks and heads as library
“magic.”

## Sources and visual continuations

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), §§3.2.2–3.2.3 — original multi-head and masked-attention definitions.
- Lena Voita, [Seq2seq and Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html) — clear animations of masking and independent heads.
- Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — visual split, concatenation, and $W^O$.
- Dive into Deep Learning, [Multi-Head Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/multihead-attention.html) — implementation with explicit shape transforms.
- Harvard NLP, [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — subsequent-position masking and multi-head implementation alongside the equations.
- PyTorch, [`scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention) — exact semantics of `attn_mask`, `is_causal`, and dropout.

**Previous:** [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V|Self-attention from the inside]] ·
**Next:** [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/03 Полный Transformer|The complete Transformer]]
