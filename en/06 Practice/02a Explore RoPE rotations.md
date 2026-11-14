---
title: "RoPE: positions and rotations"
type: practice
status: active
locale: en
translation_status: current
canonical_revision: "4cb50a26b79d92dbf2722b1a9dad8a8e6ce6d94d0760e5155388ebf10e767bad"
translation_reviewed: "2026-11-10"
translation_of: "06 Практика/02a Проверить повороты RoPE.md"
last_updated: 2026-11-10
---

# RoPE: positions and rotations

## Common shifts and relative positions

Use $q=(1,2)$, $k=(3,4)$, and a frequency of $\pi/6$ radians per
position. With $m=2$ and $n=5$, the relative rotation is $90^\circ$.
It sends $k$ to $(-4,3)$, so the dot product is $-4+6=2$.

Shift both positions from $(2,5)$ to $(3,6)$. Both vectors turn another
$30^\circ$, leaving the angle between them unchanged. Now swap only the
positions and keep $q$ and $k$ fixed. The offset is $n-m=-3$, the
relatively rotated key is $(4,-3)$, and the dot product is $-2$.
This catches a sign error that would go unnoticed with identical vectors.
The exercise omits causal masking, which is applied to the scores afterward.

See the [original derivation and figure in RoFormer, §3.2](https://arxiv.org/html/2104.09864v5#S3.SS2)
and the [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/04 RoPE|RoPE chapter]].

## Distance and frequency in Llama 3.1

Pair 32 has an original period of about 4443 positions. It completes about
1.84 turns over the original 8192 positions, so the new frequency is about
0.37112 times the old one. At a separation of 1000 positions, the two angles
are 1.41421 and 0.52485 radians. For $q=k=(1,0)$, their cosines are the
corresponding contributions to the dot product.

Set the distance to zero. Both angles become zero and both cosines become
one, but the frequency ratio does not change. The scaling band depends on
the pair's period, not on the current separation. Compare pairs 0 and 40:
their ratios are 1 and $1/8$, respectively.

See [Meta's `apply_scaling`](https://github.com/meta-llama/llama-models/blob/f45cdfd624b98b6655540f7101d8d9cb432e631c/models/llama3_1/reference_impl/model.py#L45-L79)
and the [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/04 RoPE#rope-llama31-scaling|Llama 3.1 scaling walkthrough]].
