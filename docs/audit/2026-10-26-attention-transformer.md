# Self-attention and the complete Transformer: source comparison

## Current artifacts

- `00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V.md`:
  738 lines, SHA-256
  `073f6e573481b59e93f2a2bee4bdd2e86c7ab5e1c9091764e9c506bfd18acc3c`.
- `00 Учебник/05 Attention и Transformer/03 Полный Transformer.md`:
  1094 lines, unchanged, SHA-256
  `8bda8af63d82960da7d5849e132be647542b90d4a9a9daca05ff5a0f2e66d356`.

Both current chapters were read completely, not selected by search matches.
The accepted self-attention explanation was retained. Removing the bounded
`attention-identifiability` addition and restoring its update date recovers
the previous 672-line revision, SHA-256
`d402a9a6810e8076173faf2e0b0e9043dbffa3673d7edb794e16e9c29f7ffd60`.
The checker and unit test enforce that preservation. All old figures,
code blocks, examples and anchors remain.

The resulting self-attention chapter was reread in order in three chunks
through EOF. The only later reader edit shortened the two table cells
from `q=±ln2` to `±ln2`, since the column already says Query and the
longer notation wrapped on mobile. The final table and surrounding
explanation were checked again in the browser.

## Explanation comparison

The relevant Harvard *Network Architectures* sections are attention,
lines 2234–2929, and Transformer, lines 2930–3458. Its pinned source is
`2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd`, SHA-256
`64ef2c6bab3a09846bdb089a33c0ee355ee32ecbf291447493708c276f18d9a6`.
The separate original published source page remains unchanged, SHA-256
`52c77d7f6d386e97af65decf4f55523be785930571759fe5a906069ad13fcfac`.

| Harvard mechanism | Current explanation / worked evidence | Decision |
| --- | --- | --- |
| Context-dependent representations | Mole example, prefix averaging, shared projections and directed scores | Already explained; no second synthetic introduction added |
| Q/K/V and scaled dot-product | All dimensions, separate value width, full hand calculation, softmax axis and variance assumptions | Existing explanation is more explicit |
| Learning the projections | Manual value/weight/softmax/Q/K gradients, shared-X gradient sum, finite differences and one SGD step | Retained in full |
| Parallel positions and sequence order | Qualified permutation equivariance, mask/PE boundary, separate generation dependence | Retained; not described as parallel generation |
| Attention cost | Pairwise work separated from projection work; full matrix storage identified as direct implementation | Harvard's unconditional square-memory claims not transferred |
| Residual, norm and FFN | Numerical LayerNorm/pre/post-norm, FFN forward/backward, gated-width comparison | Retained; no universal gradient-stability guarantee added |
| Encoder and decoder | Full post-norm translation model, rectangular cross-attention, PAD handling and different parameters by depth | Already covered with executable examples |
| Training versus generation | Shifted targets, valid-target denominator, CE/KL/smoothing distinctions, stored source memory and growing target prefix | Existing worked treatment retained |
| Modern causal architecture | RMSNorm, RoPE Q/K, SwiGLU, final norm and head; BERT/GPT-1/GPT-2/LLaMA boundaries | Retained; not confused with the 2017 translator |

The selected D2L self-attention/complexity section was also reread from
its pinned local source through the positional-encoding transition.
The primary references already used by the full Transformer are recorded
in its historical 21 October acceptance; this pass did not reread every
paper in that bibliography or claim full CS336 assignment execution.

## One explanatory gap filled

The sentence saying that different attention weights can yield the same
output lacked a worked example. The new subsection supplies one without
changing the rest of the chapter. Values `(1,0)`, `(0,1)`, `(.5,.5)` and
scalar keys `[0,0,1]` yield weights `[.25,.25,.5]` for `q=ln2` and
`[.4,.4,.2]` for `q=-ln2`. Both outputs are `(.5,.5)`.

The example constructs Q/K through shared projections of `X=I3`; all
logits are finite, so neither zero probabilities nor arbitrary
unrealizable stochastic matrices are assumed. The difference row sums
to zero and annihilates V. The text preserves determinism for fixed
inputs/parameters and does not claim arbitrary replacement is possible
in every head. It distinguishes mixing coefficients from a demonstrated
causal-importance interpretation.

Brunner et al., arXiv:1908.04211v3 §§2–3.2, were read for the relevant
rank/null-space question. Only a citation is reused; no paper figure or
passage was copied. The example is independent, not a verbatim course
excerpt. Source record:
`05 Источники/Originals/attention-identifiability-2026-10-26/README.md`.

The no-ai-slop check left the accepted subject-first exposition alone.
The bounded English example replaces no original explanation and adds
no dramatic framing or generic recap. It makes a previously unsupported
short caveat checkable instead of shortening the chapter.

## Executed checks

`check_self_attention_examples.py` executes all four printed blocks and
the previous exact-variance, hand-gradient, nonidentity-X, scalar-loop,
SDPA, finite-difference and permutation checks. The original worked SGD
loss remains `2.0465561246435064 → 1.8381402347944618`.
Its final SHA-256 is
`e09062b5b68072268b43aaa4f7a0333936c5201f007eb5e6467ba90cea5bdb7a`.
The new checks additionally cover 161 finite queries in `[-20,20]`:
the weights change with a nonzero derivative while the output remains
fixed and its query derivative is zero to numerical tolerance.

`check_full_transformer_examples.py` executes all five unchanged printed
blocks. Both post-norm layers match PyTorch outputs and all
input/memory/parameter gradients; the complete translation graph passes
causality, source-PAD, tied-gradient and detached-memory negative controls.
The existing translation loss remains
`9.352584927036121 → 8.497538101285013`. The modern causal model passes
dense-RoPE/SDPA, all attention gradients, RMSNorm gradcheck and SwiGLU
checks. This is not a benchmark, translation-quality evaluation or the
official CS336 test suite.

The CPU commands use cached PyTorch 2.14.1 / NumPy 2.3.5. A requested
Pillow 12.2.0 was not available in the offline cache; no installation was
attempted. The source-image checker then ran successfully with the existing
system Python 3.13 / Pillow 12.1.1. It matches all seven Transformer
objects byte-for-byte and decodes all 334 GIF frames. The four existing
self-attention figures remain covered by the central asset-registry tests.

Logs for the accepted revision:

- `/var/tmp/bookvar-attention-identifiability-examples-final.log`:
  four printed blocks and all old/new checks pass, current chapter SHA.
- `/var/tmp/bookvar-transformer-recheck-examples.log`: all five printed
  blocks and independent checks pass, unchanged chapter SHA.
- `/var/tmp/bookvar-transformer-recheck-figures.log`: all source bytes,
  dimensions and animation frames pass.
- `/var/tmp/bookvar-attention-transformer-unit.log`: 65 files / 516 tests
  pass on the full insertion before the two notation-only table edits.
- `/var/tmp/bookvar-attention-identifiability-unit-final.log`: one
  affected test passes after those table edits; CPU checks also rerun.
- `/var/tmp/bookvar-attention-transformer-build-release.log`: final
  current-source-map build, 1969 pages / 22.58 s, complete at 23:27:31.
- `/var/tmp/bookvar-attention-transformer-output-release.log`: 136 tests pass.
- `/var/tmp/bookvar-attention-transformer-links-release.log`: zero broken
  internal routes, fragments or files.
- `/var/tmp/bookvar-attention-transformer-coverage-release.log`: structural
  source coverage passes, 210 rows / 58 destinations. This does not prove
  thematic completion of the entire course.

## Rendered figures and locales

All four self-attention and seven Transformer source figures were opened
and inspected in the actual viewer at desktop 1280×900 and mobile 360×800.
They remain complete source objects; no redraw or crop was substituted.
The complex attention-stage and encoder/decoder residual figures were
also checked at 3×, including both lateral edges and the bottom.
Small labels are not all readable in a 344-pixel fit thumbnail: zoom is
still needed. The original source resolution is not called unlimited.

Both pages have body width 360/360. All 21 self-attention and 12
Transformer display-formula containers fit 328 pixels; no KaTeX error
was observed. The new table fits without horizontal scrolling, with both
query values on single lines after the final notation edit.

Saved views:

- `screenshots/2026-10-26-attention-identifiability-desktop.png`.
- `screenshots/2026-10-26-attention-identifiability-mobile.png`.
- `screenshots/2026-10-26-attention-stages-mobile-right-bottom.png`.
- `screenshots/2026-10-26-transformer-residual-mobile-right-bottom.png`.

The EN routes were visited, not assumed to be fallback. Both are separate
older English bodies dated 15 September. The new identifiability example
is absent from EN self-attention; EN full Transformer has only three
images, compared with seven in the canonical chapter. English content
parity is therefore not accepted. EN editing remains deferred by the
user's instruction; no EN file was changed. Current EN hashes:
`0abb7aad9c2d76b10cb0894cc69c4f3c321eadaf7045d790e07a7c94b52d1b8a`
and `dbb6f8470d6ad92af518013a8ecbbdaacdcc49fdd10655a50309154e727e0223`.

The browser was restored to the new RU self-attention subsection on the
release build, the viewer closed and the temporary viewport reset.

## Harvard source reading and remaining integration

Sequential source reading now reaches EOF, line 5256. This includes all
remaining selection, pitfalls, summary, code and TikZ; the long numeric
zebra paths earlier were inspected in the whole archived SVG rather than
read coordinate-by-coordinate. The whole external native-figure directory
has not been reviewed. A finished source read is not finished integration.

The later normalization/systems claims were checked before deciding to
transfer them. Independent CPU counterexamples give spectral radii
`rho(A)=rho(B)=0.5` but `rho(B@A)=4.48606797749979` for
`A=[[.5,2],[0,.5]]`, `B=A.T`. A normalization Jacobian annihilates the
all-ones translation direction, contradicting a universal eigenvalue
interval `[.5,2]`. A residual branch `F(x)=-x` cancels the identity
derivative. None of these source guarantees was added to the textbook.

Other pending source dispositions include the im2col caption's row/column
orientation, static versus data-dependent execution, hardware-specific
latency/utilization/energy numbers and hypothetical wildlife accuracy.
The source's final square-memory claim contradicts its own FlashAttention
description; its linear KV-cache formula is a different allocation.

Next: compare the whole current ranking/recommender chapter with the
DLRM section, then the normalization/gating and systems chapters with
the now-read remaining source sections. Native figure disposition remains
open. H-V1-06 remains integration-partial; H-F1-06 remains re-review.
No RL editing, translation, stage/commit/push, deployment or infrastructure
mutation occurred. The complete textbook/course goal remains active.
