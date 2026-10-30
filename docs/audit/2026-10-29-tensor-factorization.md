# Tensor decompositions: explanation review

Scope: a bounded addition to chapter 57b, not acceptance of the whole Harvard
Model Compression unit. Original English retained; no EN body changed and no
publication authorized. Exact current and previous hashes are in
[the addition journal](2026-10-29-tensor-factorization-addition.json).

## Sources and explanation

Read the complete 759-line chapter before editing. Read the full TensorLy
tensor-decomposition guide and TensorLy-Torch factorized-convolution guide;
the complete `tltorch/functional/convolution.py`; Harvard's low-rank and tensor
decomposition sections, including the original figure. The pinned text
snapshots and licenses are in `05 Источники/Originals/tensor-factorization-2026-10-29/`.

| Mechanism | Source | Retained explanation and concrete check |
| --- | --- | --- |
| CP | TensorLy guide, CP; Harvard tensor decomposition | Original definition; outer product slices; rank/storage; one explicit least-squares factor update; nonconvex joint-fit limitation |
| Tucker | TensorLy guide, Tucker | Both original explanatory paragraphs; off-diagonal core versus CP; the guide's 12×12 cross reconstructed with a 2×3 core and 66 stored entries |
| Tucker-2 convolution | TensorLy-Torch guide and `tucker_conv` | Derivation by regrouping sums; exact factor shapes; three convolution stages; bias/activation rules; 73,728 versus 9,728 weights; three stride/dilation cases |
| CP convolution | TensorLy-Torch `cp_conv` | Input/output projections and two depthwise stages; exact output equality; spatial separability and rank/quality caveat |
| TT | TensorLy guide, TT | Chain indices, scalar contraction yielding 11; 16-versus-8 small counterexample; 720-versus-1,000,000 count with bounded-rank qualification |

The old guide's CP return-value example and older stable site's `CPConv`
class names were not copied as current runnable APIs. The cross data and
original explanations are preserved; runnable calculations use NumPy/PyTorch.
Five new programs executed on CPU, then all eleven chapter programs and the
original randomized pruning/SVD/gradient checks passed. This is an algebra
check, not tensor-fitting quality evidence or a GPU speed benchmark.

## Illustrations

Reused Harvard's complete source SVG for the tensor/factor-entry relationship.
No crop, redraw, changed labels or rasterization. The source's depth-axis
K versus caption T mismatch is stated in the adjacent caption.
The already accepted eight illustrations remain unchanged.

Three TensorLy candidates were inspected and rejected: `cp-conv.png` has no
labels; the pyramid is a library overview; the benefit circles are decorative.
They are not inserted to inflate the figure count. A dedicated original
Tucker/TT diagram remains a visual improvement opportunity, not a claim that
all three methods now have individual diagrams.

## Final acceptance

Read the complete final 1,092-line chapter, including the unchanged pruning,
SVD, distillation, Minitron and deployment comparison. The 333-line addition
fits between SVD/LoRA and distillation; it connects discarded directions to
calibration and recovery instead of repeating an introductory definition.
Original quotes were checked separately from authored prose. The no-ai-slop
pass retained technical contrasts, examples and limitations; it did not
shorten the explanation or rewrite source excerpts.

| Check | Observed result |
| --- | --- |
| Chapter programs | 11 executed, including all five new programs |
| Existing numerical checks | 300 Wanda, 400 structured pruning, 400 SVD and 150 gradient trials passed |
| Publishing unit suite | 664 tests in 109 files passed |
| Build | 1,989 pages, exit 0, 27.00 seconds |
| Built HTML suite | 165 tests passed |
| Internal links | 0 broken routes, fragments or files |
| Source coverage | passed; structural check only |
| Prior chapter preservation | exact restoration to `e818fccc317a2025b020f453a2e578e43ea1a2ba39ae893a86c6ee0a89bed5d3` |

The first asset check caught an unsupported archived-SVG provenance path.
The curated copy now has an exact source-parent comparison, hard-pinned SVG
hash and full-canvas check, as well as the source license and QMD hashes.
The final suite above passed after that correction; no old figure assertion
was removed. Locale/Poppler warnings are inherited; exit status and test
results, not a silent log, establish the outcomes above.

Actual browser inspection at 1280×900, 390×844 and 360×800 found no document
overflow or KaTeX errors. All nine images loaded. The new SVG's print-sized
natural dimensions made its labels too small; a scoped CSS rule now scales
the intact vector to the article width (maximum 40rem), preserving aspect
ratio. On mobile, the complete diagram fits; the viewer's 1.5× zoom,
horizontal pan, fit and close controls were exercised. The CP matrices,
Tucker-2 formula and convolution-stage table were inspected in the actual
page, not inferred from compilation. Wide tables retain local scrolling.

Screenshots:

- `/var/tmp/bookvar-tensor-desktop-2026-10-29.png`;
- `/var/tmp/bookvar-tensor-mobile-2026-10-29.png`.

Current chapter SHA:
`9fb33a98e8ece36cfabefe9e589c1ed9a5d7a08f4f686358f4ed2c0769aad4f3`.
The figure-path change after the full reread changed no explanatory text;
the final integrity checks and browser pass use this revision.

No GPU timing, real pretrained-kernel fit, standalone TensorLy install,
whole Harvard unit acceptance, EN-body synchronization, commit, push or
deployment is claimed. The additional dedicated Tucker/TT source diagrams
remain a stated visual improvement opportunity. Whole-book remaining work
is reconciled in the final section of
[the corpus follow-through](2026-10-29-corpus-follow-through.md).
