# Harvard Shared Building Blocks: comparison with existing chapters

This is a thematic source comparison, not a new rewrite of accepted RL,
normalization or FFN explanations. Sources: the already fully read Harvard
Network Architectures chapter, pin `2bd97c5`, Shared Building Blocks
3691–4035. Relevant portions were compared again in context.

## Whole current chapters read

| Chapter | Before | Current SHA-256 |
| --- | ---: | --- |
| Initialization, normalization and residual connections | 873 lines | `019c8f6a0270bcdb4f282ade039108c289f5d7d391dde779e664fd046fa61435` |
| Pre-norm, RMSNorm, SwiGLU and residual stream | 712 lines; 718 after formula wraps | `ba4dde89572fa4c17be5addc0bde7ea24bbffb300120d15467c2282fb120ced6` |
| Dense FFN | 845 lines; 847 after formula wrap | `861b4a9b9724ef3efb0446f09ae522b8117a71758c693dca64c5258c819a5b6a` |

The complete pre-edit texts were read sequentially, not by search hits.
Their hashes matched the accepted 21/22/23 October revisions. No missing
mechanism relative to this Harvard section was found. The subsequent
changes affect only four equation layouts and update dates; the new
`publishing/tests/normalization-mobile.test.ts` reverses those exact
changes and checks the full previously accepted chapter hashes.

## Mechanism-by-mechanism comparison

| Harvard mechanism | Existing detailed explanation |
| --- | --- |
| Products of derivatives and depth | Foundations: scalar growth/decay, matrix directions, initialization assumptions and active/dead nonlinearities |
| Dense maps reused across families | MLP/CNN/RNN comparisons already done; Dense FFN gives shapes, row convention, flattening and shared weights |
| Residual learning and shape matching | Foundations: identity target, post-addition ReLU caveat, projected shortcuts, explicit compatible shapes |
| Residual backward | Both normalization chapters: `I+J_F`, two gradient paths, `F=-x` cancellation, pre/post full Jacobians and zero last projection |
| BN/LN/RMS formula and axes | Foundations: explicit 2×2 matrix, BN train/eval/variance estimates, CNN spatial axes, LN token axes, RMS forward and affine qualifications |
| Normalization derivatives | Foundations: full LN/RMS VJP with worked vector, zero-sum shift direction, small-variance large derivative; LLM chapter has independent RMS finite differences and detached-denominator control |
| Normalization placement and stability | LLM chapter: original Xiong figure and gradient profiles, numerical pre/post case, scope of warm-up evidence, final RMSNorm and Chameleon branch-output reordering |
| Gate modulates a learned signal | LLM chapter: explicit SiLU/product and shape example; Dense FFN: both projections, sigmoid/SiLU difference, matched parameter budgets and dense-vs-sparse routing distinction |
| Gated backward and systems cost | Dense FFN: both branch gradients, scalar reference/finite differences, local TP halves and input-gradient sum, stored-versus-recomputed activations |

The original English/source excerpts, source figures and explanatory prose
are not shortened. No new decorative diagram was made merely to claim
another figure. LSTM/GRU, attention and expert-routing details remain in
their dedicated chapters; this review does not pretend to reread those
chapters in full again.

## Source claims not adopted

- Skip connections do not guarantee a nonzero gradient: `F(x)=-x` cancels
  the direct path. The current chapters already show this calculation.
- Bounds on the eigenvalues of each changing Jacobian do not bound their
  product in the asserted way. CPU counterexample: A and B=A-transpose
  each have spectral radius 0.5; BA has radius 4.48606797749979.
  The tested matrix is `A=[[0.5,2],[0,0.5]]`.
- The claimed universal normalization spectrum `[0.5,2]` is false. For
  constant input, three normalized values and epsilon 0.01, the Jacobian
  has eigenvalues 0, 10, 10. The common-shift direction is zero.
- Fixed affine gamma/beta cannot invert all batches' differing statistics.
  The existing LN explanation already distinguishes information removed
  from a specific input from a fixed learned scale/shift.
- RMSNorm's historical timing tables are not universal latency reductions,
  kernel counts or full-LLM throughput. The LLM chapter pins an actual
  Table 4 workload and separates fusion, dtype, operator and model time.
- No fixed minimum batch size, universal learning-rate increase, claim
  that all networks deeper than ten layers require a norm, or unsupported
  historical “born in” assertion is transferred as a fact.

## Executed checks

All three existing example checkers passed before the notation-only edit:

- `/var/tmp/bookvar-harvard-norm-foundations.log`: three printed blocks,
  initializer/API and exact moment checks, BN state/axes, LN/RMS VJP and
  gradcheck, FP16, residual/pre-post/zero-projection cases; Torch 2.14.1.
- `/var/tmp/bookvar-harvard-norm-gating.log`: four printed blocks, full
  Jacobians, independent RMS finite differences, epsilon/causal controls,
  actual SwiGLU matrices, softmax/z-loss/capping and original Table 4.
- `/var/tmp/bookvar-harvard-gating-ffn.log`: three printed NumPy blocks,
  independent scalar forward, all-coordinate finite differences and TP
  reconstruction. No GPU/collective or recommendation benchmark.
- `/var/tmp/bookvar-harvard-jacobian-controls.log`: the spectral and
  normalization counterexamples above. The tiny negative numerical
  eigenvalue near −2e−15 represents roundoff around zero.

## Rendering repair and final local acceptance

Fresh UI inspection at viewport 360×800 found three overwide equations
in the LLM normalization page and one in Dense FFN. Foundations already
fit. The four equations were split into aligned rows without changing
their mathematical operations. Final verification after that edit:

- All four printed normalization blocks and three printed FFN blocks
  rerun with their independent checks; final hashes above confirmed.
  Logs `/var/tmp/bookvar-shared-blocks-norm-final.log` and
  `/var/tmp/bookvar-shared-blocks-ffn-final.log`.
- 67 unit files / 520 tests, 44.90 s,
  `/var/tmp/bookvar-shared-blocks-unit.log`. This includes reversal of
  the exact four formula changes to the full accepted hashes.
- Final Astro build: 1969 pages, 24.53 s,
  `/var/tmp/bookvar-shared-blocks-build.log`.
- 136 output tests, zero broken internal routes/fragments/files, structural
  source coverage PASS. Logs `/var/tmp/bookvar-shared-blocks-output.log`,
  `bookvar-shared-blocks-links.log`, `bookvar-shared-blocks-coverage.log`.
  Coverage counters are not proof of whole-course semantic completion.
- Fresh browser reload: all 18 and 23 display formulas in the changed
  pages fit the 313px content width on a 360px viewport. Body client and
  scroll widths are both 345px (15px scrollbar); KaTeX errors zero.
  All four changed equations were inspected on screen, not just measured.
  The unchanged foundations page already fit all 26 display equations.
- Desktop 1280×900: normalization page has no display-math or document
  overflow; the original complete Xiong figure and caption were inspected
  in context. Existing figures were retained, not regenerated. Loading
  was checked for all 3/5/8 original chapter figures; this is not a new
  full visual reacceptance of every old figure. Previous figure-level
  provenance/QA reports remain the evidence for unchanged objects.
- `git diff --check` passed for the changed chapters and source matrix.

Screenshots under `docs/audit/screenshots/`:
`2026-10-27-swiglu-mobile.png`, `2026-10-27-residual-jacobian-mobile.png`,
`2026-10-27-norm-gating-desktop.png`.

The comparison and four formula-layout repairs are accepted locally.
No live publication, staging/commit/push/deploy or EN body edit was
performed. H-V1-06 remains integration-partial pending systems primitive
comparisons; H-F1-06 remains re-review pending native figure disposition.
The wider book goal remains active. The next whole-chapter comparison is
GPU/CUDA/memory hierarchy and tensor storage against the already read
computational/memory/data-movement primitives, not another RL rewrite.
