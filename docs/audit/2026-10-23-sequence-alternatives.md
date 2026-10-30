# Sequence-model alternatives: explanation, examples and figures

Date: 2026-10-23. Local Russian chapter only. No authored translation, commit,
push or deployment. The general textbook, course integration and model atlas
remain under review.

Chapter: `00 Учебник/09 Dense FFN и Mixture of Experts/03 Mamba, RWKV, RetNet и гибридные архитектуры.md`.
Final SHA-256: `4f78170b76794b013076afc363b9df813563ac0df59e0465bd0d1c74633bb803`.
The complete 1,475-line input was read and saved in
`snapshots/2026-10-23-sequence-alternatives-before.md`, SHA-256
`fdadb7d3bba105cde4f8cc2cba739f842e8c3293ae7cef86af208c3c4c4d489e`.
The final 1,743-line revision was read sequentially: 1–360, 361–720,
721–1080, 1081–1280, 1281–1490 and 1491–EOF. An earlier truncated attempt
at 1081–1440 was not counted as a completed read.

## Explanation

The retained chapter already covered RNN/LSTM, state-space models, Mamba,
RWKV, RetNet, linear attention, delta rules, TTT, hybrids and stateful serving.
The revision preserves those explanations and adds five executable NumPy
blocks connected to worked examples:

- A two-state LTI recurrence and its causal convolution, including the direct
  term, ordered affine composition and nonzero initial-state boundary.
- The selective Mamba-1 operator with explicit tensor shapes and supplied
  post-projection tensors. Exact zero-order-hold discretization is distinguished
  from the reference implementation's input discretization.
- SSD as recurrence, dense matrix and blocked computation. A two-token block
  separates local output from the incoming state's contribution. The decay
  matrix excludes decay of the current write.
- Normalized linear attention versus gated delta updates. Repeated keys show
  interference, correction and the error caused by computing the residual
  before applying decay. Unit/orthogonal-key assumptions are explicit.
- Online TTT versus block-start gradients: 17.5 and 20 are different updates,
  not two execution schedules for the same update. The toy linear loss is
  separated from the paper's residual LayerNorm model, learned initial weights,
  input-dependent step and outer gradient through the inner update.

RWKV-7 now specifies the orientation of its state, separate write key and
matrix transition, with a numerical example. Mamba-3 explains endpoint
weighting, rotations, MIMO rank and the actual state shape. Stateful serving
includes a numerical accepted-prefix rollback rather than implying that an
overwritten recurrent state can be truncated like a KV sequence.

The style pass removed an abstract opening, repeated misconception boxes and
an imperative learning-outcome recap. Mechanisms, caveats, citations and the
four practice exercises remain. Nine wide equations were split into lines.
The body and metadata previously skipped ML Systems while the generated
footer did not; all now lead to S1, “Модель как часть системы”. The TTT caption
and link were corrected to v1 Figure 4, verified anchor `S1.F4`.

## Source comparison

Freshly read pinned CS336 Lecture 4 pages 4–11; Mamba-1 v2 §§2, 3.1–3.4
and the relevant theorem in §3.5; Mamba-2 v1 §§5–6.3 and §7.1;
Linear Transformers v3 §§3.1–3.3; Gated DeltaNet v1 §§2.1–2.3 and §3.1;
TTT v1 §§2.1–2.4 and §2.7; the relevant Mamba-3 v1 §3.1/3.2 and decoding/
MIMO portions; RWKV-4 v2 §§3.1.2–3.1.3; RWKV-7 v1 §4.1; and RetNet v4's
parallel-form explanation in §2.1. This does not claim a complete fresh read
of every cited paper, Mamba-2 §7.2 or all of Mamba-3 §3.3.

The Mamba reference is pinned to
`e9594ce1c732d97440f0332fdc43170a2294dbfa`. Qwen3-Next's official model card
confirms 12 × (3 GDN + 1 attention). Nemotron 3 Nano's card confirms
23 Mamba-2 + 23 MoE + 6 GQA layers; the sequence-mixer ratio is 23:6, not
the course slide's shorthand 3:1. Older MiniMax, Falcon, Jamba, phonebook
and TTT-E2E material was retained, not freshly reviewed in its entirety.

## Figures and limits of this visual pass

There are 32 figures, not 32 new figures. Two new complete Mamba-2 originals
provide Figure 5's SSD algorithm and Figure 6's block, including their full
English captions at 300 dpi. The latter replaces an old right-cropped copy
that was also mislabeled Figure 5 in the registry. Four duplicate formula
screenshots were removed from the chapter; their equations remain live math.
All old files and archives were preserved, and usage records were adjusted.

`prepare_sequence_figures.py --check` verifies the new source PDF hash, full
panel geometry, pixels and manifest. The archived HTML verifies CC BY 4.0.
Both new panels were inspected at native resolution. The other 30 objects
were checked on the rendered page and in a raster contact sheet where
applicable, but were not all freshly compared pixel-for-pixel with their
upstream originals. That source-comparison gap remains open.

Two retained objects still warrant improvement: the comparison table is
766 × 388, exactly the native embedded raster on CS336 page 11 (rendering
the PDF larger cannot restore missing detail); the old RetNet image includes
some surrounding article text. Existing permission records have not been
upgraded to verified named licenses merely because the page renders.

## Numerical evidence

`check_sequence_examples.py` executes the five actual published code blocks.
Independent references check the LTI values; 64 stream partitions with a
nonzero initial state; 132 affine parenthesizations; all eight SSD chunk sizes,
64 partitions and zero decay; direct normalized-attention prefix sums;
gated-delta transitions and nonorthogonal-key interference; block-start TTT
gradients for batch sizes 1/2/3/8; and future-token perturbations.
Additional checks cover RWKV/RetNet arithmetic, rotation, endpoint weighting,
memory accounting and accepted-prefix rollback. Wrong masks, update order
and invalid normalizers are negative controls. These are CPU algebra checks,
not a trained-checkpoint run, CUDA kernel, real serving test or speed benchmark.

## Build and actual browser

On the final chapter revision, the full build produced 1,961 pages; all 378
publisher unit tests and all 76 output tests passed. The built-link checker
found zero broken internal routes, fragments or files; `git diff --check`
passed. The final sequential run finished at about 02:36 Moscow time.

The actual 375px page loads all 32 images and 328 KaTeX expressions without
math errors or document-width overflow; no display formula is wider than its
container. Code and wide tables scroll internally. SSD was inspected in the
desktop viewer and at 2.5× on mobile; arrow-key panning changed scrollLeft
282 → 322 → 282. Actual navigation reached MoE and S1 and returned to this
chapter. The temporary viewport override was reset. A wrapped inline link
did not activate under one automated center click; keyboard activation worked,
as did footer navigation. This is not reported as a confirmed routing defect.

Screenshots: `/tmp/bookvar-sequence-ssd-desktop-final-2026-10-23.png` and
`/tmp/bookvar-sequence-ssd-mobile-2026-10-23.png`.
An additional 1280 × 900 view includes the complete diagram and original
English caption: `/tmp/bookvar-sequence-ssd-complete-2026-10-23.png`.

## Build storage and test timing

An earlier build failed with ENOSPC and left incomplete generated output;
the successful full build above restored it. Individually verified own
temporary duplicates and an ignored generated Astro cache were removed.
Canonical figures, personal files and source archives were preserved.
`BOOKVAR_LINK_ASSETS=1` now opts into local hard-linked staging, avoiding
duplicate public assets. The staged links must not be edited. Tests and an
actual output inode/hash check confirm the final dist file is an independent
regular copy. Default/CI behavior is unchanged. An attempted APFS clone was
unsupported and was not retained as an implementation.

The old registry test exceeded its unchanged 30-second timeout under
structural Buffer comparison. Equivalent native `Buffer.equals` checks
preserve exact byte comparisons and pass in the final suite; no assertion
or asset was skipped and the timeout was not increased.

## Remaining work

S1 “Модель как часть системы” subsequently received its own review in
`2026-10-23-system-foundations.md`; the next chapter is GPU/CUDA. The
visual/source gaps listed above, general review, course ledgers, model atlas,
English parity and old CS231n source-comparison item remain open. No remote
release occurred.
