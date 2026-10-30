---
status: content-and-local-ui-reviewed
chapter: 00 Учебник/10 ML Systems/07 Profiling ML-нагрузки.md
route: /textbook/ml-systems/profiling/
before_sha256: e930009037625adff0485b39cac3d91b30340abfe665a16049f439df785deac8
after_sha256: f6fdc8028d115f4026643fc090bc843f2fdd8ddfc97207332681bca22e672a80
---

# One training step: mathematical work and observed events

The preceding goal turn completed the AD addition with execution and rendered
evidence. It was progress, not a wait or a repeated RL pass. This turn continues
the Frameworks comparison in S7; it does not close the full course or textbook.

## Full reading and preservation

Read all 832 prior lines, including the existing profiler script, critical-path
examples, CUDA capture fragments, GeLU comparison, memory history and figures.
Two tool responses were truncated; overlapping reads at 510–832 and 650–730
recovered the missing text. After editing, read the complete 1103-line expanded
chapter in consecutive ranges 1–280, 281–520, 521–810 and 811–1103. The final
cross-entropy layout change was read in its surrounding paragraph and inspected
in the actual mobile page. Final length: 1105 lines.

Removing the one bounded addition and restoring the two dates recovers the
entire accepted 832-line SHA. Independent TypeScript and actual-example Python
checks verify that preservation. Original prose, math, seven source-unit IDs,
code, anchors, navigation and all four older source figures remain unchanged.
The chapter's draft status remains; no EN body or original course was rewritten.

## Source comparison

Harvard ML Systems, pinned `2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd`,
Frameworks Anatomy of a Training Step, lines 4266–4599. Read the complete
training fragment, forward/backward explanations, both calculation cells,
roofline table, launch model and backend discussion. This is part of the
already-read 4762-line source, not a restart of it.

The whole `lst-training-step-anatomy` fence is retained verbatim, including
comments and its `(32)` size error. An adjacent warning distinguishes it from
a standalone program. The complete editorial variant supplies initialization,
the tuple size, zeroing and a defined plain-SGD update. Original language,
adjacent attribution and CC BY-NC-SA 4.0 are retained.

Fresh primary-source checks: PyTorch 2.14 randint size contract; profiler
record_shapes/activities and its reference-holding overhead; CUDA asynchronous
execution/stream ordering; CrossEntropyLoss class-index and reduction rules.
These are linked beside the relevant mechanisms, not merely listed at the end.
No claim to have reread all profiler/CUDA documentation is made.

Stanford CS231n: read pinned Neural Networks Part 1, Layer-wise organization
and Example feed-forward computation, lines 125–164, with both original
network captions and the source's column-input convention. Its complete
3–4–2 network image supplies the architecture illustration; our caption
explicitly distinguishes it from the row-batched 784–256–10 worked model.

## Worked explanation

- Constructed a batch of 32 examples, weights 784×256 and 256×10, two biases
  and integer class labels. Forward preactivation/hidden/scores are tied to
  their explicit shapes, rather than listed as names of API calls.
- Derived mean cross-entropy score gradients, row sums for the biases,
  both weight gradients and the ReLU mask. Compared all four parameter
  gradients against autograd, not just the final loss.
- Demonstrated the difference between saved input values and requested input
  derivatives: fixed X still participates in the W1 derivative but does not
  require a G_X matrix product.
- Verified every SGD parameter update against theta−0.01g. Actual fixed-case
  output is `2.644112 -> 2.476242`; no universal loss-decrease guarantee is
  inferred from it.
- Recorded actual CPU operator events with stage ancestry and operand shapes.
  Fixed inputs give two forward and three backward mm events; trainable
  inputs add the 32×256 by 256×784 product. Both runs begin from identical
  cloned parameter values, with setup outside the captured window.
- Derived the matrix-only FLOP totals from those recorded shapes:
  forward 13,008,896; fixed-input backward 13,172,736; an input derivative
  adds 12,845,056. The resulting 1.013/2 ratios are not universal backward
  multipliers and exclude non-matmul work.

The source's six launches, fixed launch latency, three-operation cross-entropy
estimate and physical memory-traffic multipliers are not accepted as measured
properties of this program. An ATen event is not a CUDA launch. The CPU run
does not establish cuBLAS selection, Tensor Core usage, CUDA timing, HBM
traffic, optimized-compiler speedup or cross-backend numerical equivalence.

## Source diagram and visual evidence

Complete original CS231n JPEG, 511×350, 41,510 bytes, copied byte-for-byte
from commit `8ec57ccfc32379ae4b37c45de949af359673f92f`, SHA-256
`6d7bfc74b8315a0d7ab35d4cb8aeaae38bc439cc56be1c561b1b9a98b208fac9`.
All inputs, hidden/output nodes, arrows and English labels are retained.
No crop, artificial resolution increase, redraw, relabeling or recompression.
The page renders it at native 511×350 on desktop and proportionally on mobile.

The source Markdown matches the same pinned upstream bytes, SHA-256
`28196e335053230ac9f1fff728c8baa089e05814982d950ffc6339462828f0f3`.
MIT copyright/license evidence is retained in the earlier CS231n source archive
and public license copy. The local manifest, NOTICE and central registry agree;
unit checks bind the source image, page, license and output hashes.

Actual UI: 1280×900 and 360×800. The new diagram and its complete caption were
inspected inline and in the viewer. Both matrix products and all five result
shapes are readable in the narrow table, 328/328. All five HTML tables fit;
body 360/360. Both display formulas and all inline math fit, with zero KaTeX
errors and no duplicate IDs. The first cross-entropy formula overflowed at
368/328 and was reflowed into two aligned lines; the final page was reloaded
and inspected after that change.

All five source images load. The four unchanged original profiler/Nsight/
memory screenshots were reopened in the actual desktop viewer; their tracks,
panels and captions still match the explanation. Their fine text requires
enlargement at narrow inline widths; no new mobile pixel-detail claim is made
for those unchanged screenshots. The new network diagram's three labels
are readable on mobile without enlargement.

EN route: visibly shows the explicit unavailable-language banner and the same
new content as fallback. It is not an English translation. Screenshots are
saved under `docs/audit/screenshots/2026-10-27-training-step-*`.

## Execution and publication checks

- Final actual-example checker, exit 0, at the final SHA above:
  `/var/tmp/bookvar-training-step-examples-accepted.log`. All eleven Python
  fences parse. Five new runnable fences execute in their printed order;
  the intentionally incomplete original fragment is checked byte-for-byte,
  not run on unavailable CUDA. Its isolated integer-size call rejects on CPU.
- Both profiler runs have the exact expected mm shapes and stage labels.
  Eight additional independently dimensioned gradient cases pass, including
  requested/omitted input gradients. All previous scalar/table/CS336-source
  launch-count checks remain and pass.
- Old standalone profiler still exports one temporary step-5.json with
  767 events and three ranges per stage. GeLU output/gradient cases pass.
  Memory-history finally behavior is tested by mock; CUDA/NVTX/Memory Snapshot
  execution is not claimed. Temporary exported traces are inspected then
  removed by TemporaryDirectory, not left as permanent source assets.
- Full unit suite before the last diagram/wording/layout additions: 79 files,
  562 tests pass. After the diagram and strict source branch, both affected
  suites pass all 153 tests. After the final formula layout change the four
  added training-step tests pass again. These are not represented as a full
  final-revision 563-test run.
- A first asset run fell through to the older lecture-cache path rule. Added
  a strict case for this pinned source copy, without relaxing the older rule.
  A first output assertion counted the viewer's empty img as a source figure;
  the corrected assertion requires exactly five images with src attributes.
- Final Astro build: 1969 pages, 22.50 s, finished 21:05:22 local;
  `/var/tmp/bookvar-training-step-build-accepted.log`, exit 0.
- Final output suite: 148 tests pass in 4.32 s;
  `/var/tmp/bookvar-training-step-output-accepted.log`, exit 0.
- Final built links: 0 broken routes/fragments/files;
  `/var/tmp/bookvar-training-step-links-accepted.log`, exit 0.
- Structural source coverage: 210 rows / 58 destinations pass. This is
  bookkeeping evidence, not proof that all course knowledge is integrated.
- Scoped tracked-file whitespace check passes. No file was staged or published.

## Remaining scope

This completes the selected training-step addition. H-V1-07 remains partial:
the full source-section map must reconcile platform analysis, deployment,
framework-selection and all remaining mechanisms with current destinations.
Three held native-figure candidates and broader source/rights findings remain
open. Do not reread the now-accepted S7/S8/AD/MLP work from the beginning or
restart RL. The complete Harvard/HSE/EDLS/textbook and deferred EN objective
remains active; no commit, push, deployment, delegation or live-infra mutation
was performed.
