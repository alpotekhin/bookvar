# Local reduction tree in S8

Canonical artifact:
`00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению.md`.
Final: 1890 lines, SHA
`b51b579453cf677bc17eb37dbbd29dd6386a75727005ca30505a8c4c156e4faf`.
Route: `/textbook/ml-systems/gpu-kernels-triton/#local-reduction-tree-work-depth-synchronization`.

## Explanation review

The entire prior 1699-line chapter was read before editing and the expanded
chapter was reread, including the unchanged FLCE, Dynamo and CUDA Graphs
sections. Subsequent local wording and the mobile probability-vector wrap
were read separately. The accepted prior SHA is reconstructed by removing
only the bounded addition, two new source URLs, metadata dates and that one
math reflow. The old explanation, code, citations and figures remain intact.

The previous explanation ended the row accumulator with `tl.sum`, without
working through the local tree. The addition now connects that operation
to a complete three-round case, then distinguishes logical work, dependency
depth, resource limits and actual implementation. The original four-frame
Flogr animation supplies the visible states: 2/3/5/1/7/6/8/4, then
5/6/13/12, then 11/25, then 36. The worked table and CPU trace retain the
states when the GIF moves to another frame.

The arithmetic explains why seven additions can have depth three, and
why two workers need four ideal rounds. Odd inputs are carried forward;
padding five inputs to eight instead creates seven addition nodes rather
than four. A stale-read example gives 10 instead of 11 and explains what
must be ordered between shared-memory stages. The existing row example's
four accumulated values (13,13,6,7) then reduce to 39.

The exact current CUDA Samples helper is retained as code, with its full
public copyright/license notice. Its FP32 full-warp contract is explicit:
five shuffle stages, 160 lane-level additions, a valid final sum in lane 0,
and no claim that every lane receives the same result. The minimal binary
tree has 31 additions. Source stages and exact machine instruction counts
are distinguished; no compiler disassembly or GPU timing is inferred.
Sparse masks, shared-memory ordering and the inter-block dependency are
not silently treated as solved by a full-mask register shuffle.

The transition back to the existing online softmax explanation remains:
a scalar sum is simpler than the maximum/normalization state. No online
softmax derivation or FLCE/Dynamo/CUDA Graphs section was replaced by a summary.

## Sources and factual checks

Detailed source pins and reading scope are in
`05 Источники/Originals/reduction-primitives-2026-10-27/README.md`.

- Harris: all 38 slides' extracted text read; slides 3 and 29 rendered and
  inspected under the PDF workflow. Only mechanisms are used; historical
  G80 timings, the old 512-thread limit and implicit `volatile` warp code
  are not adopted as current guarantees. No slide-image reuse is claimed.
- NVIDIA CUDA Samples `5443602`: generic helper and selected complete
  shared-memory reductions read; copied helper is byte-equal to its source.
  Full BSD-3-Clause notice is publicly present and byte-equal to the source.
- Current CUDA guide: complete selected warp-shuffle description and mask
  constraints read, including inactive source lanes, out-of-range source
  handling and the absence of a memory-ordering guarantee from shuffles.
- Flogr: full original description/license sections and all four frames
  inspected. The published GIF is byte-equal to the original; its dimensions,
  four frames and 1500 ms frame durations are checked. Both asset manifests
  retain author, permanent description URL, CC BY-SA 4.0 and source hashes.

`check_reduction_tree_examples.py` executes the actual new Markdown fence.
It checks all partial sums, output, addition count and depth for 1025
signed-integer arrays; empty-input rejection, odd carry, zero-padding cost,
ideal worker schedules and stale-read counterexample are included. Its
full-warp CPU exchange model checks lane 0 = 528, last lane = 1024 and
160 lane-level additions versus 31 in a minimal tree. This is not a CUDA
interpreter or a device execution. Log:
`/var/tmp/bookvar-reduction-tree-examples-final.log` records the earlier
source-level revision; the final b51b579 revision was also executed directly.

The unchanged S8 CPU checkers also passed on the expanded source-level
revision: all printed GEMM matrices, padding/online softmax, 48 CE gradient
cases, buffer/queue models and the three actual Dynamo fences. The last
change after those runs was only probability-vector display wrapping;
its numbers are unchanged and the baseline contract verifies that scope.
No Triton compilation, CUDA execution, optimized Inductor backend or
performance experiment is claimed.

## Style and visual inspection

The final text was checked separately against no-ai-slop. The addition uses
specific values and dependencies rather than a generic introduction or an
importance claim. Necessary distinctions and API constraints were retained.
Original code, source excerpts and the existing Russian prose were not
rewritten for stylistic uniformity.

Desktop 1280×900: animation and original caption inspected in the chapter.
The file's original 901×512 canvas is complete. At 360×800, the inline
animation is 313×178; labels can be enlarged in the viewer. Both edges were
inspected at 2×, and the animation visibly advanced to its final sum 36.
The table's intermediate states remain available without timing the GIF.

A separate pre-existing inline vector in the online-softmax example extended
to x=349.5 with a 345px body. It was wrapped into two aligned rows, preserving
all four probabilities. Final body client/scroll widths are 345/345;
all 17 display formulas fit within one pixel of browser rounding and there
are zero KaTeX errors. Tables scroll inside their own containers.

Screenshots:

- `screenshots/2026-10-27-reduction-tree-desktop.png`;
- `screenshots/2026-10-27-reduction-tree-mobile.png`;
- `screenshots/2026-10-27-reduction-probabilities-mobile.png`.

## Release evidence and remaining work

Final current-state checks: 69 unit files / 526 tests; 138 built-output
tests; 1969-page build in 22.86 s; zero broken internal routes/fragments/files;
structural source coverage PASS. Logs:
`/var/tmp/bookvar-reduction-release-{unit,output,build,links,coverage}.log`.
The first full run caught the registry's unrecognized new source directory;
it was added to the existing strict original-GIF validation path rather
than weakening the checks.

Both root and EN outputs contain the new section and original GIF. EN is
explicitly marked “This content is not available in your language yet”:
this is a source-language fallback, not a completed English translation.
No EN body was edited.

The local reduction explanation is accepted within these bounds. Complete
course integration remains open: indexed gather/scatter, the full distributed
destination comparison and individual disposition of Harvard native figures
are next. H-V1-06 remains integration-partial; H-F1-06 remains re-review.
No staging, commit, push, deployment or live-infrastructure operation occurred.
