# Indexed gather/scatter: explanation, original figure and CPU checks

Worktree: `editorial-route-alignment`. Local work only, 2026-10-27.
Chapter: `00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению.md`.
Route: `/textbook/ml-systems/gpu-kernels-triton/#indexed-gather-scatter-and-gradients`.

Before: 1890 lines, SHA-256
`b51b579453cf677bc17eb37dbbd29dd6386a75727005ca30505a8c4c156e4faf`.
After: 2198 lines, SHA-256
`61b8e6b36d4484c9d3ba2d920942701af321638e77556b7215fde7dbaa39922b`.
The current chapter was read in full before the addition. The finished
addition and adjacent GEMM/numerics transition were reread after editing.
Removing only its bounded section and four source metadata entries restores
the complete previous SHA. Existing reasoning, figures, examples, manual
anchors and all 14 source-unit IDs are preserved. Earlier reduction,
torch.compile and CUDA Graphs preservation tests remain active.

## Explanation review

The selected original English explanations are retained, not translated.
`no-ai-slop` and its eval were read completely; the editorial passages were
checked for generic introductions, unsupported importance, repeated templates
and compression into definitions. Existing source excerpts were not edited for
style. This checklist is not evidence of mechanism correctness; the source
comparisons and worked calculations below supply that separately.

| Mechanism | Complete selected source and comparison |
| --- | --- |
| Elementwise gather | PyTorch v2.14.1 `_torch_docs.py`, full `gather` entry 4812–4849, including shape contract, empty input and example. Retained original coordinate/shape explanation with markup conversion only. Traced both matrix axes, then checked differing input/index shapes. |
| Indexed reduction | Matthias Fey `scatter.py`, all 177 lines including complete implementation, docstring, formula, atomic note, broadcasting and example; full original `add.tex`. Kept original explanation and SVG. Traced destination 0 through 5 → 6 → 8 and all four final groups; showed nonzero initial output and untouched fifth destination. |
| Assignment vs addition | Complete PyTorch `scatter_` and `scatter_add_` entries, `_tensor_docs.py` 4303–4488. Preserved the original repeated-destination assignment warning. Distinguished that contract from sum accumulation and FP32 reduction-order variation; showed the lost-update interleaving. No particular overwrite winner is promised. |
| Backward and non-invertibility | Explicit chain rule for incoming `[1,2,3,4]`, output gradient `[2,0,4,4]`; reverse scatter-add `[10,0,60,40]` counterexample; transpose/adjoint identity and occurrence counts. Also derived gradients with respect to scatter updates and initial output. Both rules checked by actual autograd and gradcheck. Integer indices remain fixed. |
| ONNX portability | Complete GatherElements-13, Gather-13, ScatterElements-18 documentation, relevant schemas, and both complete NumPy reference helper files at `3646b038`. Distinguished element output `[2,2]` from slice output `[2,2,2]`, negative index acceptance, and the overly general “inverse” wording. No blanket equivalence with PyTorch is asserted. |
| Memory access | NVIDIA CUDA C++ Best Practices Guide 13.4, §10.2.1.1–10.2.1.4 read. Index-dependent source pointer, contiguous destination, four vs 32 access units, repeated index/cache reuse and contention explained. Counts exclude index/output traffic and are not timings or HBM measurements. |
| Layer boundary | Local element destinations distinguished from process-level `torch.distributed.gather/scatter`. Linked existing DLRM explanation for pooled table gradients and existing DDP chapter for collectives instead of copying those chapters. |

Source revisions, licences, exact selected ranges and the rejected ONNX-figure
lead are retained in
`05 Источники/Originals/indexed-movement-2026-10-27/README.md`.
Unselected ONNX/PyTorch source entries are not represented as read.

## Original figure and rights

Matthias Fey, PyTorch Scatter, source revision
`f514c10f920b5aeed2eb162092f0ad20d3edee52`,
`docs/source/_figures/add.svg`.

Original/copy SHA-256:
`6f3791b14d81bd96aab6f239f0eb985536996eb5d4941d687638e66bfac3a46f`.
Complete native SVG retained byte-for-byte. No crop, redraw, translated labels
or changed paths. Original TeX data independently reproduces `[8,7,5,4]`.
MIT permission text was read in full and retained beside raw source and at
`site/public/licenses/pytorch-scatter-f514c10-MIT.txt`; author and direct
original links are adjacent to the illustration. Local `assets.json`, NOTICE
and central registry agree; the existing strict copied-source validator covers
this directory rather than a new weakened validation branch.

The original print dimensions are small. A narrowly scoped CSS selector
sets `width: min(100%, 28rem)` and `height: auto`; the original viewBox is
unchanged. This affects only this diagram, not all SVGs or other course pages.

## Executed checks

`check_indexed_movement_examples.py` runs all four actual printed Python
fences, each in its own scope, with PyTorch 2.14.1 and NumPy. Additional checks:

- 18 rank/axis/output-shape combinations in ranks 1–3, compared with a scalar
  coordinate loop, gather autograd, the adjoint identity and actual selected
  ONNX NumPy helper functions;
- Gather/GatherElements output values and shapes; ONNX negative indices vs
  core PyTorch rejection; invalid rank/non-axis extent and empty input;
- independent gradcheck for gather and for both inputs of scatter-add;
- missing-coordinate, multiplicity and invertible-permutation controls;
- lost-update and FP32 addition-order counterexamples; aligned, strided,
  permuted, repeated and misaligned access-unit counts;
- original TeX numerical arrays, retained original SVG hash and full licence.

This executes the two reviewed ONNX helper functions extracted from their AST;
it does not run ONNX Runtime or claim exporter end-to-end compatibility.
The old Triton checker initially misread inline Markdown arrays such as
`[[1,2],[3,4]]` as wiki links. It now excludes inline code as well as fenced
code before checking actual wiki links. The actual publication did not have
that failure. Its four old CPU fences and all independent checks passed after
this parser correction. Three actual Dynamo fences and the reduction-tree
checker also passed on the final chapter SHA.

Logs:

- `/var/tmp/bookvar-indexed-examples.log`: all new CPU checks PASS;
- `/var/tmp/bookvar-indexed-existing-triton.log`: old CPU examples/checks PASS;
- `/var/tmp/bookvar-indexed-existing-dynamo.log`: captured FX/guards/cache/
  graph-break/dynamic-shape checks PASS, recording backend only;
- `/var/tmp/bookvar-indexed-existing-reduction.log`: 1025 array traces,
  work/depth and all four original GIF frames PASS;
- `/var/tmp/bookvar-indexed-unit.log`: 70 files, 531 tests PASS;
- `/var/tmp/bookvar-indexed-build.log`: 1969 pages, 26.41 s, terminal exit 0;
- `/var/tmp/bookvar-indexed-output.log`: 139 built-output tests PASS;
- `/var/tmp/bookvar-indexed-links.log`: 0 broken internal routes/fragments/files;
- `/var/tmp/bookvar-indexed-coverage.log`: structural coverage PASS, 210 rows,
  58 existing destinations. This is not a claim of complete source integration.

No CUDA/Triton/Inductor execution, distributed launch or performance benchmark
was performed. CPU correctness is not GPU implementation or performance proof.
Dependencies were reused from the existing offline uv cache; no weights or
datasets were downloaded.

## Actual rendered page

The completed local build was reloaded before inspection. At 1280×900,
body width/scroll width were 1265/1265 and the complete diagram was 448×238.
At 360×800, body was 345/345 and the diagram 313×166. All 20 display formulas
and inline KaTeX expressions fit; there were no KaTeX errors. Numerals, arrows,
index/input/output labels and the adjacent explanation were read in place.
The image viewer was inspected at 2×, including both left and right edges;
native vector lettering remained clear. No label was lost to a crop.

Saved actual screenshots:

- `screenshots/2026-10-27-indexed-movement-desktop.png`;
- `screenshots/2026-10-27-indexed-movement-mobile.png`.

The new backward section was separately inspected at 360 px, including its
inline numerical example and displayed derivative. The EN route contains the
same addition/image/notice behind the explicit “This content is not available
in your language yet” fallback banner. It is not an accepted English
translation; no EN body was edited.

## Remaining work

This closes the selected local indexed-movement explanatory gap, not H-V1-06
as a whole. Next compare all 1411 lines of current
`44a Processes, collectives и DDP.md` with the already-read Harvard
Computational Primitives material; then dispose of each native figure
individually. The restored Harvard cross-attention drawing remains rejected
for textbook reuse because of its wiring, even though it now renders.
H-V1-06 stays integration-partial and H-F1-06 stays re-review. Other
Harvard/HSE/EDLS/course/rights work and the full textbook goal remain open.
Ready PPO/DPO/RLVR/GRPO were not returned to the edit queue.
No staging, commit, push, deployment or live infrastructure mutation.
