---
status: content-and-local-ui-reviewed
chapter: 00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению.md
route: /textbook/ml-systems/gpu-kernels-triton/
before_sha256: 61b8e6b36d4484c9d3ba2d920942701af321638e77556b7215fde7dbaa39922b
after_sha256: 4fab5d0025274bd232fb3122c1152dc939cf8529a41389401c3d261e1db94a23
---

# Tracing, branches and export: explanation review

The previous goal turn was content progress: module-state teaching was added
to MLP and verified. This turn continues H-V1-07 without returning to accepted
RL or starting the already-read Harvard source again. It does not close the
whole Frameworks chapter or course.

## Full destination reading and source selection

Read all 2198 lines of current S8 in seven consecutive ranges. Existing explanations
already cover fusion, masked vector addition, row softmax, the local reduction
tree, online normalizer, tiled GEMM, indexed movement/gradients, FLCE,
Dynamo capture/guards/graph breaks/dynamic shapes, CUDA Graph buffers/pools,
CPU–GPU submission overlap and the setup amortization inequality.

The old text and calculations are retained, not replaced by another overview.
In particular, the CUDA Graphs example already demonstrates why launch costs
cannot all be added to device work without accounting for overlap. Its strict
break-even calculation supplies the same setup-cost reasoning needed for
compiler amortization. The Harvard benchmark table does not have compatible
measurement provenance, so its fixed throughput/speedup values are not imported
as measurements. A full per-section Frameworks map is still outstanding.

The missing source mechanism was the sample-path failure of legacy tracing.
Selected Harvard source: Hybrid approaches, tracing/scripting and IR discussion,
lines 1170–1323; modern graph-capture discussion immediately after that.
The complete `lst-tracing-silent-failure` fence is copied with only its needed
`import torch` added. Function, call and comments remain exact source text.

Read PyTorch v2.14.1's `trace` warning and `check_trace`/`check_inputs` contract;
the pinned TorchScript deprecation note; export programming-model material
through line 335, including the complete Dynamic Data-Dependent Control Flow
example; and the Control Flow Operators introduction/prototype warning and
specialization discussion through line 51. The whole 529-line export document
and the whole trace implementation are not claimed read.

The original `M_old` and `M_new` classes are retained verbatim. No translation
or replacement summary of their code is presented. Attribution and actual
licenses stand beside the excerpts; all originals are archived under
`05 Источники/Originals/frameworks-2026-10-27/`.

## Worked mechanisms

- On `[1]`, Python and the legacy trace both return `[2]`; on `[-1]`, Python
  returns `[-3]` but the trace returns `[-2]`. IR contains multiplication,
  without the sum/comparison/conditional. `[0]` hides the wrong branch;
  `[1,-1]` exposes the sum-zero branch without zero outputs.
- Default trace validation may repeat the same example. A negative
  `check_inputs` case raises `TracingCheckError`; disabling the check would
  not restore missing control flow. The real `TracerWarning` is observed.
- `torch.compile(..., backend='eager')` produces the correct four results via
  supported graph-break/fallback behavior. This is not an Inductor or GPU run.
- The first strict-capture test unexpectedly reused the earlier compiled
  variant. Starting from `torch.compiler.reset()` exercised fresh
  `fullgraph=True` capture and raised `Unsupported`. This reset and its timing
  boundary are now printed in the chapter, not hidden in the checker.
- Original PyTorch sine/cosine branches survive explicit export through
  `higher_order.cond`; both branch subgraphs exist. Positive, negative and zero
  inputs give 0.841471/0.540302/1 with fixed shape `[1]`.
- Non-strict export of the ordinary `M_old` Python branch still raises
  `GuardOnDataDependentSymNode`. Passing length two to the shape-one exported
  program raises `AssertionError: Guard failed: x.size()[0] == 1` in this
  version. These rejected cases do not count as support for arbitrary control
  flow or arbitrary shapes.

`check_tracing_control_examples.py` executes all seven actual printed fences,
then 25 additional scalar cases and negative controls. Its first harness had
no `__name__`, so TorchScript could not name the function; the harness now
provides the normal module identity. A wrong expected exception type for the
shape rejection was corrected to the actual pinned API behavior. Assertions
about rejection were retained. The final example run exited 0.

## Preservation, style and figures

Removing only the bounded `tracing control flow 2026-10-27` addition restores
the exact entire 2198-line before SHA. All old prose, math, sources, code,
anchors and source-figure paths are unchanged. Existing baseline helpers now
remove this addition before reconstructing older accepted revisions; their
expected hashes and source-equality assertions are not weakened.

The new section was read in full after the edits, with the whole original
chapter still in context. No-ai-slop review removed generic explanation-of-
importance sentences, while retaining failure outputs, warnings, cache scope,
prototype limitations and source code. The new outcome table complements the
actual recorded IR and existing original Dynamo pipeline image. No generated
diagram or source-image substitute was added.

No claim of accelerator correctness, code-generation speedup, model training
or full export deployment is made. Native Frameworks figures, additional
forward/custom AD and the remaining runtime/training-step reconciliation are
still open. EN bodies and RL are unchanged; no git publication is authorized.

## Current-revision execution evidence

- Seven new actual printed fences and 25 additional scalar cases pass at final
  SHA `4fab5d0025274bd232fb3122c1152dc939cf8529a41389401c3d261e1db94a23`.
  `/var/tmp/bookvar-tracing-examples-final.log`, exit 0. Expected rejected
  export diagnostics remain visible in the log; they are negative controls,
  not a successfully exported implicit Python branch.
- All three old Dynamo examples pass, including the literal printed FX body,
  5 calls/3 backend invocations, two cached regions and dynamic-shape/scalar
  reuse. `/var/tmp/bookvar-tracing-old-dynamo-final.log`, exit 0.
- The prior Triton chapter checker passes on the same final SHA: all 23 fences
  parse, four CPU-model fences execute, old indexing/masking/softmax/chunked
  CE/gradient/work examples pass. `/var/tmp/bookvar-tracing-old-triton-final.log`,
  exit 0. This still does not execute Triton or CUDA.
- Full unit suite before removing the redundant explicit heading anchor:
  76 files, 551 tests, 43.35 s. `/var/tmp/bookvar-tracing-unit-final.log`.
  The six affected suites were rerun after the final anchor/map revision:
  22 tests passed. `/var/tmp/bookvar-tracing-scoped-final.log`.
- Final Astro build: 1969 pages, 21.01 s, finished 05:01:49 local.
  `/var/tmp/bookvar-tracing-build-final.log`, exit 0.
- Final output suite: 144 tests passed in 3.99 s.
  `/var/tmp/bookvar-tracing-output-final.log`, exit 0. The first output run
  exposed an anchor assertion mistakenly run on tag-stripped text and an old
  16-fence count. The anchor is now checked as an HTML attribute and all S8
  total-count assertions expect the actual 23; old checks are retained.
- Built links: zero broken routes/fragments/files.
  `/var/tmp/bookvar-tracing-links-final.log`, exit 0.
- Structural coverage: 210 rows, 58 destinations PASS.
  `/var/tmp/bookvar-tracing-coverage-final.log`, exit 0. This is not proof of
  full course integration. Section-map integrity: 37 ranges / 17 current
  destinations / 5256 source lines, zero errors. Scoped diff whitespace check
  passed.

All native process handles were observed terminal. No state-file assumption
is used as a substitute for completion.

## Actual browser evidence

Inspected the freshly built localhost route at 1280×900 and 360×800. Desktop
body width/scroll width 1265/1265; mobile 345/345. The new result table is
313/313 on mobile; all four inputs and all three output columns are visible.
All 20 display equations and inline math fit, with zero KaTeX errors and no
duplicate IDs. The new heading uses its generated ID; a redundant explicit
anchor was removed before final build.

Read the new branch explanation and original PyTorch export classes on the
rendered page. Code retains ordinary internal horizontal scrolling. The
original Dynamo pipeline diagram remains in context; small phone labels need
the viewer. At 2×, both left/right extremes were panned and inspected, including
the default Python path, Guards, transformed code, FX graphs and compiler
backend. It is the same source diagram, not a newly drawn substitute.
The byte-preservation unit checks for the original pipeline still pass.

The EN route explicitly shows the language-unavailable fallback. It contains
the new section and loads all 15 unchanged source figures; body 345/345,
zero display-math overflow/errors and no duplicate IDs. This is not a completed
English translation; no English locale body was edited.

Saved final screenshots:

- `screenshots/2026-10-27-tracing-desktop.png`
- `screenshots/2026-10-27-tracing-mobile.png`

H-V1-07 remains partial and H-F1-07 remains re-review. The next source work is
individual native-figure disposition and remaining AD/runtime/training-step
reconciliation, not another rewrite of accepted RL or S8 introductions.
No staging, commit, push, deployment, delegation or live infrastructure mutation
occurred in this turn.
