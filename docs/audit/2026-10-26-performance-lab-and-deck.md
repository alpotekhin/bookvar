# Performance Engineering lab/deck — thematic review, 2026-10-26

Destination: `06 Практика/13 Оптимизировать один Transformer step.md`.
Final SHA256:
`37572b4aef27332ce5187d2244de5073225563d3576f135ea062197326b85aab`.
Route: `/practice/13-optimizirovat-odin-transformer-step/`.
Anchor: `#performance-evidence-profiling-regression-capacity`.

## Source and explanation review

Read the entire original Harvard laboratory (1602 lines) and Beamer source
(1796 lines, including all speaker notes, 37 explicit frames, title/focus,
references and backup). Both pinned originals match the full originals
inside the repaired course wrappers. This was a source read-through, not
acceptance inferred from filenames or extraction coverage.

Read the complete previous practice page before editing, then the entire
expanded explanation. The bounded addition preserves the previous EDLS and
TinyTorch routes byte-for-byte after reversing metadata; the fidelity test
reconstructs previous SHA256
`444962666a6abf9f73459c32ce666a74da182003d90840cab1cf3aa217aa59d0`.

Source pins and exact selection scopes are in
`05 Источники/Originals/performance-practice-2026-10-26/README.md`
and `references.json`. Full Harvard instructions and feasibility expression
remain English. Two complete SciPy paragraphs are retained with only
docstring indentation removed. SciPy's full copyright/license is retained
locally and in public licenses. No translation source was authored.

The four lab decisions now have full explanations and worked solutions:

| Decision | Canonical treatment and evidence |
|---|---|
| A: bottleneck localization | Workload/timer contract, actual trace interpretation, causal hypothesis, correctness and full-step remeasurement; original CS336 trace with source/meaning/limits. Links to complete S3/S7 instrumentation. |
| B: regression evidence | Independent versus paired observation units; executable Welch mean interval; sample-size planning with significance and power; separate request-p99 and quality questions. |
| C: capacity | Rate measured under a declared SLO, additive capacity bound, utilization/headroom conventions, explicitly assumed source derating, cost and required-unit calculation; no invented p99 predictor. |
| D/synthesis: trade-off | Four rows computed by original helpers, not measured hardware/quality; feasibility before preference, changed quality budgets, no-feasible-option case, evidence handoff and rejected experiments. |

The no-ai-slop checklist was applied to authored prose without rewriting
licensed source excerpts. Mechanisms, calculations, assumptions, limitations
and source attribution remain; the extension is not a shortened glossary.
The training baseline and serving-capacity exercise explicitly have different
contracts and are not presented as one interchangeable benchmark.

## Numeric review

The source lab was not imported or launched. Its bootstrap, Marimo, ledger,
report-export and cloud code were not executed. An AST checker selects only
eight reviewed pure numeric helpers and calls them with the cloud fixture.

Observed source limitations and disposition:

1. Source `mde_pct = 1.96*sqrt(2)*CV/sqrt(n)` has no power requirement.
   The practice explains that it is a known-variance normal interval threshold,
   not an 80%-power MDE. At n=8 and sigma=6 ms, the half-width is 5.88 ms,
   an approximate 80%-power shift is 8.40 ms, and a 5-ms shift needs 23 per arm.
2. Source p99 and quality losses are configured teaching values. The practice
   never promotes them to measured request quantiles or model-quality results.
   Its separate mean fixture produces delta −4.500 ms, Welch interval
   [−6.907, −2.093] ms and p≈0.001350 under the stated assumptions.
3. The source queue multiplier drops from about 7.986 at rho=.999 to 3 at rho=1.
   This was reproduced on CPU, not adopted as a latency model.
4. Both displayed tables and the four Python fences are checked against actual
   execution, exact constraints and original helper outputs. The 12% derating
   is labelled a scenario assumption. Feasible sets for quality budgets 2%,
   1%, .5% are respectively targeted/precision, targeted, and empty.

SciPy 1.18.0 was executed in an isolated uv environment. This is CPU arithmetic
and statistics, not a GPU benchmark, training run, quality evaluation or SLO
validation. No real capacity purchase, rollout or release action happened.

## Deck disposition and figures

| Deck section | Destination / exclusion |
|---|---|
| Iron law, memory wall, roofline and batch size | Existing S2/S3/S4 and inference physics. No repeated introductory chapter added. Unqualified “zero improvement” and universal utilization percentages not adopted. |
| Fusion, FlashAttention and CUDA Graphs | Existing S8 and FlashAttention chapter. Original PyTorch traces and source-aware explanations remain; generic speedup/traffic numbers are not substituted for measurements. |
| Precision and KV compression | Existing mixed-precision, quantization and KV-compression chapters. Fixed calibration times and guaranteed proportional throughput are not universal facts. |
| Compilation | Existing S8/serving engines. Source “TensorRT static shapes” is qualified by official support for runtime dimensions and optimization profiles. |
| Speculation and MoE | Existing mechanism chapters and arithmetic. No “compute is free,” guaranteed speedup, or universal instability above 64 experts adopted. |
| Profiling, diagnostic loop, playbook and synthesis | Existing S7 plus practice A–D. The staged 70B figures remain a source illustration, not a reproduced hardware case. |
| FlashAttention arithmetic exercise | Its source 2N²d omits one of the two dense attention products; together QKᵀ and PV give 4N²d before softmax. Reading Q/K/V once is not a general tiled-HBM traffic bound. The shortcut was not transferred as a solved example. |
| Backup/reference/retrieval frames | Original archive retained. Their repeated definitions do not require duplicate canonical pages. Named MBU and its numerator/physical-traffic limitations remain a concrete open item. |

Primary checks on disputed claims, verified 2026-10-26:
[FlashAttention official Tests](https://github.com/Dao-AILab/flash-attention#tests)
uses numerical tolerances, not a bit-identical-output guarantee;
[TensorRT Working with Dynamic Shapes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-dynamic-shapes.html)
describes runtime dimensions and optimization profiles. These are live
documentation checks, not frozen source pins or executed GPU tests.

All nine existing SVG candidates from the archived deck were visually
inspected via a temporary QA contact sheet. They remain in the original
archive; none was newly embedded, cropped, redrawn or claimed as a measured
plot. Several include the same unsupported fixed speedup/traffic assumptions
as the slide text. Existing stronger canonical originals remain in place.

The practice reuses the exact existing CS336 figure at 3278 × 1986, SHA256
`7dd8944a6c6e123547e92fbf38ab2cbc88a826b657b1c9f693955981ca092f6f`,
with MIT attribution and a new `used_in` entry. No new teaching graphic was
generated. The QA contact sheet is temporary and is not textbook content.

## Structural and rendered acceptance

- 464 unit tests / 48 files passed, including four new fidelity/source tests.
- Four actual Markdown Python fences executed; both actual Markdown tables
  parsed and checked; source-helper and constraint-boundary checks passed.
- Final build: 1965 pages, complete at 14:45:38.
- 135 built-output tests passed. Built links: zero broken routes/fragments/files.
- Source coverage: PASS. This is structural evidence, not full course acceptance.
- Scoped `git diff --check`: no errors. No staging, commit, push or deployment.

The local final page was inspected at 1280×900 and 360×800. The source figure
loads at its native resolution and preserves its aspect ratio. Its viewer
opens at 6×. At 360 px there are zero overflowing display-math blocks and
zero KaTeX errors; page width is 345 px. The two wide tables intentionally
scroll inside their 313-px content columns (466/652-px content widths).
Keyboard scrolling of the candidate table was observed; it is not a clipped
static image. The temporary viewport was reset and the viewer closed.

The `/en/` route was opened and has the new section/code with an explicit
“This content is not available in your language yet” fallback. This is not
an authored English translation. No `en/` source file changed.

Screenshots:

- `screenshots/2026-10-26-performance-practice-desktop.jpg`
- `screenshots/2026-10-26-performance-practice-mobile-math.jpg`
- `screenshots/2026-10-26-performance-practice-mobile-table.jpg`
- `screenshots/2026-10-26-performance-practice-mobile-viewer.jpg`

Logs:

```
/var/tmp/bookvar-performance-practice-unit-2026-10-26.log
/var/tmp/bookvar-performance-practice-examples-final-2026-10-26.log
/var/tmp/bookvar-performance-practice-build-2026-10-26.log
/var/tmp/bookvar-performance-practice-output-2026-10-26.log
/var/tmp/bookvar-performance-practice-links-2026-10-26.log
/var/tmp/bookvar-performance-practice-coverage-2026-10-26.log
```

## Remaining work

This extension's explanation/code/render checks are complete. It does not
execute the student's EDLS assignment or reproduce the original Marimo UI.
The next bounded topic is MBU and the distinction between a model-byte
estimate and measured physical HBM traffic in inference physics. Full course,
practice, source/rights and overall textbook completion remain open. Accepted
RL and the previously accepted S3/S7/S8 content were not rewritten.
