# TensorFlow capture, SavedModel and target conversion

Status: content-and-local-ui-reviewed.

Workflow: `00 Учебник/19 Deployment, Reliability и MLOps/01 ML workflow.md`,
1428 lines, SHA-256
`82d43f96fb0bfee7d0d7c2efd5a1f669008b87d4b3bad92daf5a59b2d5cad569`.
Removing one bounded TensorFlow addition recovers the entire accepted 1088-line
revision `8d6cc94470edc9ca00b9bc2d032d009bd51a34b130bfbb49356bea3b1af2f4d8`.
The prior ONNX explanation, programs, source passages and eight figures remain.

The previous goal turn delivered the accepted ONNX export addition, so it was
content progress, not another RL review. This pass read the complete current
MLOps chapter, lines 1–390, 391–780, 781–1170, 1171–1560, 1561–1950 and
1951–2360. Its SHA remains
`cec102f93b2e1314a9aa80cb5d36421b465a49eb7083fd2c92cb08597d9ab007`.
Lifecycle, dependency/rollback compatibility, admission, hardware/software
isolation, costs and feasibility are already substantial explanations. No
definition-only rewrite or duplicated MLOps chapter is introduced.

## Source reading and fidelity

Source scope and exact downloads are recorded in the adjacent source README
and eight-entry manifest. Five Git blobs match pinned GitHub objects. The two
16 KiB HTML prefixes are marked incomplete and excluded as evidence. Complete
pinned Keras Markdown and official web pages support the Keras distinction.
The complete official TensorBoard raster is retained with the source notebook's
Apache-2.0 notice and publicly served full root license. Its abbreviated labels
are original, not crops by Bookvar.

Read the full graph-guide prose/code, SavedModel cells 0–54 and complete
selected custom saving/loading/fine-tuning/signature sections, Keras migration
Saving/Loading sections and converter Example usage. Five original code cells,
four complete TensorFlow explanatory passages and the complete Harvard
exit-path paragraph are retained in English. Exact source comparisons run in
the CPU checker. The old source Keras API, prototype-splitting release note,
versionless speedups and obsolete Estimator pipeline are not current recipes.

## Executed examples

All eleven displayed Python fences execute on CPU with TensorFlow 2.21.0 and
NumPy 2.3.5. The source scalar graph gives 3, mutation gives 6; vector dispatch
fails for the saved scalar method. The unchanged source fine-tuning cells give
`v=4.981860160827637` after ten updates, matching `5-3*.6**10`; the chapter
explains pre-update printed loss versus post-update printed variable.

Named export saves v=2; a loaded object can change to 3 without rewriting the
artifact. A fresh Python interpreter containing neither source class loads the
artifact and returns [6]. The directory contents are inspected. The multiplication
converts to Lite and invokes with builtin CPU kernels and no default delegates.
120 independently generated cases match SavedModel/Lite/scalar equations exactly.
Eight application-invalid inputs are rejected; direct float64 fails at the
runtime boundary. The restored method rejects scalar/matrix shapes while the
flattened signature wrapper accepts them for this graph. The first checker's
shape-rejection assumption failed and led to an explicit counterexample and
application validator in the actual chapter; it was not hidden by a weaker test.

Temporary model artifacts are removed. TensorFlow's bundled interpreter emits
its deprecation warning; the chapter discloses the pinned deprecated API and
points to the successor. The checker's linecache exposes the actual code fences
to normal Python source inspection; it does not modify the source programs.
No GPU, phone/MCU, hardware delegate, TensorFlow Serving process or benchmark
is claimed. An actual CPU conversion is not proof of those targets.

## Final reading and source comparison

Read all 1428 final lines in ranges 1–300, 301–600, 601–900, 901–1200 and
1201–1428 on the stated SHA. No chapter change followed that reading. Applied
the fully read no-ai-slop skill/eval with academic qualifications. Original
rhetorical wording and comments are retained; the added explanation uses
concrete state, call paths, equations and observed failures. Style review is
not proof of authorship or of completion of the whole textbook.

The explicit comparison resolves the platform item rather than leaving an
already compared topic in the queue:

| Source interval | Actual explanation or source finding |
|---|---|
| 3918–3992 TensorFlow | Eager/capture, original custom module, saved graph/state, restored gradients and explicit endpoints. Actual CPU execution, not an exclusive production-framework claim. |
| 4045–4123 comparison | Existing numerical PyTorch/JAX cases and the original TensorFlow program explain state/capture. The malformed Keras input_shape and reused JAX key remain findings, not a benchmark. |
| 4124–4151 runtime map | ONNX kernel/batch controls, SavedModel/signature and CPU Lite boundary, Edge arena and MLOps resource/isolation cases. No fixed speedup/op-count/footprint promises. |
| 4152–4178 targets | Executed CPU conversions and Edge's 20/28 KiB lifetime/device ledger. No universal 10 ms/50 KB or automatic phone/NPU/JS support. |
| 4183–4257 selection | Workflow's hard quality/capacity case and iteration budget, S8 capture, Edge memory/operators and the fully read MLOps compatibility/admission/FinOps cases supply the separate feasibility filters. |
| 4258–4265 viability | Complete original exit-path paragraph, real class-free load and missing custom kernel connect maintainability to executable dependencies, not community rankings. |

Platform references in the ladder, execution and pitfalls ranges point to these
named destinations. The globally held native-image candidates are independent
of this text's target constraints; an unreferenced image is not evidence that
an explained paragraph is missing. Those candidates remain open, not published.
JAX runtime semantics, held figures, other course units and the broad textbook
remain incomplete; H-V1-07 is not accepted as a whole.

## Actual rendered evidence

Desktop 1280×900 and mobile 360×800. The original image is 632×767.42 on
desktop and 328×398.28 inline on mobile, uncropped. Mobile inline labels are
small; the viewer was opened at 2× and actually moved from upper-left to
lower-right. Final offsets are x=344/344, y=240.5 (integer maximum 240).
Top Identity/ReLU nodes and lower MatMul input nodes were seen. Abbreviated
node names remain as in the source; captions and public license remain visible.

The new two-column table is 328/328 and was inspected on the actual page.
Body is 360/360, display formula 328/328, no inline math right overflow,
KaTeX errors or duplicate IDs. The EN route contains the new section and an
explicit “This content is not available in your language yet” warning; it
is a fallback, not a translation. Screenshots:
`docs/audit/screenshots/2026-10-27-tensorflow-export-*`.

161 scoped source/registry/map tests and all 581 unit tests in 85 files passed.
153 output tests passed, including root/EN anchors, source image and public
license bytes. First full build: 1969 pages / 25.39 s. Old workflow/feedback
programs, original-source checks and ONNX programs passed on the current SHA;
their baseline checks retain all prior assertions. The final formal-map and
published-matrix note update requires the final metadata rebuild/checks below.
No stage, commit, push, deployment, EN-body edit, delegation or infrastructure
mutation occurred.

Final metadata build after both published matrix updates: 1969 pages / 22.46 s,
completed 22:43:53. All 153 output tests passed again, links report zero broken
routes/fragments/files, structural coverage passes. The twelve affected source/
workflow/map tests passed after platform closure. Scoped diff whitespace checks
pass. Actual final source-map UI shows the closed platform item and the updated
H-V1-07 row, with JAX and three figure candidates still open.

A navigation during regeneration briefly hit the static preview's 404 page.
The specific build handle completed exit 0; reloading afterward showed the
actual final chapter and all nine distinct source figures loaded. The table's
lower rows were also inspected. This was not left as an unresolved page error.

Final SHA is unchanged from executed/reread/inspected content. All validation
handles are terminal. Logs use `/var/tmp/bookvar-tensorflow-export-` prefixes:
`cpu-final`, `unit`, `scoped-final`, `build-final2`, `output-final`, `links-final`,
`coverage-final`, `workflow-old`, `feedback-old` and `onnx-old`. The full goal
remains active; no publication, translation or broader course-completion claim.
