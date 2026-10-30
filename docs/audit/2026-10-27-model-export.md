# Model export and portability

Status: content-and-local-ui-reviewed.

Canonical destination: `00 Учебник/19 Deployment, Reliability и MLOps/01 ML workflow.md`.
Current revision: 1088 lines, SHA-256
`8d6cc94470edc9ca00b9bc2d032d009bd51a34b130bfbb49356bea3b1af2f4d8`.
Removing the one bounded model-export addition and restoring two dates recovers
the entire accepted 832-line revision, SHA-256
`329710d26fc09b696036c217d07f7151e9b0879394411dbc21e54e24d1911869`.
The previous workflow/feedback explanation, seven Python blocks and seven
source figures were not replaced.

## Explanation and source comparison

The previous goal turn only rechecked already accepted RL revisions. That was
a status answer, not content progress. This pass instead compares the full
611-line Serving engines and 832-line Workflow chapters with Frameworks
3918–4265. The serving chapter already explains request scheduling and model
execution; it is not a generic ONNX deployment chapter. Workflow is the existing
destination for a candidate becoming a versioned executable artifact.

Read both complete ONNX Runtime pages (89/79 lines), both complete PyTorch
ONNX pages (138/252), the complete MIT license and the existing full PyTorch
license. Six downloads' Git blob hashes were verified at their pinned commits.
The source README and manifest record exact files. The complete Harvard
hard-constraints callout, PyTorch tensor-capture paragraph and ORT architecture
sequence retain source English and reasoning. Only documentation reference/link
syntax is adapted. The original ORT architecture PNG is byte-identical, not
a new illustration. Its MIT copyright/license is also served publicly.

The addition explains weights versus executable graph versus application,
opset versus installed kernels, provider partitioning/fallback boundaries,
captured shape bounds versus application validation, numerical checks and the
release's external dependencies. It does not transfer obsolete ONNX hub arrows,
versionless performance/footprint/migration figures, or the old design page's
blanket CPU completeness as universal facts.

## Executed CPU evidence

`publishing/tools/check_model_export_examples.py` executed both visible fences
with PyTorch 2.14.1 / ONNX 1.23.2 / ONNX Script 0.7.2 / ONNX Runtime 1.31.1 /
NumPy 2.3.5. Only CPUExecutionProvider was registered. The two-row calculation
is exactly `[[.5,6.5],[8.5,0]]`. All seven allowed batch sizes agree with
PyTorch on the chosen random inputs at the chapter's tolerance. Separately,
140 scalar-equation cases agree at rtol 2e-6/atol 1e-5; their largest absolute
error is 7.62939453125e-06, so this wider-input control is not misreported as
universal exact equality or as an absolute 1e-6 bound.

Eight invalid application inputs and three direct runtime-invalid inputs are
rejected. Batch 9 is rejected by the originating ExportedProgram, executed by
this symbolic-batch ONNX graph, and rejected by the application wrapper.
A custom-domain model passes ONNX checking but session construction rejects
the unregistered kernel. Both source programs passed on the first run; the
extra checker's first run caught RuntimeError but PyTorch raised AssertionError
for its shape guard. The checker was corrected to recognize both, without
changing the chapter to hide the observed behavior. Missing torchvision emitted
optional registration warnings for unrelated vision operators; those operators
are absent from this graph.

No GPU backend, TensorRT conversion, task accuracy, latency, server, cluster or
firmware was executed. Structural tests and a successful export do not establish
those results.

## Final reading and rendered evidence

Read the whole final chapter in ranges 1–285, 286–565, 566–835 and 836–1088;
the later source-list blank-line adjustment was reread at 503–515. No mechanism,
code or prior paragraph changed after that read. Applied the no-ai-slop checklist
with the project's academic qualifications: source prose and comments remain
untouched, necessary terminology and meaningful distinctions are retained.
The two-row derivation, actual export boundary and negative controls carry the
explanation rather than a sequence of definitions. Style checks are editorial
guidance, not evidence that the whole textbook is complete.

Actual browser, desktop 1280×900 and mobile 360×800: the complete new image is
632×395.54 in the desktop column and 328×205.28 inline on mobile. No crop.
The mobile inline labels are too small for detailed reading; the viewer was
opened and inspected at 2× (688×430.59), including its left and right edges.
Rightmost scroll is 344/344; source Input Data/ONNX Model and Output Result
were actually seen in their respective views. No claim that all labels are
readable in the mobile inline thumbnail. Full attribution and MIT link remain
in the viewer. Eight published source figures load; the hidden retained viewer
image is not counted as a ninth teaching illustration.

Mobile body is 360/360, the one display formula is 328/328, inline math has no
right overflow, KaTeX errors and duplicate IDs are zero. The new code fence
has a 326 px viewport and 715 px scroll width; its actual right edge 389/389
was reached and then returned left. The EN route shows the new addition and
the explicit “This content is not available in your language yet” warning.
It remains a fallback, not a completed translation. No EN body was edited.

Screenshots under `docs/audit/screenshots/2026-10-27-model-export-*` record
desktop, mobile inline, both viewer edges, formula, code edge and EN warning.

Current automated evidence: both new programs and controls on final SHA;
old seven workflow fences/800 independent cases/11 source hashes/six original
Google figures, plus three feedback fences/600 cases, all passed. The inverse
test preserves the entire 832-line accepted baseline. 578 unit tests in 84
files passed, 152 built-output tests passed, and the first full Astro build
completed 1969 pages in 25.37 seconds. The new output test checks root and EN
route anchors, image, content and public-license bytes. A final rebuild follows
the published transfer-matrix acceptance-note update; links/coverage are checked
against that final build separately below.

This accepts the export addition, not the complete Frameworks platform unit.
TensorFlow-specific capture/SavedModel/target paths, full platform criteria
versus current MLOps, JAX runtime semantics and held figure decisions remain
open in the section map. H-V1-07 stays partial. No stage, commit, push,
deployment, delegation, infrastructure mutation or accepted RL edit occurred.

Final build after the published acceptance-note update completed 1969 pages
in 22.74 s at 22:23:49. All 152 output tests passed again, built links have
zero broken routes/fragments/files, and structural coverage passes (210 rows,
58 named destinations, 34 Harvard labs, 35 decks, 67 HSE artifacts). These
inventory counts do not prove their semantic integration. The nine affected
model-export/workflow/map tests passed after the map update. Scoped
`git diff --check` passed. The actual source-map page was opened and showed
the accepted export note rather than its earlier render-pending text.
The final Workflow hash is unchanged from the executed and inspected revision.

Logs: `/var/tmp/bookvar-model-export-{cpu,unit,scoped-final,build-final,output-final,links-final,coverage-final}.log`.
All process handles reached exit 0; no validation job remains running.
