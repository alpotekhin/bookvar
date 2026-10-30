# Frameworks: section-by-section reconciliation

The complete decisions are in
[2026-10-27-frameworks-section-map.json](2026-10-27-frameworks-section-map.json).
They partition all 4762 lines of the pinned source into the 37 actual H2/H3
sections, including the front matter in the first range. Fourteen destination
chapters are bound to their current SHA-256 and a named review record.

The named comparisons are now resolved for this pinned source chapter, with
source findings retained. This is not acceptance of the entire course.
The source prose, examples, captions and pseudocode were read in the preceding
Frameworks pass; long TikZ coordinate blocks were inspected as rendered figures
instead. All 56 distinct published/native byte objects have individual decisions
in the separate native-figure report. The map does not count a source link,
an archived page or a matching hash as proof that its mechanism is explained.

## Already evidenced mechanisms

The section records connect the actual source explanation to the current
worked material: tensor storage/strides, arithmetic and memory limits,
eager/capture/tracing failures, parameter registration/state restoration,
forward/reverse differentiation and custom rules, the full CPU training step,
and the new original grad/vmap/jit example. Revision-specific reports carry
the numerical and rendered evidence. The source's organizing taxonomy and
recaps remain in the source layer rather than becoming another introductory
chapter that repeats definitions.

Source findings are recorded separately from missing teaching material.
For example, the obsolete ONNX hub arrows, blanket LiteRT no-training claim,
fixed speedup/utilization/migration ranges, metaclass/load_state_dict errors
and universal backward=2x estimate are not facts to import. Their presence
in an open course is not measurement evidence. Correct worked explanations
are mapped where already available; unresolved target comparisons remain open.

## Resolved comparisons and additions

The hook gap is now closed separately by
`docs/audit/2026-10-27-module-hooks.md`: the complete original and numerical
input/output/weight derivatives, firing cases, cleanup and bypass are accepted
with executed and rendered evidence. The entire prior 838-line MLP is preserved.

The TinyML range 1893–1938 is now mapped separately in
`docs/audit/2026-10-27-micro-arena.md`: complete Harvard decisions, TFLM original
image/text diagram and C++ fragments, numerical lifetime/phase/device budgets,
operator support and actual desktop/mobile inspection. This closes that
micro-runtime explanation, not every export/delegate/platform comparison.

The PyTorch/ONNX export boundary is now accepted separately in
`docs/audit/2026-10-27-model-export.md`. Complete current Serving/Workflow
were read; Workflow retains the original hard-constraints callout, PyTorch
capture paragraph and ORT flow/figure. Actual CPU export, symbolic batch versus
range validation and an unregistered custom kernel are worked through and
executed. Desktop/mobile/viewer and explicit EN fallback were inspected.
The complete previous 832-line Workflow is preserved. This does not close
TensorFlow-specific conversion or the rest of the platform comparison.

The subsequent TensorFlow pass closes that platform comparison in
`docs/audit/2026-10-27-tensorflow-export.md`: all current 2360 MLOps lines were
read, and Workflow preserves full original TensorFlow graph/save/load/gradient/
signature explanations, five code cells and the original diagram. Eleven visible
fences, a class-free subprocess and CPU Lite conversion/invocation were checked.
Hard feasibility, lifecycle compatibility and exit paths have explicit worked
destinations; source speed/count/brand claims are recorded as findings. The
entire previous 1088-line ONNX/Workflow revision is preserved. This closes
the platform item, not the whole Frameworks unit or untested device targets.

The distinct JAX runtime comparison is now recorded in
`docs/audit/2026-10-27-jax-runtime.md`: original counter/state/ref/rank/cache/
scan and psum explanations, full original diagram, 22 printed Python fences
and 96 independent CPU cases. Four logical CPU devices execute the reduction;
the vmap contrast uses an explicit matching one-device mesh. It closes the
named source comparison, not all JAX documentation, GPU performance or
multi-host training. All 2400 prior S8 lines remain byte-identical.

The 3D placement comparison is now recorded in
`docs/audit/2026-10-28-frameworks-placement.md`: complete original SVG and
caption, 32 logical coordinates, explicit TP/DP/PP groups and global batch
30 versus the incorrect 480. The full distributed overview was read and its
917-line prior revision is preserved. Placement is not treated as physical
topology or a pipeline schedule.

The final two unreferenced rasters were compared with the entire current
1118-line Edge/FL and 893-line BERT chapters on 28 October. Both are now
reference-only: existing original figures and worked explanations already
cover their content more precisely. Their primary URLs and hashes remain
recorded; third-party redistribution permission remains unverified. No new
copy is published and no source archive or textbook chapter is changed.
See `docs/audit/2026-10-27-frameworks-native-figures.md`, final section.

The current Edge hash binds the 1118-line micro-arena revision; removing its
one addition and restoring two dates recovers the full accepted 871-line
revision. The 20 October review's older hash is not substituted. Current workflow and MLOps hashes also bind their later
extensions, with their corresponding reports. The serving revision matches
the source-originals review. No old destination revision is silently substituted
to make the map pass.

## Consistency checks and limits

The dedicated unit tests compare headings to the actual source outside code
fences, require a gap-free line partition, verify current destination hashes
and review-file existence, and reject unknown target/open-item/finding IDs.
They distinguish resolved comparisons from retained source findings. These checks
protect the record from drift; they do not read an explanation or establish
its scientific or pedagogical adequacy.

The earlier JAX gradient addition has its own 566-unit/149-output/site-build and actual-browser
acceptance. Adding this non-published source map does not change those chapter
bytes. The published transfer-matrix row was regenerated and seen in the actual
local site. After the later hooks addition, both maps use the new MLP revision
with inverse-preservation evidence. All twelve affected hook/module-state/map
tests passed; the final build has 1969 pages, 150 output tests and zero broken
links. Those historical gates are not semantic completeness. After the later
runtime, placement and final figure comparisons, H-V1-07 records thematic
review complete with findings for its 37 source intervals, not a whole-course
completion claim.
Harvard labs/decks, other course
units, deferred EN parity and the broader textbook goal remain separate work.
No source archive, English body, accepted RL chapter, git publication or live
infrastructure was changed by this map.
