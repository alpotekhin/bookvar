# Harvard links in Transformer/MoE arithmetic — 2026-10-13

## Scope

Repair the missing Harvard connection in S4 without rewriting its recently
revised explanation or claiming that every imported course object has been
integrated. Only the Russian S4 chapter, four Harvard transfer-matrix rows,
and this audit were changed. No English text, registry, navigation, original
course source, or asset was changed.

## Source inspection

Read all 970 lines of the pre-edit S4 chapter. The relevant Harvard material
was then inspected directly in the preserved source layer:

- `05 Источники/Courses/Harvard ML Systems/vol1/nn_computation.md`:
  provenance, *Model size and computational complexity*, *Forward pass
  computation*, and the backward-pass/memory-accounting explanations.
  The MLP has widths 784–128–64–10. Its parameter table includes biases;
  its FP32 training-state calculation includes gradients and two Adam moments.
- `05 Источники/Courses/Harvard ML Systems/slides/vol1/05_nn_computation.md`:
  *Training Memory Breakdown*, *Forward and Backward Pass*, *Backprop Memory
  Accounting*, *Your Turn: Training Memory Budget*, and *Batch Size*.
  These are teaching simplifications, not universal memory ratios. The local
  page also retains the original Beamer code; a source link is not a claim
  that the whole deck was reconstructed as a polished Bookvar chapter.
- `05 Источники/Courses/Harvard ML Systems/labs/vol1/lab_05_nn_compute.md`:
  narrative, imports, profile selection, shape/batch/precision controls,
  budget ratios, and the operation-ledger calls. The lab operates on
  estimated values from helper models; it is not a profiler measurement.
- `06 Практика/18 Harvard ML Systems — интерактивные design labs.md`:
  existing execution instructions explicitly distinguish preserved source
  from an application running inside Bookvar.

The book snapshot is `2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd`.
The lab/slide snapshot is `45ecc8d82fcae70c149cdce550d3b3d3411df913`.
This distinction is now recorded beside the S4 evidence in the matrix.

## Changes

S4 now connects its training-state calculation to the complete Harvard MLP
example. It explicitly limits the factor-four comparison to FP32 persistent
states versus FP32 weights. Its source/practice list links to the full
chapter, the preserved slide source, Lab 05, and the existing Marimo guide.
The suggested `cloud_fleet` exercise varies one parameter at a time and
identifies the first resource limit; the text labels outputs as model
estimates rather than GPU measurements.

The four affected transfer rows now state the actual result:

| Row | Current evidence |
| --- | --- |
| H-V1-05 | Direct chapter/section links and a recomputed small-MLP example |
| H-L1-05 | Direct lab link and concrete practice route; runtime unverified |
| H-SL1-05 | Named relevant sections and a source link; not full-deck integration |
| H-F1-05 | Source-only figure collection; no claim that Harvard figures are embedded in S4 |

Existing S4 illustrations, section titles, anchor, source-unit IDs, and
previous/next navigation were preserved. The Harvard figure files were not
deleted or replaced. Their independent presence in the source chapter does
not prove incorporation into S4.

## Executed checks

- Before the edit, `pnpm --dir publishing check:source-coverage` failed on
  exactly one destination: S4 lacked a direct Harvard source marker.
- After the edit, the same check passed: 210 matrix rows, 59 destinations,
  215 Scikit-learn MOOC pages, 212 figures, 7 course-linked canonical pages,
  34 Harvard labs, 35 slide decks, and 67 HSE artifacts. This check confirms
  existence/source connections, **not completeness of thematic transfer**.
- A Node calculation independently returned 109,386 parameters, 437,544
  bytes for FP32 weights, and 1,750,176 bytes for the four FP32 state arrays.
- The three course destinations are present in `publishing/navigation.yml`.
  The existing built Harvard chapter has the target fragment
  `model-size-and-computational-complexity`.
- `git diff --check` passed for both edited content files.

## Acceptance boundary

This is a local source-link/provenance repair. No build, browser inspection
of the changed S4 page, Marimo execution, commit, push, or deployment was
performed for this subtask. The parent editorial pass must validate the
new links in its next site build. Full Harvard thematic integration remains
a broader editorial task and is not declared complete by this audit.
