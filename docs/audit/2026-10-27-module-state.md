---
status: content-and-local-ui-reviewed
chapter: 00 Учебник/01 Основы нейронных сетей/01 Нейрон и MLP.md
route: /textbook/neural-networks/mlp/
before_sha256: 2fd8e90374a5802bd2c3997d36ee2d945e9aa706dc2e2810c023b39a4a695e05
after_sha256: 5e7aa6cd8ffc7af8ff995943582f13cbcc59c784ae2399801c16b0ee4f7547e0
---

# Module state: explanation review

This is a thematic addition from H-V1-07, not completion of Frameworks or the
Harvard course. The previous goal turn only answered the RL status question;
it made no content progress. This turn adds a missing explanation to MLP.

## Reading and source comparison

The entire prior MLP (564 lines) and backpropagation chapter (784 lines) were
read. The Harvard Frameworks source was finished through line 4762 for prose,
captions and executable/pseudocode examples. Long TikZ drawing blocks still
require actual figure inspection; H-F1-07 is not accepted on that basis.
PyTorch's full 724-line Modules note and the registration/state/mode contracts
in the pinned implementation were read. Archive and remaining source findings:
`05 Источники/Originals/frameworks-2026-10-27/README.md`.

Selected Harvard sections are Automatic parameter discovery, Mode-dependent
behavior and Hierarchical composition and serialization (3697–3871), with
the hooks example checked for the module-call contract. The complete
`lst-parameter_registration` code fence is unchanged. PyTorch's selected
sections are A Simple Custom Module, Modules as Building Blocks, Module State
and the train/eval discussion; a short original statement of recursive
parameter ownership is reproduced, with its license and attribution.

Backpropagation already covers the rule of chain, VJP/JVP, saved tensors,
graph rebuilding, leaf accumulation, freezing versus detach and eval/no_grad.
It is unchanged. MLP lacked an explanation of registration, despite invoking
`model.parameters()` in its existing worked training step.

## Mechanisms and worked cases

1. Existing MLP: its printed hierarchy maps both children to four parameter
   names, shapes and 17 scalar values. Ownership is distinguished from the
   graph of executed operations; reuse does not allocate new weights.
2. The full Harvard `CustomLayer(10,20)` has 220 parameter values and a separate
   20-value buffer. The buffer is never read/updated by this source forward;
   its name does not implement a running mean.
3. A 2×2 concrete case computes output `[0.5,6.5]`, loss 11, weight gradients
   `[1,2]` for both rows, and SGD weights `[[1.9,-1.2],[0.9,2.8]]`.
   A plain tensor receives gradient 4 but no update. A frozen registered bias
   remains present but has no new gradient. Conversion changes registered
   floating-point state while leaving the ordinary tensor attribute alone.
4. List and ModuleList use the same actual layer objects and yield the same
   forward result. Only the registered container exposes the 12 parameter
   values and six state entries. Additional tests show that the unregistered
   list can still differentiate, while recursive mode/dtype operations miss
   its contents. Shared-layer aliasing is explicit in the prose.
5. An actual BytesIO `torch.save`/`torch.load(weights_only=True)` roundtrip
   into an already-created destination gives output `[0,6]`. Freeze flags,
   scratch/plain attributes and architecture are not reconstructed. A size
   mismatch fails even under `strict=False`; missing/extra keys are distinct.
6. `.eval()` recursively changes flags but preserves parameter differentiation.
   `no_grad()` removes the output's reverse-mode recording. Module calls run
   registered hooks; direct `forward` calls bypass them.

The source's metaclass claim, frozen-parameter filtering implication and
hierarchy reconstruction claim are not repeated. The added prose also warns
that freezing does not clear an old `.grad`, backed by a separate SGD check.

## Preservation and checks

The entire accepted MLP body is recoverable by removing the bounded
`Module state 2026-10-27` addition, restoring the date and reversing two
explicit formatting-only equation reflows in `publishing/tools/mlp-review-layout.json`.
Its hash must equal the before revision above. The reflow test also strips only
layout commands to compare the old/new mathematical content. Both previous Python fences, all four source
figures, formulas, XOR proof, old forward/SGD calculations, FFN explanation
and stable headings are retained. No EN body was edited.

`publishing/tools/check_mlp_examples.py` executes all seven actual printed
Python fences, retains all old mathematical checks, compares the full Harvard
fence against the pinned source, and exercises the additional cases above.
`module-state.test.ts` checks preservation and source/license identity; these
integrity checks do not prove the prose's teaching quality.

The full 827-line chapter was reread in three consecutive ranges. The final
838-line revision additionally shortens two table headers, defines those
headers and reflows two unchanged calculations; all changed passages were read
again. The final inverse-preservation check binds the unchanged remainder.
No-ai-slop review retained necessary comparisons, mathematical reasoning and
the original source code/quotation. No introductory importance claims,
decorative recap or unsupported speedups were added. Four existing source
figures remain byte-for-byte unchanged; the new ownership mapping uses the
actual printed hierarchy and an exact name/shape table, not a synthetic diagram.

The Network Architectures map is rebound only for MLP, with this report as
current evidence. Removing the addition and reversing the recorded layout
changes restores its previously reviewed architecture explanation exactly;
all old CPU mechanisms are also rerun.
The other 16 mapped destinations were not edited in this turn.

## Executed validation

All native processes reached terminal exit 0; no running handle is inferred
from a log or state file.

- Seven printed Python blocks plus all prior/additional checks passed on the
  final SHA: PyTorch 2.14.1, NumPy 2.3.5, SciPy 1.18.1, CPU.
  `/var/tmp/bookvar-module-state-examples-final.log`.
  The first reused Torch environment lacked SciPy; an already-cached complete
  environment was used instead, without installing packages or removing checks.
- Full unit suite before the mobile-only formatting corrections: 75 files,
  548 tests passed in 43.25 s. `/var/tmp/bookvar-module-state-unit.log`.
  After the final headers/reflows/map revision, both affected integrity suites
  were rerun: six tests passed. `/var/tmp/bookvar-module-state-scoped-final.log`.
  The full 548 are not represented as a new run of the final revision.
- Final Astro build: 1969 pages, 22.00 s, finished 04:46:33 local.
  `/var/tmp/bookvar-module-state-build-final.log`.
- Final output suite: 143 tests passed in 4.19 s. New assertions cover all
  five anchors, code presence, source attribution and license path; these are
  rendering regressions, not a semantic-completeness test.
  `/var/tmp/bookvar-module-state-output-final.log`.
- Final built-link check: zero broken routes/fragments/files.
  `/var/tmp/bookvar-module-state-links-final.log`.
- Structural source check passed: 210 rows, 58 existing destinations.
  `/var/tmp/bookvar-module-state-coverage.log`. Later corrections did not change
  its source paths, links or mapping scope; it does not establish course transfer.
- The section-map checker reports 37 ranges / 17 destinations / 5256 source
  lines, no integrity errors. Scoped diff whitespace check passed.

## Actual browser inspection

Inspected the built localhost route after the final build, not stale generated
HTML. Desktop viewport 1280×900, phone 360×800. Desktop page width/scroll width
1265/1265; mobile 345/345. All 19 display equations and inline math fit at
360 px, with zero KaTeX errors. Two previously wide formulas were split into
aligned rows without changing operands or results.

The first new table initially clipped its Count column, and the state table
required horizontal scrolling. The final tables are both 313/313 on mobile;
all column text is within their right edge at x=329 (max text right 312.22
and 328.17). All four parameter rows and all four state rows were visually
inspected. The compact headings are defined in adjacent prose. Source code
retains its normal internal horizontal scrolling; it is not cropped as an image.

Four existing source images load at their original intrinsic widths
791/384/1031/1077. Their current SHA-256 values were compared with the four
accepted values in `2026-10-21-neuron-mlp.md`: all match. The inverse baseline
check separately preserves their Markdown paths and captions.
No new synthetic graphic was inserted. Prior individual figure acceptance is
retained for these unchanged files, not relabeled as newly high-resolution art.
The new API table is not claimed to close the broader native-figure audit.

The EN route was visited: it explicitly says “This content is not available in
your language yet.” The same four images load; math has no overflow/errors.
This is verified fallback, not a finished translation. EN bodies remain untouched.

Final screenshots:

- `screenshots/2026-10-27-module-state-desktop.png`
- `screenshots/2026-10-27-module-parameters-mobile.png`
- `screenshots/2026-10-27-module-state-mobile.png`

H-V1-07 remains partial; H-F1-07 needs individual native-object review.
The rest of Harvard/HSE/EDLS, source/rights findings, hardware/Marimo execution
and deferred English parity remain open. No staging, commit, push or deployment
was performed. Accepted RL chapters and live infrastructure were not changed.
