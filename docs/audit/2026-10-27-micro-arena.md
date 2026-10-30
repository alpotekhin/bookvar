---
status: content-and-local-ui-reviewed
chapter: 00 Учебник/19 Deployment, Reliability и MLOps/03 Edge и federated deployment.md
route: /textbook/deployment/edge-federated/
before_sha256: 9038a3d22ef6b746b7a99720542763f94b8117641c84543499433305a001c7e3
after_sha256: 8f10fda46877de9a593fb29200c2511d6727e4431f75f1e7b5c5283e7010cd8d
---

# Fixed-arena inference: source comparison and bounded addition

The preceding goal turn delivered JAX and hooks, with actual code and rendered
evidence. It was progress. This pass follows the next concrete Frameworks
gap instead of reopening accepted RL, AD or MLP.

Read all 871 current Edge/FL lines in consecutive ranges 1–310, 311–590 and
591–871. That revision already thoroughly explains local training, TinyTL,
LiteRT checkpoints, FL, secure aggregation, DP and fleet release. It does not
explain a micro-runtime's tensor arena and operator-registry constraints.

The addition preserves every byte of that accepted revision outside one
bounded section and two dates. Its removal and date restoration recover SHA
9038a3d2…c7e3. The six older source figures, five Python blocks, training-memory
source excerpt, all FL/DP explanations and existing anchor meanings remain.
The old TinyTL anchor is after the addition, still immediately before TinyTL.
Final chapter length is 1118 lines.

Source scope: complete Harvard Frameworks 1893–1938, including its KWS scenario
and all three decisions; exact source English is retained. The full Frameworks
chapter was already read earlier. TFLM memory management, online allocation
overview and preallocation RFC were read completely; the full porting FAQ
and selected resolver lookup/registration implementation were read. Exact pins,
hashes and reading bounds are in the new source README/manifest. Every retained
source Git blob matches the pinned GitHub contents API; no full clone or model
weights were downloaded. Only the existing RFC background diagram is reused;
the proposed external-buffer API is not presented as supported functionality.

The explanation separates model buffer, arena and whole-device memory, then
works through overlapping buffer lifetimes. Assigned sizes 8/12/8/4 KiB have
32 KiB sum but a 20 KiB simultaneous lower bound, attained by explicit offsets.
Extending the original input's lifetime makes that layout invalid and raises
the demonstrated lower bound/plan to 28 KiB. Inclusive operator endpoints and
out-of-place operation assumptions are explicit. A supplied-plan validator is
not called an optimal allocator or the actual TFLM planner.

An assigned preparation/invocation ledger gives 32 versus 24 KiB, and a separate
256 KiB SRAM ledger leaves 112 KiB after its declared reservations. These are
bookkeeping inputs, not measurements or a universal allocator phase sequence.
Both complete original C++ recording fragments and the original text arena
diagram are preserved. Missing setup, non-constant C++ array bound, alignment,
recording overhead and the separate 9568-byte documentation output are explained.
No C++ compilation, firmware, conversion, energy or timing experiment is claimed.

The new checker executes both actual Python fences, verifies whole-chapter
preservation, compares 400 independent byte/step occupancy cases and checks
invalid/aligned/boundary controls. It also verifies all seven source objects,
the exact Harvard passage, both C++ fences and text diagram. Current final-SHA
execution and the old five-block/1200-case regression are recorded below.
No-ai-slop review preserves original wording, derivation, caveats and examples;
the new content is not a short API list.

The new PNG is complete, 271×229, SHA
692475fcb0609be2a97f30078c3b9e84ac90130e14e2c705df483e922723ca6b.
Its background labels were inspected in the original before selection. The
root Apache-2.0 license and adjacent TensorFlow credit are retained; the pinned
root has no NOTICE. Native-size CSS prevents upscaling. No crop, redraw,
translation, recoloring or recompression. Both adjacent/global provenance bind
its page, image, source/license hashes and destination. A small original is
not described as a high-resolution vector or enlarged to pretend it is one.

## Final reading, execution and publication evidence

After editing, read all 1118 lines in consecutive ranges 1–380, 381–760 and
761–1118. No change followed that final read. The new checker passed on the
final SHA: two actual Python fences, 400 independently checked layouts
(179 valid, 221 rejected), exact source fragments and inverse preservation.
The older checker passed its five actual blocks, 1200 independent cases,
16 source hashes, exact LiteRT excerpt, three reproduced legacy PDF crops
and eleven retained anchors. Its first run lacked Pillow; an isolated uv
environment with NumPy/Pillow/pypdf ran it successfully. This was an environment
failure, not a failed mathematical example. The thirteen Marimo-import
regression tests also passed, retaining the old Edge quote/fingerprint check.

All 575 unit tests in 83 files passed. The final site build produced 1969
pages in 25.25 seconds. All 151 output tests passed on that build, including
the new root/explicit-EN-fallback page and served license checks. Built links:
zero broken routes/fragments/files. Source coverage passed; its structural
counters are not semantic-completeness evidence. Scoped tracked-file
git diff --check passed. Logs: /var/tmp/bookvar-micro-arena-*-final.log,
bookvar-micro-arena-old-edge.log and bookvar-micro-arena-lab-regression.log.

## Actual rendered page

Inspected the complete original image and surrounding explanation at desktop
1280×900. All four labels are readable at 271×229. At mobile 360×800, the
inline image retains exactly 271×229; the viewer was opened at 1× and all
four labels/borders and attribution inspected. It is not enlarged to column
width or described as a vector. All seven published figures load; the six
old originals remain byte-verified by the legacy checker and asset tests.

The new four-column lifetime table is 328/328, with all four rows and offsets
visible together. Body is 360/360; all thirteen display equations and inline
math fit, no KaTeX errors or duplicate IDs. The original 80-column text diagram
scrolls inside its pre block: both ends were exercised and inspected, with
scrollLeft 0 and 412.5 (integer scrollWidth minus clientWidth reports 412).
The right view shows TAIL and Highest Address; the left shows HEAD/TEMPORARY
and Lowest Address. The diagram bytes were not reflowed or redrawn.

The actual EN route shows the new section and an explicit untranslated-content
notice. This is fallback, not English parity or a new translation. Screenshots
in docs/audit/screenshots/:

- 2026-10-27-micro-arena-desktop.png
- 2026-10-27-micro-arena-lifetimes-mobile.png
- 2026-10-27-micro-arena-viewer-mobile.png
- 2026-10-27-micro-arena-ascii-mobile-left.png
- 2026-10-27-micro-arena-ascii-mobile-right.png
- 2026-10-27-micro-arena-en-fallback.png

## Remaining overall work

The selected TinyML source range can be mapped to this accepted explanation;
that does not close every source platform or export section. Broader framework/export and JAX
runtime comparisons remain open. EN bodies, immutable previous sources, RL,
staging/commit/push/deploy and live infrastructure were not changed.
