# Optimizer memory: source comparison, examples and original figures

Status: `content-and-local-ui-reviewed`, 28 October 2026.
Checkout: `editorial-route-alignment`; no stage, commit, push or deployment.

## Delivered reader material

Existing chapter:
`00 Учебник/01 Основы нейронных сетей/02 Оптимизация и стабильность обучения.md`.
Route: `/textbook/neural-networks/optimizers-schedules/#memory-efficient-optimizers`.

Before: 969 lines, SHA256
`ffd865468db1e8ea07c64ec84f74a31cd065935f4b8aeb09a0f1f5b65670e43b`.
After: 1276 lines, SHA256
`97370cdc6665f959eb59109cbe68cf2014e5247bfa1f21b2e68e9512b7a8c903`.
Removing only the bounded `optimizer memory 2026-10-28` addition,
restoring `last_updated` and removing its new `source_language: mixed`
field recovers every original byte. Both the Python checker and the unit
test independently verify this inverse, not a replacement baseline.

The entire prior chapter was read before editing. The complete final
chapter was reread in ranges 1–320, 321–640, 641–943, 944–1130 and
1131–end; the last intro/projection-orientation changes were read separately.
The full existing SGD/Adam, schedules, clipping, precision, checkpoint,
Muon/SOAP explanations and all old source/anchor content remain intact.
The English addition follows the existing advanced matrix-optimizer section;
no EN body was authored or translated and no separate short summary route
was created.

## Explanatory acceptance

| Mechanism | Source read and retained reasoning | Worked check |
|---|---|---|
| Factored second moment | Shazeer/Stern, Adafactor v1, §§3–4; physical PDF pp.2–4, full selected pages. Row/column marginal formula, nonnegativity, rank-one exactness, linearity through averaging and optional full first-moment cost. §§5–8 through the start of §8.1 read on pp.4–6 for update clipping, changing decay and relative step. | G1=[[1,3],[3,1]] gives bias-corrected factor estimate all 5, not the elementwise squared gradient [[1,9],[9,1]]. G2=[[2,0],[0,2]] with beta=.5 gives all 3. Code keeps a full matrix only as an independent reference. |
| 8-bit state | Complete official bitsandbytes usage and explanation pages, read online 28 Oct. Blockwise storage, higher-precision update, metadata, configurable 4096-element cutoff. No blanket 4x speed or 75% total-training-memory claim imported. | Ideal two one-byte moments count 32 MiB versus 128 MiB for two FP32 arrays. Weights/gradients and metadata are separate. No bitsandbytes execution or kernel reproduction claimed. |
| One-sided GaLoRE | GaLoRE v2 §§3.1–4.4, plus complete pinned README, projector and AdamW implementation. Positive gradient convention, compact Adam moments, expansion, left/right orientation, alpha, periodic SVD and retained moments across refresh. | diag(3,1), Q=e1: projected Adam changes only the first diagonal entry. Two separately supplied rank-one directions sum to rank two; with a fixed direction the sum stays rank one. History is deliberately disabled only in the geometric comparison. |
| Full rank versus full Adam | Prior chapter's coordinate-dependence example and GaLoRE §3.3, checked against code. Full-rank raw-gradient reconstruction does not imply rotation-invariant adaptive normalization. | Direct rotated/full-rank Adam counterexample differs from coordinatewise full Adam, while unnormalized reconstruction is exact. |
| Persistent versus peak storage | Shapes from the primary one-sided implementation, not a mixture with two-sided equations. State dtype inspected in an actual tiny CPU Adam coordinate proxy. | For 4096x4096/r128/all-FP32, moments 4 MiB + basis 2 MiB; weights and retained full gradient 128 MiB; accounted total 134 MiB versus 256 MiB. No peak allocator/GPU claim. |
| Per-layer update and published feasibility | GaLoRE §§4.3,5.3,5.5 and pinned README. Per-layer freeing is separated from projection; immediate microbatch updates and global clipping are qualified. | 64-A100 quality run distinguished from single-device memory feasibility; the README's checkpoint-enabled larger-batch command is not called checkpoint-free. No large model was trained. |

The original complete English README introduction is quoted without
translation and with author/pin/license attribution. New mathematical
explanation is editorial English, not a claimed verbatim course excerpt.
`no-ai-slop` was applied to that authored explanation; quoted source text
was not rewritten. The final edit removed an editorial aside about mixing
formulas and states the actual left/right tensor dimensions instead.
This checklist is not proof of AI authorship or universal pedagogical quality.

## Primary source and image inventory

Archive: `05 Источники/Originals/galore-2026-10-28/`.
Pin: `jiaweizzhao/GaLore`,
`2cc66f88cce189e505affbb91042a8e77f5bf4e9` (27 Sep 2024).
The root Apache 2.0 was read completely. The pinned recursive tree has no
separate NOTICE or per-image license; author credits are preserved in the
curated NOTICE and adjacent captions. Paper license is separately CC BY 4.0,
not silently substituted for the repository license.

| Published original PNG | Dimensions | SHA256 | Primary Git blob |
|---|---|---|---|
| galore_code_box.png | 3114x1298 | 3e390fe0e3ffd2e0b7fcf467666be5a6f6a15a584afec40e4555d7d4dc73a392 | 68fd83ed0bf9582dc162567067c572c58e251149 |
| subspace_learning.png | 855x794 | ed5a375d1dc12393467d8cc6fad9a6ef5399e857759318fc5c24f8aef0cce691 | 8494c00e5f955e38903588cae0ee38bc279e4fa8 |

Both Git blob hashes were independently recomputed and match the primary
GitHub tree response. Teaching PNGs are byte-identical to originals;
no crop, redraw, translated labels, changed geometry or recompression.
New `assets.json`/`NOTICE.md`, two entries in the public asset registry and
`site/public/licenses/galore-2cc66f88-Apache-2.0.txt` bind the same copies.
Existing generic image/source/license/decode validation was extended only
to this new directory, without disabling older asset assertions.

Other retained sources: complete README, LICENSE, projector and AdamW;
arXiv v2 HTML, original Figure 1 memory SVG and Adafactor v1 PDF. Figure 1
is reference-only, not published or marked visually accepted. No PDF pages
or Adafactor illustrations were transferred. Adafactor physical page 3
was separately rendered and inspected for the formula and Algorithm 2.
Earlier nonexistent Adafactor v4 lookup returned 404; the actual source
used is v1. No repository clone, model weights or large training dataset.

## Executed checks

`publishing/tools/check_optimizer_memory_examples.py` runs all three actual
new Python fences in source order. PyTorch 2.14.1 CPU. It additionally checks
16 row/column-history cases and 10 actual pinned-projector/Adam-state cases,
then reuse/refresh, adaptive rotation and independent byte counts. Direct
execution of the standalone upstream projector does not execute the full
upstream optimizer package. No timing, GPU peak memory, convergence or
training-quality measurement was performed.

`publishing/tools/check_optimization_examples.py` still executes the four
old fences and all previous independent assertions. It strips only the
bounded independent addition before its original four-block count.

Evidence for the final reader SHA:

- `/var/tmp/bookvar-optimizer-memory-examples-final.log`: 3 actual fences,
  26 extra marginal/projection/state cases and the additional checks PASS.
- `/var/tmp/bookvar-optimizer-memory-prior-final.log`: all previous numerical
  checks PASS on the current chapter.
- `/var/tmp/bookvar-optimizer-memory-unit.log`: 598 tests/89 files PASS,
  45.81 s; after the last two prose edits the 3 affected preservation/source
  tests reran PASS, 684 ms, `...-unit-final.log`.
- `/var/tmp/bookvar-optimizer-memory-build-final.log`: 1969 pages, 23.72 s.
- `/var/tmp/bookvar-optimizer-memory-output-final.log`: 157 tests, 4.11 s.
- `/var/tmp/bookvar-optimizer-memory-links-final.log`: 0 broken internal
  routes, fragments or files.
- `/var/tmp/bookvar-optimizer-memory-coverage-final.log`: structural coverage
  PASS. This is not full source/course or explanation completeness.
- Scoped `git diff --check` PASS; staged file list empty. All jobs terminal 0.

The first new unit-test run exposed a whitespace-sensitive sentence
assertion; it was corrected to normalize whitespace. That failed run is
not counted as acceptance. The accepted full suite and final scoped run
are the later logs above.

## Actual local UI

IAB tab 5, local port 4330. Desktop 1280x900 and mobile 360x800 inspected.
Ten chapter images load; seven Python blocks render; KaTeX errors and
duplicate IDs are zero. Mobile body is 360/360; all four tables are
328/328. The new display equations fit the column. Across all 19 displays,
one old equation reports a 1px scroll-width difference (329/328); the
page has no horizontal overflow. It was not silently rewritten.

Algorithm PNG displays at 632x263 desktop and 328x137 mobile. The mobile
inline text is small: accepted reading path is the image viewer, actually
tested at 2.5x with both extremes 0/516 and 516/516. Image region is
360px wide, height611/scrollHeight611; all source rows are available, not
cropped. The subspace figure displays at 632x587 and 328x305; its complete
desktop viewer and narrow inline view were inspected. Color is supported
by labels/arrows and the surrounding explanation.

EN route was actually opened: visible warning “This content is not
available in your language yet.”, new section present, both originals load.
This is fallback, not a finished English translation.

Screenshots in `docs/audit/screenshots/`:

- `2026-10-28-galore-algorithm-desktop.png` (refreshed final build);
- `2026-10-28-galore-subspace-desktop-viewer.png`;
- `2026-10-28-galore-mobile-math.png`;
- `2026-10-28-galore-mobile-viewer-left.png`;
- `2026-10-28-galore-mobile-viewer-right.png`;
- `2026-10-28-galore-subspace-mobile.png` (refreshed final build);
- `2026-10-28-galore-en-fallback.png` (final build).

## Harvard Training reconciliation remains open

Pinned QMD remains SHA
`9c6869cd6b115ab0a63ac020206f3537206535d1f465c0297c7cf93242e5b6eb`.
This pass advanced sequential reading from 2584 through 3848, including
prose and Python. Long TikZ coordinates 2907–2962,3363–3429,3470–3582,
3588–3694 were skipped; no claim of individually accepting those diagrams.
Earlier skipped 2051–2206 and previously selected 4656–5045 retain their
recorded disposition. Next sequential source boundary is 3849, the
mixed-precision figure; non-TikZ text resumes at 3902.

Selected GPU pp.140–185 and roofline pp.570–642 were compared, not fully
reaudited anew. They already distinguish warp/block/tile/wave and conditional
resident-block capacity. Training's universal batch-multiple-of-32 rule and
warp-as-wave argument were not imported. Other source findings in this
range are retained as findings, not facts: hard-coded loss curves used as
convergence evidence; universal linear LR scaling; synchronization payload
claimed to grow with batch; assigning all missing MFU to stalls; universal
30%/55–65% MFU thresholds; ResNet activation/gradient working-set constants
without a reproducible workload; peak memory determined solely by the
widest layer; preprocessing overlap asserted to eliminate all stalls.

The real new gap was the optimizer-memory mechanisms at 2850–2966.
Primary sources above supplied the explanation/illustrations instead of
copying the source chapter's unsupported rankings. H-V1-08 stays re-review
in the published transfer matrix. Remaining Training sections, native
figures, other Harvard/HSE/EDLS units, atlas and deferred EN parity remain
within the original active goal. Accepted RL/JAX/Frameworks were not edited.

## Later figure-selection addendum, 28 October

The historical acceptance above covered two PNGs; its reference-only
disposition for paper Figure 1 was subsequently revisited after inspecting
the native four-bar JPEG, generated redraw and primary six-bar SVG together.
The complete primary SVG and English caption/footnote are now integrated.
The original 1276-line revision is preserved by the bounded inverse check.
Current revision, license evidence, UI and numerical results are recorded in
`2026-10-28-training-reconciliation.md`; the older two-image counts above
remain evidence for that earlier revision, not the current image inventory.
