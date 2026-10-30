# tf.data cache: explanation and local rendered review

Status: `content-and-local-ui-reviewed`, 28 October 2026.

Chapter: `00 Учебник/10 ML Systems/06 Data pipeline, padding и packing.md`.
Current revision: 964 lines / 70,807 bytes, SHA-256
`5f2a58816bfc82f2ccb0e43cea2a4bdbae1d22d8aae17d0fb36f82a2baac2917`.
Route: `/textbook/ml-systems/data-pipeline-padding-packing/#cache-placement-and-epoch-state`.

## Explanation and source boundary

The entire previous 727-line S6 was read before editing. It explained stored
token IDs, worker prefetch, H2D lifetime, padding/packing, token-weighted loss,
sharding and iterator recovery. It did not work through cache placement,
epoch-dependent output, finalization or persistent cache filename reuse.

The addition uses the complete original English caching paragraphs from
TensorFlow's guide, with its complete original vector timeline. It is inserted
after loading and before H2D; it does not replace the PyTorch loader with a
TensorFlow API or create another short source-summary page. All five programs
are shown, followed by their intermediate state changes and limitations.

Read source selections:

- `tensorflow/docs` commit `e4210b940833a4e4e7c5d35a19ca46cafc27c7c5`,
  notebook cells 0–1 and 36–38 in full. Timing/plot definitions 52, 55, 57 and
  65 were also read; the whole notebook was not claimed reviewed or executed.
- TensorFlow 2.21.0, commit `a481b10260dfdf833a1b16007eead49c1d7febf3`:
  complete selected `repeat`, `shuffle` and `cache` methods/docstrings in
  `dataset_ops.py`, ranges recorded in the source archive README.
- The complete pinned SVG was inspected in the primary site and then in our
  actual page and viewer. The guide's deliberate sleeps are simulated work,
  not a measurement of our machine or GPU training.

| Mechanism | Source and worked evidence |
| --- | --- |
| Placement around deterministic `map` | Guide cells 36/38; identical `[0,1,4,9]` outputs, four versus eight runtime transformations over two passes |
| Epoch-dependent preprocessing | API `cache` reuse rule; controlled epoch 0→1 leaves cached values unchanged but produces `[100,101,102,103]` downstream |
| Shuffle placement | Complete `shuffle`/`cache` descriptions; seed 7's first order repeats before cache, while the downstream shuffle executes each iteration; no universal uniqueness promise |
| Finalization and repetition | Cache's full-input/StopIteration condition; incomplete `cache().take(2)` re-executes work, `take(2).cache()` reuses exactly two values; infinite cardinality is distinguished from finite cache completion |
| Persistent identity | Cache's documented filename behavior; three old records are returned for a changed five-record source under the same prefix, while a new prefix returns five |
| Capacity | Derived `uint8`/FP32 payloads 150,528,000 / 602,112,000 bytes, or 143.55 / 574.22 MiB; metadata, model and other buffers remain separate |

The 960-line draft was reread completely in four bounded reads. Subsequent
changes moved the figure's attribution immediately beside it and wrapped one
existing equation; those final passages were checked separately. The
no-ai-slop pass removed the generic guide-introduction sentence, retained
licensed wording verbatim, and kept the explanations and caveats around the
examples. No new mechanism is reduced to a definition alone.

## Preservation and provenance

Removing the one bounded cache addition, restoring the two review dates and
reversing only the display-equation wrap recovers every byte of the accepted
727-line chapter, SHA
`4ec7b2a6d0665e0c0c6a43337e508f4d7cb41c30009d1b850850c6559d1a25aa`.
All previous explanations, five images, code, links and numerical values remain.
Both the new Python checker and unit test assert this inverse, not a new
unrelated baseline. The old PyTorch checker strips only the independent TF
addition and keeps its existing assertions. Its CUDA loader remains parsed,
not executed on GPU.

The SVG SHA is `e2bb98742e0f946db1c53b7464c8aa114611055fc57b37a63a1b9f956904cdb7`;
its computed Git blob equals the primary GitHub contents response
`5b1aae592ab786846d5c19a5f179e57ac024c7d9`. Teaching and source copies are
byte-identical. No crop, redraw or translation was made. The pinned repository
license and notebook's own copyright/Apache notice are retained. Adjacent
`assets.json`/`NOTICE.md`, the central registry and public full-license copy
bind the source, figure and chapter. This does not clear unrelated course art.

## Execution and publication checks

- All five actual TF fences executed on TensorFlow 2.21.0 CPU, plus 33
  independent size/prefix/epoch cases. No GPU visible; no throughput benchmark.
  Final log: `/var/tmp/bookvar-tf-data-examples-accepted.log`.
- Previous collator, attention, packing, token/DDP normalization, sharding,
  pipeline timing and RGB-byte checks passed on PyTorch 2.14.1 CPU.
  Final log: `/var/tmp/bookvar-tf-data-prior-accepted.log`.
  Both final CPU logs print the current chapter SHA above.
- Full unit suite: 595 tests / 88 files, 42.94 s. After the final caption and
  equation formatting changes, all four affected cache/source/preservation
  tests passed again. Logs: `/var/tmp/bookvar-tf-data-unit-final.log` and
  `/var/tmp/bookvar-tf-data-unit-mobile-fix.log`.
- Current Astro build: 1969 pages, 23.71 s. Log:
  `/var/tmp/bookvar-tf-data-build-mobile-fix.log`.
- All 156 current output tests passed in 4.42 s. They check nine code blocks
  and six figures in root and explicit English fallback.
  Log: `/var/tmp/bookvar-tf-data-output-current.log`.
- Zero broken built routes/fragments/files; structural source coverage passes.
  Logs: `/var/tmp/bookvar-tf-data-{links,coverage}-current.log`.
- Scoped `git diff --check` passed. No staging, commit, push or deployment.

## Actual visual observations

The initial full-column SVG display reduced the row labels too far. The final
scoped CSS keeps the original at 1224×360 CSS pixels inside a horizontal
scroll window. Desktop's inner window is 630 px, with both 0/594 and 594/594
ends inspected. At 360 px, its inner window is 326 px, with 0/898 and 898/898
ends inspected. This is a scrollable complete figure, not a cropped source
file; the caption explains the controls. Other chapter images are unaffected.

The mobile viewer at 4× was inspected at both horizontal ends (0/1032 and
1032/1032). Its vertical client/scroll heights were both 611 px; all five rows
and the bottom time axis remain available. The full desktop viewer also shows
all epochs and the adjacent source/license caption. A narrow screen requires
panning; the full timeline is not claimed legible as one tiny thumbnail.

All six actual images load. At 360 px body widths are 360/360, all four tables
are 328/328 and all eight display equations are 328/328, with no KaTeX errors
or duplicate IDs. One pre-existing sum initially measured 333 px in a 328 px
column; its equality was wrapped without changing numbers. Both EN presence
and the visible untranslated-content warning were checked, not counted as a
completed translation.

Screenshots in `docs/audit/screenshots/`, prefix `2026-10-28-tf-data-`:
`desktop-left`, `desktop-right`, `desktop-viewer`, `mobile-left`, `mobile-right`,
`mobile-viewer-left`, `mobile-viewer-right`, `mobile-math`, `en-fallback` (PNG).

## Artifact inventory and remaining work

Existing files changed: S6 chapter; central asset registry; published Harvard
transfer-matrix H-V1-08 row; prior data checker; asset-registry and site-output
tests; scoped site CSS; task reconciliation anchor.

New authored files: this review, `check_tf_data_cache_examples.py`,
`tf-data-cache.test.ts`, the TF and Harvard Training source archive READMEs, and the figure's
`assets.json`/`NOTICE.md`. New source copies: notebook, API source, original SVG
and full license under `05 Источники/Originals/tf-data-2026-10-28/`, one teaching
SVG and one public license copy. New screenshots are listed above. No files
were moved. English body files and accepted RL/JAX/placement material were
not changed.

Harvard Training remains in progress. Its pinned raw QMD is now retained at
`05 Источники/Originals/training-2026-10-28/chapter-2bd97c5.qmd`, SHA
`9c6869cd6b115ab0a63ac020206f3537206535d1f465c0297c7cf93242e5b6eb`.
Read prose and Python/pseudocode through line 2583; long TikZ 2051–2206 was
not read and source figures have not all received individual visual review.
Earlier selected 4656–5045 remains read. Hard-coded activation timings,
utilization/convergence rankings, incompatible throughput units, unconditional
saved-activation claims and unsupported cost savings are source findings,
not accepted facts to import. This cache addition does not close that chapter,
its slide/lab units, the other courses, atlas or deferred English parity.
Next source cursor is 2584, Forward pass. Do not reopen completed Frameworks
or RL merely to produce another status check.
