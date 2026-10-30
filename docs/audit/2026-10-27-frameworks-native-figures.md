# Frameworks: source figures and tensor storage

## Inventory and inspection boundary

The imported chapter references 21 distinct objects. Its local asset directory
contains those same 21 files. All complete images and their original QMD
captions/alt text were inspected, including the cover and marginal graphics.
Generated TikZ figures were rendered from their archived SVG bytes for inspection;
these previews do not replace the originals.

The pinned repository's native `frameworks/images/` directory is a different
inventory. A non-truncated GitHub tree at commit
`2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd` contains **46 paths**, representing
43 distinct byte objects. Three logo spelling aliases have identical bytes.
Eight native objects also occur in the published inventory. The union contains
56 distinct objects, not 21, 46 or 67. Every distinct image was inspected; exact
byte aliases reuse that inspection rather than being counted as new content.

The native downloads occupy 12,044,731 bytes in
`/var/tmp/bookvar-frameworks-directory-2026-10-27`, outside publication. Each
download's Git blob SHA-1 matched the pinned tree. The checked path/size/blob/
SHA-256 inventory is retained in
`05 Источники/Originals/frameworks-2026-10-27/native-directory.json`.
No PDF, model weights or dependencies were downloaded. No personal files were
removed. SVG previews with a white background are inspection renders only.

Exact per-object decisions, source lines, original captions and hashes are in
[the disposition JSON](2026-10-27-frameworks-native-figures.json). The source
page and QMD remain unchanged:

| Artifact | SHA-256 |
| --- | --- |
| Imported `vol1/frameworks.md` | `c348b78528c0f0af99a477e0bd07651b03a445ebbda51b0adf046e97777c6e14` |
| Original `chapter-2bd97c5.qmd` | `2cdf087ebca6106368840273f9ef6c447aba277de197223bc3c24a7022010059` |

The import manifest's rendered-HTML snapshot hash is not the hash of the
converted Markdown. This report binds the actual Markdown bytes separately.
The generated mediabag SVGs are not claimed to be native Git image blobs.

## Selection and specific findings

The complete Tensor Memory Layout SVG, published object 13, is integrated
in the first tensor chapter beside transpose and reshape. It identifies six
values, both physical orders and the corresponding element strides. Its
original labels are retained. The source caption's byte-address qualification
is corrected in adjacent prose, not by redrawing the image.

Other published objects remain in the source archive rather than being
silently approved or deleted. Native files not referenced by the current
chapter remain pinned by URL/blob/hash in the directory inventory; their
temporary inspection copies are not new reader-facing assets:

- The eager flow says an operation is executed before the next is defined.
  GPU queue submission and completion must remain separate. S8 already has
  a worked launch-queue model, an original trace and an explicit overlap example.
- The Python-tax chart assigns five 10+1 μs pairs versus 5+4 μs. It is an
  illustrative serial timeline, not a measured 55/9 speedup. The continuum
  similarly assigns three linear cost functions; its log axis and crossovers
  do not establish universal eager/JIT/AOT rankings.
- The dispatch sparkline has no axes, units or legend. The bandwidth ladder
  omits compatible link/direction/aggregation conventions. Neither becomes
  performance evidence because it appears in a course.
- The 102 MB versus 10–15 GB ladder compares weights-only sizing with an
  assumed training footprint. It is not a symmetric runtime-memory comparison.
- Three colored DataLoader or framework-name dots provide labels, not the
  queue/transfer/implementation mechanisms described in the surrounding text.
- The TensorFlow/ONNX pipelines and native feature tables omit versions and
  conversion contracts. The smaller native ONNX SVG is legible but its
  “any framework … any runtime” slogan is not an interoperability proof.
- Covers, wordmarks, a company-logo market map and undated search-interest
  screenshots do not replace worked mechanism figures. The training/inference
  cover with illegible pseudo-labels is explicitly excluded.

The 3D GPU-placement figure was subsequently integrated on 28 October after
comparison with the full distributed overview; see
`docs/audit/2026-10-28-frameworks-placement.md`. Its complete vector and English
caption accompany 32 logical coordinates and explicit TP/DP/PP group/batch
calculations, not a new GPU benchmark. The two formerly held, unreferenced
federated-learning and transfer-learning rasters have a reference-only decision
after the destination comparison recorded below; neither receives a teaching copy.
The hospital padlocks do not establish privacy. The RGB-channel figure's
Niklas Lang credit is not overridden by a general repository license.
Individual disposition does not grant blanket acceptance or third-party rights.

## Tensor explanation review

All 749 pre-change lines were read sequentially, including the six retained
figures, numerical examples and complete NumPy fence. The accepted chapter
already distinguishes transpose from reshape but does not derive storage
addresses. The source's complete tensor/layout subsection, QMD 2793–2960,
and all of its three drawings were compared. The memory-layout TikZ and its
published SVG agree on the six values and both linear orderings.

The addition stays in English; it does not translate the existing course
excerpt or any locale body. It reproduces the source caption's first sentence
and adds an explicit worked address calculation:

- At `[1,1]`, C-order element offset 4 and F-order offset 3 both find 5.
  For float32, these are byte offsets 16 and 12 in separate buffers.
- NumPy reports byte strides `(12,4)` and `(4,8)`. The figure's element
  strides `(3,1)` and `(1,2)` are not passed off as NumPy output.
- Transposition changes the view to shape `[3,2]`, element strides `(1,3)`,
  without reordering the original buffer. A column-major copy keeps shape
  `[2,3]` and stores a different order; these are separate transformations.
- The slice `A[:, ::2]` has four float32 elements/16 logical bytes, strides
  `(12,8)` and an address span of 24 bytes. It retains shared storage; the
  printed span formula is explicitly limited to this positive-stride case.
- Writing 99 through the slice changes the original and transpose view,
  while an explicit slice copy and the F-order copy remain unchanged.

NumPy 2.3's complete selected documentation sections were checked live:
[`ndarray.strides`](https://numpy.org/doc/2.3/reference/generated/numpy.ndarray.strides.html)
and [Copies and views](https://numpy.org/doc/2.3/user/basics.copies.html).
The chapter does not adopt the Harvard text's unsupported bandwidth-utilization
percentages, universal row-major speedup, or a claim that NumPy lacks metadata.

Current chapter: 878 lines, SHA-256
`7764790a11f7cbf12fbb9e662d62d56bd94ad8d39863a8995423f29161ff6ebb`.
Removing only the 129-line bounded addition restores every byte of the accepted
749-line chapter, SHA
`f8b39f3482e80bc07ff4cafcdddd287f7fe89cb31d00c1f9d0db0d125dd6a536`.
The older equation-wrap inverse and its previous accepted baseline are retained.
All prior mathematical explanations, six figures and the old code are preserved.
The NN section map binds this new revision without changing its 37 source ranges
or the other 16 destinations.

The complete original SVG is copied byte-for-byte, SHA-256
`f0d88143e5561c8061bfba1f5ebc9f79e7c9d712e4e79fd195ce06a8e1a4f036`.
Attribution, source revision, original QMD/TikZ, CC BY-NC-SA 4.0 license evidence
and the distinction between generated and native assets are recorded in the
adjacent `NOTICE.md`, `assets.json` and central asset registry. No source art
was cropped, redrawn, relabeled or translated.

## Execution and rendered acceptance

The checker executed both actual chapter Python fences on the existing NumPy
2.3.5 runtime. In addition to all previous contraction/broadcast/head-permutation
cases, 2520 scalar addresses were checked against independent buffer bytes
across four dtypes, C/F order and 30 matrix shapes. Transpose and sliced views
were checked for value correspondence and shared storage. No GPU work or
runtime-speed claim follows from these CPU checks.

The final 878-line chapter was reread in full, not searched for selected words.
The new English prose keeps the numerical progression and source language;
the no-ai-slop review did not remove terminology, mathematical qualifications,
or the licensed caption. The meaningful distinctions between element/byte
units and transpose/copy are preserved. There is no generic introduction,
universal performance conclusion or decorative replacement art.

Observed checks so far, all terminal exit 0 after the registry repair:

- Both actual NumPy fences and all 2520 additional address cases on the
  printed current chapter SHA. No dependency installation.
- 77 unit files / 555 tests, 41.12 s. Log:
  `/var/tmp/bookvar-frameworks-native-unit-final.log`.
- 146 built-output tests, 4.46 s. They include the new figure/code/attribution
  in root and explicit EN fallback, and all 21 archived Frameworks images
  in root and EN source routes. Log:
  `/var/tmp/bookvar-frameworks-native-output-final.log`.
- Built links: zero broken internal routes/fragments/files. Log:
  `/var/tmp/bookvar-frameworks-native-links.log`.
- Structural source coverage: 210 rows / 58 existing destinations. Log:
  `/var/tmp/bookvar-frameworks-native-coverage-final.log`. This checks paths
  and source links, not whether all knowledge has been transferred.

The first unit run exposed that the new source SVG fell through the asset
validator to its default PDF-only branch. Its directory was added to the
existing unchanged-source-image validation branch. Source/output/attribution
hash checks, full-byte equality and the no-crop check remain; no check was
removed or converted to a blanket pass. The entire 555-test suite then passed.

After the published transfer-matrix changes, the final build again produced
1969 pages, in 22.28 s, terminal exit 0. Log:
`/var/tmp/bookvar-frameworks-native-build-final.log`. On that final build,
146 output tests passed in 4.22 s, links again reported zero broken targets,
and the actual NumPy fences/address checker again printed the final chapter SHA.
Release logs: `/var/tmp/bookvar-frameworks-native-{output,links,examples}-release.log`.
The final root page was reloaded, visually inspected and saved after this build.
Scoped `git diff --check` passed; no staging or publication was performed.

## Actual browser observations

Root route:
`http://127.0.0.1:4330/textbook/ml-foundations/tensors/#shape-strides-and-shared-storage`.

- At 1280×900, body client/scroll widths are 1265/1265. The complete SVG
  displays at 568×176, with both arrays and the offset formula visible.
- At 360×800, body widths are 345/345 and the complete SVG is 313×97.
  Fine labels are too small for comfortable inline reading at that width;
  the viewer at 2× was used to inspect the full left matrix/stride formula
  and both right-hand storage orders after panning to each edge. Labels
  and element indices remain sharp. No source crop was substituted.
- All 24 display equations fit to within one pixel of browser rounding;
  inline math overflow and KaTeX errors are zero. Both new address formulas
  were also read in the actual mobile screenshot.
- All seven chapter images with `src` load. The empty viewer placeholder is
  not an eighth illustration or a failed source image.
- The actual EN route explicitly says “This content is not available in your
  language yet.” It contains the addition and complete SVG. This is a checked
  fallback, not a completed translation.
- The actual course-source route renders and loads all 21 original images
  at mobile width. Loading is not a claim that every source caption is correct.

Screenshots: `screenshots/2026-10-27-tensor-storage-desktop.png`,
`screenshots/2026-10-27-tensor-storage-mobile.png`, and
`screenshots/2026-10-27-tensor-storage-mobile-viewer.png`.

The full Harvard/HSE/EDLS integration, remaining AD/runtime/training-step work,
held candidates and deferred EN parity remain open. Accepted RL chapters,
EN bodies, git staging/commit/push/deploy and live infrastructure are untouched.

## FL and transfer disposition, 28 October

The entire current Edge/FL chapter (1118 lines, SHA-256
`8f10fda46877de9a593fb29200c2511d6727e4431f75f1e7b5c5283e7010cd8d`)
and BERT chapter (893 lines, SHA-256
`fa3595eca44cec1269db1425337a04773d132d2dd685432d1b12518a15ae19b4`)
were read sequentially in this pass. Their bytes are unchanged. Both native
rasters were inspected again at original resolution; their primary pages were
also reopened. The origin/hash evidence from the placement report is retained,
not replaced by another search or download.

The NVIDIA hospital/server drawing is a cross-silo overview. Edge already
explains local versus shared models, the round's base version, returned weights
or deltas, weighted aggregation and participation. It works through 100/300
example weights giving `(4,6)`, the single accepted-client case, and local drift
`2.865` versus two centralized steps `2.92125`. Original FedAvg, scheduling and
secure-aggregation figures accompany the relevant mechanisms. The candidate
adds no missing step; its padlocks also do not specify a threat model. Keep
[Nicola Rieke's NVIDIA original](https://blogs.nvidia.com/blog/what-is-federated-learning/)
as a source reference, without a new teaching copy of its drawing.

The 560×370 TechTarget drawing gives a dog/cat task-transfer overview, without
trainable/frozen layer ownership or a target loss. BERT already has complete
original Figures 1 and 4 beside checkpoint transfer, replacement task heads
and full fine-tuning, and Alammar's original feature-extraction comparison
beside the 96.1/94.9/96.4 NER results. Edge derives the memory implications of
freezing and explains why few trainable parameters need not mean few saved
activations. The tiny candidate would duplicate a less explicit explanation.
Keep the [Gillis/Cole TechTarget original](https://www.techtarget.com/it-strategy/definition/What-is-transfer-learning)
as a reference, without a new teaching copy.

Both JSON decisions are `reference-only-not-selected`. Source URLs, blobs,
sizes and hashes remain intact. Redistribution permission for these two
third-party objects is **unverified**; selecting neither is not a rights
clearance. The source archive and both textbook chapters remain unchanged.
This resolves the last two named Frameworks comparisons, not the Harvard
labs/decks, all course units, deferred English parity or the whole textbook.

Final disposition checks on 28 October: the complete 591-test unit suite
passed in 42.14 s; after the final metadata wording changes all 11 affected
figure/map/placement tests passed again. The current site build produced
1969 pages in 22.69 s, and all 155 output tests passed in 4.30 s. Built links
have zero broken targets; structural source coverage also passes. Current
logs are `/var/tmp/bookvar-frameworks-disposition-{build,output,links,coverage}-current.log`;
the complete unit log is `/var/tmp/bookvar-frameworks-disposition-unit-final.log`.
Scoped `git diff --check` passes. No source artwork or textbook body changed.

Actual matrix inspection found that the old status cell made the Frameworks
row 1697 px high. Its detailed history now remains in the reports; a compact
summary and report path reduce the row to 381 px at desktop width and 325 px
at 360 px. Body widths are 1280/1280 and 360/360 respectively. The wide matrix
still needs horizontal scrolling; it is not claimed to be a mobile card view.
The EN route shows the same current status and explicit fallback warning.
Screenshots are `screenshots/2026-10-28-frameworks-disposition-{desktop,mobile,en-fallback}.png`.
