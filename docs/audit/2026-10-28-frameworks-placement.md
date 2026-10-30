# Harvard Device Placement Dimensions in the distributed chapter

Status: content-and-local-ui-reviewed.

Current destination: `00 Учебник/11 Pre-training и Scaling/44 Distributed training и mixed precision.md`,
1019 lines, SHA `732257fa15ab9d21f4c28ec94b07c0d9f945a6a975de38960a66cd9afcaa86ac`.
The whole prior 917-line chapter was read, then the complete expanded chapter
in ranges 1–285, 286–570, 571–820 and 821–EOF; the final one-paragraph
style correction was checked separately. Removing one bounded addition and
restoring the two dates recovers every prior byte, SHA
`fe1a7b12bc09cabd25750a0f14ca805217c05f72359549d3cbd611e3906a02a9`.
No earlier explanation, code, nine figures or manual anchor is replaced.

## Source and explanation comparison

Harvard ML Frameworks, revision `2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd`,
Distributed execution contexts and complete original TikZ/caption around
QMD lines 3329–3447 were compared with the actual SVG and the current DP/TP/
PP/EP composition section. Existing individual algorithms are already worked
through; what was missing there was a single spatial picture of their groups.

The original full SVG `2989ddaa…ff9b.svg` and entire English caption are
retained, SHA `6c3e1e350e5653c517ac8b82ec4a7184eceb7573ae8320b56c4ca222861c6b12`,
viewBox `0 0 274.96933 168.46268`. No crop, redraw, relabeling, translated
labels or raster replacement. The inspected PNG is only a preview; publication
uses the exact vector original. CC BY-NC-SA 4.0 is supported by the pinned
Harvard root license; adjacent NOTICE/assets and the global registry agree.
The published/generated SVG is not misidentified as a native Git image blob.

The explanation derives eight stacks times four GPUs = 32, rather than
mistaking eight stacks for eight devices. Logical coordinates use
`rank(p,r,t)=8p+4r+t`, consistent with all eight visible labels. The source
does not label the other 24 boxes individually; this completion is explicitly
one worked numbering, not a claim about physical IDs or backend rank order.
For GPU 0, TP is `[0,1,2,3]`, DP is `[0,4]`, PP is `[0,8,16,24]`.
The worked GPU 23 case gives TP `[20,21,22,23]`, DP `[19,23]` and
PP `[7,15,23,31]`. Fixed versus varying coordinates explain each group.

Corresponding DP parameter partitions combine gradients; TP combines
operation-specific partials; adjacent PP stages exchange activations and
their derivatives. Neither a generic all-reduce nor averaging different
layer weights follows from a device drawing. A microbatch of three with five
accumulations and two data replicas gives 30 sequences, not 480 from using
all 32 devices as independent data workers. The bridge to the original
64-device MoE case preserves its separate dense/expert replica convention.
The picture is not evidence of a physical network or a pipeline schedule.

## Execution, provenance and style

`publishing/tools/check_placement_dimensions_examples.py` executes the exact
visible Python fence. It independently verifies inverse coordinates and
TP/DP/PP membership for all 32 devices, all pairwise group intersections,
complete partitions by all three group families, eight visible labels and
the 30/480 batch calculation. SVG bytes/viewBox/labels are checked. No
process group, communication operation, GPU or timing is executed by this
example. The pre-existing overview checker still executes its original TP
program and mocked AMP ordering, with all prior arithmetic/source checks.
Its only change excludes the independent addition from old fixture counts.

No-ai-slop review retained necessary ownership contrasts, equations and the
licensed caption. Added prose describes coordinates, actual member lists,
what is exchanged and the batch consequence; no generic introduction or
performance conclusion is substituted for the existing mechanisms.

## Remaining candidates and validation

The two unreferenced FL/transfer-learning rasters were inspected again.
The hospital image does not specify a round's payload or establish privacy;
the small task-transfer image is nearly illegible and does not show frozen/
trainable ownership. They are not newly published here, and their source/
rights/destination decisions remain open rather than being hidden by the
accepted 3D figure. Edge, adaptation, RL and JAX chapter bytes are unchanged.

Their third-party origins were identified from primary publisher pages:

- TechTarget, Alexander S. Gillis and Ben Cole,
  https://www.techtarget.com/it-strategy/definition/What-is-transfer-learning
  links `https://www.techtarget.com/rms/onlineimages/how_transfer_learning_works-f_mobile.png`.
  Downloading that image produces exactly the existing 14430-byte archive
  SHA `709e3fc9a981d9d0912a66ebfc1740422240276f97ba38e59c96954a5d9d2625`.
  This is publisher artwork, not newly authored Harvard material. Its small
  labels and the missing parameter/update ownership remain selection issues.
- NVIDIA, Nicola Rieke,
  https://blogs.nvidia.com/blog/what-is-federated-learning/ (2019, updated 2024),
  links the healthcare diagram. The full original
  `https://blogs.nvidia.com/wp-content/uploads/2019/10/federated_learning_animation_still_white.png`
  is 2000×1118, SHA `b8d3885fcb9d4a9bedad14cf94dd1bef75c806db3792feb8ee65a69d3b35aff1`.
  Its compressed bytes differ from the archive's `af5dff11…98b4b`, but the
  decoded RGBA pixels are identical (Pillow comparison, same dimensions).
  The diagram's locks do not establish a privacy guarantee.

Only temporary inspection copies were downloaded; neither raster was newly
published. These origins do not themselves establish redistribution permission.
The remaining rights/destination decisions are recorded, not silently closed.

## Final execution and rendered evidence

The first actual desktop inspection exposed undersized display: the SVG's
print dimensions became an approximately 275-pixel image. A selector scoped
only to this figure now uses `width: min(100%, 40rem)` and `height: auto`.
This changes presentation, not SVG bytes, labels or viewBox; metadata and
NOTICE distinguish display scaling from source modification. After a rebuild:

- Desktop 1280×900: complete source figure 632×387.20; all three brace labels
  and eight GPU labels inspected at ordinary page scale.
- Mobile 360×800: image 328×201; body 360/360. Small inline labels benefit
  from the existing viewer. At 2×, both edges were actually visited,
  scrollLeft 0/344 and 344/344; no vertical clipping or lost labels.
- New table and coordinate formula 328/328; no display-math overflow,
  KaTeX errors or duplicate IDs. All ten published source images load.
- EN fallback warning visibly inspected; the new source section and 632-pixel
  figure are present on that route. No translated body was created.
- Actual source-matrix page shows the 28 October note and updated H-V1-07 row
  with two held rasters, rather than implying complete course transfer.

Screenshots: `docs/audit/screenshots/2026-10-28-frameworks-placement-` with
`desktop`, `mobile`, `viewer-left`, `viewer-right`, `table-mobile`,
`math-mobile` and `en-fallback` suffixes. The initial tiny desktop image was
replaced by the screenshot of the corrected display.

Final checks, all terminal exit 0:

- Exact placement fence, all 32 memberships/inverses/intersections, all three
  group-family partitions, batch negative control, SVG and inverse baseline.
  Log `/var/tmp/bookvar-placement-cpu-final.log`.
- Entire old overview arithmetic/source/TP/AMP check on the expanded chapter;
  log `/var/tmp/bookvar-placement-overview-final.log`. These are CPU/model
  checks, not actual distributed execution.
- 590 unit tests / 87 files, 45.58 s:
  `/var/tmp/bookvar-placement-unit-final2.log`.
- Final actual build: 1969 pages, 24.72 s:
  `/var/tmp/bookvar-placement-build-final2.log`.
- 155 output tests, 4.30 s:
  `/var/tmp/bookvar-placement-output-final2.log`.
- Built links: 0 broken routes/fragments/files:
  `/var/tmp/bookvar-placement-links-final2.log`.
- Structural source coverage PASS and scoped whitespace checks PASS.
  Structural checks are not all-course semantic acceptance.

This closes the placement-figure comparison, not H-V1-07, the full course,
EN parity or the whole textbook goal. No staging, commit, push, deployment,
source replacement, delegation or live-infrastructure mutation.
