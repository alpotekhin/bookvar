# Harvard Training: source comparison and original figures

28 October 2026. Status: `source-chapter-thematic-review-complete-with-findings`.
Local checkout only; no staging, commit, push or deployment.

The preceding user-facing turn was a status verification of already accepted
RL chapters, not new content progress. PPO/DPO/RLVR/GRPO hashes still match
their 24–25 October acceptances. This pass does not edit them.

## Source and destination comparison

The pinned QMD is unchanged: 6801 lines, 562666 bytes, SHA256
`9c6869cd6b115ab0a63ac020206f3537206535d1f465c0297c7cf93242e5b6eb`,
commit `2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd`.
The cumulative reading boundary now reaches EOF for prose, Python and
pseudocode. Long TikZ coordinate ranges not read as code are listed in
`2026-10-28-training-section-map.json`; rendered diagram inspection is
recorded separately, not represented as reading every coordinate.

The 32 actual H2/H3 intervals partition lines 1–6801. They map mechanisms
to 19 current destination revisions, with source findings and explicit
comparison decisions. A matching revision permits reuse of the corresponding
recorded acceptance; it does not prove an unrelated source mechanism.
No claim is made that all 19 chapters were reread in full in this pass.

S5, FlashAttention and 44b were read completely across this and the preceding
Training pass. The truncated checkpoint paragraph at 545–558 was reread
separately. S5's numerical examples cover subnormals, rounding, master
weights, weighted accumulation, FP8 scales and MXFP8 blocks. FlashAttention
already gives the online maximum update, backward reconstruction, causal
tails, gradient ownership and FA2/FA3 execution. 44b already explains
segment storage, selective checkpointing, memory-budget selection and
offload lifetimes. They are not rewritten to reproduce a weaker overview.

Current CPU evidence, unchanged destination hashes:

- `/var/tmp/bookvar-training-precision-check.log`: actual NumPy example,
  BF16/FP16 rounding/sums, master weights, loss scaling, weighted gradients,
  all 127 positive E4M3FN codes and 126 midpoints, FP8/MX cases PASS.
  SHA `9a72e3c171518af749c26271e183bf35c55d2f98f76a146cf93c8d8404efa02d`.
- `/var/tmp/bookvar-training-flash-check.log`: actual CPU examples,
  independent block/mask/tail cases, forward/backward and gradchecks PASS.
  SHA `98b6a0fddd6b6d7b7136bf1b3a9eacc32d6334f182f59bd4218b99e1c3652add`.
- `/var/tmp/bookvar-training-checkpoint-check.log`: printed examples,
  independent segment/policy/lifetime cases and gradients PASS.
  SHA `3fe01a82c9a8f60a417d7e3eacce59dac608cd767944af5d85db5ab549be1f5e`.

CUDA AMP, FlashAttention kernels, GPU peak memory, offload performance and
large-model training were not executed. These are not benchmark results.

## New reader figure

The existing optimizer chapter now includes the unchanged primary GaLoRE
Figure 1, not Harvard's four-bar redraw or its older JPEG. The primary has
six bars, including both retained-gradient and early-update controls.
The complete original English caption and footnote state the 256-token
single-device batch and absence of activation checkpointing/offload.
The adjacent explanation distinguishes projection, 8-bit state and early
gradient freeing, and separates paper estimates from the chapter's explicit
FP32 matrix ledger. No translation or EN body was authored.

Reader route:
`/textbook/neural-networks/optimizers-schedules/#galore-memory-breakdown`.
Source: `https://arxiv.org/html/2403.03507v2#S1.F1`.
SVG SHA256:
`3c395fe582f012f4f47780105693631a864aadf309a79b6a4ea6ed866d7f974f`.
Source and published copies are byte-identical. The actual arXiv v2 license
declaration is archived as `arxiv-abs-v2.html`, SHA
`17ea79ba6a8857197fa02d4753cb22bf34e1497c95704e566467bb8f0c1d176a`.
CC BY 4.0 is distinct from the two repository PNGs' Apache 2.0 license.
Attribution, original URL, license link and unchanged status are adjacent
to the figure and in its manifest/NOTICE/central registry.

The full accepted 1276-line optimizer chapter was read before this addition.
The 33 added lines were then read in their final context. Removing only the
bounded `galore memory comparison` addition restores every byte of that
accepted revision, SHA
`97370cdc6665f959eb59109cbe68cf2014e5247bfa1f21b2e68e9512b7a8c903`.
Current chapter: 1309 lines, SHA
`b9a3d638815bb981cedc989003a9ac0cc6ec3314a849c4bf2c57c0940a5ffa84`.
No numerical example was replaced. This is acceptance of the figure
addition, not a new claim that every whole-book requirement is complete.

## Native directory is not the rendered-media list

The existing web import has 27 image paths. The pinned Git directory has
33 paths / 32 distinct byte objects; the two sets have 50 distinct objects
in their union. They were not interchangeable inventories. All 33 pinned
paths were retrieved (10,087,278 bytes total), independently checked against
Git blob SHA1 and byte count, and bound to SHA256 in
`05 Источники/Originals/training-2026-10-28/native-directory.json`.
The 27 existing objects still match the original import manifest.

Every object has a disposition in `2026-10-28-training-figures.json`.
Contact sheets were visually inspected; the full parallelism-choice PNG
was also inspected separately. Semantic examples are distinguished from
cover art, vendor marketing, unsupported timing bars and duplicate formats.
Native third-party images do not inherit verified redistribution clearance
merely from the Harvard repository license. No Harvard image is newly
published by this pass. The selected GaLoRE primary figure has its own
verified license and reader QA.

Notable source findings: the choice flowchart requires dataset residence
and can select DP after a model fails one-device fit; the low-utilization
flowchart uniquely diagnoses a data bottleneck; the four-bar GaLoRE JPEG
omits controls present in the primary. Those objects are retained in the
source layer, not taught as reliable decision procedures.

Screenshots: `2026-10-28-training-pinned-{01-06,07-12,13-18,19-24,25-30,31-33}.png`
and `2026-10-28-training-native-{01-03,04-09,10-15,16-21,22-27}.png`.
These are audit contact sheets, not newly drawn textbook illustrations or
proof that all small labels would be readable inline.

## Local checks and remaining work

The first figure-addition build passed: 1969 pages / 26.65 s; 599 unit tests
in 89 files / 46.44 s; 157 output tests / 4.31 s; built links: zero broken.
Logs: `/var/tmp/bookvar-training-reconciliation-{unit,build,output,links}.log`.
Those counts predate the later section-map/figure-inventory tests and final
matrix wording; final acceptance is recorded separately below.

`/var/tmp/bookvar-training-galore-examples.log` reruns the three actual
memory fences, 26 independent cases and the pinned projector on the current
chapter. `...-galore-prior.log` reruns all four prior fences and assertions.
Both passed on CPU, with no timing or peak-GPU-memory claim.

Actual reader UI: the six-bar diagram and full caption were inspected
desktop and at 360 px. Mobile body is 360/360, no KaTeX errors. All eleven
inline images load; the hidden viewer copy is not counted as a twelfth
chapter figure. The new SVG displays at 328x170, so small inline labels
are read through the viewer. At 2.5x, both horizontal extremes were checked
at 0/516 and 516/516; image height/scrollHeight is 611/611. The long caption
scrolls separately to 147/147, exposing the source/license at its end.
EN was opened and shows the explicit unavailable-language fallback with
the same new figure; this is not an English translation.

Proof screenshots: `2026-10-28-galore-memory-{desktop,desktop-viewer,mobile,mobile-viewer-left,mobile-viewer-right}.png`.

## Energy comparison resolved without duplicating a chapter

The initial search in MLOps found no CO2e derivation there. A subsequent
whole-textbook search found the already accepted Responsible systems chapter.
All 1136 current lines were then read, not only the matching paragraphs.
Its SHA matches the 20 October acceptance:
`5e70d3ce66b8874cfc6b32928762cdea7d71b9f92b000aa6355d7cacf248cc5c`.
Its text is unchanged.

The energy section already explains sampled-power integration, counters
versus snapshots, full versus incremental consumption, facility boundaries,
one-time PUE, time-varying grid intensity, lifecycle emissions, rebound and
hardware payback. Worked cases include 6000 J versus 2000 J incremental
energy, 480 to 408 kg CO2e after shifting the same work, a 30-day
training/inference crossover and 3.75-year hardware payback. These cover
Training's energy mechanism more precisely than its constant-TDP scenario.
MLOps remains the separate destination for billing, interruptions and cost.

`/var/tmp/bookvar-training-responsible-check.log` passed on the current
revision: four actual Python blocks, 936 checks including 107 energy/carbon/
lifecycle cases, source identities and existing built anchors. No new
power measurement or benchmark. The first interpreter lacked BeautifulSoup;
the existing Homebrew Python with bs4 4.14.3 ran the unchanged checker.
No dependency installation or privileged telemetry access was needed.

The two existing original energy figures were inspected in the actual
desktop viewer. Their lifecycle shares and prefill/decode ratios are
schematic/source scenarios, not universal measurements; that qualification
is present in the surrounding explanation. S5's six existing original
figures were also inspected in the actual desktop viewer; the master-weight
proof is saved as `2026-10-28-training-precision-master.png`. No new
whole-Sustainable-AI, all-Responsible-systems-source or fresh mobile
acceptance is inferred from this selected comparison.

The provisional `energy-emissions` item is now in `resolved_items`, not
`open_items`. No new duplicate chapter or synthetic summary was added.
Thematic reconciliation is complete for this pinned Training chapter only.
H-V1-09 Data Selection is the next source chapter to compare; other Harvard/
HSE/EDLS units, atlas and deferred EN parity remain in the active whole-book
goal. RL is not reopened.

## Final acceptance evidence

After the final source-map resolution, the six affected unit tests passed
in 1.00 s. The preceding full suite passed 601 tests in 90 files / 45.56 s;
its remaining inputs were not changed by the resolution. The first new
inventory test incorrectly required every decision to exceed 60 characters;
that editorial-length proxy was replaced by nonempty decisions and explicit
disposition categories. It is not counted as an accepted run.

Final Astro build: 1969 pages / 23.91 s. Output tests: 157 / 4.26 s.
Built links: zero broken. Structural coverage passed; its counts are not
evidence that the whole courses or book are explained. Final logs:
`/var/tmp/bookvar-training-reconciliation-{scoped-final,build-current,output-current,links-current,coverage-current}.log`.
All processes exited 0. Scoped diff-check passed; staged list was empty.

The actual source matrix rendered eight cells for H-V1-08 and its new
Responsible systems link was clicked in the browser. It lands at the
energy heading and its worked 6000-J trace, saved as
`2026-10-28-training-energy-existing-destination.png`. The intermediate
18-target/provisional-gap build is not the final matrix revision.

No reader material was compressed into definitions. The original source
caption is unchanged in language; authored figure explanation was checked
with no-ai-slop and the local mathematical/academic qualifications.
No general AI-detection or universal-quality claim follows from that check.

The temporary image-gallery server and owned inspection tab are closed
after inspection; the main preview remains on the optimizer chapter.
