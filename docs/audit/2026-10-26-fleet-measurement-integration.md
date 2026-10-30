# Fleet measurement — acceptance, 2026-10-26

Canonical chapter:
`00 Учебник/10 ML Systems/03 Измерение производительности и roofline.md`.
Final SHA256:
`ee3b591a0085982c18c2ae691889e38f2f624b505cb2a414520c947c1f57323b`.
Route: `/textbook/ml-systems/performance-measurement-roofline/`.
New anchor: `#fleet-mfu-scaling-and-preserved-progress`.

The previous 755-line S3 and the full 832-line S7 were read before choosing
the addition. The final S3 explanation was reread, including the complete
new section. The earlier timer, roofline, cache, statistics, GEMM, profiling
and energy explanations were preserved. A fidelity test removes only the
bounded addition and date/language metadata, then reconstructs the previous
accepted S3 SHA256 `1ae6298b758d8d9ec1b970930abc9ac3750f15a77c2c32d309351a4889f2154a`.
S7, S8 and accepted RL chapters were not edited.

## Explanation and sources

Exact source pins, hashes and read scopes are in
`05 Источники/Originals/fleet-measurement-2026-10-26/references.json`
and its README. Seven source blocks retain the original language: the
two-paragraph Ultra-Scale explanation, three NVIDIA paragraphs, and three
Google entries. Selection boundaries are tested against archived text.

| Mechanism | Explanation and worked check |
|---|---|
| MFU versus HFU | Complete Ultra-Scale distinction; declared model/executed work, global-batch accounting and dense 6PD limits; 600/800 over 1000 gives 60%/80%. |
| Strong versus weak scaling | Fixed versus growing work; efficiency derivation; NVIDIA's full setup and both original plots; model size/global batch changes are explicit. |
| Fleet scenario | Harvard's 8/128 GPU calculation reconstructed with its assumed times and an explicit 6PD/989-TFLOP/s convention; 65.00% to 48.00%, 26.15% relative versus 17.00 percentage points. |
| Aggregation | Work over integrated capacity, rather than unweighted percentages; different-capacity example gives 32%, different-duration example gives 26%. |
| Preserved progress | Original Google entries distinguish blocking/overlapped loading and discarded work; disjoint one-hour example gives 75% productive time and 37.5% preserved-work utilization under stated assumptions. |
| Time to quality | Official MLPerf Quality measure and Run Results/Result files checked; initialization/evaluation exceptions and required logs are named, without a fixed production penalty. |

Harvard's complete Measurement at Scale and Optimization Playbook sections,
including calculations and case-study conclusions, were compared with S3/S7.
The remaining Fallacies/Summary were also read. The following were not adopted
as established facts: fleet MFU must always fall, all loss comes from the network,
one MFU point implies a fixed GPU saving, and a universal 10–20% “hero run tax.”
The staged 70B latency/throughput claims lack a run artifact/configuration in
the selected source and remain an illustrative scenario, not an accepted
benchmark. The local profile/change/reprofile workflow already exists in S7;
it was not duplicated as a new overview.

## Original figures and rights

Two complete NVIDIA PNGs were added at native 4737 × 2125, unchanged.
The seven earlier S3 figures remain. The original graph labels identify
Megatron-Core r0.8.0; the pinned README is on the historical core_r0.13.0
branch. They are not presented as current Megatron results or a Bookvar run.

Source SHA256, dimensions and general repository BSD-3-Clause terms are
recorded in `05 Источники/asset-registry.yml`. The complete NVIDIA license is
adjacent to the figures and copied into public licenses. Google Apache-2.0
and the existing Ultra-Scale Apache-2.0 records are retained. No newly drawn,
generated, translated or cropped teaching figure was used.

## Code and structural evidence

- 460 unit tests in 47 files passed before the final anchor/typesetting fixes.
  The four new fidelity tests were rerun on the final revision and passed.
- `check_roofline_examples.py` executes all three actual CPU Markdown blocks:
  the earlier sequential roofline example and the new MFU/progress examples.
  Both new tables are parsed from the chapter and checked with exact arithmetic.
  The earlier 25 rational cross-checks and worked numbers still pass; 108 exact
  strong-scaling identity cases and the weighted aggregation cases pass.
- All six Python fences parse. Three CUDA fences were not executed. No
  multi-GPU benchmark, NVIDIA result reproduction or Google cloud API run
  was performed or claimed.
- Final build: 1965 pages, complete at 14:15:42.
- Final built-output tests: 134 passed, including nine source images, six
  code blocks, new anchors/explanations and KaTeX error checks.
- Final built links: zero broken internal routes, fragments or files.
- Source coverage: PASS, 210 rows / 58 destinations. This checks source
  structure and links, not full semantic course integration.
- Scoped `git diff --check`: no errors. No staging, commit, push or deploy.

Logs:

```
/var/tmp/bookvar-fleet-measurement-fidelity-final-2026-10-26.log
/var/tmp/bookvar-fleet-measurement-examples-final-2026-10-26.log
/var/tmp/bookvar-fleet-measurement-build-final-2026-10-26.log
/var/tmp/bookvar-fleet-measurement-output-final-2026-10-26.log
/var/tmp/bookvar-fleet-measurement-links-final-2026-10-26.log
/var/tmp/bookvar-fleet-measurement-coverage-2026-10-26.log
```

## Rendered inspection

The final local build was inspected in IAB at 1280 × 900 and 360 × 800.
The original images keep their aspect ratio and full axes/legends; all nine
chapter figures loaded. The desktop screenshot includes source context,
graph, attribution and the explanatory paragraph, not an isolated asset.

At 360 px, a combined MFU/HFU display initially required 428 px within a
313-px content column. The equations were placed on separate aligned lines.
After rebuilding and reloading, there are zero overflowing display-math
blocks and zero KaTeX errors. Page width is 345 px within the 360-px viewport.
The two new tables each have 313-px content width; the calculation table
was personally inspected, including wrapped labels and numerical columns.

The strong-scaling viewer was checked at 3×; its axes/model labels are readable.
Keyboard panning was verified by scrollLeft changing 346 → 386 after the
scroll settled. “Целиком” returns to 1×. The viewer was closed and the
temporary viewport override reset before handoff.

The `/en/` route was also opened. It visibly says “This content is not available
in your language yet,” includes the new section and both loaded figures, and
has no KaTeX errors. This is a verified fallback, not an authored English
translation. No `en/` source file was changed.

Screenshots on the final build:

- `screenshots/2026-10-26-fleet-scaling-desktop.jpg`
- `screenshots/2026-10-26-fleet-scaling-mobile-formula.jpg`
- `screenshots/2026-10-26-fleet-scaling-mobile-table.jpg`
- `screenshots/2026-10-26-fleet-scaling-mobile-viewer.jpg`

## Editorial result and remaining work

The no-ai-slop checklist was applied to authored explanations, not to the
licensed original excerpts. Necessary technical contrasts, derivations,
examples and limitations were retained. No teaching gap identified in this
bounded addition remains open after the mobile formula fix.

The two course matrices record S3 as an additional thematic destination of
H-V2-09 without declaring the entire Harvard source transferred. H-L2-09 and
H-SL2-09 remain unaccepted: the full laboratory and deck require their own
read-through, topic mapping and practice/figure decisions. Only the first
180 lines of the laboratory's readable wrapper/source opening were inspected
here; that is not a full lab audit. Other course/practice/source-rights work
and the overall textbook goal remain open. RL is not returned to the queue.
