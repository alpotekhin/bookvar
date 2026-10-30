# Pretraining 42–44 — local acceptance, 2026-10-13

This record covers the Russian next-token, scaling-laws and distributed
overview chapters, plus the small Harvard source-link repair in S4. It is
not acceptance of the whole textbook or of the subsequent 44a–44h review.
All three original and revised chapters were read in full by the primary
editor. Chapter-specific findings and sources are in the same-day
`next-token`, `scaling`, `distributed-overview` and `harvard-arithmetic-links`
audit records.

## Executed checks

- `site check`: 0 errors, 0 warnings, 0 hints.
- `site build`: 1,961 pages; completed successfully. The pre-existing `/404`
  route-conflict warning remains; this is not a warning-free build claim.
- `publishing test:unit`: 231/231 passed, including the stricter asset tests.
- `site test`: 64/64 passed.
- `publishing test:output`: 12/12 passed on the resulting build.
- `publishing check:links`: no broken internal routes, fragments or files.
- `publishing check:source-coverage`: 210 rows / 59 destinations passed.
  This checks mapped destinations and links, not full knowledge transfer.
- `check_ntp_examples.py`, `check_scaling_examples.py` and
  `check_distributed_overview_examples.py`: passed. See their per-chapter
  records for numerical, autograd, negative-control and mocked checks.

The build preceded a metadata-only normalization of three MiniCPM figure
records. Chapter text and image bytes did not change after that build;
the complete publishing unit suite was then run on the final registry.

## Browser inspection

The built site was inspected at `http://127.0.0.1:4330/`, with 1280×900 and
390×844 viewport overrides; the temporary override was subsequently reset.

- Chapter 42: three images loaded, no KaTeX errors, no document overflow.
  Inspected the introduction and causal-decoder explanation. Clicking the
  decoder image opens the 2000×1056 original; browser Back returns to the
  chapter and section.
- Chapter 43: all 23 images loaded, no KaTeX errors or document overflow.
  Inspected IsoFLOP curves, WSD explanation and worked examples. Every source
  image was also inspected at native resolution; five crop boundaries were
  corrected before the build to retain axis labels and remove a neighbouring
  plot fragment.
- Chapter 44: all seven images loaded, no KaTeX errors or document overflow.
  Inspected FSDP forward/backward, the TP illustration and GPipe on desktop;
  mobile text, table containment and all image bounds were checked.
- S4: the new Harvard paragraph and its three publication links were
  inspected in the rendered DOM; the existing nearby memory illustration
  was inspected visually. This was a targeted check, not a fresh full-S4
  editorial review.

At 390px the content width is 343px. Wide, multi-panel plots necessarily
have small labels in that layout; original-image links provide enlargement.
No claim is made that every label is readable without opening the original.

## Boundaries and remaining work

No CUDA, NCCL, multi-node training, GPU memory profiling or throughput
benchmark was run for these chapters. CPU calculations are not GPU results.
New D2L figures have verified CC BY-SA provenance. Existing CS336 course
figure reformats retain explicitly unverified third-party rights metadata;
the MiniCPM underlying-paper licence is recorded separately where verified.

English, commit, push and deployment are outside this continuation. Existing
unrelated edits remain untouched. The active editorial goal continues with
the detailed distributed chapters 44a–44h and subsequent reader-facing pages.
