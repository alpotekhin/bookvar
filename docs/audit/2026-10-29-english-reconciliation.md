# English textbook reconciliation — 29 October 2026

## Result

All 29 pre-existing English textbook files now match the substantive content
of their recorded canonical revisions: 28 chapters and the contents page.
The earlier 28 partial entries are closed. This does not claim an English
translation of all 149 canonical pages. Chapters without a separate English
file still open an explicitly labelled canonical-language fallback.

The [first-pass report](snapshots/2026-10-29-english-reconciliation-first-pass.md)
and [ledger](snapshots/2026-10-29-english-reconciliation-first-pass.json)
retain the earlier one-current/28-partial result. The current
[ledger](2026-10-29-english-reconciliation.json) records the complete before,
intermediate and after revisions, canonical SHA, figure inventory and detailed
parity report for each file.

## Delivered inventory

| Group | Existing chapters | Figure placements | Executed CPU blocks | Detailed evidence |
|---|---:|---:|---:|---|
| Self-attention | 1 | 4 | 4 | First-pass report above; shared example checker |
| Transformer, positions, masks, BERT, GPT-1/2/3, T5 | 9 | 53 | 33 | [Transformer and model-family parity](2026-10-29-en-transformer-models-parity.md) |
| Decoding, KV, scheduling, engines, FlashAttention, quantization, speculation | 7 | 44 | 33 | [Inference parity](2026-10-29-en-inference-parity.md) |
| Inference physics | 1 | 8 | 5 | [Physics and contents](2026-10-29-en-index-physics-parity.md) |
| Distributed inference and prefill/decode disaggregation | 2 | 18 | 18 | [Distributed-serving parity](2026-10-29-en-distributed-parity.md) |
| Benchmarking and SLOs | 1 | 6 | 8 | [Benchmark parity](2026-10-29-en-benchmark-parity.md) |
| Vision, connectors, resolution, training, OCR, audio/video, VLM evaluation | 7 | 50 | 75 | [Multimodal parity](2026-10-29-en-multimodal-parity.md) |
| Contents | 1 index | 0 | — | [Physics and contents](2026-10-29-en-index-physics-parity.md) |

The English chapter inventory contains 183 figure placements, compared with
108 before this turn and 116 at the first checkpoint. These are placements,
not 183 newly downloaded assets. All canonical figures are retained; the
English physics and disaggregation chapters also keep useful earlier figures.
No generated substitute diagrams were introduced.

The contents now follow the canonical nine core parts and thirteen independent
modules. All 147 chapter destinations are present; 28 separate English chapters
are marked EN. Obsolete ordinal prefixes no longer appear in English textbook
navigation or H1 titles. Original course titles and Markdown filenames were
not renamed.

## Explanation and source review

Each English/canonical pair was read in full. The group reports map restored
mechanisms, intermediate calculations, caveats, original excerpts and figures.
The parent additionally read the complete final masking, physics, benchmarking,
distributed-inference and disaggregation bodies, reran delegated examples and
reviewed the built pages. Passing a hash or figure count is not used as proof
that an explanation is clear.

The completed text retains full worked examples: Transformer forward/backward
and loss calculations; actual small GPT/T5 training or generation steps;
allocation, cache offsets and speculative rollback; online-softmax backward;
TP partitions, MoE dispatch, layer-transfer dependencies; CLIP/SigLIP objectives,
Q-Former visibility, spatial coordinates and multimodal evaluation counterexamples.
Approved source-language excerpts remain unchanged and attributed.

One canonical arithmetic error was found during disaggregation review.
With capacity 16384, active 8192 and reserved 4096, an incoming 4096 fits
exactly. The failing example now uses incoming 8192, and English explains
both cases. Reversing that single line reconstructs the exact previous
canonical file SHA; code and the rest of its explanation are unchanged.

## Rendered acceptance

All 29 routes were opened at 360×800 and 1280×800 after the successful build:
58 route/width checks, 183 image placements at each width. No broken image,
KaTeX error, overflowing display equation or document-level horizontal overflow
remained. The [browser receipt](2026-10-29-en-parity-browser.json) pins these
observations to the reviewed chapter hashes.

The final mobile pass required 20 formula reflows in eight chapters.
[Exact old/new expressions](2026-10-29-en-parity-render-reflow.json) are recorded.
Reversing them restores the reviewed pre-layout files byte for byte; mathematical
tokens, prose, code and figures are unchanged. The earlier eight-reflow journal
is preserved separately; FlashAttention later received longer derivations.

The parent inspected the Molmo crop/token figure in desktop and mobile layouts,
opened its full viewer, zoomed to 2× on mobile and returned to the text.
The tensor-parallelism and Splitwise layer-transfer figures were inspected with
their captions on desktop. This selected visual inspection is distinct from
the automated all-page checks and does not claim a fresh source-semantic review
of all 183 images.

## Verification and limits

The parent executed the published CPU examples using the saved checkers,
including every delegated batch. In total, 176 displayed blocks passed;
additional cases check attention outputs/gradients, INT4 packing, residual
sampling distributions, timestamp arithmetic and layout equivalence.
GPU-dependent runtime excerpts are explicitly excluded. These are teaching
calculations, not pretrained-model quality or throughput measurements.

Integrated checks on the delivered sources:

- Build: 1989 total site routes, exit 0; 25.73 seconds.
- Publishing unit checks: 758 passed across 118 files.
- Built HTML: 165 passed.
- Site checks: 69 passed.
- Internal routes, fragments and files: zero broken targets.
- Source coverage: 210 matrix rows passed; this is structural coverage,
  not proof that entire courses were transferred.

Logs use `/var/tmp/bookvar-en-parity-*-20261029.log`; exact commands and scoped
execution logs are retained in the linked reports. Warnings from Poppler and
the host locale did not cause a failed final test run.

No stage, commit, push or deployment was performed. The local preview is
[Bookvar in English](http://127.0.0.1:4330/en/textbook/).
Full-book translation, GPU/NCCL experiments and the remaining source-course
units are outside this completed parity batch.

