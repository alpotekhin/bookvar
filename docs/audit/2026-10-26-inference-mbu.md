# Inference physics: MBU, 26 October 2026

## Scope and revision

Canonical destination:
`00 Учебник/14 Inference и оптимизация/55a Физика LLM inference — prefill, decode и roofline.md`.
The 674-line input was read in full. The final 930-line chapter retains that
explanation and adds one bounded MBU section after the Llama memory estimate.

Final SHA-256:
`fd960c1e11d964376d977525988d7753f351fde4443cbe57b294a91d5a4d63f6`.
The fidelity test removes only the new section and its date/language metadata,
then recovers input SHA
`6c459a9f5153b9eaaa35e437411bb14f4301b2e49d251c6abbe278c0c4257fd5`.
No existing explanation, figure or example was replaced.

## Explanation and source comparison

- The complete selected Databricks MBU definition and 7B / 14 ms example were
  checked. The chapter uses a short attributed paraphrase and explicitly
  distinguishes model-byte accounting from physical hardware counters.
- JAX inference sections on latency objectives, linear operations, attention,
  theoretical latency/throughput and caveats were read, together with complete
  Appendix A. Appendix A is transferred in original English without cuts;
  only the heading level and Liquid image embeds change. The test checks
  the entire transferred section against the pinned source.
- Nsight Memory, Caches, Metrics Overview/Entities, Kernel Replay and
  Application Replay were read. The chapter links the distinction between
  model traffic and physical DRAM traffic, active versus elapsed cycles and
  profiler replay. No long NVIDIA excerpt or new NVIDIA image is copied.
- The Harvard performance deck's low-MFU/high-MBU comparison is qualified
  rather than repeated as a universal optimality criterion. The full course
  remains partial in both source maps.

The added reasoning includes matching the byte budget and timed interval,
deriving the factor of batch size from `R=B/t`, KV allocation versus access,
quantization metadata, union of routed MoE experts, physical traffic versus
model bytes, and uneven per-rank bandwidth bounds. Examples identify their
assumptions and show intermediate values, not only definitions. The JAX
curves are source-reported per-layer TPU measurements; the chapter does not
present them as full-model throughput or measurements of MBU.

The final text was reviewed with no-ai-slop and the project qualifications:
the existing technical explanation and original licensed English remain
intact; necessary terminology, formulas and meaningful contrasts are kept.
No generated diagram, decorative analogy or introductory importance claim
was added.

## Arithmetic and preservation checks

`python3 publishing/tools/check_inference_mbu_examples.py` executes all five
actual Python fences, including the two previous examples. Independent
rational calculations check:

- 50% versus 35% MBU and the incorrect 140% result when batch division is lost;
- all 16 top-1 two-token routing pairs, with shared weights counted once;
- 27/28 physical bandwidth fraction versus 1/2 model MBU in a hypothetical step;
- 12 ms per-rank bound versus 7 ms aggregate ideal and aggregate 7/12 MBU;
- the previous MHA/GQA budget fixtures and the new table rows.

`publishing/tests/inference-mbu.test.ts` also checks unchanged accepted
PPO/DPO/RLVR/GRPO hashes, complete source transfer, pinned downloads, original
PNG dimensions, and identical adjacent/public MIT licenses. RL was not
re-edited. No GPU benchmark, profiler collection, Colab reproduction or
multi-GPU run was performed.

## Figures, rights and rendered checks

Two byte-exact source PNGs, 1044×662 and 1110×674, were inspected at native
resolution and on the actual page. Full original axes, titles and proportions
are retained; provenance, SHA-256, original URLs and MIT evidence are registered.
Copyright notices accompany both the figures and the publicly served license.
Databricks figures were not copied; its HTML snapshot is not a license.

Actual local route:
`http://127.0.0.1:4330/textbook/inference/llm-inference-physics/`.

At 1280 px the graphs render at 617×391 and 617×375. At 360 px they render at
313×198 and 313×190, without cropping. Inline labels are small on a phone;
the viewer is required for detailed reading. At 4×, keyboard panning reaches
both horizontal extremes, 0 and 987 px. New table widths 344 and 357 px scroll
inside a 313 px container; keyboard movement was verified. All four new
display equations fit 313 px; the page has no horizontal viewport overflow
and no KaTeX errors. This is not a claim that every pre-existing wide equation
in the chapter fits without its existing internal scroll.

The final build was reloaded in the browser. Both PNGs loaded at their original
native dimensions. The temporary viewport override was reset, the viewer
closed, and the chapter marked as the deliverable tab.

Screenshots:

- `screenshots/2026-10-26-mbu-desktop.jpg`
- `screenshots/2026-10-26-mbu-mobile-math.jpg`
- `screenshots/2026-10-26-mbu-mobile-figure.jpg`
- `screenshots/2026-10-26-mbu-mobile-viewer-left.jpg`
- `screenshots/2026-10-26-mbu-mobile-viewer-right.jpg`

## Publication checks and outstanding scope

Final evidence: 468 unit tests in 49 files, 135 built-output tests, Astro build
of 1965 pages, zero broken built routes/fragments/files, and source-coverage
PASS (210 matrix rows, 58 destinations). Logs are
`/var/tmp/bookvar-mbu-{unit,assets,output,build,links,coverage}-2026-10-26.log`.
New asset IDs initially failed registry conventions; both the IDs and the
strict validation for these two pinned originals were corrected before the
final full suite passed.

Scoped `git diff --check` passes. The whole-worktree check still reports
pre-existing Markdown hard-break/trailing spaces in derived Harvard lab/deck
pages; these unrelated files were not changed in this unit. A direct curl
check of the live JAX page failed at TLS transport, so its current generated
fragment was not independently validated that way. Pinned GitHub source,
licensed source text and source-image bytes are locally verified; no TLS
verification was disabled.

Manual English 55a remains unchanged, SHA
`fb18a0eada8a1b658ea585191a996aaa97b551f4f784b3b9984ea6aba664e19b`.
It is a separate older chapter, not a fallback, and lacks this MBU addition.
Translation work remains deferred. Existing unconfirmed rights for older
Aleksa figures are not resolved by licensing the new JAX figures.

No staging, commit, push or deploy. The full goal remains active: full course
integration, remaining practice and source/rights findings are not closed.
Next source-specific comparison is Harvard inference lab/deck 10 against the
short existing practice 14 and already accepted theory; only actual missing
explanations should be added, without restarting accepted RL or chapter 55.
