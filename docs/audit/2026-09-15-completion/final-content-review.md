---
type: editorial-review
status: bounded-pass
last_updated: 2026-09-15
---

# Final independent content review: advanced, reference, atlas, and root deltas

## Verdict

**Bounded PASS.** The original advanced/reference/atlas findings are accounted
for exactly, and no remaining P1/P2 issue was found in the eight page instances
read in full for this review. One P2 was raised during review: the renamed GPU
softmax section initially lacked its old generated slug. Root resolved it by
adding the explicit compatibility anchor
`первый-полный-пример-softmax-одной-строки` immediately before the new heading;
the current page was rechecked after that change.

This is not a claim that every advanced, reference, or atlas page was
independently reread. It is an identity/status reconciliation of all 155
recorded findings plus a complete-text substantive review of the eight page
instances listed below. The broader completion ledgers retain their own stated
full-read evidence and limitations.

## Ledger reconciliation

The original finding tuples were compared as multisets using page path plus the
six original fields `severity`, `line`, `excerpt`, `issue`, and `action`.

| Scope | Original | Completion | Identity | Recorded outcomes |
| --- | ---: | ---: | --- | --- |
| Advanced | 55 | 55 | exact | 48 fixed, 7 already-fixed, 0 open |
| Reference/research concepts | 57 | 57 | exact | 48 fixed, 9 already-fixed, 0 open |
| Atlas | 43 | 43 | exact | 38 fixed, 5 already-fixed |

The corresponding implementation evidence is internally consistent at the
declared scope: advanced reports 47/47 byte-exact task-entry snapshots (40 RU
pages and 7 authored EN pages), reference reports 45/45, and atlas reports
28/28. This review did not reinterpret those snapshot counts as independent
full-page reading.

The following later hashes are expected superseding deltas, not unexplained
ledger regressions:

- DPO now hashes to
  `2c01405ddf084d092b9838dc2b26efb43f8962fc1ee1b5b8190ea87c68f5a4ec`;
  its predecessor and root review are recorded in
  `docs/audit/2026-09-15-completion/primary-content-review.md`.
- Jamba now hashes to
  `26ba768152a94987208b53eeb06e1e01a4d6c0edfe54e88dc0cee01af44b4ac3`
  because root replaced the cropped derivative PNG with the original SVG.
- The current NTP, long-context, and GPU/Triton hashes in the evidence table
  below include the subsequent root course/anchor work reviewed here.

## Full-page sample

All files below were read from frontmatter through EOF in their current form.

| Page | Lines | Current SHA-256 | Review result |
| --- | ---: | --- | --- |
| `00 Учебник/12 Post-training и Alignment/05 DPO.md` | 789 | `2c01405ddf084d092b9838dc2b26efb43f8962fc1ee1b5b8190ea87c68f5a4ec` | PASS |
| `00 Учебник/11 Pre-training и Scaling/42 Next-token prediction.md` | 240 | `9db385e6450e19734669a2cb76d648cecec264e494ae2a86ec8226bd93380ba7` | PASS |
| `00 Учебник/08 Эффективный Attention и длинный контекст/03 Длинный контекст — расширение, разреженность и оценивание.md` | 384 | `a905656cb7446ed983d0dce75ce80a8173eb0f0346a2aa7960efe054c0dcb433` | PASS |
| `02 Атлас моделей/Семейства/Jamba.md` | 54 | `26ba768152a94987208b53eeb06e1e01a4d6c0edfe54e88dc0cee01af44b4ac3` | PASS |
| `00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению.md` | 332 | `9f57d28da5aa5e427abc497e844285d33bd6dda977b1bd3fe6f61536afa59e1f` | PASS after compatibility-anchor fix |
| `en/00 Textbook/16 Multimodal Models/64 Multimodal models.md` | 153 | `69ad3615bf15b78ae670ae8affd11ebb861b61bdcd4f11714d8e33b3a91e80a8` | PASS; authored EN, not locale fallback |
| `01 Справочник/FFN и MoE/Mixture of Experts.md` | 113 | `373913e41e4a65bf3b47747e9c09d30e0d6d2e4f6aa31acae96caf41ca212ba5` | PASS |
| `02 Атлас моделей/Семейства/DeepSeek.md` | 83 | `922432e4f44a0ec5360475bb561bcc15df5438ef78be2c0651cb08e7a1d42206` | PASS |

### Substantive checks

- **DPO:** the Bradley–Terry reduction, reference-policy substitution,
  finite-normalizer/support qualification, sequence log-probability shifting
  and masking, derivative sign and beta saturation discussion, and offline
  versus online preference-optimization boundary are mutually consistent. The
  length-normalization section labels its changed objective rather than
  presenting it as canonical DPO.
- **NTP/MTP:** masking, NLL, perplexity, valid target count, and the distinction
  between independent Gloeckle-style heads and sequential DeepSeek-style MTP
  modules are accurate. Training auxiliary prediction is not conflated with
  speculative inference; the `D=1` special case is handled coherently.
- **Long context:** the chapter separates accepted context length from usable
  context, distinguishes position extrapolation, sparse computation, state
  compression, and evaluation, and gets the local-window receptive-field and
  mixed local/full-cache arguments right. CLA reduces a layer multiplier while
  retaining per-layer queries; DSA's indexer is not mistaken for the main
  attention path.
- **Jamba:** the original hybrid/MoE ratios and the current Jamba 2 update are
  stated without turning release claims into measured local evidence. The
  original SVG now replaces the known cropped derivative.
- **GPU kernels/Triton:** the page explains program instances, tiling, masks,
  online softmax, coalescing, occupancy, and benchmark discipline without
  claiming a GPU execution in this pass. The displayed softmax is explicitly
  an algorithmic teaching example rather than a runnable Triton kernel. The
  local Stanford lecture source link resolves.
- **Authored EN multimodal page:** ViT patch/token arithmetic, CLIP versus
  SigLIP normalization, early/late fusion, connector cost, and serving concerns
  are coherent. Its reader-facing language is natural authored English and its
  local chapter/practice links resolve.
- **MoE reference:** router logits, selected-expert normalization, capacity
  factor, buffer capacity, parameter capacity, and the DeepSeek auxiliary-loss-
  free balancing qualification are kept distinct. The worked top-2 and capacity
  examples match the formulas.
- **DeepSeek atlas:** the card separates architecture, post-training, and
  serving claims, and routes MLA, MoE, MTP, DSA, RLVR, and GRPO to the correct
  level of detail. MTP training is not relabelled as speculative decoding.

## Figures, anchors, and navigation

No pre-existing figure embed disappeared from the sampled DPO, NTP,
long-context, GPU/Triton, authored EN, MoE-reference, or DeepSeek pages. NTP and
long-context add course-follow-through figures without displacing the old
figures. Jamba is the intentional exception: the cropped PNG was replaced with
`curated/course-follow-through/jamba-architecture.svg`.

The current Jamba SVG and `raw/papers/jamba/figure1-original.svg` are
byte-identical at SHA-256
`5ee975e76a73d5a9dd738b353b6eda0ca7a3c8d4f18be5928fc0c5088cb90b88`.
The asset registry records the arXiv Figure 1 source and CC BY-SA 4.0. This
review verified byte identity and provenance, but did not independently render
the SVG in a browser; browser acceptance remains with root.

Across the eight sampled pages, 52 actual local wiki/Markdown link or embed
references were checked after excluding code-syntax false positives; their file
targets resolve. Historical heading compatibility is preserved for the renamed
DPO, NTP, and long-context sections. The GPU/Triton old slug is now preserved by
the explicit anchor at the renamed softmax section.

## Executed checks

Executed in the pinned CPU environment:

- `test_course_attention_mtp.py`: 3/3 PASS (CLA, local/hybrid attention, MTP
  targets and normalization).
- `test_primary_review_examples.py`: 3/3 PASS (DPO shifted/masked sequence
  log-probability and backward pass, beta/margin saturation, plus the unrelated
  derivative regression in the same bounded test file).
- Direct scalar assertions: MoE capacity `ceil(1.2 * 100 / 4) = 30`; CLA cache
  ledger 2 GiB versus 4 GiB; DSA adaptation-token totals 2.097152B and
  943.7184B; ViT 384/16 and 768/16 patch/pair counts. PASS.

No model weights, training run, GPU kernel, GPU benchmark, full site build, or
browser render was run or inferred from these checks.

## Current primary-source spot checks

- Jamba/Jamba 2 release facts were checked against the
  [original AI21 Jamba release](https://www.ai21.com/blog/announcing-jamba/),
  [AI21 Jamba 2 release](https://www.ai21.com/blog/introducing-jamba2/), and
  [Jamba paper](https://arxiv.org/abs/2403.19887).
- DeepSeek architecture facts were checked against the
  [official DeepSeek-V3 repository](https://github.com/deepseek-ai/DeepSeek-V3)
  and DSA/current-model claims against the
  [official DeepSeek-V3.2-Exp repository](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)
  and [DeepSeek-V3.2 paper](https://arxiv.org/abs/2512.02556).

These checks validate the sampled statements and source identities; they are
not a fresh source audit of every claim in the 155-finding scope.

## Evidence boundary

The final site build and browser checks were still owned by root and are not
claimed here. The overall result is therefore an independent bounded content
acceptance for the exact sample and ledger reconciliation above, not a final
all-site release gate.
