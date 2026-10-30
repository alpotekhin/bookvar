# English textbook reconciliation — 2026-10-29

## Delivered scope

All **29 existing English textbook files** were read: 28 chapters and the index.
This is not a complete English translation of the textbook. Four EN/canonical
chapter pairs were read in full and substantially reconciled: self-attention,
scheduling, serving engines and FlashAttention. For the other chapters, the
existing English body was read and its canonical mapping and figure inventory
were checked; their complete canonical source bodies were not all reread in
this pass. They remain explicitly partial.

The result contains **one current chapter and 28 partial pages**. Partial means
an authored English body with known missing expansions; it does not mean
an original-language fallback, and does not establish content parity.
No new English chapter or source fallback was counted as a translation.

## Content changes

- **Self-attention:** realizable shared projections with $X=I_3$, the full
  forward and backward calculation, all projection/input gradients, a checked
  SGD step, permutation equivariance, and a finite non-identifiability example.
  Two original figures were added. Four source figures now match the canonical
  inventory. The old $X=Q=K$ example could not produce its displayed $V$ with
  one linear projection; the correction is explained and regression-tested.
- **Serving engines:** original Apache-2.0/MIT explanations from pinned vLLM,
  SGLang, TensorRT-LLM and FlashInfer documentation, seven native source figures,
  process counts, weight loading, overlap scheduling, ragged/mask/page/MLA
  layouts, and CPU examples separating token selection from KV creation and
  preserving request boundaries. Historical SGLang one-token pages are
  explicitly dated. PyExecutor is not described as mandatory TensorRT-engine
  compilation. Source runtime fragments are not presented as tested snippets.
- **Scheduling:** separate token and block budgets, a three-pass worked
  continuous-batching example, a checked packed causal mask, block-aligned
  full-prefix hits and the last logits-producing forward pass, and the
  distinction between reference release and cache eviction. Two native source
  figures added. Chunk size is no longer claimed to impose a hardware-independent
  time bound.
- **FlashAttention:** FA1/FA2 loop-order distinction, the rescaled online-softmax
  example yielding 30 rather than the incorrect 26, executable blocked forward,
  backward reconstruction from logsumexp, FA2 task counts and normalization,
  FA3 producer/consumer and pingpong dependencies, two original FA3 figures and
  unchanged CC BY 4.0 paragraphs on register pressure and compiler reordering.
- **Positional information:** an allocated but untrained embedding row is
  distinguished from a position outside the allocated table.
- **GPT-3:** demonstration prefixes are no longer said to be recomputed on
  every request when a compatible prefix cache can be reused.
- **Navigation:** complete Transformer points back to positions; architectural
  patterns points forward to BERT.
- **Index:** explains authored partial EN versus original-language fallback and
  does not equate availability with completed translation.

Across existing EN textbook pages: **6,832 → 8,055 lines**, **108 → 116 figure
embeds**. Line count is an inventory measure, not a quality score. All eight
additional embeds reuse already accepted source figures; no new generated
illustrations, downloads, or asset-registry changes were introduced.

## Frontmatter contract

- `translation_status: current | partial`
- `canonical_revision: "<64-character SHA256>"` of `translation_of`
- `translation_reviewed: "2026-10-29"`

The date records the reconciliation pass, not fresh validation of every
historical or version-specific claim. Existing last_verified dates were not
globally advanced. The exact file/route/revision inventory and known missing
expansions are in [the JSON ledger](2026-10-29-english-reconciliation.json).

## Verification

Executed:

```sh
LC_ALL=C LANG=C uv run --offline --with numpy==2.3.5 --with torch==2.14.1 python publishing/tools/check_english_reconciliation_examples.py
pnpm --dir publishing exec vitest run tests/english-reconciliation.test.ts
```

- 9 standalone Python blocks from the actual Markdown passed.
- 36 FlashAttention configurations passed output, logsumexp, and Q/K/V gradient
  comparisons against the direct implementation, including causal and incomplete
  tiles.
- Six Vitest checks passed: exhaustive EN inventory; status/hash contract;
  existing figure files; the projection-regression guard; and unchanged licensed
  original excerpts; exact preservation of mathematics in eight mobile reflows.
- Serving fragments 1–5 are runtime/GPU excerpts or symbolic layout examples
  and were **not** executed. No throughput or GPU performance was measured.
- Parent subsequently checked the four expanded EN chapters at 1280 and 360 px.
  No broken source images, KaTeX errors, overflowing display equations or page
  overflow remained. The current/partial/fallback notices were checked separately.
  This is not a rendered review of all 29 EN pages. See
  [the combined acceptance report](2026-10-29-one-pass-completion.md).

## Routes and remaining expansions

| English route | Status | Outstanding canonical expansion |
|---|---|---|
| /en/textbook/transformer/self-attention/ | current | None identified in the reviewed chapter pair. |
| /en/textbook/transformer/masking-heads-shapes/ | partial | Complete executable multi-head forward/backward and packed-cache examples from the canonical expansion. |
| /en/textbook/transformer/positional-information/ | partial | Extended RoPE and relative-position numerical examples; two additional canonical figures. |
| /en/textbook/transformer/full-transformer/ | partial | Expanded complete forward/loss/gradient examples and four additional canonical figures. |
| /en/textbook/transformer/encoder-decoder-patterns/ | partial | Extended executable pattern comparisons and an additional canonical figure. |
| /en/textbook/models/bert-roberta-deberta/ | partial | Expanded objective/attention comparisons and end-to-end worked examples. |
| /en/textbook/models/gpt-1/ | partial | Full GPT-1 training and fine-tuning walkthroughs, additional native figures and checks. |
| /en/textbook/models/gpt-2/ | partial | Expanded GPT-2 numerical examples and three additional canonical figures. |
| /en/textbook/models/gpt-3/ | partial | Expanded GPT-3 scoring/scaling examples and three additional canonical figures. |
| /en/textbook/models/t5/ | partial | Complete T5 executable forward and sentinel/mask examples; three additional canonical figures. |
| /en/textbook/inference/decoding/ | partial | Expanded probability calculations and executable decoding experiments. |
| /en/textbook/inference/kv-cache-batching-pagedattention/ | partial | Incremental forward equivalence, cache-offset masks and additional canonical illustration. |
| /en/textbook/inference/llm-inference-physics/ | partial | Extended roofline/bandwidth examples and four additional canonical figures. |
| /en/textbook/inference/scheduling/ | partial | Complete block allocator/hash-key walkthroughs and remaining canonical examples. |
| /en/textbook/inference/serving-engines/ | partial | Remaining canonical source framing and source-link detail; retained English explanatory sections intentionally differ from canonical ordering. |
| /en/textbook/inference/flashattention/ | partial | Detailed two-pass backward and version/backend diagnostic walkthrough; canonical CPU code and two FA3 figures are included. |
| /en/textbook/inference/quantization/ | partial | Expanded GPTQ/SmoothQuant worked calculations and four additional canonical figures. |
| /en/textbook/inference/speculative-decoding/ | partial | Complete distribution-preservation derivation, tree examples and additional native figures. |
| /en/textbook/inference/distributed-inference/ | partial | Detailed TP partition, normalization, PP and EP worked examples and six additional canonical figures. |
| /en/textbook/inference/disaggregated-serving/ | partial | Extended handoff-state/protocol walkthroughs and canonical Splitwise figures. |
| /en/textbook/inference/benchmarking-slo/ | partial | Executable timestamp ledger, tail/censoring examples and additional canonical figures. |
| /en/textbook/multimodal/models/ | partial | Expanded CLIP/SigLIP losses and gradient examples and three additional canonical figures. |
| /en/textbook/multimodal/connectors-fusion/ | partial | Full Q-Former mask/stage walkthroughs and three source-native canonical BLIP-2 figures. |
| /en/textbook/multimodal/visual-tokenization-resolution/ | partial | Expanded coordinate/rotary calculations and canonical Pixtral packing figure. |
| /en/textbook/multimodal/training-data/ | partial | Complete frozen-module gradient and loss-normalization examples; existing source figure inventory retained. |
| /en/textbook/multimodal/documents-ocr-grounding/ | partial | Expanded document-model walkthroughs and four additional canonical source illustrations. |
| /en/textbook/multimodal/video-audio-omni/ | partial | Expanded video/audio mechanisms and six additional native illustrations. |
| /en/textbook/multimodal/evaluation-serving/ | partial | Expanded metric/counterexample/serving calculations and five additional canonical figures. |
| /en/textbook/ | partial | Core-versus-independent-module contents structure still follows the older 14-part English outline; fallback/reconciliation limitations are now explicit. |

## Integration boundary

The parent owns per-page notices, route handling, the shared publication build,
browser QA, and global audit integration. No shared registry, canonical RU
chapter, publication manifest, site stylesheet, git commit, push, or deployment
was changed by this EN subtask. Completion of the remaining canonical expansions
must not be inferred from the partial notices.

