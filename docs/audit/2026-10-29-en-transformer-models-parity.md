# English Transformer and model-family reconciliation — 29 October 2026

## Delivered scope

All nine existing English chapters in the assigned Transformer/model-family group now match the substantive content of their pinned canonical chapters. Self-attention was outside this batch. No canonical chapter, shared asset registry, central English ledger, publication manifest, or site stylesheet was changed.

Each old English chapter and canonical chapter was read from beginning to end. The revised English text was then reviewed against the canonical explanation, including numerical examples, assumptions, negative results, source limitations, and figure captions. The JSON journal records the full read ranges and section-by-section correspondence. Section counts, hashes, and automated checks guard against later loss; they do not themselves establish explanatory quality.

The source text grew from 2,685 to 7,584 lines, and figure embeds from 35 to 53. These are inventory counts, not a claim that length proves quality. There are 31 newly referenced figure embeds and 13 replaced or removed embeds, a net increase of 18. All new references reuse images already present in the accepted canonical material; no new generated illustrations or source downloads were introduced. The journal lists every replacement. Older crops and generic diagrams give way to the canonical paper figures; the former RoBERTa table screenshot and DeBERTa equation screenshot are represented as a complete numeric table and rendered equations with a worked example.

| Chapter | Source lines, before → after | Figure embeds, before → after | CPU examples executed |
|---|---:|---:|---:|
| [masking-heads-shapes](../../en/00%20Textbook/05%20Attention%20and%20Transformer/03%20Masking,%20multi-head%20attention,%20and%20tensor%20shapes.md) | 287 → 779 | 5 → 5 | 6 |
| [positional-information](../../en/00%20Textbook/05%20Attention%20and%20Transformer/04%20Positional%20information.md) | 397 → 1001 | 4 → 6 | 6 |
| [full-transformer](../../en/00%20Textbook/05%20Attention%20and%20Transformer/05%20The%20complete%20Transformer.md) | 396 → 1091 | 3 → 7 | 5 |
| [encoder-decoder-patterns](../../en/00%20Textbook/06%20Encoder%20Decoder%20and%20Encoder-Decoder/01%20Three%20architectural%20patterns.md) | 430 → 860 | 5 → 6 | 4 |
| [bert-roberta-deberta](../../en/00%20Textbook/06%20Encoder%20Decoder%20and%20Encoder-Decoder/03%20BERT,%20RoBERTa,%20and%20DeBERTa.md) | 300 → 890 | 8 → 7 | 4 |
| [gpt-1](../../en/00%20Textbook/06%20Encoder%20Decoder%20and%20Encoder-Decoder/04%20GPT-1%20—%20generative%20pre-training.md) | 186 → 690 | 1 → 4 | 2 |
| [gpt-2](../../en/00%20Textbook/06%20Encoder%20Decoder%20and%20Encoder-Decoder/05%20GPT-2%20—%20zero-shot%20through%20language.md) | 182 → 677 | 3 → 6 | 2 |
| [gpt-3](../../en/00%20Textbook/06%20Encoder%20Decoder%20and%20Encoder-Decoder/06%20GPT-3%20—%20in-context%20learning.md) | 232 → 806 | 3 → 6 | 2 |
| [t5](../../en/00%20Textbook/06%20Encoder%20Decoder%20and%20Encoder-Decoder/07%20T5%20—%20text-to-text%20Transformer.md) | 275 → 790 | 3 → 6 | 2 |

## What was restored

- **Masks and heads:** mask-before-softmax normalization, padding and empty rows, Boolean API polarity, split/merge order, output projection, named axes, broadcasting, and rectangular cached attention. The full-sequence qualification distinguishes equal-length self-attention from cached rectangular passes.
- **Positions:** complete learned/sinusoidal, Shaw, T5-bucket, RoPE, and ALiBi explanations and calculations. Cache positions, repeated rotation, and extrapolation limitations remain explicit.
- **Complete Transformer:** residual/FFN/block construction, teacher forcing and masking, vocabulary probabilities and smoothing, a complete small translator, and a modern causal block with architectural distinctions.
- **Three patterns:** separate input visibility, supervised target positions, cache dependencies, span corruption, and two-layer indirect leakage in an incorrectly masked prefix LM.
- **BERT/RoBERTa/DeBERTa:** a full training example and QA head, experimental protocols, the three disentangled attention terms, numerical computation, enhanced mask decoder, and reported-versus-recomputed aggregate distinctions.
- **GPT-1:** all four task input formats, correct pooling, original implementation, auxiliary LM normalization, an actual small weight-update example, and the mixed ablation results.
- **GPT-2:** released parameter counts, byte BPE, generation and sampling, full-versus-cached forward agreement, delta `present` versus full `past`, likelihood-scoring conventions, and limits of empirical comparisons.
- **GPT-3:** unchanged parameters during in-context learning, candidate scoring, full/local attention, data mixtures and packing, task-specific protocols/results, contamination, and negative results.
- **T5:** decoder shift and visible start-PAD, norms and projection scaling, relative buckets, exact configuration arithmetic, span samplers and sequence lengths, one training step, C4 filtering, experimental regimes, mixtures/adapters/scaling, and source-dependent cache invalidation.

Approved original English excerpts remain verbatim: the CS336 block description, two CC BY 4.0 T5 captions, and three context-dependent implementation snippets. Their citations and existing license annotations remain with them. All canonical source URLs and explicit stable fragment aliases are present in the English versions.

## Checks performed

1. All **33 standalone displayed Python programs** executed successfully on CPU. The runner compares each program's Python AST against its canonical counterpart before executing it. Translated comments do not change statements. Execution was performed on the after-revisions recorded below.
2. All **53 figure embeds** match canonical order, resolve locally, and have recorded SHA-256 hashes. This check does not establish cropping or legibility.
3. All **79 display equations** match canonical expressions, allowing only translated logical words and one documented BERT line reflow. No solitary `$` display delimiter remains.
4. All **53 fenced blocks** have English/canonical hashes and execution classification. The three original implementation excerpts require their source runtimes and were **not executed**; literal text examples are not programs.
5. **48 targeted regression tests passed** in `publishing/tests/en-transformer-models-parity.test.ts`. They pin file revisions, frontmatter, assets, equations, code inventory, source quotations, URLs, aliases, and recovered technical distinctions.

Reproduce the CPU checks from repository root:

```sh
LC_ALL=C LANG=C uv run --offline \
  --with torch==2.14.1 --with numpy==2.3.5 \
  --with einops==0.8.1 --with transformers==4.57.6 \
  python publishing/tools/en-transformer-models-check.py
```

The offline command requires an existing package cache. On the first run, omit `--offline` to download dependencies. No pretrained weights are required.

Run the regression tests from `publishing/`:

```sh
npx vitest run tests/en-transformer-models-parity.test.ts
```

The examples test mechanics and small numerical cases. They do not reproduce source training runs, trained-model quality, GPU throughput, or benchmark results.

## Visual acceptance and integration

Parent review identified one BERT Q/K/V expression that exceeded a 313-pixel mobile column. It was split into three aligned lines without changing any equation. The before/after expression and resulting file hash are in the journal. The parent owns the final build, rendered desktop/mobile review, central 29-page English-ledger update, and publication. This report does not mark those separate checks complete.

All nine chapter bodies are frozen. No known substantive canonical omission remains in this batch. Whole-book or publication completion is not claimed here.

## Exact revisions

- `en/00 Textbook/05 Attention and Transformer/03 Masking, multi-head attention, and tensor shapes.md`
  - before: `9e9c18a705a9c4614551034c22f4654a68408b76d20fd0471602cfb885df62ec`
  - after: `64bbbd9085a4c30083a2f77df49316bc13087083d43f720ec7937ae781154e9d`
  - canonical: `9306be89c604c905489000ffcebffb32e16161f66e07540005d0f394097c0fd5`
- `en/00 Textbook/05 Attention and Transformer/04 Positional information.md`
  - before: `b44b50da949d9f488cab9fe7304a37636f6b6858474fee3a69d10034f71b0962`
  - after: `8e2fa5500c85ba4d850b91a22bb404b7dffcbed02c595bf87c8e875a36917090`
  - canonical: `897a19c428e9ef20a6c651e981eb32d905d6f72a616907c2ea2e20fa1842451a`
- `en/00 Textbook/05 Attention and Transformer/05 The complete Transformer.md`
  - before: `7c95420ddda31bc7134ec3994a58a96f974fe8f5925cd61b5bb761e1ab9a1723`
  - after: `64ee4c1d8984515c3baca3f9ccfd2086cc3a9a53364ebc07204eb23a87e16106`
  - canonical: `8bda8af63d82960da7d5849e132be647542b90d4a9a9daca05ff5a0f2e66d356`
- `en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/01 Three architectural patterns.md`
  - before: `967da02f188cf49752845634e168f1a0cf3664ac6f6d2093c6331b6737a81adf`
  - after: `d2104d394673c898c019becd49fa2337f01ae82abf694ac03ea3fabdbb4212a2`
  - canonical: `a99819704bcdd1ff795c39a2240b8c32331fded5b1372c302f69e381d123d6e8`
- `en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/03 BERT, RoBERTa, and DeBERTa.md`
  - before: `d3f4a8b520b2dedf910d2c53e044a1f7cadb74b72fa7d6f4ec26bceba06e2cc9`
  - after: `888b25e8db26b37c2d2fcecc366d21744b88ba8ba229df0365c18794529847c5`
  - canonical: `fa3595eca44cec1269db1425337a04773d132d2dd685432d1b12518a15ae19b4`
- `en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/04 GPT-1 — generative pre-training.md`
  - before: `1705dce2150e38f2d240437cc92318866589d0e8ab0eebbc10501e760d1e389d`
  - after: `4aff198f47b6d6e13167d8394d82a25a887b41b51977121a17acf29dda51a5ef`
  - canonical: `318e040db226fabac04ba6576bf629775600e4626a03e2ac7ce0a4ec16967499`
- `en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/05 GPT-2 — zero-shot through language.md`
  - before: `4b889ce727279d4f209de68306da37770d7a4fbf0da53f049d8a538f8447ae3a`
  - after: `6b248051dd87cc3c58521cdcba0021fe3630ffe8aa0125c728dd821bf6cefbb8`
  - canonical: `ce717410045dd566b798820440ef49db532c1204d6d0bed6b54cfec1247f6c8d`
- `en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/06 GPT-3 — in-context learning.md`
  - before: `25cfe59f394629e001fc431f255d122c383ad9e13b3e1e4e4c85a16a2f5004b6`
  - after: `48ccbfcdfa89f81c1da3aadf80bff4c8e2ba716b67089f28269bfcd3f362e609`
  - canonical: `76ed87ea2ca9e9c2bc82b7926eb95de9313d0b11c8ce14ab95ab13313e19875a`
- `en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/07 T5 — text-to-text Transformer.md`
  - before: `4f08155e8f879953d5b11ffab393133e73c7975e9cd0cdbf2a93355076305a70`
  - after: `0e5c05aa03151e24da5ef6735bcd5525a0c4ea5029b4e9414993b190d7395192`
  - canonical: `1bced617c9985be26919d79c4d01a87e4d0a3087d59e65d2e200fd2f84608dfc`

## Artifacts

- [Machine-readable parity journal](2026-10-29-en-transformer-models-parity.json): before/after and canonical hashes, read ranges, section correspondences, 53 figure hashes, all code-block hashes, execution classifications, and source-native quotations.
- [CPU example runner](../../publishing/tools/en-transformer-models-check.py).
- [Regression tests](../../publishing/tests/en-transformer-models-parity.test.ts).

No staging, commit, push, or deployment was performed.
