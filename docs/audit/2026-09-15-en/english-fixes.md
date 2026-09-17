# English editorial audit — 15 September 2026

## Outcome and scope

The owned English editorial pass is complete: **36 of 36 owned files read in full, 23 changed, 13 intentionally unchanged**. The initial authored-English inventory contained 37 Markdown files; `en/00 Textbook/_index.md` belongs to the primary agent and is excluded from these before/after checksums. The primary agent added `en/05 Sources/Courses.md` in parallel, raising the live inventory to 38; that new page is outside this existing-page audit. No fallback-only localization was created by this agent.

Seven existing VLM chapters now cover their Russian counterparts' causal explanations, notation, worked examples, diagnostic interventions, original source figures and limitations. Existing useful prose and every existing image embed were retained. This is an editorial result, **not yet final site/independent-review acceptance**.

Worktree: `/Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion`.

The initial SHA256 values were captured before this agent's first edits. Current values below reflect the final content handoff. Full 36-page records, reading status, Russian counterparts, findings, source-check scope and gaps are in [english-pages.json](english-pages.json). The primary index has no invented baseline in that inventory.

## Reading and comparison method

- Read all 36 owned English files completely, including the full original-answer archive and the five LMCache source notes. Where terminal output was truncated, continued reading the omitted ranges.
- Read all seven current Russian VLM counterparts completely. Read the preceding 38-page RU audit inventory and associated systems, advanced-model and question-answer findings under `docs/audit/2026-09-15/`.
- Compared claims, not filenames alone. The English scheduling chapter did not contain the Russian packed-token `sum(S)` mistake. Other earlier RU fixes—omitted FFN activation/current-token KV and missing Word2Vec/MHA/LoRA answers—were not mechanically inserted into unrelated English text.
- Re-read the expanded VLM pages after editing, and reviewed the other changed passages/diffs. The original archive was inspected as a separate historical layer.
- Checked niche architecture and figure claims against primary papers/official implementations. See the source ledger below. This was not a fresh empirical reproduction of all benchmark or model claims.
- Used only patches within owned files and these two reports. No publisher/site implementation, raw snapshot, asset, commit, push, merge, deployment, worktree, or site-build changes were made by this agent.

## Confirmed corrections outside the VLM expansion

### Architecture and metadata

Ten existing attention/architecture pages lacked `locale: en` and their exact `translation_of` mapping. Added the mappings from the navigation inventory, using the actual `source_en` entries.

The architecture-patterns chapter now distinguishes BERT's selected MLM loss positions from positions actually replaced by `[MASK]`: 15% selection, then 80% mask / 10% random / 10% unchanged. The loss conditions on corrupted input. The complete-Transformer equal-head trace is explicitly an MHA trace; GQA has fewer KV heads than query heads.

The positional chapter's T5-specific bias formula now uses unscaled query/key scores plus learned relative bias. The current official Transformers T5 implementation sets its attention scaling to 1; generic Transformer scaling remains unchanged elsewhere. The primary agent independently read and corrected the Russian counterpart with the implementation source; that synchronization is complete.

Existing date fields were updated on modified pages. The original-answer archive's frontmatter was deliberately **not** changed: the whole captured archive remains an exact prefix, and the appended errata has its own date. LMCache's older support matrix retains its explicit 2026-07-23 snapshot label despite this new editorial verification date.

### Inference, goodput and source figures

- Goodput is a rate, with per-time units, rather than an unqualified request count.
- TPOT's inter-token denominator is used only for output length greater than one; a one-token response has no inter-token interval. The exact latency identity requires shared token timestamps; separate response-completion events can add another interval.
- Poisson-distributed arrival counts were distinguished from exponential interarrival times.
- A completion throughput of 10 requests/s and joint SLO attainment of 90% yields 9 qualifying requests/s. This observed product is distinguished from the DistServe-style maximum admitted rate satisfying an attainment target.
- Three speculative-decoding illustrations are attributed to **Chen et al., Accelerating Large Language Model Decoding with Speculative Sampling**, not Leviathan et al. The latter remains a valid independent primary reference.
- The middle plot has lookahead K on the x-axis and accepted-output count divided by K+1, not acceptance probability at individual draft positions. The right plot measures the complete draft-plus-verification cycle, not verification alone.
- The Table 1 caption no longer claims statistical variation/significance from a table without uncertainty bars. Similar reported task metrics do not prove equality of output distributions.

### LMCache

Clarified computed state versus prefetched state, cache dependence on the current token and its history, and the overlap equation's clock as **remaining P/D tail latency**, not total request latency. Linked the verified official legacy deprecation warning. The documented Ultravox example is explicitly identified as legacy in-process evidence, not proof of current MP-mode feature parity.

### Historical question-answer archive

The entire initial UTF-8 archive—including frontmatter, wording, code, figures and all original question markers—remains byte-equivalent to the prefix of the current file. A separately headed **Editor's errata — 15 September 2026** follows it. The question index links that section.

Ten numbered errata groups address existing answer defects: TF–IDF convention/code, Bayes and class-conditional independence, SVM C direction, lemmatization/imported code artifacts, cosine geometry, classification denominators, perplexity comparability, BLEU, ROUGE, and recurrent-network parameter/gradient claims.

The literal English ROUGE candidate has **9 tokens**, not 10: `I really really loved reading reading the Hunger Games`. Its six-token reference yields precision 6/9, recall 1, and F1 12/15 = 0.8. This was caught and corrected by an executable token-count assertion before handoff. It is not a reinterpretation of an untranscribed screenshot.

The archive still contains **78 unanswered markers among 100 numbered questions**. These are missing source answers, not 78 newly discovered factual errors. No answers were invented to create parity with the newer Russian archive.

## Seven VLM chapters: explanatory coverage

Word counts here are rough whitespace counts including frontmatter/captions, used only to show the scale of change—not as acceptance criteria.

| Chapter | Before → after words | Added mechanism, example and diagnostic scope |
|---|---:|---|
| 64 Multimodal models.md | 797 → 1512 | Expanded to RU scope: perception/generation distinction, explicit patch projection shapes, CLIP loss/matrix, SigLIP normalization and source figure, architecture comparison. |
| 64a Connectors and fusion.md | 760 → 1493 | Fixed projector multiplication; added LLaVA-1.5 shape/budget trace, OneVision and DeepStack with source figures, second-stage BLIP-2, resampler worked shapes, gradient and compatibility boundaries. |
| 64b Resolution, tiling, and spatial positions.md | 563 → 1256 | Expanded RU parity: patch/merge budget table, AnyRes and Qwen2-VL source figures, 64 versus boundary tokens, interleaved axes, inverse-coordinate example, stage-specific diagnostics. |
| 64c Training VLMs — alignment, instruction tuning, and data.md | 537 → 1652 | Expanded RU parity: stage/objective table, gradient path, assistant loss masks, mixture arithmetic, LLaVA/OneVision/Qwen curricula and six additional inherited source figures (eight total), transfer ablation and reproducibility contract. |
| 64d Documents, OCR, and visual grounding.md | 496 → 1230 | Expanded RU parity with document architectures and Pix2Struct figure, invoice trace, table ambiguity, grounded counting, explicit coordinate convention, CER/WER/IoU examples and evidence/abstention contract. |
| 64e Video, audio, and omni models.md | 473 → 1435 | Expanded RU parity: event-aligned request, frame and codec arithmetic, projection versus sequence compression, OneVision/ImageBind source figures, codec objectives, Moshi text-first ordering, duplex/streaming and evaluation matrix. |
| 64f Evaluation, failure modes, and VLM serving.md | 535 → 1287 | Expanded RU parity: benchmark capability map, Winoground inequalities, POPE arithmetic, two-page admission example, 460ms TTFT trace, pre-LLM blocking, cache-layer identity, judge/security and fallback continuation. |

All pre-existing English figure embeds remain. Additional VLM figures are inherited existing assets already used in the Russian chapters; no new figure files were fabricated. Captions distinguish original-paper images from Stanford course reproductions and avoid treating figures as measured accuracy guarantees. The perception/generation split points to explicitly labeled RU fallbacks for diffusion/Chameleon and the practical, without creating unrelated English pages.

## Primary-source verification ledger

The sources below support targeted statements; a row is not a claim that every version-dependent feature in that ecosystem was tested.

| Source | What was checked |
|---|---|
| [BERT](https://arxiv.org/abs/1810.04805) | Selected MLM positions and corruption scheme; unchanged selected tokens can remain visible. |
| [Official Transformers T5 implementation](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/t5/modeling_t5.py) | T5 scaling=1 and relative bias placement; this mutable source should be revision-pinned for a publication snapshot. |
| [Chen et al., speculative sampling](https://arxiv.org/html/2302.01318v1) | Table 1 attribution, K-axis efficiency, full cycle timing; local source images visually inspected. |
| [SigLIP](https://arxiv.org/pdf/2303.15343) | Pairwise logistic objective; normalization by B, not B²; negatives remain. |
| [LLaVA-OneVision](https://arxiv.org/html/2408.03326v3) | SigLIP → two-layer MLP → Qwen2; AnyRes, staged training and modality budget differences. |
| [BLIP-2](https://arxiv.org/abs/2301.12597) | Q-Former stages and learned-query visual bottleneck. |
| [Qwen3-VL report](https://arxiv.org/html/2511.21631v1) | Interleaved M-RoPE, DeepStack, explicit timestamps, 8K/32K/256K curriculum and square-root-normalized per-token objective description. |
| [Pix2Struct](https://arxiv.org/abs/2210.03347) | Screenshot parsing with simplified structure targets; not generic text-only OCR. |
| [Moshi](https://arxiv.org/html/2410.00037v2) | Text-first head in Inner Monologue, temporal versus depth computation, semantic versus acoustic codebooks and full duplex. The 50 fps/eight-codebook arithmetic is explicitly illustrative, not Mimi's frame rate. |
| [Winoground](https://arxiv.org/abs/2204.03162) | Two text and two image inequalities required for group success. |
| [BLEU](https://aclanthology.org/P02-1040/) and [ROUGE](https://aclanthology.org/W04-1013/) | Brevity penalty direction, clipped counts and metric variants. |
| [Official SVM guide](https://scikit-learn.org/stable/modules/svm.html) | C is inverse regularization strength. |
| [FlashAttention repository](https://github.com/Dao-AILab/flash-attention) | Current FA4 scope includes Hopper/Blackwell; no EN text edit was needed. |
| [LMCache MP](https://docs.lmcache.ai/mp/index.html), [legacy mode](https://docs.lmcache.ai/legacy/index.html), [Ultravox legacy example](https://docs.lmcache.ai/getting_started/quickstart/multimodality.html) | Prefetch/transfer distinction, explicit in-process deprecation and provenance boundary for multimodal examples. |

Other primary references and original figure attributions remain in the chapter bodies and metadata. Newly written explanation paraphrases mechanisms in natural English rather than copying course narration.

## Executed verification

**Passed:** an inline Node.js assertion suite against all 36 captured original texts and current on-disk files:

1. Each captured original text hashes to its recorded initial SHA256.
2. Every current owned page has `locale: en`, an exact `translation_of` value and an existing RU target.
3. Every old image embed remains in each edited file; all edited files' image targets exist locally after resolving the vault prefix.
4. The current original-answer page starts with the exact original content; original and current both retain 78 `[!todo]` markers and the original 100 numbered question headings.
5. Checked patch/merge counts, projected-prefix lengths, inverse image coordinates, video frame budgets, audio feature/codec counts, mixture weights, IoU, Bayes substitution, literal ROUGE tokenization/F1, POPE F1, TTFT sum, goodput and the existing attention example.
6. `git diff --check -- en` exited 0 with no output.

Representative executed expectations:

```text
ViT 224/16: 196 patches + CLS = 197
LLaVA 336/14: 576 patches; +40 instruction +60 answer = 676 positions
Qwen spatial merge: 256 / 4 = 64 vectors
inverse box: (112,56,312,156) minus padding (12,6), divided by .5
             = (200,100,600,300)
tile point: ((120+500-20)/.5, (60+800)/.5) = (1200,1720)
video: 60*2*256 = 30,720; 60*8*256 = 122,880; /4 temporal pooling = 7,680
audio: 60/.02 = 3,000; illustrative codec 10*50*8 = 4,000
sqrt mixture: 70.6101%, 22.3289%, 7.0610%
IoU: 80/(100+120-80) = 4/7
Bayes: .8*.4/.86 = 16/43; original inverted substitution = 1.72
ROUGE literal candidate: 9 tokens; F1 = 2*(6/9)/(6/9+1) = .8
POPE F1: 80/110 = 8/11
TTFT: 20+80+35+120+10+180+15 = 460 ms
goodput: 10*.9 = 9 requests/s
```

The suite initially rejected an editorial draft that counted the ROUGE candidate as ten tokens; the draft and expectation were corrected to the literal nine-token result, and the complete suite then passed. No model training, benchmark execution or archived-example dependency installation was performed.

## Exact changed files and SHA256

Paths are repository-relative. Date-only/metadata changes are identified in the per-file findings. Full unchanged-file hashes are in the JSON inventory.

### en/04 Questions/100 questions about NLP — original answers.md

- Initial SHA256: `675230e0d2459894ba296cae00544f8f87fb1eb73688499c84e472d008da1966`
- Current SHA256: `7bfc73683cf4aeb1dd99c0ceaab6f5b36c835d62451830597254bc672dc50972`
- Preserved exact original archive; appended ten explicit errata groups covering actual existing answers and retained 78 unanswered markers.

### en/04 Questions/100 questions about NLP.md

- Initial SHA256: `131f8b00520e19acda29fe880d9de59d4a2b1bf2fbbb320f94a6c4339d179bc9`
- Current SHA256: `0d4bd9a207621f2dad0c18d458591bd02bda83364800203b793ef43e7ab7eb46`
- Linked separate English archive errata; retained original-answer versus unanswered distinction. Used source-heading wiki resolution.

### en/00 Textbook/05 Attention and Transformer/02 Self-Attention — Q, K, V.md

- Initial SHA256: `ea26c9f7db0532c506ebf05f65093cd46d0ac69176d5a3cfed6e0b7c621ae804`
- Current SHA256: `e100e0b09d40b8ee09ee6bd382a6c2e9572707b09b5184b92f3ee5397e4661b3`
- Added missing locale and exact navigation-mapped Russian counterpart.

### en/00 Textbook/05 Attention and Transformer/03 Masking, multi-head attention, and tensor shapes.md

- Initial SHA256: `edb535009b9aedf760614de74fa8c02038402606f12825c7331183cd94ccc9b9`
- Current SHA256: `1d07498b1ba18f1e450aba8390ab5ae04ba8b1c57ae3a82ee556ecad528fb5f7`
- Added missing locale and exact navigation-mapped Russian counterpart.

### en/00 Textbook/05 Attention and Transformer/04 Positional information.md

- Initial SHA256: `c727cea6a1d8e6ea8d5e2690840d47af73a9e8e3474badaf96028bf0b8a8dadb`
- Current SHA256: `67045dd4ca11aa5c442195576c0aad1b7ffab96b67b7299948f5785cc6bfd640`
- Added missing locale and exact navigation-mapped Russian counterpart. T5 bias formula now uses its unscaled query/key product. Added implementation source for scaling convention.

### en/00 Textbook/05 Attention and Transformer/05 The complete Transformer.md

- Initial SHA256: `a38494eab482e2502bdb7ab1534f9411180795090528590f1ad50d58083de9a8`
- Current SHA256: `b2d96e75c568d57c178374e6e58f2474ba8e51fb878872e5261903fdd1c5fe28`
- Added missing locale and exact navigation-mapped Russian counterpart. Scoped equal Q/K/V head trace to MHA and distinguished GQA.

### en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/01 Three architectural patterns.md

- Initial SHA256: `879ff2df8b5ff0bf435312b1c134943b6849e50481affa7d09323a82145a6e57`
- Current SHA256: `18beb8222ec5af93f58b26430ca0b51888eccd3b93fed5cbb2d964376e357ce5`
- Added missing locale and exact navigation-mapped Russian counterpart. Corrected selected-versus-hidden MLM positions and corrupted-input notation.

### en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/03 BERT, RoBERTa, and DeBERTa.md

- Initial SHA256: `2c524797a86db800c4c79f093faea99a6beccc42b12da68b9f9e4683623adf3b`
- Current SHA256: `534756abd309bf9469862e4492d459c639707e44bec7f51495867c9a5621a00c`
- Added missing locale and exact navigation-mapped Russian counterpart.

### en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/04 GPT-1 — generative pre-training.md

- Initial SHA256: `62530354e7dbc54e82b42b127cf3efd59cbdca5f7fa0f1129b9730d5e41995a1`
- Current SHA256: `ef937dcf4b0af32ed77be3e25bdc44389b594a4ace737f3f96359d19aca657ae`
- Added missing locale and exact navigation-mapped Russian counterpart.

### en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/05 GPT-2 — zero-shot through language.md

- Initial SHA256: `21f99813f068087de378815d5787f352ede6d8ddd717aab5c4a31a07e91237bb`
- Current SHA256: `70bf5a2c3b1f2db6501b550252a09901aa71e6ca93e01ce9ce6963ef36a27234`
- Added missing locale and exact navigation-mapped Russian counterpart.

### en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/06 GPT-3 — in-context learning.md

- Initial SHA256: `e72ac94c13d5c9ac855eaaaa15bf06b6b597f5c6e3bbd812535eff2bca11c68a`
- Current SHA256: `2d10254fcd15aeee1151a341b74a86703e4f9badd7fa0e71324dbaaeaaa726aa`
- Added missing locale and exact navigation-mapped Russian counterpart.

### en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/07 T5 — text-to-text Transformer.md

- Initial SHA256: `47f92e6023d357c4eed4a79c722125819477441e20f116279b0a9413923a4f0f`
- Current SHA256: `003721171a06d5efbf43b86e1037253ec258e6d10c4140b5c9003c789d9d4d86`
- Added missing locale and exact navigation-mapped Russian counterpart.

### en/00 Textbook/14 Inference and optimization/55 KV cache batching and PagedAttention.md

- Initial SHA256: `9bfe37c4120ed039fcfce0876e7771e9bb0c27d91deff602001d85a48b06cac5`
- Current SHA256: `28db7fc0b21edc7a227b5b66f61bf1f2fea55327bb1703d5a0cabcc2ee1b69c2`
- Restored per-time units in goodput definition.

### en/00 Textbook/14 Inference and optimization/58 Speculative decoding.md

- Initial SHA256: `baa8f7a40daa72440f0d6d6758f378a8c54aa7e432f4ecdd5c8e314345d0c9e3`
- Current SHA256: `98597274b2cd0446e71d1e23791e5fad2a567651efe772ffdd4330c3aa941c6f`
- Corrected three original figure attributions to Chen et al., while retaining Leviathan paper as an independent source. Corrected x-axis and efficiency interpretation. Corrected plot caption. Removed unsupported statistical-variation claim from Table 1 caption; no uncertainty bars or significance test are supplied.

### en/00 Textbook/14 Inference and optimization/58b Benchmarking SLOs and inference operations.md

- Initial SHA256: `35052a70f5d7ba624ef54f7ad7d0599efa27b2445886986d78e8bdcffb62231d`
- Current SHA256: `82ac891078568d66b3fa76ab8ae251ed291ae6756a0c9db4265d5b34df8a975f`
- Fixed TPOT N=1 edge, exact timestamp identity, Poisson arrival wording, and goodput/attainment arithmetic.

### en/00 Textbook/16 Multimodal Models/64 Multimodal models.md

- Initial SHA256: `af16af32017080b53e139ffe4c6747c8f8dc1b2ef65c0618f7f4c4d394756621`
- Current SHA256: `af9700d2b9cbe9d98095f506b98672e3ffafc86813909e1a90f54a94af4e4bc5`
- Expanded to RU scope: perception/generation distinction, explicit patch projection shapes, CLIP loss/matrix, SigLIP normalization and source figure, architecture comparison.

### en/00 Textbook/16 Multimodal Models/64a Connectors and fusion.md

- Initial SHA256: `734cb5010d5d0e5d11d7738683e662ebc446ec58d0e522fee088216fed74e983`
- Current SHA256: `abe04969c82c3cb2d543b555de62edb894f293c98c06a04477179758c1616d08`
- Fixed projector multiplication; added LLaVA-1.5 shape/budget trace, OneVision and DeepStack with source figures, second-stage BLIP-2, resampler worked shapes, gradient and compatibility boundaries.

### en/00 Textbook/16 Multimodal Models/64b Resolution, tiling, and spatial positions.md

- Initial SHA256: `5c0d3fa55833b10e22692bc234e7dd4fa7b08281cefe7aa334a7638276539e25`
- Current SHA256: `2777eb5c9d128e8b69871e51a8e3b8287d25a900050b0465662b44cf8f403f8f`
- Expanded RU parity: patch/merge budget table, AnyRes and Qwen2-VL source figures, 64 versus boundary tokens, interleaved axes, inverse-coordinate example, stage-specific diagnostics.

### en/00 Textbook/16 Multimodal Models/64c Training VLMs — alignment, instruction tuning, and data.md

- Initial SHA256: `9d41eacbca1b235dd2feb72ca39378201a7f789ac3bc3892a649033468ad916f`
- Current SHA256: `dfb581fa6daa2dee003f0a2806de06a145b20256e6bca109eee83014950dde6a`
- Expanded RU parity: stage/objective table, gradient path, assistant loss masks, mixture arithmetic, LLaVA/OneVision/Qwen curricula and six additional inherited source figures (eight total), transfer ablation and reproducibility contract.

### en/00 Textbook/16 Multimodal Models/64d Documents, OCR, and visual grounding.md

- Initial SHA256: `c35f14c16dd7c3e47b78a242b336208306371425f25c4799f37d73f3ac9b464e`
- Current SHA256: `b591ffa1d78e2470729efb8a422af74ed1a26bc93639bdce7a0335b2b9c3cf1d`
- Expanded RU parity with document architectures and Pix2Struct figure, invoice trace, table ambiguity, grounded counting, explicit coordinate convention, CER/WER/IoU examples and evidence/abstention contract.

### en/00 Textbook/16 Multimodal Models/64e Video, audio, and omni models.md

- Initial SHA256: `db125a97ef772e53979d86b2b08ed1df196268eac4e4ea7593425133857af0ad`
- Current SHA256: `256112e6354adf53bb521c363d44029ca125f821ca5b33c0cfc08a0f37f32a99`
- Expanded RU parity: event-aligned request, frame and codec arithmetic, projection versus sequence compression, OneVision/ImageBind source figures, codec objectives, Moshi text-first ordering, duplex/streaming and evaluation matrix.

### en/00 Textbook/16 Multimodal Models/64f Evaluation, failure modes, and VLM serving.md

- Initial SHA256: `eb1ffc3549c38e103da2b3dd2af6ff3bcecc17cad36ea0b7c3b446b5c6bce572`
- Current SHA256: `5e67947a261984fed23ba7ed0011e489c2b8204ce57338c46c702623aa733b22`
- Expanded RU parity: benchmark capability map, Winoground inequalities, POPE arithmetic, two-page admission example, 460ms TTFT trace, pre-LLM blocking, cache-layer identity, judge/security and fallback continuation.

### en/05 Sources/LMCache/LMCache — map of materials.md

- Initial SHA256: `6a5e0797cb454806f701c07fe2e408a76699d190bb6f5950dcc89b4af63ac689`
- Current SHA256: `00637916c5a6b9ed502442de397aaab77883dff6197a79f08b8a327c81b2192f`
- Corrected prefetch versus computation, current-token cache dependency, and labeled overlap equation as remaining tail latency. Linked verified legacy deprecation and scoped Ultravox example to in-process mode; no automatic MP feature-parity claim.

## Remaining issues and handoff boundaries

- Final independent editorial review, publisher build/checks, locale-route/anchor validation and browser rendering belong to the primary agent. This report does not mark those steps as passed.
- The main English textbook index is outside this agent's checksum and edit ownership; its audit is in the course-navigation report.
- All 78 original unanswered questions remain intentionally open. Historical code was preserved, with known defects described in errata; it is not an executable reference implementation.
- The LMCache support table is explicitly a historical 2026-07-23 snapshot. This pass did not prove every current backend/connector combination, security property or benchmark result. Deployment claims still require exact versions and measured end-to-end tests.
- Existing sources or external URLs can drift. Official implementation links such as T5 `main` are evidence inspected during this pass, not immutable release pins.
- Figure existence/preservation was checked mechanically. Three speculative source figures were visually checked for axes/captions; final rendering of every inherited VLM source figure remains part of the primary visual pass.
- Russian synchronization: the primary agent applied both the speculative attribution/axes/cycle-time correction and the T5-specific scaling correction to the RU counterparts. This agent did not edit those RU files.
- The seven VLM pages now cover the specified RU explanatory scope, but this is not localization of later diffusion/Chameleon/practice chapters; explicit RU fallback links remain intentional.

## Focused review refinements

The independent review requested two semantic refinements; both are now included in the current SHA256 values above:

- LMCache §6 defines the maximum of remaining prefill and pure KV transfer service as an **ideal-overlap lower bound**, not an equality for elapsed completion. A final chunk ready at 100 ms and needing 20 ms of transfer cannot arrive before 120 ms, even when total pure transfer work is at most 100 ms. Actual tail latency follows the readiness/transfer/synchronization critical path. The clock remains the start of overlap, not request arrival; additive handoff and queue terms assume sequential downstream stages. The linked [layerwise documentation](https://docs.lmcache.ai/kv_cache_optimizations/layerwise.html) is explicitly legacy in-process evidence, not MP feature parity.
- In chapters 64b and 64e, Qwen3-VL textual timestamps now explicitly **replace** Qwen2.5-VL physical-time-scaled temporal IDs, while interleaved M-RoPE remains. Checked against [Qwen3-VL §2.3](https://arxiv.org/html/2511.21631v1#S2.SS3); this is not described as merely adding text timestamps to the old time-scaling scheme.

The primary agent's visual inspection also identified inline-sized math where equations used standalone single-dollar lines. Converted exactly 44 such delimiters to double dollars across the seven VLM chapters, producing 22 display equations; inline math and every source image embed remain unchanged. The training chapter has **eight total figures: two retained and six additional inherited assets**.

Repeated the full baseline/metadata/image/archive/numerical assertion suite after these changes. The readiness example was additionally evaluated as a release-time-constrained two-chunk schedule: ready times [0,100] ms and service times [80,20] ms yield 120 ms completion versus the 100 ms ideal-overlap bound. Independent focused recheck and final rendered verification remain primary-agent coordinated. No build was run by this agent.
