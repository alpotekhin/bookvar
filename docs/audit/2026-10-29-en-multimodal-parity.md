# English multimodal reconciliation — 2026-10-29

The seven existing English chapters were read against their complete current
Russian counterparts and expanded in place. `current` now refers to the exact
canonical SHA256 recorded in each chapter, not to the presence of English
frontmatter. Canonical Russian chapters, source assets, shared registries,
central English ledger, navigation configuration, and publisher were not edited
by this pass.

## Delivered scope

All paths below are under `en/00 Textbook/16 Multimodal Models/`. Line counts use
`text.split("\n")`, matching the earlier reconciliation ledger. They describe
the revision, not a quality score.

| Chapter | Lines before → after | Original figures before → after | Executed Python examples |
|---|---:|---:|---:|
| `64 Multimodal models.md` | 157 → 533 | 3 → 6 | 6 |
| `64a Connectors and fusion.md` | 148 → 409 | 5 → 7 | 6 |
| `64b Resolution, tiling, and spatial positions.md` | 126 → 420 | 5 → 6 | 9 |
| `64c Training VLMs — alignment, instruction tuning, and data.md` | 178 → 438 | 8 → 8 | 8 |
| `64d Documents, OCR, and visual grounding.md` | 132 → 486 | 2 → 6 | 12 |
| `64e Video, audio, and omni models.md` | 206 → 608 | 4 → 10 | 17 |
| `64f Evaluation, failure modes, and VLM serving.md` | 134 → 562 | 2 → 7 | 17 |

The machine-readable companion, `2026-10-29-en-multimodal-parity.json`, records
before/after hashes, canonical hashes, each example's execution result, all
figure paths, retained fragment aliases, and a topic-by-topic reconciliation
inventory for every pair. Baselines were captured from the previous
`2026-10-29-english-reconciliation.json` before these edits; its historical
values were not overwritten.

## What was restored

The first three chapters now explain patch layout and projection, CLS and
position behavior, both CLIP loss directions, prompt ensembles, SigLIP bias and
distributed comparison, visual-prefix labels and cache timing, all three
Q-Former masks, Flamingo's zero gates and last-image mask, and the distinction
between projector, resampler, merger, and DeepStack. Resolution includes the
actual Molmo/AnyRes/InternVL/Qwen/Pixtral preprocessing calculations, not merely
a list of model names.

Training includes an executed projector update, shifted target loss, independent
packing versus shared-image branches, collection and teacher inputs for PixMo
and LLaVA, source-document leakage, sampling versus token mass, and target-length
loss normalization. The original OneVision demonstration remains visible with
its arithmetic error explained. Qwen training schedules, SAPO attribution, and
response/logit distillation retain the canonical version limits.

Documents include LayoutLMv3's distinct MLM/MIM/WPA objectives, Donut's
serialization and OCR-derived training labels, Pix2Struct's selected subtree and
reading curriculum, layout-preserving extraction, chart scales, coordinate
units and rotations, ANLS thresholds, calibration, and interventions with
explicit expected behavior.

Video/audio includes timestamp arithmetic, xGen and GenS, STFT and codec
quantization, Mimi's parallel semantic/acoustic branches, Qwen2.5 versus
Qwen3-Omni, Moshi's two autoregressive axes and delayed codes, causality,
lookahead, playback deadlines, and interruption at the audible boundary.
Evaluation/serving includes the original MM-Vet rubric, Winoground/POPE/CHAIR
details, paired counterfactuals, MMStar caveats, complete client TTFT, EPD
transfers and allocation, cache identity and UUID trust scope, and joint
quality/deadline counting.

All 50 canonical figure placements use existing source images. No generated
diagrams or new downloaded assets were substituted. Original SigLIP English
pseudocode and the MM-Vet English table were retained verbatim. Russian strings
remain only in explicitly identified original test fixtures where changing the
characters would change the CER or interval example.

## Verification

Executed offline in the existing cached NumPy 2.3.5 / PyTorch 2.14.1 environment:

```sh
LC_ALL=C LANG=C uv run --offline --with numpy==2.3.5 --with torch==2.14.1 \
  python publishing/tools/check_en_multimodal.py \
  --output docs/audit/2026-10-29-en-multimodal-parity.json
```

Result: seven chapter pairs, 75 of 75 published Python blocks passed. Their AST
multisets equal the canonical examples; translated comments and fence spelling
do not affect this check. The checker also verified canonical hashes, original
figure presence, explicit source aliases, unique IDs, filled example markers,
balanced display delimiters, and removal of obsolete bare chapter-number link
labels. All previous/next destinations remain unchanged.

`publishing/tests/en-multimodal-parity.test.ts`: five tests passed. These checks
pin the evidence to the edited files and preserve specific technical corrections;
they are not an automated judgment of prose quality. The no-ai-slop checklist was
applied to the completed prose while retaining mechanisms, limitations, original
quotations, and worked examples.

## Acceptance boundary

This is English parity for these seven chapters, not completion of every English
page or transfer of every Stanford/Berkeley course. No model weights were
downloaded, no trained VLM was evaluated, and no GPU-performance claim is made.
The examples are deterministic calculations and small CPU tensor computations.
The parent owns integrated build, rendered desktop/mobile checks, and central
ledger integration. They are not reported as completed by this local audit.
No commit, publication, or deployment was performed.

## Parent integration

The parent subsequently reflowed 18 display equations in six files for a
360-pixel viewport. The table and JSON after-hashes above include that layout
change; `before_render_sha256` preserves each earlier editorial revision.
Reversing only the recorded formula changes reconstructs those revisions
exactly. The 75 actual CPU blocks passed again after layout edits.
All seven routes passed the final two-width browser check. Selected Molmo
figures were inspected with their captions and mobile zoom controls.
See [the combined acceptance](2026-10-29-english-reconciliation.md) and
[exact reflows](2026-10-29-en-parity-render-reflow.json).
