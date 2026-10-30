# EDLS neural KV reference: 26 October 2026

## Delivered revision and preservation

`06 Практика/14 Собрать минимальный inference engine.md`: 783 lines,
SHA-256 `c47997e5b9601a3e8ef31f6a9d669286ff11d46399604d56a7e368c25ea9848a`.
The complete 433-line prior practice was read before editing; the final
783-line chapter was read in full. Removing the bounded new section recovers
prior accepted SHA `a6f5cfb5a68307f7a17c653a68a5830867fb3a5d3e03bd829fed112c71aae30e`.
The original EDLS assignment and Harvard examples were not rewritten.

The complete visible module is identical to both
`06 Практика/Code/edls_kv_reference.py` and the public downloadable file:
167 lines, SHA `ae4c58df80457d2f0f335dca5ebac4744b10d04dca2277059c146559db01cd09`.
The actual preview HTTP response for `/examples/edls_kv_reference.py` has that
same SHA. Its reader link was checked in the DOM. An ordinary Markdown link
is used because the wiki attachment resolver does not publish `.py` files.

## Sources, explanation and style

Complete EDLS Week 8 seminar/homework, engine/config and MIT license were read;
the unchanged files, source boundaries and hashes are recorded in
`05 Источники/Originals/edls-inference-2026-10-26/`. Selected complete installed
Transformers 4.57.1 forward/cache/config sections were checked. The original
homework paragraph remains in English and is compared with the notebook by
test. The new module is an attributed solution rather than purported unchanged
teacher solution. The preserved full MIT license is also served by the site.

The narrative explains prefill's first output, the pending token, `P+N-1`
cached positions, immediate EOS/one-token completion and zero-token requests.
It traces C with two cached tokens and B with five through width-5 temporary
padding: both append at physical column 5, while their logical positions are
2/5. Saving `[0,1,5]` rather than the first three columns keeps C's actual new
K/V. Each request receives independently owned compact tensors; row order is
rebuilt for every call. These mechanisms lead into the full visible code and
three-request neural example, not only an API definition.

The no-ai-slop review retained mathematical notation, required terminology,
meaningful distinctions, the English source paragraph and source examples.
Added prose states transitions and observed checks without generic importance
claims. It does not compress the explanation into definitions or label a CPU
correctness run a speedup benchmark.

Source bugs are explicit: request/cache row drift, the one-token extra step,
missing admission in the small seminar example, padding/RoPE confounds and
timers that include prefill or count requested rather than actual tokens.
Original files remain unchanged. Version and full-attention scope are stated;
no arbitrary Transformers 5.x/sliding-cache compatibility is claimed.

## Execution evidence

`publishing/tools/check_edls_kv_reference.py` passes using cached packages:

```sh
LC_ALL=C LANG=C uv run --offline --with torch==2.14.1 --with numpy==2.3.5 \
  --with transformers==4.57.1 python publishing/tools/check_edls_kv_reference.py
```

The checker executes the actual module and smoke fences from Markdown. The
22,720-parameter random Qwen3 runs on CPU, without a tokenizer/checkpoint
download. Across 316 prefix/cache comparisons it checks exact greedy IDs and
every K/V tensor against uncached full-prefix computation. The largest observed
absolute cache difference is `7.748603820800781e-07` within explicit tolerances.
These are 316 checked prefix states, not 316 different trained models.

Coverage includes all six initial orders of three requests, later reversal,
admission during another request's decode, 24 varied workloads, zero/one-token
budgets, first/second-token EOS, cancellation, context limits, empty/invalid
inputs and independent cache storage. A forward hook checks that finished
requests do not trigger an extra model pass. The smoke produces:

| Request | Output IDs | Final KV length |
|---|---|---:|
| A | 50 | 3 |
| B | 32, 27, 34, 28, 63 | 9 |
| C | 63, 59, 24 | 4 |

The existing five arithmetic/scheduler fences also pass on the current chapter,
including 300 scheduler fixtures. The checker now selects its five labeled
blocks separately from the neural checker. Wiki-link validation ignores fenced
Python, avoiding false links from nested array literals; prose links remain
checked.

## Figures and rendered page

Added one complete original HF KV-cache PNG, 1348×466, with author/source
attribution and pinned CC BY-NC-SA 4.0 dataset evidence. It is unchanged,
including its indexed-color palette. Existing original HF dynamic/continuous
batching PNGs remain in place; there are now three source illustrations.
The diagram omits head axes, which the caption distinguishes from actual GQA.

The PNG validator previously accepted only non-paletted 8-bit color types.
It now also validates indexed PNGs: PLTE structure, CRCs, inflate length,
PNG row filters and palette indices. The new test accepts the original image
and rejects absent palette, truncation and corrupt CRC. The source picture was
not recompressed to satisfy a test; existing integrity cases remain enabled.

Actual route:
`http://127.0.0.1:4330/practice/14-sobrat-minimalnyy-inference-engine/`.
Desktop 1280 px: source diagram 617×213, same aspect ratio, caption and full
worked table visible. Mobile 360 px: article width 313, no document horizontal
overflow, no KaTeX errors; both display equations fit. Seven tables measure
351/594/313/352/313/313/395 px inside 313 px scroll containers. The six-column
KV table was keyboard-panned to its attention-mask end. Code blocks scroll
within their containers rather than widening the page.

The mobile inline figure is too small for comfortable label reading. Its
viewer was checked at 4×, with left edge 0 and right-side output/legend reached
by keyboard panning. Full attribution remains in the viewer. This limitation
is recorded rather than claiming the thumbnail is fully legible.

Screenshots:

- `screenshots/2026-10-26-edls-neural-kv-desktop.png`
- `screenshots/2026-10-26-edls-neural-kv-mobile-zoom.png`
- `screenshots/2026-10-26-edls-neural-kv-mobile-table.png`

The EN route was opened: its explicit missing-translation notice remains,
and it includes the current neural section, image and download as fallback.
No translation was authored. The viewer was closed, viewport reset and RU
practice tab marked as the deliverable after inspecting the final build.

## Publication checks and remaining work

Final build: 1965 pages. 477 unit tests in 51 files and 135 built-output tests
passed. Link check: zero broken internal routes, fragments or files. Source
coverage passes: 210 matrix rows and 56 destinations with direct source links.
This structural coverage does not prove complete semantic course integration.
Logs are `/var/tmp/bookvar-edls-kv-{cpu,unit,assets,build,output,links,coverage}-2026-10-26.log`.

The EDLS and Bookvar gap matrices now distinguish this checked neural core
from remaining tokenizer/server/sampling, metrics and three-policy comparison.
No pretrained Qwen3-4B, CUDA/FlashAttention execution, GPU timing, HTTP service
or asynchronous workload benchmark was run. Those gaps remain open.

PPO/DPO/RLVR/GRPO hashes were checked against accepted revisions and match.
No RL rewrite, EN source edit, staging, commit, push or deployment was performed.
The broad textbook goal, complete course integration and remaining source/rights
findings are not declared complete by this unit.
