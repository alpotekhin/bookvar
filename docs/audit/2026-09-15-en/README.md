# English corrections and course navigation — 15 September 2026

## Result and boundaries

This batch corrects existing English content and makes imported courses easier to use alongside the textbook. It does **not** complete a translation of the entire book or certify that every useful detail in every course has been integrated.

- Initial authored-English inventory: 37 Markdown files. The editorial agent read 36 in full; the primary agent read and revised the English index separately.
- Changed 24 existing English files: 23 in the editorial pass plus the index. Added one English course-reading map. Current authored-English inventory: 38 files. Some existing changes are metadata corrections, not chapter rewrites.
- Expanded seven VLM chapters with mechanisms, tensor shapes, worked examples, diagnostic tests, limitations and original source figures. Their 29 image embeds all load in the browser. Existing figure embeds were preserved; no generated illustrations were substituted.
- Corrected 22 VLM equations from standalone single-dollar notation to display math. Inline equations were left intact.
- Preserved the original English NLP-answer archive as an exact prefix; appended ten editorial errata groups. Its 78 unanswered original questions remain unanswered.
- Added two tightly scoped RU parity fixes after reading both complete chapters: T5's unscaled query/key product in the positional-information chapter; original-paper attribution and axis/measurement explanations for three speculative-decoding figures.

All work remains local in `editorial-route-alignment`. HEAD was and remains `e77f9a027e66f6719db486d2f31d9c08b3ae1d2f`. No commit, push, merge, deployment, source-snapshot rewrite or course-ledger regeneration was performed. The worktree already contained substantial earlier changes; the whole working-tree diff is not attributable to this batch.

## Detailed evidence

- [English page inventory and hashes](english-pages.json)
- [English corrections, examples and primary sources](english-fixes.md)
- [Course-state audit and bounded source-to-chapter sample](course-state.md)
- [Machine-readable course state](course-state.json)
- [Course-navigation changes and exact file list](course-navigation-fixes.md)

The primary agent personally read all seven expanded English VLM chapters, the English index and edited course maps, reviewed the other EN changes, checked source claims and inspected rendered pages. A separate reviewer examined the full 23-file EN diff. This is not a claim that the primary agent personally reread every unchanged English page.

## How courses fit into the book

The existing Obsidian structure remains canonical. The LLM Wiki source-to-topic approach was used without introducing a new vault layout:

1. Keep each original course, lecture, figure and assignment in the source collection, in its original language.
2. Use the bilingual Courses map to move from a topic to a textbook chapter and its original course material.
3. Extend the chapter when a source contributes a missing explanation, example, figure or limitation; do not insert another independent lecture sequence into the middle of the textbook.
4. Keep exact assignment contracts, repeated introductions, course administration and useful reference-only material in the archive. List genuinely deferred topics separately.
5. Treat coverage-ledger markers as evidence of destinations, not proof of explanatory completeness. Accept content only after reading the resulting chapter and checking the figures and formulas.

The maps now expose these distinctions for Stanford CS336, Berkeley Advanced LLM Agents, Efficient DL Systems, Harvard ML Systems, and HSE. The SHAD legacy notes are not advertised as a complete archived course. Stanford's generated hub was left untouched because its activation status was already current; the authored Berkeley hub was updated while preserving all lecture anchors.

Important count qualifications: Stanford's 554 integrated semantic units include 373 assignment units and 181 lecture units. Berkeley's 137 include 102 slide-derived units and 35 reading-catalogue entries, not 35 fully reproduced papers. These counts are not percentages of educational completeness.

## Independent review

The English reviewer requested an important LMCache correction: `max(remaining prefill, pure transfer service)` is only an ideal-overlap bound when KV chunks have readiness dependencies. The corrected explanation gives a concrete 100 ms readiness + 20 ms final-transfer example and distinguishes the tail clock from total TTFT. The reviewer also requested explicit wording that Qwen3-VL textual timestamps replace Qwen2.5's physical-time-scaled position IDs, while interleaved MRoPE remains. Both findings were fixed and the focused recheck passed; all 36 owned current hashes were verified.

The course reviewer requested the correct CS336 architecture lecture (3), explicit placement of the optional supplement inside Assignment 5, and exact edited-file/check records. All were corrected; the focused content/spec recheck passed.

## Executed technical checks

| Check | Final result |
|---|---|
| Publisher unit tests, `pnpm test:unit` | 181 passed, 15 files; 15:14:19 |
| Site unit tests, `pnpm test` | 6 passed, 2 files; 15:07:09; site code unchanged afterward |
| Full local site build, `pnpm build` | Passed; 1938 pages, 18.22 s; finished 15:15:26 |
| Built-output tests, `pnpm test:output` | 11 passed; 15:15:47, after final content build |
| Built links, `pnpm check:links` | 0 broken internal routes, fragments or files |
| Scoped `git diff --check` | Passed |
| English baseline/image/archive/arithmetic assertions | Passed; see detailed report |

The course audit also ran the existing Stanford/Berkeley importer checks and source-coverage/course-ingestion validators successfully; see its report for command scope. Those checks do not prove that all original lectures have been pedagogically incorporated.

Two content errors were caught before the successful final build: new wiki links targeted HTML-only anchors, and a course map linked to an unpublished `coverage.yml`. Both were fixed in the authored Markdown. The build still reports the pre-existing `Entry docs → 404 was not found` warning; it exits successfully. No unrelated implementation change was made to suppress that warning.

## Browser checks

Local static build served only on `127.0.0.1:4330`. Temporary viewport overrides were reset afterward.

- All seven `/en/textbook/multimodal/` pages: `models`, `connectors-fusion`, `visual-tokenization-resolution`, `training-data`, `documents-ocr-grounding`, `video-audio-omni`, `evaluation-serving`. DOM/layout checks at 1280 px and final checks at 390 px: 29 image embeds load, no KaTeX errors, no page-wide horizontal overflow. Final display-math counts are 6/3/2/3/4/1/3, totaling 22. Wide equations have their own horizontal scroll area.
- Screenshots inspected: EN Courses at 1280 and 390 px; Berkeley hub at 1280 px; CLIP loss before/after display-math correction at 1280 and 390 px; CLIP original figure on a narrow screen; LLaVA training-data figure at 1280 px. The displayed figures preserve their source aspect ratios. Links to full-size originals exist.
- EN textbook index: fourteen curriculum entries and explicit partial-English/fallback explanation verified in the DOM.
- RU and EN Courses: final titles, topic links and page layout verified; no horizontal page overflow. Berkeley retains all twelve `meeting-*` anchors.
- EN speculative decoding: three loaded images and no math errors.
- EN LMCache: corrected ideal-overlap explanation and no math errors at the actual `/en/sources/lmcache/` route. A test initially used the wrong `/index/` URL; that was a test URL error, not a broken emitted link.
- EN NLP questions: clicked Editor's errata and verified the destination heading/fragment; no KaTeX errors there.

These are explicit layout/rendering checks plus selected screenshot inspections, **not a screenshot-by-screenshot visual acceptance of every page or every source figure**.

## Remaining work

- Existing RU-wide audit findings outside the preceding correction batches remain open. The entire textbook is not declared finished.
- English remains partial. Missing EN chapters retain labeled original-language fallbacks; diffusion, Chameleon and their practicals were not newly translated here.
- The course-state report identifies narrow deferred topics (including selected Stanford MTP/attention/FP4 material and Berkeley video-agent/concept-discovery material). They need source comparison and a coherent destination chapter, not bulk copying into new stubs.
- HSE and older SHAD content need a stronger section-by-section semantic map before making whole-course transfer claims.
- Source-code URLs on mutable branches should be pinned when preparing a publication snapshot. No model training or empirical reproduction of cited results was performed.

No remaining blocking issue from this batch's independent reviews or final technical checks was left unreported.
