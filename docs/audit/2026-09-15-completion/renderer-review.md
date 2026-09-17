---
type: editorial-review
status: bounded-pass
last_updated: 2026-09-15
---

# Independent renderer and publishing-adapter review

## Verdict

**Bounded PASS after three review findings were fixed.** No remaining P1/P2 was
found in the scoped renderer, explicit-anchor, or linked-course-artifact delta.
The current implementation is fail-closed for altered Harvard source payloads,
does not synthesize missing mathematical operators, does not change canonical
Markdown, and preserves literal fenced/inline code during render-only repairs
and artifact-link rewriting.

The review covers:

- `publishing/adapter/math.ts`;
- `publishing/adapter/markdown-prose.ts`;
- the explicit-anchor and `prepareLinkedCourseArtifacts` paths in
  `publishing/adapter/build.ts`;
- `site/astro.config.mjs`;
- the corresponding cases in `publishing/tests/build.test.ts`,
  `site/tests/source-math.test.ts`, and
  `site/tests/markdown-rendering.test.ts`.

It is not a review of every changed publisher test or of the whole site build.

## Findings raised and resolved during review

### P2 — duplicate native headings ceased to be ambiguous

The first explicit-anchor implementation reduced matching candidates to a set
of emitted fragments. Two native `## Repeated` headings therefore became one
set value (`repeated`) and were accepted, although Astro suffixes later duplicate
heading IDs and the previous publisher deliberately failed closed on ambiguous
headings.

The current code separately rejects more than one native candidate or more than
one explicit candidate, while still accepting the intended case of one native
heading plus one retained alias with the same fragment. Regression cases cover
duplicate native headings and duplicate explicit IDs. **Resolved.**

### P2 — lossy old projection was not a complete source-integrity guard

The old Harvard projection deletes arbitrary TeX commands. A source-only change
from `\sum` to `\prod` could therefore leave the lossy readable projection
unchanged while changing the restored formula. The review reproduced that
behavior on the retained source in the Vol. 2 distributed-training deck.

The current implementation additionally checks a route-keyed SHA-256 of the
complete retained TeX or Marimo payload, including the importer-preserved
terminal LF, before reconstruction. All 35 slide sources and the one compression
lab source match independent entries in the frozen
`labs-slides-manifest.json`. The same source-only mutation now returns the input
unchanged. The exact old-projection equality remains as the second independent
guard, so a readable-block edit also fails closed. **Resolved.**

### P2 — linked artifacts were rewritten inside literal code

`prepareLinkedCourseArtifacts` initially applied its Markdown-link regular
expression to the complete page body. A textbook example containing a course
`.py` link inside a fence or code span was consequently rewritten and caused an
artifact to be published.

The current `mapMarkdownProse` helper preserves matching inline-code runs,
backtick and tilde fences, longer closing fences, shorter nested-looking fence
runs, and unclosed fences through EOF. Four red-to-green build fixtures verify
that literal links remain byte-for-byte present and publish no artifact.
**Resolved.**

## Math-renderer assessment

The normalizer is scoped by the generated file URL to source-course routes,
including their EN projection. Canonical textbook/reference paths, unrelated
source routes, missing URLs, and unknown Harvard routes return the original
Markdown unchanged. A direct current check also confirmed canonical/no-URL
identity.

For Harvard slides and compression lab, the repaired expressions are recovered
only from the complete source already retained at the bottom of the same
archive page. Reconstruction requires both the frozen source-payload SHA and
byte-exact reproduction of the old readable projection. It does not execute
Python, infer a missing operator, or edit the complete original source block.
For the five notebook/script routes, transformations are exact lexical matches;
unknown text is retained. Display delimiters are moved to their CommonMark form
without altering the expression, and the other repairs escape existing literal
identifiers or replace an escaped currency dollar with its equivalent TeX
command.

The prose mapper and dedicated tests cover fence character/length, inline-code
delimiter length, and unclosed-fence behavior. Normalization is idempotent for
every tested source. Full frozen bodies retain their code fences and tail
sentinel, and the complete original source portion remains byte-identical.

The Astro integration wraps the configured renderer rather than weakening
KaTeX error handling. Render options and renderer metadata are forwarded. The
actual SHA-256 of `math.ts` is placed in serialized processor options so an
implementation or frozen-hash change invalidates Astro's content cache. Valid
Unicode mathematical labels on canonical pages are rendered rather than
treated as source-archive exceptions; tests still require a KaTeX result with no
error node and intact surrounding prose.

No Harvard, HSE, Linux Foundation, or Scikit-learn source-archive file is
modified in the current worktree. This review found no formula suppression,
error-class renaming, permissive KaTeX fallback, placeholder formula, or
canonical-source mutation in the scoped implementation.

## Explicit-anchor assessment

`parseMarkdownHeadings(..., true)` recognizes only standalone empty `a`/`span`
elements with a non-whitespace quoted ID and optional `aria-hidden="true"`.
The existing fence scanner excludes apparent anchors inside literal examples.
Fragment conversion reuses an already present alias or Astro's native heading
slug; it no longer injects an extra HTML node into generated chapter content.

The resolver accepts one exact native heading, one exact explicit alias, or the
deliberate one-native-plus-one-same-slug alias case. Duplicate native headings,
duplicate IDs, different resolved fragments, and fenced fake anchors remain
unresolved and cause the existing build failure. The reviewed changes therefore
preserve both stable old slugs and the previous ambiguity invariant.

## Linked-course-artifact assessment

A relative Markdown link from a non-course chapter can now point to an approved
artifact under one of the two configured course roots. Registered Markdown
pages still become publication routes. Raw Markdown artifacts require the
existing exact allowlist; other copied artifacts are limited to the declared
PDF/Python/JSON/YAML/text extensions.

Lexical containment is checked against the repository root and then against the
selected exact course root. Before cleanup, each selected artifact must exist,
and real paths of the course root and target are compared; both Stanford and
Berkeley symlink-escape fixtures fail before stale output is removed. Public
destinations are derived from fixed course route roots, path components are URL
encoded, destination collisions fail, repeated identical references deduplicate,
and only actually linked artifacts are copied. The new prose-only mapping
prevents code examples from becoming publication pointers.

## Executed verification

Current-file verification after all three fixes:

- `pnpm exec vitest run tests/source-math.test.ts tests/markdown-rendering.test.ts`
  from `site/`: **57/57 PASS**. This includes the real Astro entry-info/render
  path, 35 full Harvard slide decks, the full compression lab, five full
  notebook/script pages, RU/EN route scope, source/readable mismatch cases,
  idempotence, code preservation, and cache-key coverage.
- `pnpm exec vitest run tests/build.test.ts` from `publishing/`:
  **50/50 PASS**, including explicit-anchor ambiguity and artifact containment,
  symlink, cleanup, allowlist, prose/fence, copy, and deduplication cases.
- Direct `\sum` to `\prod` mutation of the retained distributed-training TeX:
  current normalizer returned the complete mutated input unchanged.
- Scoped `git diff --check` and `node --check site/astro.config.mjs`: **PASS**.
- Source-archive path status/diff check: no Harvard/HSE/LF/scikit-learn archive
  body changed.

Current reviewed implementation hashes:

| File | SHA-256 |
| --- | --- |
| `publishing/adapter/math.ts` | `56c15937886914a2d9a2a0b6f86cc152f5b6f91a5fa54b080706107dde3173c3` |
| `publishing/adapter/markdown-prose.ts` | `e85eaed65bf5bfb69f4e6acd3de392306eb08f4985a089f860e0cdb604631062` |
| `publishing/adapter/build.ts` | `58dc7f6fa6f82395773e6f5c789806b974c125630e1d28c3642a8538a6528d2b` |
| `site/astro.config.mjs` | `fefe2c8075a3ff0afa7ebcf5792007851b3da33e6ed50f7ac4f8fa52324adb9b` |

## Separate bounded Berkeley content reread

As a small independent supplement, the current files below were read completely
from frontmatter through EOF after the Berkeley follow-through changes:

- `00 Учебник/17 Tools и Agents/71 Агенты научного поиска и discovery.md` —
  391 lines, SHA-256
  `670e7741669b1da2d88ef7a763ed7877a6f43cf51645d4abbf0cf34e4b361e22`;
- `en/00 Textbook/16 Multimodal Models/64e Video, audio, and omni models.md` —
  202 lines, SHA-256
  `1975c1ee74a813104c013d89b9f40b53eb83772cb4ae4ffb2f59d6ed3272411d`.

No P1/P2 was found. RU 71 keeps proposal/evaluator/search boundaries, LaSR's
external numerical evaluation, ESCHER's learned-VLM-critic limitation, the
pseudo-confusion qualification, worked symbolic-regression split arithmetic,
original SVG attribution, and safety/experimental-contract boundaries coherent.
EN 64e is a real authored EN page (`locale: en`, `translation_of` present), not
a locale fallback; xGen temporal compression, GenS question-conditioned frame
selection, audio-code scheduling, streaming/full-duplex distinctions, worked
token budgets, figure provenance, and previous/next navigation are coherent.

This supplement is not a reread of the rest of either chapter family and did
not include browser visual inspection of their figures.

## Evidence boundary

No full site build, browser run, GPU execution, importer regeneration, commit,
push, or deployment was performed by this reviewer. Root owns the concurrently
running fresh full build and final generated-output acceptance; its result is
not counted among the independent checks above.

## Addendum: code-formatted Markdown link labels

**Delta-only PASS.** A later full-output check exposed that the first
`mapMarkdownProse` implementation treated the backticks inside
`` [`lecture_01.py`](...) `` as a standalone code span before the enclosing
Markdown link was passed to the artifact transformer. The link was split and
remained unresolved even though a code-formatted label is ordinary prose around
a real link target.

The current scanner gives a complete Markdown link token precedence over an
internal code span, then applies the caller's transform to that whole link. A
real outer code span still begins earlier in the input and is preserved in full,
so the whole link wrapped by an outer two-backtick span remains literal. Fence
handling is unchanged.
The ordering is deterministic and does not add placeholders or mutate the link
label.

The three positive fixtures cover a plain label, a backtick-formatted filename,
and a bold label containing a backtick-formatted filename. The separate
precedence fixture places the same code-formatted link both inside an outer
two-backtick span and in prose, requiring the first to stay byte-identical and
the second to become the publication URL. Existing backtick, tilde, longer-close,
shorter-inner-fence, and unclosed-EOF cases remain in the same test file.

Independent delta verification:

- the 17 Stanford lecture inventory links use the affected
  `` [`filename`](relative-artifact) `` shape;
- `pnpm exec vitest run tests/build.test.ts`: **53/53 PASS**;
- scoped `git diff --check` for `markdown-prose.ts` and `build.test.ts`: PASS;
- no other implementation or content file was reviewed or edited in this
  addendum.

Updated delta hashes, superseding the earlier helper hash in this report:

| File | SHA-256 |
| --- | --- |
| `publishing/adapter/markdown-prose.ts` | `5f50669e31a7d8a24d9ab8359b7c7aec280229ae0fe3d58e7a4a691919b6a865` |
| `publishing/tests/build.test.ts` | `cebfe1eb89c59491e3fd9acb39edb87d020d730e6174bcdf7819afa9e341f792` |

The fresh full build and confirmation that all 33 previously failing emitted
links are now resolved remain root-owned acceptance evidence and are not claimed
by this delta review.
