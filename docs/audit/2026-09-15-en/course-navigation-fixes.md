# Course navigation corrections

Task 3, 2026-09-15. Working-tree only; no commits, pushes or deployments.

- Reused sources/kursy for a topic → Bookvar chapter → original course map. Added its separate English source; no new sidebar item.
- Updated the English textbook index from seven to fourteen curriculum parts. Replaced complete/fully edited claims with explicit partial-English availability. Fallback indicates a missing locale page, not a failed editorial review.
- Corrected Berkeley activation and visual counts (47/87), separated slide-derived and reading-catalogue units, and placed topic links before audit internals. Preserved old anchors and lecture inventory.
- Corrected the HSE map's obsolete no-mirror claim. Archive presence is explicit; whole-course semantic transfer remains unverified.
- Exposed original-language archives, practice routes, deferred topics, and the distinction between source-only reference material and missing explanations. Stanford assignment-heavy counts are not presented as lecture coverage.
- Linked the map from the RU/EN textbook indices and root source index. No source snapshots, ledgers, importer templates or contracts changed. Stanford's generated hub remains untouched; its activation statement was already current.

Existing broader course suggestions remain reading options rather than claims of complete ingestion. Structural status is a current local-worktree snapshot, not a statement about the deployed version. Final tests/build/browser evidence is recorded in README.md: 198 tests pass, the 1938-page local build succeeds, and the built-link check reports zero broken routes, fragments or files.

## Exact edited paths

1. `en/00 Textbook/_index.md`
2. `en/05 Sources/Courses.md` — new authored English page for the existing route.
3. `05 Источники/Курсы.md`
4. `05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/_index.md`
5. `05 Источники/Source maps/HSE ML course — link map.md`
6. `00 Учебник/_index.md` — only the course-map introduction link belongs to this task.
7. `05 Источники/_index.md`
8. `publishing/navigation.yml` — only the new `source_en` mapping for `sources/kursy` belongs to this task; other changes predate it.

## Review and checks

The independent course-state reviewer found two imprecise references. Corrected Stanford architecture to lecture 3 (not lecture 2) in both maps; the optional assignment supplement is now explicitly part of Assignment 5, not a sixth assignment.

The first build exposed wiki fragments aimed at hand-authored HTML anchors. Replaced those with exact Markdown heading references so both Obsidian and the publisher can resolve them. A later built-output test found a relative link to a non-published `coverage.yml`; replaced it with the reader-facing Stanford hub plus a plain-text description of the maintainer file. No resolver or importer code was changed to accommodate these content mistakes.

Browser checks on the local build: EN Courses at 1280 and 390 px; Berkeley hub at 1280 px. Topic tables render, no page-wide horizontal overflow was observed, and all twelve Berkeley meeting anchors remain in the DOM. The English map links directly to the corrected lecture-3 heading. Final post-correction test/build results and the complete inspected-page list are in `README.md`.
