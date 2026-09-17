# English corrections and course navigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Follow task checkboxes. Do not commit.

**Goal:** Bring existing English pages into the correction workflow, repair confirmed errors, and make the recently imported courses navigable through thematic textbook destinations without confusing import coverage with editorial acceptance.

**Architecture:** Preserve the current Obsidian layers and existing routes. English authored pages in en/ and generated Russian fallbacks are distinct. Course archives remain immutable; reader-facing course maps explain what is available and point into the textbook.

**Tech Stack:** Markdown, existing source-unit/coverage/visual ledgers, Astro/Starlight, Vitest.

**Spec:** AGENTS.md; docs/superpowers/specs/2026-09-04-stanford-berkeley-ingestion-design.md, amended by the user's 2026-09-15 authorization to correct English too.

## Global Constraints

- Preserve all existing dirty changes in editorial-route-alignment. HEAD at start is e77f9a0.
- Do not commit, push, merge or deploy. Do not create another worktree.
- The prior instruction not to edit English is superseded for this task.
- Do not turn all generated fallback pages into purported English translations.
- Keep strong original English and the original source figures; do not replace them with generated diagrams.
- Read each edited chapter completely and retain its explanatory substance.
- Do not edit immutable raw files, pinned course artifacts, audit contracts, imported original-answer bodies or course snapshots.
- Source integration claims require a destination; acceptance also requires actual content and visual review. Do not mark all rows accepted mechanically.
- Do not spawn further subagents from a delegated task. The primary agent reviews the results.
- Reviews use per-task working-tree snapshots, not HEAD-to-HEAD diffs: commits are not authorized.

### Task 1: Correct authored English pages

**Files:** Own existing files under en/ except en/00 Textbook/_index.md (Task 3); own docs/audit/2026-09-15-en/english-pages.json and english-fixes.md. Read docs/audit/2026-09-15/revised-pages.json and associated RU findings. No publisher/code ownership in this task.

**Interfaces:** English pages use locale: en and translation_of to identify their Russian counterpart. Publishing uses sourceEn from publishing/navigation.yml. Generated fallbacks are not authored EN.

- [x] Inventory the 37 existing EN Markdown files. The primary agent handles the index; read all other 36 files in full, including the question archive and LMCache source notes.
- [x] For each file record initial SHA256, reading status, corresponding RU page where present, concrete findings, changes, source verification and unresolved gaps.
- [x] Propagate applicable confirmed fixes from the 38-page RU batch into authored EN. Check scheduling budget, dimensions/heads, projector and audio roles, inference metrics, source attributions and prerequisite claims. Where an English file does not contain the faulty claim, do not invent a change.
- [x] Correct other definite mathematical, technical or internal-consistency errors found in those EN files. Verify niche/version-specific assertions against primary sources.
- [x] Bring the seven short existing VLM pages up to the explanatory scope of their current RU counterparts: mechanism, notation, worked or diagnostic example, meaningful original figures, limitations and primary sources. Write natural English, not a literal translation of awkward Russian. Preserve useful existing text. Do not create new localizations for unrelated fallback-only chapters.
- [x] Preserve the original question-answer archive body. If it contains confirmed errors, put corrections in a clearly separate editor's errata section or separate linked note; distinguish unanswered original questions from corrections.
- [x] Re-read changed pages, verify examples that can be checked locally, record honest remaining gaps. Do not call a rewritten page accepted solely by word count.
- [x] Report the exact changed files and verification evidence in english-fixes.md; send the primary agent only the report path and concise status.

### Task 2: Establish actual course integration state

**Files:** Read-only across course archives, ledgers, destination chapters and existing publisher checks. Own only docs/audit/2026-09-15-en/course-state.md and course-state.json.

**Interfaces:** Use existing source-manifest.yml, source-units.yml, coverage.yml, visuals.yml, editorial-map.yml and snapshot-lock.json. Do not mutate or regenerate these files.

- [x] Inspect Stanford CS336 Spring 2026 and Berkeley Advanced LLM Agents Spring 2025 ledgers and their current hubs.
- [x] Count actual dispositions, editorial-map activation and destinations; distinguish source units from slide/page counts. Cross-check hub claims with actual data.
- [x] Read a concrete sample of integrated source sections and their target prose/figures to test what the ledgers establish. Inspect source-only reasons to distinguish intentional reference material from unfinished integration. Clearly label the sample, not a full re-audit of all slides.
- [x] Inspect the source hubs/manifests of Efficient DL Systems, Harvard ML Systems and HSE/SHAD materials added earlier. Identify exact source-to-textbook relationships already present and remaining reader-facing gaps.
- [x] Recommend thematic destinations and a concrete reader-facing course-map structure. Preserve strong source originals and useful figures. Do not propose a new short page per lecture.
- [x] Report evidence, counts, specific stale claims, sample limitations and a prioritized integration queue.

### Task 3: Implement clear bilingual course navigation

**Files:** Own en/00 Textbook/_index.md, reader-facing source/course hub Markdown and a course-reading map in the existing source layer. Modify publishing/navigation.yml only if a new reader-facing route is needed; preserve all other entries. Use exact paths discovered in Task 2. Do not change protected/generated ledgers or import snapshots.

**Interfaces:** Consume Task 2 course-state.json/md. Link to existing stable chapter routes via canonical wikilinks. Preserve existing source-hub anchors used by coverage rows.

- [x] Read the full course hubs before editing. Replace obsolete integration status and maintainer jargon in the reader-facing introduction.
- [x] Provide a thematic map with course offering, main topics, textbook destinations, original material and what remains to integrate. Distinguish structural mapping from substantive acceptance.
- [x] Preserve original lecture/assignment inventory and every anchor; move detailed maintainer information below the reading map rather than discarding it.
- [x] Update the English textbook index to the current fourteen-part curriculum and link the course-reading map. Do not present untranslated destinations as native EN.
- [x] Validate Markdown/frontmatter and internal links with the existing publication pipeline. If an importer owns a hub, adjust its authored template instead of making a change that regeneration would erase.
- [x] Record exact edited paths and verification. Send the primary agent the report and concise status.

### Task 4: Review and verify the combined result

**Files:** docs/audit/2026-09-15-en/README.md, task-scoped review packages and final status.

- [x] Review Task 1 and Task 3 diffs independently for factual accuracy, source language, preservation of figures and specification compliance.
- [x] The primary agent personally checks changed claims, representative full chapters, course mappings and review findings.
- [x] Run publishing unit tests, site tests, full local build and built-output link checks.
- [x] Inspect changed EN page types and course maps in the browser at normal and narrow widths. List inspected pages explicitly.
- [x] Record actual counts, corrections, unresolved content gaps and remaining full-book audit work. No blanket claim of complete translation or complete course transfer.
