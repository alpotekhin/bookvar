# Close Bookvar editorial findings

> Use bounded parallel domain work (superpowers:dispatching-parallel-agents) with independent review; no commits or publication.

**Execution refinement:** after verifying the 55 foundations, 60 systems and 40 advanced page sets have no overlapping paths, these three content passes may run concurrently. Existing EN ownership follows each RU page's manifest mapping. Shared navigation, source registries, rendering, question archives and practice remain with the primary. This replaces the initial serial-implementer scheduling only; snapshots, per-finding evidence and independent acceptance gates remain mandatory.

**Goal:** Resolve the outstanding page-level audit findings in the existing canonical book and authored English counterparts, and correct the course-integration and rendering gaps identified by the preceding audits.

**Architecture:** Preserve Obsidian Markdown, stable routes, original sources and figures. Work from full pages and the concrete 442-finding baseline, not keyword replacement. Keep archived answers immutable and distinguish their errata from canonical answers.

**Tech Stack:** Markdown, JSON evidence ledgers, Astro/Starlight, existing Vitest and course validators.

**Spec:** AGENTS.md; docs/superpowers/plans/2026-09-15-page-by-page-editorial-audit.md; latest user instruction to fix all remaining issues, including existing English content.

## Global Constraints

- Preserve every existing dirty change. Work only in the existing editorial-route-alignment worktree.
- No commit, push, merge, deploy, new worktree, or alteration of immutable source snapshots.
- The old no-EN constraint is superseded. Correct existing EN counterparts; missing translations remain explicit fallbacks, not invented completed translations.
- Read every assigned current page completely. Record one outcome per original finding: fixed, already-fixed, or rejected-with-evidence. Unresolved findings remain open, not silently closed by tests.
- Preserve useful explanations and all existing source illustrations. Prefer original course/book/paper figures; do not fabricate substitute diagrams. New scientific explanations cite primary sources; direct reuse respects source terms.
- Repair prose in context, not by global substitutions. Keep API and method names, translate ordinary English nouns in Russian prose. Strong original English course text stays English.
- Do not remove stable anchors/source-unit IDs, reference links, meaningful examples, or assignment contracts.
- Each worker owns only its assigned content and report/snapshots. No nested agents. Primary coordinates independent review and full builds.
- Capture pre-edit content for changed pages in task-local snapshots using apply_patch, so the review does not conflate older dirty edits.
- Per-page record: path, before/after SHA256, full-read status, each finding outcome and evidence, changed mechanisms, source URLs, retained/added figures, executed example checks, remaining limitations.

### Task 1: Foundations, neural networks and basic NLP

**Files:** The 55 exact page paths in docs/audit/2026-09-15/foundations.json and their existing EN counterparts identified in publishing/navigation.yml. Report: docs/audit/2026-09-15-completion/foundations.json and foundations.md.

- [ ] Read the 55 current pages and all 102 original findings, reconciling prior fixes.
- [ ] Resolve the remaining factual, numerical, explanatory and language findings in complete chapter context. Use the referenced HSE/D2L/Stanford/primary sources and preserve original figures.
- [ ] Re-read changed pages; verify worked arithmetic/code; update existing EN counterparts where applicable.
- [ ] Record individual outcomes, changed-file snapshots and verification, then obtain independent review.

### Task 2: LLM architecture, systems and inference

**Files:** The 60 exact page paths in docs/audit/2026-09-15/systems.json and existing EN counterparts. Report: completion/systems.json and systems.md.

- [ ] Resolve all 88 original findings against current full pages, including units, pipeline schedules, kernels, attributions, prerequisites and prose.
- [ ] Verify relevant primary sources and executable examples, retain figures and anchors, synchronize existing EN.
- [ ] Re-read changes, record individual evidence, obtain independent review.

### Task 3: Post-training, RAG, multimodality and agents

**Files:** The 40 exact page paths in docs/audit/2026-09-15/advanced.json and existing EN counterparts. Report: completion/advanced.json and advanced.md.

- [ ] Resolve all 55 findings against current complete pages; do not redo the completed seven-EN-VLM expansion.
- [ ] Check reward/verification and deployment assumptions, expand missing explanations and examples with original-source figures.
- [ ] Re-read and verify changes, synchronize EN, record outcomes and obtain independent review.

### Task 4: Reference, model atlas and research overviews

**Files:** Entries in docs/audit/2026-09-15/reference-practice.json whose paths start with 01 Справочник, 02 Атлас моделей or 03 Исследовательские линии. Report: completion/reference-models.json and reference-models.md.

- [ ] Read complete pages and resolve their original findings, replacing incomplete mechanism descriptions without duplicating textbook chapters.
- [ ] Check generation-specific architectural claims against primary reports/configurations; date and qualify scope rather than imply exhaustive latest-release coverage.
- [ ] Preserve strong figures, verify examples, record outcomes and obtain independent review.

### Task 5: Questions and practice

**Files:** All nine paths in questions.json and reference-practice.json entries under 06 Практика. Reports: completion/questions-practice.json and questions-practice.md.

- [ ] Preserve original answer archives; resolve erroneous source answers through explicit errata and links to canonical explanations in both existing locales.
- [ ] Ensure questions lead to substantive answers; distinguish absent source answers from actual corrections.
- [ ] Resolve practice findings in complete-page context, verify small examples and distinguish adapted exercises from official contracts.
- [ ] Record outcomes and obtain independent review.

### Task 6: Course gaps and navigation

**Files:** Remaining navigation entries in reference-practice.json; course maps/hubs; publishing/navigation.yml; relevant existing topic chapters. Reports: completion/courses-navigation.json and courses-navigation.md.

- [ ] Resolve remaining navigation findings and verify previous/next sequence and fourteen-part menu against contents; add focused regression coverage if needed.
- [ ] Work through the concrete deferred queue in docs/audit/2026-09-15-en/course-state.md. Integrate useful stable content into existing topic chapters with original figures; keep administration, duplicates, unsupported future claims and source contracts in the archive with reasons.
- [ ] Reconcile source-to-topic evidence without changing imported originals or equating catalogue entries with complete paper explanations.
- [ ] Record actual remaining boundaries and obtain independent review.

### Task 7: Combined acceptance

**Files:** docs/audit/2026-09-15-completion/README.md and combined findings ledger; focused publisher/site tests if defects arise.

- [ ] Reconcile all 442 original findings with per-finding outcomes, including earlier verified fixes; do not use aggregate counts as acceptance.
- [ ] Primary review of changed content and sources; independent combined review of task deltas.
- [ ] Run unit/site/output/link/course checks and full local build.
- [ ] Inspect changed pages and original figures in the browser at desktop and narrow widths, recording exact paths and limitations.
- [ ] Finish only with an honest unresolved ledger; never claim all content complete if substantive findings remain.
