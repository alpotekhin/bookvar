# Stanford and Berkeley Source Ledgers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve complete, revision-pinned source layers for Stanford CS336 Spring 2026 and Berkeley Advanced LLM Agents Spring 2025, then make every educational unit and meaningful visual mechanically accountable before textbook editing begins.

**Architecture:** Each course receives an immutable original-language source layer, a human-readable course hub, and three machine-readable YAML ledgers: source objects, educational-unit coverage, and visuals. Importers materialize official artifacts without rewriting them. Vitest validates that every scheduled lecture, assignment, reading cluster, and meaningful visual has a disposition and a stable destination or an explicit reason for exclusion.

**Tech Stack:** Obsidian Markdown, YAML, Python import scripts, Astro/Starlight publishing adapter, TypeScript/Vitest, GitHub revision URLs, SHA-256 checksums.

**Spec:** `docs/superpowers/specs/2026-09-04-stanford-berkeley-ingestion-design.md`

## Global Constraints

- Treat Stanford Spring 2026 and Berkeley Spring 2025 as distinct, named offerings.
- Fetch only official course pages, official repositories, official slide decks, official recordings, and readings linked by the official syllabus.
- Pin GitHub sources to a 40-character commit SHA. For non-Git artifacts, preserve the canonical URL, retrieval date, byte-level SHA-256, and page or slide coordinates.
- Keep source prose in its original language. Do not translate it during ingestion.
- Preserve complete official lecture source, handouts, slide decks, and student-facing assignment scaffolds; do not import solution repositories or staff-only artifacts.
- Every educational unit has exactly one disposition: `integrated`, `covered-existing`, `source-only`, or `excluded`.
- `integrated` and `covered-existing` rows identify a destination file and a stable heading anchor; the destination carries the reciprocal `source_unit_id`. `source-only` rows identify a source-hub anchor and rationale. `excluded` rows carry evidence and a reason.
- Every meaningful figure, table, animation, code trace, and multi-slide argument appears in `visuals.yml`, even if its disposition is `source-only` or `excluded`.
- A reused asset must identify author, offering, lecture or assignment, source URL, source location, local file, parent checksum, transformation, rights status, evidence, destination page, and destination anchor.
- Rights metadata uses `rights_status: licensed | permission-recorded | link-only | unknown`, plus holder, scope, license identifier/URL, and evidence. User confirmation is recorded as evidence, not silently converted into a license. `link-only` and `unknown` assets may be referenced or independently summarized but are not copied into the repository.
- Source authority uses `official-course | official-author | primary-paper | third-party-mirror | bookvar-original`. A third-party mirror is a discovery lead, not evidence for what an official assignment required.
- Do not create generated diagrams as substitutes for course visuals.
- Do not touch the English publication route, push, deploy, or merge.
- Stage only files named by the current task; the worktree contains unrelated user changes.

---

### Task 1: Define the ledger contract with failing tests

**Files:**
- Create: `publishing/tools/course-ledger.ts`
- Create: `publishing/tools/check-course-ingestion.ts`
- Create: `publishing/tests/course-ingestion.test.ts`
- Create: `publishing/tests/fixtures/course-ledger.valid.yml`
- Create: `publishing/tests/fixtures/course-ledger.invalid.yml`
- Modify: `publishing/package.json`

**Interfaces:**
- `source-manifest.yml` contains `schema_version`, `course`, `offering`, `integration_base_commit`, `retrieved_at`, and `objects[]`. Every object records `id`, `kind`, `source_authority`, lecturer/author, meeting date, title, canonical URL, revision or checksum, local path when mirrored, MIME, bytes, page count or video metadata, language, rights status, rights evidence, and retrieval date.
- `source-units.yml` is the extraction index. Executable lectures enumerate section boundaries and each rendered text/image/link event; PDFs enumerate every page, heading, figure, table, and multi-page build; assignments enumerate every task, deliverable, test interface, and evaluation requirement.
- `coverage.yml` contains one row per extracted unit with `id`, `source_object`, `source_unit`, `source_location`, `kind`, `title`, exactly one `disposition`, `destination`, `destination_anchor`, `reason`, and `primary_sources`.
- `visuals.yml` contains one row per extracted visual with `id`, `source_object`, exact pages/frames, `question`, ordered `sequence_members`, `disposition`, `destination`, `destination_anchor`, `local_file`, `parent_sha256`, `transformation`, `caption`, `attribution`, rights metadata, rendered route, desktop/narrow evidence, reviewer, and checked date.

- [ ] **Step 1: Add a valid fixture that exercises every disposition**

Include one `integrated`, `covered-existing`, `source-only`, and `excluded` unit. Require a destination for the first two and a reason for the last two.

- [ ] **Step 2: Add an invalid fixture**

Omit a source location, give one unit two dispositions, and register an integrated visual without a local file.

- [ ] **Step 3: Write the focused validation tests**

The test must reject duplicate IDs, unknown source objects or units, GitHub URLs without a pinned SHA, extracted source units absent from `coverage.yml`, integrated units without destination anchors and reciprocal `source_unit_id`, source-only rows without source-hub anchors, exclusions without reasons/evidence, missing source files, unverified artifacts marked official, reused visuals without attribution/rights evidence, unordered visual sequences, and local images absent from the existing asset registry.

- [ ] **Step 4: Run the focused test and observe failure**

```bash
cd publishing
npm test -- course-ingestion.test.ts
```

Expected: FAIL until the ledger parser and valid fixtures are complete.

- [ ] **Step 5: Implement the shared validator and make the fixture tests pass**

Implement schema parsing and referential-integrity checks in
`publishing/tools/course-ledger.ts`. The CLI and Vitest must import the same
module so importers cannot bypass a test-only contract. Do not add a runtime
dependency; use the repository's existing `yaml` package and filesystem helpers.

- [ ] **Step 6: Add a convenience script**

Add `check:course-ingestion` to `publishing/package.json` and map it to
`tsx tools/check-course-ingestion.ts`. The command validates the two real course
directories; Vitest validates the fixtures and failure modes.

- [ ] **Step 7: Commit the contract**

```bash
git add publishing/tools/course-ledger.ts publishing/tools/check-course-ingestion.ts publishing/tests/course-ingestion.test.ts publishing/tests/fixtures/course-ledger.valid.yml publishing/tests/fixtures/course-ledger.invalid.yml publishing/package.json
git commit -m "test: define course ingestion ledger contract"
```

---

### Task 2: Import and pin Stanford CS336 Spring 2026

**Files:**
- Create: `publishing/tools/import_stanford_cs336.py`
- Create: `05 Источники/Courses/Stanford CS336 Spring 2026/_index.md`
- Create: `05 Источники/Courses/Stanford CS336 Spring 2026/source-manifest.yml`
- Create: `05 Источники/Courses/Stanford CS336 Spring 2026/source-units.yml`
- Create: `05 Источники/Courses/Stanford CS336 Spring 2026/coverage.yml`
- Create: `05 Источники/Courses/Stanford CS336 Spring 2026/visuals.yml`
- Create: `05 Источники/Courses/Stanford CS336 Spring 2026/Lectures/`
- Create: `05 Источники/Courses/Stanford CS336 Spring 2026/Assignments/`
- Modify: `05 Источники/Курсы.md`
- Modify: `05 Источники/Source maps/Единый реестр покрытия источников.md`

**Interfaces:**
- Official course page: `https://cs336.stanford.edu/`.
- Official lecture repository: `https://github.com/stanford-cs336/lectures`.
- Official assignments: `assignment1-basics`, `assignment2-systems`, `assignment3-scaling`, `assignment4-data`, and `assignment5-alignment` under the `stanford-cs336` organization. The optional safety/RLHF handout is an object inside the pinned Assignment 5 source, not a sixth repository.
- The importer accepts `--refresh`, resolves and records repository SHAs, downloads official non-Git artifacts, computes SHA-256, and refuses an unpinned GitHub blob URL. `--check` validates the frozen local snapshot; `--check-upstream-drift` separately reports later changes without redefining the locked offering.

- [ ] **Step 1: Write an importer dry-run test path**

Add `--check` so the script validates the frozen local snapshot without network access. It must report missing lectures, assignments, dependency files, source hashes, and extraction units. Add a separate `--check-upstream-drift` mode for live syllabus/repository comparison.

- [ ] **Step 2: Snapshot all scheduled educational artifacts**

Preserve the complete pinned lecture-repository archive so executable lectures retain `images/`, `references.py`, `facts.py`, `lecture_util.py`, generated traces, dependency metadata, and external-asset records. Preserve lecture 1–17 source files or PDFs, recording URLs, Assignment 1–5 student handouts and scaffolds, the optional safety/RLHF handout inside pinned `assignment5-alignment`, and links for guest lectures 18–19. Do not invent a sixth safety repository. Preserve source paths and original filenames.

- [ ] **Step 3: Generate the extraction index and seed coverage rows**

Generate `source-units.yml` from the complete snapshot. Split each lecture at
actual section, derivation, experiment, worked-example, failure-mode, and visual
boundaries rather than treating a whole deck as one row. Split each assignment
by task, deliverable, mechanism, evaluation contract, and reproducibility
requirement. Every extracted unit receives a coverage row. Administrative
material may be excluded only with a reason.

- [ ] **Step 4: Seed the complete visual inventory**

Inspect executable traces, PDFs, handouts, and embedded assets event by event
and page by page. Register each figure/table/code trace and every sequence that
develops one argument; do not allow a whole-deck visual row. Preserve the
original PDF or source file even when only a licensed crop will later be used
in the textbook. Record extraction tool/version and parent checksum.

- [ ] **Step 5: Build the course hub**

The hub must list all 19 meetings, distinguish the 17 content lectures from the two guest slots, link Assignments 1–5, show pinned revisions, expose the two ledgers, and explain that textbook destinations—not this page—form the primary learning route.

- [ ] **Step 6: Update source registries**

Replace any claim that four legacy notes constitute CS336 coverage. Record the new source layer as `inventory complete; editorial integration pending` until every coverage row has a final disposition.

- [ ] **Step 7: Run focused and importer checks**

```bash
python3 publishing/tools/import_stanford_cs336.py --check
cd publishing
npm run check:course-ingestion
npm run check:source-coverage
```

Expected: all 17 content lectures, both guest slots, five assignments, and the optional safety branch are accounted for; no source or visual row is orphaned.

- [ ] **Step 8: Commit the Stanford source layer**

```bash
git add publishing/tools/import_stanford_cs336.py '05 Источники/Courses/Stanford CS336 Spring 2026' '05 Источники/Курсы.md' '05 Источники/Source maps/Единый реестр покрытия источников.md'
git commit -m "sources: pin Stanford CS336 Spring 2026"
```

---

### Task 3: Import and pin Berkeley Advanced LLM Agents Spring 2025

**Files:**
- Create: `publishing/tools/import_berkeley_agents.py`
- Create: `05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/_index.md`
- Create: `05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/source-manifest.yml`
- Create: `05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/source-units.yml`
- Create: `05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/coverage.yml`
- Create: `05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/visuals.yml`
- Create: `05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/Lectures/`
- Create: `05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025/Readings/`
- Modify: `05 Источники/Курсы.md`
- Modify: `05 Источники/Source maps/Единый реестр покрытия источников.md`

**Interfaces:**
- Official syllabus: `https://rdi.berkeley.edu/adv-llm-agents/sp25`.
- Twelve lecture bundles: recording, slides, and syllabus-linked readings for Jan 27; Feb 3, 10, 24; Mar 3, 10, 17, 31; Apr 7, 14, 21, 28.
- The importer records canonical URLs and SHA-256 for every slide deck. Readings are catalogued by canonical paper or project URL; copyrighted paper PDFs are not mirrored unless their own terms permit it. The Spring 2025 syllabus exposes 37 reading links across the twelve meetings, distributed 3, 3, 3, 3, 2, 4, 2, 4, 3, 4, 2, 4.

- [ ] **Step 1: Implement refresh and check modes**

The script must fail when the locked syllabus contains a lecture, slide deck,
recording, or any of its 37 individual reading links absent from the manifest.
It must retain the full source-object schema, including slide page count and
recording channel/video ID/title/duration. Live changes are reported only by
`--check-upstream-drift`.

- [ ] **Step 2: Preserve all twelve official slide decks and lecture metadata**

Store original decks unchanged. Where an official deck is a PDF, also create a page-index text extraction used only for search and editorial mapping; never substitute the extraction for the original deck.

- [ ] **Step 3: Catalogue all 37 syllabus-linked readings individually**

Create exactly one manifest object and at least one coverage disposition per
reading link, preserving lecture/date/order. Record title, authors, year,
canonical project URL or pinned arXiv version/date, and relationship to the
deck. Reject a single collapsed `reading cluster` row that can hide omissions.
Mark the lecture deck as a secondary explanation and the paper/project page as
the primary source for technical claims.

- [ ] **Step 4: Build section-level coverage rows**

Account for inference-time reasoning, learning to reason, memory and planning, open post-training recipes, coding agents, web agents, GUI agents, AlphaProof, autoformalization, advanced theorem proving, abstraction and discovery, and agent safety.

- [ ] **Step 5: Build the complete visual ledger**

Review every slide in all twelve decks and record reviewed page ranges. Each
deck must enumerate its orienting, mechanism, comparison, empirical, and
multi-slide visuals or give an item-level exclusion reason. Preserve ordered
page membership for Yu Su's agent-first/LLM-first argument, memory and world
models; Kaiyu Yang's Lean and theorem-proving pipeline; Charles Sutton's
vulnerability-discovery loop; Swarat Chaudhuri's LaSR/concept-library sequence;
and Dawn Song's threat model and privilege-control sequence.

- [ ] **Step 6: Record the practice provenance boundary**

The source hub must state that the public syllabus confirms a lab and project
but does not expose a verified official lab artifact on the syllabus page. A
Drive artifact, if found, must be pinned by file ID, export checksum, and
retrieval date before use. Add the Precioux student mirror as
`source_authority: third-party-mirror`, `disposition: excluded`, reason
`discovery lead only`; it cannot substantiate an official lab contract.
Describe any later Bookvar exercise as an adaptation inspired by the course.

- [ ] **Step 7: Update the source registries**

Record the source layer as `inventory complete; editorial integration pending` and add a lecture-to-destination table without placing Berkeley as a top-level textbook module.

- [ ] **Step 8: Verify and commit**

```bash
python3 publishing/tools/import_berkeley_agents.py --check
cd publishing
npm run check:course-ingestion
npm run check:source-coverage
```

Expected: twelve lecture bundles and every syllabus reading are present; all meaningful visuals have a disposition; the lab provenance warning is visible.

```bash
git add publishing/tools/import_berkeley_agents.py '05 Источники/Courses/Berkeley Advanced LLM Agents Spring 2025' '05 Источники/Курсы.md' '05 Источники/Source maps/Единый реестр покрытия источников.md'
git commit -m "sources: pin Berkeley Advanced LLM Agents Spring 2025"
```

---

### Task 4: Publish the source hubs without polluting the textbook route

**Files:**
- Modify: `publishing/navigation.yml`
- Modify: `publishing/tests/routes.test.ts`
- Modify: `publishing/tests/publication-policy.test.ts`
- Modify: `05 Источники/_index.md`

**Interfaces:**
- Stanford hub route: `/sources/courses/stanford-cs336-spring-2026/`.
- Berkeley hub route: `/sources/courses/berkeley-advanced-llm-agents-spring-2025/`.
- Raw source pages live below each hub; they must be reachable but must not appear in the primary textbook sidebar.

- [ ] **Step 1: Add failing route tests**

Require both hubs, source-native lecture pages, and original deck links. Require that neither course becomes a top-level `Учебник` section.

- [ ] **Step 2: Add source navigation entries**

Place both hubs under `Источники → Курсы`. Preserve concept-oriented textbook navigation.

- [ ] **Step 3: Make legacy Stanford pages non-canonical**

Remove the four legacy notes from visible source navigation. Do not delete them until the Stanford editorial integration plan has reconciled their unique claims.

- [ ] **Step 4: Run the publication checks**

```bash
cd publishing
npm test -- routes.test.ts publication-policy.test.ts
npm run build
npm run check:links
```

Expected: both hubs and all declared source routes build; legacy pages are absent from sidebar/search; no broken routes or fragments.

- [ ] **Step 5: Commit the source navigation**

```bash
git add publishing/navigation.yml publishing/tests/routes.test.ts publishing/tests/publication-policy.test.ts '05 Источники/_index.md'
git commit -m "docs: publish pinned Stanford and Berkeley source hubs"
```
