# Stanford and Berkeley Integration Final Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove that the Stanford and Berkeley material is complete, visually legible, narratively integrated, and publishable without hidden source, navigation, or rendering debt.

**Architecture:** The audit runs in four passes: machine-readable ledger completeness, reverse source-to-textbook reconciliation, rendered visual inspection, and continuous editorial reading. The audit produces durable reports and regression tests rather than relying on a one-time verbal assessment.

**Tech Stack:** YAML ledgers, TypeScript/Vitest, Astro/Starlight build output, Playwright or existing browser screenshot tooling, Obsidian Markdown, Git.

**Spec:** `docs/superpowers/specs/2026-09-04-stanford-berkeley-ingestion-design.md`

## Global Constraints

- Run only after both source-ledger tasks and all Stanford/Berkeley integration tasks are complete.
- Do not mark a unit complete because a source page exists; verify that the destination contains the mechanism, evidence, and source link promised by the ledger.
- Inspect generated output, not only Markdown source.
- Do not repair translation or create an English route in this pass.
- Rights status is a blocking gate for copied assets: `unknown`, `link-only`, and all-rights-reserved material can still be covered by link and independent explanation but cannot pass as a locally reused figure.
- Before every commit, stage exact files individually and inspect `git diff --cached --name-only`; never stage `tmp/` screenshots or pre-existing user changes.
- Do not push, merge, or deploy without a new explicit user instruction.
- Preserve unrelated worktree changes and stage only named audit fixes.

---

### Task 1: Add reverse-coverage and asset regression tests

**Files:**
- Modify: `publishing/tests/course-ingestion.test.ts`
- Modify: `publishing/tests/asset-registry.test.ts`
- Create: `publishing/tests/course-destination-links.test.ts`
- Create: `publishing/tests/course-visual-rendering.test.ts`

**Interfaces:**
- Every `integrated` or `covered-existing` unit points to an existing page and heading anchor carrying its stable `source_unit_id`; the ID resolves through the manifest to a specific artifact and canonical URL.
- Every integrated visual exists locally, is registered, is referenced at its destination anchor, and has a caption, attribution, parent checksum, transformation record, and permissible rights status.

- [ ] **Step 1: Write a reverse destination-link test**

For each integrated coverage row, resolve the destination file/heading and
require the exact `source_unit_id`. Resolve that ID to the source object and
canonical URL. Reject a destination that points only to the generic course hub
when a lecture or paper artifact is known.

- [ ] **Step 2: Write a visual-reference test**

Require each integrated visual's local file in the asset registry and in the destination Markdown. Require crop metadata when `transformation: crop` and sequence ordering when `sequence` is present.

- [ ] **Step 3: Write a rendered-dimension test**

Static tests inspect intrinsic dimensions, file signatures, PDF page counts,
copied assets, and figure/figcaption structure. Browser tests use Playwright at
1440×1000 and 390×844 and reject `scrollWidth > clientWidth`, zero-sized
rendered boxes, detached captions, and clipped media. Static HTML alone is not
accepted as viewport evidence.

- [ ] **Step 4: Run the focused tests and repair only real failures**

```bash
cd publishing
npm test -- course-ingestion.test.ts course-destination-links.test.ts asset-registry.test.ts course-visual-rendering.test.ts
```

Expected: all source, destination, and visual relationships pass in both directions.

- [ ] **Step 5: Commit the regression gate**

```bash
git commit -m "test: enforce course coverage and visual integrity"
```

---

### Task 2: Perform the reverse source audit

**Files:**
- Create: `05 Источники/Source maps/Stanford CS336 Spring 2026 — final coverage audit.md`
- Create: `05 Источники/Source maps/Berkeley Advanced LLM Agents Spring 2025 — final coverage audit.md`
- Modify: `05 Источники/Source maps/Единый реестр покрытия источников.md`

**Interfaces:**
- The audit reads sources in original order and verifies ledger rows against final destinations.
- Final status is derived from zero unaccounted educational units, not page count.

- [ ] **Step 1: Reverse-audit Stanford**

Read every extracted source unit from Lecture 1–17 and Assignment 1–5 at the
pinned revisions. For every section, derivation, experiment, worked example,
failure mode, task, and deliverable, record destination anchor, depth verdict,
visuals, primary sources, and remaining gap. Inspect guest slots 18–19 for
newly available official material and record the result without inventing
content.

- [ ] **Step 2: Reverse-audit Berkeley**

Read all twelve decks and all 37 individual syllabus readings in syllabus
order. Record official-artifact availability for every lecture. Verify that
each argument appears in a coherent thematic chapter and that no lecture was
reduced to a one-paragraph mention.

- [ ] **Step 3: Audit exclusions**

Challenge every `excluded` and `source-only` row. Keep it only if administrative, genuinely redundant, obsolete, illegible without a better source, or disruptive to the narrative; state the evidence.

- [ ] **Step 4: Update the unified registry**

Generate report totals from the machine ledgers and test that Markdown totals
match. Report exact counts for source objects, educational units, integrated
units, covered-existing units, source-only units, exclusions, unavailable
objects, rights-unknown objects, permission-needed objects, visuals, and reused
visuals. Avoid percentages without counts.

- [ ] **Step 5: Commit the coverage audit**

```bash
git commit -m "docs: audit Stanford and Berkeley coverage"
```

---

### Task 3: Inspect every changed chapter and visual in the rendered site

**Files:**
- Create: `00 Учебник/Аудит Stanford и Berkeley 2026-09.md`
- Create: `tmp/course-integration-screenshots/desktop/`
- Create: `tmp/course-integration-screenshots/narrow/`
- Create: `tmp/course-integration-screenshots/changed-routes.yml`
- Modify: only chapter, asset metadata, or stylesheet files that fail inspection

**Interfaces:**
- Desktop viewport: 1440×1000 or the repository's documented equivalent.
- Narrow viewport: 390×844 or the repository's documented equivalent.
- Screenshots are audit artifacts and remain untracked unless the repository already publishes audit screenshots.

- [ ] **Step 1: Build from a clean, version-locked generated-output directory**

```bash
pnpm --dir publishing build
pnpm --dir publishing check:links
pnpm --dir site check
pnpm --dir site build
pnpm --dir publishing test:output
```

- [ ] **Step 2: Generate the changed-route manifest and capture every route**

Generate `changed-routes.yml` from the integration commit range and both
ledgers. Capture every changed destination—not a sample by page type—at desktop
and narrow widths. Record route, viewport, screenshot hash, source commit,
reviewer, timestamp, and verdict. The manifest must include foundations,
equation-heavy systems pages, long slide sequences, capstones, source hubs, and
the complete advanced-agent route.

- [ ] **Step 3: Inspect every reused visual**

Record pass/fail for crop boundaries, text size, aspect ratio, sequence order,
caption usefulness, attribution, light/dark contrast, and relation to
surrounding prose. Re-crop only from the original source; update parent hash,
tool/version, crop coordinates, and ordered sequence membership.

- [ ] **Step 4: Read every changed chapter continuously**

Mark and fix abrupt lecture-note insertions, repeated definitions, machine-like throat-clearing, mixed terminology without introduction, unexplained formulas, decorative images, and transitions that do not prepare the next concept.

- [ ] **Step 5: Rebuild and repeat failed screenshots**

No visual failure may be closed from Markdown inspection alone.

- [ ] **Step 6: Commit the editorial and visual fixes**

Stage the audit report and only the exact files changed by this pass. Keep
`tmp/course-integration-screenshots/` untracked.

```bash
git commit -m "docs: complete rendered editorial audit"
```

---

### Task 4: Reconcile global navigation and the practical route

**Files:**
- Modify: `00 Учебник/_index.md`
- Modify: `00 Учебник/Карта крупных учебных модулей.md`
- Modify: `00 Учебник/Покрытие программы.md`
- Modify: `00 Учебник/Редакционная матрица Bookvar.md`
- Modify: `06 Практика/_index.md`
- Modify: `05 Источники/_index.md`
- Modify: `publishing/navigation.yml`
- Modify: `publishing/tests/routes.test.ts`
- Modify: `publishing/tests/publication-policy.test.ts`
- Create: `publishing/tests/course-navigation.test.ts`

**Interfaces:**
- Textbook sidebar remains concept-oriented.
- Source hubs remain under Sources.
- Capstones appear in dependency order after their prerequisite theory.

- [ ] **Step 1: Repair the linear textbook route**

Ensure the reader reaches language-model construction before systems optimization, then scaling/data/post-training, then agent foundations and advanced agents. Remove duplicate or obsolete sidebar entries.

- [ ] **Step 2: Repair previous/next links**

Every new chapter and capstone has a meaningful predecessor and successor. Course source pages must return to the canonical textbook destination without hijacking the primary sequence.

- [ ] **Step 3: Update curriculum maps**

List the five Stanford capstones, the Bookvar verifiable-coding practice
(course-inspired; official lab contract unverified), and the Bookvar AgentX-style
capstone with prerequisites, outputs, and evidence bundles. Reflect advanced
chapters 69–74 without labelling the Bookvar exercises as Berkeley Lab 1/2.

- [ ] **Step 4: Add navigation regression assertions**

Require unique sidebar labels within a section, numeric order for chapters
65–74 and practices 20–26, no visible legacy Stanford notes, no top-level
course-name sections under the textbook, and exact previous/next edges for the
new routes.

- [ ] **Step 5: Verify and commit**

```bash
cd publishing
npm test -- routes.test.ts publication-policy.test.ts
npm run build
npm run check:links
```

```bash
git commit -m "docs: reconcile course integration navigation"
```

---

### Task 5: Run the final publication gate and prepare a local review handoff

**Files:**
- Modify: `00 Учебник/Аудит Stanford и Berkeley 2026-09.md`
- Modify: `05 Источники/Source maps/Stanford CS336 Spring 2026 — final coverage audit.md`
- Modify: `05 Источники/Source maps/Berkeley Advanced LLM Agents Spring 2025 — final coverage audit.md`

**Interfaces:**
- Final evidence includes schema version, audit timestamp, auditor, source commits/checksums, tested Git commit, commands, test/build/link output, source totals, rights totals, visual totals, rendered-page review, commit list, and intentionally deferred items.

- [ ] **Step 1: Run all automated checks from the final tree**

```bash
pnpm --dir publishing check:course-ingestion
pnpm --dir publishing check:source-coverage
pnpm --dir publishing test
pnpm --dir publishing build
pnpm --dir publishing check:links
pnpm --dir site check
pnpm --dir site build
pnpm --dir publishing test:output
```

Expected: every command exits zero; no broken internal route, fragment, or file; all source and visual ledgers are complete.

- [ ] **Step 2: Inspect Git scope**

```bash
git status --short
git diff --check
git log --oneline --decorate -20
git diff --name-only "$INTEGRATION_BASE"...HEAD -- en/ ':(glob)**/en/**'
```

Set `INTEGRATION_BASE` to the pre-ingestion commit recorded in both source
manifests before running the command. Confirm that unrelated pre-existing user changes were not
staged or rewritten and that no `en/` file changed.

- [ ] **Step 3: Record exact completion evidence**

Generate counts from ledgers and write named exceptions into the audit
documents. Do not state `complete` while any ledger row lacks a disposition,
destination anchor/source-hub anchor/reason, rights decision, or visual review.

- [ ] **Step 4: Start the local preview and open the review route**

Use the repository's existing development command. Open the textbook index, one Stanford foundation page, one systems page, one capstone, one Berkeley advanced-agent page, and both source hubs for user inspection.

- [ ] **Step 5: Stop before external publication**

Report the local commit range and preview URLs. Wait for explicit approval before push, merge, GitHub Pages deployment, or branch publication.

- [ ] **Step 6: Commit final audit metadata if it changed**

```bash
git commit -m "docs: record final Stanford and Berkeley audit"
```
