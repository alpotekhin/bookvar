# Course integration state audit

Date: 2026-09-15  
Checkout: `editorial-route-alignment` at `e77f9a027e66f6719db486d2f31d9c08b3ae1d2f`  
Scope: read-only audit of the live worktree; no importer regeneration and no source, code, or textbook edits.

## Verdict

Stanford CS336 Spring 2026 is an active, structurally validated integration. Berkeley Advanced LLM Agents Spring 2025 is also active in the current worktree, but its reader-facing hub still describes the pre-activation state. The Berkeley integration is currently an uncommitted worktree state: its ledgers and many destinations are modified or untracked, so this audit does not claim that the same state is present in `HEAD` or on a remote branch.

The ledger numbers are semantic-unit counts, not slide counts and not a measure of prose volume. They must not be summarized as “N slides integrated.” In particular, Stanford's 554 `integrated` rows include 373 assignment rows, while Berkeley's 137 include 35 reading catalogue records whose bodies are not mirrored.

## Evidence standard used

- **Integrated / covered-existing:** the current ledger names a destination and anchor; the importer and `course-ledger.ts` validate that the file and anchor exist and that the destination carries the reciprocal `source_unit_id`.
- **Integrated visual:** in addition, the validator requires the local image file, an embed in the destination, attribution/transformation metadata, rendered route, desktop and narrow evidence, reviewer, and date.
- **Contract-only:** a Stanford assignment test interface is preserved as a normative contract without claiming that its full wording is textbook prose.
- **Source-only:** preserved and routed to the course archive. This includes both intentional catalogue/reference material and genuinely deferred topics.
- **Excluded:** reviewed but deliberately not reused. It is not “missing.”

The validators establish structural provenance and destination presence. They do not automatically establish that every destination is pedagogically complete or that every claim was independently re-derived.

## Current detailed-ledger state

| Course | Source inventory | Semantic units | Coverage dispositions | Semantic visual rows | Unique active destinations |
|---|---:|---:|---|---:|---:|
| Stanford CS336 Spring 2026 | 23 objects: 17 lectures and 6 assignment handouts/branches | 807 | 554 integrated; 24 covered-existing; 21 contract-only; 154 source-only; 54 excluded | 159: 73 integrated; 9 covered-existing; 45 source-only; 32 excluded | 56 coverage destinations; 36 visual destinations |
| Berkeley Advanced LLM Agents Spring 2025 | 64 objects: 13 PDFs, 12 recording records, 37 reading records, syllabus, and one third-party practice lead | 212 | 137 integrated; 6 covered-existing; 41 source-only; 28 excluded | 161: 47 integrated; 87 source-only; 27 excluded | 16 coverage destinations; 12 visual destinations |

### Counts that must stay separate

Stanford has 496 physical lecture-PDF pages and 176 assignment-PDF pages. Its nine executable lectures instead expose 4,526 edtrace steps through the manifest's `page_count` field. Adding those steps to physical PDF pages would create a meaningless “page” total. The 807 reviewed semantic units are a third, independent count.

Berkeley has 1,254 physical pages across 13 official PDFs, organized into 12 meeting bundles. The 212 semantic units and 161 visual rows are reviewed groupings over those pages. The 37 readings are individual metadata records, not locally mirrored paper bodies.

### Stanford: what is actually integrated

The integration reaches all major course arcs rather than only the source archive:

- foundations and building an LM: tokenization, next-token modeling, Transformer components, and `06 Практика/20`;
- resource accounting and systems: GPU/memory, roofline, profiling, Triton, FlashAttention, and `06 Практика/21`;
- distributed training: DDP, tensor/sequence/pipeline/expert parallelism, ZeRO/FSDP, and scaling practice;
- data and evaluation: corpus lineage, filtering, deduplication, mixtures, evaluation validity, and `06 Практика/23`;
- inference: KV cache, scheduling, serving engines, quantization, compression, and speculative decoding;
- post-training and multimodality: SFT, reward modeling, PPO/DPO, GRPO/RLVR, reasoning cases, and multimodal chapters.

The 554 integrated rows break down as 181 lecture-derived rows and 373 assignment-derived rows. The 154 source-only rows include 97 assignment rows—mostly exact deliverable wording and test interfaces that correctly remain normative in the pinned handouts—and 57 lecture rows.

Useful deferred Stanford topics are narrow, not evidence that the whole course is unfinished:

1. multi-token prediction as a training objective (`lecture-04-deepseek-mtp`);
2. cross-layer attention and local/sliding-window attention, which need a dedicated long-context treatment;
3. FP4 and other evolving low-precision material, after a stable systems framing exists;
4. ARC interactive reasoning, GCG adversarial evaluation, and AIR-Bench as bounded evaluation/safety case studies;
5. DeepSeek v4 attention only after released primary evidence is verified.

Course orientation, staff, historical overview, duplicate recaps, weak/obstructed figures, dated lawsuits, and exact submission wording should remain source-only.

### Berkeley: active coverage hidden by a stale hub

The active map is broad and thematic:

- 13 integrated plus 2 covered-existing rows in test-time reasoning;
- post-training routes across SFT, reward modeling, DPO, RLVR, and reasoning distillation;
- memory, planning, RAG, and tool use;
- 27 integrated rows in coding/web/computer-use agents;
- 40 integrated plus 1 covered-existing row in formal proofs and mathematical agents;
- scientific discovery, agent evaluation, and 15 integrated security rows.

Its 137 integrated rows consist of 102 deck-derived semantic units and 35 reading catalogue records. The latter establish a primary-reading route and metadata provenance; they do not mean that the copyrighted paper text was imported.

The hub `_index.md` is materially stale. It says `coverage.yml` and `visuals.yml` remain source-only/excluded until future prose and images appear, although the current importer validates active destinations. It also prints 42 integrated and 92 source-only visuals, while the current ledger and snapshot lock record 47 and 87. The coverage totals printed by the hub happen to match the current ledger, but their surrounding status explanation does not. The shared course catalogue `05 Источники/Курсы.md` is stale for both detailed courses: its Stanford block still reports the old 785-unit pending baseline and its Berkeley block still calls the 1,254-page archive a non-integrated source-only baseline.

Of Berkeley's 41 source-only units, 12 are recording metadata with no mirrored media body. Most others are intentionally retained orientation, syllabus/coursework metadata, broad or dated landscape summaries, speculative future claims, or material already represented by narrower integrated mechanisms. The useful deferred queue is:

1. xGen video and GenS temporal/generalist-agent material after the separate video integration is reviewed;
2. visual concept discovery after pinning a connected primary-source route;
3. Plan–Sequence–Learn robotics only if Bookvar gains a coherent embodied-agent route;
4. AgentTrek only after a primary upstream publication/repository and rights basis are pinned;
5. the proprietary-adjacent pages 118–121 only after explicit rights review.

Meeting 6 page 117 remains a hard `do-not-reuse` exclusion and must not be promoted by a course-level permission record.

## Bounded source-to-destination sample

This was a concrete sample, not a re-audit of all 807 Stanford units, 212 Berkeley units, or 320 visual rows.

1. **Stanford BPE trace.** The pinned executable lecture implements repeated adjacent-pair counting on `the cat in the hat`, then checks `decode(encode(text)) == text`. The destination chapter at `bpe-merge-trace` contains four preserved trace states and explains ordered merges, byte vocabulary, round-trip scope, and the stricter Assignment 1 contract. This is substantive integration, not a backlink-only claim.
2. **Stanford roofline and wave quantization.** The ledger routes Lecture 5 pages 20–23 and 45–48 plus Lecture 6 benchmarking/profiling experiments to `cs336-systems-roofline`. The destination includes a numerical roofline example, preserved shape-sweep and 1792→1793 frames, bounded captions, and an actionable benchmark protocol.
3. **Stanford MinHash/LSH.** Lecture 14 trace spans and Assignment 4 tasks route to `minhash-lsh`. The destination derives Jaccard, MinHash agreement, and `1-(1-s^r)^b`, embeds three preserved trace states, and explains finite-sample and threshold limitations.
4. **Berkeley agent-security chain.** Meeting 12 physical page 27 lists five ways model output becomes part of an attack chain. The destination converts those five levels into a Russian table, embeds the full page, and adds system-boundary mitigations without claiming that the model alone “performs SQL injection.”
5. **Berkeley formal proof agents.** Meeting 8/9 units route to the formal-math chapter. The destination separates autoformalization from proving, explains proof state → tactic → Lean execution → kernel checking, and embeds source pages for Lean, ReProver, LIPS, and LeanEuclid with bounded captions.

The sample supports the interpretation that `integrated` corresponds to real prose/figure work in representative foundation, systems, data, safety, and formal-agent areas. It does not prove uniform quality across every row; assignment-heavy counts and reading metadata make such an extrapolation invalid.

## Older course imports

### Efficient DL Systems

- Source archive: 40 imported records—10 notebooks, 20 READMEs, and 10 slide decks—at pinned commit `e632aa89...`.
- Transfer matrix: 37 rows—33 `integrate`, 3 source-only asset groups, 1 excluded illustrative dataset.
- Present relationship: the nine-week course is distributed across the ML Systems, distributed-training, inference, deployment, and practice routes. Practice pages 06–17 link directly to original notebooks, handouts, and decks.
- Evidence limit: there is no Stanford/Berkeley-style per-semantic-unit ledger. The source-coverage check proves that declared destinations exist and contain a direct source marker, not that every source section was transferred.

### Harvard ML Systems

- Source archive: 29 complete English book chapters, 34 Marimo labs, 35 Beamer slide decks, and 20 TinyTorch modules. These are source-layer originals under root `05 Источники`, not authored `/en/` textbook translations.
- Transfer matrix: 173 source rows—114 `integrate`, 59 `cross-link`. The current ML-systems gap matrix marks all 22 thematic systems concepts integrated.
- Present relationship: canonical Bookvar chapters synthesize system framing, hardware, measurement, training/distribution, inference/serving, deployment, security, robustness, and responsibility; `06 Практика/18` is a reader-facing design-lab index, while TinyTorch supports framework-from-first-principles practice.
- Evidence limit: the automated check validates 59 unique declared destinations for existence and a Harvard source marker. It does not semantically compare all 173 source rows with the prose.

### HSE ML course

- Source archive: 67 generated artifacts—31 notebooks, 26 PDFs, and 10 Python files—from two Spring 2026 tracks at pinned commit `4b210515...`; provenance records owner permission for a noncommercial teaching archive despite no repository-wide license.
- Current canonical evidence: explicit HSE links were found in the spectral/graph clustering chapter for the full seminar and unsupervised homework. The 67-artifact validator proves files, hashes, assets, provenance, and routes, not broad textbook integration.
- Stale contradiction: `HSE ML course — link map.md` still says `status: link-only` and that Bookvar does not mirror the material, while the generated archive and manifest now do mirror it under the recorded permission. Treat the source map as stale and the course as mostly source-archive coverage until a current destination ledger exists.

### SHAD LLM

- Only a legacy index and one “Week 1 — Intro to LLMs” note exist; the index says “In progress.”
- The note points to a missing `raw/courses/SHAD LLM/...` path and links mainly to the old `Concepts/` layer. It has no current canonical textbook destination map and no detailed integration ledger.
- Therefore SHAD is legacy/source-only, not an integrated course. Its intuitive RLHF/PPO-ptx explanation may be useful after source provenance is restored and mapped to current post-training chapters.

## Importer ownership and durable edit locations

| Reader-facing page | Generated? | Durable place for a future edit |
|---|---|---|
| Stanford `_index.md` | Yes | `publishing/tools/import_stanford_cs336.py`, function `hub_markdown()`; direct edits are overwritten by `write_generated_ledgers()` |
| Berkeley `_index.md` | No | Edit the hub directly; preferably add a parity assertion so totals/status cannot drift from the active ledger |
| Efficient DL Systems root hub | No | Edit `05 Источники/Courses/Efficient DL Systems.md`; the importer writes the directory artifacts and manifest only |
| Harvard ML Systems root hub | No | Edit `05 Источники/Courses/Harvard ML Systems.md` |
| Harvard `tinytorch/README.md` | Yes | Edit the overview template in `publishing/tools/import_tinytorch.py` |
| Harvard `Labs and slides.md` | Yes | Edit `write_index()` in `publishing/tools/import_harvard_labs_slides.py` |
| HSE ML course root hub | Yes | Edit the inline index template in `publishing/tools/import_hse_ml_course.py`; direct edits are overwritten |
| SHAD legacy index/week note | No importer found | Edit the legacy pages directly, but only after provenance is restored |
| Shared course catalogue | No | Edit `05 Источники/Курсы.md`; add the English counterpart at `en/05 Sources/Courses.md` and reuse its locale route rather than creating a second sidebar item |

The source maps and `ML systems — Bookvar gap matrix.md` are human-authored editorial documents; importer regeneration does not update their claims.

## Recommended thematic course navigation

Do not create a short Bookvar page per lecture. Rework the existing reader-facing `05 Источники/Курсы.md` as the “Courses and source routes” map, and add `en/05 Sources/Courses.md` as its localized counterpart on the same source route. Link the map from the textbook index/program-coverage page without adding a duplicate sidebar item. Use one card per course and two layers:

1. **Study by Bookvar theme** — links directly to canonical module maps/chapters and practice sequences.
2. **Open the original course** — links to the pinned source hub, manifest/ledger status, language, rights boundary, and source-only queue.

Recommended course cards:

- **Stanford CS336:** (a) tokenization and LM from scratch; (b) GPU/roofline/kernels; (c) distributed training and scaling; (d) data and evaluation; (e) inference; (f) post-training/RLVR and multimodality. Pair each theme with its canonical chapters and practices 20–24.
- **Berkeley Agents:** (a) reasoning and post-training; (b) memory/planning/RAG; (c) coding/web/GUI agents; (d) formal mathematics and discovery; (e) evaluation and security. Pair with chapters 65, 67–72 and practices 25–27.
- **Efficient DL Systems:** systems route by outcome—measure, accelerate input, distribute, shard, optimize a step, serve, quantize/speculate—rather than week numbers alone.
- **Harvard ML Systems:** foundations → scale → inference/operations → responsible systems, with separate “full English chapter,” “design lab,” and “TinyTorch module” links.
- **HSE:** classical-ML spine—linear models, evaluation/calibration, trees/ensembles, neural foundations, clustering/dimensionality reduction, kernels/recommendations—while labeling most links as original Russian course material until destination coverage is audited.
- **SHAD:** place in a legacy/unverified section until a pinned source and current canonical mapping exist.

Every card should display semantic-unit counts separately from physical pages and should say whether the integration state is committed or only validated in the current worktree. Root source-layer English originals must be labelled as source archives. They must not be counted as `/en/` translations: this worktree has 37 Markdown files under `en/`, of which 27 currently carry both `locale: en` and `translation_of`.

## Prioritized queue

1. **P0:** update the Berkeley hub's status and 47/87 visual counts; add ledger-to-hub parity coverage.
2. **P0:** update `05 Источники/Курсы.md`, add `en/05 Sources/Courses.md` on the same localized route, and wire the existing map from the textbook index/program-coverage route; do not duplicate source hubs or sidebar items.
3. **P0:** correct the HSE source-map contradiction and explicitly label HSE destination coverage as partial/audited separately from archive completeness.
4. **P1:** expose Stanford and Berkeley theme routes with practice pairings and source-only/deferred badges.
5. **P1:** integrate the useful deferred topics listed above only after their primary-source, rights, and destination prerequisites are met.
6. **P2:** migrate SHAD only after restoring a pinned source; otherwise preserve it as legacy.

## Executed validation

- `python3 publishing/tools/import_stanford_cs336.py --check` — passed: 17 lectures, 2 guest slots, 5 assignments plus safety supplement, 902 artifacts, 807 units/coverage rows, 159 visual rows.
- `python3 publishing/tools/import_berkeley_agents.py --check` — passed.
- `pnpm --dir publishing check:source-coverage` — passed: 210 Harvard/EDLS matrix rows, 59 destinations with direct source markers, 34 Harvard labs, 35 slide decks, and 67 HSE artifacts.
- `pnpm --dir publishing check:course-ingestion` — passed outside the sandbox because `tsx` could not create its IPC socket inside the sandbox.
