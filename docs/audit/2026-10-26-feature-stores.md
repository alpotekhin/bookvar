# MLOps — historical features and feature migrations

## Delivered revision and preservation

Canonical chapter: `00 Учебник/19 Deployment, Reliability и MLOps/02 MLOps.md`.
Route: `/textbook/deployment/mlops/`. Final: 1312 lines, SHA-256
`57ee342a14d420a5c97a7a48511dfe1efa430bcb29d3c207c8e3e676dc51739e`.

The entire accepted 907-line chapter was read before editing, and the final
chapter was read in full. Removing the two marked additions and reverting
the two frontmatter dates reconstructs the accepted SHA
`0e63a2dde76d8eae9630036aac9542f63c48cf0f15cb98a659ca2dd5fa76f454`.
Its seven original figures, examples, navigation and old anchors remain.
The only styling addition gives the new lineage table 10rem minimum cells;
an inline span identifies that table without changing its visible text.

## Explanation review

Selected Harvard source scopes were read in full, including their related
figures, listings and calculations: ML Operations 795–916; ML Operations
at Scale, Feature Store Operations through Backfill procedures. The pinned
45ecc8d8 source files are reused from the 20 October archive. Feast v0.66.0
architecture and point-in-time documentation were read in full. The SDK
get_historical_features signature, full docstring and option forwarding
were inspected; the whole SDK and all providers were not reviewed/run.
Exact source paths and revisions are in the source archive README.

| Mechanism | Preserved source explanation | Worked reasoning and limits |
|---|---|---|
| Training-serving feature definitions | Original Harvard session-length example; original Feast architecture | Historical `tickets_7d` versus current values; fixed-history boundary and duplicate tests; storage/retrieval does not supply all transformations or prove parity. |
| Point-in-time retrieval | Original Harvard noon/future-clicks SVG and full lateral SQL | Four past versus ten daily clicks; row selection, left join and no-match cases are explained before extending the query. |
| Availability, TTL and corrections | Full Feast point-in-time page; original join image | The printed fixture separates feature time and online availability. Noon returns 3, 13:00 returns 8, stale and absent entities return NULL. Original event-time query can choose either late revision; no arbitrary deterministic winner is claimed. Equal-time visibility and missing values are explicit contracts. |
| Creation timestamp in Feast | Pinned docs and API implementation | Default False, explicit opt-in, missing/NULL column and backend restrictions; warehouse creation time need not equal online availability. No backend test is implied. |
| Historical storage | Harvard's complete storage parameter/calculation block | 100M × 1000 × 8760 × 100 bytes = 87.6 decimal PB under dense hourly-retention assumptions. Record size, sparse updates, metadata, compression, replicas and temporary migration overlap are distinguished. |
| Feature migration | Full original version/lineage table, representative scenario and incremental-backfill paragraph | z-score 1 versus min-max 0.4 changes an old 0.8 threshold decision; versioned namespaces and complete partition manifests avoid mixed semantics. Missing or wrong-snapshot seventh partition blocks the fixture. Rollback restores the feature/model pair. Audit replay and a counterfactual new-definition experiment are separated. |

The prose was checked against no-ai-slop separately from these mechanisms.
Source quotations, code and table words were preserved; the first excerpt's
Quarto reference is expanded into words and table emphasis/spacing is
simplified. Editorial SQL, numerical and migration examples are identified.
No definition-only placeholder substitutes for the selected mechanisms.

## Figures and rights

Three complete originals are added: Feast architecture PNG 7322×2860,
Harvard time-travel SVG viewBox 0 0 700 480, and Feast point-in-time join PNG
1260×588. All published bytes equal the archived sources. Caption, author,
revision, source URLs, hashes, license evidence and public license copies
are retained in the local manifest and global registry.

The competing Harvard architecture SVG was inspected and rejected: it
claims skew is impossible by construction and no single database can meet
both access patterns. It remains in the source archive only, unchanged;
it is not published or redrawn. Scenario latency, fleet and engagement
numbers are not transferred as empirical results. Harvard adaptations and
new examples explicitly use CC BY-NC-SA 4.0; unchanged Feast images retain
Apache-2.0. The prior Google/SRE licenses are not replaced. The Feast root
tree at the pinned revision contains LICENSE and no root NOTICE file.

## Executed checks

- All nine chapter Python fences executed; the previous 1400 independent
  cases still pass. Fifteen old source hashes and 17 built anchors match.
- New checker executes both new Python fences, checks 300 partition cases,
  preserved source passages/table cells/SQL and the three figure identities.
- All three printed SQL blocks actually executed in DuckDB 1.4.1. The
  extended lookup matches an independent Python oracle on 200 generated
  histories / 800 prediction rows, including other definitions/snapshots,
  late values, correction ties, TTL boundaries and missing entities.
- Full unit suite: 503 tests in 57 files passed before the final bounded
  storage/table edits. The two affected feature-store tests were rerun and
  passed on the final text. Existing chapter checker was also rerun on its
  final SHA. No stale full-suite result is presented as a fresh rerun.
- Final build: 1967 pages, 21.52 seconds. Final output tests: 135 passed.
  Built link check: zero broken routes/fragments/files. Source coverage
  check passes 210 rows / 56 destinations; this is structural evidence,
  not proof of full course transfer. Scoped git diff --check passed.
- The first unit attempt caught a misplaced new YAML sequence block in
  the registry. It was moved out of the old asset entry, then the full
  suite and build passed. The source figures themselves were not changed.

Logs are `/var/tmp/bookvar-feature-stores-{unit,focused-final,build-final,
output-final,links-final,coverage-final,examples-final,
existing-examples-final}-2026-10-26.log`.

## Rendered result

The actual built page was inspected at 1280×900 and 360×800. All ten chapter
figures load. The three new originals and their surrounding explanations
were inspected. At phone width, small inline labels require the viewer;
3× zoom makes the SVG and PNG text readable and the source is not cropped.
Both horizontal ends of the SVG are reachable (0 / 658 pixels). The lineage
table's 480px row area scrolls inside 313px; its right edge was reached at
167 / 167. All five display formulas fit their 313px containers, with zero
KaTeX errors. Code blocks scroll internally. Document width is 345px in the
360px viewport: no page-wide horizontal overflow.

EN was inspected as an explicitly labelled fallback containing the new
sections and images. Its viewer controls are localized in English. No
English-body file or authored translation was changed. RU preview is left
open; temporary viewport overrides are reset. Screenshots are under
`docs/audit/screenshots/2026-10-26-feature-stores-*.png`.

## Remaining scope

This unit accepts these additions, not the complete Harvard chapters,
labs or slide decks. Resource-management/isolation source sections were
read for comparison; the missing thematic integration remains open.
Practice 17 still has no container build, actual metrics scrape, gRPC or
load experiment. No Feast deployment, distributed backfill, model-quality
experiment, RL rewrite, translation, stage, commit, push or deployment was
performed. The broader textbook goal remains active.
