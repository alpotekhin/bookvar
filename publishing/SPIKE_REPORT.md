# Publication spike audit

Audit date: 2026-07-17. Baseline: `7c72bec3cc92fb5bfa348c46c0e84409101a6ced`.

## Decision

**No-go: remain in the spike and revise the repository packaging before public-repository setup or full-corpus generation.** Starlight remains a suitable stack, but the acceptance gate is not met. A real clean checkout cannot generate the site because a referenced image lives under ignored `raw/`; the landing page still duplicates its title and contains prototype copy; the MoE diagram is not legible enough; and the user has not approved the visual result. The adapter now preflights before replacing its publication-owned asset tree, and CI now orders unit tests, build, and Pagefind-dependent output tests correctly, but neither change resolves the clean-checkout asset blocker.

Vercel remains the preferred host *after* these blockers are resolved: the build is static and host-compatible, and no spike result justifies changing hosts. No Vercel project, remote, credentials, or deployment should be created at this gate.

## Clean-checkout evidence

The temporary detached worktree was created from the baseline and removed with `git worktree remove`; source files were not copied into it.

| Command | Outcome |
|---|---|
| `git worktree add /tmp/ml-dl-handbook-spike HEAD` | Passed; detached `7c72bec`. |
| `pnpm install --frozen-lockfile` | Initial attempt could not start because pnpm was absent (`exit 127`). A temporary `/tmp` installation of pnpm 10.12.1 was used; the rerun passed in 2.51 s. |
| `pnpm --dir publishing test` | **Failed** in the prescribed order: 38 passed, 6 skipped, one suite failed because `site/dist/pagefind/pagefind.js` does not exist before a site build. |
| `pnpm --dir site check` | Sandbox attempt hit a local IPC `EPERM`; an unrestricted rerun reached the product code and **failed** because `raw/papers/attention-is-all-you-need/images/transformer_self_attention_vectors.png` is absent from Git. |
| `pnpm --dir site build` | Not runnable in the clean worktree for the same missing tracked asset. |
| `git worktree remove /tmp/ml-dl-handbook-spike` | Passed. |

The missing file exists only in the working copy and `git check-ignore -v` attributes it to `.gitignore:4:raw/`. This is a repository/package defect, not a network or sandbox failure.

## Post-defect local verification

The visual review found duplicate H1s on every generated page. Starlight already emits the sole page H1 from frontmatter, so the adapter contract now removes exactly one source-level H1 when it is the first content heading after optional blank lines. It does not remove an initial H2, a later H1 after prose, or any additional H1 in the body. Regression tests were added first and observed failing, including the real case where the short source H1 does not equal the longer frontmatter title; authored sources remain unchanged.

With the ignored local assets available:

| Command | Outcome |
|---|---|
| `pnpm --dir publishing test -- build.test.ts` | Red: the short nonmatching leading H1 remained; green rerun: 7 files, 47 tests passed, including three safety cases. |
| `pnpm --dir site check` | Passed: 0 errors, warnings, or hints; 3.82 s. |
| `pnpm --dir site build` | Passed: 12 HTML pages, Pagefind indexed 12 files; 2.68 s wall time. Vite emitted a large-chunk warning and sitemap was skipped because `site` is unset. |
| `pnpm --dir publishing test` (after the final H1 contract change) | Passed: 7 files, 47 tests. |
| `pnpm --dir site check` (root verification after final H1 change) | Passed: TypeScript passed; Astro reported 0 errors, 0 warnings, and 0 hints. |
| `pnpm --dir site build` (root verification after final H1 change) | Passed: 12 pages built and 12 files indexed by Pagefind. |
| `rg -l '^# ' site/src/content/docs/generated \| wc -l` | 0: none of the ten generated Markdown pages retains a source H1. |
| `git diff -- '00 Учебник' '01 Справочник' '02 Атлас моделей' '03 Исследовательские линии' '04 Вопросы' '05 Источники' '06 Практика' Concepts Courses Papers MOC.md Timeline.md` | Empty: generation changed no authored source. |
| `find site/src/content/docs/generated -type f \| wc -l` | 10, matching 10 manifest entries. |
| `du -sk site/dist`; `find site/dist -type f \| wc -l` | 6,724 KiB (6.6 MiB), 209 files. |

Navigation is generated from `publishing/navigation.yml`; the route test and built sidebar cover all ten manifest entries.

## Pages and visual evidence

The browser review covered the required representative set at desktop and/or 390 px mobile widths:

1. textbook index;
2. self-attention chapter;
3. MoE module map;
4. RLVR/verifiers chapter;
5. RAG module map;
6. DeepSeek model-family page;
7. DeepSeek-R1 paper page;
8. LLM question index.

Additional evidence: the attention page loaded three local images at 632 px, rendered 50 KaTeX nodes, and had no horizontal overflow. The RLVR page exposed 21 subsection headings, four code blocks, and 22 KaTeX nodes. Mobile navigation was responsive with no obvious clipping. Mermaid source rendered, but MoE labels were too small on desktop and especially at 390 px.

After the final adapter change, root verification rebuilt the corpus and browser-audited all ten manifest routes: every generated page has exactly one DOM H1, including the DeepSeek-R1 page whose short source heading differs from its frontmatter title. The hand-authored landing page still renders duplicate `ML & DL Handbook` titles and placeholder prototype copy. **Visual approval remains pending; do not treat this review as user approval.**

## Syntax and unresolved inventory

Supported and tested:

- frontmatter validation and publication metadata;
- ordinary Markdown, fenced code, inline/display math (KaTeX), and Mermaid source/rendering;
- Obsidian wikilinks by full path or alias, display aliases, and heading anchors;
- local image embeds for PNG, JPEG, WebP, SVG, and GIF, copied under `/assets/`;
- callouts mapped as `note/info/abstract -> note`, `tip -> tip`, `warning/caution -> caution`, and `danger/failure -> danger`;
- unresolved targets remain visible as plain text and are reported rather than silently dropped.

Unsupported or degraded Obsidian constructs in this adapter:

- links/embeds whose targets are outside the ten-page publication manifest do not link or embed; they degrade to visible text;
- non-image embeds are not transcluded;
- image extensions outside the allowlist are not embedded;
- block references/block embeds are not converted;
- other Obsidian-only plugin syntax has no adapter contract and must be added explicitly before full-corpus migration.

`publishing-report.json` is the complete machine-readable instance inventory: **90 unique unresolved targets across 7 pages**, all classified as known out-of-manifest targets (not parser-unknown syntax): textbook index 44; backpropagation 1; self-attention 2; MoE map 1; RAG map 3; DeepSeek-R1 paper 9; LLM questions 30. The other three generated pages have none. Full-corpus generation must either add routes or define an intentional external/plain-text policy for all 90.

The three copied attention images are explicitly attributed in the source page to Jay Alammar and the reviewed license registry records *The Illustrated Transformer* as CC BY-NC-SA 4.0. That is only a spike-corpus license pass and is insufficient for a public repository: Git tracks **32 files** under `00 Учебник/Assets/Figures/`, including apparent paper-page and UI extracts, without per-asset manifests. The count is reproducible with `git ls-files -z '00 Учебник/Assets/Figures/*' | tr -cd '\\0' | wc -c`. These files were neither deleted nor relicensed in this spike because redistribution rights require research. Before any remote is created, a whole-repository binary registry/audit must record author, source URL, license and license URL, modifications, and intended use for every asset; unverified assets must be removed or quarantined, and the repository license must distinguish third-party material.

Dependency manifests now pin every direct dependency to the exact version represented by `pnpm-lock.yaml`. CI installs with `pnpm install --frozen-lockfile`; lockfile changes require explicit review alongside manifest changes. This is a reproducibility baseline, not a complete supply-chain gate: vulnerability and dependency-license auditing, reviewed update automation, and an exception/remediation policy remain required before public CI.

## Search evidence

The built Russian Pagefind index passed both examples in the automated suite:

- `Transformer` returns `/textbook/transformer/self-attention/`;
- inflected `нейронов` returns `/textbook/foundations/backpropagation/`.

This demonstrates the required Russian inflection example for the spike; broader relevance and typo tolerance remain future evaluation work.

## Acceptance matrix

| Gate | Status | Evidence |
|---|---|---|
| Clean-checkout automated tests pass | **Fail** | CI ordering is corrected (unit tests → build → output tests), but generation still fails because a required image is ignored under `raw/`. |
| Eight representative pages render | Pass | All ten manifest routes were rebuilt and browser-audited; each has exactly one DOM H1. |
| Formulas and local images render in the spike | Pass with packaging blocker | Browser/build/tests pass locally; clean checkout lacks an image. |
| Internal links and fragments are release-safe | **Fail** | 90 unresolved targets degrade to plain text; emitted fragment IDs have not yet been validated against built HTML. This is degraded navigation and a release failure, not a passing link gate. |
| Russian inflected search works | Pass | `нейронов` test passes. |
| Generation leaves authored content unchanged | Pass | Exact authored-directory diff is empty. |
| Whole-repository asset licensing is complete | **Fail** | The three emitted spike images pass the spike-only check, but 32 tracked figure assets lack required per-asset manifests. |
| Manifest drives navigation | Pass | Ten generated pages equal ten manifest entries; sidebar is generated. |
| Unsupported Obsidian constructs listed | Pass | Construct classes and all 90 instances are accounted for above/report JSON. |
| User approves visual result | **Fail/pending** | No user approval; known landing and Mermaid defects remain. |

## Required next actions

1. Complete the whole-repository asset registry/audit for all 32 currently tracked figure assets; remove or quarantine any asset without verified redistribution permission. Do not create a public remote based on the three-image spike pass.
2. Put all publication-required assets in a tracked/licensed location (or deterministically acquire them with license verification), then repeat the exact clean-worktree sequence.
3. Add a post-build checker for emitted routes, assets, and fragment IDs; resolve or explicitly allowlist all 90 currently unresolved targets. Until then internal navigation fails the release gate.
4. Add dependency vulnerability/license scanning, reviewed update automation, and a documented exception/remediation policy while retaining frozen-lockfile installs.
5. Remove the landing-page duplicate H1 and placeholder copy; improve responsive Mermaid legibility.
6. Rebuild and repeat desktop/mobile review after the remaining landing/Mermaid fixes, then obtain explicit user approval.
