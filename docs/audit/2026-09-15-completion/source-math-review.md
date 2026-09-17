---
title: "Source-archive math rendering review"
type: editorial
status: editorial
last_updated: 2026-09-15
---

# Source-archive math rendering review

Scope: the source routes reported by the existing static build, not canonical chapters. Original course Markdown, embedded complete source, the importer and publication adapter build.ts are unchanged. Implementation is render-only in publishing/adapter/math.ts, integrated by site/astro.config.mjs.

## Result and evidence

- Existing non-EN output: 36 affected routes, 105 KaTeX error nodes. The EN projections duplicated the issue; counts below avoid counting both languages as independent source defects.
- Direct rendering of actual generated files with real file URLs: 41 changed projections, 105 errors before and 0 after. The 41 are 35 Harvard decks, one Harvard lab and five notebook/script pages. Five Harvard decks had no parse errors but contained the same lossy mathematical extraction.
- All 35 decks passed the strict old-projection match. The 30 formerly failing decks account for 95 errors; the compression lab accounts for one, and other archives account for nine.
- The evidence JSON records every route, generated-input SHA256, original archive-file SHA256, retained-source payload SHA256, before/after error and heading counts, idempotence and exact preservation of complete source blocks.
- Focused regression tests cover the configured renderer, real Astro getEntryInfo/getRenderFunction pipeline, full frozen bodies, both locale scopes for the notebook cases, canonical exclusion, unknown-route exclusion, code spans/fences, mismatched projections, source-only operator mutation, frozen manifest checks and cache invalidation.
- Full static build is owned by the parent task. A direct renderer result is not presented as proof that old dist HTML has already changed. Final build acceptance is tracked separately.

Final local verification: 57 focused tests pass. Astro check completed at 19:11:09 local time with 0 errors, 0 warnings and 0 hints; source/layout warnings printed during content sync are not TypeScript diagnostics. The production content store at site/node_modules/.astro/data-store.json, modified at 19:12:09, contains zero KaTeX error nodes across all entries. An earlier follow-up accidentally inspected the obsolete development store at site/.astro/data-store.json (August 3); its five old errors were not current production results. Temporary diagnostic logging was removed. Final rehash: all 41 archive files match their recorded input hashes; none was edited.

Machine evidence: [source-math-render-evidence.json](source-math-render-evidence.json). Tests: site/tests/source-math.test.ts and site/tests/markdown-rendering.test.ts.

Final implementation SHA256:

- publishing/adapter/math.ts: 56c15937886914a2d9a2a0b6f86cc152f5b6f91a5fa54b080706107dde3173c3
- site/astro.config.mjs: fefe2c8075a3ff0afa7ebcf5792007851b3da33e6ed50f7ac4f8fa52324adb9b
- site/tests/source-math.test.ts: 229585fd8efe6d8d4195bb4f21341fcce071512650e4b0d5bc0fe73414ef7b4e
- site/tests/markdown-rendering.test.ts: ac66ab728d654dda2dac80446d9a92c26f95fd3079297e642ec2022259bac521

## Actual causes

| Source | Original errors | Cause and treatment |
|---|---:|---|
| Harvard slides: 30 of 35 decks | 95 | The importer removed arbitrary TeX commands inside mathematical spans; empty dollar pairs then captured following prose. Recover original mathematical spans from retained TeX under a frozen SHA and exact old-projection check. |
| Harvard compression lab | 1 | The extractor treated plain Python triple-quoted strings as unprocessed text and replaced brace arguments with ellipses. Recover the plain literals, decode only the observed doubled-backslash form, and escape literal identifier underscores in TeX text. No Python code is executed. |
| HSE trees homework | 1 | A multiline display expression closed on the formula line; place the existing opening/closing delimiters on their own lines. |
| HSE random-features homework | 2 | Same display-fence defect caused downstream expressions and prose to be swallowed. Only the exact frozen expression is normalized. |
| HSE trees seminar | 2 | Nested dollar math inside TeX text collided with Markdown dollar tokenization. Use supported parenthesized TeX math delimiters inside the same text expression. |
| LF diversity metrics | 3 | Literal reco_df/train_df identifiers had unescaped underscores in text mode. Escape those underscores without changing the metric formulas. |
| Scikit-learn MOOC feature importance | 1 | Escaped currency dollar adjacent to a Markdown closing dollar was tokenized as a delimiter. Use the equivalent textdollar command. The same lexical normalization is needed for nine Harvard currency expressions. |

These were not a general lack of KaTeX support for fraction, sum or gradient macros. Blanket macro suppression, global permissive error handling, removal of formulas and error-class renaming were rejected.

## Deterministic original-source reconstruction

The frozen importer is publishing/tools/import_harvard_labs_slides.py. Its clean_tex function ends with a general command-removal expression. Applying that operation to mathematical input loses operators and their semantics. The renderer mirrors the frozen projection only to verify an exact match; it then protects original dollar-delimited mathematical spans while applying the same existing prose projection. Display fences are put on separate lines. Mathematical currency escapes receive an equivalent TeX command to avoid a Markdown lexical collision.

The guard has two independent conditions. First, the retained original payload plus one terminal LF must match the route-specific SHA256 pinned from the frozen labs-slides-manifest.json. All 35 TeX sources and the one Python source match that independently stored manifest: 36/36. The importer used rstrip before embedding, hence the explicitly restored terminal LF for original-file verification. Second, recomputing the old readable projection must exactly reproduce the present readable block. A source-only sum-to-product mutation and an editorial change to the readable block both fail closed. Neither normalization edits the original source section.

Examples inspected against the complete retained original:

- 05 Источники/Courses/Harvard ML Systems/slides/vol2/05_distributed_training.md:275 has the damaged g_k expression; line 1759 retains its fraction, summation, membership relation and gradient. Line 279 versus 1763 provides the global-gradient counterpart.
- 05 Источники/Courses/Harvard ML Systems/slides/vol1/00_course_overview.md:624–626 has the damaged Iron Law; lines 2077–2079 retain the original fractions and underbraces. Restoring its delimiters also recovers the following headings: rendered heading count rises from 14 to 38.
- 05 Источники/Courses/Harvard ML Systems/labs/vol1/lab_10_model_compress.md:61–68 has the broken runtime branch; lines 1520–1526 preserve the original cases expression inside a plain Python literal.

All three retain a complete original code section and the exact upstream link at commit 45ecc8d82fcae70c149cdce550d3b3d3411df913: [overview TeX](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/slides/vol1/00_course_overview/00_course_overview.tex), [distributed training TeX](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/slides/vol2/05_distributed_training/05_distributed_training.tex), [compression Python](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/labs/vol1/lab_10_model_compress.py). No local complete-slide PDF was found for these samples; remote directory verification was unavailable. Reconstruction therefore relies on verified retained source, not an unverified PDF or invented replacement. Existing source links and licenses remain unchanged.

## Cache and integration boundary

Astro 7.1.0 content-layer.js serializes the resolved processor options for its content-store digest, while glob.js reuses a rendered entry when its source digest is unchanged. An imported normalizer function can change without changing either old cache key. Therefore the actual SHA256 of math.ts is included in processor.options; every implementation or pinned-hash change now invalidates cached HTML on the next content sync. The processor name stays unified so Starlight still installs its own plugins. Tests verify the hash survives Astro config validation.

The wrapper keeps all render options, frontmatter and returned metadata. It scopes by generated source-course file URLs, including the EN prefix. Canonical routes and arbitrary source routes are not normalized. The SHA map also prevents broad use of the TeX projection on newly imported or changed decks.

## Verification commands

```sh
cd site
npm exec vitest run tests/source-math.test.ts tests/markdown-rendering.test.ts --silent
npm exec astro check
```

The initial notebook regression run failed all five source cases before implementation. The Harvard reconstruction initially left nine currency-related failures across four decks; the equivalent currency command eliminated those without changing mathematical values. The strict-source hash mutation check was added after independent review.

No long training, GPU execution, dependency installation, importer regeneration, full site build, commit or deployment was performed by this subtask. Full frozen bodies are rendered in the tests; code and source bytes are preserved.

## Limits

Zero KaTeX parse errors is not proof that every archived theorem, derivation or exercise is mathematically correct. This task repairs rendering of the original expressions; substantive source errata remain separate. TeX layout wrappers, speaker-note comments and original figure references still reflect the existing mechanical projection; it is not a complete Beamer visual renderer. Unsupported future input is retained rather than silently rewritten. KaTeX strict-mode layout warnings and the separate toc-language highlighting warning are distinguished from parse-error nodes and from TypeScript diagnostics.

Technical references: [remark-math documentation](https://github.com/remarkjs/remark-math), [KaTeX supported functions](https://katex.org/docs/support_table.html). Installed behavior was tested with remark-math 6.0.0, rehype-katex 7.0.1, KaTeX 0.16.47 and Astro 7.1.0; live documentation may describe a newer KaTeX release.
