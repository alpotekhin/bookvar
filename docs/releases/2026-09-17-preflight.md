# Release preflight — 17 September 2026

Target: `origin/dev` and its existing GitHub Pages workflow. The user approved
publication on 17 September. `main` is not a release target.

## Integrated revisions

- `14270e1` records the completed editorial work and the acceptance evidence in
  `docs/audit/2026-09-15-completion/final-acceptance.md`.
- The release merges `origin/dev` at `d2d1148`, retaining its expanded chapter on
  stateful sequence models, 20 illustrations, research overview and four paper
  notes. No remote commit is discarded or rewritten.
- The merged chapter also retains all nine Stanford illustrations, three course
  unit anchors, the corrected SSM direct term, RWKV/RetNet worked examples and
  explicit cache arithmetic. Gated DeltaNet uses a consistent state orientation
  and applies global decay before the directional correction.
- Two old heading fragments are retained as aliases for inbound Qwen/Mamba
  links. Jamba uses the previously reviewed original SVG instead of the cropped
  raster. The registry contains 776 unique assets; the new use of the Jamba SVG
  is recorded.

## Fresh local verification

Build: `PUBLICATION_BASE_PATH=/bookvar pnpm --dir site build`, completed
17 September at 16:26:35 Europe/Moscow, exit 0. Astro generated 1,949 routes;
including the standalone 404, the output contains 1,950 HTML files.

| Check | Result |
| --- | --- |
| Publisher unit tests | 205/205 passed |
| Site unit tests | 60/60 passed |
| Astro check | 0 errors, warnings or hints |
| Built-output tests | 12/12 passed |
| Built internal routes, fragments and files | 0 broken links |
| KaTeX error scan across 1,950 HTML files | 0 error pages |
| Stanford/Berkeley importer regressions | 45/45 passed |
| Course-ingestion ledgers | Both courses passed |
| Source coverage | Passed: 210 rows, 59 destinations, 215 MOOC pages, 212 figures, 34 Harvard labs, 35 decks, 67 HSE artifacts |
| Original audit reconciliation | 442 exact findings across 280 pages, passed |
| Historical content snapshot | 315/318 hashes unchanged; only the merged sequence chapter and two indexes differ |
| Merge example arithmetic | RWKV outputs 2, 3, 5.2; states 10.5/1.75; dense/hybrid KV 16/4 GiB; recurrent state 6.75 MiB |
| Working diff whitespace | Passed |

The initial post-merge publisher check caught two renamed section destinations.
The aliases were added to the chapter and the full publisher, build and link
checks then passed. No unresolved-link exception was added.

Browser sample: the new contents page and merged sequence chapter were opened
from the built site. The chapter contains 35 images, 127 rendered mathematical
expressions and no KaTeX errors. Its original Jamba SVG was visually checked at
desktop width and 390 px: no horizontal page overflow, oversized images or
broken loaded images. This is a bounded release check, not a fresh visual audit
of all 1,950 pages.

## Verification boundaries

The temporary Python/Lean environment used on 15 September no longer exists;
its 18 example tests were not rerun in this release preflight. Their previous
results remain dated in the original acceptance report, and the relevant
chapter/practice files retain their recorded hashes. No GPU benchmark or full
English translation is claimed.

Local JavaScript checks ran on Node 25.2.1. The unchanged GitHub workflows pin
Node 22.12.0 and pnpm 10.32.1; successful local checks do not replace checking
the actual remote workflow outcome. This file records preflight only, not a
claim that a push or deployment has already completed.
