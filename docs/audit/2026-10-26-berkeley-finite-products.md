# Berkeley: finite products and missing assumptions

Date: 2026-10-26. Local editorial work only; no staging, commit, push or deploy.

## Scope and revision

Chapter: `00 Учебник/17 Tools и Agents/70 Формальные доказательства и математические агенты.md`.
SHA-256: `a81cede17e8726c611b8455d35bc296a917d4d3012632a73214ea00b6a6841fe`.
The bounded addition is `Berkeley finite products 2026-10-26`, with anchor
`finite-products-and-missing-assumptions`.

The accepted chapter before this addition has SHA
`6a6912ffa8cf76d54452572734970180a3b2bb1e472055fd87d28da573e6bf50`.
Removing the bounded addition and its source metadata/bibliography changes
restores that exact hash. No former Lean/Coq blocks or 31 earlier figure
references changed. The historical formal-math report remains unchanged;
the references metadata retains both revisions and their separate evidence.

## Explanation and source comparison

Read and visually inspected Berkeley meeting 8, physical pages 94–107:
the infinitude-of-primes title, AIME 2020 II problem 10, the zetamath
channel reference, prime factorizations, geometric factors, Euler product,
finite-case frame and the final zero-product/prime-counting frames.
The PDF provides neither a Lean proof file nor an AlphaProof search trace.
No trace was reconstructed or invented.

| Source sequence | Integrated explanation and worked case | Limits retained |
|---|---|---|
| p.95, AIME | Cubic sum, substitution, all divisors of 332, original remainders, answer 239 | Multiplication by 4 is only a necessary condition; n=327 has remainder 100, not 17 |
| pp.98–100, factorization | Two primes and exponents 0–2; nine denominators, exact 91/36, nonprime 2/4 counterexample | Finite exponent rectangle differs from an ordinary partial sum; multiplicities cannot be discarded |
| pp.100–104, geometric factors | Fixed primes 2 and 3 with unbounded exponents; at s=2 the sum is 3/2 | Finite primes do not imply a finite indexing set |
| p.94 topic and pp.98–103 | A finite set of all primes would bound every harmonic partial sum; dyadic blocks contradict that bound | Does not assign a finite value to zeta(1) |
| p.103 and Mathlib | Full original English module introduction; finite-prime lemma, omitted tail, HasSum/HasProd and inherited f(1)=1 | Norm-summability, f(0)=0 and Re(s)>1; equality alone is not a convergence statement |
| pp.105–107 | Distinguish primes, trivial/nontrivial zeros and pole; precise non-integer-x explicit formula; prime powers 2,3,4 | Symmetric zero limit, multiplicity, trivial-zero correction and endpoints; no claim to prove the analytic formula with the finite code |

Mathlib v4.19.0 is pinned to
`c44e0c8ee63ca166450922a373c7409c5d26b00b`.
Read the introductory documentation, finite-prime and tail lemmas,
the final Euler-product proof and completely multiplicative variants.
The complete introductory prose, not a synthetic summary of it, is retained
in English and checked for whitespace-normalized identity to the original.
The code explicitly adds f(1)=1 beyond the compact module introduction.
Analytical conditions were cross-checked with NIST DLMF §25.2 E1/E11/E12
and Terence Tao, 246B Notes 4, Theorem 8. The notes are linked, not mirrored.

Source-code SHA:
`18ae67ac1cf4af8b95293b338da3339554c3f6ea470564516982b2eba4af81b3`.
The new source card, navigation route and source index are wired to
`/sources/papers/mathlib-euler-products/`.

## Facts, arithmetic and code

`python3 publishing/tools/check_formal_zeta_examples.py` passes:

- Both actual Python fences from the chapter execute.
- 400 cubic sums and 45 finite rational-product cases agree independently.
- Duplicate denominators for factors 2 and 4 preserve multiplicity.
- Harmonic dyadic lower bounds checked at 12 finite values.
- Original English, source/license bytes, PDF hash and image dimensions match.
- Re-extracting the two JPEG streams with `pdfimages -j` produces the same bytes.
- Exact restoration of the accepted baseline preserves all old code and figures.

The first execution caught the draft's erroneous remainder 183 for n=327;
the chapter, table and both assertions were corrected to the computed 100.
No new Lean/Coq fences were added. Lean 4.19.0 is not currently installed;
unchanged code inputs retain historical execution evidence only. No new
Mathlib build, AlphaProof run, model training or model benchmark is claimed.

## Style and continuity

The complete prior chapter was read before the addition; the final chapter
was read in sequential ranges. The new section follows the distinction
between checking formal actions and checking the problem statement.
It supplies concrete examples before abstract convergence conditions and
connects to the following autoformalization section. `no-ai-slop` and its
evaluation checklist were applied to editorial prose, without editing the
quoted Mathlib explanation or removing mathematical qualifications.
Figure captions explain the factorization mechanism; byte-level extraction
details are kept in provenance rather than filling the reader's captions.

## Figures and rights

Two full native JPEGs, not cropped/rerendered slide screenshots:

| File | Physical page / PDF object | Original pixels | SHA-256 |
|---|---|---|---|
| prime-factorization.jpg | 99 / 638 | 2048 × 1046 | 040dcc2c0f2ad172478ca185a1863fbacf02d4276b489e6d2db22a42b95803d7 |
| prime-geometric-factors.jpg | 100 / 645, repeated on 104 | 2048 × 1037 | f479139f3e1d2da57bfd7fdab31fabc218310e5af0df0cd56f9adbb60adb9946 |

Source PDF SHA:
`94bd68c2fdb534d57459a8b96f2a80ada96fd14a58b36626016d4a9f601866ab`.
The gray lower fade is part of each original image, not a site crop.
The lecture refers to the zetamath channel; the exact video containing
each source image has not been established. The existing user permission
record is retained, not upgraded to a named image license. Mathlib's
Apache 2.0 applies to its separate source text, not these course images.
The Apache license and a separate NOTICE are published locally.

## Publication checks and visual QA

Final build after the caption revision: 1965 pages, Complete at 06:18:41,
23.24 seconds. Full unit suite: 444 tests in 43 files, 48.43 seconds.
Final output suite: 129 tests, 4.08 seconds. Berkeley importer: 31 Python
tests; offline source check passes. Built links: zero broken internal
routes, fragments or files. Source-coverage and course-ingestion checks
pass; these structural checks do not prove whole-course explanation quality.
Scoped `git diff --check` passes.

The final revision was inspected at 1280×900 and 360×800. Both images load
with their original natural dimensions; desktop width is 632 px and
the aspect ratio is preserved. Mobile page width/scrollWidth is 360/360.
All 10 display formulas fit the 328 px content column; no KaTeX errors.
The two new tables fit. The unchanged four-column system-comparison table
scrolls internally (328/583); this turn does not claim a fresh acceptance
of every old figure or table. Mathlib's original long identifiers wrap
inside the quote without overflowing the page (327/327).

Both native images open in the viewer. At 4× the first reaches its left
sum symbol and right ellipsis; the second reaches both original blank
margins and the central geometric-series terms. The lower fade is visible
in the source and the viewer; no formula was redrawn to remove it.
In the mobile miniature the small terms require the viewer, where the
prime bases and exponent labels are legible. This is a remaining limitation
of the original layout, not a claim that every label reads at thumbnail size.

The chapter link opened the source card; its reverse link eventually
returned to the exact chapter anchor. An automation observation timed out
before that return completed; the subsequent AX state and screenshot
confirmed the destination without a direct-navigation substitute.

Evidence screenshots in `docs/audit/screenshots/`:

- `2026-10-26-finite-products-desktop.jpg` — final caption, source figure and finite formula.
- `2026-10-26-finite-products-mobile.jpg` — second source figure and final caption.
- `2026-10-26-finite-products-viewer.jpg` — actual 4× source terms on mobile.
- `2026-10-26-finite-products-conditions-mobile.jpg` — explicit formula and its qualifications.
- `2026-10-26-finite-products-source-mobile.jpg` — original English Mathlib prose.

Logs: `/var/tmp/bookvar-zeta-final-{build,unit,output,links,coverage,ingestion}-2026-10-26.log`,
`/var/tmp/bookvar-zeta-import-{check,tests}-2026-10-26.log`.
NOTICE and Apache-license copies match the published local files byte-for-byte.
No deployment took place.

## Remaining scope

The course ledger now has 143 integrated / 6 covered-existing /
35 source-only / 28 excluded semantic units, and 49 integrated /
85 source-only / 27 excluded visual records. Of the integrated semantic
units, 108 are slide-derived and 35 are reading catalogue entries.
These counts are not a claim that every paper has been reproduced.

The whole-book goal remains active. Other course semantic mappings,
previously recorded source/rights findings and unaccepted content remain
open. PPO/DPO/RLVR/GRPO are not part of this edit. No translation authored;
EN changes are catalogue counts only.
