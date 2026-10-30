# Geometric selection: explanation and publication review

Scope: new canonical chapter 41d, navigation between accepted 41c and 42,
source provenance and the actual rendered reader. No RL/EN-body edits,
delegation, stage/commit/push/deploy or external training service.

The preceding goal turn was a status verification of ready RL, not content
progress. Current work fills the geometric-selection gap in H-V1-09 rather
than reopening accepted PPO/DPO/RLVR/GRPO.

## Source-to-explanation comparison

| Mechanism | Selected source read | Reader explanation and worked case |
| --- | --- | --- |
| Max-min coverage | Erickson §J.10; Sener–Savarese v4 §§3–4.3 | Fixed L, new budget b, nearest distances and radius; six-point 0/1/4/5/9/10 trace |
| Farthest-first | Entire Erickson §J.10, pseudocode and six-panel figure; full Google class | Running min update, explicit availability mask, stable IDs, seed/no-seed cases, representation cost |
| Metric guarantee | Erickson Theorem J.7 and full proof; Sener §4.3 | Original English proof, next-point witness; separate b+1 pigeonhole proof for fixed L; no accuracy claim |
| Metric choice | MOOC quiz body; Euclidean/cosine identity worked directly | Original scaling SVGs correctly called k-means, normalized-vector ordering, squared factor four, 0/60/120 counterexample |
| Outlier variant | Sener v4 §4.3 robust mixed-integer formulation | 0/1/2/3/100 exhaustive example, ordinary radius 3 vs one-exception radius 1; rare group vs corruption not inferred |
| Loss-bound boundary | Sener v4 §4.1–4.2 and its explicit CE limitation | Bounded/Lipschitz/zero-training-error assumptions, empirical core-set term, L2 CNN lemma vs experiment CE |
| Query coreset | Har-Peled §§31.2.1–31.2.2, full definitions | Weighted loss family and all-query quantifier; no automatic neural-network CE transfer |
| Grid compression | Entire Erickson §J.11; Har-Peled §31.1 and 31.5 | Cell diameter, additive query bound, separate optimal compressed solver, continuous vs pool centres, dimension dependence; six points to two reps |
| Herding | Welling §§1–4 in full | Exact dynamic weights and telescoping identity, bounded/sublinear conditions, full four-step binary trace, repeats vs distinct subset |
| Mean matcher distinction | Welling update plus explicit algebra, not an attributed new theorem | Raw mean vs second moment; expanded next squared error and candidate-norm term, correct w0=mu condition |

The source QMD read boundary is unchanged: Harvard H-V1-09 lines 1–2240;
next unread line 2241. No claim that this pass read 2241–4741. Native H-F1-09,
semi/self-supervised reconciliation and the remaining course sections stay open.

## Reuse and visual integrity

New original image: complete six-panel Gonzalez sequence plus caption from
Erickson p.14, extracted at 288 dpi to 1500×856. Source PDF and permissions
HTML pinned by SHA in `assets.json` and the central registry. Poppler crop is
reproduced from the original in the registry validator, not merely hash-checked.
No generated replacement diagram. Two existing MOOC SVGs remain unchanged and
are attributed as scaling examples, not a measured k-center improvement.

The current Har-Peled PDF differs from an older indexed rendering and has no
explicit license notice in its extracted text. Reference only; no figures or
extended prose copied from it. Sener/Welling paper figures are also not copied.
The Google code's Apache license does not license the Sener paper figures.

## Complete-reader review and style

All saved chapter lines read in three contiguous ranges, including every
program, formula, quotation, caption and footer. No-ai-slop SKILL and eval read
in full. No invented gain, vague attribution, generic introduction, decorative
diagram or definition-only replacement. Meaningful technical contrasts and
mathematical limits are retained. Original quotations remain in English;
no English-locale body or translation is authored.

Later mathematical layout edits, if needed, must be read and verified before
acceptance. The comparison above concerns explanation quality; unit counts and
source-coverage status alone cannot close it.

## Executed examples

Checker: `publishing/tools/check_geometric_coreset_examples.py`.
First execution completed the assertions but failed JSON serialization of a
NumPy integer in the report; fixed to explicit integer conversion. This was
not reported as a passing run. Six chapter programs execute; the second uses
the preceding function intentionally. Independent tests compare small metric
problems with exhaustive discrete optima and every greedy choice, check row
permutations/duplicates/invalid inputs, additive grid error and herding prefixes.

The unchanged historical Google class body is executed with an explicit NumPy
pairwise-distance adapter and flatten_X base (imports stripped by AST). It
reproduces a selected-seed assertion and duplicate result `[0,0]` on zero features.
This is a control-flow test, not an installed historical sklearn stack.

## Accepted revision and execution evidence

Final chapter: 649 lines, SHA-256
`db32ea0fad9b98fe920150fda9408c857678c629f6f962df0878e1235f5590c8`.
Full reader read, followed by explicit rereading of the modified coreset
quantifier, expanded mean-error formula, prefix-error example, attribution and
anchors. The two long formula displays were split without changing the math.

Final actual checker log:
`/var/tmp/bookvar-geometric-coresets-examples-final-2026-10-28.log`, exit 0:
six programs, 1272 exhaustive metric cases, 1272 stable-ID permutations,
288 zero-optimum cases, 14 invalid inputs, 120 grid queries, eight herding
prefixes and 16 next-error expansions. Ordinary/outlier exact radii 3/1,
six-point queries IDs 620/412, duplicate features return distinct IDs 10/20/30.
No empirical label-efficiency, accuracy or speed superiority claim is made.

The first full unit run had 609 pass and one failure because the sidebar
label omitted “coreset” compared with the contents. After label alignment,
the full suite passed 610 tests in 93 files (48.17s). After final math/caption
edits, all 15 affected tests in four files passed again (0.760s). The preserved
41c body is verified by restoring its sole next-edge change and recovering
accepted SHA c1970f…62f81. The 42 body is similarly preserved, recovering the
pre-edit SHA 3889d5…028c. Ready RL is not rewritten.

Final build: 1975 pages, 23.04s, exit 0. Final output suite: 160 tests, exit 0.
Built-link check: 0 broken routes/fragments/files, exit 0. Source-coverage
check: 210 matrix rows / 59 existing primary destinations, exit 0; this is
structural evidence, not acceptance of all course content. Scoped tracked
diff-check passed and the new authored files have no trailing spaces.

Logs are `/var/tmp/bookvar-geometric-coresets-{unit-accepted,scoped-final,
build-final,output-final,links-final,coverage-accepted}-2026-10-28.log`.
Every launched job was observed terminal; no previous handle remains to wait.

## Browser evidence on the final reader

Desktop 1280: complete original image is 631.98×360.66, source 1500×856;
MOOC SVGs preserve their ~501×434 aspect ratio. Six panels and original caption
are not clipped. All three meaning-bearing images load.

Mobile 360: body 360/360, all ten displays 328/328, zero KaTeX errors and
zero over-wide inline-math elements. The two formerly long displays fit after
notation-only line breaks. The first table is intentionally horizontally
scrollable, width 328 / content 585: positions 0 and 257/257 inspected, including
the final Radius column 10/5/1. The herding table fits without horizontal scroll.

Farthest-first viewer at 2.5×: image region x=0/516 and x=516/516 both reached;
height 611/611, both rows available. Caption scrolled to 91/91; final green-arrow
explanation reachable. Both original MOOC plots opened and inspected at mobile
width, including titles, full plot boundaries and changed assignments. On the
current final page both plots have their own author/title/direct source link.

Actual footer clicks: 41d→41c→41d→42→41d. English route opened: explicit
“not available in your language yet” fallback, ten formulas, zero KaTeX errors;
not counted as an authored English translation. Viewport reset after inspection;
main reader left at the new chapter.

Screenshots under `docs/audit/screenshots/`:
`2026-10-28-geometric-coresets-desktop.png`, `…-mobile-left.png`,
`…-mobile-right.png`, `…-mooc-original-mobile.png`, `…-mooc-scaled-mobile.png`,
`…-table-mobile-right.png`, `…-mean-error-mobile.png`.

## Remaining scope

This closes the new geometric reader explanation, not all H-V1-09/H-F1-09.
Next: reconcile already-read semi/self-supervised passages against current
chapters and primary FixMatch/representation-learning explanations, then resume
the Harvard original at 2241. Native figure inventory, remaining Harvard/HSE/EDLS
units, model atlas and deferred English parity remain part of the active goal.
No stage, commit, push, deploy, delegation or live infrastructure action occurred.
