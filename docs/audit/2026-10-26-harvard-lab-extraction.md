# Harvard lab extraction and Edge memory accounting

Date: 2026-10-26. Local edits only; no staging, commit, push or deploy.

## Finding and cause

While comparing the remaining Harvard Edge lab with the textbook, its
published static explanation contained `M_{\\text`…`}` instead of the
original memory labels. KaTeX returned no error for these damaged expressions:
its successful parse did not establish source fidelity. The initial browser
observation recorded the altered annotations and visible placeholder output.

`extract_marimo_markdown` treated Python strings with a regular expression.
It replaced brace-enclosed LaTeX labels as though they were Python fields,
left doubled f-string braces in the output and retained Python-level doubled
backslashes instead of decoding the string. It also missed single-line
Markdown calls. The source Python archived on each page was intact.

The revised importer parses Python AST without executing source imports,
code or format expressions. String constants therefore follow Python's
escaping rules. F-string literal braces survive; actual runtime fields get
explicit symbolic names and a legend retaining their Python expressions,
conversions and format specifications. Unknown runtime-only message variables
are not invented. Markdown math detection ignores escaped currency and code.

Literal multiline Markdown is shown as the static explanation. Other UI text
and unevaluated templates remain in a collapsed block. Conditional OOM and
success messages are not presented as consecutive experimental outcomes.
HTML interfaces, plots and runtime execution are explicitly not reproduced
by this static layer. Original source is retained separately on every page.

Three original compression-lab display formulas used unescaped underscores
inside `\text{...}`. For typesetting only, the readable layer escapes those
literal underscores; code and subscript operators are not rewritten. The
original Python retains its exact bytes. This formatting change is disclosed
in the source-page notice.

## Scope and source integrity

Rebuilt 34 Marimo lab pages from their existing, SHA-verified originals.
No repository clone, fresh course download, external lab execution or model
training was needed. All 34 source hashes matched before and after rebuild.
The 35 Beamer pages received only the corrected license-notice link
(`LICENSE` → `LICENSE.md`). Their original TeX and readable extraction were
not otherwise changed; all 69 original code blocks match the source manifest.

The pinned Harvard source is
`45ecc8d82fcae70c149cdce550d3b3d3411df913`. The live pinned `LICENSE.md`
confirms CC BY-NC-SA 4.0. The 34 static pages and source notices remain in
English; no translation was authored. A repaired source extraction is not
recorded as full thematic integration of every lab.

## Edge chapter addition

Chapter: `00 Учебник/19 Deployment, Reliability и MLOps/03 Edge и federated deployment.md`.
Before: 784 lines, SHA
`80cebdaee98f4a89c709e6aaf50d15af64e2b5045b5aadc516adf1065fbb3f42`.
After: 871 lines, SHA
`9038a3d22ef6b746b7a99720542763f94b8117641c84543499433305a001c7e3`.

The complete original English “Math Peek: Training Memory Amplification”
from H-L2-11 is integrated at `harvard-mixed-state-memory`, not just linked.
The quoted explanation is checked against the decoded AST constant. Its
two equations are split across lines, with no mathematical-token changes.
Attribution, pinned lines, license and a link to the local original follow it.

The source's “5–9×” approximation does not follow from its listed arrays:
FP16 weights plus FP16 gradients plus two FP32 moments already give
`1+1+4=6` times the weights-only baseline before activations. The original
claim remains visible as a quotation; the discrepancy is explained directly
after it, not silently corrected or presented as a universal measurement.

For ten million parameters, the worked example gives 20 MB weights,
20 MB gradients and 80 MB moments, subtotal 120 MB. Adding 32 MB activations
gives 152 MB or 7.6× the 20 MB baseline; a separate 40 MB FP32 master copy
would give 192 MB or 9.6×. Workspace and application memory are excluded
explicitly. This accounting is distinguished from the earlier FP32-only
160 MB example and from measurements of a device or the Marimo simulator.

The prior five Python blocks and all six figure references are unchanged.
Old long display formulas were reformatted for mobile. After removing the
bounded addition and verification dates, the rest of the document matches
the accepted baseline when only display-math whitespace/alignment is
normalized (fingerprint
`5ffa537e1b9d09083d2efa38629e969a5187fb4a20cc6ec16e173ba2b821fa61`).
This is not described as byte identity of reformatted mathematical markup.

`no-ai-slop` was applied to the editorial bridge and accounting explanation;
the source quotation, original UI messages and code were not rewritten for
style. The entire original chapter was read; the addition and adjoining
memory discussion were read again after editing. Existing source findings
and the 20 October report were consulted, not treated as new acceptance of
the whole Harvard course.

## Checks

- Regression tests were run against the old importer first and failed on
  literal TeX, raw strings, f-string braces, symbolic fields and missing
  preservation checks. The corrected suite has 13 passing tests.
- All 69 archived Python/TeX code blocks match their manifest SHA-256.
- `--rebuild-labs --check` verifies deterministic output without writes.
- The full unit suite passes: 444 tests in 43 files, 47.30 seconds.
- The first two whole-registry checks timed out while the macOS power log
  recorded long sleep intervals. A process-scoped `caffeinate -is` rerun
  passed without changing test assertions or timeouts. The assertion ends
  with its process; no permanent power setting was changed.
- The default Homebrew Python lacks NumPy. The existing bundled runtime,
  not a new installation, executes `check_edge_examples.py`: five actual
  chapter blocks, 1200 independent cases, the additional mixed-dtype
  arithmetic, source hashes and reproducible figure crops.
- Authored chapter/tool/test edits pass ordinary `git diff --check`.
  Imported UI text retains 13 intentional Markdown two-space hard breaks;
  the derived-source check disables only `blank-at-eol`, while still checking
  blank-at-EOF and space-before-tab. Original code is not whitespace-cleaned.

Final build includes the updated gap-matrix note: 1965 pages, Complete
at 09:53:23, 21.59 seconds. Final output suite: 130 passing tests, 3.99 seconds,
including checks of all 34 lab routes, intact math and the canonical addition.
Built links: zero broken internal routes, fragments or files. Source coverage
passes with the existing 210-row/57-destination structure; that is not a
new semantic acceptance of every destination.

Logs are under `/var/tmp/bookvar-harvard-ast-*-2026-10-26.log`;
the delivered build/output/link logs are `final4-build`, `final4-output`
and `final4-links`. The source-preservation suite and deterministic extraction
check were rerun after the last source change. The final Edge checker reports
the delivered chapter hash above and all five actual Python blocks executed.

## Rendered verification

The final chapter and repaired Edge lab were inspected in IAB at 1280×900.
The old lab's broken labels and doubled backslashes are replaced by the
original decoded TeX; the memory components and fractions render correctly.
Runtime templates are initially collapsed and their control opens the named
symbolic fields. The chapter's real link opens the local original lab;
browser Back returns to the new stable chapter anchor.

At 360×800 the final Edge page has width/scrollWidth 360/360, zero KaTeX
errors, and all 13 display formulas fit their 328 px (311 px within quote)
containers. All mathematical tokens of the old formulas are retained;
only alignment, whitespace and line breaks change. The new three-column
memory table fits at 328/328. Two old wide tables retain internal horizontal
scrolling; this turn does not claim their redesign. All six earlier images
and their source bytes are preserved; no new figure or synthetic replacement
was introduced for the arithmetic addition.

Screenshots in `docs/audit/screenshots/`:

- `2026-10-26-harvard-lab-restored-desktop.jpg` — real decoded source formulas.
- `2026-10-26-harvard-edge-budget-desktop.jpg` — final source quotation and attribution.
- `2026-10-26-harvard-edge-budget-mobile.jpg` — split formula and original English explanations.
- `2026-10-26-harvard-edge-accounting-mobile.jpg` — explicit 120/152/192 MB accounting.

The viewport override was reset and the chapter left open as a local demo.
No live app, lab kernel, Android device or federated fleet was run.

## Remaining work

The Edge lab's simulator, shared helper, HTML instructions, plots, scheduling,
privacy and final policy-review parts still need their own source-to-chapter
comparison. Its complete execution and full slide transfer are not claimed.
Beamer extraction still uses a lossy TeX cleaner and needs a separate repair
and visual audit. HSE and Efficient DL mappings remain open beyond the
specific accepted chapters. No whole-book or English-version completion is
claimed; accepted PPO/DPO/RLVR/GRPO remain outside this change.
