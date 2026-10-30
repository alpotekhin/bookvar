# ML workflow: iteration budget and feedback clocks, 26 October 2026

## Revision and preserved content

`00 Учебник/19 Deployment, Reliability и MLOps/01 ML workflow.md`: 832 lines,
SHA-256 `329710d26fc09b696036c217d07f7151e9b0879394411dbc21e54e24d1911869`.
The accepted 635-line chapter was read in full before editing. After the two
additions the whole 830-line revision was reread sequentially in ranges
1–210, 211–420, 421–630 and 631–end. That reading found the old HTTP-200
completion sentence needed qualification; the final two-line extension was
read and linked to the tested streaming practice. No unrelated body was rewritten.

The preservation test removes only the two marked sections, restores the two
frontmatter dates and reverses that exact HTTP clarification. It recovers
accepted SHA `498a80237b35db4bf8376d2bc0dade25d6065f955a71958ce53d4ec9c515115e`.
All six original Google figures, four preceding examples, nine legacy anchors,
route, index and previous/next navigation are retained.

The adjacent complete MLOps chapter was read for context, not rewritten or
newly accepted by proxy. Its current SHA
`0e63a2dde76d8eae9630036aac9542f63c48cf0f15cb98a659ca2dd5fa76f454`
matches its 20 October acceptance report. Accepted RL is not reopened.

## Source scope and explanation review

The pinned Harvard source at commit
`45ecc8d82fcae70c149cdce550d3b3d3411df913` is reused from
`05 Источники/Originals/ml-workflow-2026-10-20/harvard-ml-workflow.qmd`, SHA
`25c7e8255c6db5c96c8906fcd759063803aac278d83f33f849c1f61e302d98bd`.
No second full source copy was created. New selected scopes read completely:
the IterationTax calculation and its callout, and Multi-scale feedback,
lines 2131–2142. Other sections were inspected for overlap and source claims,
but neither a full new Harvard chapter transfer nor complete lab/deck reading
is claimed. The Marimo lab's initial templates, entry point and helper imports
were inspected only; no Marimo execution.

The source calculation fragment is copied exactly after class dedentation.
The complete first feedback paragraph is copied exactly after removing only
its index markup. Original English, variable names and comments remain.
Both new Harvard-derived sections, including their added examples, are
explicitly distributed under CC BY-NC-SA 4.0. The original root license notice
and current CC deed were read; old Google notices remain separate.

The first example walks from 26 weeks to 4,368 nominal hours, 26 weekly
iterations and the chosen 100-useful-experiment cap. It computes 98.9% versus
99%, distinguishes percentages from percentage-point gains and identifies
the chosen ceiling and additive-gain assumption. A 0.03-point counterexample
gives 93%, so fast iterations are not presented as a guarantee that a smaller
model wins. Nor is the source's 0.1-point final gap a statistical test result.
Full-cycle data/evaluation/review time is distinguished from the training call.

The second example links operational, data, quality and architecture loops.
Its concrete three-record case separates prediction time, a declared 72-hour
outcome horizon and label availability. At hours 72/80/132, mature counts are
2/2/3, usable counts 0/1/2, coverage 0/0.5/2⁄3 and error ratios unavailable/1/0.5.
The text explains every transition, the missing-label denominator and why
that changing observed error is not evidence of model improvement. Label
versions and selection bias remain distinct concerns. The clinical cadences
in the source quote are explicitly illustrative, not prescribed intervals.

No-ai-slop review preserved reasoning, examples, caveats, citations and the
licensed source wording. New text was not substituted for the already accepted
explanations. The final HTTP wording also separates status 200 from a streaming
terminal event, with a direct link to practice 14a.

## Figure selection and rendered inspection

Two original margin SVGs were compared. The unlabelled iteration curves were
not published: they supply neither axes nor evidence for the numeric table.
The complete feedback-timescales SVG was selected and copied byte-for-byte,
SHA `ae5afe218654f1895e8b24b4cd4903fc8985c2bcacfcb285cae9e333c2cb9879`.
Its original viewBox is `0 0 72.63 99.9`; no crop, redraw, translated label or
geometry change. Exact source, commit, author, license and hashes are in the
adjacent assets.json/NOTICE.md and global registry.

A source-specific CSS rule displays the vector at up to 20rem instead of its
small physical point dimensions. The actual page renders it whole at 320×440
on desktop 1280 px and about 313×431 on a 360 px phone. Labels and the log-scale
annotation are readable. The mobile viewer also shows the entire SVG at 1×,
with visible close/zoom controls and a scrollable attribution caption. Quick
Look PNGs were inspection intermediates only; the website retains the SVG.

New tables have 480/514 px scrollable content inside 313 px table viewports.
The cohort table was focused and scrolled with ArrowRight to the actual
201/201 px limit, exposing the rightmost error column. All seven code fences
scroll internally. Document width is 345 inside viewport 360; no page overflow
or KaTeX errors. All seven original figures load at their source aspect ratios.

RU route: `http://127.0.0.1:4330/textbook/deployment/ml-workflow/`.
The actual EN route shows the explicit missing-translation notice and both
new sections/figure as fallback. No EN file was edited. Proof screenshots:
`screenshots/2026-10-26-workflow-feedback-desktop.png`,
`screenshots/2026-10-26-workflow-feedback-mobile.png`,
`screenshots/2026-10-26-workflow-feedback-mobile-table.png`.

## Executed verification and remaining scope

The existing checker now executes all seven current chapter Python fences and
retains its 800 independent checks of the four prior mechanisms. The new
standard-library checker separately executes the three new fences and checks
300 independent budget cases plus 300 cohort cases, boundary inputs, timestamp
translation invariance and source-fragment equality. Both ran successfully
again after the final clarification. This is deterministic teaching arithmetic,
not training, clinical validation, a live monitor or Marimo execution.

Full unit run: 501 tests in 56 files passed; after the final wording/import
clarifications, nine affected workflow/HTTP tests passed again. Final site
build: 1967 pages. Final built-output checks: 135 passed. Built links: zero
broken routes/fragments/files. Source coverage: PASS, 210 matrix rows and 56
direct-linked destinations; these structural counts do not establish complete
course integration. Build and post-build gates are recorded in the logs under
`/var/tmp/bookvar-workflow-feedback-*-2026-10-26.log`. The source matrix marks
H-V1-03 partial-reviewed rather than claiming complete book/lab/deck transfer.
Scoped tracked-file whitespace checks pass.

This closes the selected extensions, not the global textbook goal. The rest of
the Harvard workflow and MLOps sources, labs/decks, other course integration
and source/rights findings remain open. Container/gRPC/load work in practice 17
is still unexecuted; it is not silently marked complete by these arithmetic
checks. No agent, translation, staging, commit, push or deployment was used.
