# Chapter 58c: queueing and capacity planning

Date: 2026-10-20.
Canonical RU chapter:
`00 Учебник/14 Inference и оптимизация/58c Queueing и capacity planning.md`.

Final SHA-256: bbfb1d67720cfe7b264c57c8d754fa0ebb147925cfd25e9ba4e66a63f3603c74.
Local route: <http://127.0.0.1:4330/textbook/inference/queueing-capacity-planning/>.

## Scope and full reading

The entire original chapter was read before editing. The final 827-line chapter
was reread sequentially in untruncated ranges 1-280, 281-560 and 561-end after
the mechanism, wording and responsive equation corrections.

This accepts the explanations within this chapter's defined scope. It does not
accept the entire textbook, complete transfer of all linked courses, English
parity, interactive practice or a live release. No English chapter, commit,
push, deployment or infrastructure mutation was made.

The deliverables are the canonical chapter, ten executable teaching examples,
ten source snapshots, five source figures, their provenance and license
evidence, a CPU/source-fidelity checker, targeted registry tests and this report.
The existing route and previous/next links remain unchanged. Seven older
heading anchors were retained explicitly.

## Explanation review against selected sources

| Mechanism | Source scope read | Worked explanation and limitation |
|---|---|---|
| Request time budget | Harvard serving latency/queueing material, plus chapter 58b definitions; Google deadlines discussion | 1.5 s TTFT budget minus .1/.6/.1 s leaves .7 s queue budget; per-request addition is distinguished from summing marginal p95s |
| Little's Law | Complete Harvard Model Serving Little's Law section, including examples and code | Four requests with arrival, start and finish times; 12 request-seconds of occupancy, 5 request-seconds of waiting, 7 s full-cohort horizon; exact finite area identity before stable long-run averages |
| Residual service and variability | All 14 MIT 6.263J slides read as text; complete derivation and examples on slides 2-6; complete source pages 5/6 visually inspected | Unfinished service plus waiting jobs, residual triangles S-squared/2, PASTA and the waiting-time fixed point; same .1 s mean and 80% load yield .2, .4 or 1.658 s mean waiting |
| M/M/1 and model assumptions | MIT slides 2-6 and complete Harvard utilization relationship section | 5 to 9 RPS at service rate 10 changes mean response .2 to 1 s; M/M/1 doubles M/D/1 queue-only waiting, not total time; it is not a conservative bound for arbitrary service variability |
| Batch formation | Complete Harvard batching-tax section and example code | Four arrivals show first wait .3 s versus mean .15 s; batch-32 at 500 RPS gives 62 versus 31 ms before executor waiting; no-timeout full-group assumptions explicit |
| Replica queue placement | Harvard multi-server considerations and the already reviewed scheduling/PD chapter context | An occupied replica and an idle replica show private-queue starts 10/11 versus shared-queue starts 0/1 s; known release times and movable unstarted jobs are assumptions, not a real router performance claim |
| KV admission | Existing KV and PD mechanisms combined with explicit block arithmetic | 16 GiB pool, 128 KiB/position, twenty 4096-position states, 1024 future positions each; five new requests become three after honoring earlier promises; per-sequence page rounding, physical allocation and logical promises distinguished |
| Resource-weighted flow | Complete Google Pitfalls of Queries per Second and prior source-backed P/D context | 70/30 mixture gives 28160 input tokens/s, 3968 outputs/s and 3958 remaining sequence decode steps/s; 256 MiB times 10 RPS is 2.5 GiB/s; independent resource limits are not jointly achieved capacity |
| Fleet reserve | Complete Harvard Capacity planning section; Google Server Overload example and both original figures | 22 RPS at measured 6 RPS/replica gives four normal, five N+1, six with 80% post-failure use, seven for two lost replicas; placement domains, transition load and linear scaling assumptions explicit |
| Burst recovery | Complete Harvard Cold start, Reactive scaling, Warm pool management and Scaling response time analysis sections | 30 versus 24 RPS for 60 s accumulates 360; capacity 36 then takes another 60 s to drain; B/C is a frozen-capacity estimate, not actual waiting across a future capacity change |
| Deadlines and scaling metric | Complete Google Queue Management, load shedding and Latency and Deadlines sections; complete Kubernetes Algorithm details | 1.4 versus 1.7 s first-token estimate; shared deadlines and cancellation; 1200 total queued tokens at target 300/replica gives four, not twelve replicas |
| Retry amplification | Complete Google Retries and Deciding to Retry sections | Independent .2 failure, three attempts gives 1.24 attempts and .992 success; congestion correlation invalidates a fixed-probability promise; three attempts at three layers means 27, three retries means 64 |

The source snapshot manifest records the exact Harvard commit
45ecc8d82fcae70c149cdce550d3b3d3411df913, selected line ranges and dated official
web pages. It does not conflate these with the different commit of the full
imported Harvard source edition. The CMU book page was inspected for its table
of contents and retained as further reading, not represented as a fully read
or imported book.

The full requested explanation was compared with selected sections, rather
than accepted because links or source markers exist. No universal GPU startup
duration, warm-GPU billing discount, safe utilization threshold or p99 prediction
is inferred from an illustrative source scenario.

## Figures, provenance and source corrections

Five source illustrations are embedded:

- Eytan Modiano's two original MIT diagrams, slides 5 and 6, CC BY-NC-SA 4.0.
  Complete source pages were inspected before selecting the diagram regions.
  Final outputs are 2720 x 860 and 2420 x 1000 px, rendered directly at a
  6000 px maximum PDF page dimension, not upscaled from bitmaps. Diagram labels,
  arrows and geometry are unchanged. Surrounding prose/formulas are not part of
  these crops; the chapter typesets its own derived equations.
- Mike Ulrich's Google SRE Figures 22-2 and 22-3, complete original 767 x 440 WebP
  files under CC BY-NC-ND 4.0.
- Alejandro Forero Cuervo's Figure 21-2, complete original 673 x 902 WebP file,
  with the same Google license.

The Google files are byte-for-byte copies, not crops or translated artwork.
Attribution, direct figure/source links and actual licenses accompany each.
Official HTML snapshots retain exact original image locators and license
links. The MIT resource page retains the direct lecture PDF and license.
The checker reproduces both MIT crops pixel-for-pixel from the archived PDF.

The former
`00 Учебник/Assets/Figures/curated/ml-systems/harvard/inference/queuing-hockey-stick.svg`
was inspected and removed from the narrative. It labels rho/(1-rho) as
queue-only population although this is total M/M/1 population, calls the growth
exponential and claims unbounded growth above 85%. The file is preserved and
its systems manifest points here as a review reference, not a reader placement.

The former
`00 Учебник/Assets/Figures/curated/ml-systems/harvard/inference/ttft-tpot-timeline.svg`
was removed to avoid repeating the timing explanation already established in
58b. Its retained manifest reference is this audit; no fresh factual or license
clearance for unrelated old figures is claimed. The older batching-strategies
SVG is not embedded in the revised chapter either. Shared source bytes remain
unchanged.

An initial MIT crop contained a strip of adjacent slide prose. Visual inspection
caught it, and the final crop excludes that strip while retaining every diagram
label. The manifest and reproduced-pixel check use only the corrected crop.

## Separate wording review

The no-ai-slop skill and eval were read before substantial editing.
The entire final chapter was checked separately for generic setup,
definition-only fragments, misleading simplifications, empty contrasts,
unsupported attribution and mixed-language filler. Explanations retain
intermediate states, units, assumptions, mathematical steps and examples.

The new opening uses the actual queue balance, including the equal-rate case:
raising service to equal arrivals stops growth but does not drain backlog.
A vague failure heading was replaced with the concrete cluster-overload
mechanism. A defensive bibliography disclaimer was removed; the audit, not the
reader's source list, records what was and was not read.

Technical terms and normal Russian mathematical punctuation are retained.
Source artwork labels, code identifiers and quotations were not rewritten
for style. The checklist is editorial guidance, not proof of human authorship.

## Executed checks

The CPU checker executes all ten actual chapter blocks, not an independent
replacement notebook. It then tests 300 cases each for:

1. FCFS schedules and occupancy/queue areas;
2. service-time moments and residual-work identities;
3. batch formation;
4. private/shared replica schedules;
5. physical block rounding and future reservations;
6. resource-weighted workload flows;
7. minimal replica count under lost capacity and headroom;
8. fluid work conservation and split-interval burst recovery;
9. first-token deadline accounting;
10. retry outcomes and expected attempt counts.

Boundary/negative cases include invalid arrival order or duration, unstable
M/G/1 input, impossible moments, partial groups, per-sequence page fragmentation,
invalid replica reserve and retry probability. Numerical checks cover units,
one-first-token-per-prefill accounting, HPA normalization, SLO-conditioned
request cost, 360 accumulated requests and 60 s recovery.

The same run verifies ten source hashes, five figure hashes, five license
evidence identities, two reproduced PDF crops, three unchanged original WebPs,
removed legacy figure paths and stable aliases/navigation.

Observed final evidence:
- CPU/source checker: ten blocks and 3000 randomized cases passed.
- Asset-registry tests: 40 passed, including new negative provenance tests.
- Adapter generation and actual Astro site build: complete, 1961 pages.
- Built-output tests: 12 passed.
- Built-link check: zero broken internal routes, fragments or files.
- Targeted git diff --check: clean.

No GPU run, serving-engine benchmark, workload-capacity measurement, model
quality evaluation or HPA/controller experiment was performed. Conditional
prices and timings are labeled teaching inputs, not current market claims.

## Actual browser review

Final page inspected at 1365 x 900 and 390 x 844 in the in-app browser.
All five images loaded with expected natural dimensions and preserved aspect
ratios. Each was visually inspected in context at both widths; the tall retry
stack's lower portion and caption were also inspected on desktop.

Initial mobile inspection found three display formulas wider than the content
area. They were split into aligned lines without changing the mathematics,
and the final site was rebuilt. All eight final display equations fit their
343 px mobile content area with zero KaTeX errors; document width is 375 px
within the 390 px viewport. Code and tables retain their own horizontal scroll.

An actual scroll moved the first table to scrollLeft 375 px, with a 343 px
client width and 810 px full content. Opening the MIT residual diagram reached
the actual 2420 x 1000 PNG; opening the retry stack reached the actual
673 x 902 WebP. Fine labels on a phone can still require opening an original,
which is not described as full inline mobile label legibility.

Proof images:
- /tmp/bookvar-queueing-desktop-2026-10-20.jpg
- /tmp/bookvar-queueing-mobile-2026-10-20.jpg

The desktop proof was embedded in the conversation. The temporary viewport
override was reset and the chapter tab retained as a deliverable.

## Remaining scope

No unresolved explanatory gap was identified within this chapter's reviewed
scope. Model assumptions, fine-label mobile limits and unmeasured performance
are retained explicitly. The next canonical chapter is ML workflow in the
deployment/reliability section; it is not accepted by this report.

The full textbook goal remains active. No claim is made that the entire book,
source-course integration, translations, interactive practice or live
publication is finished.
