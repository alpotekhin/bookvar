# Chapter 58b: benchmarking, SLOs and inference operations

Date: 2026-10-20. Canonical RU chapter:
`00 Учебник/14 Inference и оптимизация/58b Benchmarking, SLO и эксплуатация inference.md`.

Final SHA-256: `633f22fd09c7fe64a86d67b05cba487d18f7f48d20e254b89e2c94677b758f53`.
Local route: <http://127.0.0.1:4330/textbook/inference/benchmarking-slo/>.

## Scope and reading

The whole original chapter was read before editing. The revised chapter was
read in order, including a final reread of lines 1-375, 376-560 and 561-end
after the technical corrections. The final file has 750 lines; a terminal
blank line was removed after render validation without changing rendered
content. This report accepts this chapter's stated scope, not the entire book.

Deliverables are the canonical chapter, seven executable educational examples,
17 pinned or dated source snapshots, four new source-figure files, provenance
records, a CPU/source-fidelity checker and this audit. No English chapter,
commit, push or deployment was made. Unrelated dirty work was preserved.

## Mechanism and source review

| Topic | Primary-source scope actually read | Explanation and check in the chapter |
|---|---|---|
| Measurement boundary and harness | Harvard benchmarking: Micro/Macro/End-to-end, Benchmark harness, System specifications, Run rules; LoadGen roles, responsibilities and LoadGen over Network | Kernel time versus full forward versus client E2E; 80+20 to 80+10 ms Amdahl example; untimed QSL preparation is not end-user latency |
| Streaming timestamps | Full pinned vLLM streaming completion handler; AIPerf v0.13 streaming metrics through prefill throughput; original Aleksa Gordic figure | First-token and last-token boundaries, O-1 intervals, one-token output with undefined TPOT, stream-close overhead, six tokens in three SSE events and 60 versus 75 ms normalization |
| Workload generation and client throttling | Complete vLLM get_request, calculate_metrics, benchmark through gather and goodput parsing | 20 closed-loop clients reduce offered rate when latency rises; concurrency-one dispatch moves scheduled arrivals 0/.1/.2/.3 to actual sends 0/1/2/3 s; client wait remains visible; finite normalized gamma schedules are not literally independent Poisson gaps |
| Goodput and percentiles | DistServe abstract/introduction and complete Figure 1; Harvard inference-metrics and statistical sections | Six-request cohort, 2/6 eligible requests passing joint limits; marginal 90% limits may yield only 80% joint success; DistServe maximum rate under attainment is a different quantity |
| Tail estimates and aggregation | Harvard selected statistics sections; independent mathematical derivations stated with assumptions | Nearest-rank versus linear quantiles, token- versus request-weighting, 100 zero-failure samples allowing a roughly 2.95% upper failure bound, average replica p99 of 505 ms versus merged p99 of 10 ms |
| Paired lengths and dependent traces | Complete Stanford CS336 lecture 10 throughput function; complete MLCommons agentic article body | Same input/output marginals but more than nine times the counted decode attention key positions; explicitly not a prediction of whole-model speed; recorded dependent trajectories distinguished from a live agent |
| Phase accounting | Full EDLS model setup and timing-function cells; the original notebook has no saved outputs | Full generate includes prefill; 120 ms prefill and 300 ms full generation for ten outputs leave nine steps and 20 ms TPOT, not ten decode steps |
| Capacity and reliability accounting | DistServe source setup, Google Implementing SLOs selected complete sections and Alerting on SLOs sections 4-6 | Last measured passing rate is not a global maximum; overloaded 8 versus 6.2 RPS accumulates 108 requests in 60 s; finite cohort/drain accounting differs from a steady window |
| Error budgets and alerts | Complete Google burn-rate, multiple-burn-rate and multiwindow discussions | 150 bad out of 10,000 at 99.9% SLO gives burn rate 15; 14.4 for one hour consumes 2% of a 30-day budget under constant traffic; long and short alert windows use AND |

The source archive README and references.json record the exact section scopes,
URLs, commits/versions, dates and hashes. The full Harvard book, full linked
courses and unread paper sections are not claimed as reviewed. A fresh CS336
download was hash-matched with the canonical imported lecture. Source programs
were read as data, not installed or executed.

The vLLM and AIPerf terminology is not homogenized: vLLM's event-gap array and
AIPerf's scalar ITL / ICL distribution are version-specific. Successful-only
latency populations remain distinct from all eligible requests. Timeouts,
rejections and incomplete streams do not silently disappear from SLO accounting.

## Source figures and correction

All five displayed illustrations were checked against their recorded sources.

- The original MLPerf LoadGen SVG is unchanged, pinned and accompanied by the
  Apache-2.0 license, copied byte-for-byte.
- The inherited Aleksa Gordic latency image is unchanged; its attribution and
  existing registry metadata are retained.
- DistServe Figure 1 is re-extracted from the original PDF page 2 at a 6000 px
  maximum page dimension. Both panels and the original caption are retained.
  The output is 1890 x 1860 px. The checker independently reproduces the crop
  from the archived PDF and exact crop coordinates.
- Google SRE Figures 5-4 and 5-6 are unchanged complete WebP originals under
  CC BY-NC-ND 4.0. No crop, translation, recoloring or reconstruction was made.
  Source HTML records both original image locators and the license link.

The previous local DistServe plot does not match the original points and
layout. It was removed from the narrative and marked source-identity-unverified
in the registry, rather than continuing to call it an original crop. Its bytes
were preserved. The new extraction retains the existing permission record;
this audit does not assert a new named license or fresh rights clearance for
DistServe. The notice states that limitation.

## Separate editorial pass

The no-ai-slop skill and eval were read. The full finished chapter was reviewed
for generic introductions, mixed-language filler, unsupported universal claims,
empty contrasts and definition-only fragments. Technical terminology, source
labels, code and mathematics were retained. Heading changes preserve three
explicit older anchors and the previous/next navigation.

The text explains mechanisms through timestamp records, intermediate
calculations and counterexamples rather than shortening them into definitions.
Hypothetical measurements, historical paper measurements and current source
implementation observations are explicitly distinguished. Assertions about
ignore_eos, prefill warmup, output token counts and cross-host clock subtraction
were narrowed to the actual conditions.

This checklist is an editorial pass, not proof of human authorship or teaching
quality.

## Executed checks

The chapter checker executes the actual seven Python blocks and compares them
with independent CPU calculations:

- 300 timestamp cases and 300 cohort aggregations;
- 1500 nearest-rank quantile cases;
- 300 limited-concurrency schedules and overlap invariants;
- 500 explicit context-work sums;
- 300 phase-accounting cases;
- 300 error-budget/burn-rate cases.

Negative and boundary cases include a one-token response, invalid timestamps,
uncompleted requests, marginal versus joint SLOs, bundled SSE tokens, unequal
replica populations, zero-failure uncertainty and the overloaded queue balance.
The same run verifies 17 snapshot hashes, five figure identities, one reproduced
PDF crop, three unchanged licensed source images, the LoadGen license copy,
the notebook's lack of saved outputs and stable navigation/anchors.

Observed results:
- CPU/source checker: pass.
- Asset-registry tests: 39 passed.
- Publication build: 1961 pages, complete.
- Built-output tests: 12 passed.
- Built-link check: zero broken internal routes, fragments or files.
- Targeted git diff --check: clean after removing the terminal blank line.

No GPU benchmark, serving-engine run, hardware performance measurement,
network experiment or task-quality evaluation was performed. The examples
test definitions and arithmetic; they do not reproduce the paper's performance.

## Browser evidence

The rebuilt page was reloaded and checked in the actual in-app browser at
1365 x 900 and 390 x 844. All five images loaded with expected natural
dimensions; desktop and mobile views of each were visually inspected.
Three display equations rendered, with no KaTeX errors. On mobile each display
equation fits its 343 px content area. The document stays within the viewport.

The two wide tables use their own horizontal overflow, rather than widening
the document. An actual horizontal scroll of the first table changed scrollLeft
to 158.5 px within a 343 px viewport and 502 px table content. Code blocks also
retain horizontal overflow.

Small labels in source plots are not fully legible inline on a phone.
Clicking the Google multiwindow image was verified to open the actual
1410 x 654 file. The full original remains accessible instead of being replaced
by a thumbnail or redrawn schematic.

Proof images:
- /tmp/bookvar-benchmark-desktop-2026-10-20.jpg
- /tmp/bookvar-benchmark-mobile-2026-10-20.jpg

The desktop proof was embedded in the conversation. The viewport override was
reset and a chapter preview tab retained.

## Remaining work

No unresolved mechanism gap was identified within this review's stated scope.
Fine source labels on mobile still require opening the original. Inherited
DistServe rights were not freshly audited. English and live publication are
unchanged. The larger textbook goal remains active.

The next full page, 58c Queueing и capacity planning, has been read in its
entirety to prepare the next edit. It is not accepted by this report. Its next
review must cover queue-model assumptions, variability, resident KV, measured
replica capacity, burst/cold-start behavior, admission and failure reserve,
with original source visuals and worked examples.

