# Neural admission policies and actual CPU receipts, 26 October 2026

## Revision and preserved work

`06 Практика/14 Собрать минимальный inference engine.md`: 1264 lines,
SHA-256 `d8f167f2d531b6d893a3db2a244d9eb0b195918527965fcc79ac26a21d6aaee5`.
The preceding 974-line practice was read in full. Its neural core, receipt
calculation, original source excerpts/figures and Harvard examples remain.
The new module, experiment driver, plotter and explanation were read in full
while preparing them; the final caption was revised after actual page review.

The preservation test removes the bounded new section and restores only the
superseded two-line pending-experiment sentence. It recovers prior SHA
`2adf0af6a979805bd8136d27a075b1e103b0eb3e01c5c640d9e952b0d2e0686e`.
The nested earlier 783/433/31-line preservation checks also pass. The existing
neural-core SHA is unchanged:
`ae4c58df80457d2f0f335dca5ebac4744b10d04dca2277059c146559db01cd09`.

## Explanation and source boundary

The complete EDLS scheduler/metric interfaces were revisited. Their full
homework/API/benchmark reading and immutable source hashes are recorded in the
preceding reports and archive. The original English three-policy assignment
and manual-greedy paragraph remain unchanged. The adaptation retains the full
course MIT notice and links; it is not presented as an unchanged teacher solution.

The worked admission case uses active A, waiting B, one free slot, six free
KV positions, B's five-position reservation and 5/20 ms waiting times. The
decision table separately checks free slots and insufficient KV. The text
explains FIFO, the source single-admission baseline, refill, wait-threshold
admission, one-token completion before decode and same-step decode of new
unfinished requests. A threshold is not promised as a hard queue-time bound.

The full 146-line `edls_admission_reference.py` is visible in the chapter and
identical to the canonical and served copies. External IDs remain the receipt
keys. Cancellation and error are terminal outcomes; reservations and cache
references are released, and an incomplete failing call does not publish
internal candidate tokens. The logical reservation is distinguished from
actual peak memory, including temporary padded batches and copies.

The clock contract was checked against complete selected Python 3.14
`monotonic`, `perf_counter` and `sleep` documentation sections. Only local
processes share this clock origin; no cross-host clock subtraction is claimed.

No-ai-slop review retained source language, equations, concrete transitions,
examples and limitations. After viewing the page, the long procedural caption
was replaced with the actual phase/marker definitions. The prose does not
claim a universal policy ranking from a laptop run.

## Executed correctness and experiment

`publishing/tools/check_edls_admission_reference.py --experiment` ran with
cached Torch 2.14.1, NumPy 2.3.5 and Transformers 4.57.1. It verified:

- 54 deterministic combinations of policy, slots and initial request order;
- source-ID preservation, FIFO/reservation boundaries, zero/one-token budgets,
  capacity rejection, waiting/active cancellation, terminal immutability,
  cache release and recovery after an injected partially mutating call error;
- 15 real neural terminal cases: first/second-token EOS, context limit, zero
  budget and cancellation, across all three policies;
- 54 actual CPU runs: two workloads × slots 1/2/4 × three policies × three
  repeats, with all 648 output sequences equal to independent uncached greedy.

The model has 22,720 random Qwen3 parameters, CPU FP32, one PyTorch thread,
Python 3.14.3 on Darwin arm64. No weights/tokenizer were downloaded. Warmup,
oracle computation and terminal correctness checks occur outside timed windows.
The two fixed manifests contain twelve requests each. EOS is disabled in the
timed token-ID fixtures to hold output work fixed; EOS correctness is tested
separately, and this distinction is explicit in the final text. Prefill-heavy emits
18 tokens; decode-heavy emits 156. The measured duration includes the finite
submission/drain interval, not summed concurrent request latencies.

An initial threaded-producer diagnostic exposed tens-of-milliseconds lateness.
It was not used for the published snapshot. The final driver uses an independent
standard-library subprocess, a readiness handshake and a future common origin.
Its actual submission timestamps pass through a pipe; consumer observation
delay is stored separately. Maximum final submission lateness is still about
5.00 ms against a planned 2 ms spacing. It is reported rather than erased.
Three repeats and shared-CPU noise do not establish a causal policy ranking,
a production p99 or server saturation capacity.

Canonical measurement artifacts:
`06 Практика/Results/edls-admission-2026-10-26/`.
The full JSON SHA is
`9f7b5094faefb3503dc0edf8cf1248612e3a281b0e0b8525b3c2994900067103`;
the scheduler SHA is
`f2808f08819ed700edb0b2a70847cea5898ce1b9614cda32b08d3237bbf5f58c`.
The JSON binds the exact core, scheduler, experiment and actual receipt-fence
hashes. Its medians reproduce the published six-row table. `single`'s one-active
invariant means its cross-column variation cannot be called slot scaling.

The five new unit tests check preservation, complete visible/downloadable code,
source hashes, every recorded cell/ID/outcome/token/time ordering, and measurement
plot provenance. The original five CPU example fences and receipt fence still
pass, including their 300 scheduler and 200 affine-clock cases. The unchanged
core/fence checks preserve applicability of the earlier 316 KV comparisons.

## Actual plot, page and downloads

`phase-timeline.png` is a standard Matplotlib 3.10.7 plot of recorded phases,
not an AI-generated illustration or a replacement for the original HF diagrams.
It uses the final decode-heavy, two-slot, repeat-0 records, explicit phase
patterns/legend and actual submission markers. It is registered as an original
measurement in the adjacent hash manifest, separately from source-copy assets.
The original three HF illustrations remain unchanged.

The 1800×990 PNG was inspected at native resolution and on the final page:
`http://127.0.0.1:4330/practice/14-sobrat-minimalnyy-inference-engine/`.
At desktop 1280 px it is 617×339, whole and uncropped. On mobile 360 px it is
313×172, so small labels need the viewer. Zoom 4× and all horizontal/vertical
extremes were checked: x 0/987, y 0/144, with caption retained. The new tables
are 337/390 px inside 313 px scroll containers; keyboard access reaches their
last columns. Both display formulas fit 313 px, there is no KaTeX error and
no document horizontal overflow. The full code remains a scrollable code block.

Actual HTTP responses for the scheduler and complete JSON match the canonical
SHA values above. Their reader links were observed in the DOM. The EN route
shows the explicit missing-translation notice and current sections/plot/links
as fallback, not a new translation. Viewport was reset, viewer closed and
the final RU tab marked as the deliverable.

Screenshots:

- `screenshots/2026-10-26-edls-admission-desktop.png`
- `screenshots/2026-10-26-edls-admission-mobile-zoom.png`
- `screenshots/2026-10-26-edls-admission-mobile-table.png`

## Publication evidence and remaining scope

486 unit tests in 53 files passed; after the final caption/EOS-scope clarifications all
17 affected fidelity/measurement tests passed again. Final build: 1965 pages.
Final built-output tests: 135 passed; built links: zero broken routes/fragments/
files; source coverage: PASS, 210 matrix rows and 56 direct-linked destinations.
Scoped tracked-file diff checks pass. Logs:
`/var/tmp/bookvar-edls-admission-{unit,targeted,build,output,links,coverage}-2026-10-26.log`.
Actual neural run diagnostics:
`/var/tmp/bookvar-edls-admission-final-cpu-2026-10-26.log`.

This delivers a working neural CPU admission reference and recorded event
experiment inside the practice. It does not deliver an asynchronous streaming
HTTP service, real-tokenizer/checkpoint workload, sampling bonus, physical
peak-memory measurement or GPU/SLO-capacity benchmark. The source matrices
remain partial; original source availability and a green build are not
full course integration. Those requirements and the broader content/rights
queue remain open. Accepted RL, EN source, staging, commit, push and deployment
were not changed.
