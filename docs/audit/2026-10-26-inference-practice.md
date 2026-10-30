# Inference practice: source integration, 26 October 2026

## Delivered revision

`06 Практика/14 Собрать минимальный inference engine.md`: 433 lines,
SHA-256 `a6f5cfb5a68307f7a17c653a68a5830867fb3a5d3e03bd829fed112c71aae30e`.
The initial 31-line EDLS assignment and metric requirements were read in full
and preserved outside a bounded extension. The fidelity test reconstructs
input SHA `5315e35516a200b68cc8d50c1fad635d33dfce549c039b882563f4be1283c165`.
The final practice was read in full, with subsequent unit/wording fixes checked.

## Source and explanation review

The complete Harvard lab (2219 lines) and deck (1940 lines, including notes,
focus formula, questions and backup) were read. All original source files are
pinned to commit `45ecc8d82fcae70c149cdce550d3b3d3411df913`. Full original
source passages on cost equations, state and feasibility are retained in
English and compared byte-for-text against their source sections by tests.
English worked elaborations are an attributed adaptation; the original Russian
EDLS route remains intact. No EN source body was authored or edited.

The extension develops each calculation through inputs, intermediate steps,
result and limitations:

- Cost crossover: 50 million requests/day at assigned USD 0.001/request,
  USD 50,000/day, 40-day crossover, 44.44 days after a 10% saving and
  USD 1.825 million/year saved. Units and optimization fraction are explicit.
- GQA memory: eight KV heads × head dimension 128 × 80 layers, FP16 K/V,
  decimal/binary distinction, quantization-metadata caveat, explicit 8 GB
  reserve, boundary batches 27/13 at contexts 4096/8192.
- Scheduler: four requests traced through closed cohorts and iteration-level
  refill; arrivals, useful rows and completion times are shown. Equal tick
  cost, already-prefilled requests and absent scheduler overhead are explicit.
  Logical slot occupancy is not labeled GPU utilization, MFU or MBU.
- Feasibility: all constraints must pass simultaneously. Memory capacity is
  separated from a measured request rate. Four assigned configurations have
  explicit cost, p99 and quality assumptions; none is a real benchmark.
- Cold-start backlog: explicit 300-second delay, 6000 queued requests and
  another 75 seconds to drain under a deterministic fluid model. Rejections,
  expiry and censored latency measurements are discussed.

The final section connects these examples to the original EDLS model engine
and names the required correctness and measurement checks. It does not claim
that the reference scheduler contains a neural model, allocator, cancellation
or working GPU inference. The practice is expanded teaching material, not a
completed real engine or reproduction of the Marimo UI.

Read primary evidence: Orca's original USENIX abstract for iteration-level
scheduling; complete HF “Chunked prefill” and “Continuous batching” sections
with their figures and algorithm. Full source read scopes and hashes are in
`05 Источники/Originals/inference-practice-2026-10-26/`.

## Source errors not promoted to facts

The deck computes `100/(8*5)` as 0.25 instead of 2.5. Its P2C asymptotics do
not establish exact p99 values; its reasoning-token discussion does not prove
an automatic switch to compute-bound execution. Its focus KV formula needs
head dimension when H denotes a head count. The lab uses hard-coded scheduler
scores and quality rules; selecting a routing widget does not alter the numeric
serving calculation. These limitations are documented, not hidden by copying
the source's conclusions.

The Harvard batching figure's printed utilization numbers are not derived
from its bars. The continuous-batching figure had already been withdrawn for
a zero-waste claim and was not reintroduced. The cold-start SVG's phase labels
and timeline markers disagree. Instead of redrawing these figures, the
practice reuses the two complete original licensed HF diagrams.

## Execution and preservation

`python3 publishing/tools/check_inference_practice_examples.py` passes on
the actual five Markdown fences. It independently checks rational cost math,
KV boundary batches, all emitted tokens/arrival constraints over 300 scheduler
fixtures, invalid IDs/lengths/slots, idle gaps, candidate failures, backlog and
the queue-slide arithmetic. Six reviewed source helper definitions are
AST-extracted and called without importing or executing the Marimo module.
Source proxy results are not presented as measurements.

The no-ai-slop review preserved the original source voice, formulas and useful
contrasts. Added prose explains calculations rather than labeling them
important or compressing them into definitions. Literal currency dollar signs
were replaced with explicit USD to prevent accidental inline math; the two
new display equations were split into readable aligned lines.

Existing RL chapters, accepted inference theory and the MBU addition were not
rewritten. No staging, commit, push or deploy.

## Figures and actual UI

Reused source PNGs: dynamic 2801×1030 and continuous 2722×2569, with pinned HF
image-dataset CC BY-NC-SA 4.0 evidence. Both were inspected at native resolution
and on the actual page; no new image copy, crop, redraw or recompression was
created. The registry now records both their original chapter and practice 14.
The strict registry test checks those two destinations and retains all its
negative license/hash/crop cases.

Route: `http://127.0.0.1:4330/practice/14-sobrat-minimalnyy-inference-engine/`.
At 1280 px the complete three-step diagram and attribution are present. At
360 px the figures fit 313 px without cropping, but their inline labels are
small. Reading individual labels requires the viewer. Zoom 4×, both horizontal
extremes (0/987) and vertical extremes (0/662), including the complete legend,
were tested with keyboard panning. The caption remains available in the viewer.

Both new display formulas fit 313 px. Four tables measure 352/313/313/395 px
inside 313 px containers; wider ones scroll internally, including keyboard
movement. There is no viewport horizontal overflow and no KaTeX error. Final
build was visited again; temporary viewport reset, viewer closed and RU page
marked as the deliverable tab.

The `/en/practice/14-sobrat-minimalnyy-inference-engine/` route was also opened:
it explicitly shows “This content is not available in your language yet.” It
is a fallback, not a completed English translation.

Screenshots:

- `screenshots/2026-10-26-inference-practice-desktop.jpg`
- `screenshots/2026-10-26-inference-practice-mobile-math.jpg`
- `screenshots/2026-10-26-inference-practice-mobile-viewer-start.jpg`
- `screenshots/2026-10-26-inference-practice-mobile-viewer-legend.jpg`

## Build evidence and remaining work

472 unit tests in 50 files passed; the four source/preservation tests were
rerun after the final wording/source-pin adjustment. Final CPU checker passes.
Final Astro build: 1965 pages. Final built-output tests: 135 passed. Built
links: zero broken routes, fragments or files. Source coverage passes, 210
matrix rows and 58 destinations. Scoped `git diff --check` passes; the unrelated
derived-source trailing spaces recorded in the prior audit are not changed.
Logs: `/var/tmp/bookvar-inference-practice-{unit,build,output,links,coverage}-2026-10-26.log`.

The initial registry test assumed exactly one use of the HF images; it was
updated to require the two explicit destinations rather than disabling the
check. No provenance check was removed.

Full course integration remains open. H-L2-10 and H-SL2-10 now record partial
reviewed integration instead of treating `preview-audited` as current completion.
Next compare the complete EDLS Week 8 seminar/homework with this practice and
add a source-backed executable neural KV/greedy reference where missing. Do not
restart accepted RL or rewrite the already accepted scheduling explanation.
