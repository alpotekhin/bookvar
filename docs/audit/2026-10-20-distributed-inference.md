# Chapter 58a: distributed inference

Date: 2026-10-20. Canonical RU chapter:
`00 Учебник/14 Inference и оптимизация/58a Распределённый inference и disaggregated serving.md`.

SHA-256: `7219c5e6fdef1ca9cd3fa92c50097cfe77a6ad664afa3ac43e9e6d8c117d7625`.

Local route: <http://127.0.0.1:4330/textbook/inference/distributed-inference/>.

## Scope

The full original 391-line chapter was read before editing. The revised chapter
was reread in order in two complete, untruncated parts; the final terminology,
KV-symbol definitions and preserved-anchor edits were reviewed separately.
The result replaces short descriptions with a sequence of tensor layouts,
worked computations and constraints. This is acceptance of this chapter's
explanation, not completion of the whole textbook or full integration of every
source course.

Canonical Markdown, source snapshots, one viewport-only source-figure excerpt,
shared asset usage, a CPU checker and this report are local deliverables. No
English page, commit, push or deployment was made. Unrelated dirty files and
existing original assets were preserved. The preview server was restarted on
127.0.0.1:4330 after the old server was found stopped.

## Explanation review and sources read

| Mechanism | Source scope actually read | Worked explanation retained or added |
|---|---|---|
| Placement and collectives | How to Scale Your Model, Sharded Matrices: Computation With Sharded Arrays, Cases 1-4; complete NCCL Collective Operations page, version 2.32.3 | Concatenated coordinates versus summed contributions; two-rank AG, RS, AR and AllToAll; latency/bandwidth ring arithmetic |
| Tensor parallel MLP/attention | Megatron-LM 1909.08053v4, section 3, Figure 3 and norm/output-embedding paragraphs; Harvard Inference at Scale, Tensor parallelism | Complete two-rank MLP, hidden values, partial output values, one-time output bias, paired SwiGLU columns, head/output-projection split |
| DP and topology | How to Scale Your Model, Distributing Inference Over Multiple Accelerators, Prefill and Generation; Harvard Request routing for sharded models | Independent dense replicas versus shared MoE experts; TP8 on two nodes versus TP16; explicit DP2 x TP4 x PP2 rank groups |
| PP | Harvard Pipeline parallelism for inference and Request routing for sharded models, including original panels | Four-stage, three-microbatch schedule; finite fill/drain utilization; unequal-stage bottleneck; autoregressive feedback and layer-local KV |
| SP and CP | Complete Megatron Core Context Parallelism article and Figure 1; Ultra-Scale Playbook original TP/SP artwork and its recorded source metadata | Full-width token-row LayerNorm, AG before TP and RS after; CP group membership; explicit attention-denominator merge including maxima and empty masked blocks |
| EP | Harvard Expert parallelism and load balancing and Routing failure modes; Efficient DL Systems week 6 complete source slides, PDF pp.129-130 | Eight top-2 assignments for four tokens; per-destination counts, per-expert loads, grouped computation, return order and one-time gate combination |
| KV layout and capacity | How to Scale Your Model, Sharding the KV Cache, including full algorithm and source figure caption | GQA KV-head replication floor; 70B memory-only TP4/8/16 comparison; query-head to request-batch permutation and its inverse |

The chapter links to the prior FlashAttention derivation for online-softmax
statistics and to prior training chapters for gradient-specific operations.
No backward pass or dropout from a training figure is silently executed as part
of the inference explanation. NCCL communication-order/type requirements are
separate from the single-process algebra examples.

## Figures and source fidelity

Eleven figures are embedded. Five existing NCCL/Megatron figures are retained.
Harvard column/row splitting, two Ultra-Scale Playbook figures and two Efficient
DL Systems figures are reused from existing verified source assets. Their complete
mechanism panels are preserved. The two EDLS source PDF pages and their selected
figure extractions were visually inspected side by side.

The former Harvard TP request-routing figure was rejected: its embedded claim
of one AllReduce per Transformer block conflicts with Megatron's classic two.
The replacement shows column versus row partitioning without that claim.

For PP, both complete request-flow panels remain visible, but the source footer
claiming four times single-stage throughput is excluded. A new SVG changes only
viewBox height 480 to 365. No labels, paths, colors or geometry are edited. The
original remains unchanged. The adjacent caption, local NOTICE and registry
identify the excerpt and license. A validator checks exact viewport-only
derivation and rejects changed labels, invalid bounds and unpinned metadata.

The former Harvard EP overview with illustrative millisecond timings is not
embedded. EDLS's actual matrix-placement diagrams replace it. The chapter does
not promote those discarded illustrative numbers to measured serving results.

Five source snapshots and all eleven figure hashes are recorded in
`05 Источники/Originals/distributed-inference-2026-10-20/references.json`.
Pinned Harvard and HF originals and the current official NCCL/CP images were
compared byte-for-byte. This establishes identity, not a new reproduction-rights
grant for existing documentation. Existing NCCL rights metadata was not newly
reclassified or claimed to have been legally re-audited. Existing Harvard
CC BY-NC-SA, HF Apache-2.0 and EDLS MIT notices are preserved.

## Separate style review

The no-ai-slop skill and eval were read. The finished prose was reviewed for
generic setup, unsupported importance/measurement claims, mixed-language filler,
repeated definition-only paragraphs and empty recap sections. Technical terms,
useful contrasts, derivations, examples and limitations are retained. New toy
numbers are explicitly educational assumptions, not invented benchmark evidence.
English original source figures are not translated or rewritten for style.
Style review does not substitute for the mechanism and source checks above.

## Executed checks

`publishing/tools/check_distributed_inference_examples.py` executes all eight
Python blocks from the actual chapter, then compares the displayed functions
and formulas with dense/reference calculations:

- 800 TP MLP cases and a duplicated-output-bias negative control;
- 800 paired SwiGLU cases;
- 800 SP normalization/gather/reduce-scatter cases;
- 500 stable softmax-merge cases with random shard counts, large scores and masks,
  plus explicit empty/all-masked handling;
- 300 routed EP cases, including original-order recovery and assignment counts;
- 96 balanced pipeline schedules and one unequal-stage schedule;
- 200 KV query-head/request-batch permutations versus separately looped attention;
- written ring-communication and memory-capacity arithmetic;
- five snapshot hashes, eleven figure hashes, exact viewport crop and old anchors.

These are CPU algebra/provenance tests. No GPU, NCCL process group, model serving,
quality evaluation, measured capacity or speedup benchmark was executed.

The final local publication build succeeds: 1961 pages. Asset-registry tests:
38 passed. Built-output tests: 12 passed. Built-link check: zero broken internal
routes, fragments or files. Targeted git diff --check: clean.

## Browser acceptance

Actual browser viewports were checked at 1365 x 900 and 390 x 844. An initial
fresh tab used its default 1280 x 720; that was observed and corrected before
claiming the desktop breakpoint. All eleven final images load. Source-panel
boundaries, labels, aspect ratios and adjacent explanation were inspected.

Mobile document width is 375 px inside the 390 px viewport, with 343 px content.
All eight display equations fit that content width; zero KaTeX errors or display
equation overflows were observed. Code and tables keep their horizontal scrolling.
Wide matrix/network diagrams still have small labels on a phone; they require
opening the original for detail. Clicking the TP/SP figure was verified to open
the actual 1332 x 1474 original PNG, not a preview thumbnail. This limitation is
not described as perfect inline legibility.

Proof images:
- `/tmp/bookvar-distributed-desktop-2026-10-20.png`
- `/tmp/bookvar-distributed-mobile-2026-10-20.png`

## Remaining work

No explanatory gap identified in this chapter's defined scope remains open.
Wide-figure phone reading still relies on the original-image link. No fresh
rights review is claimed for the five inherited NCCL/Megatron documentation assets.
The English counterpart and live site remain unchanged. The larger textbook
goal is active; this report does not close it. Next canonical page: 58a2,
separate prefill/decode serving, then 58b benchmarking. Chapter 58a2 was read for
the boundary between topics but has not been revised or accepted in this pass.
