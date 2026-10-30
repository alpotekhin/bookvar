# Inference physics: editorial and arithmetic review

Canonical chapter: `00 Учебник/14 Inference и оптимизация/55a Физика LLM inference — prefill, decode и roofline.md`.
RU only; no translation, commit, push or deployment.

The primary agent read the entire existing chapter and the revised text.
This is a continuation of the systems/pretraining revision, not a claim
that the complete inference module has been reviewed.

## Sources and corrections

- Re-read the pinned CS336 Lecture 10 source
  (`8b59b50730766695c2ffedd1a79c50cd09b9eb91`), especially the explicit
  MLP/attention transfer ledger and Llama parameter/memory calculation.
  Checked the live JAX Scaling Book inference chapter as a second
  primary teaching source.
- CS336's displayed MLP sequence writes U/G but its short traffic sum
  does not count their later reads. The chapter now gives a complete
  four-operation ledger and a separately stated fused-operation model.
  This is not represented as measured HBM traffic or an implemented kernel.
- Attention's read-once formula is an optimistic input/output model:
  FlashAttention may reload tiles. MHA and GQA are kept distinct, with
  explicit shared-K/V reuse assumptions.
- Distinguished allocated parameters from weights used by linear layers.
  An input embedding lookup does not read the whole V×d table; the output
  logits projection normally does. Worked Llama estimates now separate
  allocated memory, accounted reads and peak-bandwidth time.
- A nearly-full 80 GB allocation is not presented as a runnable server.
  The GQA counterfactual changes both KV and projection parameters and
  is not called an allocator-only optimization.
- Fixed comparison of FLOP count with FLOP/s: the prefill time comparison
  uses 2PS/C. Added actual TTFT/ITL/TPOT timestamps and streaming-chunk
  granularity caveat.
- Replaced meta-introduction and execution-trace captions with subject
  explanation and readable source attribution. Five existing figures
  retained without byte changes; no original graphic generated.
  Existing unconfirmed rights for Gordić figures remain explicitly
  unconfirmed, not silently upgraded to a named license.
- Preserved all five source_unit_id values, route, links and the old
  worked-estimate heading anchor.

## Executed numerical evidence

`publishing/tools/check_inference_physics_examples.py` PASS with Python 3.12.
Executes the two actual Markdown blocks, then checks independent operation
ledgers for MLP; large-matrix parameter counts and separate embedding
lookup; MHA/GQA input/output traffic; causal pair counts; H100 peak-ratio
arithmetic; latency timestamps.

Example B=64 MHA:
79,717,990,400 allocated bytes, 79,390,965,760 accounted read bytes,
23.69879574925373 ms at the assumed peak bandwidth.
These are arithmetic results, not a GPU benchmark.

Final chapter SHA-256:
`6c459a9f5153b9eaaa35e437411bb14f4301b2e49d251c6abbe278c0c4257fd5`.
Checker SHA-256:
`94003a76cd5e6123788b45d9f4c18018d3aa6bffe6fccfce2fd5b58ab3131a58`.

## Acceptance boundary

No actual fused kernel, HBM hardware counters, GPU throughput, server SLOs
or multi-device inference were tested. Build and desktop/mobile acceptance
are recorded in the accompanying integrated 44e–44h/55a acceptance note.
