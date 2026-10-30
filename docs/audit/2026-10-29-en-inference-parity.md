# English inference: full chapter reconciliation

Seven English chapters now contain the explanations, calculations, code,
figures and qualifications in their current canonical counterparts.
This report covers 54, 55, 55b, 55c, 56, 57 and 58 only.
Physics (55a), benchmarking (58b), distributed inference (58a) and
disaggregation (58a2) belong to the parallel parent/sibling work.

The machine-readable journal is
`docs/audit/2026-10-29-en-inference-parity.json`.
It records the before/current English SHA, canonical SHA, complete reading
ranges, chapter mechanisms, all 180 canonical section headings, their
English counterparts, fragment aliases and hashes of all 44 figure files.
This is a thematic parity review of the current chapters, not a claim that
every external course or paper has been reproduced in full.

## Material restored

| Chapter | Explanations retained in the English body |
|---|---|
| 54 Decoding | Stable softmax; pooled beam expansion, EOS, length normalization and non-monotonic beam-width example; temperature and corrected figure interpretation; top-k/top-p/min-p and order; penalty sign and shift dependence; local grammar masking versus full-string conditioning; stop-string boundaries; real random-weight Llama generation. |
| 55 KV cache | Layerwise causal invariance; chunk masks; multilayer GQA/RoPE and cache mutation; selected-token versus processed-token timing; left padding, cache positions and RoPE positions; exact memory accounting; block addressing and global softmax; fragmentation, block size, sharing, copy-on-write and release synchronization. |
| 55b Scheduling | Static slot accounting; iteration scheduling and packed attention; separate token and KV budgets; assigned versus completed writes; preemption/recomputation; chunk latency and context-dependent cost; chained prefix hashes; full-hit recomputation; free versus evicted blocks; RadixAttention splitting, reuse, isolation and fairness. |
| 55c Serving | Existing full original vLLM, SGLang, TensorRT-LLM and FlashInfer explanations retained; packed-request CPU example; NUMA CPU and memory placement; thread-pool oversubscription; mmap first touch, meta initialization and parameter assignment; checkpoint loading and readiness limits. |
| 56 FlashAttention | Memory sizes versus IO counts; complete online-softmax recurrence and rescaling example; full backward derivation and numerical gradients; dropout replay; atomics versus two-pass ownership; incomplete causal tiles; FA2 partitioning; FA3 overlap dependencies, buffer ownership and register pressure; FP8 and dated implementation support. |
| 57 Quantization | Symmetric/asymmetric grids, zero points and packing; calibration and clipping; worked GPTQ compensation, AWQ and SmoothQuant; LLM.int8 channel split; QAT/STE update; NF4, exact Double Quantization excerpt and paged optimizers; FP8/MXFP4/NVFP4 accounting; rotations and SpinQuant training/deployment constraints. |
| 58 Speculation | Greedy and exact sampling; residual-mass proof and complete accept/reject cycle; pending-token KV rollback; total variation and expected progress; lookahead costs; EAGLE-1 shifted embeddings, EAGLE-2 tree reranking/masks/residual verification, EAGLE-3 self-generated training states; experimental and measurement limits. |

## Verification on the saved chapters

`node publishing/tools/en-inference-parity.mjs` passes on all seven files:

- 180 canonical sections represented, with 170 explicit compatibility aliases;
- 44 unchanged source figures in canonical order;
- 70 display-math blocks with the same mathematics;
- 43 code fences preserved, including 39 Python blocks;
- 20 complete licensed excerpts preserved;
- all canonical external URLs and normalized Obsidian targets retained;
- no duplicate section IDs, single-dollar display delimiters or unresolved
  drafting placeholders.

The ten pre-existing serving excerpts had no boundary markers in English.
Their original paragraphs are checked in order, excluding the separately
checked figures and editorial captions. The remaining named excerpts are
checked byte for byte. The existing packed-attention example translates
one comment and its diagnostic print; its computation is unchanged.

The expanded FlashAttention derivation replaces four shorter historical
English displays with complete canonical derivations. The new displays
retain line breaks rather than restoring the previously overflowing
single-line formulas. Formula labels such as `bits/weight`, `ms` and
`on the chosen grid` are translated; values and operations are unchanged.

`publishing/tests/en-inference-parity.test.ts`: **2 tests passed**.
The test checks actual chapter content against the journal and canonical
files; recorded editorial review remains a separate human judgment.

CPU execution command:

```sh
LC_ALL=C LANG=C HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --offline --with torch==2.14.1 --with transformers==4.57.1 \
  --with numpy==2.3.5 python publishing/tools/check_en_inference_examples.py
```

All **33 runnable Python blocks** passed. Independent checks also passed:
36 blocked-attention output/logsumexp/Q/K/V-gradient cases, all 256 signed
INT4 pairs, 102 residual-distribution cases and the QAT/STE update.
The CPU receipt records each executed block and current chapter SHA in
`docs/audit/2026-10-29-en-inference-examples.json`.

## Limits and handoff

The five original serving runtime/layout fragments and the FlashAttention
CUDA diagnostic were not executed. No pretrained weights, endpoint, EAGLE
runtime, GPU kernel or throughput benchmark was run. CPU checks establish
the stated example calculations, not model quality or serving speed.

All source and prior English chapter bodies were read in full. The final
English prose was reread for missing causal steps, source qualifications
and the project-qualified no-ai-slop checklist. Code and licensed source
passages were not rewritten for style.

No canonical chapter, shared English ledger, central asset registry,
publication adapter or navigation index was changed in this EN pass.
No commit, push or deployment was performed. The parent owns the aggregate
build, current mobile/desktop rendering inspection and publication decision;
those stages are not claimed by this report.
