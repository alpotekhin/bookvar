---
type: editorial-review
status: accepted-bounded
last_updated: 2026-09-15
---

# Independent acceptance: foundations and systems

## Verdict

**PASS for the bounded editorial acceptance.** No unresolved P1/P2 content error was found in the complete-file sample below. The two material evidence gaps found during review were resolved before this verdict:

1. The existing EN quantization chapter was materially shallower than the RU page after the FP4 course follow-through. It now covers the FP8 ranges, FP4/MXFP4/NVFP4 scale/layout distinctions, W4A16/W8A8 memory ledger, and SpinQuant rotation/fusion mechanism; I reread the final 326-line page. The exact handoff, sources, checks, and hash are in `course-fp4-eval.md`.
2. `systems.json` originally exposed 22 suite-level numerical checks but did not attach an executed-check/remaining-limit record to every page. It now has explicit `executed_example_checks` and `limitations` on 60/60 RU records and 11/11 EN records. Empty arrays mean that no page-specific executable example was run; they are not test claims.

This is not an independent claim that every one of the 115 RU chapters or every authored EN counterpart was reread. Mechanical ledger, snapshot, figure, and hash checks cover the full two scopes; substantive content review covers the 19 page instances listed below.

## Exact review coverage

The following pages were read completely in their current form for substantive/mechanistic review.

Foundations, 7 RU + 3 EN:

- `00 Учебник/00 Математические и ML-основания/02 Производная и градиент.md`
- `00 Учебник/01 Классическое машинное обучение/06a Спектральная кластеризация и графовый Laplacian.md`
- `00 Учебник/00 Математические и ML-основания/03 Вероятность, правдоподобие и логарифм.md`
- `00 Учебник/00 Математические и ML-основания/04 Функции потерь.md`
- `00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V.md`
- `en/00 Textbook/05 Attention and Transformer/02 Self-Attention — Q, K, V.md`
- `00 Учебник/05 Attention и Transformer/03 Полный Transformer.md`
- `en/00 Textbook/05 Attention and Transformer/05 The complete Transformer.md`
- `00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение.md`
- `en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/04 GPT-1 — generative pre-training.md`

Systems, 6 RU + 3 EN:

- `00 Учебник/09 Dense FFN и Mixture of Experts/02 Mixture of Experts — routing, capacity и serving.md`
- `00 Учебник/11 Pre-training и Scaling/44c Tensor и sequence parallelism.md`
- `00 Учебник/11 Pre-training и Scaling/44d Pipeline parallelism.md`
- `00 Учебник/14 Inference и оптимизация/58a Распределённый inference и disaggregated serving.md`
- `en/00 Textbook/14 Inference and optimization/58a Distributed inference and disaggregated serving.md`
- `00 Учебник/14 Inference и оптимизация/58a2 Раздельное обслуживание prefill и decode.md`
- `en/00 Textbook/14 Inference and optimization/58a2 Disaggregated prefill and decode serving.md`
- `00 Учебник/14 Inference и оптимизация/57 Квантизация языковых моделей.md`
- `en/00 Textbook/14 Inference and optimization/57 Quantizing language models.md`

The derivative/spectral superseding edits were checked against `primary-content-review.md`. The EN57 superseding edit was checked against `course-fp4-eval.md`. The selected probability/loss, attention/full Transformer/GPT-1, MoE, TP/PP, distributed inference, P/D timing, and quantization explanations are mechanistically coherent; sampled RU/EN pairs preserve the relevant contract rather than merely similar headings.

## Ledger and provenance reconciliation

- Foundations: all 102 original tuples `(path, severity, line, excerpt, issue, action)` match the completion ledger exactly; dispositions are 96 `fixed` and 6 `already-fixed`, with no rejected/open record.
- Systems: all 88 original tuples match exactly; dispositions are 82 `fixed` and 6 `already-fixed`, with no rejected/open record.
- Snapshot hashes: 65/65 foundations page instances and 59/59 changed systems page instances match their recorded task-entry `before_sha256`.
- The systems evidence enrichment preserved all 88 findings and the page count. JSON parses successfully. It changes evidence granularity and the four completed course-follow-through hashes, not finding dispositions.

Current hash mismatches are all documented concurrent/superseding content work, not unexplained regressions:

- foundations derivative: current `454937e034fa5ed30480d19fb8d5ae49350389505aadf5e75f2a7a2f9daaa69d`, matching `primary-content-review.md`;
- foundations spectral clustering: current `52af70baf5e70420404e3ee70366a638857ae7b3cc21c8b2b6567f08799cae82`, matching `primary-content-review.md`;
- systems long context: current `a905656cb7446ed983d0dce75ce80a8173eb0f0346a2aa7960efe054c0dcb433`, root-owned course follow-through;
- systems next-token prediction: current `9db385e6450e19734669a2cb76d648cecec264e494ae2a86ec8226bd93380ba7`, root-owned course follow-through;
- systems scaling laws: current `9d5e6959000c6d6ab678d01c7c8e7bbb21d8894e90f482e99da36433a29587ab`, later full-read correction owned by `complete_foundations`.

The completed FP4/evaluation follow-through hashes were folded into `systems.json`: RU57 `cda51ab…`, EN57 `048baec7…`, evaluation 59 `819a8896…`, and Responsible systems `072e39de…` (full values remain in the JSON and handoff report).

## Figures and retained structure

- Foundations: the embed multiset matches the task-entry snapshot for all 65 RU/EN page instances, including the derivative and spectral superseding edits.
- Systems: no task-entry embed was removed from any of the 59 changed RU/EN page instances. The only multiset differences are one added curated course-follow-through figure in long context and one in next-token prediction; subtraction against each snapshot reports zero removed embeds.
- This check establishes retention of referenced embeds, not publication rights, image quality, or browser rendering.

## Executed checks

Executed in this acceptance:

- `task-1-numerical-checks.cjs`: PASS.
- `task-1-page-code-checks.py` under `/private/tmp/bookvar-editorial-venv`: PASS with PyTorch 2.8 CPU.
- `test_primary_review_examples.py`: 3/3 PASS.
- Independent NumPy/Python CPU checks reproduced MoE capacity/reservation arithmetic, copied-expert normalization, TP forward/backward equivalence, both published PP schedules and live-activation peaks, ring-collective latency arithmetic, P/D handoff timing, and the quantization grid/all-zero/metadata examples.

The FP4 follow-through report additionally records a passing runnable CPU program for all 256 FP8 byte encodings, FP4 shared-scale error, payload accounting, and orthogonal rotation. I reviewed that program and its stated boundary; I do not reclassify it as hardware/model validation.

## Remaining evidence boundary

No GPU/CUDA/Triton kernel, NCCL/network collective, multiprocess DDP, serving engine, quantized model, benchmark suite, model training, or full sklearn pipeline was executed in this acceptance. The 34 Harvard lab runtime is not part of these ledgers and is not claimed. No full site build, browser/mobile visual QA, link crawl, commit, push, or deployment was performed here.

The absence of a page-specific executable check is now explicit in `systems.json`; full-page reading, primary-source inspection, contract analysis, and executable examples remain separate evidence types. Final publishing/build evidence belongs to the root acceptance.
