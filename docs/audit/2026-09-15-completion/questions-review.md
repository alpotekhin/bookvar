# Task 5 question delta review

Date: 2026-09-15  
Scope: only the three current question pages compared with
`.superpowers/sdd/2026-09-15-close-editorial-findings/task-5-before`, plus
`test_question_examples.py` and `questions.json`. No full build or unrelated
question/practice review was performed.

## Verdict: PASS

No P1 or P2 findings remain. The archive-preservation boundary, all 45 archive
resolutions, and the two small executable examples pass this review.

The previously reported Xiong theorem-scope P2 is resolved. The erratum now
states the single-head square-matrix simplification, Gaussian inputs, bounded
loss derivatives, `W_Q=W_K=0` and therefore uniform initial attention, Xavier
initialization of the remaining matrices, zero biases, and initial LayerNorm
parameters. It continues to identify the last layer's second FFN matrix,
Frobenius norm, total depth, concentration assumptions, high-probability upper
bounds, initialization-time scope, and limited warm-up conclusion. Finding 28's
ledger resolution was updated consistently. This matches
[Xiong et al., §3.3 and Theorem 1](https://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf).

## Checks that passed

- The original Russian archive body is byte-identical to the task-local
  snapshot after removing only the inserted warning and trailing errata.
- All 37 original embeds remain in the preserved body. The errata adds one
  attributed local copy of LoRA Figure 1; no original figure was removed.
- The retained ROUGE screenshot shows a seven-token candidate and six-token
  reference. The erratum correctly separates its `6/7`, `12/13` result from
  the neighboring nine-token repeated-word candidate's `6/9`, `0.8` result.
- LoRA is correctly described as a trainable scaled low-rank update rather
  than an SVD of the frozen weight. Matrix shapes, initialization, and the
  `4096`, rank-`8` parameter count are correct.
- QLoRA correctly separates frozen NF4 storage, double quantization,
  dequantization to the compute type, trainable LoRA weights, activation
  memory, checkpointing, and CPU-RAM/GPU paging of optimizer state. This agrees
  with [Dettmers et al., §3](https://arxiv.org/html/2305.14314v1#S3).
- `python3 docs/audit/2026-09-15-completion/test_question_examples.py` ran four
  tests successfully. The tests execute the Python blocks extracted from the
  Markdown, rather than a duplicated implementation.
- Completion JSON parses, all before/after hashes match the reviewed files,
  and its 45 archive entries reproduce the baseline findings in order. The
  separate unresolved LLM publication finding remains explicitly open, so the
  aggregate `47 fixed / 1 open` claim is honest.
- The NLP index preserves unanswered-source labels while linking readers to
  the separate errata. The LLM edit correctly distinguishes pass@k as a metric
  from self-consistency and best-of-N selection. No missing English answers
  were represented as newly translated in this delta.

## Review limitation

Rendered links and desktop/mobile presentation remain pending the combined
build, as the completion ledger says. Primary-source checks here were focused
on the requested Xiong, LoRA/QLoRA, and ROUGE risks; this was not a new audit of
all archived answers or the unchanged English archive.
