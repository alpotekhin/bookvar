# Recorded neural CPU admission experiment

`runs.json` contains 54 actual finite runs, not assigned latency fixtures:
two token-ID workloads × three slot limits × three policies × three repeats.
Each run retains its twelve request receipts, prefill/decode call intervals,
runtime and code hashes. Its 648 output sequences were checked against an
independent uncached greedy reference.

The source contract is EDLS Week 8 homework, commit
`e632aa89ca9e6638d52e1b686095e7442faffbb0`, MIT. The complete source notice is
retained in `05 Источники/Originals/edls-inference-2026-10-26/`. This is an
adapted Bookvar solution, not a claimed completed teacher implementation.

Reproduce from the repository root:

```sh
uv run --offline --with torch==2.14.1 --with numpy==2.3.5 \
  --with transformers==4.57.1 \
  python publishing/tools/check_edls_admission_reference.py --experiment
```

JSON goes to stdout and verification messages to stderr. Wall-clock values
will change; IDs, budgets and outputs are the correctness invariants. The
existing snapshot is not overwritten by that command.

The final driver uses a separate standard-library producer process. A first
threaded-producer diagnostic revealed tens of milliseconds of submission
lateness and was not used for the published plot/table. Process readiness and
a future common clock origin precede the measured interval. Actual producer
timestamps remain in the records; final maximum lateness was about 5.00 ms,
not zero, despite planned 2 ms spacing.

`phase-timeline.png` is a standard Matplotlib plot of the final recorded
decode-heavy, two-slot, repeat-0 phase intervals. It is an original measurement
artifact, not a reused third-party illustration or a synthetic teaching
diagram. Hatching and the legend identify phases without relying only on color.
`manifest.json` records its data/script hashes and origin separately from
the book's reused source-illustration registry.

```sh
uv run --offline --with matplotlib==3.10.7 \
  python publishing/tools/plot_edls_admission.py \
  '06 Практика/Results/edls-admission-2026-10-26/runs.json' /tmp/admission-timeline.png
```

The model is random Qwen3, 22,720 parameters, CPU FP32, one PyTorch thread.
Warmup and uncached checks are not timed. The experiment has no HTTP,
tokenizer/checkpoint quality measurement, GPU timing, peak physical-memory
measurement or production p99/SLO-capacity claim. Logical KV reservations
exclude padded temporary batches and copies. Shared-CPU noise and realized
arrival jitter prevent causal policy-ranking claims from three repeats.

Acceptance: `docs/audit/2026-10-26-edls-admission.md`.
