# EDLS metric receipts and service boundaries, 26 October 2026

## Artifact and reading

`06 Практика/14 Собрать минимальный inference engine.md`: 974 lines,
SHA-256 `2adf0af6a979805bd8136d27a075b1e103b0eb3e01c5c640d9e952b0d2e0686e`.
The complete prior 783-line practice was read. The new section was read with
the whole existing chapter in context; the finished receipt calculation,
worked table, benchmark analysis and policy/service boundaries were checked
against the sources. Removing its bounded marker restores prior SHA
`c47997e5b9601a3e8ef31f6a9d669286ff11d46399604d56a7e368c25ea9848a`.
Nested preservation checks also recover the accepted 433-line practice and
initial 31-line assignment.

Read in full at EDLS commit `e632aa89ca9e6638d52e1b686095e7442faffbb0`:
`scheduler_manager.py` (100 lines), `metric_manager.py` (44), `api_server.py`
(194), `launch.py` (65), and `test_benchmark.py` (230). Relevant homework
Parts 2.1–3 were revisited after the earlier complete notebook reading. All
five new original files are unchanged and hash-pinned in the source archive's
`references.json`, with the complete course MIT license. The three-policy
assignment paragraph remains verbatim in English and has a fidelity test.

## Mechanisms and executable example

The section distinguishes queue admission, server token availability, client
receipt and GPU timing. Request A is traced through submission 0.00 s,
admission 0.10 s and token times 0.30/0.50/0.90 s. Its queue wait, TTFT,
individual ITLs and TPOT are derived before the calculation code.

The full visible `request_report` fence is executed by
`python3 publishing/tools/check_edls_metric_receipts.py`. It validates receipt
identity, terminal outcome, finite/ordered timestamps and interval boundaries.
The five assigned requests demonstrate:

- one-token and rejected requests have undefined TPOT, not zero;
- one cancellation's emitted token contributes to total emitted-token rate,
  while completed-request token count remains a separate quantity;
- seven emitted tokens, six completed-request tokens, three completions and
  five offered requests over a declared one-second window;
- mean request TPOT 0.20 s versus pooled mean ITL about 0.2333 s;
- goodput 2 requests/s under the assigned TTFT/mean-TPOT SLO, versus 1 under
  its strict per-interval version.

The checker verifies the actual published example, table values and populations,
invalid receipts, duplicate IDs, empty/zero-output outcomes, both goodputs and
200 affine-clock cases. Counts remain invariant, intervals scale and rates
scale inversely. It AST-extracts and executes only the reviewed original
mode-config and baseline-policy functions; it does not import the source
server, spawn workers or send benchmark requests.

The existing five CPU examples still pass, including 300 scheduler fixtures.
The neural module/smoke text remains unchanged; its exact equality tests pass,
so the earlier 316 neural prefix/cache comparisons remain applicable. No
neural timing or language-quality claim is added.

## Source observations and pedagogical review

The endpoint returns completed JSON rather than token streaming. Its client
latency timestamp is taken at response headers before reading/parsing JSON;
it cannot provide TTFT or token intervals. Prompt length is characters, not
tokenized length. Source latency percentiles include both success/error
responses. Target submission rate, achieved submission rate and finite
submission-plus-drain denominator are distinguished.

The original baseline admits one request only when no unfinished request is
active. The chapter specifies two alternative admissions to compare, while
requiring all three to obey free request slots and memory capacity. Whole-
prompt admission is not mislabeled chunked prefill or a token budget. The
assignment remains open until those implementations and their experiment run.

The API's ID and scheduler's separate counter agree only under the same
successful registration sequence; future cancellation/rejection/retry work
must propagate one ID. The source retains resolved futures, does not guard
`set_result` against cancellation, and waits for readiness without a timeout.
These are source-level findings, not reproduced service incidents. No HTTP
fix is claimed.

No-ai-slop review kept concrete events, units, source language, exact code,
limitations and causal examples. It did not replace the original assignment
with a generic summary or remove meaningful technical contrasts. The final
policy paragraph was tightened to state slot/memory limits explicitly.

## Actual page and publication checks

Preview: `http://127.0.0.1:4330/practice/14-sobrat-minimalnyy-inference-engine/`.
Desktop 1280 px: full receipt table and explanation were inspected. Mobile
360 px: its 487 px table scrolls within 313 px, including keyboard access to
the TPOT/ITL columns. Its 900 px code block scrolls within 311 px. Document
width is 345 inside the 360 px viewport; no horizontal page overflow or KaTeX
errors. Existing original diagrams are unchanged; their zoom acceptance is in
the preceding neural-KV report. The final build's HF figure was also viewed
again, with full source credit and pending-token table visible.

Screenshots: `screenshots/2026-10-26-edls-receipts-desktop.png`,
`screenshots/2026-10-26-edls-receipts-mobile.png`, and the refreshed
`screenshots/2026-10-26-edls-neural-kv-desktop.png`.

The EN route explicitly retains the unavailable-translation notice and current
metric/policy sections as fallback. No EN source was edited. Temporary viewport
was reset and the final RU practice was marked as the deliverable tab.

481 unit tests in 52 files passed. After the last prose-only slot/memory
clarification, all 12 affected source/fidelity/receipt tests passed again.
Final build: 1965 pages; final built-output tests: 135 passed; built links:
zero broken internal routes/fragments/files; source coverage: PASS, 210 rows
and 56 direct-linked destinations. Scoped tracked-file `git diff --check`
passes. Logs: `/var/tmp/bookvar-edls-receipts-{unit,final-targeted,build,output,links,coverage}-2026-10-26.log`.

## Remaining work

The current deliverable adds the complete receipt calculation and source-backed
explanations. Assigned receipt times are not observed server events. Actual
event instrumentation, the three admission policies and fixed-workload service
experiment remain open, as do tokenizer/HTTP/sampling extensions and GPU work.
No original Marimo reproduction or full EDLS completion is claimed.

The transfer/gap matrices reflect this partial state. Next implement the
request-ID/terminal-event contract and three admissions around the verified
neural core, then compare them on the same offered workload. Keep correctness,
CPU timings and any future GPU benchmark distinct. No accepted RL rewrite,
translation, staging, commit, push or deployment occurred; the global textbook
goal and remaining course/source-rights requirements remain active.
