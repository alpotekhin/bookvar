# Task 5 practice independent review

Date: 2026-09-15  
Scope: the 29 current Markdown pages under `06 Практика` represented in
`practice.json`, compared with the task-local `task-5-before/06 Практика`
snapshot; the 38 original practice findings; the current practice contracts;
and the bounded executable examples. No full site build, GPU/NCCL run, model
training, networked agent experiment, or Harvard Marimo runtime was performed.

## Verdict: PASS

No content or evidence P1/P2 remains in the reviewed pages. The 38 original findings are
resolved accurately in complete-page context, and the current text consistently
distinguishes a small runnable starter, an implementation contract, an adapted
project, and a full course assignment.

The initially reported evidence-contract P2 was repaired without changing page
content or finding dispositions. Every page object in
`docs/audit/2026-09-15-completion/practice.json` now records changed mechanisms,
source URLs, retained/added figure counts, only the checks actually performed,
and page-specific remaining limitations. JSON validation confirms 29 complete
records, 38 findings (`34 fixed`, `4 already-fixed`, `0 open`), and zero current
or snapshot image embeds across the practice pages.

## Independent checks

- Read all 29 current pages in full, including the index and the long Stanford
  and Berkeley projects. Re-read the edited mechanisms against the 38 baseline
  issues. The completion ledger reproduces those 38 issues exactly, in the same
  order and at the same severity; it does not silently drop an original finding.
- Recomputed every current and snapshot SHA-256. All 58 values match the hashes
  recorded in `practice.json`; the four `edited_in_tranche: false` pages are
  byte-identical to their snapshots, and the other 25 are distinct as recorded.
- Searched every current and snapshot practice Markdown page for Markdown,
  wiki-embed, and HTML image syntax. Both sets contain zero image embeds, so no
  practice-page figure was removed. These zero counts are now explicit in every
  per-page completion record.
- Ran the bounded suite with the requested Python environment, Lean 4.19.0, an
  Agg backend, and writable cache directories:

  ```text
  BOOKVAR_LEAN_BIN=/private/tmp/lean-4.19.0-darwin_aarch64/bin \
    /private/tmp/bookvar-editorial-venv/bin/python \
    docs/audit/2026-09-15-completion/test_practice_examples.py

  Ran 8 tests in 1.215s — OK
  ```

  The suite extracts code from the current Markdown rather than testing copied
  implementations. It covers bigram pseudocounts, causal attention and SDPA,
  FP16 underflow/loss scaling, checkpoint RNG preservation, the 65-point scaling
  fixture, the RLVR mask/reward/gradient/digest, Lean bad/good/repeat, and the
  symbolic-regression AST. Runtime versions independently matched the ledger:
  Python 3.12.14, torch 2.8.0, NumPy 2.2.6, SciPy 1.18.1, Matplotlib 3.11.2, and
  Lean/Lake 4.19.0 at commit `6caaee842e94`.
- Counted 21 Assignment 1 adapter functions and 48 test functions in the pinned
  local source tree, matching Practice 20. The A1/A2 contract language now
  correctly treats adapter bodies as learner-owned while preserving signatures,
  semantics, tests, and fixtures. Practices 20–24 link to locally present pinned
  assignment handouts, repositories, tests, snapshots, and data fixtures.
- Extracted 367 Stanford/Berkeley `source_unit_id` values from Practices 20–27;
  every value occurs in the corresponding pinned course source-unit registries.
  This checks route provenance, not the scientific truth of every source unit.
- Confirmed 34 archived Harvard lab pages and inspected the existing evidence
  artifact: all 34 entries report exact pinned-upstream inventory, matching
  upstream hashes, complete archived Python bodies apart from final whitespace,
  and passing Python syntax. Its machine-readable boundary remains
  `runtime: not-run`, `visual: not-reviewed`; Practice 18 states that limitation
  explicitly and does not claim successful Marimo execution.

## Content acceptance by practice type

- **Runnable inline starters:** 01, 02, 06a, 10, 22, 24, 26, and 27 now contain
  bounded examples with stated scope and independently passing checks. Their
  outputs do not stand in for the larger project.
- **Pinned course fixtures/contracts:** 20–24 expose real local Stanford
  handouts, tests, fixtures, and machine-readable contracts. Practice 20's
  `cpu_local` profile is explicitly a Bookvar recommendation, not a nonexistent
  upstream script. Practice 23 accurately labels the 25-document corpus as work
  the learner must prepare, while pointing to the real upstream Moby and dedup
  fixtures.
- **Protocols or full projects without a packaged starter:** 03–09, 11–17, 19,
  and 25 are presented as trace, measurement, distributed, service, vision, or
  agent projects rather than falsely advertised turnkey labs. Practice 25's
  YAML is an illustrative task contract, not a supplied twelve-task harness.
- **Imported interactive sources:** Practice 18 is a reader route over the 34
  pinned Harvard source bodies. Syntax/body preservation evidence is adequate
  for archive acceptance, but not runtime acceptance, and the page says so.
- **Adapted Berkeley projects:** 25–27 explicitly say they are Bookvar
  adaptations rather than official Berkeley labs. Practice 26 separates the
  verified one-theorem Lean starter from the learner-created twelve-theorem
  prover experiment; Practice 27 similarly separates its five-candidate
  polynomial fixture from the eight-law evolutionary/LLM comparison.
- **Index:** `_index.md` removes unverified duration promises and the guaranteed
  speedup implication. It now warns that 20–27 are substantially larger than
  the small examples and that successful starter execution is not project
  completion.

## Mechanism checks that materially support the verdict

- Bigram training optimizes the same pseudocount-smoothed target that it is
  compared against, while unsmoothed MLE remains separately named.
- Attention fixes layout, causal-mask polarity, `dropout_p=0`, SDPA semantics,
  merged shape, tolerances, and future-token invariance.
- KV-cache text separates logical occupied tokens, configured capacity, and
  allocated bytes; W8A8 separates calibration, validation selection, and test.
- FSDP equality is conditioned on global batch, sample order, normalization,
  stochastic state, and numeric rather than automatic bitwise equivalence.
- The speculative-decoding trace keeps draft and target cache offsets consistent
  through partial acceptance, correction, rollback, and full-acceptance bonus.
- Agent evaluation reports conditional and overall ASR with their own
  denominators and `N/A` for zero exposure. Outer isolation remains enabled even
  for intentionally open profiles; path containment covers traversal, symlinks,
  TOCTOU, and fail-closed behavior.
- Lean `pass@k` counts independent complete search restarts, not dependent tactic
  branches. The checked `n + 0 = n` starter fails on the missing lemma, succeeds
  twice with `rfl` in clean processes, and does not claim an LLM result.
- Symbolic regression uses a validated bounded AST, train-only selection,
  expanded primitive cost for learned abstractions, held-out reporting, and an
  explicit protected-division convention. The starter's six negative cases are
  executed.

## Remaining limitations

Rendered links and desktop/narrow presentation remain for combined acceptance.
No claim is made for GPU performance, distributed correctness beyond the text's
contracts, full Stanford training, Berkeley agent outcomes, or runtime behavior
of the 34 Harvard Marimo labs. Source-unit presence and archive hashes establish
provenance and preservation, not a fresh re-derivation of every cited paper.
