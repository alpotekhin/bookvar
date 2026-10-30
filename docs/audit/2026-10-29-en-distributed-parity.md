# English distributed-serving parity — 29 October 2026

## Scope and result

The two existing English distributed-inference chapters were reconciled against
their complete canonical Russian chapters. Both source-language and English
bodies were read in full, including code, captions, caveats, and source lists.
The work adds the missing explanations and calculations inside their relevant
sections rather than appending short summaries.

| English chapter | Before → after lines | Figures before → after | Executed examples |
|---|---:|---:|---:|
| Parallelism and collective communication | 346 → 911 | 5 → 11 | 8 |
| Disaggregated prefill and decode | 440 → 951 | 4 → 7 | 10 |

The JSON companion records exact before/after hashes, every figure path, the
44 canonical-to-English heading mappings, topic reconciliation, and per-example
execution results. Line counts are inventory, not evidence of prose quality.

### Parallelism

The chapter now derives collectives from actual values and tensor ownership.
The TP example computes the full MLP output from two ranks, explains why the
first nonlinear operation needs the right split, and counts the output bias
once. SwiGLU and attention retain their distinct placement requirements.

The pipeline schedule shows completion times, bubble occupancy, the bottleneck
of unequal stages, the autoregressive dependency between successive passes, and
layer-local KV ownership. SP follows token rows through normalization, gather,
TP projections, and ReduceScatter. CP derives the stable merge of attention
statistics: averaging local normalized outputs is explicitly shown to be wrong.

MoE includes the eight top-2 assignments, dispatch matrix, expert grouping,
reverse permutation, weighting, padding, and actual cross-rank bytes. The
configuration section describes process groups rather than blindly multiplying
all parallelism degrees. The 70B example compares TP4/8/16 memory bounds with
GQA replication and tests request-axis KV sharding by moving queries and outputs.

All 11 canonical original figure placements remain, with source captions
explaining what the training diagrams do and do not imply for inference.

### Disaggregated serving

The handoff distinguishes cached prompt positions from the already-selected
but uncached first output token. Client timestamp definitions, one-token output
handling, arithmetic-intensity assumptions, and the MLA state-shape boundary
are explicit.

The KV section covers bytes, suffix-reuse prerequisites, TP layout restoration,
page mappings, model/adapter compatibility, push/pull ownership, asynchronous
cancellation hazards, and layer-transfer dependencies. The Splitwise original
English excerpt is preserved exactly with its existing attribution. The
layer-streaming example computes the final transfer tail rather than replacing
it with an invalid maximum of aggregate compute/transfer times.

Routing compares local prefix, queueing, remote fetch, and network speed.
Physical reservation is separated from forecast admission. The chapter retains
the versioned SGLang allocation restrictions, Dynamo v1.5.1 metadata and failure
behavior, and the limitation on assuming decode-prefix deduction.

Provisioning includes average-flow bounds, the first output token produced on P,
18 total GPUs rather than seven instances, Little's-law concurrency, the
M/D/1 TP/PP example, and all nine constraints in the Splitwise source figure.
Mooncake's chunked pipeline is explained through two causal attention stages
and checked against whole-input execution. Goodput uses the joint SLO result
per request and all GPUs.

All six canonical PD figures are present. The pre-existing English Dynamo
overview is also retained as an additional seventh figure, explicitly separated
from the version-specific protocol claims.

## One authorized correction in the canonical chapter

Full reading found one arithmetic error in the paragraph before the reservation
example. Capacity 16384 minus active 8192 minus reserved 4096 leaves 4096
positions. An incoming 4096 therefore fits exactly; the text incorrectly said it
did not fit. Parent explicitly extended ownership for this one correction.

The changed line is:

- Before: “8192 активных и 4096 зарезервированных новый запрос на 4096 позиций уже не”
- After: “8192 активных и 4096 зарезервированных новый запрос на 8192 позиций уже не”

No code or other canonical prose changed. The existing less-than-or-equal
allocation check was already correct. English explains both the failing
8192-position case and the exact-fit 4096-position case.

Canonical before SHA-256:
b7b7f12fe4726130c9d5d75697e7cda4d3314216fbb3a8195a474af77cb5efe4

Canonical after SHA-256:
7973703ca7ec77248f0fdc77321e7c5f9805164d182508234fb5c49c3e86a13d

The checker reverses only that line in memory and requires the result to match
the complete original file hash. This bounds the canonical edit precisely.

## Verification obtained

- All 18 published Python blocks executed on CPU with the cached offline NumPy
  2.3.5 environment. Blocks run in chapter order with a shared namespace, as
  explicitly required by continuation examples. Each chapter gets a fresh
  subprocess.
- All 18 Python ASTs match their canonical counterparts in sequence. Removed
  Russian comments do not change executable statements.
- All 17 canonical figures exist locally and are retained, plus the existing
  additional Dynamo overview. No new images were generated or downloaded.
- Every canonical heading has a destination alias in English. Existing explicit
  source aliases and changed old English heading fragments are also retained.
- All displayed-math delimiters are paired; no standalone single-dollar
  delimiters or unfinished insertion markers remain.
- Exact canonical revisions and current status are set in both English files.
- Six focused Vitest tests passed. Scoped diff whitespace verification passed.

Commands used:

~~~sh
LC_ALL=C LANG=C uv run --offline --with numpy==2.3.5 --with torch==2.14.1 \
  python publishing/tools/check_en_distributed.py \
  --output docs/audit/2026-10-29-en-distributed-parity.json

# Working directory: publishing
LC_ALL=C LANG=C pnpm exec vitest run tests/en-distributed-parity.test.ts
~~~

This is execution of teaching calculations, not a GPU benchmark, model-quality
evaluation, or network experiment. Integrated build and desktop/mobile visual
acceptance belong to the parent; neither is claimed by this child report.

## Frozen revisions and owned deliverables

| File | SHA-256 |
|---|---|
| EN 58a | 91dbf85d69db6a4389b9a23c5978eb61f9b5c5e8343da50bc494c73345435629 |
| EN 58a2 | 3a4dd8e20f7f3c4e45bf0361010773b6fe13d554bc85f5ffb7db2048a5b9ebb5 |
| Canonical 58a, unchanged | 7219c5e6fdef1ca9cd3fa92c50097cfe77a6ad664afa3ac43e9e6d8c117d7625 |
| Canonical 58a2, single-line correction | 7973703ca7ec77248f0fdc77321e7c5f9805164d182508234fb5c49c3e86a13d |

Owned paths:

- en/00 Textbook/14 Inference and optimization/58a Distributed inference and disaggregated serving.md
- en/00 Textbook/14 Inference and optimization/58a2 Disaggregated prefill and decode serving.md
- 00 Учебник/14 Inference и оптимизация/58a2 Раздельное обслуживание prefill и decode.md — one approved line only
- publishing/tools/check_en_distributed.py
- publishing/tests/en-distributed-parity.test.ts
- docs/audit/2026-10-29-en-distributed-parity.md
- docs/audit/2026-10-29-en-distributed-parity.json

Shared registries, central English ledgers, the publisher, navigation, and
previously frozen multimodal files were not changed in this pass. No git
staging, commit, push, deployment, or remote mutation was performed.

