# JAX runtime: source explanations and executed state/placement cases

Status: content-and-local-ui-reviewed.
This is a thematic S8 extension, not acceptance of all JAX documentation,
the complete Harvard course, EN translation or the whole textbook.

## Current artifact and preservation

`00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению.md`
has 2980 lines, SHA-256
`8826db488145077e23c59f37abb23d42e66afe4d57688bd30f109d521817ea08`.
Removing the one bounded JAX addition recovers all 2400 lines of the prior
accepted chapter, SHA
`4fab5d0025274bd232fb3122c1152dc939cf8529a41389401c3d261e1db94a23`.
No previous source, code, figure, metadata or anchor is changed. The JAX
addition sits outside the CUDA Graphs review boundary; its old anchor still
belongs to CUDA Graphs. The accepted backprop/JAX and RL chapters are unchanged.

The whole expanded chapter was read in consecutive ranges 1–430 (the
truncated middle recovered at 185–270), 431–710, 711–1000, 1001–1300,
1301–1600, 1601–1982, 1976–2275, 2276–2580 and 2581–EOF. After localized
style/bridge changes, all 578 lines of the new section were reread together.
The unchanged earlier explanation is also bound by the inverse hash check.

## Explanation comparison

All 659 lines of the pinned JIT guide, 363 of State, 353 of Control flow,
245 of Placement, and complete Shard-map selections 1–179/654–809 were read.
The remaining Shard-map material is archived, not claimed integrated.
Source revision: `dc4af684446e65ee98eee8090e1ff6f8211473fb`, JAX 0.12.0.1.
The source archive README records exact reading and reuse boundaries.

| Mechanism | Retained explanation and worked case |
|---|---|
| Python state versus recorded computation | Full original Counter and jaxpr inspection. Plain calls print 1/2/3; the fresh jitted counter returns 1/1/1 and mutates its Python field only to 1. The constant result is explained from the missing state input. |
| Explicit state | Full original CounterV2; compiled calls consume/return state and produce 1/2/3. Training-state inputs are connected to the already accepted gradient-pytree example rather than duplicating it. |
| Refs | Original current-JAX ref update; three executions produce `[0,3,0]`. Explicit constraints distinguish intercepted ref operations from arbitrary Python mutations; ambiguous shared-ref vmap writes are rejected. |
| Specialization and hidden values | Original rank branch/cache-key passage, six calls/four observed traces with shape/dtype/static-value changes and prior-cache reuse. Closed-over gain stays at 2 after Python changes it to 5; explicit input gives 15 instead of 6. Traces are not called kernel counts. |
| Runtime control flow | Python if rejects an abstract scalar; compiled cond gives 2/-4/0. Matching branch structure/shapes/dtypes and batching-to-selection boundary are stated. Dynamic Boolean-index length and unequal branch shapes are independently rejected. |
| Loop state and derivatives | Full original scan reference and cumulative-sum example. Carry values 1/3/6/10 and derivative 4/3/2/1 are derived and checked. Fixed carry structure, while-loop reverse-mode boundary and static fori-loop distinction are stated. No universal constant compile-time claim is imported. |
| Ownership | Actual donation leaves output `[1,2,3,4]` and invalidates the consumed input. Reuse raises; ineffective donation and measurement limits are explicit. |
| Placement and communication | Committed single-device inputs reject incompatible addition. Original four-shard psum program and original full diagram produce `[22,20,12,17]` on four CPU replicas. Explicit one-device mesh/vmap gives row sums `[9,22,21,19]`, not a collective. |
| Completion/timing | Warmup and `.block_until_ready()` are required for the intended completed-call measurement; no timings or hardware speedup are claimed. |

Harvard's callable-transform exclusivity, all-host-overhead elimination and
automatic-pmap-communication claims remain source findings. Pure functions
are not claimed to make arbitrary program parallelization safe. Current refs
are included to avoid teaching a stale blanket no-mutation rule.

## Executed evidence

`publishing/tools/check_jax_runtime_examples.py` executes all 22 printed
Python fences, plus 96 independently checked cases: five scan lengths,
51 signed scalar branch cases and 40 four-device collective inputs. JAX and
JAXlib are both 0.12.0.1, NumPy 2.3.5, CPU only, four logical devices in one
process. All results and all four replicas are compared with independent
NumPy sums. Negative controls cover abstract Python Boolean conversion,
unequal conditional shapes, changing carry shape, dynamic Boolean indexing,
shared-ref writes, consumed donation and incompatible placement.

The first placement assumption failed. The active four-device mesh makes
new arrays replicated there; moving the array to CPU 0 alone then conflicts
with that context. The chapter now supplies a matching one-device mesh and
explains both observations. The check was rerun, not relaxed to hide the
failure. No GPU, network, multi-host, model quality or performance claim.

Old Triton/FLCE/buffer/queue and tracing/export CPU checks pass after the
extension. Old numerical implementations/assertions are unchanged; the
tracing inverse check removes only the independent JAX addition. The five
site-output fence counts change from 23 to 45 because 22 visible programs
were added, not because an existing rendering requirement was removed.

## Sources, image and style

Eight archived objects match the pinned Git blob IDs: five complete Markdown
files, original PNG, LICENSE and AUTHORS. Seven selected original fragments
are compared exactly after declared MyST-to-Markdown conversion. English
code/comments and explanations are retained; added reasoning is separate.
The full 1941×918 psum PNG is byte-identical, SHA
`ab23d21957bed8ff35f25b068861d691d28668b94170c630a2ceb750fb1d3914`.
Apache-2.0 and AUTHORS are published beside the source attribution; no root
NOTICE exists in the pinned tree. Adjacent and global asset records agree.

No-ai-slop review removed an editorial aside and a blanket contrast, added
the missing introduction of the four control-flow operators, and retained
the full source explanations and numerical reasoning. The checklist does
not serve as evidence that the mechanisms are complete.

## Actual browser inspection

Desktop 1280×900: complete original figure and caption visibly present.
Mobile 360×800: body 360/360, all four tables 328/328, no display-math or
inline-code overflow, KaTeX errors 0. PNG renders at 327.99×155.125 from
1941×918. Its small mobile inline labels remain a limitation; viewer 2× was
actually panned to both edges 0/344 and 344/344, with all Device and x/y
labels visible. No crop or replacement drawing was used.

Screenshots under `docs/audit/screenshots/2026-10-27-jax-runtime-*` record
desktop, mobile inline, table and both viewer edges. After the final build,
root was reloaded and the complete addition revisited. All 16 different
source images load; the unused viewer image with an empty src is not counted.
All IDs are unique; the old CUDA anchor's containing paragraph directly
precedes the CUDA heading, not the JAX section. The new table's top and bottom
were inspected at mobile width. EN visibly states that the content is not yet
available in the selected language and contains the new section; it is not
counted as translated. The real published Harvard matrix shows both the
new paragraph and updated H-V1-07 row with three held candidates still open.
Local changes only; no staging, commit, push, deploy or EN body edit.

## Final validation

The first full unit run exposed the stale S8 revision in the separate Network
Architectures map. Its target now points to the current chapter and this
inverse-preservation review; source decisions and prior mechanisms are not
changed. It also exposed an existing TensorFlow reference-HTML digest drift:
the converter snapshot's mtime is 22:45:14, after the earlier 22:35 manifest.
The current complete 64099-byte file and its entire Example usage were compared
with the live official reference. Bytes were left untouched; the corrected
manifest retains the old digest and explicitly records that the prior bytes
and cause of the mismatch are not established. No copied teaching material
depends on that HTML, and the actual earlier CPU conversion is unchanged.

Final validation continued on 28 October after the overnight run. That run
ended in wall-clock timeouts in asset decoding/inventory and the built-link
walk, not failed content assertions (elapsed durations 3663.92/1034.30 s).
The cause of those elapsed-time anomalies was not established. No timeouts
or assertions were relaxed. Fresh serial runs on the same bytes pass:

- 587 unit tests / 86 files, 43.21 s, `unit-final3.log`.
- 154 built-output tests, 4.47 s, `output-final2.log`.
- Final actual Astro build: 1969 pages, 26.01 s, `build-final.log`.
- Built links: 0 broken routes/fragments/files, `links-final.log`.
- Structural source coverage PASS: 210 rows, 58 destinations, `coverage.log`.
  This does not establish all-course semantic completeness.
- Current 22 fences/96 CPU cases and eight source-object Git blob checks
  pass; the printed chapter hash is the 8826db48 revision above.
- Scoped diff whitespace check passes. Current destination hashes/line counts
  bind both Frameworks and Network Architectures maps; no older revision is
  substituted to make their tests pass.

Logs use `/var/tmp/bookvar-jax-runtime-` as their prefix. No validation job
remains running. Remaining Frameworks items are the three held native
placement/FL/transfer candidates; other Harvard/HSE/EDLS units and deferred
EN remain part of the full active textbook goal.
