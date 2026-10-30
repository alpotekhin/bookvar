# Distributed training details 44a–44d: integrated acceptance

Date: 2026-10-13. This is a local editorial/build acceptance record, not a
deployment or completion record for the entire textbook.

## Scope and editorial review

The primary agent fully read the revised DDP, checkpoint/offload and
tensor/sequence/context chapters and wrote the pipeline revision. Delegated
results were reviewed, not accepted solely from summaries. The pipeline
chapter and its checker also received independent review; the one reported
source-list typo was corrected.

Canonical files remain in `00 Учебник/11 Pre-training и Scaling/`:

| Chapter | Final SHA-256 |
|---|---|
| `44a Processes, collectives и DDP.md` | `27abe16642e5c9a9a3eeb12f3e6c82ded7d6472010098fc2fe55fde2fbfe0c0a` |
| `44b Gradient checkpointing и offload.md` | `a9009322b62bfcb8b06083959dca935f55ec8607a2fb497c9546cb3b037cce6f` |
| `44c Tensor и sequence parallelism.md` | `2a7cd5f3cdc437d2d176593ceac6266f81ce05da9c01296d3bea410af7e667b2` |
| `44d Pipeline parallelism.md` | `ad5d16a1b639016ead8a5abd52fa7e01307349640fe753ceeeb970e36d37002e` |

Chapter-specific source, editorial and numerical evidence is in
`2026-10-13-ddp.md`, `2026-10-13-checkpoint-offload.md`,
`2026-10-13-tensor-sequence.md` and `2026-10-13-pipeline.md` alongside this file.
No English source files, navigation routes or remote Git refs were changed.

## Executed numerical checks

The primary agent executed all four chapter checkers, not just the site's
tests. All passed:

- `check_ddp_examples.py`: two real CPU/Gloo processes; actual chapter code,
  collective semantics, unequal valid-token counts, accumulation, local-zero
  targets, global-zero rejection, Adam states, no_sync, join and sampler.
  Loopback hostname warnings occurred; this was not a multi-node run.
- `check_checkpoint_offload_examples.py`: actual six Markdown examples,
  saved tensors versus unique storage, segment/offload/transfer arithmetic,
  dropout RNG, selective checkpoint operation counts and finite differences.
  The initial standard SAC gradcheck failed because it repeated backward
  through a consumed cache; the corrected test uses independent graphs for
  all 32 perturbed coordinates. Maximum error was 4.434e-10, not a skipped test.
- `check_tensor_sequence_examples.py`: single-process NumPy simulations,
  column/row MLP and SwiGLU derivatives, SP adjoints, norm parameter sums,
  GQA replication, Ulysses permutation and Ring online softmax. This is not
  torch.distributed/DTensor execution.
- `check_pipeline_examples.py`: both actual chapter timelines, 256 strict
  schedules and a nonlinear four-stage model with explicitly returned
  boundary gradients. Includes uneven/masked labels and a zero-label
  microbatch. This is a single-process CPU schedule simulation.

PyTorch checks used 2.14.1 CPU; NumPy checks used the bundled Python runtime.
No CUDA/NCCL, real GPU offload, network overlap, peak GPU-memory or throughput
measurement was performed. Arithmetic examples are not benchmarks.

## Illustrations and rights

The four chapters contain 17 inline source figures: 7, 3, 4 and 3 respectively.
All 17 actual usages match entries in `05 Источники/asset-registry.yml`.
Five upstream assets were newly added in this block: three HF TP/SP/CP
figures, one HF interleaved-pipeline SVG, and one DeepSpeed offload PNG.
These originals are byte-for-byte copies with pinned provenance and Apache-2.0
records. Existing PyTorch and Harvard images retain their own licenses.

The primary agent personally inspected the new originals and the retained
DDP/TP/checkpoint/pipeline diagrams, including original-size views. Rejected
Stanford crops were not published. Existing full slide files remain intact;
their removed chapter usages point to the respective audit records. Their
historical user-permission metadata is not a new verified rights claim.

The full DeepSpeed license is included in `site/public/licenses/` and its
local HTTP response was hashed as
`c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`, matching
the pinned upstream copy. The NOTICE and registry record the public copy.

## Rendered UI acceptance

The actual local preview at port 4330 was inspected with the in-app browser
at 1280×900 and 390×844. On all four routes:

- No KaTeX errors; every chapter image loaded.
- No document-wide horizontal overflow at either tested viewport.
- All image bounds remain within the mobile content column.
- Meaningful diagrams and captions were inspected visually, not only via
  DOM dimensions. Desktop checks included complete TP/SP panels, causal
  zigzag assignment, all checkpoint/offload figures and pipeline schedules.
- Original-image navigation and return worked. Dense wide diagrams still
  need enlargement for their smallest labels on a phone; fitting a diagram
  into 358 pixels is not a claim that every label is comfortably readable.

Mobile review found a checkpoint-memory formula whose long underbrace labels
required scrolling. The primary agent moved those labels into adjacent prose,
reran its checker, rebuilt, and visually verified that the complete expression
now fits. The final browser viewport override was reset; the DDP page is open.

## Final build and publication checks

- `pnpm --dir site check`: 0 errors, warnings or hints.
- Final `pnpm --dir site build`: 1961 pages, success at 16:48:41 local time.
  The existing duplicate `/404` route warning remains; it was not introduced
  by these chapter edits.
- `pnpm --dir publishing test:unit`: 231/231 passed after registry integration.
  Intermediate runs failed on assets/audit files still being integrated;
  the final run is the acceptance result.
- `pnpm --dir publishing test:output`: 12/12 passed on the final build.
- `pnpm --dir publishing check:links`: 0 broken routes, fragments or files.
- `pnpm --dir publishing check:source-coverage`: passed, 210 matrix rows and
  59 destinations, plus its existing course-artifact checks. This checks
  source connectivity, not complete pedagogical coverage of every course.
- `git diff --check`: passed.

No commit, push or deployment. Unrelated `AGENTS.md` and `.agents/` changes
were preserved. The broader editorial goal remains active: chapters 44e–44h
(ZeRO/FSDP2, expert/hybrid parallelism, network/storage/checkpoints and fault
tolerance), subsequent reader-facing pages and final whole-goal reconciliation
are not claimed complete by this record.
