# Fault tolerance and fleet orchestration: editorial review

Canonical chapter: `00 Учебник/11 Pre-training и Scaling/44h Fault tolerance и fleet orchestration.md`.
Local editorial work only. No English edits or publication.

## Read and revised

The primary agent read the original chapter completely and rewrote the
explanation with explicit assumptions: independent failure rates versus shared
failure domains; timeouts versus root cause; finite results versus correctness;
checkpoint bytes versus a correct training state; Young/Daly compute-only
intervals; full torchrun worker-group restart; data consumption versus prefetch;
elastic batch/normalization rules; admission, placement, backfill and recovery.
The chapter and checker received an independent read-only review; no further
mathematical error was found.

Sources include the pinned Harvard QMD chapters at
`45ecc8d82fcae70c149cdce550d3b3d3411df913`, PyTorch 2.14 torchrun documentation,
Llama 3 report v3 section 3.3.4, official Slurm and Kueue documentation.
The primary agent rendered and inspected Daly PDF pp. 304 and 312, checking
the interval definition and equations (37)/(38). Harvard's assertion that
ordinary TorchElastic preserves surviving worker memory was not carried over.

## Illustrations

Four source figures are used: failure-domains, failure-types, gang-deadlock
and topology-placement. All were personally viewed at original size.
The two newly downloaded originals (failure-types and gang-deadlock) have
pinned provenance, hashes and CC BY-NC-SA 4.0 attribution in their
NOTICE/registry. The other two figures were already present.

Captions identify illustrative MTBF/bandwidth assumptions and explain that
the centralized failure-types drawing is not the topology of DDP. It does
not establish that a timeout proves the failure cause or that every
computational error is detectable by a checksum.

Retired from revised RU 44h, source bytes preserved:
`00 Учебник/Assets/Figures/curated/ml-systems/harvard/distributed/young-daly-optimization.svg`.
The supposed total curve is below a component; this is misleading for a sum
of nonnegative costs. An alternative checkpoint-recovery-timeline.svg was
inspected and rejected because it marks a checkpoint saved before the save.
No replacement custom diagram was drawn. The worked table and derivative
explain the Young optimum; an accurate licensed plot remains an optional
visual improvement, not a fabricated asset.

## Executed evidence on 2026-10-13

`publishing/tools/check_fault_tolerance_examples.py` passed with PyTorch 2.14.1:

- Executes the actual Markdown code. Exact next CPU step after restoring
  model, Adam, schedule, RNG and data cursor; missing optimizer, RNG and cursor
  are all detected. No-optimizer case retains the same pre-update loss and
  gradient but changes the next weights.
- Checks independent-device failure arithmetic, Young/Daly conventions and
  numerical examples, overhead table, global batch and placement example.
- Launches two isolated torchrun workers; worker 0 deliberately exits with
  code 23. Both workers get new PIDs and complete on restart 1. Bounded
  timeout/cleanup targets only this test's process group.

Initial lifecycle attempts used `--standalone`; local hostname resolution
failed. An explicit rendezvous endpoint alone did not change the worker
bootstrap hostname. `--local-addr=127.0.0.1` allowed worker startup. A further
attempt that included Gloo hit a barrier timeout. The final lifecycle test
therefore tests only the launcher's group-replacement contract using a local
readiness handshake, not distributed collectives or checkpoint restoration.
This limitation is explicit in the checker output. Real DCP/Gloo resharding
is independently tested in 44g; the two tests are not presented as an
end-to-end multi-node recovery experiment.

No CUDA/NCCL, hardware-failure durability, cluster scheduler, shared storage,
object-store or performance validation was performed here. Build and actual
page QA belong to the integrated 44e–44h acceptance record.
