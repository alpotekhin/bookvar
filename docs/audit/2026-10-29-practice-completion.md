# Executable practice: completion evidence, 29 October 2026

## Scope

This pass adds executable exercises to practices 09, 14, 17 and 18. It does not
reclassify the earlier 54 CPU experiments or the existing HTTP/NDJSON loopback
tests as new work. No remote machine, cluster, paid service, git publication or
deployment was used. Other agents' theory and English changes were preserved.

The local host has CPU, Apple MPS, Gloo and a working Docker daemon. CUDA is
unavailable. The recorded environment and negative capability checks are in
`06 Практика/Results/runtime-2026-10-29/capabilities.json`.

## Delivered files

| Status | Files | What they contain |
|---|---|---|
| Extended | `06 Практика/09 Реализовать ring all-reduce.md` | Real process exchange, a step-by-step three-rank trace, DDP equivalence and the original Harvard ring illustration. |
| Extended | `06 Практика/14 Собрать минимальный inference engine.md` | Tokenizer training, safetensors checkpoint reload, actual text requests and cached/uncached CPU and MPS comparisons. |
| Extended | `06 Практика/17 Развернуть наблюдаемый ML сервис.md` | A running HTTP/gRPC fixture service, readiness, bounded admission, actual Prometheus scraping and open-loop load results. |
| Extended | `06 Практика/18 Harvard ML Systems — интерактивные design labs.md` | Launching the original Marimo applications with their matching source dependencies; a completed interactive inference lab. |
| New | `06 Практика/Code/ring_collective_lab.py`, `ddp_step_lab.py` | Gloo ring collectives and a DDP gradient/update oracle; prepared, capability-gated NCCL branch. |
| New | `06 Практика/Code/text_checkpoint_lab.py`, `check_runtime_capabilities.py` | Tiny random Qwen3 checkpoint, fixture tokenizer, CPU/MPS checks and explicit unavailable-device results. |
| New | `06 Практика/Code/prepare_harvard_runtime.py`, `check_harvard_runtime.py`, `run_harvard_lab.py` | Pinned original course downloads, source identity checks, all-app default-state execution and local interactive launch. |
| New | `06 Практика/Code/run_observable_lab.py`, `observable_service/` | Local Docker experiment, service, protobuf, model fixture, dependency recipe and cleanup of only its own containers/network. |
| New | `06 Практика/Code/RUNTIME-LABS.md` | Commands, dependencies, result interpretation and limitations. |
| New | `06 Практика/Results/runtime-2026-10-29/` | Machine-readable receipts, initial and corrected Marimo results, source manifests and Docker build log. |
| New | `site/public/examples/runtime-labs-2026-10-29.zip` | Downloadable code/results package, unchanged existing KV reference module and EDLS MIT notice. |
| New | `publishing/tests/practice-runtime.test.ts` | Six regression tests for the new runtime evidence, checkpoint, collectives, service, package and historical baseline. |
| New | `publishing/tests/helpers/practice-runtime-baseline.ts`, `publishing/tools/practice_runtime_baseline.py` | Scoped reversal of only the new practice sections for older preservation checks. |
| Adjusted | Five EDLS/inference preservation tests; `publishing/tools/check_ops_scale_examples.py` | Apply the scoped restoration helper before comparing historical prose. The serving worker owns the separate MLOps restoration and current ops-scale result hashes. |
| New | `docs/audit/screenshots/2026-10-29-marimo-runtime.png`, `2026-10-29-marimo-completed.png` | Personally inspected original UI with live controls/results and the saved report. |

## Observed executions

### Ring and DDP

`ring.json` contains 24 configurations: 2, 3 and 4 ranks, int64 and float32,
and four vector lengths per configuration. All 72 per-rank results equal both
the analytical sum and `torch.distributed.all_reduce`. Uneven vectors are
zero-padded to equal chunk sizes; padding is removed from the final result.

For three ranks and five elements, rank 0 records four actual send/receive
steps. The result is `[30, 33, 36, 39, 42]`; each rank sends eight elements over
the complete reduce-scatter/all-gather operation. This is a correctness test
on local processes, not an interconnect benchmark.

Both two-rank Gloo DDP reports match a single-process global batch of eight.
The largest gradient difference is `7.450580596923828e-9`; the resulting SGD
states match. The local torchrun command specifies `--local-addr=127.0.0.1`
because this host's machine name did not resolve for the initial rendezvous.

### Tokenizer, checkpoint and neural KV

The script trains an 18-token WordLevel test tokenizer on four English/Russian
sentences and creates a random 19,776-parameter Qwen3 configuration. The saved
checkpoint bundle occupies 84,944 bytes. CPU logits before and after offline
reload are exactly equal. Four requests pass cached/uncached reference checks
on CPU, and the same four pass on MPS. The tokenizer and model are fixtures,
not Qwen's pretrained tokenizer or a pretrained language model. Generated text
is deliberately not interpreted as a quality result.

CUDA and NCCL commands fail before rendezvous or result creation on this host.
No GPU throughput or peak-memory result is inferred from CPU or MPS results.

### Harvard Marimo

The launcher retrieves 261 original files, 4,417,974 bytes in total, at Harvard
commit `45ecc8d82fcae70c149cdce550d3b3d3411df913`. Git blob IDs and SHA256 hashes
are recorded. It does not clone the full book. The initial execution using the
bundled wheels passed 30/34 applications: four required APIs present in the
same-commit source modules but absent from the older wheels. Both the failed
and corrected receipts are retained. With matching source modules, all 34
default states executed successfully in separate processes/home directories.

The original V2 Lab 10 inference UI was also operated in a browser. Changing
demand from 100 to 200 changed daily cost from 77,760 to 155,520 USD and the
displayed crossover from 25.7 to 12.9 days. Gates A–D were completed, the UI
reported `SAVED`, and the actual downloadable report was retained as
`marimo-lab10-ui.json`, with `incomplete_fields: []`. These are course-scenario
calculations, not hardware measurements. Only this lab received a complete
interactive gate walkthrough; 34/34 refers to default-state execution.

### Observable Docker service

The service computes a three-class linear classifier over RGB means using an
identity matrix. It is not EDLS's VGG model or an LLM. Its injected 20 ms delay
is explicit and has a separate metric from model execution, queueing and
serialization. Admission is bounded to one executing and three waiting
requests. The protobuf wire contract is retained from MIT-licensed EDLS.

Actual network checks observed readiness HTTP 503, successful HTTP and gRPC
classification, HTTP 400 and gRPC `INVALID_ARGUMENT` for invalid shapes, and
HTTP 429 under overload. The implemented gRPC overload mapping to
`RESOURCE_EXHAUSTED` was not exercised in this run.

| Offered rate | Duration | Offered | Completed | HTTP 429 | Successful-request p95 | Offered requests meeting 60 ms SLO |
|---|---:|---:|---:|---:|---:|---:|
| 10 requests/s | 2 s | 20 | 20 | 0 | 34.94 ms | 100% |
| 100 requests/s | 2 s | 200 | 91 | 109 | 96.73 ms | 1% |
| 200 requests/s | 2 s | 400 | 95 | 305 | 92.08 ms | 0.5% |

The Prometheus HTTP API returned `up=1`, completed HTTP count 207 (one
correctness request plus 206 completed load requests), and inflight 0. The
620 arrival/completion receipts and image IDs are saved in `service.json`.
These short local runs demonstrate the measurement contract, not production
capacity or stable performance.

The normal public-image build initially stalled in the host's Docker
credential helper. The runner now uses a temporary empty Docker client config
with the existing local daemon endpoint for public pulls. User Docker settings
were not changed. The final cleanup left only pre-existing containers
`sphere-db`, `sphere-redis`, `sphere-kafka`, `open-webui` and no lab network.
Images/build cache remain; unrelated containers were not stopped.

## Validation and preservation

The targeted Vitest run passed seven files and 34 tests:
`practice-runtime`, `edls-kv-reference`, `inference-practice`,
`edls-admission-reference`, `edls-metric-receipts`, `edls-http-reference`,
`edls-deployment-sources`. Historical practice 14 and 17 SHA256 hashes are
recovered exactly by the new scoped helpers; no global baseline relaxation was
introduced. The separate MLOps-02 preservation conflict belongs to the serving
worker's edits, and its integration was coordinated with that worker.

| Chapter | Before SHA256 | After SHA256 |
|---|---|---|
| 09 | `8a279f5f0a5d876f9326bcd90e30c4208285ee95867b97a00f60dfe9d6fbfd51` | `7fd8aa867f668690590e7d98f6eb7f82907cf1c37d5f73e9abd2c5ebd6515eb6` |
| 14 | `df419eaed79373346f3e66f900cd635714e733bcb754fa97d335fbcd6dd31fc4` | `be07fec6d21b0db0c5f1bb5e691d6f0d05f741b95075beb8dc7cadf4d76a3219` |
| 17 | `ca35e72e3d1817c212497aaf0da43bf49f4c2d8bd51b065f52ae4289de89f47b` | `36116beffe2e9348124f27228fb6209b53282d095e114c6ac253b9abc3301c4a` |
| 18 | `b23d6edb62045a06be81c24d096784d3171e826cdc3a9809e9981d83e3b593de` | `7c36fa6ded69b1340901023814326ce48ba8619bb37eb0038f61505aade4fa8e` |

Archive SHA256:
`35675a47329723ba7a5d0d7cff21856eb3c73fefcc541c6492e21b98bed44f4c`.

The ring figure is the existing original Harvard SVG, with its pinned source
and CC BY-NC-SA 4.0 attribution in the chapter. The existing neural KV module
and reproduced protobuf retain the EDLS MIT notice. Original Marimo text and
plots were not translated or replaced. No generated diagram substitutes for a
course figure.

Parent visual review found that the original ring/tree slide's crossover rule
is not a universal equality. Practice 09 now explicitly identifies the displayed
tree as an unsegmented full-message-per-level model and the crossover rule as a
source heuristic dependent on the tree algorithm, segmentation and network.
The original SVG remains unchanged. A regression assertion checks this caveat;
no numerical runtime input or result changed, and the code/results ZIP contains
no chapter Markdown, so its hash remains unchanged.

## Remaining acceptance boundaries

Parent integration still needs the full site checks and rendered Bookvar page
review, including the ring SVG. The original Marimo UI itself was visually and
interactively checked. Global source maps and gap matrices were deliberately
left to the parent to avoid concurrent edits.

The following are not supplied by this pass: CUDA/NCCL execution, multi-node
collective bandwidth and overlap profiles, full pretrained Qwen3-4B capacity,
GPU SLO/peak-memory measurements, all 34 Marimo gate walkthroughs, VGG image
preprocessing/accuracy, Grafana, distributed tracing, or cancellation
propagation. They must not be marked executed on the basis of these receipts.
