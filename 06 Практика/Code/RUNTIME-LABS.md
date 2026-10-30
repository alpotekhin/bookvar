# Executable practice package

Extract the archive and run commands from the directory containing `Code/` and
`Results/`. Source code is also canonical under Bookvar's `06 Практика/Code`.
Python dependencies are installed in the `uv` cache. The scripts do not download
model weights or datasets. Only the service experiment needs Docker.

## Text, tokenizer, checkpoint and request-owned KV

```bash
uv run --with torch==2.14.1 --with numpy==2.3.5 --with transformers==4.57.1 \
  python Code/text_checkpoint_lab.py --device cpu --output Results/text-cpu.json
```

Use `--device mps` on an Apple device with MPS support. The code generates a tiny
random Qwen3 checkpoint and deletes its temporary files after checking reload.
The 18-token WordLevel tokenizer is a test fixture, not Qwen's pretrained
tokenizer. Meaningless generated text is not a model-quality result.

## Ring and DDP correctness

```bash
uv run --with torch==2.14.1 --with numpy==2.3.5 \
  python Code/ring_collective_lab.py --output Results/ring.json

uv run --with torch==2.14.1 --with numpy==2.3.5 \
  python -m torch.distributed.run --standalone --local-addr=127.0.0.1 \
  --nproc-per-node=2 Code/ddp_step_lab.py --backend gloo --output Results/ddp
```

The first command runs 2, 3 and 4 local processes and records every ring step.
The second compares one DDP update with the global-batch reference. It uses
equal local batch sizes and a mean MSE loss. Neither is a network benchmark.
For the prepared two-GPU path, use `--backend nccl`; unavailable CUDA fails
before rendezvous and before creating a result. That path has not been run on
this Apple host. NCCL throughput, GPU overlap profiles and full-model capacity
are not supplied by the CPU result.

## Original Harvard Marimo applications

```bash
python3 Code/run_harvard_lab.py --lab vol2/lab_10_inference.py --port 4338
```

Open `http://127.0.0.1:4338`. Change `--lab` to another preserved name, such as
`vol1/lab_11_hw_accel.py` or `vol2/lab_12_ops_scale.py`. The launcher downloads
4.42 MB of original labs and shared modules at commit
`45ecc8d82fcae70c149cdce550d3b3d3411df913`, verifies Git blob IDs and writes a
SHA256 manifest. It does not clone the book. Dependency installation uses
additional disk space in the `uv` cache. The source modules are required because
four apps in this commit refer to APIs absent from its older bundled wheels.

`check_harvard_runtime.py` runs all 34 default states in isolated subprocesses
and isolated home directories. It does not claim that all assignments have been
completed. The retained Lab V2-10 UI snapshot separately records all A–D gates
completed and export unlocked. Its estimates are course scenarios, not hardware
measurements. Original text and visualizations are unchanged.

Harvard source and licensing:
[repository at the pinned revision](https://github.com/harvard-edge/cs249r_book/tree/45ecc8d82fcae70c149cdce550d3b3d3411df913),
[license](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/LICENSE.md).
The launcher retains that license with the fetched files. Book text/labs and
software components have their respective source notices; do not infer that
one license covers every dependency.

## HTTP, gRPC and real Prometheus scraping

```bash
uv run --with grpcio==1.76.0 --with grpcio-tools==1.76.0 \
  python Code/run_observable_lab.py --output Results/service
```

Requires a local Docker daemon; remote daemon endpoints are rejected. The script
builds the image, uses loopback-only host ports, starts its own scraper, runs
620 fixed-arrival requests, saves receipts, and removes its containers/network.
Images/build cache remain available for another run. Existing containers are
not stopped or changed. The runtime image uses a non-root user.

The service computes a 3×3 identity classifier over RGB means. Its deliberate
20 ms service delay makes saturation visible and has its own metric, separate
from model time. It is neither the course's VGG model nor an LLM. The protobuf
contract is reproduced from the MIT-licensed
[EDLS Week 7](https://github.com/mryab/efficient-dl-systems/tree/e632aa89ca9e6638d52e1b686095e7442faffbb0/week07_application_deployment).
The neural KV module is an attributed solution to EDLS Week 8. The full EDLS
MIT notice is included as `LICENSE-MIT.txt`.

`service.json` records image IDs, workload events, stage times, Prometheus
responses and outcomes. Base-image tags and dependency versions identify the
recipe, but do not make it a fully locked, bit-reproducible supply chain.
The run is not evidence for GPU SLO capacity, VGG accuracy, downstream
cancellation, a Grafana dashboard or distributed tracing.

## Retained evidence

`Results/runtime-2026-10-29/` contains the actual CPU/MPS, Gloo, container and
Marimo results. `marimo.json` deliberately retains the initial four failures
with stale wheels; `marimo-source-runtime.json` records the corrected 34/34
default-state executions. No failure has been rewritten as a passing run.
Wall-clock values will change; mathematical checks and request outcomes are
the reproducible contracts.
