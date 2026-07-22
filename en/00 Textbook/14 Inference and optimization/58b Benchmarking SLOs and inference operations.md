---
title: "Benchmarking, SLOs, and LLM inference operations"
type: textbook-chapter
status: canonical
locale: en
translation_of: "00 Учебник/14 Inference и оптимизация/58b Benchmarking, SLO и эксплуатация inference.md"
last_updated: 2026-07-22
last_verified: 2026-07-22
primary_sources:
  - https://cs336.stanford.edu/spring2025/
  - https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin
  - https://docs.vllm.ai/en/latest/cli/bench/serve.html
---

# Benchmarking, SLOs, and LLM inference operations

A `tokens/s` figure without a workload description says almost nothing about
service quality. The same server may deliver high offline-batch throughput while
making an interactive user wait seconds for the first token. It may handle the
average request quickly yet repeatedly fail on long requests in the tail of the
distribution. An LLM inference benchmark should therefore ask not “how fast is
the GPU?” but “what real workload can the system sustain at the required quality
of service?”

Stanford CS336 distinguishes the latency experienced by one user from aggregate
throughput. DistServe refines that view with separate SLOs for the two generation
phases. The official [`vllm bench serve`](https://docs.vllm.ai/en/latest/cli/bench/serve.html)
tool operationalizes these definitions: it measures TTFT, TPOT, ITL, and
end-to-end latency, and supports percentiles, concurrency limits, controlled
request rates, and goodput calculation.

## The timeline of one request

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/vllm-latency_diagram.png]]

*Queueing, prefill, and sequential decode steps contribute distinct parts of
user-visible latency. Source: Aleksa Gordić,
[Inside vLLM](https://www.aleksagordic.com/blog/vllm),
[original image](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2025-vllm-anatomy/latency_diagram.png).*

Suppose a request arrives at $t_0$, its first token is received at $t_1$, and
the last of $N$ output tokens arrives at $t_N$. **Time to first token (TTFT)** is

$$
TTFT=t_1-t_0.
$$

It includes queueing, scheduling, tokenization and transport, prefill, and the
first generation step. TTFT is especially salient in chat and code completion:
the interface cannot display meaningful progress until that first token arrives.

**Inter-token latency (ITL)** is the interval between adjacent tokens,
$ITL_i=t_i-t_{i-1}$. It is a sequence, not a scalar; spikes appear to users as
streaming stalls. **Time per output token (TPOT)** is usually the request-level
mean interval after the first token:

$$
TPOT=\frac{t_N-t_1}{N-1}.
$$

Mean TPOT smooths pauses, so a streaming interface should also retain
percentiles of the individual ITLs. Finally, **end-to-end latency (E2E)** is
$t_N-t_0$. Approximately, $E2E=TTFT+(N-1)TPOT$, but exact values should come
from request timestamps: network buffering, retries, and post-processing can
violate the simple model.

Each metric answers a different product question. TTFT governs how quickly an
assistant begins to respond; TPOT and tail ITL govern reading comfort; E2E and
cost matter for batch summarization; in an agent making many short calls, both
latency components accumulate across the chain.

## Throughput and request rate

Throughput has at least three useful units:

- completed requests/s measures completion frequency;
- output tokens/s measures newly generated tokens;
- total tokens/s counts both input and output tokens.

They are not interchangeable. A workload with long prompts and short answers
may achieve high total tokens/s through efficient prefill while generating few
output tokens/s. Many short requests can raise requests/s at the same token
volume. A report should include request count, input and output throughput, and
the length distributions together.

As offered load grows, throughput rises and then saturates. Beyond saturation,
new requests mainly enlarge the queue and tail latency. The highest observed
throughput is therefore not the service's operating capacity: the saturation
point may violate every user-facing SLO.

## Goodput: count only acceptable service

**Goodput** is the rate of completed requests that satisfy the specified
constraints. If request $r$ is tested against $TTFT_r\le S_{TTFT}$ and
$TPOT_r\le S_{TPOT}$, then over a window of duration $T$,

$$
G=\frac{1}{T}\sum_r
\mathbf 1[TTFT_r\le S_{TTFT}\land TPOT_r\le S_{TPOT}].
$$

E2E or other conditions may be added. Throughput counts all completions;
goodput counts only those with acceptable service. A system that admits too
much work and creates a long queue can increase throughput while reducing
goodput.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-advanced/distributed-benchmarking/distserve-fig1-slo-curves.png]]

*Figure 1 from Yinmin Zhong et al., [DistServe: Disaggregating Prefill and
Decoding for Goodput-optimized Large Language Model Serving](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf),
OSDI 2024. For a synthetic 13B-model workload, dashed lines mark SLOs of 0.4 s
P90 TTFT and 0.04 s P90 TPOT. Colocated serving sustains only about 1.6 rps
within both limits, although each phase alone can sustain a higher rate. The
image is cropped from the original vector figure in the public USENIX PDF.*

DistServe also uses **SLO attainment**, the fraction of requests within the
limits. “Goodput of 10 rps at 90% attainment” means the system must admit a load
at which at least 90% of requests satisfy the conditions jointly. An average
TTFT below the threshold does not establish that result.

## Why percentiles are necessary

A mean hides distribution shape. Ten requests with 100 ms TTFT and one with
10 s TTFT average roughly one second, although most users see a fast response
and one experiences a serious failure. The median p50 represents a typical
request; p90, p95, and p99 expose progressively rarer tail events. An SLO usually
specifies both a high percentile and a window, for example “p99 TTFT no greater
than 800 ms over the last ten minutes.”

A percentile always needs a population and a window. A daily p99 over all
requests can conceal a five-minute incident, while a p99 calculated for every
one-minute window is stricter and exposes instability. Useful slices include
model, endpoint, tenant, prompt length, output length, and status code; an
aggregate can hide a systematically slow request class.

ITL requires special care. One can pool all inter-token intervals and report a
token-level p99, or first compute a p99 or mean within each request and then
aggregate requests. Long answers contribute more intervals and receive greater
weight under the first method. The aggregation method must accompany the result.

## Open-loop and closed-loop load

In a **closed-loop** benchmark, each virtual user submits its next request only
after the previous one completes. When the server slows, the generator reduces
offered load as well. This models a fixed population of sequential clients, but
can hide overload because rising latency automatically throttles arrivals.

In an **open-loop** test, requests arrive on an external schedule independent of
completions. Inter-arrival times are often modeled as a Poisson process, and
`--request-rate` in `vllm bench serve` sets the mean rate. The queue grows when
offered load exceeds capacity, revealing the stability boundary. Burstiness is
as important as the mean: real traffic arrives in bursts, and two streams with
equal mean rps can produce very different p99 latency.

`--max-concurrency` caps simultaneous in-flight requests; it is not a request
rate. Concurrency is accumulated work, related approximately to throughput and
latency by Little's law, whereas rate is the speed of arrivals. A benchmark
matrix should vary both or explicitly hold one fixed.

## The workload matters more than the GPU name

Prefill cost depends on prompt length, decode cost on generated-token count and
current context length, and memory on the sum of live KV caches. A pair of mean
lengths is therefore insufficient. Preserve the joint input/output-length
distribution, their correlation, shared-prefix frequency, stop reasons, and the
arrival process.

Fixed-length synthetic workloads are useful for isolating mechanisms. A grid
such as inputs of 128/2K/16K tokens and outputs of 32/256/1K tokens reveals
memory boundaries. It does not replace replay of an anonymized production
trace. Real workloads have heavy tails, system prompts, multi-turn cache reuse,
cancellations, structured outputs, tool calls, and diverse sampling parameters;
each changes scheduling behavior.

A popular ShareGPT-like dataset provides a reproducible baseline, but its result
does not generalize to document-heavy RAG or reasoning with thousands of output
tokens. A good report includes at least one standardized synthetic scenario for
comparison and one scenario representative of the target product.

## A reproducible protocol

Benchmarking begins by recording the environment, not by running a command.
Capture:

1. the exact model revision, tokenizer, and chat template;
2. dtype or quantization, maximum context, and parallelism settings;
3. serving engine commit/version, CUDA, driver, kernels, and GPU topology;
4. GPU count and type, interconnect, power limits, and clock modes;
5. scheduler settings, KV-cache budget, prefix caching, chunked prefill, and speculative decoding;
6. dataset revision, filters, seed, request count, and length distributions;
7. request rate, burstiness, concurrency, streaming, and timeout policy;
8. sampling, stop conditions, and whether `ignore_eos` is enabled;
9. warm-up procedure, measurement duration, and raw per-request timestamps.

`ignore_eos` fixes output length and reduces variance, but produces an artificial
load: in production some sequences terminate earlier and release KV-cache
blocks. Results using it must be labeled, and product benchmarks should be
repeated with natural termination.

Warm the model before measurement by loading weights, compiling kernels and
CUDA graphs, and priming the allocator. Warm-up is excluded from steady-state
latency, but **cold start** should be measured separately because it affects
availability under autoscaling. Repeat each scenario, publish dispersion and
raw data, and do not report only the best run.

The official [`vllm bench serve`](https://docs.vllm.ai/en/latest/cli/bench/serve.html)
documentation lists `--percentile-metrics ttft,tpot,itl,e2el`, percentile
selection through `--metric-percentiles`, result persistence, and `--goodput`
constraints such as `ttft:... tpot:...`. These flags belong to the current vLLM
version and may change. The durable content is the protocol; obtain exact
commands from the documentation for the version that produced the result.

## Finding operating capacity

One run at a “maximum” request rate does not establish capacity. Build a load
curve: start at low rps, increase offered load in steps, and wait for steady state
at each step. Record throughput, goodput, p50/p95/p99 TTFT, TPOT, ITL, and E2E;
queue depth; running and waiting requests; KV-cache use; GPU utilization; HBM;
and network traffic.

Operating capacity is the greatest rate at which every SLO holds with the
required attainment and the system remains stable: the queue does not grow from
window to window, and errors and cancellations remain within budget. Reserve
headroom for bursts, changing lengths, and replica failure. Peak throughput with
a permanently full queue characterizes the hardware and scheduler, but is not a
safe production setting.

## SLIs, SLOs, and operational alerts

An **SLI** is an observed quantity such as request-level TTFT. An **SLO** defines
a target and compliance fraction, for example “99% of interactive requests have
TTFT below 1 s over 28 days.” Tie SLOs to user classes: batch jobs and chat need
not share one threshold. The **error budget** is the permitted violation share;
it supports decisions about change and reliability instead of promising an
unattainable 100%.

HTTP latency is insufficient for a streaming endpoint: status 200 may be sent
before the first meaningful token. Server-side timestamps should cover admission,
prefill start, first token, every subsequent token, and completion. Correlate
them by request ID with client observations to separate queueing, GPU execution,
network delivery, and client backpressure.

GPU utilization alone is a weak alert. Memory-bound decode may be healthy below
full compute utilization, while 100% GPU use says nothing about SLO compliance.
Combine error-budget burn rate with leading indicators: queue depth, KV-cache
utilization, preemption and eviction, batch composition, prefix-cache hit rate,
KV-transfer time, NCCL errors, and replica imbalance.

## Degradation and overload

When offered load exceeds capacity, an unbounded queue turns a short burst into
a prolonged incident. Admission must be controlled: rejecting some requests
with an explicit status is often better than accepting all of them and violating
latency for everyone. The policy may consider tenant priority, deadlines,
expected length, and available KV cache, but it must be represented in the
benchmark.

Controlled degradation options include reducing maximum output length, routing
to a smaller model, disabling expensive best-of or speculation modes, or moving
batch work to a separate queue. Do not silently alter sampling semantics or
truncate context: that buys performance through an invisible quality change.
Test GPU or node failure, worker restart, loss of a decode instance after
prefill, and the time needed to restore capacity separately.

## The minimum honest report

A final report should make the result—not only the command—reproducible. Include
environment versions and configuration, workload definition, the open- or
closed-loop model, an offered-load curve, every metric definition, percentiles
and windows, throughput and goodput, errors and cancellations, GPU-hour cost,
and raw results. When comparing engines, keep model revision, sampling, request
distribution, and termination criteria identical; list engine-specific
optimizations separately.

This protocol turns a benchmark from a showcase number into a test of an
architectural decision. It establishes not only which stack is faster in one
regime, but why it sustains the target workload, where saturation begins, and
how much headroom remains before the user-facing SLO fails.

## Sources and further reading

- Stanford CS336, [Lecture 10: Inference](https://cs336.stanford.edu/spring2025/) — latency, throughput, and the inference compute profile.
- Zhong et al., [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin), OSDI 2024 — TTFT/TPOT SLOs, attainment, and per-GPU goodput.
- vLLM, [`vllm bench serve`](https://docs.vllm.ai/en/latest/cli/bench/serve.html) — the official interface for measuring serving workloads and definitions of available metrics.
- vLLM, [Benchmarking Dashboard](https://docs.vllm.ai/en/latest/benchmarking/dashboard.html) — a published approach to throughput/latency comparison; specific results are version-dependent.
