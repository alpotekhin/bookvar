---
title: "Benchmarking, SLOs, and LLM inference operations"
type: textbook-chapter
status: canonical
locale: en
translation_status: current
canonical_revision: "1d453c6e9f77e4aee10c81305bc406fe0bf3a50e69fc4edd22748a555e5a0b9c"
translation_reviewed: "2026-10-29"
translation_of: "00 Учебник/14 Inference и оптимизация/58b Benchmarking, SLO и эксплуатация inference.md"
last_updated: 2026-10-29
last_verified: 2026-09-15
source_unit_id:
  - lecture-10-section-4-understanding-the-inference-workload
  - lecture-10-dynamic-workload-failure-mode
primary_sources:
  - https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_10.py
  - https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/benchmarking/benchmarking.qmd
  - https://github.com/vllm-project/vllm/blob/b8e9001bf9568c801985889deff34d3b9fe6d0b0/vllm/benchmarks/serve.py
  - https://docs.nvidia.com/aiperf/v0.13.0/reference/ai-perf-metrics-reference
  - https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf
  - https://sre.google/workbook/alerting-on-slos/
---

# Benchmarking, SLOs, and LLM inference operations

During a ten-second test, a server completed five of six requests. One answer
started late; two others had long pauses during generation. Only two met
all the latency requirements. The report can show 0.5 successful completions
per second or 0.2 acceptable completions per second. Both describe the same run.

These are teaching inputs. Computing the corresponding metrics from a real
test requires request-level records: scheduled arrival, actual dispatch,
received content and timestamps, and the stream's termination status. Those
records explain errors and queues that a total tokens-per-second figure omits.

<a id="что-входит-в-измерение"></a>

## Define the measurement boundary

[Harvard ML Systems, Benchmarking Granularity](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/benchmarking/benchmarking.qmd#sec-benchmarking-benchmarking-granularity-3855)
distinguishes an operation, a complete model, and the end-to-end request path.
An attention-kernel test compares implementations for specified tensor shapes.
A complete forward pass includes the other layers and communications.
A client test adds transport, input processing, queues, and delivery.
A speedup measured at one level needs another measurement at the next.

Suppose a sequential path spends 80 ms outside the GPU and 20 ms in the
computation being optimized. Making that computation twice as fast changes
total latency from 100 to 90 ms, a speedup of about 1.11×. Even instantaneous
computation would leave 80 ms. This is Amdahl's law with the other stages held
constant. Under load, queueing may also change, so measure the resulting
end-to-end improvement.

A *benchmark harness* prepares requests, schedules them, invokes the system,
and collects results. Some of this work may lie outside its timer. MLPerf
LoadGen assigns these responsibilities to separate components:

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-benchmark-review-2026-10/mlperf-loadgen.svg]]

*MLCommons, [LoadGen: Integration Example and Flow](https://github.com/mlcommons/inference/blob/3fbc329939999c13d0a7b5e67fb2092287e06047/loadgen/README.md#integration-example-and-flow),
[unchanged original SVG](https://github.com/mlcommons/inference/blob/3fbc329939999c13d0a7b5e67fb2092287e06047/loadgen/loadgen_integration_diagram.svg),
Apache-2.0. Data preparation, request generation, the system under test, and
result collection have distinct roles.*

LoadGen does not interpret the input data or evaluate model accuracy itself.
Its user implements sample loading and invocation of the system under test.
MLPerf permits some Query Sample Library preparation outside the timed path.
That result cannot automatically be called latency from a user's text input
to the answer. State whether tokenization, the chat template, retrieval,
network routing, and postprocessing belong to your own timer.

<a id="временная-шкала-одного-запроса"></a>

## The timeline of one request

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/vllm-latency_diagram.png]]

*Aleksa Gordić, [Inside vLLM](https://www.aleksagordic.com/blog/vllm),
[original image](https://github.com/vllm-project/vllm-project.github.io/blob/main/assets/figures/2025-vllm-anatomy/latency_diagram.png).
The arrows mark dispatch and client receipt of output tokens. TTFT ends at the
first token; ITL measures the interval between adjacent tokens.*

Let dispatch occur at $t_0$, with output tokens received at $t_1,\ldots,t_O$.
Then

$$
\begin{aligned}
TTFT&=t_1-t_0,\\
ITL_i&=t_i-t_{i-1}\quad(i\ge2).
\end{aligned}
$$

Client TTFT includes everything between those timestamps: request transport,
server queueing, prefill, selection of the first token, and delivery.
The first token is selected from the prefill output; it does not require
an additional complete decode step. Time to the last token and the mean
interval after the first are

$$
\begin{aligned}
E2E&=t_O-t_0,\\
TPOT&=\frac{t_O-t_1}{O-1}\quad(O>1),\\
E2E&=TTFT+(O-1)TPOT.
\end{aligned}
$$

An answer arriving at 200, 260, 320, and 380 ms has TTFT 200 ms, three
60-ms intervals, and E2E 380 ms. Arrivals at 120, 140, 440, and 460 ms
give mean TPOT about 113.3 ms but include a 300-ms pause. Retaining the
interval sequence preserves what averaging loses.

```python
import numpy as np

def request_metrics(sent_ms, token_times_ms):
    times = np.asarray(token_times_ms, dtype=float)
    assert len(times) > 0 and times[0] >= sent_ms
    assert np.all(np.diff(times) >= 0)
    intervals = np.diff(times)
    return {
        'ttft': times[0] - sent_ms,
        'e2e': times[-1] - sent_ms,
        'itl': intervals.tolist(),
        'tpot': float(intervals.mean()) if len(intervals) else None,
    }

a = request_metrics(0., [200., 260., 320., 380.])
d = request_metrics(0., [120., 140., 440., 460.])
assert a['ttft'] == 200. and a['tpot'] == 60.
assert d['itl'] == [20., 300., 20.]
assert np.isclose(d['tpot'], 340. / 3)
assert request_metrics(0., [250.])['tpot'] is None
```

For one output token, there is no interval after the first: TPOT is N/A,
and E2E equals TTFT. Closing the HTTP stream, saving the result, or other
subsequent work may finish later. Include that time in a separately defined
extended completion latency.

<a id="событие-потока-и-токен-не-всегда-совпадают"></a>

### Stream events are not necessarily individual tokens

An SSE stream can contain role metadata, keepalives, usage statistics,
reasoning fragments, and answer text. A content-bearing event may include
several tokens. A network packet, an SSE event, and a model token are
different units.

Suppose six tokens arrive in three two-token chunks at 200, 300, and 500 ms.
The observed inter-chunk gaps are 100 and 200 ms. The client does not know
the individual emission times inside each chunk. Computing
$(500-200)/(6-1)=60$ ms does not recover those six generation timestamps.
Normalizing only by tokens after the first chunk instead gives
$(500-200)/(6-2)=75$ ms. Name the normalization used in the report.

Tools do not use identical metric names. In
[vLLM commit b8e9001](https://github.com/vllm-project/vllm/blob/b8e9001bf9568c801985889deff34d3b9fe6d0b0/vllm/benchmarks/lib/endpoint_request_func.py),
the corresponding backend's content-event intervals populate `itl`, while
`tpot` uses the output-token count. In
[AIPerf v0.13.0](https://docs.nvidia.com/aiperf/v0.13.0/reference/ai-perf-metrics-reference#inter-token-latency-itl),
ITL names an average similar to our TPOT; the distribution of inter-event
pauses is called ICL. It also handles the first response chunk separately.
Two columns labeled “ITL p99” are not comparable until these definitions agree.

For a reasoning model, distinguish the first token of any kind from the first
final-answer token. The user may never see the reasoning stream. A nonstreaming
endpoint permits measurement of the full response but cannot recover client
TTFT from its one final timestamp.

A broken connection may leave a partial answer. Retain it for diagnosis without
counting it as a complete successful result. A tool's backend-specific `success`
flag should agree with the expected protocol termination and generation
stop reason.

## Throughput and request rate

[Stanford CS336, Lecture 10, throughput_and_latency](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_10.py#L338)
contrasts per-request latency with aggregate batch output. If eight requests
each receive one token in a 50-ms step, aggregate output is
$8/0.05=160$ tokens/s. A batch of 32 taking 80 ms produces 400 tokens/s.
Yet each request now waits 80 rather than 50 ms for its next token.
These are illustrative values, not measurements of Llama or a particular GPU.

For a complete run, distinguish completion rate, output-token throughput, and
input-token throughput. Input throughput commonly counts logical request
tokens, including reused prefixes. It is not the number of positions newly
computed. Serving a system prefix from cache can increase logical input
throughput without measuring how quickly that prefix could be recomputed.

Give each successful request in the opening example 1024 input tokens.
The output lengths are 4, 4, 4, 4, and 1. Across ten seconds, the result is
0.5 successful completions/s, 1.7 output tokens/s, and 512 logical input
tokens/s. Combined input and output throughput is 513.7 tokens/s.
For consistency with completed-request metrics, these totals exclude any
work spent on the unsuccessful request.

<a id="goodput-считать-только-пригодный-результат"></a>

## Goodput: count acceptable completed results

Use these teaching constraints: TTFT at most 300 ms, mean TPOT at most 80 ms,
and successful completion. For a one-token answer, test TTFT without
applying the nonexistent TPOT.

| Request | TTFT, ms | Intervals after the first token, ms | TPOT, ms | Result |
|---|---:|---|---:|---|
| A | 200 | 60, 60, 60 | 60 | both latency constraints met |
| B | 500 | 20, 20, 20 | 20 | first token too late |
| C | 100 | 100, 100, 100 | 100 | generation too slow |
| D | 120 | 20, 300, 20 | 113.3 | generation too slow |
| E | 250 | none | N/A | TTFT constraint met |
| F | no complete result | none | N/A | timeout after 10 s |

Define $g_r=1$ only when request $r$ succeeds and satisfies every applicable
constraint. Over interval $T$,

$$
G=\frac{\sum_r g_r}{T}.
$$

A and E pass, giving $G=2/10=0.2$ requests/s. Five complete successfully,
so completion throughput is $5/10=0.5$ requests/s. Attainment among successful
requests is $2/5$; among all six sent requests it is $2/6$. Removing timeouts
from a product SLO's denominator would change whose experience it describes.

Run the Python blocks in order: the summary below uses `request_metrics`
defined above.

```python
records = [
    {'id': 'A', 'tokens': [200., 260., 320., 380.]},
    {'id': 'B', 'tokens': [500., 520., 540., 560.]},
    {'id': 'C', 'tokens': [100., 200., 300., 400.]},
    {'id': 'D', 'tokens': [120., 140., 440., 460.]},
    {'id': 'E', 'tokens': [250.]},
    {'id': 'F', 'tokens': None},
]

def summarize(records, duration_s, ttft_limit, tpot_limit):
    assert len(records) > 0 and duration_s > 0
    successful, good, output_tokens = 0, 0, 0
    for record in records:
        if record['tokens'] is None:
            continue
        metrics = request_metrics(0., record['tokens'])
        successful += 1
        output_tokens += len(record['tokens'])
        good += (metrics['ttft'] <= ttft_limit and
                 (metrics['tpot'] is None or metrics['tpot'] <= tpot_limit))
    return {
        'completed_rps': successful / duration_s,
        'goodput_rps': good / duration_s,
        'output_tps': output_tokens / duration_s,
        'attainment_all': good / len(records),
    }

summary = summarize(records, 10., 300., 80.)
assert summary == {
    'completed_rps': .5, 'goodput_rps': .2,
    'output_tps': 1.7, 'attainment_all': 2 / 6,
}
```

[vLLM's calculate_metrics](https://github.com/vllm-project/vllm/blob/b8e9001bf9568c801985889deff34d3b9fe6d0b0/vllm/benchmarks/serve.py#L587)
applies the selected goodput constraints jointly to successful requests.
It uses zero TPOT for the one-token goodput check but excludes such answers
from the ordinary TPOT distribution. Our record keeps N/A; the stated rule
gives the same pass/fail result without calling an absent measurement zero
latency.

DistServe also uses a different quantity: the maximum offered rate sustaining
a target SLO attainment, normalized by allocated GPUs. This is the boundary
of a load experiment, not the number of passing requests per second at an
arbitrary operating point. Label the run's `request_goodput` separately from
DistServe's sustainable-rate metric.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-benchmark-review-2026-10/distserve-figure1-original.png]]

*Yinmin Zhong et al., [DistServe, Figure 1, PDF p. 2](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf#page=2).
Original panels and caption. The authors use a 13B model, A100 80GB, 512 input
tokens, and 64 output tokens. Dashed limits concern P90 TTFT 0.4 s and
TPOT 0.04 s. These are historical measurements, not Bookvar results.*

The blue colocated curve crosses the TPOT limit around 1.6 requests/s,
before crossing the TTFT limit. Orange and green describe isolated phases.
Choosing the larger of their rates does not predict throughput for a complete
disaggregated service: count prefill resources, decode resources, and KV
transfer, as in [[00 Учебник/14 Inference и оптимизация/58a2 Раздельное обслуживание prefill и decode|the preceding chapter]].

Separate attainments of 90% do not guarantee 90% joint attainment. Among
100 requests, suppose ten fail only TTFT and another ten fail only TPOT.
Each metric passes for 90 requests, but both pass for only 80.
Compute the joint indicator from the original records.

<a id="почему-нужны-percentiles"></a>

## Percentiles, samples, and failure records

For $n$ sorted observations, the *nearest-rank* percentile selects element
$\lceil pn\rceil$, with $0<p\le1$. Other estimators interpolate neighboring
elements. For 10, 10, 10, and 1000 ms, nearest-rank p95 is 1000 ms;
NumPy's linearly interpolated p95 is 851.5 ms. The server did not behave
differently: the estimators differ on a four-observation sample. The reviewed
vLLM version uses `np.percentile`; record its method and library version.

A separate binomial model helps quantify what zero observed violations implies.
With violation probability $q$, the probability of seeing none in $n$
independent trials is $(1-q)^n$. Setting this to $\alpha=0.05$ gives a
one-sided 95% upper bound $q_{\mathrm{upper}}=1-\alpha^{1/n}$.

```python
from math import ceil

def nearest_rank(values, p):
    assert len(values) > 0 and 0 < p <= 1
    return sorted(values)[ceil(p * len(values)) - 1]

sample = [10., 10., 10., 1000.]
assert nearest_rank(sample, .95) == 1000.
assert np.isclose(np.percentile(sample, 95, method='linear'), 851.5)

# Один короткий ответ содержит один длинный интервал; другой — 99 коротких.
pooled_intervals = [500.] + [10.] * 99
assert np.percentile(pooled_intervals, 95) == 10.
assert np.mean([500., np.mean([10.] * 99)]) == 255.

# Односторонняя 95%-граница вероятности нарушения при нуле наблюдённых нарушений.
def zero_failure_upper_bound(n, alpha=.05):
    assert n > 0 and 0 < alpha < 1
    return 1 - alpha ** (1 / n)

assert .029 < zero_failure_upper_bound(100) < .030
assert .00029 < zero_failure_upper_bound(10_000) < .00030
```

Pooling all ITLs gives long answers greater weight. The example's 99
ten-millisecond intervals hide its single 500-ms interval: pooled p95 is
10 ms. Averaging each of the two requests first, then averaging requests,
gives 255 ms. These summarize different populations.

A hundred independent requests without a violation does not establish a
violation probability below 1%. The upper bound above is about 2.95%;
with ten thousand trials it is about 0.03%. This calculation assumes
independent trials sharing one probability. Bursts and common outages create
dependence. Thousands of neighboring tokens are not thousands of independent
repetitions of the experiment.

<a id="окна-и-объединение-реплик"></a>

### Windows and aggregation across replicas

For each quantile, state the window, sample count, and request class.
A daily p99 can hide a short incident; one distribution across all lengths
can hide systematically slow long prompts. Examine model, endpoint, tenant,
lengths, and termination reasons separately.

Do not average replica p99s. If one replica has a thousand observations and
another ten, two quantiles contain neither the right weights nor their full
distributions. Suppose the first thousand latencies are all 10 ms and the
other ten are all 1000 ms. The mean of the two p99s is 505 ms, but nearest-rank
p99 over all 1010 observations is 10 ms. Merge the observations or compatible
histogram bucket counts. Buckets limit precision: a bucket containing all
latencies from 500 to 1000 ms cannot reveal an exact p99 within it.

A failed request does not have latency zero. A five-second timeout says that
the request failed and no complete answer arrived before its deadline.
It does not reveal when the server would have finished without cancellation.
Report successful-response quantiles alongside error and timeout fractions,
not as a distribution covering every user.

<a id="open-loop-и-closed-loop-нагрузка"></a>

## Open-loop and closed-loop load

In a closed-loop test, a fixed number of clients sends the next request after
the preceding response, optionally with user think time. For $C=20$ clients,
mean response time $W=2$ s, and no think time, a stable rate is approximately
$C/W=10$ requests/s. If $W$ increases to four seconds, the same clients offer
about five requests/s. This models a fixed population of sequential users.

An open-loop test has an external arrival schedule independent of responses.
A homogeneous Poisson process has independent exponential inter-arrival
times with mean $1/\lambda$. Changing the interval distribution changes
burstiness even at the same mean rate. No chosen distribution establishes
fidelity to real traffic without a timestamped trace.

<a id="где-появляется-клиентская-очередь"></a>

### A concurrency limit can move the queue into the client

Schedule four requests at 0, 0.1, 0.2, and 0.3 s, but permit only one
in-flight request. Each takes one second and yields its first token 0.2 s
after dispatch. Actual dispatches occur at 0, 1, 2, and 3 s:

```python
import heapq

def dispatch_with_limit(arrivals, service_s, concurrency):
    assert concurrency >= 1 and service_s >= 0
    assert all(arrival >= 0 for arrival in arrivals)
    assert all(b >= a for a, b in zip(arrivals, arrivals[1:]))
    free = [0.] * concurrency
    sends, finishes = [], []
    for arrival in arrivals:
        sent = max(arrival, heapq.heappop(free))
        finished = sent + service_s
        heapq.heappush(free, finished)
        sends.append(sent)
        finishes.append(finished)
    return np.asarray(sends), np.asarray(finishes)

arrivals = np.array([0., .1, .2, .3])
sends, finishes = dispatch_with_limit(arrivals, 1., 1)
assert np.allclose(sends, [0., 1., 2., 3.])
first_tokens = sends + .2
assert np.allclose(first_tokens - sends, [.2] * 4)
assert np.allclose(first_tokens - arrivals, [.2, 1.1, 2., 2.9])
```

Starting each timer at dispatch gives TTFT 0.2 s for all four requests.
Relative to scheduled arrivals, the delays are 0.2, 1.1, 2.0, and 2.9 s.
The queue is in the generator, and the server can appear lightly loaded.
This example models a client limit, not continuous batching inside the GPU.

In [vLLM b8e9001](https://github.com/vllm-project/vllm/blob/b8e9001bf9568c801985889deff34d3b9fe6d0b0/vllm/benchmarks/serve.py#L984),
semaphore waiting is recorded as `client_queue_time`; backend request timing
starts after admission. Combining `--request-rate` and `--max-concurrency`
can separate intended rate from actual rate. Check both and dispatch lag.

This revision's `get_request` first generates gamma intervals, then, for a
constant rate, normalizes their total duration to `N/request_rate`.
The resulting finite schedule is therefore not literally a set of independent
exponential intervals even at `burstiness=1`. Save actual timestamps rather
than only the name of an arrival distribution.

If the test never creates requests that should have arrived during a slowdown,
their waiting disappears from the statistics: *coordinated omission*.
Closed-loop may be appropriate for fixed sequential users; it cannot replace
an independent arrival process without changing the modeled workload.
Inspect generator CPU, connections, and network capacity as well as the server.

<a id="workload-важнее-названия-gpu"></a>
<a id="состав-запросов-и-состояние-кэша"></a>

## Request composition and cache state

Identical mean prompt and answer lengths are insufficient. Consider two pairs:

| Workload | First request: input, output | Second request: input, output |
|---|---|---|
| A | 128, 512 | 8192, 32 |
| B | 128, 32 | 8192, 512 |

Both contain 8320 input tokens and 544 output tokens. In B, however, the long
answer follows the long prompt, so every decode step reads a longer context.

For one dense causal-attention layer, count key positions available during
ordinary decode. Prefill already selects the first output token.
At step $j=1,\ldots,O-1$, attention covers $S+j$ positions, including the
current token. The sum is $(O-1)S+O(O-1)/2$.

```python
def decode_key_positions(prompt_tokens, output_tokens):
    assert prompt_tokens >= 0 and output_tokens >= 1
    return (output_tokens - 1) * prompt_tokens + output_tokens * (output_tokens - 1) // 2

workload_a = [(128, 512), (8192, 32)]
workload_b = [(128, 32), (8192, 512)]
assert sum(s for s, o in workload_a) == sum(s for s, o in workload_b) == 8320
assert sum(o for s, o in workload_a) == sum(o for s, o in workload_b) == 544
a_positions = sum(decode_key_positions(s, o) for s, o in workload_a)
b_positions = sum(decode_key_positions(s, o) for s, o in workload_b)
assert b_positions > 9 * a_positions
```

This part of the work is more than nine times larger in B. That is not a
prediction of a ninefold full-model slowdown: weights, MLPs, prefill, kernels,
and parallelism are excluded. It shows what information is lost by
permuting output lengths independently of inputs.

Retain paired lengths, arrival order, shared prefixes, conversation branches,
sampling, and stop reasons. An agent's next call may depend on a tool result
and contain the preceding conversation. Shuffling independent HTTP calls
does not reproduce that dependency.

[MLPerf Agentic Inference, July 2026](https://mlcommons.org/2026/07/agentic-inference-for-mlperf-inference/)
describes a trace-based example. Within a trajectory, the client waits for
the complete response before sending the next turn. Future inputs come
from the recorded trace rather than the current model's answer.
That preserves sequential arrivals and reproducible history, allowing
serving comparisons, but differs from a live agent choosing its next action
from its own outputs. Task success requires separate evaluation.

A warm prefix cache and an empty cache define different experiments.
Repeating a dataset can accelerate the second run through accumulated KV.
For a cold-cache test, use a supported reset or independent same-length
prefixes and verify that hits are absent. For a warm test, specify prewarming
and measure actual reuse. Request hit rate differs from the fraction of reused
tokens: one very long prefix can dominate the latter.

Fix synthetic lengths after tokenization, not by character count. Retain
the chat template, BOS/EOS policy, truncation, and actual output lengths.
`ignore_eos` disables stopping on EOS; it does not by itself guarantee an
exact length, since other limits and stop conditions still apply. This regime
isolates compute shapes but changes KV release relative to natural stopping.

<a id="воспроизводимый-протокол"></a>

## A reproducible measurement protocol

Save the configuration needed to reconstruct the system: weight and tokenizer
revisions, chat template, precision and quantization, engine commit, CUDA and
driver versions, kernels, GPU count/type, TP/PP/DP/EP, topology, power limits,
and context limits. Record scheduler settings, KV budget, prefix caching,
chunked prefill, and speculative decoding separately.

For the workload, preserve the dataset revision, seed, request order, sampling,
stop rules, rate, concurrency, timeout, and retry policy. Raw records should
distinguish scheduled arrival, actual dispatch, first and subsequent outputs,
stream completion, and status. Correlate client and server records by request
ID. A difference between timestamps on different machines is not a measured
one-way network latency without clock alignment and an uncertainty estimate.

Separate warm-up from measurement. Cold startup can include GPU provisioning,
weight loading, communication-group creation, compilation, and graph
preparation. Warm-up can also change the allocator and device clocks.
Stable warm-kernel latency does not establish how quickly a cold replica
accepts real traffic. Measure the two regimes separately.

<a id="воспроизводимое-измерение-qwen3-4b"></a>
<a id="измерение-отдельных-фаз-пример-efficient-dl-systems"></a>

### Separating phases: the Efficient DL Systems example

[Week 8's seminar](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week08_inference_software/seminar.ipynb)
uses Qwen3-4B, Transformers 4.53.0, and FP16. The complete original notebook is
in the [[05 Источники/Courses/Efficient DL Systems/week08_inference_software/seminar.ipynb|course library]].
It follows a cached forward pass and `model.generate`, which performs both
prompt processing and answer continuation.

Suppose prefill takes 120 ms and complete `generate` takes 300 ms for ten
output tokens. Full output throughput is $10/0.3\approx33.3$ tokens/s.
After the first token, nine steps occupy the remaining 180 ms: 20 ms per
step, or 50 tokens/s per request. Dividing all ten tokens by 180 ms would
incorrectly report 55.6 tokens/s.

```python
def phase_accounting(prefill_s, total_s, output_tokens):
    assert 0 <= prefill_s <= total_s and output_tokens > 1
    remaining = output_tokens - 1
    decode_s = total_s - prefill_s
    assert decode_s > 0
    return {
        'full_output_tps': output_tokens / total_s,
        'decode_tpot_s': decode_s / remaining,
        'decode_output_tps': remaining / decode_s,
    }

phases = phase_accounting(.12, .30, 10)
assert np.isclose(phases['full_output_tps'], 100 / 3)
assert np.isclose(phases['decode_tpot_s'], .02)
assert np.isclose(phases['decode_output_tps'], 50.)
```

Subtracting these times is meaningful only if the first-token boundary
and other overheads agree. For a strict phase split, measure prefill and
subsequent steps within the same run rather than subtract an average prefill
from a different run with different input shapes.

CUDA work is asynchronous relative to the CPU. A Python-call timer that never
waits for the GPU can measure only enqueueing. Choose synchronization or
CUDA events for the intended boundary, as explained in
[[00 Учебник/10 ML Systems/07 Profiling ML-нагрузки|the profiling chapter]].

In the source seminar, prefill warm-up uses `use_cache=False`, measurement
uses `use_cache=True`, and allocator caches are cleared before attempts.
The input counter uses padded tensor size. When adapting it, distinguish
useful from padded positions, pass the correct attention mask, and warm up
the measured path. Compare requested output count with actual output count.
The notebook contains no saved GPU measurements; 120/300 ms above are
teaching inputs.

<a id="производительность-и-качество-образуют-один-контракт"></a>

### Keep the quality contract fixed

Changing precision or pruning can change answers. Evaluate quality on the
chosen task and performance for that same model revision. For sampling,
hold selection rules, constraints, and stopping fixed. For speculative
decoding, check the assumptions preserving the desired distribution rather
than demanding identical sampled trajectories.

An A/B comparison uses the same requests and arrival schedule. Alternate
repeated runs, for example A–B–B–A, so progressive warm-up or external load
does not coincide with only one configuration. Preserve independent repeated
runs, distributions, and raw records, not only the best result. Requests in
one overloaded queue are dependent; bootstrapping individual tokens does
not estimate uncertainty across complete experiments.

[[00 Учебник/18 Evaluation и методология/59 Оценивание моделей и контаминация|Model evaluation]]
covers quality and statistical comparisons. The object compared here includes
the model, runtime, scheduler, kernels, hardware, and workload.

<!-- serving-operations-completion: energy start -->
## Energy per result: measurement boundary and time window

A server can draw less power and still spend more energy on each answer.
Suppose configuration A consumes 600 W while completing 20 acceptable
responses per second. Configuration B consumes 400 W but completes only
10 acceptable responses per second. Their energy costs are 30 and 40 J per
acceptable response respectively. The lower-power configuration loses this
comparison because it takes longer to produce the same amount of useful work.
These are teaching inputs, not measurements of a GPU or serving engine.

Power is a rate, measured in watts: one watt is one joule per second.
Energy accumulates over time. For a measurement interval of duration $T$,

$$
\begin{gathered}
E=\int_0^T P(t)\,dt,\\
\overline P=E/T,\\
e_{\mathrm{good}}=E/N_{\mathrm{good}}.
\end{gathered}
$$

Here $N_{\mathrm{good}}$ counts results that meet the declared acceptance
conditions in the same interval. If $G=N_{\mathrm{good}}/T$ is goodput, then
$e_{\mathrm{good}}=\overline P/G$. Dividing watts by requests per second
gives joules per request. Dividing watts by the number of requests, without
the duration, does not.

### What the power meter includes

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/serving-operations-2026-10-29/power-measurement-boundaries.svg]]

*Vijay Janapa Reddi and contributors, Harvard ML Systems,
[Power Measurement Boundaries](https://github.com/harvard-edge/cs249r_book/blob/2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd/book/quarto/contents/vol1/benchmarking/benchmarking.qmd#sec-benchmarking-power-measurement-boundaries-982c),
[complete original SVG](https://mlsysbook.ai/vol1/benchmarking/benchmarking_files/mediabag/68403b70d9a611957fc377cc27f4711283a8df2f.svg),
CC BY-NC-SA 4.0. The original course diagram follows Tschand et al.,
[MLPerf Power](https://arxiv.org/abs/2410.12032v2). No cropping or redrawing.*

The diagram distinguishes a chip, an inference system-on-chip with external
DRAM, an inference node and a training installation. Its solid boundary and
dashed exclusions carry the meaning even without colour. In the node
example, accelerators, host memory, NIC, local storage and active cooling
are inside the boundary; remote storage is outside. In the training
example, compute nodes and measured or estimated network switches count,
while the storage node and data-centre cooling do not. These are the
illustrated accounting boundaries, not a claim that every benchmark uses
the same exclusions.

A GPU power counter measures a different object from an external meter
at the node's power inlet. Adding GPU telemetry to node-level AC power
would usually count the GPU twice. Conversely, reporting only device
power omits the host and the rest of that node. A comparison must name
the measurement point, included devices and any estimated components.
The [MLPerf Power paper, §III](https://arxiv.org/html/2410.12032v2#S3)
specifies these boundaries separately for the different deployment classes.

### Integrate the trace, then divide by results

Consider a 60-second interval with three constant-power phases. The same
server remains allocated throughout; 900 completed responses meet the
declared latency and quality conditions.

| Interval | Power | Duration | Energy |
|---|---:|---:|---:|
| 0–10 s | 300 W | 10 s | 3000 J |
| 10–40 s | 600 W | 30 s | 18,000 J |
| 40–60 s | 300 W | 20 s | 6000 J |
| Whole interval | 450 W average | 60 s | 27,000 J |

The energy per accepted result is $27{,}000/900=30$ J. The simple average
of the three listed power levels is 400 W, which is wrong for this trace:
the middle level lasts longer than the first. The time weights must stay
with the readings.

```python
from math import isclose, isfinite

def interval_energy(boundaries_s, interval_watts, good_results):
    if len(boundaries_s) != len(interval_watts) + 1 or not interval_watts:
        raise ValueError('one power value per time interval is required')
    if not all(isfinite(x) for x in [*boundaries_s, *interval_watts]):
        raise ValueError('times and power must be finite')
    durations = [b - a for a, b in zip(boundaries_s, boundaries_s[1:])]
    if any(dt <= 0 for dt in durations) or any(p < 0 for p in interval_watts):
        raise ValueError('positive durations and nonnegative power are required')
    if type(good_results) is not int or good_results < 0:
        raise ValueError('good_results must be a nonnegative integer')
    joules = sum(dt * p for dt, p in zip(durations, interval_watts))
    seconds = sum(durations)
    return joules, joules / seconds, (joules / good_results if good_results else None)

assert interval_energy([0, 10, 40, 60], [300, 600, 300], 900) == (27000, 450, 30)
assert interval_energy([0, 60], [400], 600) == (24000, 400, 40)
assert interval_energy([0, 60], [400], 0) == (24000, 400, None)
assert isclose(27000 / 3_600_000, .0075)  # joules to kWh
```

This integration is exact for the piecewise-constant teaching trace.
Point samples of an unknown physical signal require a declared interpolation
rule and enough sampling bandwidth; the code cannot recover a power spike
between samples. Cumulative energy counters can provide interval energy
directly, but their update cadence, reset/wrap behaviour and device boundary
still need to be checked. With several devices, align their measurement
windows before adding their energies.

If the idle baseline is 250 W, subtracting $250\cdot60=15{,}000$ J gives
12,000 J of energy above that baseline. This is an *incremental* estimate,
not the 27,000 J actually consumed within the stated boundary. It depends
on a valid baseline under comparable conditions. Report total and
baseline-subtracted energy under different names. Shared background work
also prevents attribution of the entire measured trace to one tenant
without a separate allocation rule.

### Sustained operation and the workload matter

Harvard's discussion of standardized measurement connects batching,
idle time and thermal behaviour in one paragraph:

<!-- source-excerpt: harvard-power-batch-idle-thermal -->
> Batch size creates a nonlinear relationship with power consumption that
> single-point measurements cannot characterize. Larger batches improve
> compute efficiency (better amortization of memory loads) but increase
> memory pressure and peak power requirements, meaning the most efficient
> batch size for throughput may differ from the most efficient batch size
> for energy. Measurement across multiple batch sizes is essential for a
> complete efficiency profile. System idle states deserve equal attention,
> particularly for intermittent edge workloads: a wake-word detection
> TinyML system that actively processes audio for only a small fraction
> of operating time may be dominated by idle power consumption rather than
> inference energy. Finally, sustained ML workloads can cause temperature
> increases that trigger thermal throttling and alter power consumption
> patterns—an effect particularly acute in edge devices, where thermal
> constraints limit sustained performance and make extended benchmarking
> runs essential for realistic characterization.
<!-- /source-excerpt -->

*Harvard ML Systems, [Standardized power measurement](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/benchmarking/benchmarking.qmd#sec-benchmarking-standardized-power-measurement-7fae),
CC BY-NC-SA 4.0. Complete original paragraph; only line wrapping changes.*

Those trends do not supply a universal batch-size curve. Measure the
chosen engine, shapes, cache regime and arrival process at each operating
point. Keep SLO and quality gates unchanged. Counting rejected requests
as energy-efficient responses would reward a server for discarding work;
counting only successful requests while omitting the energy spent on
failures would remove part of its bill.

For a finite workload, define whether energy accounting includes startup,
warm-up and the final drain. For a steady-state window, report which work
crosses its boundaries, as with throughput above. A short GPU kernel
benchmark cannot establish the energy per user answer for a service that
also tokenizes text, performs retrieval and waits on remote tools.

PUE applies only when converting an IT-energy boundary into a specified
facility-energy estimate. Applying it to a meter that already includes
the same cooling overhead counts that overhead again. A device TDP or
configured power limit is not a measurement of mean power. Keep the
instrument trace, timestamps, accepted-result count and boundary description
with the report so that joules per result can be recomputed.

*The Harvard-derived explanation and Bookvar teaching trace in this section
use CC BY-NC-SA 4.0. No power measurement was executed here.*

<!-- serving-operations-completion: energy end -->

<a id="поиск-рабочей-ёмкости"></a>

## Finding sustainable capacity

Increase offered load in steps. At each step, record intended and actual
arrival rates, completions, joint SLO attainment, latencies, errors, and
queue change. This illustrative table holds request composition fixed and
uses 60-second steps:

| Offered requests/s | Completed requests/s | TTFT p95, ms | TPOT p95, ms | Both SLOs met | Queue: start → end |
|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 250 | 30 | 100% | 0 → 0 |
| 4 | 4 | 500 | 40 | 99% | 0 → 0 |
| 6 | 6 | 900 | 60 | 97% | 0 → 0 |
| 8 | 6.2 | 3500 | 100 | 70% | 0 → 108 |

Require at least 95% joint attainment of TTFT ≤1000 ms and TPOT ≤70 ms.
Among tested points, six requests/s meets the requirement. Eight offers
slightly more completions but fails latency and adds approximately
$(8-6.2)\cdot60=108$ queued requests. The queue continues growing under
those rates; this is not a stationary operating point.

The result is a tested grid point, not a proven global maximum. Refine the
boundary between six and eight, repeat measurements, and test bursts.
If requests are rejected or cancelled, include those departures in the
queue balance; successful completions alone no longer explain the change.

Do not mix two time-accounting schemes. In a finite-cohort test, send a
specified set and wait for its completion; duration includes dispatch and
the final drain. In a steady-state window, count events occurring within
the window; some requests arrived earlier or will finish later. Taking the
count from one scheme and duration from the other produces a ratio that
describes neither experiment.

Also test changed length distributions, replica failure, and cold capacity
addition. [[00 Учебник/14 Inference и оптимизация/58c Queueing и capacity planning|Queueing and capacity planning]]
covers replica allocation and operating headroom.

<a id="sli-slo-и-алерты-в-эксплуатации"></a>

## SLIs, SLOs, and operational alerts

An SLI specifies an observed property and its counting rule. For a streaming
API, it might be the fraction of requests completing successfully within
TTFT and pause limits. An SLO supplies the target fraction and time window.
HTTP 200 sent before generation does not establish a successful answer.

At SLO 99.9%, the allowed bad-event fraction is 0.1%: ten bad events per
10,000 eligible requests. If 150 are bad, the observed fraction is 1.5%,
or 15 times the allowance. This ratio is the *burn rate*.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-benchmark-review-2026-10/sre-burn-budget.webp]]

*Steven Thurgood et al., Google, [The Site Reliability Workbook, Figure 5-4](https://sre.google/workbook/alerting-on-slos/#4-alert-on-burn-rate),
CC BY-NC-ND 4.0. Unchanged original graph. With the authors' constant-traffic
model, burn rates 1, 2, and 10 exhaust a 30-day budget in 30, 15, and 3 days.*

```python
def burn_rate(bad, eligible, slo=.999):
    assert eligible > 0 and 0 <= bad <= eligible and 0 < slo < 1
    return (bad / eligible) / (1 - slo)

burn = burn_rate(150, 10_000)
assert np.isclose(burn, 15.)
assert np.isclose(30 / burn, 2.)  # Прогноз при постоянной будущей доле и потоке.
assert np.isclose(14.4 * 1 / (30 * 24), .02)

def two_window_page(burn_1h, burn_5m, threshold=14.4):
    return burn_1h > threshold and burn_5m > threshold

assert two_window_page(20., 20.)
assert not two_window_page(20., 0.)
```

The longer window estimates accumulated consumption; the shorter window
checks whether deterioration continues. Google's example pages only when
burn rate exceeds 14.4 over both one hour and five minutes. At constant
traffic, one hour at that rate consumes 2% of a 30-day budget.
These thresholds illustrate an alerting policy, not mandatory LLM-service limits.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-benchmark-review-2026-10/sre-alert-windows.webp]]

*Thurgood et al., [The Site Reliability Workbook, Figure 5-6](https://sre.google/workbook/alerting-on-slos/#6-multiwindow-multi-burn-rate-alerts),
CC BY-NC-ND 4.0. Unchanged original. After a spike ends, the short green
estimate falls before the long orange one; the alert requires both to
exceed the threshold.*

With unequal traffic, sum bad and eligible event counts rather than averaging
minute-level percentages. One failure among ten night-time requests has a
different weight from a hundred among a hundred thousand daytime requests.
With no requests the ratio is undefined; missing data is not a measurement
of zero errors.

Define eligibility beforehand. Invalid client input may be outside a chosen
population, but a timeout of an admitted valid request cannot be removed
merely because its answer is incomplete. Specify how user cancellations
and retries are counted.

Join client and server traces by request ID. Compare queueing, prefill,
generation pauses, KV occupancy, preemption, KV transfer, and communication
errors to find the cause. GPU utilization can explain latency; it cannot
replace the user-facing SLO.

<a id="деградация-и-перегрузка"></a>

## Degradation and overload

When arrivals persistently exceed service, bounding the queue turns some
waiting into explicit rejections. Those rejections remain in the report.
Serving ten requests quickly and rejecting ninety does not meet a
hundred users' SLO just because the successful ten have low latency.

Retries add offered work. With up to two retries, one original request
can produce three attempts during a broad failure. Preserve both original
request IDs and attempt numbers: user success and server attempt rate
measure different levels.

Reducing maximum output, switching to a smaller model, or shortening context
changes the delivered result. Declare those as separate configurations with
their own quality evaluation. Test GPU failure, decode-worker loss after
prefill, restart, and drain separately. A stable successful run does not
demonstrate recovery after failure.

<a id="минимальный-честный-отчёт"></a>
<a id="что-сохранять-вместе-с-результатом"></a>

## Keep enough evidence to recompute the report

A reproducible result includes configuration, original workload, individual
request records, and aggregates with definitions. Each number needs an
identifiable numerator, measurement interval, and policy for excluding errors
from latency distributions.

Report the measured path, versions and topology, cache and warm-up regimes,
scheduled and actual load, attempts and errors, quantile populations, joint
SLO attainment, GPU time, and variation across repeated runs. Load and queue
plots show the operating regime behind the result. Preserve anonymized raw
records so aggregates can be recalculated without another GPU run.

<a id="практика-и-первоисточники"></a>

## Practice and primary sources

- [[05 Источники/Courses/Harvard ML Systems/vol1/benchmarking|Harvard ML Systems: complete Benchmarking chapter]]. Relevant sections: Micro/Macro/End-to-end benchmarks, Benchmark harness, System specifications, Run rules, Inference metrics, and power measurement.
- [Stanford CS336, Lecture 10](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_10.py): request latency versus aggregate throughput.
- [[05 Источники/Courses/Efficient DL Systems/week08_inference_software/seminar.ipynb|Efficient DL Systems, week 8]]: cached forward and complete generate in the original language.
- [[06 Практика/17 Развернуть наблюдаемый ML сервис|Practice: an observable ML service]].
- MLCommons, [LoadGen README, commit 3fbc329](https://github.com/mlcommons/inference/blob/3fbc329939999c13d0a7b5e67fb2092287e06047/loadgen/README.md): LoadGen, QSL, and system-under-test boundaries.
- Zhong et al., [DistServe, OSDI 2024](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf): Section 1 and Figure 1, sustainable load under TTFT/TPOT constraints.
- vLLM, [calculate_metrics and benchmark, commit b8e9001](https://github.com/vllm-project/vllm/blob/b8e9001bf9568c801985889deff34d3b9fe6d0b0/vllm/benchmarks/serve.py), [stream backend](https://github.com/vllm-project/vllm/blob/b8e9001bf9568c801985889deff34d3b9fe6d0b0/vllm/benchmarks/lib/endpoint_request_func.py), [CLI](https://docs.vllm.ai/en/stable/cli/bench/serve/). Version-specific metric and schedule definitions; canonical snapshot 20 October 2026.
- NVIDIA, [AIPerf v0.13.0: Metrics Reference](https://docs.nvidia.com/aiperf/v0.13.0/reference/ai-perf-metrics-reference): TTFT, ITL, ICL, and response chunks.
- Google, [Implementing SLOs](https://sre.google/workbook/implementing-slos/) and [Alerting on SLOs](https://sre.google/workbook/alerting-on-slos/): populations, windows, and error budgets.
- MLCommons, [Agentic Inference for MLPerf Inference](https://mlcommons.org/2026/07/agentic-inference-for-mlperf-inference/): dependent multi-turn workloads.

**Previous:** [[00 Учебник/14 Inference и оптимизация/58a2 Раздельное обслуживание prefill и decode|Disaggregated prefill and decode serving]]

**Next:** [[00 Учебник/14 Inference и оптимизация/58c Queueing и capacity planning|Queueing and capacity planning]]
