---
title: Оптимизировать один Transformer step
type: practice
status: canonical
last_updated: 2026-10-26
last_verified: 2026-10-26
source_language: mixed
---

# Оптимизировать один Transformer step

Это итоговая работа по training performance. Выполняйте [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week06_dl_arithmetic/seminar/practice.ipynb|EDLS Week 6 seminar notebook]] и сверяйтесь с [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week06_dl_arithmetic/lecture.pdf|полной 146-слайдовой лекцией]].

Зафиксируйте baseline Transformer step. До оптимизации составьте три ведомости:

1. parameters, gradients, optimizer states и peak activations в bytes;
2. FLOP по attention, MLP и vocabulary projection;
3. HBM и communication traffic, который можно оценить.

После этого по одному примените доступные оптимизации: fused optimizer,
compiled/fused elementwise path, FlashAttention, activation checkpointing,
mixed precision и улучшение data pipeline. После каждого изменения проверяйте
loss/gradients и повторяйте одинаковый benchmark.

Итоговая таблица должна содержать step time, tokens/s, MFU либо достижимую
roofline efficiency, peak memory и качество. Для отклонённой оптимизации тоже
сохраните результат: объяснение, почему она не помогла, является частью работы.

<!-- BEGIN REVIEW: performance evidence 2026-10-26 -->

<a id="performance-evidence-profiling-regression-capacity"></a>

## Profile, compare, size, and choose

An optimization can shorten a kernel without shortening the training step.
A faster training step can require more steps to reach the same quality.
A serving configuration can produce more tokens per second while missing
the latency target. Keep these as separate experiments: the training route
above measures an optimizer step; the capacity exercise below concerns a
serving workload. Their timing and quality contracts are different.

Harvard's [Performance Engineering laboratory](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/labs/vol2/lab_09_perf_engineering.py),
Harvard Edge contributors, CC BY-NC-SA 4.0, gives this complete sequence.
The original English is retained:

<!-- BEGIN SOURCE: Harvard performance lab sequence -->
1. Localize the bottleneck before optimizing.
2. Prove regressions with baseline, variance, and detectability.
3. Convert measured bottlenecks into capacity headroom and cost.
4. Defend an optimization trade-off and reject an alternative.
5. Save a memo that V2-10 inference can inherit.
<!-- END SOURCE: Harvard performance lab sequence -->

“V2-10” is the next inference lab in that course. Here the handoff is a
reproducible experiment report. The original lab uses configurable teaching
constants for its profiles, “p99,” capacity and quality loss. Those numbers
are not measurements. Use your own observations for an optimization claim;
the small numerical examples below are labelled fixtures or assumptions.

### A. Find the interval that delays the result

Start with the unchanged EDLS baseline. Record the model revision, global
batch, sequence lengths, padding, dtype, attention implementation and
optimizer. Include hardware, driver, CUDA and framework versions. State
whether the timer includes loading data, copying it to the device and the
optimizer update. Preserve the warmup policy and the input sequence so
the candidate runs the same workload.

Use the timing boundaries in the
[[02 Areas/ML & DL/00 Учебник/10 ML Systems/03 Измерение производительности и roofline|measurement chapter]].
Record ordinary timings without a profiler, then collect a short profile
with labels for input preparation, H2D, forward, backward and optimizer.
The [[02 Areas/ML & DL/00 Учебник/10 ML Systems/07 Profiling ML-нагрузки|profiling chapter]]
contains a complete PyTorch Profiler example and an Nsight capture command.
Data already on the GPU is a useful controlled experiment, but it does not
test the real input pipeline.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/a2-nsight-trace.png]]

*Stanford CS336 course staff, [Assignment 2, Figure 1, p. 5](https://github.com/stanford-cs336/assignment2-systems/blob/ca8bc81a59b70516f7ebb2da4808daade877c736/cs336_assignment2_systems.pdf#page=5),
[MIT](https://github.com/stanford-cs336/assignment2-systems/blob/ca8bc81a59b70516f7ebb2da4808daade877c736/LICENSE).
Existing lossless figure crop, 3278 × 1986, reused without further changes.
The trace belongs to the course, not to the benchmark you are about to run.*

On this trace, locate the module ranges at the bottom, then the corresponding
CUDA kernels above them. Compare the same time interval across tracks.
GPU Active, warp occupancy and DRAM bandwidth describe different properties.
A busy GPU can still execute an inefficient kernel; a low sum of CPU work
can still contain a wait that delays the next GPU operation.

Write one causal hypothesis before editing code. For example, if the GPU
waits for the next batch, predict that preparing it concurrently will remove
that exposed interval. If short kernels are separated by host dispatch gaps,
predict that capture or fusion will change those gaps. After the change,
check both the predicted trace change and the unprofiled full-step time.
For a training optimization, compare outputs and gradients on fixed inputs,
then compare the learning behavior required by the task. Record peak memory
even if the candidate is faster: an improvement that causes OOM on the
required sequence length is not usable for that workload.

Do not add the durations of overlapping kernels to reconstruct step time.
For an explicitly assumed schedule with 40 ms of preparation completely
overlapped by 70 ms of compute, the interval is 70 ms, not 110 ms. Reducing
preparation to 20 ms does not shorten it. If preparation instead takes
90 ms, 20 ms becomes exposed. Confirm which schedule the trace supports
before applying this calculation to the program.

### B. Separate an observed difference from the ability to detect one

Use repeated, independently started runs as the unit of comparison when
the question is run-to-run performance. Warm iterations in one process share
caches, allocation and thermal state. Randomize or alternate A/B order under
a declared cache policy. For paired runs sharing the same experimental
conditions, analyze paired differences; the independent-samples example
below deliberately assumes independence instead.

SciPy's [two-sample t-test documentation](https://github.com/scipy/scipy/blob/54ef5423f2e4376230ec3bfda6912a07a50958e3/scipy/stats/_stats_py.py#L6551),
SciPy developers, [BSD-3-Clause](/licenses/scipy-1.18.0-LICENSE.txt), states:

<!-- BEGIN SOURCE: SciPy independent means -->
Calculate the T-test for the means of *two independent* samples of scores.

This is a test for the null hypothesis that 2 independent samples
have identical average (expected) values. This test assumes that the
populations have identical variances by default.
<!-- END SOURCE: SciPy independent means -->

Set `equal_var=False` for Welch's test when you do not assume equal population
variances. Give the candidate as the first argument so the estimated difference
is candidate minus baseline. SciPy defines the sign as follows:

<!-- BEGIN SOURCE: SciPy statistic sign -->
The statistic is calculated as ``(np.mean(a) - np.mean(b))/se``, where
``se`` is the standard error. Therefore, the statistic will be positive
when the sample mean of `a` is greater than the sample mean of `b` and
negative when the sample mean of `a` is less than the sample mean of
`b`.
<!-- END SOURCE: SciPy statistic sign -->

The following eight values per variant are an illustrative fixture of
per-run mean step times in milliseconds. They are not GPU measurements.
The code requires SciPy 1.18.0; the confidence-interval API exists from 1.11.
Save this first block as `compare_runs.py` and run it in an isolated environment:

```bash
uv run --with scipy==1.18.0 python compare_runs.py
```

The remaining three Python blocks use only the standard library.

```python
from statistics import mean
from scipy.stats import ttest_ind

baseline = [100, 103, 98, 101, 99, 104, 97, 102]
candidate = [96, 98, 95, 97, 94, 99, 93, 96]
result = ttest_ind(candidate, baseline, equal_var=False)
interval = result.confidence_interval(confidence_level=0.95)
print(f"Mean difference, ms: {mean(candidate) - mean(baseline):.3f}")
print(f"95% CI, ms: [{interval.low:.3f}, {interval.high:.3f}]")
print(f"p-value: {result.pvalue:.6f}")
```

The mean difference is −4.500 ms, the interval is [−6.907, −2.093] ms,
and the p-value is about 0.001350. Under the independent-samples assumptions,
the interval supports a decrease in the population mean. It does not give
an interval for request p99, guarantee this effect on another GPU, or establish
that answer quality is unchanged. The p-value is not the probability that
the candidate is slower. Failure to reject equality would also not prove
that a candidate satisfies an allowed-regression margin.

For a request-p99 claim, retain request-level observations and the run grouping.
Do not relabel run means as p99 or average run p99s into an overall p99.
The [[02 Areas/ML & DL/00 Учебник/10 ML Systems/03 Измерение производительности и roofline#Разброс времени и неопределённость оценки|measurement chapter]]
explains quantiles, uncertainty and dependence between observations.

#### Plan sample size before seeing the result

Harvard's local regression helper computes
$1.96\sqrt{2}\,\sigma/\sqrt n$ and labels it a minimum detectable effect.
For two independent normal means with known equal standard deviation,
this is the 95% two-sided interval half-width. A power calculation also
needs the probability of detecting the chosen shift.

The [NIST sample-size guide](https://itl.nist.gov/div898/handbook/prc/section2/prc222.htm)
separates significance $\alpha$, missed-effect probability $\beta$, variance
and the shift to detect. Extending its known-variance normal calculation
to two equal-sized independent samples uses

$$
SE_\Delta=\sigma\sqrt{\frac{2}{n}},
\qquad h=z_{1-\alpha/2}SE_\Delta.
$$

An approximate positive shift for power $1-\beta$ is

$$
\delta\approx
\left(z_{1-\alpha/2}+z_{1-\beta}\right)SE_\Delta.
$$

Assume a baseline mean of 100 ms, known standard deviation 6 ms,
$\alpha=0.05$ and power 80%. With eight observations per variant,
$h\approx5.88$ ms, while $\delta\approx8.40$ ms. A true shift equal
to $h$ has only about 50% power in this model. A 5-ms target shift
requires about 23 observations per variant under the same assumptions.

```python
from math import ceil, sqrt
from statistics import NormalDist

normal = NormalDist()
sigma = 6.0  # Known standard deviation assumed for this planning model.
alpha, desired_power, n = 0.05, 0.80, 8
z_cut = normal.inv_cdf(1 - alpha / 2)
z_power = normal.inv_cdf(desired_power)
se = sigma * sqrt(2 / n)
half_width = z_cut * se
planned_shift = (z_cut + z_power) * se
target_shift = 5.0
n_per_variant = ceil(2 * sigma**2 * ((z_cut + z_power) / target_shift)**2)
power_at_half_width = (
    1 - normal.cdf(z_cut - half_width / se)
    + normal.cdf(-z_cut - half_width / se)
)
print(f"CI half-width: {half_width:.2f} ms")
print(f"Approximate 80%-power shift: {planned_shift:.2f} ms")
print(f"Power at the half-width: {100 * power_at_half_width:.2f}%")
print(f"Planned observations per variant: {n_per_variant}")
```

This is a normal planning approximation with known variance, not a p99
sample-size formula. In an actual benchmark, variance is estimated and
observations may be dependent. State the estimand, effect size, model and
decision rule before collecting the final comparison. Repeatedly testing
until a small p-value appears changes the error rate.

### C. Turn a serving rate into a capacity bound

For serving, measure completed useful work per unit time under the required
input/output lengths, arrival process, error policy and latency SLO. A low-load
p99 is not a service rate. Dividing a baseline p99 by an optimized p99 does
not by itself establish how many tokens per second a replica can sustain.
Keep the throughput and latency observations from the same load experiment.

If each identical serving unit can sustain rate $\mu$ under the stated
conditions, $u$ units give an ideal additive capacity $C=u\mu$. With demand
$\lambda$, define utilization and the lab's headroom convention as

$$
\rho=\frac{\lambda}{C},
\qquad h=\frac{C}{\lambda}-1.
$$

The fraction of capacity left unused is instead $1-\rho$. Requiring
25% headroom means $C\ge1.25\lambda$, hence $\rho\le0.8$ and
20% of capacity unused. A “25% reserve” requirement must specify which
denominator it uses.

For a calculation based on the original lab's cloud constants, assume
4500 tokens/s per unit, demand 19000 tokens/s, daily unit cost 980 USD and
a deliberately assumed 12% derating. This coefficient is a source-scenario
input, not a measured universal production penalty. The effective rate
is $4500\cdot0.88=3960$ tokens/s per unit.

```python
from math import ceil

rate = 4500 * (1 - 0.12)  # Assumed scenario inputs, not a benchmark.
demand = 19000
target_headroom = 0.25
unit_cost = 980
required_units = ceil(demand * (1 + target_headroom) / rate)
for units in (5, required_units):
    capacity = units * rate
    utilization = demand / capacity
    headroom = capacity / demand - 1
    print(units, int(capacity), f"{100 * utilization:.2f}%",
          f"{100 * headroom:.2f}%", units * unit_cost)
```

| Units | Assumed capacity, tokens/s | Utilization | Headroom | Cost/day, USD |
|---:|---:|---:|---:|---:|
| 5 | 19800 | 95.96% | 4.21% | 4900 |
| 6 | 23760 | 79.97% | 25.05% | 5880 |

Five units exceed average demand but miss the headroom target. Six satisfy
this arithmetic bound. Neither row proves a p99 SLO: shared resources,
batching and queues can change the measured capacity. Verify the selected
configuration under load, including bursts and long requests. For an
unbounded queue with sustained arrivals above departures, there is no
stationary finite latency distribution to extrapolate from a low-load p99.
The [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/58b Benchmarking, SLO и эксплуатация inference|serving benchmark protocol]]
sets out that experiment.

### D. Apply constraints before ranking candidates

An option can be cheap and fast but fail the quality requirement. Check
each constraint separately before ranking feasible options. The original
lab summarizes its conjunction as follows; this is a decision rule, not
a claim that its configured quality-loss constants were measured:

<!-- BEGIN SOURCE: Harvard performance feasibility -->
```
feasible =
  p99 <= track_slo
  and headroom >= target_headroom
  and quality_loss <= risk_budget
  and daily_cost <= cost_ceiling

score = p99/SLO + headroom_penalty + cost_penalty + quality_penalty
```
<!-- END SOURCE: Harvard performance feasibility -->

These rows are the original helper's calculated outputs for the default
cloud scenario: demand multiplier 1, five units, risk budget 2% and cost
ceiling 6615 USD. They are fixtures, not hardware results or measured quality:

| Candidate | Assumed p99, ms | Headroom | Cost/day, USD | Assumed quality loss | Failed constraints |
|---|---:|---:|---:|---:|---|
| Targeted bottleneck fix | 157.36 | 58.24% | 5390 | 0.7% | none |
| Add capacity | 200.50 | 72.94% | 10584 | 0.1% | latency, cost |
| Precision/cache compression | 149.49 | 58.24% | 4410 | 1.3% | none |
| Algorithmic path | 306.82 | 19.72% | 5880 | 1.0% | latency, headroom |

With the source's 180-ms latency limit, 25% headroom and 6615-USD cost limit,
two candidates pass at a 2% quality budget. Tightening only quality to 1%
rejects compression and leaves the targeted fix. These are conditional
decisions on the supplied table. They do not establish that compression
has the same quality effect on a real model.

```python
rows = [
    {"name": "targeted", "p99": 157.36, "headroom": .5824,
     "cost": 5390, "quality": .7},
    {"name": "scale", "p99": 200.50, "headroom": .7294,
     "cost": 10584, "quality": .1},
    {"name": "precision", "p99": 149.49, "headroom": .5824,
     "cost": 4410, "quality": 1.3},
    {"name": "algorithmic", "p99": 306.82, "headroom": .1972,
     "cost": 5880, "quality": 1.0},
]

def feasible(row, quality_budget):
    return (row["p99"] <= 180 and row["headroom"] >= .25
            and row["cost"] <= 6615 and row["quality"] <= quality_budget)

for budget in (2.0, 1.0, 0.5):
    allowed = [row["name"] for row in rows if feasible(row, budget)]
    print(budget, allowed if allowed else "hold: no feasible option")
```

The 0.5% case has no feasible option. A ranking function must not turn its
least-bad row into launch approval. Within the feasible set, a weighted
score expresses a chosen preference among cost, latency and quality; its
weights do not come from the performance measurements. Preserve the
individual metrics so another reviewer can apply different constraints
or examine the Pareto frontier.

For a real quality gate, name the metric, dataset, model revision and
uncertainty. A percentage-point accuracy loss, a relative increase in error
rate and a safety-failure rate are not interchangeable “risk percentages.”
Do not hide these distinctions in one configurable scalar.

### Save the evidence and the rejected experiment

The report should connect each conclusion to an artifact. Store the workload
and software configuration, ordinary per-run timings, correctness and quality
checks, the short trace that supports the diagnosis, the candidate change,
the post-change trace and the final unprofiled comparison. Retain failed
runs, timeouts and rejected optimizations with their conditions. A result
on one isolated kernel is not a full-step result; a CPU execution of the
statistics fixtures is not a CUDA performance experiment.

For the serving extension, also retain offered load, completions, errors,
request lengths, latency distribution, capacity assumptions and cost basis.
Conclude with the accepted or held configuration and the evidence required
to change that decision. The original
[[05 Источники/Courses/Harvard ML Systems/labs/vol2/lab_09_perf_engineering|Harvard lab]]
and [[05 Источники/Courses/Harvard ML Systems/slides/vol2/09_performance_engineering|complete lecture with speaker notes]]
remain available separately; their scenario constants are not benchmark data.

<!-- END REVIEW: performance evidence 2026-10-26 -->

## Альтернативная итоговая работа: TinyTorch

[[05 Источники/Courses/Harvard ML Systems/tinytorch/20_capstone|TinyTorch 20 — Capstone]]
— отдельный проект со своим учебным фреймворком, зависимостями и предыдущими
модулями. Его вход — реализованные компоненты TinyTorch, а результат —
сравнение исходной и оптимизированной модели по протоколу этого курса.
Он не является готовой реализацией расположенного выше задания EDLS и не
подключается к его notebook без адаптации. Выберите один маршрут; если
проходите оба, сохраняйте окружения и отчёты раздельно.
