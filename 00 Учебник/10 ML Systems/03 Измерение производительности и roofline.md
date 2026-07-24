---
title: Измерение производительности и roofline
type: textbook-chapter
status: draft
last_verified: 2026-07-24
source_language: mixed
---

# Измерение производительности и roofline

Benchmark — воспроизводимый эксперимент, а не одно число.

## Полные источники и практикум

- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol1/benchmarking|Harvard CS249r — Benchmarking]]: спецификация benchmark, harness, статистика и правила отчётности.
- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol2/performance_engineering|Harvard CS249r — Performance Engineering]]: Iron Law, roofline и диагностический процесс.
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week01_intro/seminar.ipynb|EDLS Week 1 — seminar notebook]]: измерения, которые можно повторить локально.

Встроенные копии сохраняют полный английский текст, исходные рисунки и
notebook. Глава ниже нужна как последовательность действий: сначала определить
границу эксперимента, затем получить распределение измерений и только после
этого объяснять результат через roofline.

## Benchmark specification и harness

До запуска фиксируют:

- commit, model/checkpoint, framework/compiler/CUDA/driver и kernel backend;
- accelerator, clocks/power mode, topology, CPU/RAM/storage;
- dtype, shapes, sequence lengths, batch, seeds и реальные input distribution;
- границу пути: kernel, operator, iteration, request или full service;
- warmup, repetitions, synchronization, cache policy, concurrency и load model.

Harness должен валидировать output, чередовать A/B, хранить raw samples и
отделять setup:

```python
def measure(fn, make_input, warmup=20, repeats=200):
    for _ in range(warmup):
        fn(make_input())
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        x = make_input()               # вне device interval
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(); y = fn(x); end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples
```

Код показывает границу device interval. Warmup поглощает allocation,
autotuning, JIT/compile и cache
fill. Cold-start измеряется отдельным сценарием. Locator: EDLS Week 1 PDF
pp. 17–20; Harvard Benchmarking § “Benchmark harness”,
`sec-benchmarking-benchmark-harness-09ea`, § “System specifications”,
`sec-benchmarking-system-specifications-6e80`, и § “Run rules”,
`sec-benchmarking-run-rules-c33f`.

## Гранулярности и метрики

| Уровень | Training | Inference |
|---|---|---|
| Kernel/operator | latency, achieved FLOP/s, bandwidth | то же |
| Step/request | step time, tokens/s, samples/s | TTFT, TPOT/ITL, E2E latency |
| Job/service | time-to-train, convergence, cost | throughput, p50/p95/p99, errors |

Для training обязательно задают global batch, sequence/token mix и target
quality: “samples/s” бессмысленен, если варианты сходятся к разному качеству.
Для online inference строят sweep offered load/concurrency; latency измеряют при
одинаковом arrival process и output lengths. Saturation видна как рост queueing
и хвостов, а не только plateau throughput.

Сообщают median, p90/p95/p99, confidence interval/bootstrap либо dispersion,
число samples и независимых runs. Minimum полезен как нижняя оценка kernel cost,
но не как production SLO.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/foundation/benchmarking_confidence_detectability.svg]]

*Оригинальная иллюстрация Harvard CS249r, Vol. I, Benchmarking,
§ “Statistical confidence”, locator
`sec-benchmarking-statistical-confidence`, commit `45ecc8d…`,
CC BY-NC-SA 4.0; файл не изменён. Маркер сопоставляет размер выборки с
минимальным различимым изменением и не даёт принять шум за регрессию.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/foundation/benchmarking_tail_latency_gap.svg]]

*Оригинальная иллюстрация Harvard CS249r Vol. I, Benchmarking,
§ “Inference metrics”, locator `sec-benchmarking-inference-metrics-78d4`,
CC BY-NC-SA 4.0.*

## Power и energy

Power — W в моменте, energy — $\int P(t)dt$ в J. Сравнивают также
J/token, J/sample и energy-to-quality. Sampling должен охватывать warm steady
state; короткий kernel может быть быстрее telemetry interval. Фиксируют power
cap, clocks, ambient/thermal state. Ускорение 2× при power 1.5× уменьшает energy
на работу до 0.75×; “ниже watts” само по себе не означает эффективнее.

## Roofline: уровень памяти имеет значение

$$I_L=\frac{F}{Q_L},\qquad
P\le\min(P_{\rm peak}, BW_L I_L),\qquad I_L^*=\frac{P_{\rm peak}}{BW_L}.
$$

$Q_L$ — traffic на выбранной границе L: HBM, L2 или shared memory. Поэтому
multi-level roofline содержит несколько наклонных ceilings. Kernel может быть
HBM-bound, но не L2-bound; cache hit меняет $Q_{\rm HBM}$, а не FLOP.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/performance/roofline-model.svg]]

*Оригинальная иллюстрация Harvard CS249r Vol. II, Performance Engineering,
§ “The roofline model”, locator `sec-performance-engineering-roofline`,
CC BY-NC-SA 4.0.*

### Worked ridge и A/B

Пусть $P_{\rm peak}=120$ TFLOP/s, HBM $BW=1.5$ TB/s. Тогда
$I^*=80$ FLOP/B.

- **A:** $I=20$ FLOP/B → ceiling $30$ TFLOP/s. Измеренные 24 TFLOP/s означают
  80% HBM-roof efficiency; удваивать peak compute почти бесполезно.
- **B (fusion/tiling):** те же FLOP, но traffic в 4 раза меньше:
  $I=80$ FLOP/B → ceiling 120 TFLOP/s. Измеренные 72 TFLOP/s = 60% нового
  ceiling и 3× speedup относительно A.

B достиг ridge: следующий bottleneck может быть instruction mix, occupancy,
Tensor Core eligibility или lower-level bandwidth. Нельзя объяснять недостающие
40% только “плохой эффективностью”.

Для GEMM $F\approx2mnk$, а идеальный
$Q_{\min}=s(mk+kn+mn)$. При $m=n=k=4096$, BF16:
$F\approx137.4$ GFLOP, $Q_{\min}\approx100.7$ MB и
$I\approx1365$ FLOP/B. Это algorithmic lower bound; profiler traffic включает
повторные reads, write allocation и intermediates.

## Защита от benchmark gaming

- не менять precision, accuracy или workload только у победителя;
- не исключать preprocessing/copies, если product path их оплачивает;
- не сравнивать warmed A с cold B;
- не выбирать единственную удачную shape;
- не скрывать OOM, failures, queueing и tail latency;
- публиковать both microbenchmark и end-to-end effect, raw samples и protocol.

## Источники

- [EDLS Week 1 lecture, pinned e632aa8](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week01_intro/lecture.pdf) — PDF pp. 13–20
- [Harvard CS249r Vol. I, Benchmarking, pinned 45ecc8d](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/benchmarking/benchmarking.qmd)
- [Harvard CS249r Vol. II, Performance Engineering, pinned 45ecc8d](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/performance_engineering/performance_engineering.qmd)

← [[02 Areas/ML & DL/00 Учебник/10 ML Systems/02 GPU, CUDA и иерархия памяти|GPU, CUDA и иерархия памяти]] ·
[[02 Areas/ML & DL/00 Учебник/10 ML Systems/04 Арифметика Transformer и MoE|Арифметика Transformer и MoE]] →
