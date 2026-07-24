---
title: Измерение производительности и roofline
type: textbook-chapter
status: draft
last_verified: 2026-07-23
source_language: mixed
---

# Измерение производительности и roofline

Секунды без протокола измерения мало что значат. Нужно определить workload,
границы измеряемого пути, warmup, синхронизацию и распределение результатов.

## Bandwidth, FLOP/s и arithmetic intensity

Для операции:

$$I=\frac{F}{Q}\quad[\text{FLOP/byte}],
$$

где $F$ — арифметическая работа, $Q$ — байты между вычислителем и выбранным
уровнем памяти. Roofline ограничивает достигнутую скорость:

$$P(I)\le\min(P_{\text{peak}},\,BW\cdot I).
$$

Ridge point $I^*=P_{\text{peak}}/BW$. Слева операция bandwidth-bound, справа
compute-bound.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/foundation/hw_acceleration_roofline_elbow.svg]]

*Источник: Harvard CS249r, Hardware Acceleration,
`hw_acceleration_roofline_elbow.svg`, CC BY-NC-SA 4.0.*

### GEMM: считать bytes правильно

Для $C_{m\times n}=A_{m\times k}B_{k\times n}$:

$$F\approx2mnk,
\qquad Q_{\min}=s(mk+kn+mn),
$$

где $s$ — bytes/element. При $m=n=k=4096$, BF16:
$F\approx137{,}4$ GFLOP, $Q_{\min}\approx100{,}7$ MB,
$I\approx1365$ FLOP/byte. Это оптимистическая оценка с идеальным reuse; лишние
materialization и reread снижают intensity.

## Корректный timing CUDA

> Due to possible side-effects (preallocation, caching), warmup and
> randomization are often necessary.

Минимальный протокол:

```python
for _ in range(10):
    fn()
torch.cuda.synchronize()

samples = []
for _ in range(100):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(); fn(); end.record()
    end.synchronize()
    samples.append(start.elapsed_time(end))
```

Обычный `time.time()` без synchronize измерит enqueue на CPU. CUDA Events
измеряют интервал на device timeline. Warmup исключает lazy initialization,
JIT/compile, autotuning и первые allocation. PyTorch caching allocator означает,
что reserved memory и live tensor memory — разные величины.

`torch.backends.cudnn.benchmark=True` ищет быстрый kernel для наблюдавшихся
shapes; при меняющихся shapes стоимость поиска и нестабильность могут перевесить
выигрыш.

## Не одно число, а распределение

Сообщают минимум/median/p90/p95/p99, число повторов, разброс и cold/warm режим.
Minimum приближает «чистую» kernel cost, median — типичный запуск, хвосты —
помехи и operational risk.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/foundation/benchmarking_tail_latency_gap.svg]]

*Источник: Harvard CS249r, Benchmarking,
`benchmarking_tail_latency_gap.svg`, CC BY-NC-SA 4.0.*

Для сравнения A/B чередуют варианты, фиксируют inputs/seeds/clocks и повторяют
на нескольких запусках. Иначе thermal throttling, соседние процессы и
динамические clocks смешиваются с эффектом оптимизации.

## End-to-end против microbenchmark

Kernel microbenchmark отвечает «насколько быстр этот оператор?». End-to-end
benchmark включает tokenization, loading, copies, synchronization, framework
dispatch и postprocessing. Обе цифры нужны: первая локализует предел, вторая
проверяет эффект на продукт.

## Источники

- [EDLS week 1 lecture](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week01_intro/lecture.pdf)
- [Harvard CS249r, Benchmarking](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/benchmarking/benchmarking.qmd)
- [PyTorch benchmark utilities](https://pytorch.org/docs/stable/benchmark_utils.html)

← [[02 Areas/ML & DL/00 Учебник/10 ML Systems/02 GPU, CUDA и иерархия памяти|GPU, CUDA и иерархия памяти]] ·
[[02 Areas/ML & DL/00 Учебник/10 ML Systems/04 Арифметика Transformer и MoE|Арифметика Transformer и MoE]] →
