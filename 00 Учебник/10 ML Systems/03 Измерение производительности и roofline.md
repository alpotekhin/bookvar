---
title: Измерение производительности и roofline
type: textbook-chapter
status: draft
last_verified: 2026-09-06
source_language: mixed
source_unit_id:
  - lecture-05-roofline-performance
  - lecture-05-matrix-mystery
  - lecture-05-performance-recap-transition
  - lecture-06-benchmark-profile-experiment
  - lecture-06-section-169-add-dim-2048
  - lecture-06-section-199-matmul-dim-2048
  - lecture-06-section-202-matmul-dim-128
  - assignment-02-task-benchmarking-script
  - assignment-02-benchmark-synchronization
---

<a id="cs336-systems-roofline"></a>

# Измерение производительности и roofline

Утверждение «новое ядро быстрее на 30%» имеет смысл только вместе с условиями
измерения. Первый запуск включает компиляцию и выделение памяти, следующий может
читать данные из кэша, а несинхронный вызов CUDA закончится для CPU раньше, чем
GPU выполнит работу. Даже честно полученное среднее скрывает редкие задержки,
которые определяют время синхронного шага или пользовательский SLO.

Поэтому benchmark — это воспроизводимый эксперимент, а не одно число. Сначала
фиксируют границу измеряемого пути и рабочую нагрузку, затем собирают
распределение повторных измерений и только после этого объясняют результат через
объём вычислений, движение данных и модель roofline.

## Нулевой результат: сначала доказать корректность

Быстрый kernel, вычисляющий другую функцию, не является оптимизацией. До первого
таймера сравнивают output с понятной reference-реализацией, а для обучаемого
оператора — ещё и gradients. Проверяют не одну удобную форму, а минимум:

- обычный размер и размер с неполным последним tile;
- несколько batch и sequence lengths;
- causal и non-causal режимы, если оба поддерживаются;
- dtype, для которого заявляется ускорение;
- конечность результата и численный допуск, соответствующий порядку редукции.

Именно такой порядок задаёт Stanford CS336 Assignment 2: reference PyTorch
implementation и тесты предшествуют benchmarking. Допуск не выбирают после
того, как увидели ошибку; его фиксируют вместе с контрактом операции.

## Спецификация эксперимента и измерительный стенд

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
`sec-benchmarking-statistical-confidence`; [исходный SVG](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/benchmarking/images/svg/benchmarking_confidence_detectability.svg),
CC BY-NC-SA 4.0; файл не изменён. Маркер сопоставляет размер выборки с
минимальным различимым изменением и не даёт принять шум за регрессию.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/foundation/benchmarking_tail_latency_gap.svg]]

*Оригинальная иллюстрация Harvard CS249r Vol. I, Benchmarking,
§ “Inference metrics”, locator `sec-benchmarking-inference-metrics-78d4`;
[исходный SVG](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/benchmarking/images/svg/benchmarking_tail_latency_gap.svg),
CC BY-NC-SA 4.0. Сравните медиану с правым хвостом распределения: небольшой
сдвиг типичного запроса может скрывать существенно худший p99.*

## Мощность и энергия

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
§ “The roofline model”, locator `sec-performance-engineering-roofline`;
[исходный SVG](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/performance_engineering/images/svg/roofline-model.svg),
CC BY-NC-SA 4.0.*

### Численный пример и сравнение A/B

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

## Почему соседние размеры дают разное время

Roofline задаёт верхнюю границу, но не обещает гладкую зависимость скорости от
размера матрицы. Один kernel разбивает output на tiles, а scheduler запускает
blocks волнами по конечному числу SM. Если новая строка или колонка создаёт ещё
один tile, последний wave может оказаться почти пустым. FLOP добавилось мало, но
число scheduling waves выросло на единицу.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/l05-p20.png]]

*Измеренный shape sweep в [Stanford CS336 Lecture 5, p. 20](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_05.pdf).
Зубцы повторяются при росте размера, хотя число операций растёт плавно. Это
наблюдение относится к показанным kernel и accelerator, а не к GEMM как
математической операции.*

Курс разбирает загадку в два шага. Сначала сравнивает размеры, согласованные и
не согласованные с внутренними tiles, затем показывает границу, на которой
добавление единственного элемента создаёт новый неполный wave.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/l05-p47.png]]

*Stanford CS336 Lecture 5, p. 47: периодичность измерения сопоставляется с
aligned/unaligned shapes. Сравнивать нужно одинаковые dtype, backend и режим
прогрева.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/stanford-cs336-2026/systems/l05-p48.png]]

*Stanford CS336 Lecture 5, p. 48: переход 1792→1793 используется как конкретный
пример wave quantization. Число 1792 не является универсальным оптимальным
размером: tile shape и число resident blocks зависят от выбранного kernel и GPU.*

Практический вывод — запускать shape sweep вокруг рабочих размеров, включая
`n-1`, `n` и `n+1` у предполагаемой границы tile. Если зубцы сохраняются,
профиль должен подтвердить изменение grid, tail tiles или occupancy. Если нет,
нужно искать другой механизм: recompilation, cache transition, алгоритм
библиотеки или изменение Tensor Core eligibility.

## Защита от benchmark gaming

- не менять precision, accuracy или workload только у победителя;
- не исключать preprocessing/copies, если product path их оплачивает;
- не сравнивать warmed A с cold B;
- не выбирать единственную удачную shape;
- не скрывать OOM, failures, queueing и tail latency;
- публиковать both microbenchmark и end-to-end effect, raw samples и protocol.

## Практика и первоисточники

- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol1/benchmarking|Harvard CS249r — Benchmarking]]: спецификация benchmark, harness, статистика и правила отчётности.
- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol2/performance_engineering|Harvard CS249r — Performance Engineering]]: Iron Law, roofline и диагностический процесс.
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week01_intro/seminar.ipynb|EDLS Week 1 — seminar notebook]]: измерения, которые можно повторить локально.
- [[05 Источники/Courses/Harvard ML Systems/tinytorch/19_benchmarking|TinyTorch 19 — Benchmarking]]: исполняемый модуль с единым измерительным стендом, повторными запусками и стандартным форматом результата.

- [EDLS Week 1 lecture, pinned e632aa8](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week01_intro/lecture.pdf) — PDF pp. 13–20
- [Harvard CS249r Vol. I, Benchmarking, pinned 45ecc8d](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/benchmarking/benchmarking.qmd)
- [Harvard CS249r Vol. II, Performance Engineering, pinned 45ecc8d](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/performance_engineering/performance_engineering.qmd)
- [Stanford CS336 Lecture 5, pinned `8b59b507`, pp. 20–23 and 40–48](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_05.pdf) — roofline, tiling и matrix-size mystery.
- [Stanford CS336 Lecture 6, pinned `8b59b507`, benchmark/profile experiment](https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_06.py) — измерения elementwise и matmul kernels на разных shapes.
- [Stanford CS336 Assignment 2, pinned `ca8bc81`, pp. 3–5](https://github.com/stanford-cs336/assignment2-systems/blob/ca8bc81a59b70516f7ebb2da4808daade877c736/cs336_assignment2_systems.pdf) — synchronization-safe benchmark contract.

← [[02 Areas/ML & DL/00 Учебник/10 ML Systems/02 GPU, CUDA и иерархия памяти|GPU, CUDA и иерархия памяти]] ·
[[02 Areas/ML & DL/00 Учебник/10 ML Systems/04 Арифметика Transformer и MoE|Арифметика Transformer и MoE]] →
