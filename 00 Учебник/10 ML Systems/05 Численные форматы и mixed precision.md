---
title: Численные форматы и mixed precision
type: textbook-chapter
status: draft
last_verified: 2026-09-06
source_language: mixed
source_unit_id:
  - lecture-05-low-precision-intensity
  - assignment-02-task-mixed-precision-accumulation
  - assignment-02-task-benchmarking-mixed-precision
---

<a id="cs336-systems-precision"></a>

# Численные форматы и mixed precision

Меньше бит означает меньше memory/communication traffic и доступ к быстрым
Tensor Cores, но одновременно меньше precision или dynamic range.

## От определения формата до работающего training loop

Практический критерий здесь двойной: ускорение должно быть измерено на
целевом железе, а численная эквивалентность — проверена по loss curve и
downstream quality, не только по отсутствию `NaN`.

## Encoding, rounding и пределы

Нормальное двоичное число хранит
$x=(-1)^s2^{e-\mathrm{bias}}(1.f)$. Экспонента задаёт range, fraction —
расстояние между соседними числами. При cast значение округляется (обычно
round-to-nearest, ties-to-even); слишком малое становится subnormal или нулём,
слишком большое — `inf` или крайним конечным значением согласно формату и
операции. FP16 точнее BF16 около единицы, но BF16 сохраняет порядок range FP32.
Накопление длинной суммы поэтому делают в FP32 даже при узких operands.

## Форматы

| Формат | Знак | Экспонента | Мантисса | Практический контекст |
|---|---:|---:|---:|---|
| FP32 | 1 | 8 | 23 | storage/accumulation, стабильная база |
| TF32 | 1 | 8 | 10 | режим Tensor Core для matmul на NVIDIA Ampere+, не storage dtype |
| FP16 | 1 | 5 | 10 | точнее BF16 около 1, но малый range |
| BF16 | 1 | 8 | 7 | range FP32, меньше precision; recent CPU/TPU/GPU |
| FP8 E4M3 | 1 | 4 | 3 | weights/activations в FP8 training |
| FP8 E5M2 | 1 | 5 | 2 | больший range, часто gradients |

TF32 был включён по умолчанию для некоторых PyTorch matmul на Ampere до
PyTorch 1.12 — это version-specific поведение, которое нельзя переносить на
другую версию. FP8 throughput также зависит от GPU (в EDLS пример — H100),
CUDA, kernels, shapes и accumulation dtype.

Tensor Core доступен не по одному `dtype`: kernel должен поддерживать format,
layout и shapes. Эффективность выше при подходящем tile/alignment, а маленькая
или «хвостатая» матрица может не насытить устройство. Фиксируют GPU/CUDA/
PyTorch/math mode, затем в Profiler/Nsight проверяют имя kernel и Tensor Core
instructions. Догадываться об использовании Tensor Core по типу tensor нельзя.

## Mixed precision

Выполнять всё обучение в FP16 обычно нельзя без потери устойчивости. Матричные
умножения хорошо работают в узком формате, тогда как softmax, нормализация и
накопление сумм часто требуют более высокой точности.

В стандартном PyTorch AMP параметры модели остаются FP32. `autocast` выбирает
тип отдельных операций: подходящие матричные умножения могут использовать FP16,
а чувствительные редукции — FP32. Тип накопителя также может быть шире типа
операндов. Это не то же самое, что заранее перевести все параметры модели в FP16.

Другая схема явно хранит рабочие веса пониженной точности и отдельную FP32-копию
для обновления оптимизатором (master weights). Её поддерживает конкретная система
обучения; сам вызов `torch.autocast` такой постоянной пары копий не требует.
См. [PyTorch AMP](https://docs.pytorch.org/docs/2.8/amp.html).

Ниже — один шаг native AMP; предполагаются `model` с параметрами FP32,
`optimizer` и заранее созданный `torch.amp.GradScaler("cuda")`:

```python
optimizer.zero_grad(set_to_none=True)
with torch.autocast("cuda", dtype=torch.float16):
    loss = model(batch)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)       # до clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
scaler.step(optimizer)           # пропускает step при inf/nan
scaler.update()
```

При gradient accumulation деление loss на число microsteps корректно для
микропакетов с одинаковым числом учитываемых токенов. При разных размерах
нужна нормировка по их общему числу, как в главе о distributed training.
`step/update` вызывают только на границе логического batch. GEMM partial sums, reductions,
softmax statistics и optimizer update обычно сохраняют в более широком типе.

### Два dtype у одной матричной операции

Запись «GEMM выполняется в BF16» неполна. Нужно отдельно назвать тип operands,
тип накопителя и тип результата. Tensor Cores могут умножать BF16 inputs,
накапливать partial sums в FP32, а затем записать BF16 output. Если принудительно
оставить accumulator в узком формате, каждое сложение округляется раньше и
ошибка растёт с длиной редукции.

Assignment 2 предлагает наблюдать это на матричном умножении, меняя только
accumulation policy. Содержательный результат эксперимента — не «BF16 плох», а
траектория ошибки при росте внутренней размерности K. Небольшая ошибка отдельного
произведения становится систематической, когда тысячи rounded partial sums
складываются в один элемент.

Для отчёта tracing проходит через весь шаг:

| Компонент | Что зафиксировать |
|---|---|
| параметры и activation inputs | storage dtype |
| GEMM operands | dtype после autocast |
| GEMM accumulator | фактическая accumulation policy/backend |
| softmax и norm statistics | reduction dtype |
| loss и loss scale | dtype до/после scaling |
| gradients | storage dtype и dtype редукции между ranks |
| optimizer moments/master weights | persistent dtype |

Таблица важнее одного глобального слова `mixed`: два запуска с одинаковыми
BF16 weights могут отличаться accumulators, reduction kernels и обновлением
optimizer state.

### Loss scaling

Малые FP16 gradients могут округлиться в ноль. Умножаем loss на $s$:

$\tilde L=sL,\qquad \nabla\tilde L=s\nabla L,
$

затем перед step делим gradients на $s$. Dynamic scaler уменьшает $s$ при
Inf/NaN и постепенно увеличивает после стабильных шагов. BF16 обычно меньше
нуждается в scaling благодаря 8-битной экспоненте.

## Почему AMP не всегда экономит память оптимизатора

Для Adam полный расчёт памяти выглядит так:

- обычный FP32 Adam: вес 4 + градиент 4 + два момента 8 =16 байт на параметр;
- показанный native AMP с FP32-параметрами: те же постоянные 16 байт на параметр;
  временные преобразования, кэш autocast и активации считаются отдельно;
- явная схема с рабочими весами FP16/BF16: вес 2 + master copy 4 + градиент 2
  (или 4) + моменты 8 =16–18 байт на параметр. Это отдельное соглашение хранения.

Главный выигрыш памяти часто приходит от activations, а не states. Это не
противоречит ускорению: GEMM и communications всё равно могут стать быстрее.

## FP8 и MXFP8

FP8 требует scale, потому что один tensor может содержать разные диапазоны.
Per-tensor scaling прост, но outlier ухудшает использование уровней. Per-block
scaling даёт каждой группе свой scale. **MXFP8** — microscaling: маленькие blocks
FP8 values разделяют компактно представленный scale. Это повышает локальную
адаптацию range, но требует hardware/software support и корректного выбора осей
block. Результаты H100/Transformer Engine нельзя объявлять свойством «FP8
вообще».

Формально $q=\operatorname{cast}_{FP8}(x/s)$, $\hat x=sq$. Операция
$\operatorname{cast}_{FP8}$ округляет к сетке выбранного формата с заданным
правилом насыщения, а не до ближайшего целого. Например, в E4M3 около единицы
шаг равен 0.125: при $s=1$ число 1.2 округляется до 1.25, а не до 1.
Масштаб
выбирают по `amax=max(abs(x))`, иногда по истории amax: current scaling быстрее
реагирует, delayed scaling дешевле, но отстаёт от смены распределения. Нужны
отдельные scales для weights, activations и gradients, а также saturation,
zero-rate, amax history и loss telemetry.

Эта формула — схема quantize/dequantize, а не точное описание любой FP8
реализации. Реальный контракт должен указать granularity scale, формат E4M3 или
E5M2, saturation rule, stochastic/deterministic rounding, accumulation dtype и
то, когда обновляется `amax` history.

В MXFP8 маленький block (типичный размер в microscaling-спецификациях — 32
значения; поддержка platform-specific) делит общий power-of-two scale. Outlier
портит квантование только своего блока, но появляются metadata, требования к
axis/layout и зависимость от MX-aware kernels.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/performance/block-quantization.svg]]

*Оригинальная иллюстрация Harvard CS249r, Vol. II, Performance Engineering,
§ “Block quantization”, locator `sec-performance-engineering-quantization`;
[исходный SVG](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/performance_engineering/images/svg/block-quantization.svg),
CC BY-NC-SA 4.0; файл не изменён. Разбиение tensor на
локальные блоки показывает, почему выброс ухудшает scale только своей группы.*

## Worked example: memory throughput

Оператор читает два и пишет один tensor по $10^8$ элементов. FP32 traffic —
1,2 GB, BF16 — 0,6 GB. При эффективных 1,5 TB/s:

$t_{FP32}\ge0{,}80\text{ ms},\qquad t_{BF16}\ge0{,}40\text{ ms}.$

Это верхняя надежда на 2× от bytes, не обещание end-to-end: launch, conversion
и compute остаются. Для Adam на 1 млрд параметров native AMP с FP32-параметрами сохраняет около 16 GB
постоянного состояния; явная схема с master copy — 16–18 GB по таблице выше.
Объём некоторых активаций уменьшается, но не обязательно ровно вдвое для всего графа.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/foundation/training_optimizer_memory.svg]]

*Оригинальная иллюстрация Harvard CS249r, Vol. I, Training,
§ “Memory decomposition”, locator `sec-model-training-memory-decomposition`;
[исходный SVG](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/training/images/svg/training_optimizer_memory.svg),
CC BY-NC-SA 4.0; файл не изменён. Ledger отделяет веса,
градиенты и состояния оптимизатора: узкий forward dtype не означает такое же
уменьшение всей памяти training.*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/foundation/hw_acceleration_energy_ladder.svg]]

*Источник визуала: Harvard CS249r, Hardware Acceleration; [исходный
SVG](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol1/hw_acceleration/images/svg/hw_acceleration_energy_ladder.svg),
CC BY-NC-SA 4.0.
Числа technology-specific; здесь важен качественный вывод о data movement.*

## MFU и HFU

$MFU=\frac{\text{model FLOP per step}/t_{\text{step}}}
{P_{\text{peak}}}.
$

MFU считает полезную модельную арифметику; recompute обычно не добавляют в
числитель. HFU считает фактически выполненную hardware arithmetic, поэтому при
checkpointing HFU может быть выше MFU. Сравнивать числа можно только при
одинаковой формуле FLOP и одном пиковом значении для выбранной точности.
Значение `MFU > 45%` иногда используют как грубый ориентир, но оно не является
универсальной границей качества реализации.

## Практика и первоисточники

- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week02_fast_pipelines/lecture.pdf|EDLS Week 2 — лекция]]: численные форматы, Tensor Cores и mixed precision.
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week02_fast_pipelines/seminar/practice.ipynb|EDLS Week 2 — семинарская тетрадь]]: исходный код и измерения.
- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol1/training|Harvard CS249r — Training]]: связь численной точности с полным training pipeline.

- [EDLS week 2 lecture](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week02_fast_pipelines/lecture.pdf) — “Floating point numbers”, “Tensor Cores”, “Mixed precision training”, “Memory savings of AMP”, “FP8 training”; title locators used because incremental slides repeat in the PDF.
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
- [PyTorch AMP](https://pytorch.org/docs/stable/amp.html)
- [Stanford CS336 Assignment 2, pinned `ca8bc81`, pp. 6–9](https://github.com/stanford-cs336/assignment2-systems/blob/ca8bc81a59b70516f7ebb2da4808daade877c736/cs336_assignment2_systems.pdf) — accumulation experiment, component dtype trace and mixed-precision benchmark.

← [[02 Areas/ML & DL/00 Учебник/10 ML Systems/04 Арифметика Transformer и MoE|Арифметика Transformer и MoE]] ·
[[02 Areas/ML & DL/00 Учебник/10 ML Systems/06 Data pipeline, padding и packing|Data pipeline, padding и packing]] →
