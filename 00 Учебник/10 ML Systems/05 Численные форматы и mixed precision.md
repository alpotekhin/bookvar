---
title: Численные форматы и mixed precision
type: textbook-chapter
status: draft
last_verified: 2026-07-23
source_language: mixed
---

# Численные форматы и mixed precision

Меньше бит означает меньше memory/communication traffic и доступ к быстрым
Tensor Cores, но одновременно меньше precision или dynamic range.

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

> **Адаптация, не дословная цитата:** EDLS week 2, PDF p. 18,
> slide “Mixed precision training”. Pure FP16 обычно нестабилен: GEMM можно
> выполнять в узком формате, тогда как softmax, normalization и accumulation
> часто требуют более широкой точности.

Autocast выбирает low precision для подходящих GEMM/convolution и оставляет
чувствительные reductions/normalization в более широком формате. Accumulation
часто шире input. Оптимизатор обновляет FP32 **master weights**, после чего
рабочая low-precision копия используется в forward.

Полный AMP-flow:

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

При gradient accumulation loss нормируют на microsteps, а `step/update`
вызывают только на границе логического batch. GEMM partial sums, reductions,
softmax statistics и optimizer update обычно сохраняют в более широком типе.

### Loss scaling

Малые FP16 gradients могут округлиться в ноль. Умножаем loss на $s$:

$$\tilde L=sL,\qquad \nabla\tilde L=s\nabla L,
$$

затем перед step делим gradients на $s$. Dynamic scaler уменьшает $s$ при
Inf/NaN и постепенно увеличивает после стабильных шагов. BF16 обычно меньше
нуждается в scaling благодаря 8-битной экспоненте.

## Почему AMP не всегда экономит optimizer memory

EDLS ledger для Adam:

- FP32: weight 4 + gradient 4 + moments 8 = 16 B/parameter;
- AMP: low-precision weight 2 + master weight 4 + gradient 2 (иногда 4) +
  moments 8 = 16–18 B/parameter.

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

Формально $q=\mathrm{clip}_{FP8}(\mathrm{round}(x/s))$, $\hat x=sq$. Scale
выбирают по `amax=max(abs(x))`, иногда по истории amax: current scaling быстрее
реагирует, delayed scaling дешевле, но отстаёт от смены распределения. Нужны
отдельные scales для weights, activations и gradients, а также saturation,
zero-rate, amax history и loss telemetry.

В MXFP8 маленький block (типичный размер в microscaling-спецификациях — 32
значения; поддержка platform-specific) делит общий power-of-two scale. Outlier
портит квантование только своего блока, но появляются metadata, требования к
axis/layout и зависимость от MX-aware kernels.

## Worked example: memory throughput

Оператор читает два и пишет один tensor по $10^8$ элементов. FP32 traffic —
1,2 GB, BF16 — 0,6 GB. При эффективных 1,5 TB/s:

$$t_{FP32}\ge0{,}80\text{ ms},\qquad t_{BF16}\ge0{,}40\text{ ms}.$$

Это верхняя надежда на 2× от bytes, не обещание end-to-end: launch, conversion
и compute остаются. Для Adam на 1B параметров AMP всё ещё может занимать
16–18 GB model states, хотя activations уменьшаются вдвое.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/foundation/hw_acceleration_energy_ladder.svg]]

*Источник визуала: Harvard CS249r, Hardware Acceleration,
`hw_acceleration_energy_ladder.svg`, commit `45ecc8d…`, CC BY-NC-SA 4.0.
Числа technology-specific; здесь важен качественный вывод о data movement.*

## MFU и HFU

$$MFU=\frac{\text{model FLOP per step}/t_{\text{step}}}
{P_{\text{peak}}}.
$$

MFU считает полезную модельную арифметику; recompute обычно не добавляют в
числитель. HFU считает фактически выполненную hardware arithmetic, поэтому при
checkpointing HFU может быть выше MFU. Сравнивать числа можно только при
одинаковой формуле FLOP и одном hardware precision peak. EDLS приводит
`MFU > 45%` лишь как rule of thumb, не как универсальную границу качества.

## Источники

- [EDLS week 2 lecture](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week02_fast_pipelines/lecture.pdf) — “Floating point numbers”, “Tensor Cores”, “Mixed precision training”, “Memory savings of AMP”, “FP8 training”; title locators used because incremental slides repeat in the PDF.
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
- [PyTorch AMP](https://pytorch.org/docs/stable/amp.html)

← [[02 Areas/ML & DL/00 Учебник/10 ML Systems/04 Арифметика Transformer и MoE|Арифметика Transformer и MoE]] ·
[[02 Areas/ML & DL/00 Учебник/10 ML Systems/06 Data pipeline, padding и packing|Data pipeline, padding и packing]] →
