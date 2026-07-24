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

## Mixed precision

> Training in pure FP16 hardly works. Some operations (matrix multiplication)
> can work, others (softmax, batch normalization) need higher precision.

Autocast выбирает low precision для подходящих GEMM/convolution и оставляет
чувствительные reductions/normalization в более широком формате. Accumulation
часто шире input. Оптимизатор обновляет FP32 **master weights**, после чего
рабочая low-precision копия используется в forward.

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

- [EDLS week 2 lecture](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week02_fast_pipelines/lecture.pdf)
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
- [PyTorch AMP](https://pytorch.org/docs/stable/amp.html)

← [[02 Areas/ML & DL/00 Учебник/10 ML Systems/04 Арифметика Transformer и MoE|Арифметика Transformer и MoE]] ·
[[02 Areas/ML & DL/00 Учебник/10 ML Systems/06 Data pipeline, padding и packing|Data pipeline, padding и packing]] →
