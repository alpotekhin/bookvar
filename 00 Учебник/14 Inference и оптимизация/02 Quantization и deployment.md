---
title: Quantization и deployment
type: textbook-chapter
status: canonical
last_updated: 2026-07-16
---

# Quantization и deployment

> [!abstract] Идея главы
> Большая часть времени decode уходит на чтение weights. Если хранить их в
> меньшем числе bits, модель занимает меньше памяти и быстрее переносится к
> compute units. Но реальный выигрыш зависит от kernels и hardware.

Quantization представляет weights и иногда activations меньшим числом bits.

| Режим | Что квантуется | Типичное применение |
|---|---|---|
| Weight-only INT8/INT4 | weights | memory-bound decode |
| W8A8/FP8 | weights + activations | server accelerators |
| GPTQ/AWQ | calibrated post-training weights | open-weight deployment |
| QLoRA | frozen 4-bit base + LoRA | дешёвый fine-tuning |

Для symmetric quantization:

$$q=\operatorname{round}(w/s),\qquad \hat w=sq.$$

Scale может быть общим для tensor, channel или небольшой group. Меньшая group
лучше следует локальному диапазону weights, но требует больше metadata.

Номинальные 4 bits не равны реальному bytes/parameter: нужны scales, metadata,
иногда zero points. Качество зависит от group size, outliers, calibration и
поддержки kernels.

Deployment — это также tokenizer, chat template, scheduler, cache policy,
observability и evaluation конкретной quantized версии.

## Проверка качества

Нельзя ограничиваться perplexity:

- math и code могут деградировать сильнее общего текста;
- long context чувствителен к quantized KV-cache;
- tool-call JSON ломается от редких token errors;
- конкретный backend может использовать другой quantization layout.

Сравнивать нужно точный artifact, runtime и hardware.

- [[02 Areas/ML & DL/Concepts/Inference/Quantization]]
- [[02 Areas/ML & DL/Papers/QLoRA]]
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Tim Dettmers](https://timdettmers.com/)
