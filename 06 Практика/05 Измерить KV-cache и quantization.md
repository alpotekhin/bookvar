---
title: Измерить KV-cache и quantization
type: practice
status: canonical
last_updated: 2026-07-16
---

# Измерить KV-cache и quantization

Для выбранной модели рассчитайте:

$$KV = 2\cdot L\cdot T\cdot n_{kv}\cdot d_h\cdot bytes$$

Сравните MHA и GQA при одинаковых `L`, `T`, `d_h`. Затем запустите модель в
[llama.cpp](https://github.com/ggml-org/llama.cpp) в двух quantization formats и
измерьте:

- файл weights;
- peak RAM/VRAM;
- prompt tokens/s;
- generation tokens/s;
- loss или небольшой task eval.

Quantization считается полезной только вместе с measured quality, а не по размеру
файла.

