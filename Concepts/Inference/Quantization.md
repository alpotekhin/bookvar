---
title: Quantization
aliases:
  - Quantization
  - Квантизация
  - Model Quantization
type: concept
status: redirect
category: Inference
papers:
  - '[[02 Areas/ML & DL/Papers/QLoRA|QLoRA]]'
  - '[[02 Areas/ML & DL/Papers/LLaMA|LLaMA]]'
courses: []
sources:
  - '[Dettmers et al. — QLoRA (2023)](https://arxiv.org/abs/2305.14314)'
  - >-
    [Hugging Face —
    Quantization](https://huggingface.co/docs/transformers/quantization)
last_updated: '2026-10-29'
canonical_topic: 00 Учебник/14 Inference и оптимизация/57 Квантизация языковых моделей.md
---
# Quantization

Полное объяснение: [[00 Учебник/14 Inference и оптимизация/57 Квантизация языковых моделей|Квантизация языковых моделей]].

<!-- Compatibility anchors for existing Obsidian and web links. -->
<span id="quantization--квантизация-моделей" aria-hidden="true"></span>
<span id="зачем-это-нужно" aria-hidden="true"></span>
<span id="иерархия-числовых-форматов" aria-hidden="true"></span>
<span id="таблица-потребление-памяти-для-7b-модели" aria-hidden="true"></span>
<span id="post-training-quantization-ptq" aria-hidden="true"></span>
<span id="наивная-квантизация-round-to-nearest" aria-hidden="true"></span>
<span id="gptq--одноразовая-квантизация-весов" aria-hidden="true"></span>
<span id="awq--activation-aware-weight-quantization" aria-hidden="true"></span>
<span id="quantization-aware-training-qat" aria-hidden="true"></span>
<span id="nf4--normalfloat4-qlora" aria-hidden="true"></span>
<span id="gguf-и-llamacpp-квантизация-для-cpu" aria-hidden="true"></span>
<span id="деградация-качества" aria-hidden="true"></span>
<span id="когда-квантизовать-а-когда-нет" aria-hidden="true"></span>
<span id="сравнение-методов" aria-hidden="true"></span>
<span id="related-concepts" aria-hidden="true"></span>
