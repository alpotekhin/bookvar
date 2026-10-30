---
title: Triton и GPU kernels для LLM inference
type: concept
status: redirect
last_updated: '2026-10-29'
last_verified: 2026-07-21T00:00:00.000Z
canonical_topic: 05 Источники/Implementation studies/Triton GPU kernels.md
---
# Triton и GPU kernels для LLM inference

Полное объяснение: [[05 Источники/Implementation studies/Triton GPU kernels|Triton и GPU kernels для LLM inference]].

<!-- Compatibility anchors for existing Obsidian and web links. -->
<span id="triton-и-gpu-kernels-для-llm-inference" aria-hidden="true"></span>
<span id="модель-исполнения-программа-обрабатывает-tile" aria-hidden="true"></span>
<span id="почему-fusion-ускоряет-softmax" aria-hidden="true"></span>
<span id="matmul-три-измерения-и-память" aria-hidden="true"></span>
<span id="grouped-ordering-и-l2-locality" aria-hidden="true"></span>
<span id="autotuning" aria-hidden="true"></span>
<span id="где-triton-встречается-в-llm-runtime" aria-hidden="true"></span>
<span id="от-kernel-к-serving-engine" aria-hidden="true"></span>
<span id="практический-маршрут-чтения" aria-hidden="true"></span>
<span id="связанные-страницы" aria-hidden="true"></span>
<span id="источники" aria-hidden="true"></span>
