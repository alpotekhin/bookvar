---
title: Test-time Compute
aliases:
  - Test-time Compute
  - Inference-time Scaling
  - Test-time Scaling
  - TTS
  - Thinking Tokens
type: concept
status: redirect
category: Reasoning
papers:
  - '[[02 Areas/ML & DL/Papers/DeepSeek-R1 Reasoning via RL|DeepSeek-R1]]'
courses: []
sources:
  - >-
    [HuggingFace Blog — What is Test-time Compute and How to Scale
    It?](https://huggingface.co/blog/Kseniase/testtimecompute)
  - >-
    [Build ML — Test-Time Compute Scaling: A Practical
    Guide](https://buildml.substack.com/p/test-time-compute-scaling-a-practical)
  - >-
    [Introl Blog — Inference-Time Scaling
    Research](https://introl.com/blog/inference-time-scaling-research-reasoning-models-december-2025)
  - >-
    [Emerge Haus — Test-Time Compute in Generative
    AI](https://www.emerge.haus/blog/test-time-compute-generative-ai)
last_updated: '2026-10-29'
canonical_topic: 00 Учебник/13 Reasoning и Test-time Compute/01 Test-time compute.md
---
# Test-time Compute

Полное объяснение: [[00 Учебник/13 Reasoning и Test-time Compute/01 Test-time compute|Вычисления во время ответа]].

<!-- Compatibility anchors for existing Obsidian and web links. -->
<span id="test-time-compute" aria-hidden="true"></span>
<span id="зачем-это-нужно-новая-ось-масштабирования" aria-hidden="true"></span>
<span id="ключевой-эксперимент-o1" aria-hidden="true"></span>
<span id="что-сделал-openai-сентябрь-2024" aria-hidden="true"></span>
<span id="scaling-curve-thinking-tokens-vs-accuracy" aria-hidden="true"></span>
<span id="механизмы-test-time-compute" aria-hidden="true"></span>
<span id="1-extended-chain-of-thought" aria-hidden="true"></span>
<span id="2-self-verification" aria-hidden="true"></span>
<span id="3-majority-voting--best-of-n" aria-hidden="true"></span>
<span id="4-tree-search" aria-hidden="true"></span>
<span id="как-обучить-модель-думать-дольше" aria-hidden="true"></span>
<span id="подход-deepseek-r1-pure-rl" aria-hidden="true"></span>
<span id="подход-o1-предположительно-rl--process-supervision" aria-hidden="true"></span>
<span id="подход-qwen3-thinking-budget" aria-hidden="true"></span>
<span id="зачем-это-paradigm-shift" aria-hidden="true"></span>
<span id="train-time-vs-test-time-сравнение" aria-hidden="true"></span>
<span id="экономический-аргумент" aria-hidden="true"></span>
<span id="compute-optimal-inference" aria-hidden="true"></span>
<span id="timeline" aria-hidden="true"></span>
<span id="related-concepts" aria-hidden="true"></span>
