---
title: Self-Consistency
aliases:
  - Self-Consistency
  - CoT-SC
  - Majority Voting
  - self-consistency decoding
type: concept
status: redirect
category: Reasoning
papers:
  - '[[02 Areas/ML & DL/Papers/Self-Consistency|Self-Consistency]]'
courses: []
sources:
  - >-
    [Wang et al. — Self-Consistency Improves Chain of Thought Reasoning in
    Language Models (ICLR 2023)](https://arxiv.org/abs/2203.11171)
last_updated: '2026-10-29'
canonical_topic: 00 Учебник/13 Reasoning и Test-time Compute/01 Test-time compute.md
canonical_heading: 4. Self-consistency
---
# Self-Consistency

Полное объяснение: [[00 Учебник/13 Reasoning и Test-time Compute/01 Test-time compute#4. Self-consistency|Вычисления во время ответа]].

<!-- Compatibility anchors for existing Obsidian and web links. -->
<span id="self-consistency" aria-hidden="true"></span>
<span id="зачем-это-нужно-одного-рассуждения-недостаточно" aria-hidden="true"></span>
<span id="алгоритм-три-шага" aria-hidden="true"></span>
<span id="шаг-1-prompt" aria-hidden="true"></span>
<span id="шаг-2-sample-diverse-reasoning-paths" aria-hidden="true"></span>
<span id="шаг-3-marginalize--majority-vote" aria-hidden="true"></span>
<span id="почему-majority-vote-а-не-weighted-average" aria-hidden="true"></span>
<span id="результаты-поразительные-улучшения" aria-hidden="true"></span>
<span id="arithmetic-reasoning" aria-hidden="true"></span>
<span id="commonsense-reasoning" aria-hidden="true"></span>
<span id="новые-sota" aria-hidden="true"></span>
<span id="scale-dependency-чем-больше-модель-тем-больше-прирост" aria-hidden="true"></span>
<span id="оптимальное-число-paths" aria-hidden="true"></span>
<span id="robustness" aria-hidden="true"></span>
<span id="uncertainty-estimation-бонусный-эффект" aria-hidden="true"></span>
<span id="zero-shot-cot--self-consistency" aria-hidden="true"></span>
<span id="связь-с-другими-методами" aria-hidden="true"></span>
<span id="self-consistency-vs-tree-of-thoughts-tot" aria-hidden="true"></span>
<span id="self-consistency-vs-ensemble" aria-hidden="true"></span>
<span id="self-consistency-vs-verifier" aria-hidden="true"></span>
<span id="ограничения" aria-hidden="true"></span>
<span id="key-papers" aria-hidden="true"></span>
<span id="related-concepts" aria-hidden="true"></span>
<span id="дополнительные-ресурсы" aria-hidden="true"></span>
