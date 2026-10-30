---
title: Distillation
aliases:
  - Distillation
  - Knowledge Distillation
  - KD
type: concept
status: redirect
category: Training
papers:
  - '[[02 Areas/ML & DL/Papers/Phi-2|Phi-2]]'
  - '[[02 Areas/ML & DL/Papers/DeepSeek-R1 Reasoning via RL|DeepSeek-R1]]'
  - '[[02 Areas/ML & DL/Papers/Gemma|Gemma]]'
courses: []
sources:
  - >-
    [Hinton et al. — Distilling the Knowledge in a Neural Network
    (2015)](https://arxiv.org/abs/1503.02531)
last_updated: '2026-10-29'
canonical_topic: 00 Учебник/12 Post-training и Alignment/08 Reasoning distillation.md
canonical_heading: Дистилляция промежуточных представлений
---
# Distillation

Полное объяснение: [[00 Учебник/12 Post-training и Alignment/08 Reasoning distillation#Дистилляция промежуточных представлений|Дистилляция моделей рассуждения]].

<!-- Compatibility anchors for existing Obsidian and web links. -->
<span id="distillation--knowledge-distillation" aria-hidden="true"></span>
<span id="идея-передача-знаний-от-большой-модели-к-маленькой" aria-hidden="true"></span>
<span id="soft-labels-и-temperature-scaling" aria-hidden="true"></span>
<span id="temperature-в-softmax" aria-hidden="true"></span>
<span id="loss-function" aria-hidden="true"></span>
<span id="почему-kl-а-не-mse" aria-hidden="true"></span>
<span id="типы-distillation-для-llm" aria-hidden="true"></span>
<span id="1-response-based-distillation-самый-распространённый" aria-hidden="true"></span>
<span id="2-synthetic-data-distillation" aria-hidden="true"></span>
<span id="3-reasoning-distillation-deepseek-r1" aria-hidden="true"></span>
<span id="4-feature-based-distillation" aria-hidden="true"></span>
<span id="когда-distillation-лучше-обучения-с-нуля" aria-hidden="true"></span>
<span id="capacity-gap-problem" aria-hidden="true"></span>
<span id="ограничения" aria-hidden="true"></span>
<span id="distillation-в-индустрии-20242025" aria-hidden="true"></span>
<span id="related-concepts" aria-hidden="true"></span>
