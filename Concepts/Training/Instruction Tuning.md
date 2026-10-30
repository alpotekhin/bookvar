---
title: Instruction Tuning
aliases:
  - instruction fine-tuning
  - SFT
  - Supervised Fine-Tuning
  - инструкционная настройка
type: concept
status: redirect
category: Training
papers:
  - '[[02 Areas/ML & DL/Papers/Flan-T5-PaLM]]'
  - '[[02 Areas/ML & DL/Papers/InstructGPT]]'
  - '[[02 Areas/ML & DL/Papers/LLaMA 2]]'
courses:
  - >-
    [[02 Areas/ML & DL/Courses/SHAD LLM/Week 1 -- Intro to LLMs|SHAD LLM -- Week
    1]]
sources:
  - >-
    [Wei et al. -- Finetuned Language Models Are Zero-Shot Learners (FLAN,
    2022)](https://arxiv.org/abs/2109.01652)
  - >-
    [Ouyang et al. -- Training language models to follow instructions with human
    feedback (InstructGPT, 2022)](https://arxiv.org/abs/2203.02155)
  - >-
    [Chung et al. -- Scaling Instruction-Finetuned Language Models (Flan-PaLM,
    2022)](https://arxiv.org/abs/2210.11416)
last_updated: '2026-10-29'
canonical_topic: 00 Учебник/12 Post-training и Alignment/01 SFT и instruction data.md
---
# Instruction Tuning

Полное объяснение: [[00 Учебник/12 Post-training и Alignment/01 SFT и instruction data|SFT и инструктивные данные]].

<!-- Compatibility anchors for existing Obsidian and web links. -->
<span id="instruction-tuning" aria-hidden="true"></span>
<span id="зачем-это-нужно-превращение-lm-в-assistant" aria-hidden="true"></span>
<span id="формат-данных" aria-hidden="true"></span>
<span id="базовый-формат-instruction--input--output" aria-hidden="true"></span>
<span id="chat-формат-modern-sft" aria-hidden="true"></span>
<span id="multi-turn-диалоги" aria-hidden="true"></span>
<span id="два-подхода-к-instruction-tuning" aria-hidden="true"></span>
<span id="подход-1-task-scaling-flanflan-t5flan-palm" aria-hidden="true"></span>
<span id="подход-2-quality-over-quantity-instructgptllama-2" aria-hidden="true"></span>
<span id="сравнение-подходов" aria-hidden="true"></span>
<span id="sft-как-step-1-в-rlhf-pipeline" aria-hidden="true"></span>
<span id="ключевые-результаты" aria-hidden="true"></span>
<span id="instructgpt-13b-vs-gpt-3-175b" aria-hidden="true"></span>
<span id="flan-t5-vs-t5" aria-hidden="true"></span>
<span id="качество-данных-что-работает-а-что-нет" aria-hidden="true"></span>
<span id="работает" aria-hidden="true"></span>
<span id="не-работает--вредит" aria-hidden="true"></span>
<span id="open-source-instruction-tuning-датасеты" aria-hidden="true"></span>
<span id="хронология" aria-hidden="true"></span>
<span id="key-papers" aria-hidden="true"></span>
<span id="related-concepts" aria-hidden="true"></span>
<span id="дополнительные-ресурсы" aria-hidden="true"></span>
