---
title: Causal Language Modeling
aliases:
  - CLM
  - autoregressive language modeling
  - autoregressive LM
  - авторегрессивное языковое моделирование
type: concept
status: redirect
category: Training
papers:
  - '[[02 Areas/ML & DL/Papers/GPT 2.0]]'
  - '[[02 Areas/ML & DL/Papers/GPT 3.0]]'
  - '[[02 Areas/ML & DL/Papers/LLaMA]]'
  - '[[02 Areas/ML & DL/Papers/OPT]]'
  - '[[02 Areas/ML & DL/Papers/Formal Algorithms for Transformers]]'
courses:
  - >-
    [[02 Areas/ML & DL/Courses/SHAD LLM/Week 1 -- Intro to LLMs|SHAD LLM -- Week
    1]]
sources:
  - >-
    [Radford et al. -- Language Models are Unsupervised Multitask Learners
    (GPT-2,
    2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
  - >-
    [Brown et al. -- Language Models are Few-Shot Learners (GPT-3,
    2020)](https://arxiv.org/abs/2005.14165)
  - >-
    [Jay Alammar -- The Illustrated
    GPT-2](https://jalammar.github.io/illustrated-gpt2/)
last_updated: '2026-10-29'
canonical_topic: 00 Учебник/11 Pre-training и Scaling/42 Next-token prediction.md
---
# Causal Language Modeling

Полное объяснение: [[00 Учебник/11 Pre-training и Scaling/42 Next-token prediction|Предсказание следующего токена]].

<!-- Compatibility anchors for existing Obsidian and web links. -->
<span id="causal-language-modeling-clm" aria-hidden="true"></span>
<span id="зачем-это-нужно-простейший-и-мощнейший-objective" aria-hidden="true"></span>
<span id="как-работает-clm" aria-hidden="true"></span>
<span id="causal-mask-запрет-подглядывания-в-будущее" aria-hidden="true"></span>
<span id="teacher-forcing-параллельное-обучение" aria-hidden="true"></span>
<span id="авторегрессивная-генерация-inference" aria-hidden="true"></span>
<span id="архитектура-decoder-only-transformer" aria-hidden="true"></span>
<span id="gpt-2-zero-shot-generalization" aria-hidden="true"></span>
<span id="gpt-3-in-context-learning" aria-hidden="true"></span>
<span id="clm-vs-mlm-когда-что-лучше" aria-hidden="true"></span>
<span id="clm-выигрывает" aria-hidden="true"></span>
<span id="mlm-выигрывает" aria-hidden="true"></span>
<span id="из-harnessing-llms-survey" aria-hidden="true"></span>
<span id="данные-и-токенизация" aria-hidden="true"></span>
<span id="токенизаторы-clm-моделей" aria-hidden="true"></span>
<span id="хронология" aria-hidden="true"></span>
<span id="key-papers" aria-hidden="true"></span>
<span id="related-concepts" aria-hidden="true"></span>
<span id="дополнительные-ресурсы" aria-hidden="true"></span>
