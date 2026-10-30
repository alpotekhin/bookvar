---
title: Sampling
aliases:
  - decoding strategies
  - temperature sampling
  - nucleus sampling
  - top-p sampling
  - top-k sampling
  - greedy decoding
  - beam search
type: concept
status: redirect
category: Inference
papers:
  - '[[02 Areas/ML & DL/Papers/Speculative Decoding]]'
courses:
  - >-
    [[02 Areas/ML & DL/Courses/SHAD LLM/Week 1 — Intro to LLMs|SHAD LLM — Week
    1]]
sources:
  - >-
    [Holtzman et al. — The Curious Case of Neural Text Degeneration
    (2020)](https://arxiv.org/abs/1904.09751)
  - >-
    [Hugging Face — How to generate text
    (2020)](https://huggingface.co/blog/how-to-generate)
last_updated: '2026-10-29'
canonical_topic: >-
  00 Учебник/14 Inference и оптимизация/54 Декодирование и выбор следующего
  токена.md
---
# Sampling

Полное объяснение: [[00 Учебник/14 Inference и оптимизация/54 Декодирование и выбор следующего токена|Декодирование и выбор следующего токена]].

<!-- Compatibility anchors for existing Obsidian and web links. -->
<span id="sampling-decoding-strategies" aria-hidden="true"></span>
<span id="зачем-это-нужно-проблема-выбора-следующего-токена" aria-hidden="true"></span>
<span id="greedy-decoding-самый-простой-самый-плохой" aria-hidden="true"></span>
<span id="temperature-sampling-регулятор-творчества" aria-hidden="true"></span>
<span id="top-k-sampling-отсечение-хвоста" aria-hidden="true"></span>
<span id="nucleus-sampling-top-p-адаптивный-top-k" aria-hidden="true"></span>
<span id="beam-search-несколько-гипотез-параллельно" aria-hidden="true"></span>
<span id="сравнительная-таблица" aria-hidden="true"></span>
<span id="практические-рекомендации" aria-hidden="true"></span>
<span id="комбинирование-стратегий" aria-hidden="true"></span>
<span id="speculative-decoding-ускорение-без-изменения-стратегии" aria-hidden="true"></span>
<span id="self-consistency-sampling-для-reasoning" aria-hidden="true"></span>
<span id="key-papers" aria-hidden="true"></span>
<span id="related-concepts" aria-hidden="true"></span>
<span id="дополнительные-ресурсы" aria-hidden="true"></span>
