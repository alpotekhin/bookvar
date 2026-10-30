---
title: MMLU
aliases:
  - MMLU
  - Massive Multitask Language Understanding
  - Massive Multitask Test
type: concept
status: redirect
category: Evaluation
papers:
  - '[[02 Areas/ML & DL/Papers/MMLU|MMLU]]'
courses:
  - >-
    [[02 Areas/ML & DL/Courses/Stanford CS224N/Lecture 11 — Benchmarking and
    Evaluation|CS224N Lecture 11]]
sources:
  - >-
    [Hendrycks et al. (2021) — Measuring Massive Multitask Language
    Understanding](https://arxiv.org/abs/2009.03300)
  - '[MMLU GitHub — hendrycks/test](https://github.com/hendrycks/test)'
  - '[MMLU Wikipedia](https://en.wikipedia.org/wiki/MMLU)'
  - '[Klu — MMLU Benchmark](https://klu.ai/glossary/mmlu-eval)'
last_updated: '2026-10-29'
canonical_topic: 00 Учебник/18 Evaluation и методология/59 Оценивание моделей и контаминация.md
canonical_heading: 'MMLU: предметные знания в формате выбора ответа'
---
# MMLU

Полное объяснение: [[00 Учебник/18 Evaluation и методология/59 Оценивание моделей и контаминация#MMLU: предметные знания в формате выбора ответа|Оценивание моделей и контаминация]].

<!-- Compatibility anchors for existing Obsidian and web links. -->
<span id="mmlu-massive-multitask-language-understanding" aria-hidden="true"></span>
<span id="зачем-это-нужно-разрыв-между-бенчмарками-и-реальными-способностями" aria-hidden="true"></span>
<span id="дизайн-бенчмарка" aria-hidden="true"></span>
<span id="57-задач-4-категории" aria-hidden="true"></span>
<span id="уровни-сложности" aria-hidden="true"></span>
<span id="формат-multiple-choice-4-варианта" aria-hidden="true"></span>
<span id="статистика" aria-hidden="true"></span>
<span id="оценка" aria-hidden="true"></span>
<span id="human-baselines" aria-hidden="true"></span>
<span id="прогрессия-модельных-результатов" aria-hidden="true"></span>
<span id="ключевые-наблюдения-из-оригинальной-статьи" aria-hidden="true"></span>
<span id="критика-mmlu" aria-hidden="true"></span>
<span id="формат-multiple-choice" aria-hidden="true"></span>
<span id="contamination" aria-hidden="true"></span>
<span id="потолок-достигнут" aria-hidden="true"></span>
<span id="что-mmlu-не-измеряет" aria-hidden="true"></span>
<span id="почему-mmlu-остаётся-важным" aria-hidden="true"></span>
<span id="related-concepts" aria-hidden="true"></span>
<span id="дополнительные-ресурсы" aria-hidden="true"></span>
