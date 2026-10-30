---
title: HumanEval
aliases:
  - HumanEval
  - pass@k
  - Codex
  - HumanEval+
  - Code Generation Benchmark
type: concept
status: redirect
category: Evaluation
papers:
  - '[[02 Areas/ML & DL/Papers/HumanEval|HumanEval]]'
courses: []
sources:
  - >-
    [Chen et al. (2021) — Evaluating Large Language Models Trained on
    Code](https://arxiv.org/abs/2107.03374)
  - '[GitHub — openai/human-eval](https://github.com/openai/human-eval)'
  - >-
    [Deepgram — HumanEval LLM
    Benchmark](https://deepgram.com/learn/humaneval-llm-benchmark)
  - >-
    [Papers With Code — HumanEval
    SOTA](https://paperswithcode.com/sota/code-generation-on-humaneval)
  - '[IBM — What is HumanEval?](https://www.ibm.com/think/topics/humaneval)'
last_updated: '2026-10-29'
canonical_topic: 00 Учебник/18 Evaluation и методология/59 Оценивание моделей и контаминация.md
canonical_heading: 'HumanEval: проверять программу исполнением'
---
# HumanEval

Полное объяснение: [[00 Учебник/18 Evaluation и методология/59 Оценивание моделей и контаминация#HumanEval: проверять программу исполнением|Оценивание моделей и контаминация]].

<!-- Compatibility anchors for existing Obsidian and web links. -->
<span id="humaneval-evaluating-large-language-models-trained-on-code" aria-hidden="true"></span>
<span id="зачем-это-нужно-bleu-для-кода-не-работает" aria-hidden="true"></span>
<span id="бенчмарк-164-рукописных-задачи" aria-hidden="true"></span>
<span id="дизайн" aria-hidden="true"></span>
<span id="метрика-passk-ключевой-вклад" aria-hidden="true"></span>
<span id="идея" aria-hidden="true"></span>
<span id="unbiased-estimator" aria-hidden="true"></span>
<span id="оптимальная-температура-зависит-от-k" aria-hidden="true"></span>
<span id="codex-модель-для-генерации-кода" aria-hidden="true"></span>
<span id="архитектура-и-обучение" aria-hidden="true"></span>
<span id="результаты" aria-hidden="true"></span>
<span id="sample-selection-без-oracle" aria-hidden="true"></span>
<span id="ограничения-codex-и-humaneval" aria-hidden="true"></span>
<span id="scaling-complexity" aria-hidden="true"></span>
<span id="prompt-sensitivity" aria-hidden="true"></span>
<span id="164-задачи----мало" aria-hidden="true"></span>
<span id="эволюция-от-humaneval-к-современным-бенчмаркам" aria-hidden="true"></span>
<span id="humaneval-liu-et-al-2023" aria-hidden="true"></span>
<span id="прогрессия-на-humaneval-pass1" aria-hidden="true"></span>
<span id="практическое-значение-passk" aria-hidden="true"></span>
<span id="related-concepts" aria-hidden="true"></span>
<span id="дополнительные-ресурсы" aria-hidden="true"></span>
