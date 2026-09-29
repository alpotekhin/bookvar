---
title: Learning to Plan & Reason for Evaluation with Thinking-LLM-as-a-Judge
type: source-note
status: canonical
last_verified: 2026-09-21
source_url: https://proceedings.mlr.press/v267/saha25b.html
authors: [Swarnadeep Saha et al.]
year: 2025
license: CC BY 4.0
---

# EvalPlanner

Оценщик сначала составляет план проверки по запросу без кандидатов, затем
выполняет его на паре ответов и формирует вердикт. Синтетические последовательности
с правильным и ошибочным вердиктами дают данные для итеративного обучения.

В [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/03 Reward modeling#планирование-проверки-перед-вердиктом|главе о модели награды]]
сохранено различие между обучением судьи и обучением модели, написавшей ответы.
Использована настоящая Figure 2 EvalPlanner, а не схема Self-Taught Evaluator,
которой ранее ошибочно иллюстрировался этот раздел.

PDF финальной статьи ICML: `Source PDFs/evalplanner-icml2025.pdf`.
Физическая страница 4, извлечение при 240 DPI без перерисовки; CC BY 4.0.
Контрольные суммы и координаты указаны в `05 Источники/asset-registry.yml`.
