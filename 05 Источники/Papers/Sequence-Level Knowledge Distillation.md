---
title: Sequence-Level Knowledge Distillation
type: source-note
status: canonical
last_verified: 2026-09-21
source_url: https://aclanthology.org/D16-1139/
authors: [Yoon Kim, Alexander M. Rush]
year: 2016
---

# Sequence-Level Knowledge Distillation

Работа EMNLP 2016 переносит дистилляцию на распределение целых выходных
последовательностей в машинном переводе. Практическое приближение использует
ответы учителя с высокой вероятностью, полученные поиском по лучу; ученик
максимизирует их правдоподобие. Это исторический предшественник SFT на
синтетических ответах современных языковых моделей, но не тот же эксперимент
и не источник результатов DeepSeek-R1.

В [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/08 Reasoning distillation|главе о дистилляции]]
работа подтверждает различие между передачей целого текста и мягких вероятностей
в каждой позиции. [Оригинал и библиография](https://aclanthology.org/D16-1139/).
Рисунки статьи в учебник не переносились.
