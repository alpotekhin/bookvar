---
title: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
type: source-note
status: reviewed
last_verified: 2026-09-21
authors: Zhihong Shao et al.
year: 2024
source_url: https://arxiv.org/abs/2402.03300v3
---

# DeepSeekMath и исходный GRPO

В разделе 4.1 вводится групповая оценка преимущества без отдельной модели
ценности. Статья рассматривает награду за результат и за промежуточные шаги;
GRPO поэтому не тождественен RLVR. Формула включает ограничение отношений
вероятностей токенов, усреднение по длине каждого ответа и штраф относительно
опорной модели.

В [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/07 GRPO и DeepSeek-R1|учебной главе]]
эта исходная постановка отделена от Dr. GRPO, GSPO и настроек современных
тренеров. Рисунок сравнения PPO/GRPO сохранён из ранее добавленного материала;
новая схема пошагового вычисления взята из авторской книги Nathan Lambert
с отдельной MIT-атрибуцией.

Первоисточник: [DeepSeekMath v3, §4.1](https://arxiv.org/html/2402.03300v3#S4.SS1).
