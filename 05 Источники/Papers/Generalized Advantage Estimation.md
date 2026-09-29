---
title: High-Dimensional Continuous Control Using Generalized Advantage Estimation
type: source-note
status: canonical
last_verified: 2026-09-21
source_url: https://arxiv.org/abs/1506.02438v6
authors: [John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel]
year: 2015
---

# Generalized Advantage Estimation

Работа выводит оценку преимущества как взвешенную сумму временных разностей.
Коэффициент λ регулирует, насколько сильно оценка зависит от прогноза функции
ценности и насколько — от последующих наблюдаемых наград. При λ=1 и корректной
обработке конца эпизода получается возврат Монте-Карло за вычетом базового уровня.

В [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/04 Policy gradient и PPO для LLM|главе о PPO]]
вывод дополнен вычислением для трёх токенов и кодом с переменной длиной ответов.
Числовой пример учебный, не результат эксперимента статьи. Первичная работа
исследует непрерывное управление; применение к языковым моделям отдельно
сопоставляется с RLHF Book и Stanford CS336.

Ссылка на [версию v6](https://arxiv.org/abs/1506.02438v6) фиксирует проверенный
текст. Рисунки статьи в учебник не копировались.
