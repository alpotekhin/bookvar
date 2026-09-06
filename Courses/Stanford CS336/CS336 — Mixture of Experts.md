---
title: "CS336 — Mixture of Experts"
type: source-note
status: legacy
source_only: true
course: "Stanford CS336"
last_updated: 2026-09-07
canonical_target: "[[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/02 Mixture of Experts — routing, capacity и serving|Mixture of Experts — routing, capacity и serving]]"
robots: noindex
search_exclude: true
---

# CS336 — Mixture of Experts

> [!note] Историческая карточка источника
> Эта страница больше не является отдельной учебной главой. Материал курса
> встроен в единый маршрут от dense FFN к router, expert parallelism и serving.

**Основная глава:** [[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/02 Mixture of Experts — routing, capacity и serving|Mixture of Experts — routing, capacity и serving]].

Глава выводит top-k routing из dense FFN, показывает арифметику активных и
полных параметров, различает статистическую и системную балансировку, разбирает
capacity и dropped assignments, два all-to-all при expert parallelism,
обучение router, upcycling, эволюцию DeepSeekMoE и ограничения inference.

**Предыдущая глава:** [[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/01 Dense FFN — token-wise вычисление, expansion и gating|Dense FFN]].

**Системная арифметика:** [[02 Areas/ML & DL/00 Учебник/10 ML Systems/04 Арифметика Transformer и MoE|Transformer и MoE: параметры, FLOPs и коммуникация]].

**Архив курса:** [[02 Areas/ML & DL/05 Источники/Courses/Stanford CS336 Spring 2026/_index|Stanford CS336 Spring 2026]].

Старая заметка сохранена только ради входящих ссылок. Её короткие оценки
Mixtral, DeepSeek и «трендов» не используются как учебное доказательство без
модельной конфигурации, топологии, нагрузки и первичного источника.
