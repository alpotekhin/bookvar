---
title: "CS336 — Scaling Laws"
type: source-note
status: legacy
source_only: true
course: "Stanford CS336"
last_updated: 2026-09-07
canonical_target: "[[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/43 Scaling laws|Законы масштабирования]]"
robots: noindex
search_exclude: true
---

# CS336 — Scaling Laws

> [!note] Историческая карточка источника
> Эта страница больше не является отдельной учебной главой. Формулы, кампании
> малых запусков и выводы Stanford CS336 перенесены в каноническую главу.

**Основная глава:** [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/43 Scaling laws|Законы масштабирования]].

Там степенная аппроксимация сначала отделена от закона природы, после чего
последовательно разобраны определения параметров, токенов и FLOPs, три способа
поиска compute-optimal точки, дизайн scaling campaign, перенос learning rate и
batch size, подгонка с остатками и неопределённостью, WSD, μP, семейства
современных рецептов и различие между training-optimal и inference-optimal
решениями.

**Данные для кампании:** [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/01 Данные и pre-training|Данные и предобучение]].

**Практика:** [[02 Areas/ML & DL/06 Практика/22 Воспроизвести scaling campaign|воспроизводимая scaling campaign]].

**Архив курса и Assignment 3:** [[02 Areas/ML & DL/05 Источники/Courses/Stanford CS336 Spring 2026/_index|Stanford CS336 Spring 2026]].

Старая заметка сохранена как стабильный адрес. Упрощённые числа вроде
«20 токенов на параметр» и оценки конкретных бюджетов не переносятся как
универсальные правила: их смысл зависит от семейства моделей, данных,
оптимизатора, горизонта обучения и цены последующего inference.
