---
title: Построить roofline и найти bottleneck
type: practice
status: canonical
last_updated: 2026-07-24
---

# Построить roofline и найти bottleneck

В этой работе измерение превращается в объяснение: ограничена ли операция
памятью, вычислениями или накладными расходами запуска.

## Читать и выполнять

- [[02 Areas/ML & DL/05 Источники/Courses/Harvard ML Systems/vol2/performance_engineering|Harvard CS249r — Performance Engineering]];
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week01_intro/seminar.ipynb|EDLS Week 1 — исходный notebook]].

Выберите vector add, reduction и GEMM. Для каждого размера оцените полезные FLOP
и bytes, которые должны пройти через HBM:

$$I=\frac{F}{Q_{\mathrm{HBM}}},\qquad
P_{\mathrm{roof}}=\min(P_{\mathrm{peak}}, BW_{\mathrm{HBM}}I).
$$

Измерьте sustained bandwidth отдельным memory-bound kernel и sustained GEMM
throughput на тех же dtype и устройстве: рекламные peak values оставьте на
графике только как вторую, явно подписанную границу. Нанесите реальные точки на
roofline и посчитайте эффективность относительно применимого потолка.

## Проверка диагноза

Измените одну причину за раз:

- объедините две elementwise операции;
- увеличьте reuse через tiling или используйте готовый fused kernel;
- для маленькой операции увеличьте batch либо число независимых запусков.

Если классификация верна, memory-bound вариант должен реагировать прежде всего
на bytes/reuse, compute-bound — на эффективный compute, launch-bound — на число
запусков. Сдайте notebook, график, таблицу assumptions и profiler trace до/после.
