---
title: "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters"
type: source-note
status: canonical
last_verified: 2026-09-21
source_url: https://arxiv.org/abs/2408.03314v1
year: 2024
---

# Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters

Работа разделяет изменение распределения предложений и использование
проверяющей модели. Сопоставляются независимые ответы, последовательные
исправления, PRM-поиск и распределение бюджета по сложности задач.
Результаты получены в определённой конфигурации PaLM2-S* на MATH,
а не являются универсальным законом для всех моделей.

В [[02 Areas/ML & DL/00 Учебник/13 Reasoning и Test-time Compute/01 Test-time compute|главе о test-time compute]]
уточнены три важных условия: PRM оценивает вероятность успешного продолжения
данной моделью; цепочки исправлений используют специально дообученную модель;
стоимость 2048 генераций для оценки сложности исключена из основного сравнения.
Проверены разделы 4–7 и приложения C–E версии v1.

Figures 2, 3 и 5 извлечены со страниц 8, 9 и 11 с полными легендами.
Цвет выбора PRM не означает доказанную правильность. Столбики Figure 3
наложены друг на друга, а не складываются. Лицензия — CC BY 4.0.
Локальный PDF: `Source PDFs/snell-test-time-compute-v1.pdf`.

Контрольные суммы PDF и рисунков, разрешение и координаты извлечения
записаны в asset registry. Результаты моделей в этой редакции не воспроизводились.

