---
title: "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
type: source-note
status: canonical
last_verified: 2026-09-21
source_url: https://arxiv.org/abs/2305.10601v2
year: 2023
---

# Tree of Thoughts: Deliberate Problem Solving with Large Language Models

ToT организует поиск по содержательным промежуточным шагам. В Game of 24
один шаг выбирает два числа и операцию; другой запрос оценивает достижимость
цели из нового состояния. Оценка языковой модели не доказывает допустимость
перехода. Нужна отдельная проверка правил и итогового выражения.

В [[02 Areas/ML & DL/00 Учебник/13 Reasoning и Test-time Compute/01 Test-time compute#поиск-по-промежуточным-состояниям|главе о поиске]]
разобраны последовательность операций, глобальный отбор пяти состояний
и эвристические баллы авторского кода. Проверены разделы 3–4 версии v2
и [реализация](https://github.com/princeton-nlp/tree-of-thought-llm/tree/8050e67d0e3a0fddc424d7fa5801538722a4c4cc).
Четвёртый шаг в коде формирует полное выражение после трёх операций.

Figures 1 и 2 извлечены целиком со страниц 2 и 5, без перерисовки.
Лицензия PDF — CC BY 4.0; лицензия кода — MIT.
Локальный PDF: `Source PDFs/tree-of-thoughts-v2.pdf`.

Контрольные суммы PDF и рисунков, разрешение и координаты извлечения
записаны в asset registry. Результаты моделей в этой редакции не воспроизводились.

