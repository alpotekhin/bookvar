---
title: "Self-Discover: Large Language Models Self-Compose Reasoning Structures"
type: source-note
status: canonical
last_verified: 2026-09-21
source_url: https://arxiv.org/abs/2402.03620v1
year: 2024
---

# Self-Discover: Large Language Models Self-Compose Reasoning Structures

Self-Discover сначала выбирает текстовые приёмы решения, адаптирует их
к классу задач и оформляет структуру ответа. Затем эта структура используется
для отдельных примеров. SELECT, ADAPT и IMPLEMENT не являются обучаемыми
блоками Transformer: это вызовы модели, создающие текстовую процедуру.

В [[02 Areas/ML & DL/00 Учебник/13 Reasoning и Test-time Compute/01 Test-time compute#декомпозиция-и-выбор-reasoning-модулей|главе о test-time compute]]
различаются общий подготовительный этап и решение отдельной задачи.
Проверены раздел 2 и Figure 2 версии v1: структура формируется по примерам
без ответов, но исходные модули и образец её формата заданы авторами.
В опыте MATH из §3.1 структуры создаются отдельно для каждого примера
с one-shot демонстрацией; этот вариант имеет другую подготовительную стоимость.

Из страницы 3 извлечена полная Figure 2 с обоими этапами.
Лицензия — CC BY 4.0. Локальный PDF: `Source PDFs/self-discover-v1.pdf`.

Контрольные суммы PDF и рисунков, разрешение и координаты извлечения
записаны в asset registry. Результаты моделей в этой редакции не воспроизводились.
