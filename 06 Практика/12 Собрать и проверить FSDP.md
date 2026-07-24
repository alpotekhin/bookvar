---
title: Собрать и проверить FSDP
type: practice
status: canonical
last_updated: 2026-07-24
---

# Собрать и проверить FSDP

Опорные материалы: [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week05_fsdp/lecture.pdf|EDLS Week 5 lecture]], [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week05_fsdp/seminar.pdf|seminar]] и [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week05_fsdp/homework/README|полное исходное задание]].

Запустите один и тот же model/optimizer/batch сначала в DDP, затем в FSDP.
Проверьте loss и один полный optimizer step. Для каждого режима снимите memory
snapshot по rank и timeline collectives.

В отчёте разложите память на parameters, gradients, optimizer states и
activations; отметьте, какие части sharded, replicated или materialized
временно. Затем сравните минимум две wrap policies и режимы prefetch. Объясните
пики памяти через all-gather lifetime, а gaps — через зависимости между
compute и communication.

Checkpoint должен пережить сохранение, новый запуск с другим числом ranks и
восстановление следующего шага. Если это не проверено, работа с FSDP не
закончена.
