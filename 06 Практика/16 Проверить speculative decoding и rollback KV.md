---
title: Проверить speculative decoding и rollback KV
type: practice
status: canonical
last_updated: 2026-07-24
---

# Проверить speculative decoding и rollback KV

Полное условие и кодовые заготовки находятся в [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week09_inference_algorithms/homework/homework.ipynb|EDLS Week 9 homework notebook]], краткое описание — в [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week09_inference_algorithms/homework/README|README задания]].

Реализуйте draft generation, одну проверку target model и принятие/отклонение
tokens так, чтобы итоговое распределение соответствовало target decoding.
Сначала тестируйте детерминированный greedy случай, затем sampling на маленьком
словаре, где распределения можно посчитать напрямую.

KV-cache должен коммитить accepted prefix и удалять rejected suffix. Добавьте
инварианты длины cache после каждого шага и тест с частичным acceptance.

Измерьте acceptance length, число target forward passes, tokens/s, TPOT и peak
KV memory при разных $k$, draft models и типах prompts. Ускорение объясняйте
через saved target invocations с учётом стоимости draft и verification; низкая
acceptance rate должна проявиться как отсутствие выигрыша.
