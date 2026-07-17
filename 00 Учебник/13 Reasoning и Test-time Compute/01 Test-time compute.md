---
title: Test-time compute
type: textbook-chapter
status: canonical
last_updated: 2026-07-16
---

# Reasoning и test-time compute

> [!abstract] Идея главы
> После обучения можно потратить больше вычислений на один вопрос: дать модели
> больше времени, получить несколько решений, запустить поиск или проверить
> ответ. Качество растёт только тогда, когда дополнительный compute создаёт
> разнообразные кандидаты и есть способ отличить хороший.

Test-time compute расходует дополнительные вычисления после training:

- длинная chain of thought;
- несколько независимых samples;
- self-consistency;
- search по дереву решений;
- verifier/reranker;
- tool execution и feedback.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/tot-figure1-hq.png]]

*Tree of Thoughts сравнивает обычную последовательную chain of thought с поиском
по нескольким промежуточным состояниям.*

Больше tokens не гарантирует лучший ответ: модель может повторяться или
рационализировать ошибку. Эффективность зависит от diversity, verifier и
обучения модели пользоваться дополнительным budget.

Thinking mode — продуктовое управление generation policy и post-training, а не
обязательно отдельный neural block.

## Основные режимы

- **длиннее одна траектория** — полезно, если модель умеет проверять шаги;
- **best-of-N** — несколько независимых samples и verifier;
- **self-consistency** — majority по финальному ответу;
- **search** — ветвление и отбрасывание промежуточных states;
- **tools** — вычисления и внешние observations.

Без verifier больше tokens может лишь сделать ошибку длиннее и увереннее.

- [[02 Areas/ML & DL/Concepts/Reasoning/Test-time Compute]]
- [[02 Areas/ML & DL/Concepts/Inference/Chain of Thought]]
- [[02 Areas/ML & DL/Papers/Self-Consistency]]
- [[02 Areas/ML & DL/Papers/Tree of Thoughts]]
