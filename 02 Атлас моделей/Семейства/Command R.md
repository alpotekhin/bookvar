---
title: "Command"
type: model-family
organization: Cohere
first_release: 2024
latest_verified_release: Command A+
last_verified: 2026-09-15
architecture_base: decoder-only Transformer then MoE
modalities: [text, image]
status: active
---

# Command: от RAG-ориентированного Command R к Command A+

Буква R в Command R обозначала не новую attention-формулу, а post-training вокруг retrieval-augmented generation: модель училась читать переданные документы, отвечать с цитатами и вызывать инструменты. В 2025–2026 Cohere продолжила линию как Command A/A+, поэтому каноническая страница должна показывать и архитектурный, и продуктовый diff.

## Релизы

| Релиз | Ключевые свойства |
|---|---|
| Command R / R+ (2024) | 35B/104B, 128K; multilingual RAG, citations, tool use. |
| Command R7B (2024) | 7B, 128K; компактная открытая модель той же прикладной линии. |
| Command A (13.03.2025) | 111B dense, 256K, 255K vocabulary; новая опубликованная архитектура. |
| Command A Reasoning (21.08.2025) | reasoning post-training и контролируемый enterprise режим. |
| Command A+ (20.05.2026) | Apache-2.0 MoE: 218B total / 25B active по текущей документации; текст и изображения на входе, текст на выходе, 48 языков. |

## Архитектура Command A

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/atlas-remainder-official/command-a-architecture.png]]

*Рисунок: Cohere, [Command A Technical Report](https://cohere.com/research/papers/command-a-technical-report.pdf), Figure 2; локальная копия. Три sliding-window слоя с RoPE чередуются с одним full-attention слоем без positional embeddings. Оба используют GQA; MLP — SwiGLU, input/output embeddings разделяют веса.*

Комбинация 3:1 ограничивает большую часть attention локальным окном, а периодический full attention переносит глобальную информацию. NoPE в full layers устраняет жёсткое позиционное преобразование там, где нужен глобальный доступ; RoPE сохраняется в local layers. Parallel transformer block вычисляет attention и MLP от общего входа, а большой 255K multilingual vocabulary увеличивает embedding matrix, поэтому weight tying существенно экономит память.

Для Command A+ [официальная документация](https://docs.cohere.com/docs/command-a-plus) указывает разреженную MoE, 218B суммарных и 25B активных параметров, окно 128 000 токенов и максимум 64 000 выходных токенов. Вход — текст и изображения, выход — текст: поддержка изображений не означает их генерацию. В [анонсе от 20 мая](https://cohere.com/blog/cohere-releases-command-a-plus) активный размер указан иначе, 24B; здесь используется актуальная спецификация документации, расхождение источников не скрывается. Свойства dense Command A — в частности окно 256K и схема local/full 3:1 — нельзя автоматически переносить на A+.

Изменение MoE разобрано в [[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/02 Mixture of Experts — routing, capacity и serving|главе о маршрутизации экспертов]], а интерфейс изображения — в [[02 Areas/ML & DL/02 Атлас моделей/Семейства/Мультимодальные семейства|карте мультимодальных семейств]]. Например, при вопросе о графике на слайде зрительный вход должен быть превращён в признаки до текстового ответа; список поддерживаемых файлов сам по себе не раскрывает конкретный encoder и соединитель A+. Эта карточка проверяет перечисленные поколения, а не гарантирует исчерпывающий охват всех новых релизов.

## Данные и post-training

Command A обучен на multilingual web, code, synthetic и специализированных качественных данных. Report описывает deduplication, ML quality filters и pruning смеси с помощью малых моделей, но не полный corpus. После pre-training идут несколько стадий SFT, preference optimization, model merging и safety alignment.

RAG-поведение создаётся специальным форматом: приложению передаются документы с идентификаторами, а модель должна связать утверждения с источниками. Tool use аналогично генерирует структурированный вызов, получает результат и формирует ответ. Это способности policy; base decoder сам по себе не гарантирует корректные citations.

## Serving

GQA и sliding-window слои уменьшают KV-cache; редкие full-attention слои сохраняют глобальную цену. Cohere сообщает, что Command A работает на двух A100/H100 и даёт более высокий throughput, чем R+ 08-2024, но сравнение зависит от стека. A+ как MoE потребует разместить все эксперты, даже если активна часть. Документация привязывает запуск на одном B200 или двух H100 к W4A4, а не BF16. Одна лишь масса 218B весов заняла бы примерно 436 GB в BF16 против идеальных 109 GB при 4 битах на вес; это нижняя оценка без шкал квантизации, KV-cache и рабочих буферов, а не измеренный расход памяти.

## Опубликовано и неизвестно

Для Command R доступны model cards и веса, но полная pre-training recipe ограничена. Command A имеет подробный technical report. Command A+ официально открыт под Apache 2.0 20 мая 2026 года; проверенные здесь спецификации не раскрывают точную схему зрительного кодировщика и соединителя или полный корпус, поэтому эти детали не восстанавливаются по свойствам Command A.

## Источники

- [Command R model card](https://huggingface.co/CohereForAI/c4ai-command-r-v01) и [R+](https://huggingface.co/CohereForAI/c4ai-command-r-plus).
- [Command A Technical Report](https://cohere.com/research/papers/command-a-technical-report.pdf).
- [Command A release](https://docs.cohere.com/changelog/command-a).
- [Command A+ documentation](https://docs.cohere.com/docs/command-a-plus) — параметры, направления модальностей и W4A4-условие развёртывания.
- [Command A+ official release](https://cohere.com/blog/cohere-releases-command-a-plus).
