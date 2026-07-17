---
title: RAG как система
type: textbook-chapter
status: canonical
last_updated: 2026-07-16
---

# RAG как система

> [!info] Карта углублённого модуля
> Архитектуры, отдельные уроки, практики и первичные источники:
> [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/00 Карта модуля и источники]].

> [!abstract] Идея главы
> RAG не «подключает базу знаний» одной командой. Ответ проходит через ingestion,
> chunking, indexing, retrieval, reranking и сборку context. Ошибка любого этапа
> выглядит в конце как галлюцинация LLM.

RAG добавляет внешнюю память, но качество определяется всей цепочкой.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/retrieval-augmented-generation-for-knowledge-intensive-nlp-t/rag-fig1.png]]

*Оригинальный RAG соединяет parametric memory генератора и non-parametric
memory индекса. Современные системы добавляют parsing, hybrid retrieval,
reranking и citations.*

## Ingestion

PDF/HTML нужно корректно разобрать, сохранить headings, tables и metadata.
Chunk — не просто отрезок фиксированной длины: он должен быть достаточно мал для
точного поиска и достаточно полон для ответа.

## Основные failure modes

- нужный факт не попал в index;
- chunk разрезал смысл;
- retriever не нашёл документ;
- reranker удалил правильный;
- context переполнен шумом;
- модель проигнорировала evidence;
- citation не поддерживает claim.

Advanced RAG меняет отдельные этапы: query rewriting, multi-query, HyDE,
parent-child retrieval, contextual compression, adaptive retrieval, Self-RAG и
corrective loops. Нельзя оценивать только финальный answer: нужны retrieval
recall, context precision, groundedness и task success.

## Правильная диагностика

Для каждого ошибочного ответа спросите:

1. был ли факт в source;
2. попал ли он в parsed text;
3. сохранился ли в chunk;
4. вошёл ли chunk в top-k;
5. прошёл ли reranker;
6. был ли помещён в prompt;
7. поддерживает ли citation конкретный claim.

Только последний пункт относится непосредственно к генератору.

- [[02 Areas/ML & DL/Concepts/Retrieval/Retrieval-Augmented Generation]]
- [[02 Areas/ML & DL/Papers/Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks]]
- [[02 Areas/ML & DL/Papers/Self-RAG]]
- [[01 Projects/X5/Retrival exp]]
