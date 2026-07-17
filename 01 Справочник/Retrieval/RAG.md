---
title: Retrieval-Augmented Generation
aliases: [RAG]
type: concept
status: canonical
last_updated: 2026-07-16
primary_sources:
  - https://arxiv.org/abs/2005.11401
---

# Retrieval-Augmented Generation (RAG)

**RAG** подаёт генератору внешние документы, найденные по запросу. Знание находится не только в weights модели, но и в обновляемом корпусе.

```mermaid
flowchart LR
  Q["вопрос"] --> RET["retriever"]
  IDX["корпус / индекс"] --> RET
  RET --> DOCS["evidence"]
  Q --> GEN["generator"]
  DOCS --> GEN
  GEN --> ANS["ответ + ссылки"]
```

## Полный цикл

1. ingest и очистка документов;
2. parsing, chunking, metadata;
3. indexing;
4. query transformation при необходимости;
5. retrieval и reranking;
6. context assembly;
7. generation;
8. citations, logging и evaluation.

RAG не гарантирует factuality: retriever может не найти нужное, контекст может конфликтовать, generator — игнорировать evidence или приписывать источнику лишнее.

## Где искать причину ошибки

```text
нет evidence в top-k → retrieval/index problem
evidence есть, но не прошло rerank/context → orchestration problem
evidence в prompt, ответ неверен → generation/grounding problem
ответ верен, citation неверна → attribution problem
```

Это важнее одной агрегированной «RAG accuracy». Оценивайте retrieval recall, relevance, context precision, answer correctness, faithfulness/entailment, citation correctness, latency и cost.

RAG — системный паттерн, а не конкретная vector database или framework. Он может использовать sparse, dense или hybrid [[02 Areas/ML & DL/01 Справочник/Retrieval/Retrieval|retrieval]], графы, SQL и tools.

## Источники

- [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
- [[02 Areas/ML & DL/Papers/Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks|Paper note: RAG]]
- [[02 Areas/ML & DL/Concepts/Retrieval/Retrieval-Augmented Generation|Legacy: RAG]]
