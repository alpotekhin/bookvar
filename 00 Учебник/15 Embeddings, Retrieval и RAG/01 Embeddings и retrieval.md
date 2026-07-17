---
title: Embeddings и retrieval
type: textbook-chapter
status: canonical
last_updated: 2026-07-16
---

# Embeddings и retrieval

> [!info] Карта углублённого модуля
> Полная траектория от BM25 до trainable retrievers, reranking и stage-wise
> evaluation: [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/00 Карта модуля и источники]].

> [!abstract] Идея главы
> Поиск начинается не с LLM. Сначала система должна быстро найти небольшой набор
> кандидатов среди миллионов документов, а затем более дорогая модель может
> внимательно сравнить каждый кандидат с запросом.

## Sparse retrieval

BM25 сопоставляет query terms с document terms, учитывает редкость слова и
нормализует длину документа. Он особенно силён для имён, кодов, артикулов и
точных формулировок.

## Dense retrieval

Bi-encoder отдельно кодирует query и documents:

$$
q=f_\theta(\text{query}),\qquad d=g_\phi(\text{document}).
$$

Similarity — dot product или cosine. Document vectors можно посчитать заранее и
искать approximate nearest neighbors.

Sparse retrieval (BM25) сопоставляет terms. Dense retrieval обучает vectors.
Hybrid объединяет lexical precision и semantic recall.

Bi-encoder кодирует query/document отдельно и масштабируется. Cross-encoder
читает пару совместно, точнее, но дорог. ColBERT хранит token-level vectors и
использует late interaction как промежуточный компромисс.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/colbert-figure2-hq.png]]

*ColBERT paper: representation-based bi-encoder, interaction-based
cross-encoder и late interaction между ними.*

Hard negatives критичны: без похожих, но нерелевантных документов retriever
учится слишком простой границе.

## Hybrid search

Dense retrieval ловит paraphrases, sparse — точные terms. Reciprocal rank fusion
или learned fusion объединяет результаты. Hybrid — не временный костыль: два
канала ошибаются по-разному.

## Оценка

- recall@k: попал ли нужный document в candidates;
- MRR/NDCG: насколько высоко он стоит;
- latency и index memory;
- performance по языкам и типам запросов;
- robustness к свежим и редким entities.

- [[02 Areas/ML & DL/Concepts/Retrieval/Dense Retrieval]]
- [[02 Areas/ML & DL/Concepts/Retrieval/ColBERT]]
- [[02 Areas/ML & DL/Papers/ColBERT]]
