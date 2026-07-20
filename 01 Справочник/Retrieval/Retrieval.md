---
title: Information Retrieval
aliases: [Retrieval, Dense Retrieval, Sparse Retrieval, Hybrid Retrieval]
type: concept
status: canonical
last_updated: 2026-07-16
primary_sources:
  - https://arxiv.org/abs/2004.04906
---

# Retrieval

**Retrieval** выбирает из корпуса кандидатов, релевантных запросу. Практический pipeline разделяет быстрый recall и дорогой precision.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-first-60-64/dpr-dual-encoder-and-loss.png]]

*DPR кодирует вопрос и passages раздельно, сравнивает их скалярным произведением
и обучается отличать positive passage от negatives. Источник: Vladimir Karpukhin
et al., [Dense Passage Retrieval for Open-Domain Question Answering, Figure
1](https://arxiv.org/abs/2004.04906).*

Это схема первого, recall-oriented этапа. После поиска по предварительно
вычисленным passage vectors небольшой top-k можно передать более дорогому
cross-encoder или генератору; применять совместную модель ко всему корпусу
слишком дорого.

## Подходы

| Метод | Сильная сторона | Слабая сторона |
|---|---|---|
| Sparse (BM25) | точные слова, редкие термины | слабее семантические перефразы |
| Dense bi-encoder | семантическое сходство, быстрый ANN | compression loss, domain shift |
| Late interaction (ColBERT) | token-level matching | индекс больше и поиск сложнее |
| Cross-encoder reranker | высокая точность пары query-document | нельзя дёшево применить ко всему корпусу |
| Hybrid | объединяет lexical и semantic | fusion и calibration требуют настройки |

ANN-индекс ускоряет nearest-neighbor search ценой approximation. Метрики retrieval оценивают, попал ли релевантный документ в кандидаты: Recall@k, MRR, nDCG. Offline relevance не заменяет end-to-end оценку [[02 Areas/ML & DL/01 Справочник/Retrieval/RAG|RAG]].

Chunking является частью index design: размер, overlap и границы определяют, что именно система способна вернуть. Bigger chunks дают контекст, smaller chunks — точность и больше кандидатов.

## Подробнее

BM25, dense retrieval, hybrid fusion и двухступенчатый поиск сопоставлены в
главе [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/61 Retrieval — от BM25 до dense и hybrid|Retrieval — от BM25 до dense и hybrid]].

## Источники

- [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906)
- [ColBERT](https://arxiv.org/abs/2004.12832)
