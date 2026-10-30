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

ANN-индекс ускоряет поиск ближайших соседей ценой приближённого результата.
Качество кандидатов и их порядка измеряют по-разному:

- **Recall@k** — доля всех известных релевантных документов, попавших в первые $k$.
- **MRR** — среднее по запросам от $1/r$, где $r$ — ранг первого релевантного результата; при отсутствии находки вклад нулевой. Если выдача обрезана на $k$, это MRR@k.
- **nDCG@k** — дисконтированная полезность первых $k$ результатов, делённая на полезность идеального порядка; допускает градации релевантности. Например, $DCG@k=\sum_{i=1}^k(2^{rel_i}-1)/\log_2(i+1)$, где $rel_i$ — оценка релевантности на позиции $i$. Соглашения о gain и запросах без релевантных документов фиксируют в протоколе.

Если релевантны два документа, а в первых трёх найден один на позиции 2,
Recall@3 равен $1/2$, reciprocal rank — $1/2$. Перестановка этой находки
на позицию 1 не изменит Recall, но повысит reciprocal rank до $1$ и nDCG.
В задачах с одним достаточным ответом отдельно используют hit rate@k —
долю запросов хотя бы с одной находкой, не общий Recall@k.
См. [Manning, Raghavan, Schütze: оценка ранжированной выдачи](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html).
Метрики поиска не заменяют оценку полного [[02 Areas/ML & DL/01 Справочник/Retrieval/RAG|RAG]].

Разбиение на фрагменты — часть проектирования индекса: размер, перекрытие и границы определяют, что система способна вернуть. Крупный фрагмент сохраняет окружение, но может смешать темы и расходовать контекстный бюджет. Мелкий точнее локализует факт, однако может отделить его от условия или исключения. Например, фраза «срок — 30 дней» без следующего абзаца «для архивных данных — 90» может привести к неверному ответу. Уменьшение фрагмента не гарантирует большей точности; размер проверяют вместе с поиском и сборкой контекста.

## Подробнее

BM25, dense retrieval, hybrid fusion и двухступенчатый поиск сопоставлены в
главе [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/61 Retrieval — от BM25 до dense и hybrid|Retrieval — от BM25 до dense и hybrid]].

## Источники

- [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906)
- [ColBERT](https://arxiv.org/abs/2004.12832)
