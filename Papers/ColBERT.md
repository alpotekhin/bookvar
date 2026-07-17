---
title: "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT"
url: https://arxiv.org/abs/2004.12832
authors: [Omar Khattab, Matei Zaharia]
year: 2020
date_reviewed: 2026-04-07
type: source-note
status: legacy
category: paper
tags:
  - retrieval
  - IR
  - BERT
  - efficiency
Organization: Stanford
concepts:
  - "[[02 Areas/ML & DL/Concepts/Retrieval/Dense Retrieval|Dense Retrieval]]"
  - "[[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]]"
raw: "[[02 Areas/ML & DL/raw/papers/colbert/paper.txt]]"
---

# ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT

**Authors:** Omar Khattab, Matei Zaharia (Stanford)
**Published:** 2020 (SIGIR 2020, arXiv:2004.12832)
**URL:** https://arxiv.org/abs/2004.12832

## TL;DR

**ColBERT** (Contextualized Late Interaction over BERT) — ranking модель, которая сочетает эффективность bi-encoder подхода (предвычисление document embeddings offline) с выразительностью fine-grained token-level interaction. Query и document независимо кодируются через BERT в наборы token embeddings, а relevance score вычисляется через сумму MaxSim операций (для каждого query token — max cosine similarity с document tokens). Результат: **170x быстрее** BERT-base ranker, **14,000x меньше FLOPs**, при сопоставимом MRR@10 на MS MARCO.

## Problem

BERT-based ранкеры (cross-encoders) достигли SOTA на retrieval benchmarks, но:
- Требуют пропустить каждую пару query-document через BERT целиком
- 100-1000x дороже предшествующих моделей (KNRM, Duet)
- BERT-large: 32,900 ms на re-ranking top-1000 (MS MARCO) — неприемлемо для production
- Latency > 100ms уже влияет на UX и revenue

Существующие подходы к ускорению:
- **Representation-based** (DSSM, SNRM): query и document -> одно embedding -> dot product. Быстро, но низкое quality — теряют fine-grained matching
- **Interaction-based** (KNRM, Duet): моделируют word-level interactions, но не используют deep contextual representations
- **NLU-augmented** (doc2query, DeepCT): улучшают BM25 через NLU offline, но существенно уступают BERT по precision

## Method

### Late Interaction Architecture (Section 3.1)

Ключевая идея: задержать (delay) взаимодействие query-document, сохранив при этом token-level granularity.

```
S_{q,d} = sum_{i in |Eq|} max_{j in |Ed|} (Eq_i . Ed_j^T)
```

1. **Query encoder** fQ: q -> набор embeddings Eq (контекстуализированные через BERT)
2. **Document encoder** fD: d -> набор embeddings Ed (предвычисляется offline)
3. **Scoring**: для каждого query embedding — MaxSim с всеми document embeddings -> суммирование по query tokens

MaxSim — ключевой выбор:
- Дешевый: dot product + max + sum
- Pruning-friendly: позволяет использовать vector similarity indexes (FAISS) для end-to-end retrieval
- Альтернативы (average similarity) — менее amenable to pruning

### Query Encoder (Section 3.2)

- Prepend [Q] token после [CLS]
- **Query augmentation**: padding до фиксированной длины Nq=32 с [MASK] tokens. BERT генерирует embeddings для [MASK] позиций — soft mechanism для query expansion и term re-weighting
- Linear projection: BERT hidden dim -> m-dimensional embeddings (m=128)
- L2 normalization -> cosine similarity = dot product

### Document Encoder (Section 3.2)

- Prepend [D] token после [CLS]
- Без padding (в отличие от query)
- Фильтрация embeddings пунктуации
- Linear projection + L2 normalization
- **Offline indexing**: все document embeddings предвычисляются и хранятся (32-bit или 16-bit)

### Training (Section 3.3)

- Pairwise softmax cross-entropy loss на тройках <q, d+, d->
- Shared BERT model для query и document encoders (различаются через [Q]/[D] tokens)
- Fine-tuning с lr=3e-6, batch=32, 200k iterations на MS MARCO

### Re-ranking (Section 3.5)

1. Загрузить предвычисленные document embeddings в память
2. Для query q: вычислить Eq (один BERT forward pass)
3. Batch dot product Eq x D (3D tensor) -> cross-match matrices
4. Max-pool по document tokens, sum по query tokens -> scores
5. Sort

Cost: BERT вызывается только 1 раз (для query), а не k раз (для каждого document). Доминирующий cost — gathering и transfer embeddings CPU -> GPU.

### End-to-end Retrieval (Section 3.6)

Двухэтапная процедура с FAISS (IVFPQ index):
1. **Filtering**: для каждого из Nq query embeddings — поиск top-k' ближайших document embeddings через FAISS -> K уникальных документов
2. **Refinement**: exhaustive re-ranking K документов через стандартный ColBERT scoring

FAISS index: P=1000 partitions (k-means), s=16 sub-vectors (product quantization), 1 byte per sub-vector.

## Key Results

### MS MARCO Re-ranking (Table 1)

| Method | MRR@10 (Dev) | Latency (ms) | FLOPs/query |
|--------|-------------|---------------|-------------|
| BERTbase [Nogueira] | 34.7 | 10,700 | 97T (13,900x) |
| BERTlarge | 36.5 | 32,900 | 340T (48,600x) |
| **ColBERT (BERTbase)** | **34.9** | **61** | **7B (1x)** |
| fastText+ConvKNRM | 29.0 | 28 | 78B (11x) |

ColBERT: -1.6 MRR@10 vs BERTlarge, но **170x быстрее** и **14,000x меньше FLOPs** vs BERTbase.

### MS MARCO End-to-end Retrieval (Table 2)

| Method | MRR@10 (Dev) | Latency (ms) | Recall@1000 |
|--------|-------------|---------------|-------------|
| BM25 (Anserini) | 18.7 | 62 | 85.7 |
| docTTTTTquery | 27.7 | 87 | 94.7 |
| **ColBERT (end-to-end)** | **36.0** | **458** | **96.8** |
| ColBERT (re-rank BM25) | 34.8 | - | 81.4 |

End-to-end ColBERT превосходит re-ranking ColBERT по MRR@10 (+1.2) и recall (+15.4 R@1000) благодаря лучшему покрытию через vector similarity search.

### TREC CAR (Table 3)

ColBERT: MAP 31.3, сопоставимо с BM25 + BERTbase (31.0). BERTlarge: 33.5.

### Ablation (Figure 5)

- Late interaction > representation-based (BERT [CLS] dot-product): +10 MRR@10
- MaxSim > average similarity: значительное преимущество
- Query augmentation ([MASK] padding): существенный вклад в effectiveness
- 12-layer > 5-layer: +6 MRR@10

### Indexing (Section 4.5)

- MS MARCO (8.8M passages): ~3 часа на 4 GPU
- Space: десятки GiB при m=128, 32-bit values
- Embedding dimension m: робастен в диапазоне 32-256

## My notes

- ColBERT стал основой для целого направления: ColBERTv2 (2022), PLAID, ColBERT-XM (multilingual). Принцип late interaction оказался очень влиятельным.
- На практике ColBERT + PLAID используется в RAGatouille, Stanford DSP framework, и ряде production систем. Но overhead по памяти (хранение per-token embeddings для каждого документа) остается проблемой — ColBERTv2 решает это через residual compression.
- Query augmentation через [MASK] tokens — элегантная идея для learned query expansion. Модель сама решает, какие "виртуальные" термины добавить к запросу.
- MaxSim vs dot product на [CLS]: разница огромная. Fine-grained token-level matching принципиально важен для retrieval quality. Это подтверждает, что compression query/document до одного вектора теряет слишком много информации.
- Ограничение: storage cost. При m=128 и 32-bit, каждый token = 512 bytes. Документ в 200 токенов = 100KB. 10M документов = ~1TB. ColBERTv2 снижает это через quantization.
