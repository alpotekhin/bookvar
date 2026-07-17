---
title: "ColBERT"
aliases: [ColBERT, ColBERTv2, Late Interaction, Contextualized Late Interaction]
type: concept
category: Retrieval
papers:
  - "[[02 Areas/ML & DL/Papers/ColBERT|ColBERT (Khattab & Zaharia, 2020)]]"
sources:
  - "[Khattab & Zaharia — ColBERT (SIGIR 2020)](https://arxiv.org/abs/2004.12832)"
  - "[Weaviate — Late Interaction Overview](https://weaviate.io/blog/late-interaction-overview)"
  - "[Santhanam et al. — ColBERTv2 (2022)](https://arxiv.org/abs/2112.01488)"
courses: []
---

# ColBERT: Late Interaction over BERT

## Зачем это нужно: три парадигмы neural ranking

Задача neural Information Retrieval: по запросу $q$ найти наиболее релевантный документ $d$ из коллекции. К 2020 году сложились три подхода, каждый с критическим недостатком:

### 1. Bi-encoder (Representation-based)

```
Query  →  BERT  →  [CLS]  →  q⃗  (один вектор)
                                    ↓  dot product
Document →  BERT  →  [CLS]  →  d⃗  (один вектор)
```

Запрос и документ **независимо** кодируются в один вектор. Relevance = dot product или cosine similarity.

**Плюс:** документы кодируются offline → поиск за milliseconds через FAISS/HNSW.
**Минус:** **сжатие всего документа в один вектор** теряет token-level информацию. «Was the Eiffel Tower built in Paris?» и «Was the Eiffel Tower built in London?» могут получить почти одинаковые эмбеддинги.

### 2. Cross-encoder (All-to-all interaction)

```
[CLS] query tokens [SEP] document tokens [SEP]  →  BERT  →  relevance score
```

Запрос и документ **конкатенируются** и подаются в BERT. Полное взаимодействие между всеми токенами через self-attention.

**Плюс:** максимальная expressiveness — BERT видит fine-grained interactions.
**Минус:** нужно прогнать BERT **для каждой пары** (query, document). Для 1000 документов — 1000 прогонов BERT. На MS MARCO: **10,700 ms** на запрос (BERT-base). Невозможно использовать для retrieval из миллионов документов.

### 3. Late Interaction (ColBERT) — лучшее из двух миров

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/colbert/colbert-framework.png]]
*Архитектура ColBERT: query и document независимо кодируются, relevance вычисляется через MaxSim (источник: Stanford).*

```
Query    →  BERT  →  [q₁, q₂, ..., qN]     (N token embeddings)
                                                    ↓  MaxSim + sum
Document →  BERT  →  [d₁, d₂, ..., dM]     (M token embeddings, offline)
```

Запрос и документ кодируются **независимо** (как bi-encoder), но relevance вычисляется через **token-level interactions** (как cross-encoder). BERT вызывается для документов **offline** (один раз при индексации), для запроса — **один раз** при поиске.

## Архитектура ColBERT: пошаговый разбор

### Query Encoder $f_Q$

1. Добавляем специальный токен `[Q]` в начало запроса
2. Пропускаем через BERT
3. **Query augmentation**: если длина запроса < $N_q = 32$, дополняем токенами `[MASK]`
4. Linear projection: $768 \rightarrow m = 128$ dimensions
5. L2-нормализация каждого токена

$$\mathbf{E}_q = \text{Normalize}\left(\text{CNN}_{m}\left(\text{BERT}([Q] \; q_1 \; q_2 \; \ldots \; q_n \; \underbrace{[\text{MASK}] \ldots [\text{MASK}]}_{N_q - n - 1})\right)\right)$$

**Query augmentation с [MASK]** — ключевая инновация. [MASK] токены проходят через BERT self-attention и «узнают» о контексте запроса. BERT может «додумать» релевантные концепты, которых нет в запросе явно. Это **мягкое расширение запроса** (soft query expansion): если запрос «capital of France», [MASK] токены могут «активировать» semantics связанные с «Paris», «city», «Europe».

### Document Encoder $f_D$

1. Добавляем специальный токен `[D]` в начало документа
2. Пропускаем через BERT
3. **Фильтрация пунктуации** (точки, запятые, скобки — не несут semantic signal)
4. Linear projection: $768 \rightarrow 128$
5. L2-нормализация

**Критически важно:** document embeddings предвычисляются **offline** и хранятся в индексе. BERT вызывается для каждого документа **один раз** при индексации, а не при каждом запросе.

### MaxSim: scoring function

Relevance score между запросом $q$ и документом $d$:

$$S_{q,d} = \sum_{i \in |E_q|} \max_{j \in |E_d|} \mathbf{E}_{q_i}^\top \cdot \mathbf{E}_{d_j}$$

Пошагово:

1. Для каждого query embedding $\mathbf{E}_{q_i}$ найти **максимально похожий** document embedding $\mathbf{E}_{d_j}$ (cosine similarity, благодаря L2-нормализации = dot product)
2. Это **MaxSim** — максимальное сходство для данного query token
3. **Суммировать** MaxSim по всем query tokens

```
Query: "When was the Eiffel Tower built?"
                ↓
           q₁="When"   q₂="was"   q₃="Eiffel"   q₄="Tower"   q₅="built"

Document: "The Eiffel Tower was constructed in 1889 in Paris."
               d₁       d₂      d₃      d₄            d₅      d₆    d₇

MaxSim для каждого query token:
  q₁="When"   → max similarity: d₅="1889"     (0.3)
  q₂="was"    → max similarity: d₃="was"      (0.9)
  q₃="Eiffel" → max similarity: d₁="Eiffel"   (0.95)
  q₄="Tower"  → max similarity: d₂="Tower"    (0.93)
  q₅="built"  → max similarity: d₄="constructed" (0.85)

Score = 0.3 + 0.9 + 0.95 + 0.93 + 0.85 = 3.93
```

MaxSim — это **мягкое** поточечное сопоставление. В отличие от dot product одного вектора (bi-encoder), ColBERT сохраняет granularity отдельных токенов. В отличие от cross-encoder, вычисление **дешёвое** (dot products вместо full BERT forward pass).

### Обучение

Pairwise softmax cross-entropy loss на тройках $\langle q, d^+, d^- \rangle$:

$$\mathcal{L} = -\log \frac{e^{S_{q,d^+}}}{e^{S_{q,d^+}} + e^{S_{q,d^-}}}$$

Query и document энкодеры разделяют один BERT (shared weights), но с разными special tokens ([Q] vs [D]).

## Два режима: re-ranking и end-to-end retrieval

### Re-ranking

Классический сценарий: BM25 (или другой дешёвый retriever) находит top-1000 кандидатов, ColBERT переранжирует их.

- BERT вызывается **один раз** для query
- Document embeddings уже предвычислены
- MaxSim — **batch dot products** через 3D tensor operations

**Результат:** 170x быстрее BERT-base ranker, 14,000x меньше FLOPs, при сопоставимом качестве.

### End-to-end retrieval (без BM25)

Двухэтапная процедура с FAISS:

1. **Candidate generation:** для каждого query embedding → top-$k'$ ближайших document embeddings через FAISS (IVFPQ index). Объединение: набор уникальных документов-кандидатов.
2. **Exhaustive re-ranking:** для каждого кандидата — полный MaxSim score. Выбор top-$k$.

End-to-end ColBERT ищет **напрямую** по коллекции 8.8M passages без BM25, достигая MRR@10 36.0 и Recall@1000 96.8%.

## Сравнение трёх парадигм

| Метрика | Bi-encoder | Cross-encoder (BERT) | **ColBERT** |
|---------|-----------|---------------------|------------|
| Query latency | **~10 ms** | 10,700 ms | **58 ms** (re-rank) |
| FLOPs/query | ~500M | ~340B | **24B** (14,000x < BERT) |
| Offline index | Да (1 vec/doc) | Нет | Да (N vecs/doc) |
| Token-level interaction | Нет | Да | **Да** |
| MRR@10 (MS MARCO) | ~33% (DPR) | 34.7% | **34.9%** |
| Storage | ~1 GB/M docs | 0 | ~25 GB/M docs |

ColBERT достигает **качества cross-encoder** при **скорости, близкой к bi-encoder**. Главный tradeoff — **storage**: per-token embeddings для каждого документа занимают значительно больше места, чем один вектор.

## Ablation: что даёт каждый компонент

Из Section 4.4 оригинальной статьи:

| Конфигурация | MRR@10 | Delta |
|-------------|--------|-------|
| ColBERT (full) | 34.9 | — |
| Без query augmentation ([MASK]) | 33.4 | -1.5 |
| [CLS]-only (как bi-encoder) | 24.7 | **-10.2** |
| Без document filter (пунктуация) | 34.5 | -0.4 |
| Avg pooling вместо MaxSim | 32.0 | -2.9 |

**Late interaction vs single-vector:** переход от ColBERT к [CLS] dot product теряет **10+ MRR@10** — колоссальная разница. Это подтверждает центральный тезис: **сжатие в один вектор теряет слишком много информации**.

**Query augmentation** даёт +1.5 MRR@10 — значимый вклад soft query expansion.

## ColBERTv2: сжатие эмбеддингов

Главная проблема ColBERT v1: **storage**. MS MARCO (8.8M passages) → ~25 GB per-token embeddings. Для Wikipedia (21M passages) — ~60 GB. Непрактично для больших коллекций.

**ColBERTv2** (Santhanam et al., 2022) решает это через **residual compression**:

1. Обучается кодбук (codebook) центроидов
2. Каждый token embedding = **ближайший центроид** + **residual** (разница, квантизованная в 1-2 бита)
3. Размер на embedding: с 256 bytes → 20-36 bytes (**6-10x сжатие**)

Плюс: **distillation-based training** — ColBERTv2 обучается на soft labels от cross-encoder, улучшая качество retrieval.

**Результат:** ColBERTv2 — и **точнее**, и **компактнее** оригинального ColBERT.

## Наследие: late interaction как парадигм

ColBERT породил целое направление:

| Модель | Описание |
|--------|----------|
| **ColBERTv2** | Residual compression, distillation training |
| **PLAID** | Optimized engine для ColBERTv2 |
| **ColBERT-XM** | Multilingual ColBERT (mBERT-based) |
| **ColPali** | Late interaction для **визуальных** документов (PDFs) |
| **ColQwen** | Late interaction на Qwen2-VL backbone |
| **RAGatouille** | Python library для easy ColBERT usage |

ColBERT также часто используется как **re-ranker** в [[02 Areas/ML & DL/Concepts/Retrieval/Retrieval-Augmented Generation|RAG]] пайплайнах: bi-encoder для первичного retrieval → ColBERT для re-ranking → generator.

## Хронология

| Год | Milestone | Статья |
|-----|-----------|--------|
| 2019 | DPR, ANCE — dense bi-encoders | Karpukhin et al., Xiong et al. |
| **2020** | **ColBERT — late interaction** | **Khattab & Zaharia (SIGIR)** |
| 2022 | ColBERTv2 — residual compression | Santhanam et al. |
| 2023 | PLAID — optimized ColBERTv2 engine | Santhanam et al. |
| 2023 | RAGatouille — easy-to-use ColBERT | Clavié |
| 2024 | ColPali — late interaction для vision | Faysse et al. |

## Related concepts

- [[02 Areas/ML & DL/Concepts/Retrieval/Dense Retrieval|Dense Retrieval]] — bi-encoder подход, который ColBERT расширяет
- [[02 Areas/ML & DL/Concepts/Retrieval/Retrieval-Augmented Generation|RAG]] — ColBERT как retriever/re-ranker в RAG
- [[02 Areas/ML & DL/Concepts/Architectures/BERT|BERT]] — backbone энкодеров в ColBERT
- [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]] — self-attention внутри BERT энкодеров

## Дополнительные ресурсы

- [Khattab & Zaharia — ColBERT (2020)](https://arxiv.org/abs/2004.12832) — оригинальная статья
- [Santhanam et al. — ColBERTv2 (2022)](https://arxiv.org/abs/2112.01488) — residual compression + distillation
- [Weaviate — Overview of Late Interaction Models](https://weaviate.io/blog/late-interaction-overview) — обзор ColBERT, ColPali, ColQwen
- [RAGatouille — GitHub](https://github.com/bclavie/RAGatouille) — простая Python библиотека для ColBERT
- [Stanford DSPy](https://github.com/stanfordnlp/dspy) — framework от авторов ColBERT
