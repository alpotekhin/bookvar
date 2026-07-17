---
title: "Dense Retrieval"
aliases: [dense passage retrieval, DPR, bi-encoder retrieval, semantic search, neural retrieval]
type: concept
status: legacy
category: Retrieval
papers:
  - "[[02 Areas/ML & DL/Papers/Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks]]"
  - "[[02 Areas/ML & DL/Papers/REALM]]"
  - "[[02 Areas/ML & DL/Papers/ColBERT]]"
courses:
  - "[[02 Areas/ML & DL/Courses/MIPT NLP/ANCE — Dense Retrieval Training|MIPT NLP — ANCE]]"
sources:
  - "[Karpukhin et al. — Dense Passage Retrieval for Open-Domain QA (2020)](https://arxiv.org/abs/2004.04906)"
  - "[Khattab & Zaharia — ColBERT (2020)](https://arxiv.org/abs/2004.12832)"
---

# Dense Retrieval

## Зачем это нужно: семантика вместо ключевых слов

Традиционный information retrieval (IR) строится на **sparse** представлениях: BM25, TF-IDF. Документ — это мешок слов, запрос — тоже мешок слов, matching — пересечение множеств.

Проблема очевидна: запрос «как лечить головную боль» не найдёт документ «терапия мигрени аспирином» — ни одного общего слова, хотя семантически это прямой ответ.

**Dense Retrieval** заменяет sparse keyword matching на **dense vector representations**: запрос и документ кодируются в плотные векторы одного пространства, похожесть = dot product / cosine similarity. Семантически близкие тексты оказываются рядом в embedding space **независимо от совпадения слов**.

После DPR (Karpukhin et al., 2020) dense retrieval стал доминирующим подходом в neural information retrieval и **backbone'ом** для RAG систем.

## Bi-Encoder Architecture (DPR)

### Основная идея

Два отдельных BERT-энкодера: один для запросов ($E_Q$), другой для документов ($E_D$):

$$\text{score}(q, d) = E_Q(q)^T \cdot E_D(d)$$

**Offline** (один раз):
1. Encode все документы: $d_i \to E_D(d_i)$ → dense vectors (768-мерные)
2. Построить index для быстрого поиска (FAISS)

**Online** (каждый запрос):
1. Encode запрос: $q \to E_Q(q)$ → один 768-мерный вектор
2. MIPS (Maximum Inner Product Search) → top-$k$ ближайших документов

**Ключевое преимущество**: документы кодируются **один раз offline**. Inference = один forward pass (query encoder) + ANN lookup. На 21M Wikipedia chunks это занимает **миллисекунды**.

### Approximate Nearest Neighbor (ANN)

Точный поиск ближайшего соседа в 768-мерном пространстве среди 13-21M векторов — слишком медленно. Используются приближённые методы:

| Метод | Как работает | Плюсы / Минусы |
|-------|-------------|----------------|
| **FAISS** (Facebook) | IVF + PQ (кластеризация + product quantization) | Стандарт индустрии, GPU-accelerated |
| **HNSW** | Hierarchical Navigable Small World graph | Высокий recall, не требует GPU |
| **IVF** | Inverted File Index с кластеризацией | Быстрый, configurable trade-off |
| **ScaNN** (Google) | Anisotropic quantization | Оптимизирован для MIPS |

RAG paper: Wikipedia разбита на **21M chunks** по 100 слов. FAISS с HNSW index. Top-$k = 5{-}10$ документов.

### Training: Contrastive Learning

DPR обучается через **contrastive loss** с positive/negative pairs:

$$\mathcal{L} = -\log \frac{\exp(d^+ \cdot q)}{\exp(d^+ \cdot q) + \sum_k \exp(d^-_k \cdot q)}$$

Для каждого вопроса:
- **Positive**: пассаж, содержащий ответ
- **Negative**: (1) random пассажи из batch (in-batch negatives), (2) **hard negatives** — пассажи, найденные BM25, но не содержащие ответ

**Hard negatives критически важны**: обучение только с random negatives даёт посредственный retriever. BM25 hard negatives заставляют модель различать семантически похожие, но неправильные документы.

### ANCE: динамические hard negatives

**ANCE** (Approximate Nearest Neighbor Negative Contrastive Estimation, Xiong et al., 2021):

Проблема: hard negatives из BM25 — фиксированы и со временем становятся «лёгкими» для модели.

Решение: периодически обновлять hard negatives **из собственного ANN index** модели:
1. Train DPR для N шагов
2. Пересчитать все document embeddings
3. Для каждого query найти top-$k$ «ложных» документов из свежего index
4. Использовать их как hard negatives для следующих N шагов

Результат: ANCE значительно превосходит static-negative DPR на MS MARCO и NQ.

## BM25 vs Dense Retrieval: когда что лучше

| Аспект | BM25 (Sparse) | Dense Retrieval |
|--------|--------------|-----------------|
| **Matching** | Lexical (exact keywords) | Semantic (meaning) |
| **«головная боль» → «мигрень»** | Не найдёт | Найдёт |
| **Acronyms/jargon** | Находит exact match | Может промахнуться |
| **Speed** | Очень быстрый (inverted index) | Быстрый (ANN) |
| **Training data** | Не нужны | Нужны пары (query, positive) |
| **Index size** | Компактный | Большой (768 dims * N docs) |
| **Из коробки** | Хорошо работает | Нужен fine-tuning |

**На практике**: **hybrid retrieval** (BM25 + DPR) — стандарт в production:
1. BM25 для высокого recall (быстро, ловит exact matches)
2. Dense retriever для re-ranking (семантика, ловит paraphrases)

Или reciprocal rank fusion: объединение ranked lists из обоих retriever'ов.

## ColBERT: Late Interaction — золотая середина

Из [[02 Areas/ML & DL/Papers/ColBERT]] (Khattab & Zaharia, 2020):

### Проблема: bi-encoder vs cross-encoder

| Подход | Accuracy | Speed | Как работает |
|--------|----------|-------|-------------|
| **Bi-encoder** (DPR) | Good | Fast | Отдельные embeddings → dot product |
| **Cross-encoder** (BERT) | Best | Slow (~170ms/pair) | Concat(q, d) → full cross-attention |

Bi-encoder быстрый (pre-computed docs), но теряет взаимодействие token-level между query и doc. Cross-encoder точный, но нужен отдельный forward pass для **каждой пары** (q, d) — нереально для 21M документов.

### Решение: MaxSim

ColBERT кодирует query и document **отдельно** (как bi-encoder), но сохраняет **per-token embeddings** (не CLS-pooling):

$$S(q, d) = \sum_{i \in \text{tokens}(q)} \max_{j \in \text{tokens}(d)} E_q[i] \cdot E_d[j]^T$$

Каждый query token ищет **наиболее похожий** document token → MaxSim. Сумма MaxSim'ов по всем query tokens = score.

**Архитектура**:
1. Query encoder (BERT): `[Q]` + query tokens → Q embeddings (per-token)
2. Document encoder (BERT): `[D]` + doc tokens → D embeddings (**pre-computed offline**)
3. Late interaction: MaxSim over all query-document token pairs

**Результаты** (MS MARCO Ranking):
- BERT cross-encoder: MRR@10 ~0.365, latency ~170ms/query
- ColBERT (re-rank): **competitive MRR, 170x speedup, 14000x fewer FLOPs**
- ColBERT (full retrieval): end-to-end SOTA среди non-cross-encoder подходов

### Сравнительная таблица подходов

| Подход | Тип | Accuracy | Speed | Index Size |
|--------|-----|----------|-------|------------|
| BM25 | Sparse keyword | Moderate | Very fast | Small |
| DPR bi-encoder | Dense | Good | Fast | Medium (1 vec/doc) |
| **ColBERT** | Dense + late interaction | Better | Fast | Large (N vecs/doc) |
| Cross-encoder | Interaction-based | Best | Slow | N/A (no pre-compute) |

**Trade-off ColBERT**: хранит per-token embeddings → index значительно больше (128 dims * avg_tokens * N docs). ColBERTv2 решает это через residual compression.

## Dense Retrieval в RAG pipeline

Dense retrieval — **критический компонент** RAG. Качество retrieval напрямую определяет качество generation:

```
Query → Dense Retriever → Top-k documents → LLM generates answer
         ↑                                    ↑
    Bi-encoder / ColBERT              Retrieved context
```

Из [[02 Areas/ML & DL/Papers/Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks]]:
- DPR retriever в RAG: NQ 44.5 EM, TQA 56.8 EM
- BM25 retriever в RAG: значительно хуже
- **Retrieval quality = generation quality ceiling**

## Современные embedding models

Эволюция dense retrieval models:

| Model | Год | Dims | Особенности |
|-------|-----|------|-------------|
| DPR | 2020 | 768 | BERT bi-encoder, NQ-trained |
| Contriever | 2022 | 768 | Unsupervised contrastive pre-training |
| E5 | 2023 | 768-1024 | Instruction-tuned retriever |
| BGE | 2023 | 768-1024 | BAAI, multi-lingual |
| GTE | 2023 | 768 | Alibaba, efficient |
| Nomic Embed | 2024 | 768 | Long context (8K) |
| **Cohere Embed v3** | 2023 | 1024 | SOTA, multi-lingual, binary quantization |
| **OpenAI text-embedding-3** | 2024 | 256-3072 | Variable dims, Matryoshka |

Trend: instruction-tuned retrievers + Matryoshka representations (variable dimensionality) + binary quantization для компрессии.

## Key papers

- [[02 Areas/ML & DL/Papers/Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks]] — DPR bi-encoder в RAG context
- [[02 Areas/ML & DL/Papers/REALM]] — retrieval-augmented pre-training; end-to-end differentiable retrieval
- [[02 Areas/ML & DL/Papers/ColBERT]] — late interaction для 170x speedup vs cross-encoder

## Related concepts

- [[02 Areas/ML & DL/Concepts/Retrieval/Retrieval-Augmented Generation|Retrieval-Augmented Generation]] — RAG, использующий dense retrieval
- [[02 Areas/ML & DL/Concepts/Retrieval/Open-domain QA|Open-domain QA]] — основной бенчмарк для dense retrieval
- [[02 Areas/ML & DL/Concepts/Retrieval/Knowledge-Intensive NLP|Knowledge-Intensive NLP]] — класс задач, требующих retrieval
- [[02 Areas/ML & DL/Concepts/Retrieval/REALM|REALM]] — end-to-end retrieval pre-training
- [[02 Areas/ML & DL/Concepts/Architectures/BERT|BERT]] — backbone для большинства retriever'ов

## Дополнительные ресурсы

- [Karpukhin et al. — Dense Passage Retrieval (EMNLP 2020)](https://arxiv.org/abs/2004.04906) — оригинальная DPR статья
- [Khattab & Zaharia — ColBERT (SIGIR 2020)](https://arxiv.org/abs/2004.12832) — late interaction model
- [FAISS documentation](https://faiss.ai/) — библиотека для ANN search
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — бенчмарк embedding models
