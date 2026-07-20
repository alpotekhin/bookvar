---
title: Embeddings
aliases: [Vector Embeddings, Sentence Embeddings]
type: concept
status: canonical
last_updated: 2026-07-16
primary_sources:
  - https://arxiv.org/abs/1301.3781
  - https://arxiv.org/abs/1908.10084
---

# Embeddings

**Embedding** — обученное отображение дискретного или сложного объекта в dense vector. Близость векторов должна отражать полезное для задачи сходство.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-first-60-64/sbert-bi-vs-cross-encoder.png]]

*Слева предложения кодируются независимо и сравниваются после pooling; справа
cross-encoder обрабатывает пару совместно. Источник: UKP Lab / Hugging Face,
[Sentence Transformers — Cross-Encoder
applications](https://www.sbert.net/examples/cross_encoder/applications/README.html).*

Для sequence embeddings важна именно левая ветвь: независимость кодирования
позволяет заранее вычислить document vectors. Правая ветвь обычно точнее для
конкретной пары, но уже не создаёт универсальный индексируемый embedding.

## Три разных значения

1. **Token embeddings** — строки embedding table внутри языковой модели.
2. **Contextual token embeddings** — представления токенов после Transformer.
3. **Sequence embeddings** — один вектор для предложения, запроса или документа.

Для semantic retrieval часто используют bi-encoder: query и document кодируются независимо, поэтому документы можно индексировать заранее. Contrastive loss сближает positive pairs и отдаляет negatives.

$$s(q,d)=\frac{e_q^\top e_d}{\|e_q\|\|e_d\|}\quad\text{or}\quad e_q^\top e_d.$$

Cosine и dot product эквивалентны для L2-normalized vectors. Выбор similarity должен совпадать с training/model card.

## Ограничения

Один vector сжимает документ и теряет детали; качество зависит от домена, языка, pooling, instruction prefix и negatives. Dense embeddings плохо гарантируют exact keyword match, поэтому [[02 Areas/ML & DL/01 Справочник/Retrieval/Retrieval|hybrid retrieval]] сочетает dense и sparse сигналы.

## Подробнее

Metric learning, negative sampling, pooling и оценивание retrieval-эмбеддингов
разобраны в главе [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/60 Embeddings и metric learning|Embeddings и metric learning]].

## Источники

- [word2vec](https://arxiv.org/abs/1301.3781)
- [Sentence-BERT](https://arxiv.org/abs/1908.10084)
