---
title: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG)"
url: https://arxiv.org/abs/2005.11401
authors: [Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela]
year: 2020
date_reviewed: 2026-04-06
type: source-note
status: legacy
category: paper
tags:
  - Feature
  - LLM
Date: 2020-05-22
Organization: Meta
Status: Done
concepts:
  - "[[02 Areas/ML & DL/Concepts/Retrieval/Retrieval-Augmented Generation|RAG]]"
  - "[[02 Areas/ML & DL/Concepts/Retrieval/Dense Retrieval|Dense Retrieval]]"
  - "[[02 Areas/ML & DL/Concepts/Retrieval/Knowledge-Intensive NLP|Knowledge-Intensive NLP]]"
  - "[[02 Areas/ML & DL/Concepts/Retrieval/Open-domain QA|Open-domain QA]]"
  - "[[02 Areas/ML & DL/Concepts/Architectures/Encoder-Decoder|Encoder-Decoder]]"
raw: "[[02 Areas/ML & DL/raw/papers/retrieval-augmented-generation-for-knowledge-intensive-nlp-t/paper.txt]]"
---

# Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG)

**Authors:** Patrick Lewis, Ethan Perez et al. (Facebook AI Research / UCL / NYU)
**Published:** 2020 (NeurIPS 2020; arXiv:2005.11401v4, Apr 2021)
**URL:** https://arxiv.org/abs/2005.11401

## TL;DR

RAG — первая общая рецептура fine-tuning для моделей, комбинирующих **параметрическую** (seq2seq LM) и **непараметрическую** (dense vector index Wikipedia) память. Ретривер (DPR на основе BERT) и генератор (BART-large) обучаются **end-to-end** совместно без прямого supervision на то, какой документ нужно найти. RAG устанавливает SOTA на трёх задачах open-domain QA и генерирует более фактичные, специфичные и разнообразные ответы, чем parametric-only baselines.

## Problem

Параметрические LM (GPT, T5, BART) хранят знания в своих весах, но:
1. **Не могут легко обновить знания** — требуется переобучение
2. **Не могут дать provenance** — неизвестно, откуда взялся факт
3. **Галлюцинируют** — уверенно генерируют неверные факты
4. **Ограниченная ёмкость** — не могут хранить всю Wikipedia в 400M параметрах

Предыдущие подходы с non-parametric memory (REALM, ORQA) работали только с extractive downstream задачами и masked LM. RAG расширяет это на **generative seq2seq** задачи с общей fine-tuning рецептурой.

## Method

### Архитектура (Figure 1)

Два компонента:
1. **Ретривер** `p_η(z|x)` — возвращает top-K документов для запроса x
2. **Генератор** `p_θ(y_i|x, z, y_{1:i-1})` — генерирует ответ на основе запроса + найденных документов

### Retriever: DPR (Dense Passage Retriever)

Bi-encoder архитектура:
```
p_η(z|x) ∝ exp(d(z)ᵀ q(x))
d(z) = BERT_d(z)    # document encoder
q(x) = BERT_q(x)    # query encoder
```
Top-k поиск = Maximum Inner Product Search (MIPS), решается за сублинейное время через FAISS HNSW индекс.

**Wikipedia index:** декабрь 2018, статьи разбиты на 100-словные chunks → **21M документов**.

Pre-trained DPR ретривер, обученный на TriviaQA и Natural Questions. Во время fine-tuning обновляется только **query encoder** (document encoder и индекс фиксируются для экономии compute).

### Generator: BART-large

BART-large (400M параметров) — pretrained seq2seq transformer с denoising objective.
Конкатенация запроса и найденного документа: `input = [x; z]` → BART генерирует y.

### Два варианта маргинализации (Section 2.1)

**RAG-Sequence:** один документ для всей последовательности
```
p_RAG-Seq(y|x) ≈ Σ_{z∈top-k} p_η(z|x) · p_θ(y|x,z)
               = Σ_{z∈top-k} p_η(z|x) · Π_i p_θ(y_i|x,z,y_{1:i-1})
```
Декодирование: beam search по каждому документу отдельно, затем маргинализация.

**RAG-Token:** разный документ для каждого токена
```
p_RAG-Token(y|x) ≈ Π_i Σ_{z∈top-k} p_η(z|x) · p_θ(y_i|x,z,y_{1:i-1})
```
Декодирование: обычный beam decoder с составной transition probability.

### Training

Joint end-to-end обучение ретривера и генератора:
```
minimize Σ_j -log p(y_j|x_j)    # отрицательный marginal log-likelihood
```
Никакого прямого supervision на retrieved documents. SGD с Adam.

### Decoding details

- RAG-Token: стандартный beam decoder
- RAG-Sequence:
  - **Thorough Decoding**: beam search per document, дополнительные forward passes для отсутствующих hypotheses
  - **Fast Decoding**: ≈ 0 для гипотез, не появившихся в beams документа (быстрее, незначительно хуже)

## Key Results

### Open-Domain QA (Table 1, Exact Match)

| Модель | NQ | TQA (wiki) | WQ | CT |
|--------|----|---------|----|-----|
| T5-11B (closed book) | 34.5 | 50.1 | 37.4 | – |
| T5-11B+SSM | 36.6 | 60.5 | 44.7 | – |
| REALM | 40.4 | – | 40.7 | 46.8 |
| DPR | 41.5 | 66.1 | 41.1 | 50.6 |
| **RAG-Token** | **44.1** | **66.1** | **45.5** | **50.0** |
| **RAG-Sequence** | **44.5** | **68.0** | **45.2** | **52.2** |

RAG устанавливает SOTA на NQ, WQ, CuratedTrec. На TQA wiki test — превосходит T5-11B+SSM при 400M параметрах (vs 11B).

**Важное наблюдение:** RAG генерирует верные ответы даже когда ответа нет ни в одном retrieved document — **11.8% accuracy на NQ** в таких случаях (extractive → 0%).

### Abstractive QA: MS-MARCO (Table 2)

RAG-Sequence превосходит BART на +2.6 BLEU и +2.6 ROUGE-L. Приближается к SOTA-моделям, которые используют **gold passages** — RAG работает без них.

### Jeopardy Question Generation (Table 2, Table 4)

| Метрика | BART | RAG-Token |
|---------|------|-----------|
| Q-BLEU-1 | 19.7 | **22.2** |

**Human evaluation (452 пары):**
- RAG более фактичен: **42.7%** vs BART 7.1%
- RAG более специфичен: **37.4%** vs BART 16.8%
- Оба хороши: 11.7% фактичность, 11.8% специфичность

### FEVER Fact Verification (Table 2)

| Модель | 3-way label accuracy | 2-way |
|--------|---------------------|-------|
| SOTA pipeline | 76.8 | 92.2* |
| RAG-Token | **72.5** | **89.5** |

Всего -4.3% от SOTA на 3-way, несмотря на отсутствие retrieval supervision. Для 2-way: -2.7% от supervised RoBERTa модели с gold evidence.

Top-1 retrieved document из gold evidence: **71% случаев**; в top-10: **90% случаев**.

### Generation Diversity (Table 5)

Distinct trigrams ratio:
- BART: 70.7% (MS-MARCO), 32.4% (Jeopardy)
- RAG-Token: 77.8%, 46.8%
- RAG-Sequence: ~82-83%, ~50%+

RAG генерирует значительно более разнообразный текст без специальных diversity-promoting методов декодирования.

### Index Hot-Swapping

Замена индекса 2016→2018 меняет ответы для 70% изменившихся world leaders — демонстрация возможности обновления знаний без переобучения.

### Retrieval Ablation (Table 6)

Learned retrieval лучше frozen retrieval на всех задачах кроме FEVER. RAG-DPR значительно превосходит BM25 на QA задачах.

## Limitations

- Индексирование Wikipedia требует значительных вычислений (embedding 21M документов)
- Обновление document encoder = пересборка всего индекса
- Скорость инференса ниже parametric-only моделей (нужно MIPS)
- MS-MARCO результаты ограничены: часть вопросов не отвечаема из Wikipedia

## My notes

- RAG — **основополагающая работа** для современных RAG систем. Заложила архитектурный шаблон: retriever + generator + end-to-end training.
- **RAG-Token vs RAG-Sequence**: Token лучше для задач, где ответ объединяет знания из нескольких документов (Jeopardy). Sequence лучше для задач с единым источником.
- **Index hot-swapping** — принципиальное преимущество перед параметрическими моделями. Современные RAG системы используют это активно (corpora updates, domain switching).
- Интересный inductive bias: параметрическая модель (BART) может **дополнять** непараметрическую. Пример с Хемингуэем: DPR находит нужный документ → BART дописывает заголовок книги из параметрической памяти.
- Первичная реализация открыта в HuggingFace Transformers — это значительно ускорило adoption.
- Open-domain QA без gold passages: RAG конкурирует с T5-11B при 400M параметрах — огромная разница в эффективности.
