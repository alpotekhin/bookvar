---
title: "Open-domain QA"
aliases: [open-domain question answering, ODQA, OpenQA, open-domain QA]
type: concept
category: Retrieval
papers:
  - "[[02 Areas/ML & DL/Papers/Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks]]"
  - "[[02 Areas/ML & DL/Papers/REALM]]"
courses: []
sources:
  - "[Chen et al. — Reading Wikipedia to Answer Open-Domain Questions (DrQA, 2017)](https://arxiv.org/abs/1704.00051)"
  - "[Karpukhin et al. — Dense Passage Retrieval (DPR, 2020)](https://arxiv.org/abs/2004.04906)"
  - "[Lewis et al. — RAG (NeurIPS 2020)](https://arxiv.org/abs/2005.11401)"
---

# Open-domain QA

## Зачем это нужно: ответить на любой вопрос о мире

Open-domain QA — задача: дан произвольный factoid-вопрос, найди (или сгенерируй) точный ответ. **Без** предоставленного контекстного параграфа — модель должна **сама** найти информацию или «вспомнить» её из параметров.

Это отличает Open-domain QA от Reading Comprehension (RC): в RC тебе дают пассаж и просят найти ответ в нём (SQuAD). В Open-domain QA пассаж **не дан** — ответ может быть в любом из миллионов документов Wikipedia.

Аналогия: RC — это open-book экзамен с конкретной страницей учебника. Open-domain QA — экзамен, где можно использовать **всю библиотеку**, но нужно самому найти нужную книгу и страницу.

Open-domain QA стал **каноническим бенчмарком** для knowledge-intensive NLP и главным testbed для RAG-систем.

## Постановка задачи

**Вход**: произвольный factoid вопрос
```
"What is the birthplace of Albert Einstein?"
```

**Выход**: точный ответ
```
"Ulm, Germany"
```

**Evaluation**: **Exact Match (EM)** — строгое string matching с нормализацией (lowercase, remove articles/punctuation). EM — консервативная метрика: «Albert Einstein» vs «Einstein» = mismatch, хотя ответ верный.

## Бенчмарки

| Датасет | Размер | Тип вопросов | Источник |
|---------|--------|-------------|---------|
| **NaturalQuestions (NQ)** | ~100K | Реальные Google queries | Google Search |
| **TriviaQA (TQA)** | 95K | Trivial pursuit style | Trivia questions |
| **WebQuestions (WQ)** | 6K | Entity-centric | Google Suggest API |
| **CuratedTrec (CT)** | 2.2K | Разнообразные | MSNSearch, AskJeeves |
| **SQuAD-Open** | — | RC questions without gold context | SQuAD adaptation |
| **HotpotQA** | 113K | Multi-hop reasoning | Crowdsourced |

NaturalQuestions и TriviaQA — два **стандартных** бенчмарка. NQ ценится особенно, потому что вопросы отражают **реальные** поисковые запросы (не придуманы специально).

## Эволюция подходов: от DrQA к RAG

### Era 1: DrQA — Retrieve + Extract (2017)

Chen et al. (Facebook AI, 2017) — первая масштабная система для Open-domain QA:

```
Question → TF-IDF Retriever → Top-k Wikipedia docs → LSTM Reader → Extract span
```

**TF-IDF retriever**: sparse bigram hashing, быстрый но lexical. **Document Reader**: LSTM с attention, extract answer span из retrieved documents.

DrQA показал жизнеспособность подхода, но retriever был **heuristic** (TF-IDF) — не обучался на задаче.

### Era 2: DPR — Dense Retrieval (2020)

Karpukhin et al. (Facebook AI, 2020) заменили TF-IDF на **learned dense retriever**:

```
Question → BERT Query Encoder → FAISS MIPS → Top-k docs → BERT Reader → Extract span
```

Подробнее: [[02 Areas/ML & DL/Concepts/Retrieval/Dense Retrieval|Dense Retrieval]].

**Результаты**: DPR значительно превзошёл BM25/TF-IDF retriever на NQ:
- NQ: DPR **41.5 EM** vs DrQA 27.1 EM (+14.4 absolute)
- TQA: DPR **57.9 EM**

### Era 3: REALM — Retrieval-Augmented Pre-Training (2020)

Из [[02 Areas/ML & DL/Papers/REALM]] (Guu et al., Google Research):

```
Pre-training: MLM + Learned Retriever → Fine-tuning: Open-QA
```

Ключевая инновация: retriever обучается **end-to-end** через MLM loss ещё на этапе pre-training. Не нужны пары (query, document) — unsupervised.

**Результаты**: REALM 330M побеждает **всё**:
- NQ: **40.4 EM** (vs T5-11B 36.6, vs ORQA 33.3)
- WQ: **40.7 EM**
- CT: **46.8 EM**
- +4-16% absolute improvement над всеми предшествующими системами

Подробнее: [[02 Areas/ML & DL/Concepts/Retrieval/REALM|REALM]].

### Era 4: RAG — Retrieve + Generate (2020)

Из [[02 Areas/ML & DL/Papers/Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks]] (Lewis et al., Meta AI):

```
Question → DPR Retriever → Top-k docs → BART Generator → Generate answer
```

Революционный сдвиг: вместо **extract span** из документа — **generate answer** через seq2seq модель. Это позволяет:
- Синтезировать ответ из нескольких документов
- Генерировать ответы, которых нет verbatim в документах
- Работать с abstractive QA

**Результаты** (RAG-Sequence):
| Benchmark | RAG | DPR (extractive) | T5-11B (closed-book) |
|-----------|-----|-------------------|---------------------|
| **NQ** | **44.5** | 41.5 | 34.5 |
| **TQA** | **56.8** | 57.9 | 50.1 |
| **WQ** | **45.2** | 42.4 | 37.4 |
| **CT** | **52.2** | — | — |

Key insight из статьи:
> «Despite these being extractive tasks, we find that unconstrained generation outperforms previous extractive approaches.»

Генеративный подход лучше extractive **даже на extractive задачах** — модель не ограничена точным span'ом из документа.

### Era 5: LLM-based (2023+)

Из [[02 Areas/ML & DL/Papers/Harnessing the Power of LLMs in Practice- A Survey on ChatGPT and Beyond]]:

```
Question → LLM (GPT-4 / Claude) → Answer (parametric knowledge)
        or
Question → Search API → Retrieved docs → LLM → Answer (augmented)
```

- GPT-3 few-shot на NQ: конкурирует с fine-tuned retrieval models
- GPT-4 на MMLU (knowledge-intensive): **86.4%** (5-shot) — SOTA
- TriviaQA zero-shot LLM: comparable с many fine-tuned approaches

**Knowledge-intensive QA = одна из задач, где LLM реально превосходят** fine-tuned encoder-only модели. Это оправдывает cost of scale.

Но даже GPT-4 benefit'ится от RAG: актуальность, attribution, снижение hallucination.

## Три парадигмы: сравнение

| Парадигма | Как работает | Плюсы | Минусы |
|-----------|-------------|-------|--------|
| **Closed-Book** | Ответ «из головы» (T5-11B, GPT-4) | Простота, нет retrieval latency | Hallucination, stale knowledge, непрозрачно |
| **Open-Book Extractive** | Retrieve → extract span (DPR) | Точный ответ из документа | Ограничен span'ом, не может синтезировать |
| **Open-Book Generative** | Retrieve → generate (RAG) | Гибкость, синтез из нескольких sources | Может проигнорировать retrieved context |

В **production** доминирует гибрид: LLM + retrieval (ChatGPT Browse, Perplexity, Google SGE).

## Wikipedia как стандартный knowledge source

De facto стандарт для Open-domain QA:
- **Размер**: ~21M chunks по 100 слов (из ~6M статей English Wikipedia)
- **FAISS index**: dense embeddings для всех chunks
- **Granularity**: 100-word non-overlapping chunks (DPR/RAG) или 288 wordpieces (REALM)

Выбор chunk size — trade-off: маленькие chunks → точный retrieval, но меньше контекста. Большие chunks → больше контекста, но retrieval noise.

## Открытые проблемы

1. **Multi-hop QA**: вопросы, требующие нескольких шагов retrieval (HotpotQA). Одного retrieval шага недостаточно
2. **Temporal questions**: «Кто сейчас президент?» — ответ зависит от времени
3. **Ambiguous questions**: «What is Amazon?» — компания? река? сериал?
4. **Evaluation beyond EM**: EM слишком строгая метрика. Нужны semantic similarity метрики
5. **Attribution**: какой документ подтверждает какую часть ответа?

## Key papers

- [[02 Areas/ML & DL/Papers/Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks]] — RAG для Open-QA; NQ/TQA/WQ/CT results
- [[02 Areas/ML & DL/Papers/REALM]] — retrieval-augmented pre-training для Open-QA

## Related concepts

- [[02 Areas/ML & DL/Concepts/Retrieval/Retrieval-Augmented Generation|Retrieval-Augmented Generation]] — доминирующий подход к Open-QA
- [[02 Areas/ML & DL/Concepts/Retrieval/Dense Retrieval|Dense Retrieval]] — retriever backbone (DPR, ColBERT)
- [[02 Areas/ML & DL/Concepts/Retrieval/Knowledge-Intensive NLP|Knowledge-Intensive NLP]] — Open-QA как каноническая KI задача
- [[02 Areas/ML & DL/Concepts/Retrieval/REALM|REALM]] — retrieval-augmented pre-training
- [[02 Areas/ML & DL/Concepts/Retrieval/Self-RAG|Self-RAG]] — adaptive retrieval с reflection tokens для QA

## Дополнительные ресурсы

- [Chen et al. — DrQA (ACL 2017)](https://arxiv.org/abs/1704.00051) — пионерская система Open-domain QA
- [Karpukhin et al. — DPR (EMNLP 2020)](https://arxiv.org/abs/2004.04906) — dense retriever для Open-QA
- [Lewis et al. — RAG (NeurIPS 2020)](https://arxiv.org/abs/2005.11401) — generative approach к Open-QA
- [Guu et al. — REALM (ICML 2020)](https://arxiv.org/abs/2002.08909) — retrieval-augmented pre-training
