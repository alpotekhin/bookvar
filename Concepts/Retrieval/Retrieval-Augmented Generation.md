---
title: "Retrieval-Augmented Generation"
aliases: [RAG, retrieval-augmented generation, Retrieval Augmented Generation]
type: concept
status: legacy
category: Retrieval
papers:
  - "[[02 Areas/ML & DL/Papers/Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks|RAG (Lewis et al., 2020)]]"
  - "[[02 Areas/ML & DL/Papers/REALM|REALM (Guu et al., 2020)]]"
  - "[[02 Areas/ML & DL/Papers/Self-RAG|Self-RAG (Asai et al., 2023)]]"
sources:
  - "[Lewis et al. — RAG for Knowledge-Intensive NLP Tasks (2020)](https://arxiv.org/abs/2005.11401)"
  - "[Lilian Weng — Open-Domain QA](https://lilianweng.github.io/posts/2020-10-29-odqa/)"
  - "[AWS — What is RAG?](https://aws.amazon.com/what-is/retrieval-augmented-generation/)"
  - "[Pinecone — RAG Guide](https://www.pinecone.io/learn/retrieval-augmented-generation/)"
courses: []
---

# Retrieval-Augmented Generation (RAG)

## Зачем это нужно: проблема параметрической памяти

Большие языковые модели (LLM) хранят знания **в своих весах** — это параметрическая память. У такого подхода три фундаментальные проблемы:

1. **Галлюцинации** — модель генерирует уверенные, но фактически неверные ответы, потому что «помнит» не факт, а статистический паттерн.
2. **Устаревание знаний** — знания заморожены на момент cutoff date обучения. GPT-4 не знает о событиях после его обучения, и единственный способ «обновить» — переобучение за миллионы долларов.
3. **Непрозрачность** — невозможно проверить, откуда модель взяла конкретный факт. Нет цитирования, нет provenance.

**Идея RAG** (Lewis et al., 2020, Facebook AI Research): дать модели доступ к **внешней памяти** — базе документов. Перед генерацией ответа модель **ищет** релевантные документы и использует их как контекст. Это комбинация параметрической (веса LM) и непараметрической (индекс документов) памяти.

Аналогия: представь студента на экзамене. Параметрическая модель — студент, который отвечает только по памяти. RAG — студент, которому разрешили пользоваться конспектами. Второй ответит точнее, может показать источник, и его конспекты можно обновить без переобучения студента.

## Полный pipeline: от запроса до ответа

### Оригинальная архитектура RAG (Lewis et al., 2020)

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/retrieval-augmented-generation-for-knowledge-intensive-nlp-t/rag-fig1.png]]
*RAG pipeline: Query Encoder кодирует запрос, MIPS находит top-k документов из индекса, Generator (BART) генерирует ответ с маргинализацией по retrieved документам (источник: Lewis et al., 2020)*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/retrieval-augmented-generation-for-knowledge-intensive-nlp-t/retrieval-with-embeddings.png]]
*Retrieval с BERT-эмбеддингами: запрос и документы кодируются в плотные векторы, nearest neighbor search находит семантически близкие чанки (источник: Jay Alammar)*

Два компонента работают вместе:

### 1. Retriever: Dense Passage Retrieval (DPR)

DPR — bi-encoder на основе BERT. Два независимых энкодера:

$$p_\eta(z|x) \propto \exp\left(\mathbf{d}(z)^\top \cdot \mathbf{q}(x)\right)$$

где $\mathbf{d}(z) = \text{BERT}_d(z)$ — document encoder, $\mathbf{q}(x) = \text{BERT}_q(x)$ — query encoder.

**Индекс:** Wikipedia (Dec 2018), разбитая на **21 миллион чанков по 100 слов**. Каждый чанк закодирован в плотный вектор через $\text{BERT}_d$. Векторы хранятся в **FAISS** (Facebook AI Similarity Search) с HNSW-индексом для sub-linear поиска (Maximum Inner Product Search, MIPS).

**Ключевой момент:** document encoder $\text{BERT}_d$ **заморожен** при обучении RAG. Обновление индекса при каждом шаге градиентного спуска слишком дорого. Обновляется только query encoder $\text{BERT}_q$ + генератор BART.

### 2. Generator: BART-large

BART-large (400M параметров) — encoder-decoder seq2seq модель. Вход: конкатенация запроса $x$ и retrieved документа $z$. Выход: сгенерированный ответ $y$.

BART был предобучен с denoising objective (маскирование, перестановка предложений, удаление токенов), что делает его сильным генератором свободного текста — в отличие от extractive моделей (DPR reader), которые могут только выделять спан из документа.

## RAG-Sequence vs RAG-Token: два способа маргинализации

Документы $z$ — скрытая (latent) переменная. Вопрос: как комбинировать информацию из $k$ retrieved документов?

### RAG-Sequence: один документ на весь ответ

$$p_\text{RAG-Seq}(y|x) \approx \sum_{z \in \text{top-}k} p_\eta(z|x) \prod_{i=1}^{N} p_\theta(y_i|x, z, y_{1:i-1})$$

Для каждого документа генерируется **полный ответ**, затем вероятности взвешиваются. Интуитивно: «выбери лучший источник и ответь по нему».

**Лучше для:** задач, где ответ целиком содержится в одном документе (Open-domain QA: NQ, WebQuestions).

### RAG-Token: разные документы для каждого токена

$$p_\text{RAG-Token}(y|x) \approx \prod_{i=1}^{N} \sum_{z \in \text{top-}k} p_\eta(z|x) \cdot p_\theta(y_i|x, z, y_{1:i-1})$$

На каждом шаге генерации модель может «переключаться» между документами. Маргинализация происходит **внутри** произведения, а не снаружи.

**Лучше для:** генерации, требующей синтеза из нескольких источников (Jeopardy question generation — нужно объединить факты из разных документов в один вопрос).

### Обучение

Совместная оптимизация retriever + generator без supervision на том, какой документ правильный:

$$\mathcal{L} = \sum_j -\log p(y_j | x_j)$$

Модель сама учится, какие документы полезны, через маргинальное правдоподобие.

## Self-RAG: адаптивный retrieval

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/self-rag/fig1.png]]
*Figure 1 из Self-RAG (Asai et al., 2023): слева — стандартный RAG всегда делает retrieval; справа — Self-RAG адаптивно решает, когда retrievить, используя специальные reflection tokens.*

Стандартный RAG **всегда** делает retrieval — даже когда модель уже знает ответ (например, «Сколько будет 2+2?»). Это добавляет latency и может привести к ухудшению из-за нерелевантных документов.

**Self-RAG** (Asai et al., 2023) решает эту проблему через **reflection tokens** — специальные маркеры, которые модель генерирует сама:

| Token | Назначение |
|-------|-----------|
| `[Retrieve]` | Нужен ли retrieval для этого сегмента? (Yes/No/Continue) |
| `[IsRel]` | Релевантен ли retrieved документ запросу? |
| `[IsSup]` | Поддерживает ли документ сгенерированный ответ? (Fully/Partially/No) |
| `[IsUse]` | Полезен ли итоговый ответ? (Rating 1-5) |

Self-RAG обучается через reinforcement learning с critic-моделью, которая генерирует reflection tokens на обучающих данных. На инференсе модель сама решает: нужен retrieval или нет, и если да — критически оценивает retrieved документы.

## CRAG: Corrective RAG

CRAG (Yan et al., 2024) добавляет **evaluator** после retrieval шага. Если retrieved документы оценены как нерелевантные (confidence ниже порога), CRAG переключается на web search. Три стратегии:

- **Correct** — документы релевантны → используем как есть
- **Incorrect** — документы нерелевантны → web search
- **Ambiguous** — неуверенность → комбинируем оба источника

## Стратегии чанкинга (chunking)

Качество RAG критически зависит от того, как документы разбиваются на чанки:

| Стратегия | Размер | Плюсы | Минусы |
|-----------|--------|-------|--------|
| Fixed-size | 100-512 токенов | Простота, предсказуемость | Может разрезать предложения |
| Sentence-based | 1-5 предложений | Семантическая целостность | Разный размер чанков |
| Recursive | Адаптивный | Уважает структуру документа | Сложнее имплементация |
| Semantic | По эмбеддингам | Группирует по смыслу | Дорого вычислительно |

Оригинальный RAG использовал **100-word chunks** из Wikipedia — это фиксированный подход. Современные системы часто используют recursive chunking с overlap (например, 512 токенов с overlap 50).

**Overlap** критически важен: без него контекст на границе чанков теряется. С overlap 10-20% — значительно лучше retrieval качество.

## Modern RAG: Production-системы

Современные RAG-системы далеко ушли от оригинальной архитектуры Lewis et al.:

### Hybrid Retrieval: BM25 + Dense

Чисто dense retrieval (DPR) плохо работает с exactly matching терминами и аббревиатурами. BM25 (TF-IDF based) отлично находит точные совпадения. **Hybrid** комбинирует оба: dense retrieval для семантического поиска + BM25 для keyword matching. Reciprocal Rank Fusion (RRF) или learned score combination.

### Re-ranking

После retrieval top-k (обычно k=20-100), **cross-encoder** переранжирует документы. Cross-encoder (например, на основе [[02 Areas/ML & DL/Concepts/Retrieval/ColBERT|ColBERT]] или BERT) видит query+document **вместе** и даёт более точный relevance score, но работает медленно. Поэтому: сначала bi-encoder для speed, потом cross-encoder для precision.

### Query Expansion / Transformation

- **HyDE** (Hypothetical Document Embeddings): LLM генерирует гипотетический ответ, потом ищет документы, похожие на этот гипотетический ответ
- **Multi-query**: один запрос переформулируется в несколько вариантов для лучшего coverage
- **Step-back prompting**: абстрагирование запроса для поиска более общих документов

## Ключевые результаты (Lewis et al., 2020)

**Open-domain QA** (Exact Match score):

| Модель | NQ | TQA | WQ | CT |
|--------|-----|-----|-----|-----|
| T5-11B (closed-book) | 34.5 | 50.1 | 37.4 | — |
| DPR (extractive) | 41.5 | 57.9 | 41.1 | 50.6 |
| **RAG-Sequence** | **44.5** | **56.8** | **45.2** | **52.2** |
| **RAG-Token** | 44.1 | 55.2 | 45.5 | 50.0 |

RAG-Sequence — SOTA на Natural Questions, WebQuestions и CuratedTrec. Ключевое: RAG превосходит и closed-book T5-11B (модель в 27x больше), и extractive DPR.

**Jeopardy Question Generation:** RAG генерирует более фактуальные (human eval), более специфичные и более разнообразные вопросы, чем BART baseline.

**FEVER (Fact Verification):** within 4.3% от SOTA pipeline-моделей **без supervision на retrieval**.

## REALM: предшественник RAG

REALM (Guu et al., 2020) — раннее retrieval-augmented pre-training:

| | REALM | RAG |
|--|-------|-----|
| Retrieval при | **Pre-training** (MLM) | Fine-tuning |
| Retriever training | End-to-end (обновляет document encoder) | Только query encoder |
| Generator | BERT (extractive) | BART (abstractive) |
| Задачи | Только extractive QA | QA + generation + verification |

REALM показал +4-16% absolute accuracy на Open-QA, но ограничен extractive сценариями. RAG обобщил идею на любые seq2seq задачи.

## Хронология

| Год | Milestone | Статья |
|-----|-----------|--------|
| 2019 | ORQA — first open-retrieval QA | Lee et al. |
| 2020 | REALM — retrieval-augmented pre-training | Guu et al. |
| 2020 | DPR — dense passage retrieval baseline | Karpukhin et al. |
| **2020** | **RAG — retrieval + generation** | **Lewis et al.** |
| 2021 | FiD (Fusion-in-Decoder) — concat all docs | Izacard & Grave |
| 2023 | Self-RAG — adaptive retrieval с reflection tokens | Asai et al. |
| 2024 | CRAG — corrective RAG с web fallback | Yan et al. |
| 2024-25 | Agentic RAG — multi-step retrieval с tool use | Различные |

## Почему RAG стал стандартом в production

1. **Factuality** — retrieved документы = проверяемый источник фактов. RAG галлюцинирует значительно реже, чем чисто параметрический LLM.
2. **Обновляемость** — заменяем документный индекс → обновляем знания без retraining. Стоимость: минуты vs миллионы долларов.
3. **Интерпретируемость** — документы из retrieval можно показать пользователю → provenance для ответа.
4. **Стоимость** — RAG с маленькой моделью (7B-13B) + хорошим индексом часто превосходит огромную модель (70B+) без retrieval.
5. **Безопасность** — доступ к документам контролируется, можно ограничить scope знаний.

## Related concepts

- [[02 Areas/ML & DL/Concepts/Retrieval/ColBERT|ColBERT]] — late interaction retrieval, часто используется как re-ranker в RAG
- [[02 Areas/ML & DL/Concepts/Retrieval/Dense Retrieval|Dense Retrieval]] — основа retriever в RAG
- [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]] — cross-attention между query и retrieved контекстом
- [[02 Areas/ML & DL/Concepts/Inference/In-context Learning|In-context Learning]] — retrieved документы как контекст для ICL
- [[02 Areas/ML & DL/Concepts/Architectures/BERT|BERT]] — основа DPR энкодеров
- [[02 Areas/ML & DL/Concepts/Architectures/Encoder-Decoder|Encoder-Decoder]] — BART генератор в RAG

## Дополнительные ресурсы

- [Lewis et al. — RAG for Knowledge-Intensive NLP Tasks (2020)](https://arxiv.org/abs/2005.11401) — оригинальная статья
- [Pinecone — RAG Guide](https://www.pinecone.io/learn/retrieval-augmented-generation/) — практическое руководство
- [Weaviate — Introduction to RAG](https://weaviate.io/blog/introduction-to-rag) — обзор с диаграммами
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/) — код-первый подход
