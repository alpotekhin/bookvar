---
title: "REALM: Retrieval-Augmented Language Model Pre-Training"
url: https://arxiv.org/abs/2002.08909
authors: [Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, Ming-Wei Chang]
year: 2020
date_reviewed: 2026-04-07
type: source-note
status: legacy
category: paper
tags:
  - RAG
  - retrieval
  - pre-training
  - open-QA
Organization: Google Research
concepts:
  - "[[02 Areas/ML & DL/Concepts/Retrieval/Retrieval-Augmented Generation|RAG]]"
  - "[[02 Areas/ML & DL/Concepts/Retrieval/Dense Retrieval|Dense Retrieval]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Masked Language Modeling|MLM]]"
raw: "[[02 Areas/ML & DL/raw/papers/realm/paper.txt]]"
---

# REALM: Retrieval-Augmented Language Model Pre-Training

**Authors:** Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, Ming-Wei Chang (Google Research)
**Published:** 2020 (arXiv:2002.08909, ICML 2020)
**URL:** https://arxiv.org/abs/2002.08909

## TL;DR

**REALM** (Retrieval-Augmented Language Model) — framework, который добавляет обучаемый neural retriever к language model pre-training. Модель учится retrieve документы из Wikipedia и использовать их для предсказания замаскированных токенов. Retriever обучается end-to-end через backpropagation сигнала от MLM loss через retrieval step. На Open-domain QA: **4-16% absolute improvement** над всеми предшествующими системами (включая T5-11B, который в 30x больше), на NaturalQuestions-Open, WebQuestions, CuratedTrec.

## Problem

Language model pre-training (BERT, T5) хранит мировое знание **имплицитно** в параметрах нейросети:
- Невозможно определить, какое знание хранится и где
- Для покрытия большего числа фактов нужно увеличивать модель (T5 base -> large -> 11B)
- Экспоненциальный рост parameters vs линейный прирост knowledge
- Нет модульности и интерпретируемости

Предшествующие retrieval-based системы:
- Используют heuristic retrieval (TF-IDF, BM25, entity linking) — не обучаемый
- Или learned retrieval, но не применяют его к pre-training (ORQA: learned retrieval только при fine-tuning, fixed MIPS index)
- kNN-LM: retrieval similar LM examples, но не адаптируется к downstream tasks

## Method

### Generative Process (Section 3.1)

REALM декомпозирует p(y|x) на два шага:

```
p(y|x) = sum_{z in Z} p(y|z,x) * p(z|x)
```

где z — документ из knowledge corpus Z (Wikipedia), x — input (masked sentence или question), y — output.

### Knowledge Retriever: p(z|x) (Section 3.2)

Dense inner product model:

```
f(x,z) = Embed_input(x)^T * Embed_doc(z)
p(z|x) = softmax(f(x,z)) over all z in Z
```

Embedding functions реализованы через BERT:
- Embed_input(x) = W_input * BERT_CLS(join_BERT(x))
- Embed_doc(z) = W_doc * BERT_CLS(join_BERT(z_title, z_body))

Линейная проекция W для снижения размерности. Retrieval distribution — softmax по relevance scores.

### Knowledge-Augmented Encoder: p(y|z,x) (Section 3.2)

Отдельный Transformer (не тот, что в retriever), который получает конкатенацию x и z_body.

**Pre-training (MLM):**
```
p(y_j|z,x) ~ exp(w_j^T * BERT_MASK(j)(join_BERT(x, z_body)))
```

**Fine-tuning (Open-QA):** extractive — ответ y как span в документе z:
```
p(y|z,x) ~ sum_{s in S(z,y)} exp(MLP([h_START(s); h_END(s)]))
```

### Training: Marginal Likelihood (Section 3.3)

Maximize log p(y|x) = log sum_{z in Z} p(y|z,x) * p(z|x).

**Computational challenge:** суммирование по всем документам в Z (~13M).
- Approximation: суммирование только по top-k документов (k=8 при pre-training, k=5 при fine-tuning)
- Top-k retrieval через Maximum Inner Product Search (MIPS) — sub-linear по числу документов

### Asynchronous MIPS Index Refresh (Section 3.3)

Поскольку Embed_doc обновляется при каждом gradient step, MIPS index устаревает. Решение:
1. **Primary trainer job**: выполняет gradient updates на параметрах
2. **Secondary index builder job**: получает snapshot параметров theta', пересчитывает все document embeddings, перестраивает MIPS index
3. Refresh каждые ~500 training steps
4. После retrieval top-k по stale index, p(z|x) пересчитывается с fresh theta

При fine-tuning: MIPS index строится один раз (frozen Embed_doc), обновляется только Embed_input.

### Gradient Analysis: What Does the Retriever Learn?

```
grad log p(y|x) = sum_z r(z) * grad f(x,z)
r(z) = (p(y|z,x) / p(y|x) - 1) * p(z|x)
```

Документ z получает положительный update если p(y|z,x) > p(y|x) — т.е. если z помогает предсказать y лучше, чем среднее по всем документам. Retriever учится reward-ить информативные документы.

### Inductive Biases (Section 3.4)

1. **Salient span masking**: маскируются named entities и даты (CoNLL tagger + regex), а не случайные токены. Фокус на world knowledge, а не syntax.
2. **Null document**: пустой документ в top-k — для случаев, когда retrieval не нужен.
3. **Prohibit trivial retrievals**: если x из документа z, этот z исключается (иначе retriever учится string matching).
4. **Warm-start**: Embed_input и Embed_doc инициализируются через Inverse Cloze Task (ICT). Encoder — через BERT-base uncased.

## Key Results

### Open-QA Benchmarks (Table 1)

| System | Architecture | NQ (79K/4K) | WQ (3K/2K) | CT (1K/1K) | Params |
|--------|-------------|-------------|------------|------------|--------|
| BERT-Baseline | Sparse Retr.+Transformer | 26.5 | 17.7 | 21.3 | 110M |
| ORQA | Dense Retr.+Transformer | 33.3 | 36.4 | 30.1 | 330M |
| T5-11B | Transformer Seq2Seq | 34.5 | 37.4 | - | 11,318M |
| **REALM (Wiki, Wiki)** | Dense Retr.+Transformer | **39.2** | **40.2** | **46.8** | **330M** |
| **REALM (CC-News, Wiki)** | Dense Retr.+Transformer | **40.4** | **40.7** | **42.9** | **330M** |

REALM превосходит T5-11B (в 30x больше) по exact match на NQ и WQ. На CuratedTrec: +16.7 absolute points vs ORQA.

### vs ORQA

REALM vs ORQA — identical setup, одинаковые hyperparameters, одинаковые training data. Разница только в pre-training method. REALM: +5.9 на NQ, +3.8 на WQ, +16.7 на CT. Чисто от улучшения pre-training retriever + encoder.

### Ablation (Table 2, NQ Dev)

| Ablation | EM | Retrieval Recall@5 |
|----------|----|--------------------|
| REALM | 38.2 | 38.5 |
| REALM retriever + Baseline encoder | 37.4 | 38.5 |
| Baseline retriever + REALM encoder | 35.3 | 13.9 |
| Baseline (ORQA) | 31.3 | 13.9 |
| Random uniform masks (вместо salient span) | 32.3 | 24.2 |
| Random span masks | 35.3 | 26.1 |
| 30x stale MIPS | 28.7 | 15.1 |

- Оба компонента (retriever + encoder) выигрывают от REALM pre-training
- Salient span masking критичен: +5.9 EM vs random token masking
- Stale MIPS (редкий refresh) сильно вредит: -9.5 EM

### Qualitative (Table 3)

Пример: "An equilateral triangle ... because 3 is a ___ prime"
- BERT: p("Fermat") = 1.1e-14
- REALM (с документом "257 is... a Fermat prime..."): p("Fermat"|z) = 1.0
- REALM (marginal): p("Fermat") = 0.129

## My notes

- REALM — один из первых работ, показавших, что learned retrieval + pre-training дают синергию. Прямой предшественник RAG (Lewis et al., 2020), RETRO, Atlas.
- Ключевой insight: обучение retriever через MLM signal — unsupervised. Не нужны пары query-document с разметкой. Retriever учится из "что помогает лучше предсказать masked token".
- Asynchronous MIPS refresh — инженерно тяжелое решение, но необходимое. Без refresh retriever не учится. Это ограничивает масштабируемость pre-training.
- Salient span masking — простой, но мощный inductive bias. Случайное маскирование (BERT-style) не создает достаточного давления на retriever, т.к. большинство masked tokens можно предсказать из локального контекста.
- Ограничения: (1) pre-training на 64 TPUs — дорого; (2) при fine-tuning MIPS index не обновляется (frozen Embed_doc) — потенциально субоптимально; (3) только extractive QA, нет generation.
- REALM на 330M params бьет T5-11B — сильный аргумент за retrieval-augmented подход vs scaling parameters. Знание лучше хранить эксплицитно в корпусе, а не имплицитно в весах.
