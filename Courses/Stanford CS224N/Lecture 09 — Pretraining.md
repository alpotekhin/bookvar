---
title: "CS224N — Lecture 9: Pretraining"
course: "Stanford CS224N"
lecture: 9
type: course-note
raw: "[[02 Areas/ML & DL/raw/courses/Stanford CS224N/slides/cs224n-spr2024-lecture09-pretraining-updated]]"
concepts: ["[[02 Areas/ML & DL/Concepts/Training/Pre-training|Pretraining]]", "[[02 Areas/ML & DL/Concepts/Architectures/BERT|BERT]]", "[[02 Areas/ML & DL/Concepts/NLP/GPT|GPT]]", "[[02 Areas/ML & DL/Concepts/NLP/Subword Tokenization|Subword Tokenization]]", "[[02 Areas/ML & DL/Concepts/NLP/ELMo|ELMo]]"]
---

# Lecture 9: Pretraining

## Эволюция: от word embeddings к pretraining всей модели

### Circa 2015: pretrained word embeddings

Подход: [[02 Areas/ML & DL/Concepts/NLP/Word2Vec|Word2Vec]] / GloVe дают фиксированные вектора слов. Всё остальное (верхние слои сети) инициализируется **случайно** и обучается на downstream task.

Проблема: большинство параметров случайные. Downstream dataset должен обучить всё понимание контекста с нуля. Одно и то же слово имеет одинаковый вектор вне зависимости от контекста ("bank" = берег или банк?).

### Circa 2017-2018: контекстуализированные embeddings

**TagLM** (Peters et al., 2017): добавить hidden states из pretrained BiLSTM LM как дополнительные фичи. Первый шаг к контексту.

**[[02 Areas/ML & DL/Concepts/NLP/ELMo|ELMo]]** (Peters et al., 2018): **E**mbeddings from **L**anguage **Mo**dels.
- 2-layer BiLSTM, обученный на language modeling (forward + backward)
- Для каждого слова: **конкатенация скрытых состояний** всех слоёв, взвешенная task-specific весами
- Ключевая инновация: representation слова **зависит от контекста**
- Результат: +3-5% на многих NLP tasks (NER, SRL, coreference, QA)

Вес каждого слоя обучается per-task: разные задачи используют разные слои (syntax vs semantics).

### Circa 2018+: pretraining всей модели

**GPT-1** (Radford et al., 2018): Transformer decoder, pretrained на language modeling, fine-tuned на downstream tasks. Все параметры pretrained.

**[[02 Areas/ML & DL/Concepts/Architectures/BERT|BERT]]** (Devlin et al., 2019): Transformer encoder, pretrained на Masked LM + Next Sentence Prediction. Bidirectional context.

Революция: модель **уже "знает" язык** после pretraining. Fine-tuning -- лишь адаптация к конкретной задаче с минимальным количеством данных.

## [[02 Areas/ML & DL/Concepts/NLP/Subword Tokenization|Subword Tokenization]]

### Проблема словаря

Word-level tokenization: фиксированный словарь 50-100K слов. Любое новое/редкое слово = `<UNK>`. Character-level: слишком длинные последовательности, теряет word-level семантику.

### Byte-Pair Encoding (BPE)

Алгоритм:
1. Начинаем со словаря из отдельных символов
2. Итеративно объединяем самую **частую** пару символов в новый токен
3. Повторяем до достижения нужного размера словаря

Результат:
- Частые слова -- целиком в словаре ("the", "running")
- Редкие слова -- разбиваются на подслова ("unforgettable" -> "un" + "forget" + "table")
- Нет `<UNK>` -- любое слово можно представить

Варианты: **WordPiece** (BERT), **SentencePiece** (T5, LLaMA) -- unigram model вместо greedy merge. Современные модели: ~32K-100K subword vocabulary.

## Три парадигмы pretraining

### 1. Encoder-only: [[02 Areas/ML & DL/Concepts/Architectures/BERT|BERT]]

**Masked Language Modeling (MLM)**: маскируем 15% токенов, модель предсказывает замаскированные по **bidirectional** контексту.

Стратегия маскирования (80/10/10):
- 80% -> `[MASK]`
- 10% -> случайный токен
- 10% -> оставляем без изменения

Зачем 10/10? При fine-tuning нет `[MASK]` токенов -- нужно, чтобы модель привыкла предсказывать и обычные токены.

**Next Sentence Prediction (NSP)**: бинарная классификация -- следует ли предложение B за предложением A. Оказалось **не очень полезным** (RoBERTa убрала NSP и стала лучше).

**Fine-tuning BERT**: добавить classification head поверх `[CLS]` токена для classification; поверх каждого токена для sequence labeling (NER).

Хорош для: classification, NER, QA (extractive), sentence similarity.

### 2. Decoder-only: [[02 Areas/ML & DL/Concepts/NLP/GPT|GPT]]

**Autoregressive Language Modeling**: предсказывает следующий токен. **Unidirectional** context (masked self-attention -- видит только предыдущие позиции).

Хорош для: text generation, dialogue, code generation, few-shot learning (GPT-3).

GPT -> GPT-2 (1.5B) -> GPT-3 (175B): каждое увеличение размера открывает **новые способности** ([[02 Areas/ML & DL/Concepts/NLP/Emergent Abilities|Emergent Abilities]]).

### 3. Encoder-Decoder: T5

**Span Corruption**: маскируем случайные **spans** (не отдельные токены), модель генерирует замаскированный текст.

$$\text{Input: Thank you } \langle X \rangle \text{ me to your party } \langle Y \rangle \text{ week}$$
$$\text{Output: } \langle X \rangle \text{ for inviting } \langle Y \rangle \text{ last }$$

Хорош для: summarization, translation, Q&A (generative), text-to-text tasks.

**T5 philosophy** (Raffel et al., 2020): **все задачи** как text-to-text. Единый формат: "translate English to French: ...", "summarize: ...", "sentiment: ...".

### Сравнение парадигм

| | Encoder-only (BERT) | Decoder-only (GPT) | Encoder-Decoder (T5) |
|---|---|---|---|
| Pretraining | MLM (bidirectional) | Autoregressive LM (left-to-right) | Span corruption |
| Context | Full bidirectional | Causal (left only) | Encoder: bidirectional, Decoder: causal |
| Best for | Classification, NER, extractive QA | Generation, few-shot, dialogue | Seq2seq (translation, summarization) |
| Examples | BERT, RoBERTa, DeBERTa | GPT-2/3/4, LLaMA, Claude | T5, FLAN-T5, UL2 |
| Доминирует в 2024+ | Нет | **Да** | Частично (FLAN) |

### Почему decoder-only победил?

1. **Масштабируемость**: autoregressive LM -- простой, унифицированный objective
2. **Emergent abilities**: few-shot, chain-of-thought, tool use -- появляются при масштабировании decoder-only
3. **Одна модель для всего**: не нужны task-specific heads, всё через generation
4. **Данные**: web crawl -> language modeling, не нужна разметка

## Детали BERT

### Архитектура

| Параметр | BERT-Base | BERT-Large |
|----------|-----------|------------|
| Layers | 12 | 24 |
| Hidden size | 768 | 1024 |
| Attention heads | 12 | 16 |
| Parameters | 110M | 340M |

Pretraining data: BooksCorpus (800M слов) + English Wikipedia (2,500M слов).

### Fine-tuning BERT

Для каждой задачи добавляем minimal task-specific layer:

- **Classification** (sentiment, NLI): `[CLS]` token -> linear layer -> softmax
- **Token-level** (NER, POS): каждый token -> linear layer -> tag
- **QA** (extractive): для каждого токена предсказываем P(start) и P(end) -- span extraction
- **Sentence pair** (NLI, paraphrase): `[CLS] sent1 [SEP] sent2 [SEP]` -> classification

**Ключевой момент**: при fine-tuning обновляются **все** параметры BERT (не только classification head). Это работает при маленьких dataset (даже 5K примеров), потому что модель уже "знает" язык.

### RoBERTa: Robustly Optimized BERT Approach

Liu et al. (2019) показали, что BERT был **undertrained**:
- Убрали NSP (не помогает)
- Больше данных (160GB vs 16GB)
- Longer training (500K vs 1M steps, больше compute)
- Dynamic masking (пересэмплируем маски каждую эпоху)
- Bigger batches (8K vs 256)

Результат: значительное улучшение на всех бенчмарках при **той же архитектуре**.

## Детали GPT

### Масштабирование от GPT к GPT-3

| Модель | Год | Params | Training Data | Context |
|--------|-----|--------|--------------|---------|
| GPT-1 | 2018 | 117M | BooksCorpus (5GB) | 512 |
| GPT-2 | 2019 | 1.5B | WebText (40GB) | 1024 |
| GPT-3 | 2020 | 175B | Common Crawl + books + Wiki (570GB filtered) | 2048 |

GPT-2 key insight: zero-shot task transfer. Модель, обученная только на web text, может:
- Суммаризировать (добавь "TL;DR:" после текста)
- Переводить (формат "English sentence = French sentence")
- Отвечать на вопросы (формат "Q: ... A: ...")

**Без единого примера задачи** -- просто продолжает текст в правильном формате.

GPT-3 key insight: few-shot in-context learning. С несколькими примерами в промпте модель "понимает" формат и решает задачу. Это привело к [[02 Areas/ML & DL/Concepts/Inference/In-context Learning|In-Context Learning]] как новой парадигме.

## Что выучивает pretraining?

Manning показывает примеры probing experiments:

- **Синтаксис**: "I put ___ fork down on the table" -> "the" / "a" (определённый/неопределённый артикль)
- **Coreference**: "The woman walked across the street, checking for traffic over ___ shoulder" -> "her"
- **Семантика/topic**: "Overall, the value of exercise for human health is incredible, and regular exercise is good for your heart, muscles, joints, bones, fish, turtles, seals, and ___" -> модель скатывается к topic "marine life" из-за последних слов
- **Sentiment**: "The movie was ___" -> "good" / "great" / "terrible" (вероятности зависят от предшествующего контекста)
- **World knowledge**: "Stanford University is located in ___, California" -> "Stanford" / "Palo Alto"

### Probing experiments: что хранится в каких слоях?

Исследования (Tenney et al., 2019; Hewitt & Manning, 2019) показывают:
- **Нижние слои** (1-4): морфология, POS-теги, local syntax
- **Средние слои** (5-8): syntax trees, dependency relations
- **Верхние слои** (9-12): семантика, coreference, world knowledge

BERT выучивает **иерархическую структуру языка** через self-supervised pretraining.

## Хронология pretraining

| Год | Milestone |
|-----|-----------|
| 2013 | Word2Vec -- first practical word embeddings |
| 2014 | GloVe -- global vectors from co-occurrence |
| 2018 | ELMo -- contextualized embeddings (BiLSTM) |
| 2018 | GPT-1, BERT -- Transformer-based pretraining |
| 2019 | GPT-2, RoBERTa, XLNet, ALBERT |
| 2020 | GPT-3, T5 -- scaling + few-shot learning |
| 2022 | ChatGPT -- pretraining + SFT + RLHF |
| 2023 | GPT-4, LLaMA, Claude -- frontier models |

## Concepts covered

- [[02 Areas/ML & DL/Concepts/Training/Pre-training|Pretraining]] -- предобучение на огромных неразмеченных данных
- [[02 Areas/ML & DL/Concepts/Architectures/BERT|BERT]] -- bidirectional encoder, MLM + NSP
- [[02 Areas/ML & DL/Concepts/NLP/GPT|GPT]] -- autoregressive decoder, basis for ChatGPT
- [[02 Areas/ML & DL/Concepts/NLP/Subword Tokenization|Subword Tokenization]] -- BPE, WordPiece, SentencePiece
- [[02 Areas/ML & DL/Concepts/NLP/ELMo|ELMo]] -- contextualized embeddings на BiLSTM (предшественник BERT)
