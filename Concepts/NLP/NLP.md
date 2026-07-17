---
title: "Natural Language Processing (NLP)"
aliases: [NLP, Natural Language Processing, обработка естественного языка, NLU, NLG, Computational Linguistics]
type: concept
status: legacy
category: NLP
papers:
  - "[[02 Areas/ML & DL/Papers/Attention Is All You Need|Attention Is All You Need]]"
  - "[[02 Areas/ML & DL/Papers/BERT]]"
  - "[[02 Areas/ML & DL/Papers/GPT 3.0]]"
courses:
  - "[[02 Areas/ML & DL/Courses/Stanford CS224N/07 — Attention|CS224N Lecture 7]]"
  - "[[02 Areas/ML & DL/Courses/Stanford CS224N/08 — Transformers|CS224N Lecture 8]]"
  - "[[02 Areas/ML & DL/Courses/SHAD LLM/Week 1 — Intro to LLMs|SHAD LLM — Week 1]]"
sources:
  - "[Stanford CS224N — Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/)"
  - "[Jurafsky & Martin — Speech and Language Processing (3rd ed.)](https://web.stanford.edu/~jurafsky/slp3/)"
  - "[Lena Voita — NLP Course for You](https://lena-voita.github.io/nlp_course.html)"
---

# Natural Language Processing (NLP)

## Что такое NLP

Natural Language Processing (NLP) --- область на пересечении компьютерных наук, лингвистики и машинного обучения, занимающаяся **взаимодействием между компьютерами и человеческим языком**. Цель --- научить машины понимать, интерпретировать и генерировать текст на естественном языке.

NLP делится на два больших направления:
- **NLU (Natural Language Understanding)** --- понимание: классификация, извлечение информации, ответы на вопросы, определение тональности
- **NLG (Natural Language Generation)** --- генерация: машинный перевод, summarization, диалог, написание текстов

В эпоху LLM граница между NLU и NLG размывается: decoder-only модели (GPT, LLaMA) решают обе задачи через генерацию.

## Основные задачи NLP

### Задачи понимания (NLU)

| Задача | Описание | Пример | Benchmark |
|--------|----------|--------|-----------|
| **Text Classification** | Присвоить тексту метку | Отзыв -> positive/negative | SST-2, IMDB |
| **NER** (Named Entity Recognition) | Найти именованные сущности | "Apple was founded in Cupertino" -> [Apple: ORG, Cupertino: LOC] | CoNLL-2003 |
| **NLI** (Natural Language Inference) | Отношение между предложениями | premise + hypothesis -> entailment/contradiction/neutral | MNLI, SNLI |
| **Sentiment Analysis** | Определение тональности | "Great movie!" -> positive | SST, Yelp |
| **QA** (Question Answering) | Ответ на вопрос по тексту | Passage + question -> answer span | SQuAD |
| **Coreference Resolution** | Разрешение ссылок | "She picked up the ball. **It** was red." -> It = ball | WSC |
| **POS Tagging** | Части речи | "The cat sat" -> DET NOUN VERB | Penn Treebank |
| **Parsing** | Синтаксический разбор | Дерево зависимостей | UD Treebanks |

### Задачи генерации (NLG)

| Задача | Описание | Пример |
|--------|----------|--------|
| **Machine Translation** | Перевод между языками | "Hello" -> "Привет" |
| **Summarization** | Сжатие текста | Статья -> 3 предложения |
| **Dialogue** | Ведение диалога | Chatbot, QA system |
| **Text Generation** | Генерация текста | Продолжение текста, написание кода |
| **Paraphrase** | Перефразирование | "It's raining" -> "The weather is wet" |

## Эволюция NLP: пять эр

### Эра 1: Rule-based (1950--1990)

Первые NLP-системы --- вручную написанные правила и грамматики.

- **ELIZA** (Weizenbaum, 1966) --- первый чат-бот, pattern matching
- **SHRDLU** (Winograd, 1972) --- понимание команд в ограниченном мире кубиков
- Формальные грамматики (CFG, HPSG) для синтаксического разбора
- Экспертные системы для machine translation

**Проблема:** правила не масштабируются. Язык слишком неоднозначен, исключений больше, чем правил.

### Эра 2: Statistical NLP (1990--2013)

Переход от правил к **статистическим моделям**, обученным на данных.

- **N-gram Language Models** --- $P(w_t | w_{t-1}, \ldots, w_{t-n+1})$
- **Hidden Markov Models (HMM)** для POS-tagging и NER
- **Conditional Random Fields (CRF)** --- улучшение HMM для sequence labeling
- **Bag-of-Words + TF-IDF + SVM/Naive Bayes** для classification
- **Statistical Machine Translation (SMT)** --- IBM Models, phrase-based MT (Moses)
- **WordNet, FrameNet** --- лингвистические ресурсы

**Прорыв:** данные > правила. Но модели всё ещё требуют ручной feature engineering.

### Эра 3: Neural NLP (2013--2018)

Нейронные сети заменяют ручные признаки **обученными представлениями**.

| Год | Прорыв | Значение |
|-----|--------|----------|
| 2013 | **Word2Vec** (Mikolov et al.) | Плотные эмбеддинги слов; king - man + woman = queen |
| 2014 | **GloVe** (Pennington et al.) | Матричная факторизация co-occurrence |
| 2014 | **Seq2Seq** (Sutskever et al.) | Encoder-decoder RNN для translation |
| 2014 | **Attention** (Bahdanau et al.) | Декодер «смотрит» на энкодер |
| 2015 | **BiLSTM + CRF** (Lample et al.) | SOTA для NER |
| 2018 | **ELMo** (Peters et al.) | Контекстуальные эмбеддинги из BiLSTM |

Ключевые архитектуры: [[02 Areas/ML & DL/Concepts/NLP/RNN/RNN|RNN]], LSTM, GRU, BiLSTM, CNN для текста.

**Главная проблема:** RNN последовательны ($O(n)$ шагов) и плохо обучаются на длинных зависимостях.

### Эра 4: Pre-train + Fine-tune / Transformer (2017--2022)

[[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] и парадигма «pre-train на огромном корпусе, fine-tune на задачу».

| Год | Прорыв | Значение |
|-----|--------|----------|
| **2017** | **Transformer** (Vaswani et al.) | Self-attention вместо RNN; параллелизм |
| **2018** | **BERT** (Devlin et al.) | Bidirectional pre-training + fine-tuning; GLUE 80.5 |
| 2018 | **GPT** (Radford et al.) | Unidirectional pre-training + fine-tuning |
| 2019 | **RoBERTa, XLNet, ALBERT** | Оптимизация рецепта обучения BERT |
| 2019 | **T5** (Raffel et al.) | Text-to-text: все задачи как генерация |
| 2020 | **GPT-3** (Brown et al.) | 175B параметров; in-context learning |

Парадигма [[02 Areas/ML & DL/Concepts/NLP/Transfer Learning|Transfer Learning]]: модель один раз предобучается на огромном корпусе (Wikipedia, Common Crawl), затем дообучается на конкретную задачу с минимумом данных.

### Эра 5: LLM / ChatGPT (2022+)

Масштабирование + alignment = AI-ассистенты для массового использования.

| Год | Прорыв | Значение |
|-----|--------|----------|
| **2022** | **ChatGPT** (OpenAI) | RLHF + диалог; 100M пользователей за 2 месяца |
| 2023 | **GPT-4** (OpenAI) | Мультимодальность, reasoning |
| 2023 | **LLaMA** (Meta) | Open-source LLM, демократизация |
| 2023 | **Claude** (Anthropic) | Constitutional AI, safety |
| 2024 | **o1, o3** (OpenAI) | Chain-of-thought reasoning |

Ключевой сдвиг: от «одна модель на задачу» к «одна модель на всё через промпты». [[02 Areas/ML & DL/Concepts/NLP/Prompt-based Learning|Prompt-based Learning]] заменяет fine-tuning для многих задач.

## Уровни обработки языка

NLP можно разложить по уровням лингвистического анализа:

| Уровень | Задача NLP | Пример |
|---------|-----------|--------|
| **Фонетика/фонология** | Speech recognition | Звук -> текст |
| **Морфология** | Tokenization, stemming | "running" -> "run" + "-ing" |
| **Синтаксис** | Parsing, POS-tagging | Дерево зависимостей |
| **Семантика** | NER, SRL, WSD | Значение слов и предложений |
| **Дискурс** | Coreference, coherence | Связи между предложениями |
| **Прагматика** | Sentiment, intent | Намерение говорящего |

Современные LLM работают **сразу на всех уровнях** --- end-to-end, без явного разбиения на стадии.

## Ключевые концепции NLP (навигация)

### Представления текста
- [[02 Areas/ML & DL/Concepts/NLP/Tokenization|Tokenization]] --- разбиение текста на токены (BPE, WordPiece)
- [[02 Areas/ML & DL/Concepts/NLP/Positional Encoding|Positional Encoding]] --- кодирование позиции в последовательности

### Архитектуры
- [[02 Areas/ML & DL/Concepts/NLP/RNN/RNN|RNN / LSTM / GRU]] --- рекуррентные сети (2013--2017)
- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] --- доминирующая архитектура (2017+)
- [[02 Areas/ML & DL/Concepts/Architectures/BERT|BERT]] --- encoder-only, NLU
- [[02 Areas/ML & DL/Concepts/Architectures/GPT-3|GPT-3]] --- decoder-only, генерация

### Механизмы
- [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]] --- основной механизм Transformer
- [[02 Areas/ML & DL/Concepts/NLP/Self-Attention|Self-Attention]] --- внимание внутри последовательности
- [[02 Areas/ML & DL/Concepts/NLP/Feed-Forward Network|Feed-Forward Network]] --- FFN в Transformer

### Парадигмы обучения
- [[02 Areas/ML & DL/Concepts/NLP/Transfer Learning|Transfer Learning]] --- pre-train + fine-tune
- [[02 Areas/ML & DL/Concepts/NLP/Prompt-based Learning|Prompt-based Learning]] --- pre-train + prompt + predict
- [[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]] --- alignment через human feedback

### Оценка
- [[02 Areas/ML & DL/Concepts/NLP/GLUE|GLUE / SuperGLUE]] --- стандартные benchmarks NLU
- [[02 Areas/ML & DL/Concepts/NLP/Text Classification|Text Classification]] --- фундаментальная задача-benchmark

### Безопасность
- [[02 Areas/ML & DL/Concepts/NLP/Adversarial Promting|Adversarial Prompting]] --- атаки на LLM
- [[02 Areas/ML & DL/Concepts/NLP/Emergent Abilities|Emergent Abilities]] --- способности, возникающие при масштабировании

## Дополнительные ресурсы

- [Stanford CS224N — NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) --- лучший университетский курс по NLP
- [Jurafsky & Martin — Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) --- каноничный учебник (free online)
- [Lena Voita — NLP Course for You](https://lena-voita.github.io/nlp_course.html) --- отличный визуальный курс
- [Hugging Face NLP Course](https://huggingface.co/course) --- практический курс с кодом
- [d2l.ai — NLP chapters](https://d2l.ai/) --- учебник с PyTorch/TensorFlow реализациями
