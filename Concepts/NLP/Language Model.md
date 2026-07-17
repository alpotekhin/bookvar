---
title: "Language Model"
aliases: [Language Model, LM, Языковая модель]
type: concept
category: NLP
papers: []
courses:
  - "[[02 Areas/ML & DL/Courses/Stanford CS224N/Lecture 05 — Language Models and RNNs|CS224N L05]]"
  - "[[02 Areas/ML & DL/Courses/Stanford CS224N/Lecture 09 — Pretraining|CS224N L09]]"
sources:
  - "[Jurafsky & Martin — Speech and Language Processing, Ch. 3 (N-gram LMs)](https://web.stanford.edu/~jurafsky/slp3/)"
  - "[Bengio et al. — A Neural Probabilistic Language Model (2003)](https://www.jmlr.org/papers/v3/bengio03a.html)"
  - "[Radford et al. — Language Models are Few-Shot Learners (GPT-3, 2020)](https://arxiv.org/abs/2005.14165)"
---

# Language Model — языковая модель

## Определение

**Языковая модель (LM)** — это вероятностное распределение над последовательностями токенов. Формально, для последовательности токенов $w_1, w_2, \ldots, w_T$ языковая модель задаёт:

$$P(w_1, w_2, \ldots, w_T)$$

По правилу произведения это раскладывается в цепочку условных вероятностей:

$$P(w_1, \ldots, w_T) = \prod_{t=1}^{T} P(w_t \mid w_1, \ldots, w_{t-1})$$

То есть задача LM — для любого контекста предсказать **распределение вероятностей следующего токена**. Если модель умеет это делать, она умеет:
- Оценивать «правдоподобность» текста (например, для ASR или MT)
- **Генерировать текст** через последовательное сэмплирование
- Служить основой для downstream-задач через transfer learning

## Зачем это нужно

Языковая модель — один из самых фундаментальных объектов в NLP. Её применения:

1. **Speech Recognition:** выбор между «I scream» и «ice cream» — акустика неоднозначна, LM выбирает более вероятный вариант
2. **Machine Translation:** оценка fluency кандидатов перевода
3. **Spelling correction:** «the teh car» → LM предпочитает «the car»
4. **Text generation:** чат-боты, автодополнение, creative writing
5. **Pretraining for transfer learning:** современная парадигма NLP — предобучить LM, потом fine-tune на задачу
6. **Few-shot learning:** достаточно большая LM ([[02 Areas/ML & DL/Concepts/Architectures/GPT-3|GPT-3]]) решает задачи по нескольким примерам в промпте без fine-tuning

## Оценка качества: perplexity

Стандартная метрика — [[02 Areas/ML & DL/Concepts/Evaluation/Perplexity|perplexity]]:

$$\text{PPL}(w_{1:T}) = P(w_1, \ldots, w_T)^{-1/T} = \exp\left(-\frac{1}{T}\sum_{t=1}^{T}\log P(w_t \mid w_{<t})\right)$$

Интуиция: «среднее количество равновероятных альтернатив», которое LM рассматривает на каждом шаге. Идеальная модель имеет PPL = 1, uniform distribution по словарю $V$ — PPL = $|V|$. Современные LM на общих корпусах — PPL 10-30 (word-level).

## Два основных типа

### Autoregressive (causal) LM

Классическое определение: предсказывать следующий токен по предыдущим.

$$P(w_t \mid w_1, \ldots, w_{t-1})$$

**Генерация:** слева направо, токен за токеном. **Обучение:** teacher forcing — на каждом шаге подаём правильный предыдущий токен.

**Архитектуры:**
- [[02 Areas/ML & DL/Concepts/NLP/N-gram|N-gram]] — статистическая AR-LM
- RNN/LSTM LM — neural AR-LM
- **GPT-family** ([[02 Areas/ML & DL/Concepts/Architectures/GPT-2|GPT-2]], [[02 Areas/ML & DL/Concepts/Architectures/GPT-3|GPT-3]], [[02 Areas/ML & DL/Concepts/Architectures/GPT-4|GPT-4]]) — transformer-based AR-LM с causal self-attention
- Все современные [[02 Areas/ML & DL/Concepts/Architectures/Decoder-only|Decoder-only]] модели (LLaMA, Mistral, Qwen, DeepSeek)

Преимущество: прямой генеративный интерфейс. Недостаток: каждый токен видит только прошлое, что ограничивает представления.

### Masked LM

Альтернатива: случайно маскировать часть токенов и предсказывать их, используя **двусторонний** контекст.

$$P(w_t \mid w_1, \ldots, w_{t-1}, w_{t+1}, \ldots, w_T)$$

**Архитектуры:**
- [[02 Areas/ML & DL/Concepts/Architectures/BERT|BERT]] и его потомки ([[02 Areas/ML & DL/Concepts/Architectures/RoBERTa|RoBERTa]], [[02 Areas/ML & DL/Concepts/Architectures/DeBERTa|DeBERTa]])
- Все [[02 Areas/ML & DL/Concepts/Architectures/Encoder-only|Encoder-only]] модели

Преимущество: представления видят весь контекст → лучше для understanding-задач (classification, NER, QA-extractive). Недостаток: не может нативно генерировать текст — MLM objective не даёт авторегрессии.

### Гибрид: prefix / seq2seq LM

T5, BART, UL2 комбинируют: encoder видит вход двунаправленно, decoder генерирует авторегрессивно. Строго говоря, это **условные** LM: $P(y \mid x)$, а не чистая LM $P(x)$.

## Эволюция языковых моделей

### 1. N-gram LMs (1948 — 2000s)

Марковское приближение: $P(w_t | w_{<t}) \approx P(w_t | w_{t-n+1}, \ldots, w_{t-1})$. Оценка — по частотам в корпусе + сглаживание (Laplace, Good-Turing, Kneser-Ney).

**Плюсы:** быстро, интерпретируемо, отлично работает на очень больших корпусах. **Минусы:** sparsity (большинство n-грамм не встречались), отсутствие обобщения на семантически близкие контексты.

См. [[02 Areas/ML & DL/Concepts/NLP/N-gram|N-gram]].

### 2. Neural Probabilistic LM (Bengio et al., 2003)

Первая нейронная LM: эмбеддинги слов + feed-forward сеть на фиксированное окно контекста. Решила sparsity через distributed representations — семантически близкие слова получают похожие эмбеддинги и модель обобщает.

### 3. RNN LM (Mikolov et al., 2010)

Рекуррентная сеть хранит скрытое состояние всего контекста: $h_t = f(h_{t-1}, x_t)$. В теории — **неограниченный контекст**, на практике ограничено vanishing gradients. LSTM/GRU смягчили проблему.

### 4. Transformer LM (2017 — настоящее время)

[[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] заменил рекуррентность на self-attention. Три волны:

- **2018:** ELMo (biLSTM), [[02 Areas/ML & DL/Concepts/Architectures/BERT|BERT]] (encoder), GPT-1 (decoder) — появление pretrain paradigm
- **2019-2020:** [[02 Areas/ML & DL/Concepts/Architectures/GPT-2|GPT-2]], [[02 Areas/ML & DL/Concepts/Architectures/GPT-3|GPT-3]] — демонстрация emergent abilities при масштабировании
- **2022-2026:** ChatGPT, LLaMA, Qwen, DeepSeek-R1 — instruction tuning, RLHF, reasoning, agents

Масштабы выросли от ~100M параметров (BERT-base) до ~1T+ (GPT-4, DeepSeek-V3). Контексты — от 512 токенов до 1M+.

## Скейлинг и emergent abilities

Ключевое открытие 2020-х: многие способности LLM возникают **скачкообразно** при достижении определённого масштаба (см. [[02 Areas/ML & DL/Concepts/NLP/Emergent Abilities|Emergent Abilities]]):
- In-context learning (GPT-3, 175B)
- [[02 Areas/ML & DL/Concepts/Inference/Chain of Thought|Chain of Thought]] reasoning (~60B+)
- Instruction following (после SFT на разнообразных задачах)
- Tool use, agentic behavior (после RLVR/GRPO)

Масштаб определяется эмпирическими **scaling laws** (Kaplan et al. 2020, Chinchilla 2022): оптимум по параметрам/данным/compute следует степенным зависимостям.

## Языковая модель как универсальный интерфейс

К 2023 году стало ясно: **многие задачи NLP можно свести к language modeling** через формулировку промпта. Классификация, QA, MT, summarization, code generation — всё выражается как «продолжи текст» для достаточно большой LM. Это radical simplification:

```
Before: separate model per task (NER-model, MT-model, QA-model, ...)
Now:    one LM + prompt engineering + (optional) fine-tuning
```

LM стала **операционной системой** для NLP.

## Современные тренды (2024-2026)

- **Long context:** 128K — 10M токенов (Gemini, Claude, Qwen)
- **Reasoning LM:** DeepSeek-R1, o1 — LM, специально обученные на chain-of-thought через RL
- **Multimodal LM:** GPT-4V, Gemini — объединение текста с изображениями/аудио/видео
- **Mixture of Experts:** Mixtral, DeepSeek-V3 — разреженные LM с триллионами параметров
- **Agents:** LM + tools + планирование → автономные системы

## Related concepts

- [[02 Areas/ML & DL/Concepts/NLP/N-gram|N-gram]] — статистический предшественник
- [[02 Areas/ML & DL/Concepts/Evaluation/Perplexity|Perplexity]] — стандартная метрика оценки
- [[02 Areas/ML & DL/Concepts/Architectures/GPT-3|GPT-3]] — канонический пример масштабированной AR-LM
- [[02 Areas/ML & DL/Concepts/Architectures/BERT|BERT]] — канонический пример MLM
- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] — архитектурная основа современных LM
- [[02 Areas/ML & DL/Concepts/Architectures/Decoder-only|Decoder-only]] — доминирующий тип архитектуры для AR-LM
- [[02 Areas/ML & DL/Concepts/NLP/Emergent Abilities|Emergent Abilities]] — феномен скачкообразного появления способностей
- [[02 Areas/ML & DL/Concepts/Inference/Chain of Thought|Chain of Thought]] — способ использования LM для reasoning
