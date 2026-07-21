---
title: "Causal Language Modeling"
aliases: [CLM, autoregressive language modeling, autoregressive LM, авторегрессивное языковое моделирование]
type: concept
status: legacy
category: Training
papers:
  - "[[02 Areas/ML & DL/Papers/GPT 2.0]]"
  - "[[02 Areas/ML & DL/Papers/GPT 3.0]]"
  - "[[02 Areas/ML & DL/Papers/LLaMA]]"
  - "[[02 Areas/ML & DL/Papers/OPT]]"
  - "[[02 Areas/ML & DL/Papers/Formal Algorithms for Transformers]]"
courses:
  - "[[02 Areas/ML & DL/Courses/SHAD LLM/Week 1 -- Intro to LLMs|SHAD LLM -- Week 1]]"
sources:
  - "[Radford et al. -- Language Models are Unsupervised Multitask Learners (GPT-2, 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)"
  - "[Brown et al. -- Language Models are Few-Shot Learners (GPT-3, 2020)](https://arxiv.org/abs/2005.14165)"
  - "[Jay Alammar -- The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)"
---

# Causal Language Modeling (CLM)

## Зачем это нужно: простейший и мощнейший objective

Causal Language Modeling -- это **предсказание следующего токена** по всем предыдущим. Формально:

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t \mid x_1, x_2, \ldots, x_{t-1}; \theta)$$

Это самый старый objective в NLP -- от n-gram моделей (1950-е) до GPT-4. Идея тривиальна: дай модели начало текста, она предсказывает продолжение. Но именно эта простота позволила **масштабировать** CLM до сотен миллиардов параметров и триллионов токенов обучения, получив emergent abilities: reasoning, code generation, few-shot learning.

Название **"causal"** отсылает к причинности: токен $x_t$ зависит только от **прошлых** токенов $x_{<t}$, а не от будущих -- как в причинно-следственной цепочке.

## Как работает CLM

### Causal Mask: запрет подглядывания в будущее

В self-attention каждый токен "смотрит" на все остальные. Для CLM это нужно **ограничить**: токен на позиции $t$ должен видеть только позиции $1, 2, \ldots, t$.

Реализуется через **causal mask** -- нижнетреугольная матрица:

$$M_{ij} = \begin{cases} 0 & i \geq j \\ -\infty & i < j \end{cases}$$

Нулевое значение сохраняет связь с текущей или предыдущей позицией, а
$-\infty$ обнуляет вероятность связи с будущей позицией после softmax.

```
Attention matrix (до softmax) + causal mask:

       x1    x2    x3    x4
x1  [ a11  -inf  -inf  -inf ]
x2  [ a21   a22  -inf  -inf ]
x3  [ a31   a32   a33  -inf ]
x4  [ a41   a42   a43   a44 ]

После softmax: -inf --> 0, модель игнорирует будущие позиции
```

Из [[02 Areas/ML & DL/Papers/Formal Algorithms for Transformers]] Algorithm 10: это единственное отличие decoder-only Transformer от encoder-only. Вся остальная архитектура (multi-head attention, FFN, LayerNorm, residual connections) -- идентична.

### Teacher Forcing: параллельное обучение

При обучении CLM используется **teacher forcing**: вместо того чтобы генерировать токены один за другим (авторегрессивно), модель получает **всю последовательность сразу** и предсказывает каждый следующий токен параллельно:

```
Вход:      [BOS]  The   quick  brown  fox
Target:     The   quick brown  fox    jumps

         |      |      |      |      |
         v      v      v      v      v
Loss:  L(The) L(quick) L(brown) L(fox) L(jumps)
```

Causal mask гарантирует, что при предсказании "brown" модель видит только "[BOS] The quick", а не "fox" -- даже если все токены обрабатываются параллельно. Это **максимально эффективно** по GPU utilization.

**Важно**: loss вычисляется по **каждому токену**, а не по 15% как в MLM. Это даёт CLM ~6.7x больше обучающего сигнала на одну последовательность.

### Авторегрессивная генерация (inference)

При inference модель генерирует **по одному токену за раз**:

```
для каждого шага t:
  logits = model(x[1:t-1])
  x[t] ~ sampling(logits)     # top-k, top-p, temperature
  append x[t] to context
  if x[t] == EOS: stop
```

Это **последовательный** процесс -- основной bottleneck inference. Каждый новый токен требует нового forward pass через всю модель (оптимизируется через [[02 Areas/ML & DL/Concepts/Inference/KV-Cache|KV-Cache]]).

### Архитектура decoder-only Transformer

```
Input Embeddings + Positional Encoding
         |
         v
   +-----------+
   | LayerNorm |  (pre-norm, GPT-2+)
   +-----------+
         |
   +------------------+
   | Masked Multi-Head |
   | Self-Attention     |  <-- causal mask
   +------------------+
         |
   + Residual Connection
         |
   +-----------+
   | LayerNorm |
   +-----------+
         |
   +-----+
   | FFN |  (expand 4x, GeLU, project back)
   +-----+
         |
   + Residual Connection
         |
   [Repeat N times]
         |
   +-----------+
   | LayerNorm |  (final)
   +-----------+
         |
   +------------+
   | Unembedding |  (linear projection to vocab)
   +------------+
         |
   +----------+
   | Softmax  |  --> P(next token | context)
   +----------+
```

Ключевые отличия GPT-2 от оригинального Transformer decoder:
- **Pre-normalization** (LayerNorm перед, а не после sub-layer) -- стабильнее при глубоких стеках
- **Нет cross-attention** -- decoder-only, работает с одной последовательностью
- **GeLU** вместо ReLU в FFN

## GPT-2: zero-shot generalization

Radford et al. (2019) показали ключевой insight: достаточно большая CLM модель может решать задачи **без fine-tuning**, через **prompt engineering**:

```
Перевод:
  "Translate English to French:
   sea otter => loutre de mer
   cheese => fromage
   The quick brown fox =>"

Суммаризация:
  "[Article text]
   TL;DR:"
```

Модель не обучалась на перевод или суммаризацию -- она просто продолжает текст в нужном формате. GPT-2 (1.5B) показал, что это работает, хотя и неидеально.

## GPT-3: in-context learning

Brown et al. (2020) масштабировали CLM до 175B параметров и показали **in-context learning**: модель решает задачи, получив несколько примеров прямо в prompt (few-shot), без обновления весов.

| Режим | Описание | Пример |
|-------|----------|--------|
| Zero-shot | Только описание задачи | "Translate: cat =>" |
| One-shot | 1 пример + задача | "cat => кот, dog =>" |
| Few-shot | N примеров + задача | "cat => кот, dog => собака, house =>" |

Из GPT-3: **scaling** -- ключ к in-context learning. Маленькие модели (125M) почти не улучшаются от few-shot примеров. Большие (175B) -- значительно.

## CLM vs MLM: когда что лучше

### CLM выигрывает

- **Open-ended generation** (creative writing, dialog, code)
- **Few-shot / zero-shot** learning (in-context learning)
- **Knowledge-intensive tasks** (closed-book QA) -- больше знаний в весах
- **Масштабирование**: CLM модели доминируют при >10B параметров

### MLM выигрывает

- **Traditional NLU** с аннотированными данными: NER (~2x лучше), токсичность, классификация
- **Span-extraction QA** (SQuAD)
- **Sentence embeddings** (BERT > GPT для semantic similarity)
- **Compute efficiency при fine-tuning**: BERT-base (110M) fine-tuned > GPT-3 175B zero-shot на многих задачах

### Из Harnessing LLMs Survey

Практический вывод: если есть достаточно размеченных данных и задача -- classification/NER/QA, **BERT-family лучше**. Если данных мало и задача -- generation/reasoning, **GPT-family лучше**.

## Данные и токенизация

### Токенизаторы CLM моделей

| Модель | Токенизатор | Vocab size | Особенности |
|--------|------------|-----------|-------------|
| GPT-2 | BPE (byte-level) | 50,257 | Byte-level: может кодировать любой текст |
| GPT-3 | BPE (byte-level) | 50,257 | Как GPT-2 |
| LLaMA | SentencePiece BPE | 32,000 | Цифры разбиваются поодиночке |
| LLaMA 2 | SentencePiece BPE | 32,000 | Как LLaMA |

Выбор токенизатора влияет на **fertility** (сколько токенов на слово) и на качество для разных языков. Byte-level BPE гарантирует, что любой unicode текст может быть закодирован.

## Хронология

| Год | Milestone | Модель/params |
|-----|-----------|---------------|
| 2018 | GPT-1: CLM + fine-tuning | 117M |
| 2019 | GPT-2: zero-shot через prompt | 1.5B |
| 2020 | GPT-3: in-context learning | 175B |
| 2022 | Chinchilla: optimal compute allocation | 70B, 1.4T tokens |
| 2023 | LLaMA: open-source efficient CLM | 7B-65B, 1-1.4T tokens |
| 2023 | GPT-4: multimodal CLM | undisclosed |
| 2024 | LLaMA 3: 15T tokens training | 8B-405B |

## Key papers

- [[02 Areas/ML & DL/Papers/GPT 2.0]] -- zero-shot multi-task через CLM
- [[02 Areas/ML & DL/Papers/GPT 3.0]] -- few-shot in-context learning, scaling laws
- [[02 Areas/ML & DL/Papers/LLaMA]] -- efficient CLM с 1T+ tokens, open-source

## Related concepts

- [[02 Areas/ML & DL/Concepts/Training/Masked Language Modeling|Masked Language Modeling (MLM)]] -- альтернативный objective для encoder-моделей
- [[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]] -- CLM как основной pre-training objective
- [[02 Areas/ML & DL/Concepts/Architectures/Decoder-only|Decoder-only]] -- архитектура, использующая CLM
- [[02 Areas/ML & DL/Concepts/Inference/Sampling|Sampling]] -- стратегии генерации при CLM inference
- [[02 Areas/ML & DL/Concepts/Inference/KV-Cache|KV-Cache]] -- оптимизация авторегрессивного inference
- [[02 Areas/ML & DL/Concepts/Inference/In-context Learning|In-context Learning]] -- emergent ability CLM моделей
- [[02 Areas/ML & DL/Concepts/Training/nanoGPT|nanoGPT]] -- образовательная реализация CLM

## Дополнительные ресурсы

- [Jay Alammar -- The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) -- визуализация CLM и decoder-only Transformer
- [Lena Voita -- Transfer Learning](https://lena-voita.github.io/nlp_course/transfer_learning.html) -- сравнение CLM и MLM
- [Karpathy -- Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) -- 2-часовой tutorial по CLM
