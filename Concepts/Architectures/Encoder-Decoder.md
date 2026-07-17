---
title: "Encoder-Decoder"
aliases: [seq2seq, sequence-to-sequence, encoder-decoder transformer]
type: concept
category: Architectures
papers:
  - "[[02 Areas/ML & DL/Papers/T5]]"
  - "[[02 Areas/ML & DL/Papers/Flan-T5-PaLM]]"
  - "[[02 Areas/ML & DL/Papers/Formal Algorithms for Transformers]]"
courses:
  - "[[02 Areas/ML & DL/Courses/SHAD LLM/Week 1 — Intro to LLMs|SHAD LLM — Week 1]]"
sources:
  - "[LM Po — Encoder-Decoder Transformer Models: BART and T5](https://medium.com/@lmpo/encoder-decoder-transformer-models-a-comprehensive-study-of-bart-and-t5-132b3f9836ed)"
  - "[Yi Tay — What happened to BERT & T5?](https://www.yitay.net/blog/model-architecture-blogpost-encoders-prefixlm-denoising)"
---

# Encoder-Decoder

## What it is

Encoder-Decoder (seq2seq) — вариант [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] архитектуры из оригинальной статьи "Attention Is All You Need" (Vaswani et al., 2017). Состоит из двух частей: **encoder** с bidirectional attention (читает весь входной контекст) и **decoder** с causal attention + cross-attention к encoder (генерирует выходную последовательность). Применяется для seq2seq задач: machine translation, summarization, text paraphrasing.

## How it works

Из [[02 Areas/ML & DL/Papers/Formal Algorithms for Transformers]] (Algorithm 8 — EDTransformer):

### Encoder часть

Обрабатывает **входную** последовательность z:
```
Для каждого encoder блока ℓ:
  Z ← Z + MHAttention(LayerNorm(Z), LayerNorm(Z), Mask=1)  # bidirec.
  Z ← Z + FFN(LayerNorm(Z))
```
Результат: contextualized представления всех токенов входа, видящих весь контекст.

### Decoder часть

Обрабатывает **выходную** последовательность x, имея доступ к encoder output Z:
```
Для каждого decoder блока ℓ:
  X ← X + MHAttention(LayerNorm(X), LayerNorm(X), Mask=causal)  # unidirec.
  X ← X + MHAttention(LayerNorm(X), LayerNorm(Z), Mask=1)       # cross-attn
  X ← X + FFN(LayerNorm(X))
```

## Cross-Attention: ключевой механизм (детальный разбор)

Cross-attention — **главное отличие** encoder-decoder от decoder-only. Это механизм, через который decoder «видит» входную последовательность.

### Как работает cross-attention

В обычном self-attention Query, Key, Value приходят из **одной** последовательности. В cross-attention:
- **Query** — из decoder (текущее состояние генерации)
- **Key, Value** — из encoder output (представления входной последовательности)

$$\text{CrossAttention}(X, Z) = \text{softmax}\left(\frac{Q_X K_Z^T}{\sqrt{d_k}}\right) V_Z$$

где $Q_X = X W^Q$, $K_Z = Z W^K$, $V_Z = Z W^V$.

### Интуиция

Представьте перевод предложения. Decoder генерирует перевод слово за словом. На каждом шаге cross-attention отвечает на вопрос: **«Какие слова входного предложения сейчас релевантны для генерации следующего слова перевода?»**

Для генерации французского "chat" (кошка) cross-attention будет назначать высокий вес английскому "cat", а не "the" или "sat". Это автоматический **alignment** — модель сама учится выравнивать входные и выходные слова.

### Cross-attention vs concatenation (decoder-only подход)

Decoder-only модели обрабатывают input и output как **одну** последовательность: `[input; output]`. Вместо cross-attention используется causal self-attention, где output-токены видят input через обычный attention.

| Аспект | Cross-attention (enc-dec) | Concatenation (decoder-only) |
|--------|--------------------------|------------------------------|
| Attention type | Explicit input→output | Implicit через causal mask |
| Input context | Bidirectional (encoder) | Unidirectional (causal) |
| Compute | Encoder: O(n_in^2), Decoder: O(n_out^2 + n_out*n_in) | O((n_in + n_out)^2) |
| KV-cache при inference | Encoder KV = const | Input KV = const |
| Alignment | Explicit attention weights | Implicit in representations |

**Когда cross-attention лучше:** задачи с чётким разделением input/output, где важно **полное bidirectional понимание** входа (перевод, summarization с длинным входом).

**Когда concatenation лучше:** open-ended generation, where input/output boundary is fuzzy (chatbot, continuation).

### Три вида attention в decoder'е

1. **Causal self-attention** — decoder видит только уже сгенерированные выходные токены (autoregressive)
2. **Cross-attention** — query из decoder, key/value из encoder; позволяет decoder'у обращаться к любой части входа
3. **FFN** — position-wise преобразование

### Задача: seq2seq prediction

Цель — максимизировать:
```
P(x | z) = P(x[1] | z) · P(x[2] | x[1], z) · ... · P(x[T] | x[1:T-1], z)
```

## T5 vs BART: два подхода к encoder-decoder

### T5: Text-to-Text Transfer Transformer

[[02 Areas/ML & DL/Papers/T5]] использует enc-decoder архитектуру и переформулирует **все NLP задачи как text-to-text**:
- Classification: вход = "classify: {text}", выход = метка как текст ("positive"/"negative")
- Translation: вход = "translate English to German: {text}", выход = перевод
- Summarization: вход = "summarize: {text}", выход = краткое изложение

**Pre-training objective: span corruption.**
Случайные spans (не отдельные токены) заменяются sentinel tokens (`<extra_id_0>`, `<extra_id_1>`, ...). Модель должна восстановить удалённые spans.

Пример:
- Input: `"The <X> brown <Y> jumps over the <Z> dog"`
- Target: `"<X> quick <Y> fox <Z> lazy"`

**Ключевое:** T5 маскирует **несколько consecutive tokens** одним sentinel → модель должна определить, **сколько** токенов пропущено (в отличие от BERT, где 1 mask = 1 token).

### BART: Denoising Autoencoder

**Pre-training objective:** reconstruction из corrupted input. BART применяет **пять видов noise** к входу:
1. Token masking (как BERT)
2. Token deletion (без placeholder)
3. Text infilling (span → один `<mask>`, нужно угадать длину)
4. Sentence permutation (перемешивание предложений)
5. Document rotation (текст начинается со случайного токена)

**Отличие от T5:** BART восстанавливает **весь оригинальный текст**, а не только masked spans. Encoder получает corrupted text, decoder генерирует полный clean text.

### Детальное сравнение

| Аспект | T5 | BART |
|--------|-----|------|
| Pre-training | Span corruption | Denoising (5 noise types) |
| Target | Только masked spans | Полный оригинальный текст |
| Task formulation | Text-to-text prefix | Standard seq2seq |
| Sizes | 60M — 11B | 140M — 400M |
| Best for | **Multitask, instruction following** | **Summarization** |
| ROUGE (CNN/DM) | 43.5 (11B) | **44.2** (large, 400M) |
| GLUE | **89.7** (11B) | 88.4 (large) |
| Successor | Flan-T5, mT5, UL2 | mBART, PEGASUS (inspired) |

### Когда использовать какой

| Сценарий | Рекомендация | Почему |
|----------|-------------|--------|
| Summarization | BART | Лучшие ROUGE scores, оптимизирован для generation |
| Multitask / instruction following | T5 / Flan-T5 | Text-to-text framework, instruction tuning |
| Translation | mT5 / mBART | Multilingual variants |
| Classification (с fine-tuning) | T5 | Text-to-text unification |
| Data-to-text generation | BART | Denoising objective помогает |

## Why it matters (и почему encoder-decoder проиграл decoder-only)

Encoder-decoder — **исторически первая** Transformer архитектура (2017), созданная именно для machine translation. Но к 2023 году decoder-only доминирует.

### Причины проигрыша

1. **Сложность:** два компонента vs один. Encoder-decoder сложнее в реализации, debugging, inference optimization
2. **Zero/few-shot:** encoder-decoder требует explicit input/output separation, что затрудняет prompt engineering
3. **Scaling:** при >100B параметрах decoder-only показывает лучший scaling behaviour (Yi Tay, Google Research)
4. **KV-cache overhead:** encoder KV-cache + decoder KV-cache → больше memory при inference

### Где encoder-decoder всё ещё актуален (2024+)

| Задача | Почему enc-dec лучше | Пример |
|--------|---------------------|--------|
| Machine translation | Explicit alignment через cross-attention | mT5, NLLB |
| Medical NLP | Structured input→output, limited data | Clinical BART |
| Document understanding | Long input, short output | LED (Longformer Encoder-Decoder) |
| Speech recognition | Audio→text = natural seq2seq | Whisper (encoder-decoder) |

**Whisper (OpenAI)** — encoder-decoder для speech recognition. Audio encoder (CNN + Transformer) → text decoder. Это **natural fit**: audio и text — разные модальности, encoder-decoder идеально подходит.

**NLLB (Meta)** — No Language Left Behind. Encoder-decoder для перевода 200+ языков. Cross-attention обеспечивает alignment между языками.

## Представители (расширенная таблица)

| Модель | Год | Params | Pre-training | Best for |
|--------|-----|--------|-------------|----------|
| **Original Transformer** | 2017 | 65M | NMT | Machine translation |
| **T5** | 2019 | 60M-11B | Span corruption | Multitask NLU/NLG |
| **BART** | 2020 | 140M-400M | Denoising | Summarization |
| **mT5** | 2020 | 300M-13B | Span corruption | Cross-lingual |
| **mBART** | 2020 | 680M | Denoising | Multilingual translation |
| **Flan-T5** | 2022 | 80M-11B | T5 + instruction tuning | Few-shot NLU/NLG |
| **UL2** | 2022 | 20B | Mixture-of-Denoisers | Unified NLU/NLG |
| **Whisper** | 2022 | 39M-1.5B | Audio→text | Speech recognition |
| **NLLB** | 2022 | 600M-54B | Translation | 200+ languages |

## Key papers

- [[02 Areas/ML & DL/Papers/T5]] — канонический современный enc-decoder; text-to-text framework
- [[02 Areas/ML & DL/Papers/Flan-T5-PaLM]] — instruction tuning поверх T5/PaLM
- [[02 Areas/ML & DL/Papers/Formal Algorithms for Transformers]] — формальные алгоритмы EDTransformer (Algorithm 8)

## In courses

- [[02 Areas/ML & DL/Courses/SHAD LLM/Week 1 — Intro to LLMs|SHAD LLM — Week 1]]: machine translation, text summarization, text paraphrasing

## Related concepts

- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]]
- [[02 Areas/ML & DL/Concepts/Architectures/Encoder-only|Encoder-only]]
- [[02 Areas/ML & DL/Concepts/Architectures/Decoder-only|Decoder-only]]
- [[02 Areas/ML & DL/Concepts/Architectures/T5|T5]]
- [[02 Areas/ML & DL/Concepts/NLP/Self-Attention|Self-Attention]]
- [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]]

## Дополнительные ресурсы

- [Yi Tay — What happened to BERT & T5?](https://www.yitay.net/blog/model-architecture-blogpost-encoders-prefixlm-denoising) — глубокий анализ от Google Research: почему encoder-decoder проиграл и где ещё актуален
- [LM Po — BART and T5 Comparison](https://medium.com/@lmpo/encoder-decoder-transformer-models-a-comprehensive-study-of-bart-and-t5-132b3f9836ed) — детальное сравнение двух подходов
- [HuggingFace — Encoder Decoder Models](https://huggingface.co/docs/transformers/model_doc/encoder-decoder) — практическое руководство
