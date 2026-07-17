---
title: "Encoder-only"
aliases: [masked LM architecture, encoder-only transformer]
type: concept
status: legacy
category: Architectures
papers:
  - "[[02 Areas/ML & DL/Papers/BERT]]"
  - "[[02 Areas/ML & DL/Papers/RoBERTa]]"
  - "[[02 Areas/ML & DL/Papers/DeBERTa]]"
courses:
  - "[[02 Areas/ML & DL/Courses/SHAD LLM/Week 1 — Intro to LLMs|SHAD LLM — Week 1]]"
sources:
  - "[Devlin et al. — BERT (2018)](https://arxiv.org/abs/1810.04805)"
  - "[Formal Algorithms for Transformers (Phuong & Hutter, 2022)](https://arxiv.org/abs/2207.09238)"
  - "[Lilian Weng — BERT Overview](https://lilianweng.github.io/posts/2019-01-31-lm/)"
---

# Encoder-only

## Зачем нужен encoder-only

Представь задачу: определить sentiment предложения *"The movie was surprisingly good despite the terrible trailer."* Чтобы понять, что sentiment **позитивный**, нужно видеть слово «good» **до того** как обрабатываешь «surprisingly», и «despite» **после**. Нужен **двунаправленный контекст**.

Decoder-only (GPT) видит только **прошлое**: обрабатывая «surprisingly», он не знает, что дальше будет «good». Encoder-only (BERT) видит **всё предложение целиком** — каждый токен может обращать внимание на любой другой токен, включая будущие.

Это принципиальное архитектурное различие определяет, для чего каждая архитектура подходит лучше.

## Как устроен encoder-only Transformer

Из [[02 Areas/ML & DL/Papers/Formal Algorithms for Transformers]] (Algorithm 9 — BERT):

```
Input: token embeddings X + position embeddings

Для каждого блока ℓ (1..L):
  X ← X + MHAttention(LayerNorm(X), LayerNorm(X), Mask=1)  # bidirectional!
  X ← X + FFN(LayerNorm(X))

Output: contextualized representations для каждого токена
```

**Ключевое:** `Mask = 1` — **нет маскировки**. Токен на позиции $t$ может обращать внимание на **любой** другой токен в последовательности, включая позиции $t+1, t+2, \ldots, T$. Это полная attention matrix размера $T \times T$.

### Сравнение attention masks

```
Encoder-only (BERT):        Decoder-only (GPT):
1 1 1 1 1                   1 0 0 0 0
1 1 1 1 1                   1 1 0 0 0
1 1 1 1 1                   1 1 1 0 0
1 1 1 1 1                   1 1 1 1 0
1 1 1 1 1                   1 1 1 1 1
(всё видно)                 (только прошлое)
```

Bidirectional attention означает: каждый выходной вектор $h_i$ содержит информацию о **всей** последовательности, а не только о $x_1, \ldots, x_i$.

## Pre-training: Masked Language Modeling (MLM)

Если модель видит всю последовательность, нельзя обучать через next token prediction (модель просто «подглядит» ответ). Решение — **Masked Language Modeling**:

1. Берём последовательность: `The cat sat on the mat`
2. Случайно выбираем 15% токенов
3. Из выбранных: 80% → `[MASK]`, 10% → random token, 10% → без изменений
4. Задача: предсказать оригинальные токены

```
Input:  The [MASK] sat on the mat
Target: cat (для позиции [MASK])
```

**Почему 80/10/10:** если заменять все 100% на `[MASK]`, модель увидит `[MASK]` только при обучении, но никогда при inference → distribution mismatch. 10% random + 10% unchanged учат модель **всегда строить хорошие representations**, даже когда входной токен не маскирован.

Математически: модель максимизирует

$$\mathcal{L} = \sum_{i \in C} \log p_\theta(x_i | \tilde{X})$$

где $C$ — множество маскированных позиций, $\tilde{X}$ — входная последовательность с масками.

### Почему bidirectional важнее causal для NLU

Ablation из оригинального BERT paper (Table 5):

| Модель | SQuAD F1 | MNLI |
|--------|----------|------|
| BERT (bidirectional) | **91.8** | **86.6** |
| LTR (GPT-style, left-to-right) | **81.1** | **82.1** |

**-10.7 F1 на SQuAD** при замене bidirectional на causal attention. Для extractive QA (найти span ответа в тексте) двунаправленный контекст **критичен** — нужно понимать и вопрос, и контекст, и их взаимосвязь.

## Fine-tuning на downstream задачах

Encoder-only модели fine-tune'ятся добавлением **task-specific head** поверх contextualized representations:

### Классификация текста

```
[CLS] This movie is great [SEP]
  ↓
BERT Encoder (12-24 layers)
  ↓
[CLS] vector → Linear(768, num_classes) → softmax → label
```

`[CLS]` token — специальный токен, чей выходной вектор агрегирует информацию о всей последовательности через self-attention.

### Named Entity Recognition (NER)

```
[CLS] John lives in New York [SEP]
  ↓
BERT Encoder
  ↓
Каждый токен → Linear(768, num_tags) → BIO tags
      John → B-PER
      lives → O
      in → O
      New → B-LOC
      York → I-LOC
```

### Extractive QA (SQuAD)

```
[CLS] Where does John live? [SEP] John lives in New York. [SEP]
  ↓
BERT Encoder
  ↓
Два вектора S и E обучаются: 
  Start score: S · T_i  для каждого токена i
  End score:   E · T_i  для каждого токена i
  Answer span: argmax(Start) .. argmax(End)
```

### Sentence pair tasks (NLI, paraphrase)

```
[CLS] The cat sat on the mat [SEP] A cat was resting [SEP]
  ↓
BERT Encoder (видит оба предложения через bidirectional attention)
  ↓
[CLS] vector → Linear → {entailment, contradiction, neutral}
```

## Encoder-only vs Decoder-only: когда что использовать

Из [[02 Areas/ML & DL/Papers/Harnessing the Power of LLMs in Practice- A Survey on ChatGPT and Beyond]]:

| Задача | Encoder-only (fine-tuned) | Decoder-only LLM | Победитель |
|--------|--------------------------|-------------------|------------|
| Text classification (много данных) | **BERT/RoBERTa fine-tuned** | GPT-4 zero/few-shot | **Encoder-only** |
| NER (CoNLL03) | **~2x лучше** | GPT-4 | **Encoder-only** |
| Summarization (по ROUGE) | **Fine-tuned BART/T5** | GPT-4 | **Encoder-only** |
| Summarization (по human eval) | — | **GPT-4** | **Decoder-only** |
| Toxicity detection | **BERT (Perspective API)** ~2x лучше GPT-4 | GPT-4 | **Encoder-only** |
| Open-ended generation | Не применим | **GPT-4** | **Decoder-only** |
| Few-shot / zero-shot | Не применим | **GPT-4** | **Decoder-only** |
| Chatbot / real-world tasks | Не применим | **GPT-4** | **Decoder-only** |

**Правило:** если есть labeled данные и задача — classification/extraction/tagging → fine-tuned encoder-only. Если нужна генерация или мало данных → decoder-only LLM.

## Эволюция encoder-only моделей

| Модель | Год | Params | Ключевое нововведение | GLUE |
|--------|-----|--------|----------------------|------|
| **BERT** | 2018 | 110M/340M | MLM + NSP, bidirectional | 80.5 |
| **RoBERTa** | 2019 | 355M | Убрал NSP, больше данных, dynamic masking | **88.5** |
| **ALBERT** | 2019 | 12M-235M | Weight sharing, factorized embeddings | 89.4 |
| **DistilBERT** | 2019 | 66M | Knowledge distillation (40% меньше, 60% быстрее) | 77.0 |
| **ELECTRA** | 2020 | 110M/335M | Replaced Token Detection вместо MLM | 89.4 |
| **DeBERTa** | 2020 | 304M/1.5B | Disentangled attention, Enhanced Mask Decoder | **89.9 (SuperGLUE)** |
| **DeBERTa-v3** | 2021 | 304M | ELECTRA-style RTD + disentangled attention | Production standard |

### Что каждое улучшение дало

- **RoBERTa → BERT**: BERT был недообучен. Правильный recipe = +8% GLUE.
- **ALBERT → BERT**: можно уменьшить модель без потери качества через weight sharing.
- **DistilBERT → BERT**: knowledge distillation для production (скорость > точность).
- **ELECTRA → BERT**: MLM неэффективен — учит только 15% токенов за batch. RTD учит на **всех** токенах.
- **DeBERTa → RoBERTa**: архитектурные улучшения дают +2-3% даже при меньших данных.

## Embeddings: скрытое применение encoder-only

Помимо classification/NER, encoder-only модели широко используются для **text embeddings**:

- **Mean pooling** всех token representations → sentence embedding
- **[CLS] token** → sentence embedding (менее популярно)
- Используются в **RAG** (retrieval), semantic search, clustering

Sentence-BERT, E5, BGE — все построены на encoder-only backbone.

## Почему encoder-only «проиграл» decoder-only (2022-2024)

К 2023 году decoder-only LLM (ChatGPT, GPT-4) стали **доминирующим** подходом. Причины:

1. **Универсальность**: одна decoder-only модель решает ВСЕ задачи через prompting, encoder-only требует отдельного fine-tuning на каждую.
2. **Zero/few-shot**: decoder-only работает без labeled данных, encoder-only требует annotations.
3. **Генерация**: encoder-only не может генерировать текст, только классифицировать/извлекать.
4. **Scale**: decoder-only масштабируется до 175B-1T+ параметров; encoder-only остановились на ~1.5B.

Но encoder-only **не умер** — для classification, NER, embeddings, и production NLU с labeled данными он остаётся **лучшим выбором**.

## Key papers

- [[02 Areas/ML & DL/Papers/BERT]] — первый и канонический encoder-only
- [[02 Areas/ML & DL/Papers/RoBERTa]] — оптимизированный pre-training
- [[02 Areas/ML & DL/Papers/DeBERTa]] — disentangled attention

## In courses

- [[02 Areas/ML & DL/Courses/SHAD LLM/Week 1 — Intro to LLMs|SHAD LLM — Week 1]]: text classification, NER, NLU

## Related concepts

- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] — базовая архитектура
- [[02 Areas/ML & DL/Concepts/Architectures/BERT|BERT]] — канонический encoder-only
- [[02 Areas/ML & DL/Concepts/Architectures/RoBERTa|RoBERTa]] — оптимизированный BERT
- [[02 Areas/ML & DL/Concepts/Architectures/DeBERTa|DeBERTa]] — архитектурно улучшенный
- [[02 Areas/ML & DL/Concepts/Training/Masked Language Modeling|Masked Language Modeling (MLM)]] — pre-training objective
- [[02 Areas/ML & DL/Concepts/Architectures/Decoder-only|Decoder-only]] — альтернативная архитектура
- [[02 Areas/ML & DL/Concepts/Architectures/Encoder-Decoder|Encoder-Decoder]] — альтернативная архитектура
- [[02 Areas/ML & DL/Concepts/NLP/GLUE|GLUE]] — основной бенчмарк

## Дополнительные ресурсы

- [Jay Alammar — The Illustrated BERT](https://jalammar.github.io/illustrated-bert/) — визуализация BERT и MLM
- [Lilian Weng — Generalized Language Models](https://lilianweng.github.io/posts/2019-01-31-lm/) — обзор encoder-only vs decoder-only
- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers/model_doc/bert) — реализация и примеры
