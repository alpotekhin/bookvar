---
title: "Voita NLP — Transfer Learning"
type: course-note
course: "Lena Voita NLP"
---

# Voita NLP — Transfer Learning

> Как перенести знания из одной задачи в другую. Путь от Word2Vec до BERT и GPT.

**Курс:** [[Lena Voita NLP/_index|Lena Voita NLP Course]]
**Страница:** https://lena-voita.github.io/nlp_course/transfer_learning.html
**Связанные концепты:** [[Transfer Learning]], [[Pre-training]], [[Fine-tuning]], [[Masked Language Modeling]], [[Causal Language Modeling]], [[PEFT]], [[LoRA]]

---

## Зачем нужен Transfer Learning

**Проблема:** для большинства NLP-задач мало размеченных данных. Sentiment analysis, NER, QA — тысячи-десятки тысяч примеров.

**Решение:** обучить модель на **огромном** неразмеченном тексте (pre-training), затем адаптировать к конкретной задаче (fine-tuning).

```
Неразмеченный текст (TB)  → [Pre-training]  → Language Model
                                                    ↓
Размеченные данные (KB)   → [Fine-tuning]   → Task-Specific Model
```

Ключевое наблюдение: **языковое моделирование** — бесплатная задача, данные — весь интернет.

---

## Две великие идеи

Voita выделяет две фундаментальные идеи в развитии transfer learning:

### Идея 1: От слов к контекстуальным представлениям

**Проблема static embeddings:** Word2Vec/GloVe дают один вектор на слово. "Bank" (берег реки) и "bank" (финансовый) неразличимы.

**Решение:** представления должны зависеть от контекста.

#### CoVe (Contextualized Vectors, 2017)

Первая попытка: обучить LSTM encoder на задаче машинного перевода (EN→DE), затем использовать его representations для downstream tasks.

```
English text → [Trained NMT Encoder] → contextual vectors
                                         ↓
                          [Task-specific layers] → prediction
```

Результат: улучшения на text classification, NLI, QA.

#### ELMo (Embeddings from Language Models, 2018)

**Архитектура:** двунаправленный LSTM language model + character-level CNN для входов.

```
Forward LM:   → h₁→ → h₂→ → h₃→   (left-to-right)
Backward LM:  ← h₁← ← h₂← ← h₃←  (right-to-left)

ELMo_k = γ · Σ_l s_l · [h_l→; h_l←]
```

**Инновация:** вместо использования только последнего слоя, ELMo берёт **взвешенную сумму всех слоёв** с task-specific весами s_l:

```
Разные слои кодируют разное:
  Слой 1: syntax (POS, chunking)
  Слой 2: semantics (word sense, coreference)
```

Задача сама выбирает, какие слои ей важнее.

**Результат:** "huge improvement for several tasks" — Best Paper Award NAACL 2018.

**Ограничение ELMo:** модель используется как **feature extractor** — embeddings подаются в отдельную task-specific архитектуру. Каждая задача всё ещё требует своей архитектуры.

---

### Идея 2: Замена task-specific архитектур

**Парадигмальный сдвиг:** вместо "pretrained embeddings + task-specific model" → "pretrained model = IS the model, fine-tune end-to-end."

---

## GPT: Generative Pre-Training (2018)

### Архитектура

12-слойный Transformer **decoder** (left-to-right, causal attention):

```
[t₁, t₂, ..., t_n] → Transformer Decoder → [p₁, p₂, ..., p_n]
```

### Pre-training: [[Causal Language Modeling]]

Стандартное авторегрессионное моделирование — предсказываем следующий токен:

```
L_pretrain = -Σ_t log p(t_t | t₁, ..., t_{t-1})
```

### Fine-tuning

**Ключевая идея:** минимальные изменения архитектуры. Для разных задач — разный формат входа:

**Sentence classification:**
```
[text tokens] → Transformer → [last token representation] → Linear → label
```

**Sentence pair (NLI, similarity):**
```
[sentence_1] [DELIM] [sentence_2] → Transformer → [last token] → Linear → label
```

**Question answering:**
```
[context] [DELIM] [question] [DELIM] [answer_i] → Transformer → score
```

**Combined loss при fine-tuning:**
```
L = L_task + λ · L_LM
```

Дополнительная LM loss при fine-tuning помогает сохранить выученные representations.

### GPT → GPT-2 → GPT-3

| | GPT-1 | GPT-2 | GPT-3 |
|--|-------|-------|-------|
| Параметры | 117M | 1.5B | 175B |
| Данные | BookCorpus | WebText (40GB) | Common Crawl + books (570GB) |
| Инновация | Pre-train + fine-tune | Zero-shot via prompting | Few-shot in-context learning |

GPT-2 показал: достаточно большая модель решает задачи **без fine-tuning** — через правильный prompt. GPT-3 развил это в [[02 Areas/ML & DL/Concepts/Inference/In-context Learning|In-context Learning]].

---

## BERT: Bidirectional Encoder Representations (2018)

### Ключевое отличие от GPT

GPT — **однонаправленный** (видит только left context). BERT — **двунаправленный** (видит и левый, и правый контекст).

```
GPT:  p(token | left_context)           — causal
BERT: p(token | left_AND_right_context) — bidirectional
```

Bidirectional context критически важен для задач понимания (NLI, QA, NER), где нужно видеть весь контекст.

### Архитектура

Transformer **encoder** (не decoder!) с специальными токенами:

```
[CLS] sentence_1 [SEP] sentence_2 [SEP]
```

**Input representation** = token embedding + position embedding + segment embedding (A/B для пар предложений).

### Pre-training Objectives

**1. [[Masked Language Modeling]] (MLM):**

Случайно выбираем 15% токенов и:
- 80% заменяем на [MASK]
- 10% заменяем на случайный токен
- 10% оставляем без изменений

Модель предсказывает оригинальные токены:

```
Input:  "The [MASK] sat on the mat"
Target: "cat"
```

Зачем 10% random + 10% unchanged? При fine-tuning нет [MASK] токенов — если модель привыкнет предсказывать только после [MASK], будет mismatch. Random и unchanged уменьшают этот эффект.

**2. [[Next Sentence Prediction]] (NSP):**

Бинарная классификация: идут ли два предложения друг за другом?

```
[CLS] Sentence_A [SEP] Sentence_B [SEP] → IsNext / NotNext
```

50% позитивных пар (реальные последовательные), 50% негативных (случайные). Позже показали, что NSP не особо помогает (RoBERTa убрал его).

### Fine-tuning BERT

**Classification:** [CLS] representation → Linear → label

**Token tagging (NER):** каждый токен → Linear → tag

**QA (SQuAD):** предсказываем start и end позиции ответа в контексте

```
[CLS] question [SEP] context_tokens [SEP]
                      ↓          ↓
                    start       end
```

### BERT Sizes

| | BERT-base | BERT-large |
|--|-----------|------------|
| Layers | 12 | 24 |
| Hidden | 768 | 1024 |
| Heads | 12 | 16 |
| Params | 110M | 340M |

---

## Adapters: параметрически эффективный transfer

### Проблема full fine-tuning

Fine-tuning BERT обновляет **все** параметры. Для каждой задачи — отдельная копия модели:

```
BERT-base (110M × 4 bytes) × 10 задач = 4.4 GB
```

### Adapter Solution

Вставляем маленькие trainable модули (adapters) в каждый Transformer-слой. Original weights **заморожены**:

```
[Self-Attention] ← frozen
     ↓
[Adapter] ← trainable (маленький!)
     ↓
[FFN] ← frozen
     ↓
[Adapter] ← trainable
```

**Adapter architecture:**

```
x → [Down-project: d → m] → [ReLU] → [Up-project: m → d] → + x (residual)
```

m << d (например, m=64 при d=768). Всего ~3.6% дополнительных параметров при сопоставимом качестве.

**Современные наследники:** [[LoRA]], [[PEFT]] — ещё более эффективные методы.

---

## Analysis: что знает BERT

Voita разбирает research по интерпретации BERT:

### Attention Patterns

- Некоторые головы следят за позицией (next token, previous token)
- Некоторые фокусируются на [SEP] и [CLS] (специальные токены)
- Некоторые кодируют **синтаксические зависимости** (аналогично dependency parsing)

### FFN как Key-Value Memory

Voita описывает интересное наблюдение: FFN слои можно интерпретировать как key-value memory:

```
Key:   первый линейный слой → активирует "концепт"
Value: второй линейный слой → генерирует соответствующее vocabulary distribution
```

### Probing: NLP Pipeline в BERT

Weighted probing по всем слоям показывает, что BERT восстанавливает **классический NLP pipeline**:

```
Слой 1-4:   POS tagging (синтаксические категории)
Слой 5-8:   Syntax (dependency parsing)
Слой 9-12:  Semantics (coreference, NER)
```

### Factual Knowledge

BERT может отвечать на factual вопросы в формате cloze:

```
"The capital of France is [MASK]." → "Paris"
"Einstein was born in [MASK]."     → "Germany"
```

Модель хранит factual knowledge в параметрах без явного обучения на QA.

---

## GPT vs BERT: когда что использовать

| Аспект | GPT (decoder) | BERT (encoder) |
|--------|--------------|----------------|
| Направление | Left-to-right | Bidirectional |
| Pre-training | Causal LM | Masked LM + NSP |
| Генерация | Естественная | Невозможна (bidirectional) |
| Понимание | Слабее (нет правого контекста) | Сильнее |
| Современное использование | LLM (GPT-4, LLaMA) | Embeddings, classification |

**Тренд 2023+:** GPT-style (decoder-only) доминирует, потому что масштабирование + in-context learning делают fine-tuning ненужным для многих задач.

---

## Ключевые выводы

1. **Transfer learning** решает проблему малых данных: pre-train на TB → fine-tune на KB
2. **ELMo** — первые контекстуальные embeddings, task-specific weighing слоёв
3. **GPT** — decoder-only, causal LM, minimal fine-tuning architecture changes
4. **BERT** — encoder, MLM + NSP, bidirectional context, доминировал 2018-2022
5. **Adapters** — 3.6% параметров вместо full fine-tuning, предшественники [[LoRA]]
6. BERT хранит лингвистическую иерархию: POS → syntax → semantics по слоям

---

## Источники

- Lena Voita, NLP Course: Transfer Learning — https://lena-voita.github.io/nlp_course/transfer_learning.html
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers" (2018)
- Radford et al., "Improving Language Understanding by Generative Pre-Training" (GPT, 2018)
- Peters et al., "Deep contextualized word representations" (ELMo, 2018)
- Houlsby et al., "Parameter-Efficient Transfer Learning for NLP" (Adapters, 2019)

---

**См. также:** [[Transfer Learning]], [[Pre-training]], [[Fine-tuning]], [[Masked Language Modeling]], [[Causal Language Modeling]], [[LoRA]], [[PEFT]], [[Voita NLP — Word Embeddings]], [[Voita NLP — Seq2Seq and Attention]]
