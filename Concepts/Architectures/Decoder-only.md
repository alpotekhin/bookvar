---
title: "Decoder-only"
aliases: [autoregressive architecture, causal LM architecture]
type: concept
category: Architectures
papers:
  - "[[02 Areas/ML & DL/Papers/GPT 2.0]]"
  - "[[02 Areas/ML & DL/Papers/GPT 3.0]]"
  - "[[02 Areas/ML & DL/Papers/GPT 4.0]]"
  - "[[02 Areas/ML & DL/Papers/LLaMA]]"
  - "[[02 Areas/ML & DL/Papers/LLaMA 2]]"
  - "[[02 Areas/ML & DL/Papers/OPT]]"
courses:
  - "[[02 Areas/ML & DL/Courses/SHAD LLM/Week 1 — Intro to LLMs|SHAD LLM — Week 1]]"
sources:
  - "[Cameron Wolfe — Decoder-Only Transformers: The Workhorse of Generative LLMs](https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse)"
  - "[Yi Tay — What happened to BERT & T5?](https://www.yitay.net/blog/model-architecture-blogpost-encoders-prefixlm-denoising)"
---

# Decoder-only

## What it is

Decoder-only — вариант [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] архитектуры с **causal (masked / unidirectional) self-attention**: токен на позиции t может видеть только токены на позициях $\leq t$. Обучается на autoregressive language modeling (предсказать следующий токен). Доминирующая архитектура современных LLM: GPT-2/3/4, LLaMA, OPT, PaLM, BLOOM.

## How it works

### Архитектура (из [[02 Areas/ML & DL/Papers/Formal Algorithms for Transformers]], Algorithm 10 — GPT)

```
Для каждого блока ℓ:
  X ← LayerNorm(X)
  X_attn ← X + MHAttention(X, X, Mask=causal)  # causal attention
  X ← X_attn + FFN(LayerNorm(X_attn))
return Unembedding(LayerNorm(X))
```

### Causal Mask: почему и как

**Causal Mask**: `Mask[t_z, t_x] = [[t_z ≤ t_x]]` — верхний треугольник матрицы attention scores заполняется $-\infty$ перед softmax. В результате token на позиции $t_x$ обновляет своё представление только через токены $1..t_x$ (включая себя), игнорируя $t_{x+1}..T$.

**Зачем:** causal mask обеспечивает **авторегрессивность** — при генерации модель не может «подглядывать» в будущее. Это ключевое свойство: именно causal masking позволяет генерировать текст token-by-token.

**Реализация:** attention weights вычисляются как:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V$$

где $M$ — causal mask matrix:

$$M_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}$$

После softmax: $e^{-\infty} = 0$, поэтому будущие позиции получают нулевой вес.

### Pre-training: Causal Language Modeling (CLM)

Обучение — максимизация log-likelihood:
```
L = Σ_t log P(x[t] | x[1:t-1]; θ)
```

Все позиции обрабатываются параллельно при обучении (teacher forcing), но inference — строго авторегрессивный. При каждом шаге генерации: append нового токена → forward pass → sample следующий.

**Эффективность обучения:** при teacher forcing каждая позиция даёт gradient signal (T loss terms из T токенов). Это **более эффективно**, чем MLM, где обычно маскируется только 15% токенов (только 0.15T loss terms).

### LLaMA-style архитектурные улучшения

Из [[02 Areas/ML & DL/Papers/LLaMA]] — три ключевых отличия от оригинального GPT:
1. **Pre-normalization с RMSNorm** вместо post-norm LayerNorm: нормализуем вход каждого подслоя вместо выхода → улучшение стабильности при обучении
2. **SwiGLU активация** вместо ReLU в FFN: `SwiGLU(x, W, V) = Swish(xW) ⊙ (xV)` → лучшее качество
3. **Rotary Positional Embeddings (RoPE)** вместо learned absolute PE: кодируют позицию через вращение в комплексном пространстве → лучше обобщается на длинные контексты

Эти три модификации стали **де-факто стандартом** open decoder-only LLM: LLaMA 2, Mistral, Falcon и большинство последующих.

## Почему decoder-only победил: детальный разбор

### Хронология доминирования

| Период | Доминирующая архитектура | Пример |
|--------|--------------------------|--------|
| 2017-2018 | Encoder-Decoder | Original Transformer, T5 |
| 2018-2019 | Encoder-only (NLU) + Decoder-only (NLG) | BERT + GPT |
| 2020 | Decoder-only начинает доминировать | GPT-3 |
| 2021-2022 | Decoder-only монопольно | PaLM, Chinchilla, GPT-3.5 |
| 2023+ | Decoder-only безоговорочно | GPT-4, LLaMA, Mistral, все major LLM |

### Причина 1: Простота и масштабируемость

Decoder-only — **самая простая** Transformer архитектура:
- Только один стек слоёв (vs два у encoder-decoder)
- Нет cross-attention (экономия ~33% параметров в каждом блоке)
- Один training objective (next token prediction)
- Один input format (concatenation of all text)

При масштабировании до 100B+ параметров эта простота **критически важна**: меньше компонентов → меньше потенциальных точек отказа → стабильнее обучение.

### Причина 2: KV-cache и эффективный inference

При авторегрессивной генерации decoder-only модели **переиспользуют** KV-cache:
- Каждый новый токен требует вычисления Q, K, V только для **одной позиции**
- K и V предыдущих позиций берутся из кэша
- Encoder-decoder должен хранить и encoder KV, и decoder KV — двойной overhead

### Причина 3: Scaling behavior

Исследование от Google (Yi Tay) и независимые reproduction показали:

> **Encoder-decoder модели показывают преимущество при low compute, но decoder-only модели лучше масштабируются при high compute.**

При маленьких моделях (~1B) T5 (encoder-decoder) бьёт GPT того же размера. Но при 100B+ разница исчезает, а decoder-only становится проще масштабировать. Поскольку индустрия движется к всё большим моделям — decoder-only выигрывает.

### Причина 4: Zero-shot и few-shot generalization

Decoder-only модели показывают **лучшую zero-shot generalization**: causal LM objective обучает модель предсказывать любое продолжение любого текста. Encoder-decoder привязан к input→output формату.

GPT-3 показал, что **prompt engineering** (формулировка задачи как text completion) позволяет решать произвольные задачи без fine-tuning. Это невозможно (или значительно сложнее) с encoder-decoder.

### Причина 5: Унификация задач

Все NLP задачи можно переформулировать как text completion:
- **Classification:** `"Review: ... Sentiment:"` → `"positive"`
- **Translation:** `"English: Hello French:"` → `"Bonjour"`
- **Summarization:** `"Article: ... TL;DR:"` → summary
- **QA:** `"Context: ... Question: ... Answer:"` → answer
- **Code:** `"# Function that sorts a list\ndef"` → implementation

Encoder-decoder требует explicit input/output separation. Decoder-only — просто продолжает текст.

## Causal Attention: механика в деталях

### Матричное представление

Для последовательности длины $T$:

$$\text{Scores} = \frac{QK^T}{\sqrt{d_k}} = \begin{pmatrix} s_{11} & -\infty & -\infty & \cdots \\ s_{21} & s_{22} & -\infty & \cdots \\ s_{31} & s_{32} & s_{33} & \cdots \\ \vdots & \vdots & \vdots & \ddots \end{pmatrix}$$

После softmax: нижний треугольник — ненулевые веса, верхний — нули. Токен $t$ «видит» только $1, 2, \ldots, t$.

### Отличие от bidirectional attention

| Свойство | Causal (decoder-only) | Bidirectional (encoder-only) |
|----------|----------------------|------------------------------|
| Mask | Нижний треугольник | Полная матрица |
| Контекст | Только прошлое | Прошлое + будущее |
| Генерация | Да (авторегрессивная) | Нет (нельзя генерировать) |
| NLU | Хуже (unidirectional) | Лучше (bidirectional) |
| Training efficiency | ~100% токенов дают loss | ~15% токенов (masked) |

### Prefix LM: компромисс

**Prefix LM** (например, PaLM использует вариант) — гибрид:
- Первая часть входа (prefix) обрабатывается **bidirectional** (как encoder)
- Остальная часть — **causal** (как decoder)
- Нет отдельного encoder/decoder — один стек

Это позволяет лучше понимать input (bidirectional context) при сохранении генеративных возможностей.

## Полная таблица сравнения архитектур

| | Decoder-only | Encoder-only | Encoder-Decoder | Prefix LM |
|--|---|---|---|---|
| **Пример** | GPT, LLaMA | BERT | T5, BART | UL2 |
| **Attention** | Causal | Bidirectional | Bidirec + Causal + Cross | Bidirec prefix + Causal |
| **Training** | Next token | Masked tokens | Seq2seq | Hybrid |
| **NLU quality** | Хорошая (при scale) | Лучшая | Очень хорошая | Очень хорошая |
| **NLG quality** | Лучшая | Нет | Хорошая | Хорошая |
| **Zero-shot** | Лучший | Нет | Ограниченный | Хороший |
| **Few-shot** | Лучший | Нет | Ограниченный | Хороший |
| **Scaling** | Лучший | Ограниченный | Хороший | Хороший |
| **Простота** | Максимальная | Высокая | Средняя | Средняя |
| **Популярность (2024+)** | Доминирует | Embeddings, NER | Niche (MT, medical) | Нишевый |

## Use cases vs Encoder-only

Из [[02 Areas/ML & DL/Papers/Harnessing the Power of LLMs in Practice- A Survey on ChatGPT and Beyond]]:

| Задача | Encoder-only (fine-tuned) | Decoder-only (LLM) |
|--------|--------------------------|-------------------|
| Text classification (много данных) | Лучше | — |
| NER (CoNLL03) | ~2x лучше | — |
| Summarization (по метрикам) | ROUGE лучше | — |
| Summarization (по human eval) | — | Люди предпочитают |
| Open-ended generation | — | Лучше |
| Few-shot / zero-shot | — | Лучше |
| OOD / adversarial data | — | Лучше |
| Chatbot / real-world tasks | — | Лучше |
| Knowledge-intensive QA | — | Лучше |

**Когда encoder-only всё ещё лучше:** fine-tuned BERT/DeBERTa остаётся конкурентоспособным для structured NLU задач (NER, classification) при наличии labeled данных и фиксированном deployment. Cheaper inference, better latency.

## Key papers

- [[02 Areas/ML & DL/Papers/GPT 2.0]] — zero-shot generalization; BPE tokenization; 1.5B
- [[02 Areas/ML & DL/Papers/GPT 3.0]] — few-shot learning at scale; 175B; in-context learning
- [[02 Areas/ML & DL/Papers/LLaMA]] — эффективные open decoder-only LLM (7B-65B); стандарт архитектурных улучшений
- [[02 Areas/ML & DL/Papers/LLaMA 2]] — 70B + RLHF; chat variants

## In courses

- [[02 Areas/ML & DL/Courses/SHAD LLM/Week 1 — Intro to LLMs|SHAD LLM — Week 1]]: QA, summarization, генерация текста

## Related concepts

- [[02 Areas/ML & DL/Concepts/Training/Causal Language Modeling|Causal Language Modeling]]
- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]]
- [[02 Areas/ML & DL/Concepts/Inference/Sampling|Sampling]]
- [[02 Areas/ML & DL/Concepts/Inference/In-context Learning|In-context Learning]]
- [[02 Areas/ML & DL/Concepts/Inference/Prompting|Prompting]]
- [[02 Areas/ML & DL/Concepts/Architectures/Encoder-only|Encoder-only]]
- [[02 Areas/ML & DL/Concepts/Architectures/Encoder-Decoder|Encoder-Decoder]]

## Дополнительные ресурсы

- [Cameron Wolfe — Decoder-Only Transformers](https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse) — лучший обзор почему decoder-only победил
- [Yi Tay — What happened to BERT & T5?](https://www.yitay.net/blog/model-architecture-blogpost-encoders-prefixlm-denoising) — взгляд от Google Research на архитектурные выборы
