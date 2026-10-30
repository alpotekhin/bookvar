---
title: "Improving Language Understanding by Generative Pre-Training"
url: https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
authors: [Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever]
year: 2018
date_reviewed: 2026-04-13
type: source-note
status: legacy
category: paper
tags:
  - LLM
  - GPT
  - pre-training
  - fine-tuning
Organization: OpenAI
concepts:
  - "[[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Fine-tuning|Fine-tuning]]"
  - "[[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]]"
---

# GPT-1: Improving Language Understanding by Generative Pre-Training

## Ключевая идея

GPT-1 (Radford et al., 2018) — первая работа, системно показавшая, что **generative pre-training на неразмеченных данных + discriminative fine-tuning** даёт state-of-the-art результаты на широком спектре NLP-задач. До GPT-1 NLP полагался на supervised learning с task-specific архитектурами. Эта работа установила парадигму **pre-train → fine-tune**, которая определила развитие NLP на годы вперёд.

## Мотивация

К 2018 году главная проблема NLP — **нехватка размеченных данных**. Для каждой задачи (классификация, NLI, QA) нужны дорогостоящие аннотации. Неразмеченного текста — терабайты, но как его использовать?

Предыдущие подходы (word2vec, ELMo) использовали unsupervised pre-training для обучения представлений слов, но:
- word2vec — контекстно-независимые embeddings
- ELMo — bidirectional LSTM, embeddings зависят от контекста, но архитектура не масштабируется

GPT-1 предложил использовать **Transformer decoder** для language modeling на большом корпусе, а затем дообучать на downstream tasks.

## Архитектура

### Модель

GPT-1 использует **decoder-only Transformer**:
- **12 слоёв** (transformer blocks)
- **768 hidden size**, 12 attention heads
- **3072** размерность FFN (4x hidden)
- **117M параметров**
- Контекст: **512 токенов**
- Learned positional embeddings (не синусоидальные)
- GELU activation (вместо ReLU)
- BPE tokenizer (~40 000 merges)

По сравнению с оригинальным [[02 Areas/ML & DL/Papers/Attention Is All You Need|Transformer]] (encoder-decoder), GPT-1 использует **только decoder** — causal (unidirectional) self-attention, где каждый токен «видит» только предыдущие.

### Этап 1: Unsupervised Pre-training

Обучение стандартным языковым моделированием (causal language modeling) на BookCorpus (~7000 книг, ~800M слов):

$$\mathcal{L}_1 = -\sum_{i} \log P(u_i \mid u_{i-k}, \ldots, u_{i-1}; \Theta)$$

где $u_i$ — токены, $k$ — размер контекстного окна.

**Почему BookCorpus:** длинные связные тексты (целые книги) обеспечивают long-range dependencies, которые невозможно выучить из коротких предложений.

### Этап 2: Supervised Fine-tuning

Для каждой downstream задачи добавляется **task-specific linear head** поверх последнего hidden state:

$$P(y \mid x^1, \ldots, x^m) = \text{softmax}(h_l^m W_y)$$

где $h_l^m$ — hidden state последнего токена на последнем слое, $W_y$ — обучаемая проекция.

**Ключевой трюк: auxiliary language modeling loss.** При fine-tuning используется комбинированный loss:

$$\mathcal{L}_3 = \mathcal{L}_2(C) + \lambda \cdot \mathcal{L}_1(C)$$

где $\mathcal{L}_2$ — supervised loss для задачи, $\mathcal{L}_1$ — language modeling loss на тех же данных. Авторы нашли, что добавление LM loss **улучшает generalization** и ускоряет convergence. $\lambda = 0.5$.

### Input Transformations

Для задач с нестандартным форматом входа (пара предложений, множественный выбор) используются **input transformations** — специальная разметка входа:

| Задача | Формат входа |
|--------|-------------|
| Classification | `[Start] text [Extract]` |
| Entailment | `[Start] premise [Delim] hypothesis [Extract]` |
| Similarity | `[Start] text1 [Delim] text2 [Extract]` + обратный порядок |
| Multiple Choice | `[Start] context [Delim] answer_i [Extract]` для каждого варианта |

Это позволяет использовать **одну pre-trained модель** для всех задач, меняя только формат входа и linear head.

## Результаты

GPT-1 достиг state-of-the-art на **9 из 12** тестируемых NLP-задач:

| Задача | Бенчмарк | GPT-1 | Предыдущий SOTA |
|--------|----------|-------|----------------|
| Natural Language Inference | MNLI | 82.1 | 80.6 |
| Question Answering | RACE | 59.0 | 53.3 |
| Commonsense Reasoning | Story Cloze | 86.5 | 77.6 |
| Sentiment Analysis | SST-2 | 91.3 | 90.2 |
| Semantic Similarity | STS-B | 82.0 | 81.0 |

Улучшение на commonsense reasoning (+8.9%) и question answering (+5.7%) особенно впечатляет — эти задачи требуют «понимания» языка, а не pattern matching.

## Ablation Studies

Авторы провели важные ablation experiments:

1. **Без pre-training** — performance падает на ~15% на всех задачах. Pre-training критически важен.
2. **Без auxiliary LM loss** — ухудшение на крупных датасетах, но не на маленьких. LM loss помогает как регуляризатор.
3. **LSTM вместо Transformer** — ухудшение на ~5%. Transformer лучше capture long-range dependencies.
4. **Количество слоёв** — performance растёт монотонно с 1 до 12 слоёв.

## Zero-shot поведение

Интересное наблюдение: даже **без fine-tuning** GPT-1 показывает нетривиальный performance на некоторых задачах. Авторы отмечают, что performance растёт по мере pre-training — модель постепенно учится решать NLP-задачи как побочный эффект language modeling.

Это наблюдение стало ключевым для [[02 Areas/ML & DL/Papers/GPT 2.0|GPT-2]], который полностью отказался от fine-tuning в пользу zero-shot/few-shot prompting.

## Историческое значение

GPT-1 — **переломная работа** в NLP:

1. **Установил парадигму pre-train → fine-tune**, которая доминировала в NLP 2018-2022 (GPT, [[02 Areas/ML & DL/Papers/BERT|BERT]], T5)
2. **Показал, что decoder-only Transformers** — жизнеспособная архитектура для NLU (а не только генерации)
3. **Продемонстрировал transfer learning** в NLP на уровне, сравнимом с ImageNet в CV
4. **Заложил основу для масштабирования** — если 117M параметров дают такие результаты, что будет с 1B? 100B? Ответ дали [[02 Areas/ML & DL/Papers/GPT 2.0|GPT-2]] (1.5B), [[02 Areas/ML & DL/Papers/GPT 3.0|GPT-3]] (175B) и [[02 Areas/ML & DL/Papers/GPT 4.0|GPT-4]]

## Сравнение с BERT

Через несколько месяцев после GPT-1 вышел [[02 Areas/ML & DL/Papers/BERT|BERT]] (Google, 2018), который использовал **encoder-only** Transformer с bidirectional pre-training (Masked Language Modeling). BERT превзошёл GPT-1 на большинстве NLU-задач, потому что bidirectional context даёт более богатые представления для понимания текста.

Но GPT-линейка выиграла в долгосрочной перспективе: decoder-only архитектура лучше масштабируется и естественно поддерживает генерацию — ключевое свойство для modern LLM.

## Related concepts

- [[02 Areas/ML & DL/Papers/GPT 2.0|GPT-2]] — масштабирование до 1.5B, отказ от fine-tuning
- [[02 Areas/ML & DL/Papers/GPT 3.0|GPT-3]] — 175B, few-shot learning как основной режим
- [[02 Areas/ML & DL/Papers/GPT 4.0|GPT-4]] — мультимодальная модель
- [[02 Areas/ML & DL/Papers/BERT|BERT]] — альтернативный подход (encoder-only, bidirectional)
- [[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]] — ключевой этап, обоснованный GPT-1
- [[02 Areas/ML & DL/Concepts/Training/Fine-tuning|Fine-tuning]] — второй этап парадигмы GPT-1
