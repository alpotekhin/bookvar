---
title: "Language Models are Unsupervised Multitask Learners"
url: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
authors: [Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever]
year: 2019
date_reviewed: 2026-04-12
type: source-note
status: legacy
category: paper
tags:
  - GPT
  - LLM
  - arch
  - zero-shot
Date: 2019-01-02
Organization: OpenAI
Parent item: null  # GPT-1 paper page does not exist in vault
Status: Done
concepts:
  - "[[02 Areas/ML & DL/Concepts/Architectures/GPT-2|GPT-2]]"
  - "[[02 Areas/ML & DL/Concepts/Architectures/Decoder-only|Decoder-only]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Causal Language Modeling|CLM]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]]"
  - "[[02 Areas/ML & DL/Concepts/Inference/Zero-shot Learning|Zero-shot Learning]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/Tokenization|Tokenization]]"
raw: null
---

# Language Models are Unsupervised Multitask Learners (GPT-2)

**Authors:** Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever (OpenAI)
**Published:** 2019
**URL:** https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

## TL;DR

Авторы показали, что достаточно большая языковая модель (1.5B параметров), обученная на разнообразном корпусе текстов (WebText, 40GB), может решать downstream-задачи NLP **в zero-shot режиме** — без какого-либо fine-tuning или task-specific обучения. GPT-2 достигает SOTA на 7 из 8 языковых бенчмарков в zero-shot, демонстрируя, что language modeling неявно обучает модель множеству задач одновременно.

## Problem

Стандартная парадигма NLP в 2018-2019: pre-train + fine-tune (BERT, GPT-1). Каждая задача требует отдельного supervised dataset и дообучения. Это:
1. **Плохо масштабируется** — нужны labeled данные для каждой новой задачи
2. **Не отражает способности человека** — люди решают новые задачи без примеров, просто по инструкции
3. **Узкие модели** — fine-tuned модели хорошо работают на одной задаче, но не generalize

Вопрос: может ли достаточно мощная LM, обученная на достаточно разнообразных данных, решать задачи без явного обучения на них?

## Method

### Подход: задачи как текст (§2)

Ключевой инсайт: любую NLP-задачу можно сформулировать как предсказание следующего токена. Например:
- Перевод: `translate english to french: [text] =`
- QA: `answer the question: [question] context: [passage]`
- Summarization: `TL;DR:` после текста

Модель не получает явных task-специфичных сигналов — задача определяется контекстом (conditioning).

### Данные: WebText (§2.2)

- Собран по ссылкам с Reddit с karma >= 3 (фильтр качества через "человеческую курацию")
- ~8M документов, 40GB текста после дедупликации
- Исключена Wikipedia (чтобы не мешать evaluation на WikiText)
- Намного разнообразнее предыдущих корпусов (BookCorpus, 1B Word Benchmark)

### Токенизация: Byte Pair Encoding (§2.2)

- BPE на уровне байтов (Byte-level BPE), а не символов или слов
- Словарь: 50,257 токенов
- Позволяет кодировать любой UTF-8 текст без `<UNK>` токенов
- Специальное правило: BPE не мержит через границы категорий символов (буквы, цифры, пунктуация) — предотвращает субоптимальные мержи типа `dog.`, `dog!`, `dog?`

### Архитектура (§2.3)

Decoder-only Transformer (как GPT-1), с модификациями:

| Параметр | GPT-2 Small | GPT-2 Medium | GPT-2 Large | GPT-2 XL |
|----------|-------------|--------------|-------------|----------|
| Параметры | 117M | 345M | 762M | **1.5B** |
| Слои | 12 | 24 | 36 | **48** |
| d_model | 768 | 1024 | 1280 | **1600** |
| Heads | 12 | 16 | 20 | **25** |
| Контекст | 1024 | 1024 | 1024 | **1024** |

Изменения относительно GPT-1:
- **Layer Normalization перемещена на вход** каждого sub-block (pre-norm вместо post-norm) — стабилизирует обучение глубоких сетей
- **Дополнительная Layer Norm** после финального self-attention блока
- **Инициализация весов** residual layers масштабирована как 1/√N (N — число residual layers) — предотвращает взрывной рост активаций
- Контекст увеличен с 512 до **1024 токенов**
- Batch size увеличен с 64 до **512**

## Key Results

### Zero-shot Performance (Table 2-3)

| Задача | Метрика | GPT-2 (zero-shot) | Предыдущий SOTA (supervised) |
|--------|---------|-------------------|------------------------------|
| LAMBADA | Accuracy | **63.24%** | 59.23% |
| LAMBADA | Perplexity | **8.63** | 99.8 |
| CBT-NE | Accuracy | **89.05%** | 85.3% |
| CBT-CN | Accuracy | 93.45% | **96.3%** |
| WikiText-2 | PPL | **29.41** | 39.14 |
| PTB | PPL | **35.76** | 46.54 |
| 1B Word | PPL | **44.575** | 23.7 (supervised on same data) |
| Winograd | Accuracy | 70.70% | — |
| CoQA | F1 | 55.0 | 89.8 (supervised) |

GPT-2 достигает SOTA на **7 из 8** language modeling бенчмарков в zero-shot. На задачах, требующих reasoning (CoQA), отстает от supervised моделей, но показывает нетривиальную способность.

### Children's Book Test (§3.2)

На CBT Named Entities: 89.05% zero-shot (предыдущий supervised SOTA 85.3%). Модель "понимает" кореференцию и контекст без обучения на задаче.

### Reading Comprehension (§3.4)

На CoQA (conversational QA): 55 F1 zero-shot. Для сравнения: supervised baseline 89.8 F1. Но 55 F1 без единого примера — это значительный результат, показывающий emergent ability к чтению и ответам.

### Summarization (§3.6)

При добавлении `TL;DR:` после статьи, GPT-2 генерирует разумные саммари. ROUGE-L ~15 на CNN/DailyMail (supervised SOTA ~30), но качественно текст связный.

### Translation (§3.7)

EN→FR: 5 BLEU zero-shot (supervised SOTA ~40 BLEU). FR→EN: 11.5 BLEU. Слабый результат, но модель вообще не обучалась на параллельных корпусах — перевод как emergent behavior.

### Scaling (§3)

Все метрики монотонно улучшаются с ростом модели (117M → 345M → 762M → 1.5B), и log-loss на WebText не показывает насыщения. Это подтверждает, что дальнейшее масштабирование даст еще лучшие результаты (что подтвердилось в GPT-3).

## My notes

- GPT-2 — концептуально переломная работа: первая демонстрация того, что LM "неявно учится" выполнять NLP-задачи без supervised сигнала. Это прямой предшественник парадигмы "scale is all you need", развитой в GPT-3.
- Byte-level BPE стала стандартом токенизации для почти всех последующих LLM (GPT-3, LLaMA, Mistral). Решает проблему OOV и многоязычности одним приемом.
- Pre-norm архитектура (LayerNorm перед attention/FFN) тоже стала стандартом де-факто — LLaMA, Mistral, Falcon все используют pre-norm.
- Инициализация 1/√N для residual — простой но критически важный трюк для обучения глубоких трансформеров. Позже заменен/дополнен другими техниками (DeepNorm в DeepSeek).
- WebText approach (фильтрация через Reddit karma) — предшественник более сложных data curation пайплайнов (The Pile, RedPajama, FineWeb).
- Работа изначально вызвала дискуссию об AI safety: OpenAI задержали релиз полной модели из-за опасений по генерации фейковых текстов. Это один из первых случаев "responsible disclosure" в AI.
- Контекст 1024 токена — сегодня кажется крошечным (GPT-4: 128K, Gemini: 1M), но для 2019 это был значительный шаг вперёд от 512 в GPT-1.
- Интересно, что модель не обучалась на Wikipedia, но все равно побила PPL на WikiText-2 — свидетельство того, что разнообразие данных важнее domain match.
