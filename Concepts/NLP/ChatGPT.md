---
title: "ChatGPT"
aliases: [ChatGPT, GPT-3.5-turbo, GPT-4-turbo, чатгпт]
type: concept
category: NLP
papers:
  - "[[02 Areas/ML & DL/Papers/InstructGPT]]"
  - "[[02 Areas/ML & DL/Papers/GPT 4.0]]"
  - "[[02 Areas/ML & DL/Papers/Harnessing the Power of LLMs in Practice- A Survey on ChatGPT and Beyond]]"
  - "[[02 Areas/ML & DL/Papers/GPT 3.0]]"
courses:
  - "[[02 Areas/ML & DL/Courses/SHAD LLM/Week 1 — Intro to LLMs|SHAD LLM — Week 1]]"
sources:
  - "[OpenAI Blog — Introducing ChatGPT (Nov 2022)](https://openai.com/blog/chatgpt)"
  - "[OpenAI — GPT-4 Technical Report (Mar 2023)](https://arxiv.org/abs/2303.08774)"
  - "[Lilian Weng — Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)"
---

# ChatGPT

## Что это такое

ChatGPT --- диалоговый AI-ассистент от OpenAI, запущенный **30 ноября 2022 года**. Построен на базе GPT-3.5, дообученного с [[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]] для ведения многошагового диалога. Набрал **100 миллионов пользователей за 2 месяца** --- быстрейший продуктовый рост в истории (для сравнения: TikTok --- 9 месяцев, Instagram --- 2.5 года).

ChatGPT не является отдельной архитектурой --- это **продукт**, объединивший три компонента: мощную базовую модель ([[02 Areas/ML & DL/Concepts/Architectures/GPT-3|GPT-3.5/4]]), alignment через RLHF ([[02 Areas/ML & DL/Papers/InstructGPT|InstructGPT]]) и удобный интерфейс для диалога.

## Как это устроено: от GPT-3 к ChatGPT

### Базовая модель: GPT-3 / GPT-3.5

[[02 Areas/ML & DL/Papers/GPT 3.0|GPT-3]] (175B параметров) --- авторегрессивная [[02 Areas/ML & DL/Concepts/Architectures/Decoder-only|decoder-only]] модель, обученная на 300B токенов. Умеет генерировать текст, но без alignment ведёт себя непредсказуемо: может продолжить текст вместо ответа на вопрос, генерировать токсичный контент, «галлюцинировать».

GPT-3.5 --- промежуточная версия с дополнительным обучением на коде (Codex) и инструкциях.

### RLHF Pipeline: три шага alignment

Из [[02 Areas/ML & DL/Papers/InstructGPT]] --- pipeline, который превращает базовую LM в полезного ассистента:

**Шаг 1. Supervised Fine-Tuning (SFT):**
- Наёмные аннотаторы пишут **идеальные ответы** на пользовательские запросы
- GPT-3 дообучается на ~13K таких пар (prompt, ideal_response)
- Модель учится формату: отвечать на вопросы, а не просто продолжать текст

**Шаг 2. Reward Model (RM):**
- Модель генерирует **несколько ответов** на один запрос
- Аннотаторы **ранжируют** ответы от лучшего к худшему
- На этих парных сравнениях обучается reward model (~33K comparisons)
- RM предсказывает скаляр: «насколько хорош ответ с точки зрения человека»

**Шаг 3. PPO (Proximal Policy Optimization):**
- RL fine-tuning: модель генерирует ответ, RM оценивает его
- Модель обновляется, чтобы максимизировать reward
- KL-penalty не даёт модели слишком далеко уйти от base model (чтобы не «забыть» язык)

$$\text{objective}(\theta) = E_{x \sim D, y \sim \pi_\theta}\left[R_\phi(x, y) - \beta \cdot D_{KL}(\pi_\theta \| \pi_{\text{ref}})\right]$$

### Результат: InstructGPT vs GPT-3

Из [[02 Areas/ML & DL/Papers/InstructGPT]]:
- Аннотаторы предпочитают InstructGPT (1.3B) ответы GPT-3 (175B) в **85% случаев** --- модель в 100x меньше, но субъективно лучше
- Hallucination rate ~2x ниже
- Toxicity ~25% ниже
- InstructGPT --- предшественник ChatGPT, по сути тот же pipeline

### Conversational Design

ChatGPT добавляет к InstructGPT:
- **Multi-turn context:** модель видит всю историю диалога (system + user + assistant turns)
- **System prompt:** скрытая инструкция, задающая поведение ("You are a helpful assistant...")
- **Safety guardrails:** отказ на harmful requests, предупреждения о неопределённости
- **Chat Markup Language (ChatML):** специальные токены для разделения ролей

```
<|system|> You are a helpful assistant.
<|user|> What is self-attention?
<|assistant|> Self-attention is a mechanism...
```

## Хронология версий

| Версия | База | Контекст | Дата | Ключевое |
|--------|------|:--------:|------|----------|
| ChatGPT | GPT-3.5 | 4K | Nov 2022 | Первый релиз |
| ChatGPT Plus | GPT-4 | 8K/32K | Mar 2023 | Мультимодальность, reasoning |
| GPT-3.5-turbo | Оптим. GPT-3.5 | 4K/16K | Mar 2023 | 10x дешевле API |
| GPT-4-turbo | GPT-4 | 128K | Nov 2023 | Длинный контекст, JSON mode |
| GPT-4o | Оптим. GPT-4 | 128K | May 2024 | Аудио, видео, скорость |

## GPT-4: качественный скачок

Из [[02 Areas/ML & DL/Papers/GPT 4.0|GPT-4 Technical Report]]:

**Профессиональные экзамены:**

| Экзамен | GPT-3.5 (ChatGPT) | GPT-4 | Человек (median) |
|---------|:-----------------:|:-----:|:----------------:|
| Bar Exam (Uniform) | ~10th percentile | **~90th percentile** | 50th |
| SAT Math | 70th percentile | **89th percentile** | 50th |
| GRE Verbal | ~63rd percentile | **~99th percentile** | 50th |
| AP Biology | 3/5 | **5/5** | --- |

GPT-4 --- мультимодальная модель: принимает изображения на вход. Архитектура не раскрыта, но предположительно Mixture of Experts.

## Влияние на NLP и индустрию

### Competitive response: гонка LLM-ассистентов

ChatGPT запустил «гонку вооружений»:
- **Google:** Bard (Feb 2023) -> Gemini (Dec 2023)
- **Anthropic:** Claude (Mar 2023) -> Claude 2 -> Claude 3
- **Meta:** LLaMA (Feb 2023) -> LLaMA 2 (Jul 2023) -> LLaMA 3 (Apr 2024) --- open-source
- **Mistral:** Mistral 7B (Sep 2023), Mixtral (Dec 2023) --- open-source

### Benchmark сдвиг

До ChatGPT основной benchmark --- [[02 Areas/ML & DL/Concepts/NLP/GLUE|GLUE/SuperGLUE]] (classification, NLI). После ChatGPT:
- **MMLU** (57 предметов, knowledge) --- стандарт для LLM
- **HumanEval** (код) --- оценка coding abilities
- **Arena ELO** (LMSYS) --- human preference ranking в live setting

### Paradigm shift в NLP

| До ChatGPT | После ChatGPT |
|-----------|--------------|
| Fine-tune отдельную модель на каждую задачу | Одна модель через промпты |
| Encoder-only (BERT) для NLU | Decoder-only для всего |
| Академические benchmarks (GLUE) | Практические задачи (coding, writing) |
| ML-инженер нужен | Пользователь пишет промпт |

## Инциденты и проблемы

Из [[02 Areas/ML & DL/Papers/Harnessing the Power of LLMs in Practice- A Survey on ChatGPT and Beyond]] (Section 5):

| Инцидент | Дата | Суть |
|----------|------|------|
| **Samsung data leak** | Mar 2023 | Инженеры вставили proprietary код; OpenAI хранит inputs |
| **Italy GDPR ban** | Mar 2023 | Garante заблокировал ChatGPT за нарушение GDPR |
| **Lawyer hallucination** | Jun 2023 | Юрист подал в суд фиктивные прецеденты, сгенерированные ChatGPT |
| **Copyright lawsuits** | 2023--2024 | NYT, авторы книг --- иски о нарушении авторских прав |

**Системные проблемы:**
- **Hallucinations:** модель уверенно генерирует несуществующие факты
- **Sycophancy:** модель соглашается с пользователем, даже когда он неправ
- **Temporal cutoff:** знания ограничены датой обучения
- **Privacy:** входы пользователей могут использоваться для дальнейшего обучения

## Почему ChatGPT стал настолько важен

1. **Массовый adoption:** первый AI-продукт, понятный любому человеку --- не нужно уметь программировать
2. **Proof of concept RLHF:** InstructGPT/ChatGPT доказал, что alignment через human feedback превращает сырую LM в полезного ассистента
3. **Экономический impact:** создал рынок AI-ассистентов на десятки миллиардов долларов; Microsoft инвестировал $10B в OpenAI
4. **Демократизация NLP:** вместо «обучи модель на GPU-кластере» --- «напиши промпт в браузере»
5. **Сдвиг исследований:** NLP-сообщество переключилось с encoder-only моделей и GLUE на decoder-only модели, alignment, safety

## Key papers

- [[02 Areas/ML & DL/Papers/InstructGPT]] --- технические основы RLHF alignment (GPT-3 -> InstructGPT -> ChatGPT)
- [[02 Areas/ML & DL/Papers/GPT 4.0]] --- GPT-4 technical report, backend ChatGPT Plus
- [[02 Areas/ML & DL/Papers/Harnessing the Power of LLMs in Practice- A Survey on ChatGPT and Beyond]] --- систематический анализ capabilities и рисков
- [[02 Areas/ML & DL/Papers/GPT 3.0]] --- архитектурная основа (GPT-3)

## Related concepts

- [[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]] --- метод alignment, превращающий LM в ассистента
- [[02 Areas/ML & DL/Concepts/Training/Instruction Tuning|Instruction Tuning]] --- SFT на инструкциях
- [[02 Areas/ML & DL/Concepts/Architectures/GPT-3|GPT-3]] --- базовая архитектура
- [[02 Areas/ML & DL/Concepts/Architectures/GPT-4|GPT-4]] --- backend ChatGPT Plus
- [[02 Areas/ML & DL/Concepts/NLP/Prompt-based Learning|Prompt-based Learning]] --- парадигма, которую ChatGPT популяризировал
- [[02 Areas/ML & DL/Concepts/NLP/Adversarial Promting|Adversarial Prompting]] --- атаки на ChatGPT и защита

## Дополнительные ресурсы

- [OpenAI Blog — Introducing ChatGPT](https://openai.com/blog/chatgpt) --- оригинальный анонс
- [Lilian Weng — Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) --- как эффективно использовать ChatGPT
- [LMSYS Chatbot Arena](https://chat.lmsys.org/) --- live comparison LLM через human preference
