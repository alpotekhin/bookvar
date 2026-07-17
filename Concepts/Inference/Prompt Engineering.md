---
title: "Prompt Engineering"
aliases: [prompt design, prompt optimization, промпт-инжиниринг]
type: concept
status: legacy
category: Inference
papers:
  - "[[02 Areas/ML & DL/Papers/Pre-train, Prompt, and Predict- A Systematic Survey of Prompting Methods in Natural Language Processing]]"
  - "[[02 Areas/ML & DL/Papers/Harnessing the Power of LLMs in Practice- A Survey on ChatGPT and Beyond]]"
  - "[[02 Areas/ML & DL/Papers/COT]]"
courses: []
sources:
  - "[Lilian Weng — Prompt Engineering (2023)](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)"
  - "[OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)"
  - "[Anthropic Prompt Engineering Guide](https://docs.anthropic.com/claude/docs/prompt-engineering)"
---

# Prompt Engineering

## Зачем это нужно: от интуиции к системе

Prompting — это **что** мы делаем (форматируем вход). Prompt Engineering — это **как** мы это делаем систематически, чтобы получить максимальный performance на конкретной задаче.

Одна и та же задача с разными промптами может дать accuracy от 30% до 90%. Prompt Engineering превращает «угадывание правильной формулировки» в **инженерный процесс** с воспроизводимыми результатами.

Аналогия: промптинг — это умение задавать вопросы. Prompt Engineering — это систематическое исследование того, какие вопросы дают лучшие ответы, с тестированием гипотез и итерацией.

## Ключевые техники

### 1. Chain-of-Thought (CoT) Prompting

Из [[02 Areas/ML & DL/Papers/COT]] (Wei et al., 2022) — одна из самых влиятельных техник.

**Идея**: вместо прямого ответа попросить модель рассуждать **пошагово**. Добавляем примеры с промежуточными шагами:

```
Q: У Роджера 5 теннисных мячей. Он покупает ещё 2 упаковки по 3 мяча.
   Сколько у него мячей?
A: Изначально у Роджера 5 мячей. Он покупает 2 * 3 = 6 мячей.
   Итого: 5 + 6 = 11 мячей.
   Ответ: 11
```

**Результаты** (PaLM-540B, GSM8K): standard prompting 18% → CoT prompting **57%**. Тройное улучшение на math reasoning.

**Почему работает**: модель «раскладывает» сложную задачу на простые подзадачи. Каждый промежуточный шаг — отдельная простая операция, которую LLM выполняет надёжно.

**Zero-shot CoT**: добавление фразы "Let's think step by step" без примеров. Простейший trigger, который активирует reasoning mode.

### 2. Few-shot Example Selection

Не все примеры одинаково полезны. Ключевые факторы:

**Релевантность примеров**: KATE (Liu et al., 2022) — kNN-retrieved examples (ближайшие к целевому входу в embedding space) **значительно** лучше случайных. На GPT-3: +10-15% accuracy.

**Порядок примеров**: Lu et al. (2022) показали, что порядок few-shot examples может менять accuracy от near-random до near-SOTA. Решение: entropy-based оптимизация порядка.

**Разнообразие примеров**: примеры должны покрывать разные edge cases и категории, не повторяя один паттерн.

**Label balance**: при classification равное количество примеров каждого класса снижает bias.

### 3. Structured Output

Указание **формата ответа** значительно улучшает парсинг и качество:

```
Respond in the following JSON format:
{
  "sentiment": "positive" | "negative",
  "confidence": 0.0-1.0,
  "key_phrase": "string"
}
```

Форматы от простого к сложному:
- **Simple answer**: "Answer with just 'yes' or 'no'" — минимизирует лишний текст
- **Bullet points**: структурированный ответ для complex queries
- **JSON/XML**: машиночитаемый output для API
- **Markdown tables**: сравнительный анализ

### 4. Role Prompting

Задание **роли** (persona) модели через system prompt:

```
You are an expert Python developer with 15 years of experience.
Review the following code for bugs and performance issues.
```

Эмпирически: role prompting улучшает результат на domain-specific задачах. Механизм: модель «активирует» знания, ассоциированные с ролью в training data.

**Caveat**: role prompting не даёт модели новых знаний — только переключает distributional prior.

### 5. Self-Ask и Decomposition

Разбиение сложного вопроса на подвопросы:

```
Question: Was the founder of Apple born before the inventor of the World Wide Web?
Are follow-up questions needed here: Yes.
Follow-up: Who founded Apple?
Intermediate answer: Steve Jobs.
Follow-up: When was Steve Jobs born?
Intermediate answer: February 24, 1955.
Follow-up: Who invented the World Wide Web?
Intermediate answer: Tim Berners-Lee.
Follow-up: When was Tim Berners-Lee born?
Intermediate answer: June 8, 1955.
So the final answer is: Yes.
```

Полезно для multi-hop reasoning, где модель должна связать несколько фактов.

### 6. Retrieval-Augmented Prompting

Добавление **релевантного контекста** из внешних источников:

```
Context: [retrieved documents]
Based on the above context, answer the following question:
Q: {question}
```

Ключевой паттерн для [[02 Areas/ML & DL/Concepts/Retrieval/Retrieval-Augmented Generation|RAG]] систем. Снижает hallucination, обеспечивает актуальность информации.

## Автоматическая оптимизация промптов

### Discrete Search

| Метод | Как работает | Trade-off |
|-------|-------------|-----------|
| **AutoPrompt** | Gradient-based поиск токенов | Быстро, но промпты нечитаемые |
| **LPAQA** | Mining n-грамм из корпуса | Простой, но ограниченный |
| **APE** (Auto-Prompt-Engineer) | LLM генерирует и оценивает кандидаты | Эффективный, но дорогой |

### Continuous Optimization (Soft Prompts)

| Метод | Что обучается | Число параметров |
|-------|---------------|------------------|
| **Prefix Tuning** | Vectors на всех слоях | ~0.1% от LM |
| **Prompt Tuning** | Vectors только на input | ~0.01% от LM |
| **P-Tuning v2** | Deep continuous prompts | ~1-3% от LM |

При 11B+ параметрах модели prompt tuning **конкурирует с full fine-tuning** при доле стоимости.

## Answer Engineering: не забываем про формат ответа

**Verbalizer** — маппинг между ответными токенами и метками:

| Задача | Manual verbalizer | Автоматический |
|--------|-------------------|----------------|
| Sentiment | "good"/"bad" → pos/neg | Оптимизация через gradient |
| NLI | "yes"/"maybe"/"no" | Calibration через null-input |
| Topic | Category names | Learned embeddings |

Неправильный verbalizer может стоить 10-20% accuracy. «terrible» лучше чем «bad» для negative sentiment на некоторых моделях — потому что training data содержит разные distribution'ы для этих слов.

## Calibration: борьба с bias'ом

LLM bias'ированы к определённым ответам независимо от входа:
- GPT-3 предпочитает последний few-shot example (recency bias)
- Модели чаще выбирают «negative» для sentiment (majority label bias)
- Длинные ответы оцениваются выше коротких (verbosity bias)

**Contextual Calibration** (Zhao et al., 2021): оцениваем bias на null-input (пустой вход) → корректируем logits. Простая техника, дающая +5-15% accuracy на classification.

## Limitations и open problems

Из [[02 Areas/ML & DL/Papers/Pre-train, Prompt, and Predict- A Systematic Survey of Prompting Methods in Natural Language Processing]] (§10):

1. **Structured inputs**: нет хорошей методологии для таблиц, графов, деревьев в промптах
2. **Joint optimization**: template и answer space обычно оптимизируются последовательно, а не совместно
3. **Transferability**: оптимальный промпт для одной задачи не переносится на другую
4. **Calibration**: LLM confidence плохо откалиброван — наиболее уверенный ответ != правильный
5. **Evaluation**: нет стандартных метрик качества промптов

## Практический чеклист

```
1. Начни с clear instruction (что делать, в каком формате)
2. Добавь role (если domain-specific задача)
3. Дай 3-5 diverse examples (few-shot)
4. Попроси reasoning (CoT, если задача сложная)
5. Укажи format output (JSON, list, etc.)
6. Тестируй на 20+ примерах, итерируй
7. Если accuracy < target: попробуй AutoPrompt или RAG
```

## Key papers

- [[02 Areas/ML & DL/Papers/Pre-train, Prompt, and Predict- A Systematic Survey of Prompting Methods in Natural Language Processing]] — полная таксономия (2021)
- [[02 Areas/ML & DL/Papers/COT]] — chain-of-thought как ключевой prompt engineering trick
- [[02 Areas/ML & DL/Papers/Harnessing the Power of LLMs in Practice- A Survey on ChatGPT and Beyond]] — практические рекомендации для production

## Related concepts

- [[02 Areas/ML & DL/Concepts/Inference/Prompting|Prompting]] — базовая концепция промптинга
- [[02 Areas/ML & DL/Concepts/Inference/In-context Learning|In-context Learning]] — few-shot обучение через контекст
- [[02 Areas/ML & DL/Concepts/Inference/Chain of Thought|Chain of Thought]] — пошаговое рассуждение
- [[02 Areas/ML & DL/Concepts/Inference/Zero-shot Learning|Zero-shot Learning]] — выполнение задач без примеров
- [[02 Areas/ML & DL/Concepts/NLP/Prompt-based Learning|Prompt-based Learning]] — парадигма обучения через промпты

## Дополнительные ресурсы

- [Lilian Weng — Prompt Engineering (2023)](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) — наиболее полный практический обзор
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering) — рекомендации от OpenAI
- [Anthropic Prompt Engineering Guide](https://docs.anthropic.com/claude/docs/prompt-engineering) — рекомендации от Anthropic
- [DAIR.AI — Prompt Engineering Guide](https://www.promptingguide.ai/) — community-driven руководство
