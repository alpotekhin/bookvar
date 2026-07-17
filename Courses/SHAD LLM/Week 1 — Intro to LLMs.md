---
title: "SHAD LLM — Week 1: Intro to LLMs"
course: "SHAD LLM"
week: 1
type: source-note
status: legacy
raw: "[[02 Areas/ML & DL/raw/courses/SHAD LLM/Week 1/Week 1]]"
concepts:
  - "[[02 Areas/ML & DL/Concepts/Architectures/Encoder-only|Encoder-only]]"
  - "[[02 Areas/ML & DL/Concepts/Architectures/Encoder-Decoder|Encoder-Decoder]]"
  - "[[02 Areas/ML & DL/Concepts/Architectures/Decoder-only|Decoder-only]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Instruction Tuning|Instruction Tuning]]"
  - "[[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Causal Language Modeling|CLM]]"
  - "[[02 Areas/ML & DL/Concepts/Inference/Sampling|Sampling]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/ChatGPT|ChatGPT]]"
---

# SHAD LLM — Week 1: Intro to LLMs

## Key points

- До эпохи больших трансформеров под каждую задачу обучался отдельный инстанс модели — не было универсальных моделей
- Eliza (1964) использовала Reflection и Deflection — ранний пример диалогового агента
- LLMs изменили парадигму: одна модель — много задач

## Архитектурные паттерны

- **[[02 Areas/ML & DL/Concepts/Architectures/Encoder-only|Encoder-only]]** — хорошо подходят для: Text classification, NER, NLU
- **[[02 Areas/ML & DL/Concepts/Architectures/Encoder-Decoder|Encoder-Decoder]]** — для: Machine translation, Text summarization, Text paraphrasing
- **[[02 Areas/ML & DL/Concepts/Architectures/Decoder-only|Decoder-only]]** — для: QA, Summarization и вообще для генерации текста

## Параметры генерации

- `context window`: ограничение на `len(prompt + completion)`
- **[[02 Areas/ML & DL/Concepts/Inference/Sampling|TopK]]** — берём K слов с наибольшей вероятностью, сэмплируем из них
- **[[02 Areas/ML & DL/Concepts/Inference/Sampling|TopP (nucleus)]]** — сэмплируем из слов, чья кумулятивная вероятность не превышает TopP

## Pipeline обучения GPT-3 / ChatGPT

1. **[[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]]** — GPT на данных интернета (CLM-цель)
2. **[[02 Areas/ML & DL/Concepts/Training/Instruction Tuning|SFT (Supervised Fine-Tuning)]]** — обучение на инструкциях
3. **[[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]]** — обучение с обратной связью от людей

**ChatGPT** = LLM(2020) + Instruct-tuning (2021) + adaptation for chat (2022)

## RLHF детали

- Reward Model (RM) обучается на датасете с парными сравнениями ответов — предсказывает, какой ответ лучше
- **Проблема**: по мере обновления PPO его выходы начинают отличаться от того, на чём обучалась RM → плохие оценки reward
  - **Решение**: KL-штраф, не дающий PPO-модели слишком далеко отклоняться от SFT-результата
- **Проблема**: чистый RL-objective ведёт к деградации на многих NLP-задачах
  - **Решение**: добавить auxiliary LM-objective на pre-training данных → вариант **PPO-ptx**

### RLHF vs SFT

| | RLHF | SFT |
|---|---|---|
| Training signal | Reward (nuanced) | Autoregressive loss |
| Uses model generations | ✅ RM critiques actual completions | ❌ не использует |
| Data efficiency | Выше | Ниже |
| Captures preference | Прямее | Косвенно |
| Feedback granularity | На всю генерацию | На каждый токен |

## Concepts covered

- [[02 Areas/ML & DL/Concepts/Architectures/Encoder-only|Encoder-only]]
- [[02 Areas/ML & DL/Concepts/Architectures/Encoder-Decoder|Encoder-Decoder]]
- [[02 Areas/ML & DL/Concepts/Architectures/Decoder-only|Decoder-only]]
- [[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]]
- [[02 Areas/ML & DL/Concepts/Training/Instruction Tuning|Instruction Tuning / SFT]]
- [[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]]
- [[02 Areas/ML & DL/Concepts/Training/Causal Language Modeling|Causal Language Modeling]]
- [[02 Areas/ML & DL/Concepts/Inference/Sampling|Sampling (TopK / TopP)]]
- [[02 Areas/ML & DL/Concepts/NLP/ChatGPT|ChatGPT]]

## Notes

- Курс даёт хороший интуитивный обзор зачем нужен RLHF и почему PPO-ptx лучше чистого PPO
- Раздел «Key Models» в оригинальных заметках содержит изображение (Untitled 2 3.png) без подписи — вероятно сравнительная таблица моделей; детали недоступны
