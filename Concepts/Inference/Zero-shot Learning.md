---
title: "Zero-shot Learning"
aliases: [zero-shot inference, zero-shot generalization, zero-shot prompting, zero-shot]
type: concept
status: legacy
category: Inference
papers:
  - "[[02 Areas/ML & DL/Papers/GPT 2.0]]"
  - "[[02 Areas/ML & DL/Papers/GPT 3.0]]"
  - "[[02 Areas/ML & DL/Papers/Flan-T5-PaLM]]"
courses: []
sources:
  - "[Radford et al. — Language Models are Unsupervised Multitask Learners (GPT-2, 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)"
  - "[Brown et al. — Language Models are Few-Shot Learners (GPT-3, 2020)](https://arxiv.org/abs/2005.14165)"
  - "[Chung et al. — Scaling Instruction-Finetuned Language Models (Flan, 2022)](https://arxiv.org/abs/2210.11416)"
---

# Zero-shot Learning

## Зачем это нужно: обобщение без примеров

В традиционном ML каждая новая задача требует размеченных данных и обучения (или хотя бы fine-tuning'а). Zero-shot Learning — парадигма, в которой модель выполняет задачу **без единого task-специфичного примера**, опираясь исключительно на знания из pre-training.

Представь, что человек, прочитавший миллион книг, впервые слышит задание «переведи это на французский» — и справляется, потому что видел примеры переводов в прочитанных книгах, хотя его никто не учил переводить целенаправленно. Именно так работает zero-shot: модель «помнит» паттерны из pre-training корпуса и применяет их к новым инструкциям.

## Парадигмы: от GPT-2 к ChatGPT

### GPT-2: zero-shot через task conditioning (2019)

Из [[02 Areas/ML & DL/Papers/GPT 2.0]] (Radford et al.) — первая систематическая оценка zero-shot capabilities LLM.

Ключевая идея: **переформулировать задачу как text completion**. Модель не получает инструкцию в явном виде — задача «закодирована» в формате текста:

| Задача | Формат | Что видит модель |
|--------|--------|------------------|
| Summarization | article + "TL;DR:" | Модель генерирует краткое содержание как «естественное продолжение» |
| Translation | text + "in French:" | Модель продолжает текст переводом |
| QA | "Q: {question} A:" | Модель генерирует ответ после «A:» |

Это работало, потому что WebText (40GB интернет-текста) содержал естественные примеры всех этих форматов. Модель не «понимает задание» — она продолжает текст в наиболее вероятном направлении.

**Результаты GPT-2** были скромными: конкурентоспособны с LSTM на некоторых задачах (LAMBADA accuracy 70.7%), но значительно хуже fine-tuned моделей. Главное открытие — **сам факт**, что языковая модель может выполнять задачи без обучения на них.

### GPT-3: систематическое сравнение zero/one/few-shot (2020)

Из [[02 Areas/ML & DL/Papers/GPT 3.0]] (Brown et al.) — формализация трёх режимов использования LLM:

| Режим | Что в промпте | Аналогия из жизни |
|-------|---------------|-------------------|
| **Zero-shot** | Только описание задачи | «Переведи это на немецкий» — без примеров |
| **One-shot** | Описание + 1 пример | «Вот пример перевода: ... А теперь переведи это» |
| **Few-shot** | Описание + K примеров | «Вот 10 примеров. А теперь ты» |

Пример zero-shot промпта для GPT-3:
```
Translate English to French:
cheese =>
```

GPT-3 175B в zero-shot режиме значительно превзошёл GPT-2, но уступал few-shot на большинстве задач. На TriviaQA: zero-shot 64.3% vs few-shot 71.2%. Однако на некоторых задачах (SuperGLUE) zero-shot GPT-3 был **конкурентоспособен с fine-tuned BERT**.

### Критическое открытие: scale dependency

GPT-3 показал, что zero-shot performance **резко улучшается с ростом модели**:

- Маленькие модели (~1B) почти не способны к zero-shot — им нужны примеры
- Средние (~10B) начинают следовать инструкциям
- Большие (~100B+) демонстрируют **emergent abilities** — zero-shot на задачах, которые мелкие модели не решают вообще

Это стало одним из главных аргументов за scaling laws: некоторые способности **возникают** только при достаточном размере.

## Zero-shot CoT: «Let's think step by step»

Kojima et al. (2022) обнаружили, что добавление фразы **"Let's think step by step"** к zero-shot промпту резко улучшает reasoning:

```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. 
Each can has 3 tennis balls. How many tennis balls does he have now?
A: Let's think step by step.
```

Без этой фразы InstructGPT давал неправильный ответ. С ней — правильный, с полной цепочкой рассуждений. На GSM8K zero-shot CoT приблизился к few-shot standard prompting.

Интуитивно: фраза работает как **trigger**, переключающий модель из режима «выдай ответ» в режим «рассуждай пошагово». Модель видела подобные паттерны в training data (учебники, пояснения, решения задач).

## Instruction Tuning: revolution в zero-shot

### Проблема raw LLM

GPT-3 в zero-shot режиме часто «не понимал», что от него хотят: вместо выполнения задачи продолжал текст в произвольном направлении. Raw language model оптимизирована на **prediction**, а не на **instruction following**.

### Решение: обучение на инструкциях

Из [[02 Areas/ML & DL/Papers/Flan-T5-PaLM]] (Chung et al., 2022): **instruction tuning** на 1.8K+ задачах в формате «инструкция → ответ» радикально улучшает zero-shot:

- **Flan-T5-XL (3B)** zero-shot > **GPT-3 (175B)** zero-shot на многих задачах
- Модель в 60x меньше, но обученная следовать инструкциям, обходит гиганта
- Ключевой insight: разнообразие задач при instruction tuning → обобщение на **новые** инструкции

Это объясняет, почему ChatGPT, Claude и LLaMA-Chat так хорошо работают в zero-shot: они прошли instruction tuning + RLHF, оптимизирующие именно **понимание и выполнение инструкций**.

### InstructGPT → ChatGPT pipeline

```
Pre-training (GPT-3) → Supervised Fine-Tuning (SFT) на инструкциях → RLHF
     ↓                        ↓                                          ↓
   Completion model       Instruction follower                   Aligned assistant
   (плохой zero-shot)     (хороший zero-shot)               (отличный zero-shot)
```

## Zero-shot в Computer Vision

Отдельная ветка исследований — zero-shot для задач классификации, где модель должна распознавать **классы, не встречавшиеся при обучении**:

- **Attribute-based**: описание нового класса через атрибуты («полосатое, четвероногое, хищник» → тигр)
- **CLIP** (Radford et al., 2021): zero-shot image classification через выравнивание изображений и текстовых описаний классов. CLIP zero-shot конкурировал с ResNet fine-tuned на ImageNet
- **Semantic embeddings**: перенос знаний через общее embedding space (Word2Vec, BERT)

## Почему zero-shot важен

1. **Минимальный effort**: не нужны размеченные данные, не нужны примеры — только инструкция
2. **Истинное обобщение**: zero-shot performance = честный тест на то, насколько модель «понимает» задачу, а не запоминает паттерны из примеров
3. **Productionability**: один чекпоинт модели для тысяч задач — не нужно fine-tuning'ить модель под каждый usecase
4. **Emergent abilities**: zero-shot раскрывает способности, «скрытые» в весах модели после pre-training

## Ограничения

- **Чувствительность к формулировке**: перефразирование инструкции может драматически изменить результат
- **Нестабильность**: одна и та же задача с разными формулировками даёт разброс accuracy до 30%
- **Bias из pre-training**: модель «помнит» преобладающие паттерны, что может приводить к систематическим ошибкам
- **Не для всего работает**: задачи с нестандартной структурой (таблицы, графы) плохо решаются zero-shot

## Key papers

- [[02 Areas/ML & DL/Papers/GPT 2.0]] — первая систематическая оценка zero-shot LLM
- [[02 Areas/ML & DL/Papers/GPT 3.0]] — zero/one/few-shot систематическое сравнение, scale dependency
- [[02 Areas/ML & DL/Papers/Flan-T5-PaLM]] — instruction tuning как способ драматически улучшить zero-shot

## Related concepts

- [[02 Areas/ML & DL/Concepts/Inference/In-context Learning|In-context Learning]] — few-shot: обучение через примеры в контексте
- [[02 Areas/ML & DL/Concepts/Inference/Prompting|Prompting]] — форматирование входа для управления LLM
- [[02 Areas/ML & DL/Concepts/Inference/Chain of Thought|Chain of Thought]] — пошаговое рассуждение; zero-shot CoT = «Let's think step by step»
- [[02 Areas/ML & DL/Concepts/Training/Instruction Tuning|Instruction Tuning]] — обучение на инструкциях для улучшения zero-shot
- [[02 Areas/ML & DL/Concepts/Inference/Prompt Engineering|Prompt Engineering]] — оптимизация промптов, включая zero-shot

## Дополнительные ресурсы

- [Radford et al. — Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — пионерская работа по zero-shot из LM
- [Brown et al. — Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) — систематическое исследование zero/few-shot
- [Kojima et al. — Large Language Models are Zero-Shot Reasoners (2022)](https://arxiv.org/abs/2205.11916) — «Let's think step by step»
- [Wei et al. — Emergent Abilities of Large Language Models (2022)](https://arxiv.org/abs/2206.07682) — emergent zero-shot abilities при scaling
