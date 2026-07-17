---
title: "Instruction Tuning"
aliases: [instruction fine-tuning, SFT, Supervised Fine-Tuning, инструкционная настройка]
type: concept
category: Training
papers:
  - "[[02 Areas/ML & DL/Papers/Flan-T5-PaLM]]"
  - "[[02 Areas/ML & DL/Papers/InstructGPT]]"
  - "[[02 Areas/ML & DL/Papers/LLaMA 2]]"
courses:
  - "[[02 Areas/ML & DL/Courses/SHAD LLM/Week 1 -- Intro to LLMs|SHAD LLM -- Week 1]]"
sources:
  - "[Wei et al. -- Finetuned Language Models Are Zero-Shot Learners (FLAN, 2022)](https://arxiv.org/abs/2109.01652)"
  - "[Ouyang et al. -- Training language models to follow instructions with human feedback (InstructGPT, 2022)](https://arxiv.org/abs/2203.02155)"
  - "[Chung et al. -- Scaling Instruction-Finetuned Language Models (Flan-PaLM, 2022)](https://arxiv.org/abs/2210.11416)"
---

# Instruction Tuning

## Зачем это нужно: превращение LM в assistant

Pre-trained LLM -- это **генератор продолжений текста**. Если дать ей вопрос "What is the capital of France?", она может продолжить его как "What is the capital of France? What is the capital of Germany?" вместо того, чтобы ответить "Paris". Модель обучена предсказывать **наиболее вероятное продолжение**, а не **следовать инструкциям**.

**Instruction Tuning** (оно же **Supervised Fine-Tuning, SFT**) решает эту проблему: fine-tune модель на парах (инструкция, желаемый ответ), чтобы она научилась **выполнять задания** на естественном языке. Это превращает LM из "дополнителя текста" в "помощника".

```
До Instruction Tuning:
  User: "Summarize this article: [text]"
  Model: "Summarize this article: [другой текст, продолжение]"

После Instruction Tuning:
  User: "Summarize this article: [text]"
  Model: "The article discusses three main points: ..."
```

## Формат данных

### Базовый формат: instruction + input + output

```
Instruction: "Translate the following to French."
Input: "The quick brown fox"
Output: "Le renard brun rapide"
```

Или без input (zero-input tasks):

```
Instruction: "Write a haiku about machine learning."
Output: "Data flows like streams
   Patterns emerge from chaos
   Machines learn to see"
```

### Chat формат (modern SFT)

Современные модели используют **chat template** с ролями:

```
<|system|>You are a helpful assistant.</s>
<|user|>What is the capital of France?</s>
<|assistant|>The capital of France is Paris.</s>
```

Loss вычисляется **только по assistant-части** (не по user/system). Это критически важно: модель не должна учиться генерировать вопросы пользователя.

### Multi-turn диалоги

```
<|user|>What is Python?</s>
<|assistant|>Python is a programming language...</s>
<|user|>Show me a hello world example.</s>
<|assistant|>Here's a simple example:
print("Hello, World!")</s>
```

## Два подхода к Instruction Tuning

### Подход 1: Task scaling (FLAN/Flan-T5/Flan-PaLM)

Идея Wei et al. (2022): чем **больше задач** в instruction tuning, тем лучше модель обобщается на **невиданные задачи** (zero-shot generalization).

```
Instruction Tuning Dataset:
  Task 1: Sentiment Analysis (SST-2)
  Task 2: NLI (MNLI)
  Task 3: Translation (WMT)
  ...
  Task 1836: Code Generation
  
  + 9 Chain-of-Thought датасетов
  
  --> Модель обобщается на Task 1837 (невиданную задачу)
```

Из Flan-PaLM (Chung et al., 2022) -- три ключевых scaling фактора:

| Фактор | Наблюдение | Статус |
|--------|-----------|--------|
| Число задач | 1836 задач > 282; saturation не достигнут | Чем больше, тем лучше |
| Размер модели | IT работает тем лучше, чем больше модель | Масштаб = ключ |
| CoT данные | 9 CoT датасетов критичны для reasoning | Без CoT -- деградация |

**Flan-PaLM 540B** vs PaLM 540B:
- Average: **+9.4%** на широком наборе бенчмарков
- MMLU: 75.2% (5-shot)
- TyDiQA: **+14.9%** (1-shot multilingual)

### Подход 2: Quality over quantity (InstructGPT/LLaMA 2)

Ouyang et al. (2022) и Touvron et al. (2023) показали противоположное: **качество > количество**.

Из InstructGPT (раздел 3.3):
- ~13K human-written demonstrations (промпты + желаемые ответы)
- Labelers -- 40 людей, тщательно отобранных и обученных
- Fine-tune GPT-3, 16 эпох, cosine LR decay, dropout 0.2
- **Thousands** качественных примеров достаточно; **millions** низкокачественных -- нет

Из LLaMA 2 (раздел 3.1):
- "Tens of thousands" high-quality annotations
- Third-party SFT data (низкого качества) **отвергнута** -- ухудшала результаты
- Вывод: *"We found that SFT annotations in the order of tens of thousands was sufficient to achieve a high quality result"*

### Сравнение подходов

| | FLAN (task scaling) | InstructGPT (quality) |
|-|--------------------|-----------------------|
| Данные | 1.8K задач, millions примеров | ~13K high-quality demonstrations |
| Фокус | Zero-shot generalization на новые задачи | Following user intent, alignment |
| Модель | Encoder-decoder (T5) или decoder | Decoder-only (GPT-3) |
| Следующий шаг | Модель готова к использованию | Step 1 в RLHF pipeline |

## SFT как Step 1 в RLHF pipeline

В современном alignment pipeline, Instruction Tuning (SFT) -- это **первый этап**:

```
Pre-trained LM
     |
     v
[SFT on demonstrations]     <-- Instruction Tuning
     |
     v
[Reward Model training]     <-- Human preferences
     |
     v
[PPO / DPO]                 <-- RL optimization
     |
     v
Aligned LM (ChatGPT, Claude, etc.)
```

**Зачем SFT перед RL?** SFT "сужает" distribution модели до формата "assistant": модель уже знает, что нужно отвечать на вопросы, а не продолжать текст. Без SFT, RL должен одновременно учить формат И alignment -- это значительно сложнее и нестабильнее.

Из Flan-PaLM: SFT without RLHF уже даёт **большое улучшение** над base LM. RLHF добавляет safety, helpfulness и тонкий alignment.

## Ключевые результаты

### InstructGPT 1.3B vs GPT-3 175B

Шокирующий результат из InstructGPT:

| Модель | Params | Human preference |
|--------|--------|-----------------|
| GPT-3 | 175B | baseline |
| **InstructGPT** | **1.3B** | **preferred by humans** |

**1.3B модель предпочитается** людьми по сравнению со **175B** -- при разнице в **100x** в числе параметров. Это показывает, что alignment (SFT + RLHF) важнее чистого масштаба.

Дополнительно:
- Hallucination: 21% (InstructGPT) vs 41% (GPT-3)
- Toxicity: значительно меньше

### Flan-T5 vs T5

| Модель | MMLU | BBH | Avg NLU |
|--------|------|-----|---------|
| T5-XL (3B) | baseline | baseline | baseline |
| Flan-T5-XL (3B) | **+15.6%** | **+18.9%** | **+12.3%** |

Instruction tuning буквально **бесплатно** улучшает модель на 10-20% по всем бенчмаркам.

## Качество данных: что работает, а что нет

### Работает

1. **Human-written demonstrations** от квалифицированных labelers (InstructGPT, LLaMA 2)
2. **Diverse task mix** с инструкциями на естественном языке (FLAN)
3. **Chain-of-Thought** примеры для reasoning задач (Flan-PaLM)
4. **Multi-turn диалоги** для conversational ability (LLaMA 2-Chat)

### Не работает / вредит

1. **Low-quality crowd-sourced data** (LLaMA 2 отвергла third-party SFT data)
2. **Только one-turn** инструкции для chat-модели (нужны multi-turn)
3. **Дисбаланс задач** -- если 90% данных -- перевод, модель "забудет" остальное
4. **Слишком длинное обучение** -- overfitting на SFT данные, потеря general knowledge

## Open-source Instruction Tuning датасеты

| Датасет | Размер | Описание |
|---------|--------|----------|
| FLAN Collection | 1.8K tasks | Академические NLP задачи с инструкциями |
| Alpaca | 52K | GPT-3.5-generated, простые инструкции |
| Dolly | 15K | Human-written (Databricks employees) |
| OpenAssistant | 161K | Crowd-sourced, multi-turn |
| ShareGPT | ~90K | Real ChatGPT conversations |
| UltraChat | 1.5M | Multi-turn, model-generated |

## Хронология

| Год | Milestone | Работа |
|-----|-----------|--------|
| 2021 | FLAN: первый масштабный instruction tuning | Wei et al. |
| 2022 | InstructGPT: SFT + RLHF = ChatGPT | Ouyang et al. |
| 2022 | Flan-PaLM: scaling to 1.8K tasks | Chung et al. |
| 2023 | Alpaca: дешёвый IT через distillation | Stanford |
| 2023 | LLaMA 2-Chat: quality > quantity SFT | Meta |
| 2023 | Mistral-Instruct: efficient IT | Mistral |

## Key papers

- [[02 Areas/ML & DL/Papers/Flan-T5-PaLM]] -- scaling instruction finetuning (1.8K tasks, +9.4%)
- [[02 Areas/ML & DL/Papers/InstructGPT]] -- RLHF с SFT как Step 1; alignment at scale
- [[02 Areas/ML & DL/Papers/LLaMA 2]] -- quality > quantity SFT philosophy

## Related concepts

- [[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]] -- следующий шаг после SFT в alignment pipeline
- [[02 Areas/ML & DL/Concepts/Training/Fine-tuning|Fine-tuning]] -- general fine-tuning, от которого IT отличается структурой данных
- [[02 Areas/ML & DL/Concepts/Training/PEFT|PEFT]] -- parameter-efficient версия instruction tuning
- [[02 Areas/ML & DL/Concepts/Inference/Chain of Thought|Chain-of-Thought]] -- reasoning data критичны для IT
- [[02 Areas/ML & DL/Concepts/Inference/Prompting|Prompting]] -- IT улучшает prompt-following ability
- [[02 Areas/ML & DL/Concepts/Architectures/Flan-T5|Flan-T5]] -- открытая instruction-tuned модель

## Дополнительные ресурсы

- [Longpre et al. -- The FLAN Collection (2023)](https://arxiv.org/abs/2301.13688) -- обзор instruction tuning датасетов
- [Sebastian Raschka -- Instruction Tuning LLMs](https://magazine.sebastianraschka.com/p/instruction-tuning-llms) -- практический обзор
- [HuggingFace -- Fine-tuning with SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) -- tutorial с кодом
