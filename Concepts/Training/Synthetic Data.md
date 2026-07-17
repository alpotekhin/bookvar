---
title: "Synthetic Data"
aliases: [Synthetic Data, Synthetic Training Data, Textbook Quality Data]
type: concept
category: Training
papers:
  - "[[02 Areas/ML & DL/Papers/Phi-2|Phi-2]]"
  - "[[02 Areas/ML & DL/Papers/Phi-4 Technical Report|Phi-4]]"
  - "[[02 Areas/ML & DL/Papers/Nemotron-4 340B Technical Report|Nemotron-4]]"
courses: []
sources:
  - "[NVIDIA Blog — Nemotron-4 Synthetic Data Generation Pipeline](https://blogs.nvidia.com/blog/nemotron-4-synthetic-data-generation-llm-training/)"
  - "[Latent Space — 2024 in Synthetic Data and Smol Models](https://www.latent.space/p/2024-syndata-smolmodels)"
  - "[Aussie AI — Synthetic Data for LLM Training](https://www.aussieai.com/research/synthetic-data)"
  - "[GitHub — LLM-Synthetic-Data Reading List](https://github.com/pengr/LLM-Synthetic-Data)"
---

# Synthetic Data для обучения LLM

## Зачем это нужно

К 2024 году LLM-индустрия столкнулась с **data wall**: высококачественные текстовые данные из интернета конечны. По оценкам Epoch AI, к 2026-2028 будет исчерпан весь «полезный» web-text для обучения frontier моделей.

Одновременно стало ясно, что **качество данных важнее количества**:
- Phi-1 (1.3B) обучена на 6B «textbook-quality» токенов → превзошла модели в 10x больше по параметрам на coding
- Phi-3 (3.8B) конкурировала с LLaMA-3-8B при вдвое меньшем размере

**Synthetic data** — генерация обучающих данных с помощью LLM — стала ответом на оба вызова.

## Phi Series: «textbook quality» тезис

### Phi-1: доказательство концепции (2023)

Microsoft Research задали вопрос: **что если обучать модель только на данных уровня учебника?**

**Рецепт Phi-1:**
1. Взять сильную модель (GPT-3.5/4)
2. Генерировать «учебниковые» объяснения и упражнения по программированию
3. Обучить маленькую модель (1.3B) на этих данных
4. Результат: **51% на HumanEval** — лучше, чем GPT-3.5 (48.1%) при 100x меньшем размере

**Почему это шокировало:** считалось, что для сильного coding нужны миллиарды параметров. Phi-1 показала, что **правильные данные заменяют размер**.

### Phi-1.5 → Phi-2 → Phi-3 → Phi-4: масштабирование подхода

| Модель | Params | Данные | Ключевой результат |
|--------|--------|--------|--------------------|
| Phi-1 | 1.3B | 6B synthetic coding | 51% HumanEval |
| Phi-1.5 | 1.3B | 30B synthetic reasoning | Превзошла LLaMA-7B |
| Phi-2 | 2.7B | 250B mixed | Сопоставима с LLaMA-2-13B |
| Phi-3 mini | 3.8B | 3.3T heavily filtered | Конкурирует с LLaMA-3-8B |
| Phi-4 | 14B | 9.8T + synthetic | Превзошла GPT-4o mini |

### Эволюция метода

**Phi-1 → Phi-2:** чисто synthetic данные, сгенерированные GPT-4.

**Phi-3 → Phi-4:** сдвиг к **hybrid** — сочетание web данных + synthetic. Phi-4 использует synthetic data не для замены web data, а для **дополнения**: генерация math задач, reformulation текстов в учебный стиль, синтез multi-step reasoning примеров.

**Phi-4 инновации:**
- Synthetic data для standardization (LaTeX-формулы, форматирование)
- Question-answer pairs по сложным топикам
- Multi-step reasoning trajectories

## Nemotron Pipeline: индустриальный масштаб

### Задача NVIDIA

NVIDIA создала **полный pipeline** генерации synthetic данных для pre-training и post-training LLM.

### Nemotron-4 340B

Семейство из трёх моделей, формирующих замкнутый цикл:

1. **Nemotron-4-340B-Base** — pre-trained модель
2. **Nemotron-4-340B-Instruct** — instruction-tuned модель, генерирует synthetic данные
3. **Nemotron-4-340B-Reward** — reward model, оценивает quality

**Pipeline:**
```
Seed prompts → Instruct model → Synthetic responses → Reward model → Quality filtering → Training data
```

**Масштаб:** 98% данных для SFT/alignment Nemotron были **синтетическими**, сгенерированными из всего ~20K human-written seeds.

### Nemotron-CC: synthetic для pre-training

Nemotron-CC — переработка Common Crawl через synthetic augmentation:
- Reformulation низкокачественных web-текстов в высококачественные
- Синтез Q&A pairs из сырых документов
- Генерация math/code контента из естественноязыковых описаний

**Результат:** модели, обученные на Nemotron-CC, показывают +5.6 MMLU по сравнению с DCLM (чистый web crawl) при том же объёме данных.

## Таксономия подходов к synthetic data

### По стадии обучения

| Стадия | Примеры | Масштаб |
|--------|---------|---------|
| **Pre-training** | Nemotron-CC, Phi-4 synthetic | Триллионы токенов |
| **SFT** | Alpaca (GPT-4 generated), Nemotron SFT | Тысячи-миллионы примеров |
| **RLHF/RL** | Constitutional AI, RLAIF | Тысячи comparisons |
| **Evaluation** | Synthetic benchmarks | Тысячи задач |

### По методу генерации

**Rephrasing/Reformulation:** перезапись web-текста в «учебный» стиль. Пример: взять Wikipedia статью и переписать как главу учебника с примерами.

**Question Generation:** генерация вопросов и ответов по исходному документу. Используется для создания instruction-following данных.

**Seed-based Expansion:** начать с малого набора высококачественных примеров, расширить через variation и generalization.

**Self-Instruct / Self-Play:** модель генерирует инструкции для себя, затем учится на своих лучших ответах (используется в Alpaca, WizardLM).

## Риски и ограничения

### Model Collapse

При обучении на synthetic данных от LLM, следующие поколения моделей могут терять разнообразие — **model collapse**. Каждое поколение «усредняет» распределение, теряя tail knowledge.

**Митигация:** всегда смешивать synthetic с real data. Empirical result: pure synthetic < mixed synthetic+real.

### Hallucination Amplification

LLM-генератор может производить фактически неверные данные. Если обучить на них — модель «закрепит» галлюцинации.

**Митигация:** verification pipeline (reward model, rule-based checks, human spot-checks).

### Bias Reproduction

Synthetic data наследует biases генерирующей модели. Это может усиливать существующие предубеждения.

### Legal Concerns

Использование output одной модели для обучения другой — юридически спорная территория (Terms of Service OpenAI запрещают обучение конкурентов на output GPT-4).

## Формула успеха (эмпирическая)

Из обзора 2024-2025 года вырисовывается паттерн:

$$\text{Quality} \propto \text{Data Quality} \times \log(\text{Data Quantity}) \times \text{Model Size}^{0.5}$$

**Качество данных имеет суперлинейный эффект:** удвоение quality данных даёт больший эффект, чем удвоение quantity. Synthetic data — способ **конвертировать compute в quality**.

## Кто это использует (2024-2025)

| Компания | Подход | Результат |
|----------|--------|-----------|
| Microsoft (Phi) | Textbook-quality synthetic | Маленькие модели с большой силой |
| NVIDIA (Nemotron) | Full pipeline с reward filtering | 98% synthetic alignment data |
| Meta (LLaMA 3) | Synthetic для post-training rounds | Iterative self-improvement |
| DeepSeek (R1) | Synthetic reasoning traces для distillation | 800K CoT samples |
| Alibaba (Qwen) | Synthetic для math/code pre-training | Enhanced reasoning |

## Related concepts

- [[02 Areas/ML & DL/Concepts/Training/Fine-tuning|Fine-tuning]] — основной потребитель synthetic data
- [[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]] — synthetic preferences
- [[02 Areas/ML & DL/Concepts/Training/Instruction Tuning|Instruction Tuning]] — self-instruct method
- [[02 Areas/ML & DL/Concepts/Architectures/Falcon|Falcon]] — контраст: RefinedWeb (real data focused)
- [[02 Areas/ML & DL/Concepts/Training/Scaling Laws|Scaling Laws]] — data quality vs quantity trade-off
- [[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]] — synthetic data для pre-training
