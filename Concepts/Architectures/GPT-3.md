---
title: "GPT-3"
aliases: [GPT 3, GPT3, Language Models are Few-Shot Learners]
type: concept
status: legacy
category: Architectures
papers:
  - "[[02 Areas/ML & DL/Papers/GPT 1.0|GPT 1.0]]"
  - "[[02 Areas/ML & DL/Papers/GPT 2.0]]"
  - "[[02 Areas/ML & DL/Papers/GPT 3.0]]"
  - "[[02 Areas/ML & DL/Papers/GPT 4.0]]"
courses: []
sources:
  - "[Jay Alammar — The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)"
  - "[Jay Alammar — How GPT-3 Works](https://jalammar.github.io/how-gpt3-works-visualizations-animations/)"
  - "[Lilian Weng — The Transformer Family v2](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)"
---

# GPT-3 — Language Models are Few-Shot Learners

## Зачем это нужно: от fine-tuning к in-context learning

До GPT-3 (2020) парадигма NLP работала по схеме: **pre-train → fine-tune**. Каждая новая задача требовала:
1. Собрать labeled dataset (тысячи-десятки тысяч примеров)
2. Провести fine-tuning с gradient updates
3. Подобрать гиперпараметры

Это создавало три проблемы:
- **Практическая**: для многих задач сложно собрать большой labeled dataset
- **Spurious correlations**: fine-tuned модель переобучается на артефакты узкого датасета, теряя обобщение
- **Несоответствие человеку**: людям достаточно инструкции и 1-2 примеров, чтобы выполнить новую задачу

**GPT-3 (Brown et al., 2020)** показал, что при достаточном масштабе (175B параметров) модель способна выполнять задачи **без gradient updates** — используя только текстовые примеры в промпте. Это назвали **in-context learning**.

## Эволюция GPT: от 117M к 175B

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-30/gpt-2-transformer-xl-bert-3.png]]
*Линейка моделей 2018-2019: GPT использует Transformer decoder, BERT — encoder. Масштаб растёт (источник: Jay Alammar)*

### GPT-1 (Radford et al., 2018) — proof of concept

- **117M параметров**, 12 слоёв, $d_{\text{model}}$ = 768
- Decoder-only Transformer с causal attention
- Pre-training: left-to-right LM на BooksCorpus (7000 книг)
- Fine-tuning на каждую downstream задачу
- Показал: unsupervised pre-training + supervised fine-tuning = SOTA на 9 из 12 NLU задач

### GPT-2 (Radford et al., 2019) — zero-shot generalization

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-30/gpt2-sizes-hyperparameters-3.png]]
*Четыре размера GPT-2: от 117M до 1.5B параметров (источник: Jay Alammar)*

- **1.5B параметров**, 48 слоёв, $d_{\text{model}}$ = 1600
- Обучен на WebText (40GB, 8M веб-страниц с Reddit karma > 3)
- Архитектурные улучшения: **pre-normalization** (LayerNorm перед attention/FFN, а не после), modified initialization ($1/\sqrt{N}$ для residual layers)
- Ключевой тезис: *"Language models are unsupervised multitask learners"* — модель выполняет задачи zero-shot, без fine-tuning
- Результат: zero-shot state-of-the-art на 7 из 8 language modeling бенчмарков

### GPT-3 (Brown et al., 2020) — emergence of in-context learning

- **175B параметров**, 96 слоёв, $d_{\text{model}}$ = 12288 — **в 116x больше GPT-2**
- Тот же архитектурный дизайн + sparse attention
- Ключевое открытие: **in-context learning scales with model size**

### GPT-4 (OpenAI, 2023) — multimodal + RLHF

- Параметры не раскрыты (предположительно MoE, ~1.8T)
- Мультимодальный (текст + изображения)
- Обучен с RLHF (Reinforcement Learning from Human Feedback)
- Стабильно проходит bar exam, LSAT, GRE

## Архитектура GPT-3

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-30/gpt2-transformer-block-vectors-2.png]]
*Decoder block GPT-2/3: Layer Norm → Masked Self-Attention → Residual → Layer Norm → FFN → Residual (источник: Jay Alammar)*

GPT-3 наследует архитектуру GPT-2 с минимальными изменениями:

**Decoder block:**
1. **Pre-Layer Norm** (нормализация входа, а не выхода — стабильнее при большом масштабе)
2. **Masked Multi-Head Attention** (causal mask: каждый токен видит только предшествующие)
3. **Residual connection**
4. **Pre-Layer Norm**
5. **Feed-Forward Network** ($d_{\text{model}}$ → $4 \times d_{\text{model}}$ → $d_{\text{model}}$)
6. **Residual connection**

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-30/self-attention-and-masked-self-attention.png]]
*Masked self-attention (справа) vs обычный self-attention (слева): будущие позиции заблокированы (источник: Jay Alammar)*

### Семейство моделей (Table 2.1 из статьи)

| Модель | Параметры | Layers | $d_{\text{model}}$ | Heads | Context |
|--------|-----------|--------|---------------------|-------|---------|
| GPT-3 Small | 125M | 12 | 768 | 12 | 2048 |
| GPT-3 Medium | 350M | 24 | 1024 | 16 | 2048 |
| GPT-3 Large | 760M | 24 | 1536 | 16 | 2048 |
| GPT-3 XL | 1.3B | 24 | 2048 | 24 | 2048 |
| GPT-3 6.7B | 6.7B | 32 | 4096 | 32 | 2048 |
| GPT-3 13B | 13B | 40 | 5140 | 40 | 2048 |
| **GPT-3 175B** | **175B** | **96** | **12288** | **96** | **2048** |

Все модели обучены на **~300B токенов**. Batch size: от 0.5M до **3.2M токенов** для 175B.

Архитектурная деталь: GPT-3 использует **alternating dense and locally banded sparse attention** (чередование полного и локального attention в слоях), аналогично Sparse Transformer (Child et al., 2019).

### Входные представления

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-30/gpt2-token-embeddings-wte-2.png]]
*Token embeddings + learned positional embeddings (источник: Jay Alammar)*

- **Token embedding**: BPE с словарём ~50K (наследие GPT-2)
- **Positional embedding**: обучаемые позиционные векторы (не синусоидальные)
- Context window: **2048 токенов**

### Авторегрессивная генерация

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-30/gpt2-autoregression-2.gif]]
*Авторегрессия: каждый новый токен генерируется с учётом всех предыдущих (источник: Jay Alammar)*

На каждом шаге модель предсказывает распределение вероятностей следующего токена по всему словарю. Выбранный токен добавляется ко входу, процесс повторяется. Loss: стандартный cross-entropy языковой модели.

## Training Data

Из Section 2.2 — **взвешенная смесь** корпусов:

| Источник | Вес при обучении | Tokens | Эпохи |
|----------|-----------------|--------|-------|
| Common Crawl (filtered) | 60% | ~400B | 0.44 |
| WebText2 | 22% | ~19B | 2.9 |
| Books1 | 8% | ~12B | 1.9 |
| Books2 | 8% | ~55B | 0.43 |
| Wikipedia | 3% | ~3B | 3.4 |

**Ключевой приём**: высококачественные корпуса (Books, Wikipedia) семплируются **2-3x чаще** пропорционально размеру, CommonCrawl **меньше 1x**. Это повышает качество при том же compute budget.

Common Crawl фильтровался с помощью классификатора, обученного отличать WebText (качественный) от случайного CC (шумный).

## In-Context Learning: Zero/One/Few-Shot

Главная инновация GPT-3 — определение трёх evaluation режимов **без gradient updates**:

| Режим | Что в промпте | Аналогия |
|-------|--------------|----------|
| **Zero-shot** | Только описание задачи на естественном языке | Инструкция на Mechanical Turk без образца |
| **One-shot** | Описание + 1 пример | Один пример человеку |
| **Few-shot** | Описание + K=10-100 примеров (сколько влезет в 2048 контекст) | Несколько примеров |

**Ключевое свойство**: larger models **лучше** используют in-context information. Кривые few-shot learning **круче** у больших моделей — это emergent ability, не наблюдаемая у маленьких.

Из Figure 1.3 статьи: zero-shot performance растёт плавно с размером модели, но few-shot performance растёт **быстрее** — демонстрируя, что большие модели более proficient at in-context learning.

**Мета-learning интерпретация** (Figure 1.1): при unsupervised pre-training модель развивает pattern recognition abilities. В in-context learning она использует эти abilities для быстрой адаптации — это «inner loop» мета-обучения, происходящий **внутри одного forward pass**.

## Ключевые результаты

### Closed-Book Question Answering

| Бенчмарк | Zero-shot | One-shot | Few-shot | Fine-tuned SOTA |
|----------|-----------|----------|----------|-----------------|
| TriviaQA | 64.3% | 68.0% | **71.2%** | 68.0% |
| NaturalQuestions | 14.6% | 23.0% | **29.9%** | 36.5% |
| WebQuestions | 14.4% | 25.3% | **41.5%** | 44.4% |

Few-shot GPT-3 **превосходит** fine-tuned SOTA на TriviaQA — без единого gradient update.

### Reading Comprehension

| Бенчмарк | Zero-shot | Few-shot |
|----------|-----------|----------|
| CoQA F1 | 81.5 | **85.0** |
| DROP F1 | 23.6 | **36.5** |

### SuperGLUE

Few-shot GPT-3 приближается к fine-tuned BERT_LARGE на многих задачах, но значительно **проигрывает** на NLI задачах (WinoGrad, RTE) — где bidirectional контекст критичен.

### Арифметика (few-shot, Table 3.10)

| Задача | Точность |
|--------|---------|
| 2-digit addition | 100% |
| 3-digit addition | 80.2% |
| 4-digit addition | 25.5% |
| 2-digit subtraction | 98.9% |
| 3-digit subtraction | 75.6% |

Модель способна к базовой арифметике **без fine-tuning** — это emergent ability, не наблюдаемая у моделей меньше ~13B.

### Генерация новостей

Человеческие эксперты различают GPT-3-сгенерированные новости и человеческие **лишь в 52% случаев** (близко к случайному угадыванию).

## Limitations (из Section 5 статьи)

1. **Text generation quality**: GPT-3 иногда теряет когерентность на длинных текстах, повторяется, генерирует бессмысленные абзацы.
2. **Architectural**: decoder-only архитектура принципиально хуже для задач, требующих bidirectional контекста (NLI). Авторы отмечают, что bidirectional модель того же масштаба могла бы быть лучше.
3. **Efficiency**: модель обучена на ~300B токенов, но Chinchilla (2022) позже показал, что оптимально было бы ~3.7T токенов для 175B модели — GPT-3 **сильно недообучен**.
4. **Interpretability**: непонятно, действительно ли in-context learning — это «обучение» или «распознавание паттернов из pre-training данных».

## Почему GPT-3 изменил всё

1. **Парадигмальный сдвиг**: от «pre-train → fine-tune per task» к «pre-train → prompt» (или «pre-train → align»). Запустил взрыв prompt engineering research.

2. **Scaling hypothesis подтверждена**: показал, что emergent abilities появляются при достаточном масштабе. Это мотивировало гонку к 540B (PaLM), 1.8T (GPT-4), и далее.

3. **Decoder-only доминирование**: установил decoder-only как **доминирующую парадигму** для LLM. До GPT-3 encoder-only (BERT) и encoder-decoder (T5) были конкурентоспособны. После — decoder-only стал default.

4. **API-first подход**: GPT-3 стал доступен только через API, запустив тренд закрытых моделей и «LLM-as-a-service» бизнес-модели.

5. **Inference-time compute**: идея, что model capabilities определяются не только training, но и тем, сколько compute выделено на inference (длинные промпты, chain-of-thought).

## Хронология GPT-линейки

| Год | Модель | Параметры | Ключевая идея |
|-----|--------|-----------|---------------|
| 2018 | GPT-1 | 117M | Unsupervised pre-training + supervised fine-tuning |
| 2019 | GPT-2 | 1.5B | Zero-shot generalization, *"too dangerous to release"* |
| 2020 | GPT-3 | 175B | In-context learning, few-shot prompting |
| 2022 | InstructGPT | 175B | RLHF alignment |
| 2022 | ChatGPT | ~175B | Conversational RLHF tuning → массовое adoption |
| 2023 | GPT-4 | ~1.8T (MoE?) | Multimodal, RLHF, bar exam passing |
| 2024 | GPT-4o | ? | Natively multimodal (audio+vision+text) |

## Key papers

- [[02 Areas/ML & DL/Papers/GPT 3.0]] — оригинал (Brown et al., 2020)
- [[02 Areas/ML & DL/Papers/GPT 2.0]] — predecessor (1.5B, zero-shot generalization)
- [[02 Areas/ML & DL/Papers/GPT 4.0]] — successor (multimodal, RLHF)

## Related concepts

- [[02 Areas/ML & DL/Concepts/Architectures/Decoder-only|Decoder-only]] — архитектурная парадигма
- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] — базовая архитектура
- [[02 Areas/ML & DL/Concepts/Inference/In-context Learning|In-context Learning]] — ключевая способность
- [[02 Areas/ML & DL/Concepts/Inference/Prompting|Prompting]] — способ взаимодействия
- [[02 Areas/ML & DL/Concepts/Inference/Zero-shot Learning|Zero-shot Learning]]
- [[02 Areas/ML & DL/Concepts/NLP/Emergent Abilities|Emergent Abilities]]
- [[02 Areas/ML & DL/Concepts/Training/Scaling Laws|Scaling Laws]]
- [[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]] — alignment technique

## Дополнительные ресурсы

- [Jay Alammar — The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) — визуализация архитектуры GPT-2/3
- [Jay Alammar — How GPT-3 Works](https://jalammar.github.io/how-gpt3-works-visualizations-animations/) — анимации in-context learning
- [Lilian Weng — The Transformer Family v2](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) — контекст decoder-only моделей
- [GPT-3 Paper](https://arxiv.org/abs/2005.14165) — 75 страниц, стоит прочитать целиком
