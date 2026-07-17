---
title: "Prompt-based Learning"
aliases: [prompt-based learning, pre-train prompt predict, промпт-парадигма]
type: concept
status: legacy
category: NLP
papers:
  - "[[02 Areas/ML & DL/Papers/Pre-train, Prompt, and Predict- A Systematic Survey of Prompting Methods in Natural Language Processing]]"
  - "[[02 Areas/ML & DL/Papers/GPT 3.0]]"
  - "[[02 Areas/ML & DL/Papers/BERT]]"
courses:
  - "[[02 Areas/ML & DL/Courses/SHAD LLM/Week 1 — Intro to LLMs|SHAD LLM — Week 1]]"
sources:
  - "[Liu et al. — Pre-train, Prompt, and Predict (2021)](https://arxiv.org/abs/2107.13586)"
  - "[Lester et al. — The Power of Scale for Parameter-Efficient Prompt Tuning (2021)](https://arxiv.org/abs/2104.08691)"
  - "[Li & Liang — Prefix-Tuning (2021)](https://arxiv.org/abs/2101.00190)"
---

# Prompt-based Learning

## Что это такое и почему это революция

Prompt-based Learning --- парадигма NLP, в которой вместо адаптации модели под задачу (fine-tuning) **задача переформулируется в формат, понятный языковой модели**. Модель «решает» задачу через text completion или заполнение пропуска.

Каноническая таксономия из survey Liu et al. (2021) [[02 Areas/ML & DL/Papers/Pre-train, Prompt, and Predict- A Systematic Survey of Prompting Methods in Natural Language Processing]]:

| Парадигма | Эпоха | Суть | Пример |
|-----------|-------|------|--------|
| Feature engineering | до 2013 | Вручную создаём признаки | BoW + SVM |
| Architecture engineering | 2013--2018 | Подбираем нейросеть | BiLSTM, TextCNN |
| **Pre-train, Fine-tune** | 2018--2020 | Модель адаптируется к задаче | BERT + classification head |
| **Pre-train, Prompt, Predict** | 2020+ | Задача адаптируется к модели | GPT-3 few-shot prompting |

Ключевой сдвиг: **не модель подстраивается под задачу, а задача формулируется на языке модели**.

## Как это работает: формальная нотация

Из [[02 Areas/ML & DL/Papers/Pre-train, Prompt, and Predict- A Systematic Survey of Prompting Methods in Natural Language Processing]]:

**Шаг 1. Prompt Addition** --- добавление шаблона к входу:

$$x' = f_{\text{prompt}}(x)$$

Пример: $x$ = "I love this movie." -> $x'$ = "I love this movie. Overall, it was a **[Z]** movie."

**Шаг 2. Answer Search** --- модель заполняет пропуск:

$$\hat{z} = \arg\max_{z \in \mathcal{Z}} P(f_{\text{fill}}(x', z) \mid \theta)$$

Модель оценивает, какое слово наиболее вероятно на месте [Z]: "good", "bad", "great"...

**Шаг 3. Answer Mapping (Verbalizer)** --- отображение слова в метку:

$$\hat{y} = g(\hat{z})$$

Например: "good" / "great" / "fantastic" -> **positive**, "bad" / "terrible" -> **negative**.

### Компоненты в деталях

| Компонент | Описание | Пример |
|-----------|----------|--------|
| $x$ | Входной текст | "I love this movie." |
| $f_{\text{prompt}}(x)$ | Template function | Добавляет "Overall, it was a [Z] movie." |
| $x'$ | Prompted input (без ответа) | "I love this movie. Overall, it was a [Z] movie." |
| $z$ | Answer candidate | "good", "bad", "fantastic" |
| $\hat{z}$ | Best answer | argmax $P(z \mid x', \theta)$ |
| $g$ | Verbalizer | "good" -> positive |
| $\hat{y}$ | Final label | positive |

## Два типа промптов

### Cloze prompt (заполнение пропуска)

Слот **в середине** или произвольном месте шаблона. Естественно для masked LM ([[02 Areas/ML & DL/Concepts/Architectures/BERT|BERT]]-style моделей):

```
Template: "[X] Overall, it was a [Z] movie."
Input:    "I love this movie. Overall, it was a [Z] movie."
Model:    P([Z] = "great") > P([Z] = "terrible")
Label:    positive
```

Хорошо работает для: classification, NLI, fact verification.

### Prefix prompt (генерация продолжения)

Слот **в конце** --- модель генерирует продолжение. Естественно для autoregressive LM (GPT-style):

```
Template: "Translate English to French: [X]. French:"
Input:    "Translate English to French: The sky is blue. French:"
Model:    generates "Le ciel est bleu."
```

Хорошо работает для: translation, summarization, open-ended generation.

## Verbalizers: мост между словами и метками

Verbalizer --- ключевой компонент, который определяет, какие слова из словаря модели соответствуют каким меткам задачи.

| Тип verbalizer | Описание | Пример |
|----------------|----------|--------|
| **Manual** | Эксперт выбирает слова | positive: "good", "great"; negative: "bad", "terrible" |
| **Search-based** | Автоматический поиск по корпусу | Поиск слов, коррелирующих с меткой |
| **Soft / Learned** | Обучаемый эмбеддинг вместо конкретного слова | WARP (Hambardzumyan et al., 2021) |
| **Multi-token** | Ответ из нескольких слов | "not good" -> negative |

Качество verbalizer критически влияет на результат. Плохой verbalizer может снизить accuracy на 10--20% даже при хорошем шаблоне.

## Стратегии обучения

Из [[02 Areas/ML & DL/Papers/Pre-train, Prompt, and Predict- A Systematic Survey of Prompting Methods in Natural Language Processing]] (Table 6):

| Стратегия | LM параметры | Prompt | Пример | Когда использовать |
|-----------|:------------:|:------:|--------|-------------------|
| **Tuning-free** | Frozen | Fixed | GPT-3 zero/few-shot | Нет данных, большая LM |
| **Fixed-LM Prompt Tuning** | Frozen | Tuned | Prefix Tuning, P-Tuning | Мало данных, дорого обучать LM |
| **Fixed-prompt LM Tuning** | Tuned | Fixed | PET, LM-BFF | Есть данные, нужна точность |
| **Prompt + LM Fine-tuning** | Tuned | Tuned | PTR, P-Tuning v2 | Максимальная точность |

### Soft Prompts: обучаемые виртуальные токены

Вместо текстовых промптов --- **обучаемые непрерывные векторы**, которые конкатенируются с входными эмбеддингами:

**Prefix Tuning (Li & Liang, 2021):**
- Добавляет обучаемые префиксы к K и V в каждом слое attention
- Обучает только **0.1%** параметров
- На генеративных задачах (table-to-text, summarization) --- сравнимо с full fine-tuning

**Prompt Tuning (Lester et al., 2021):**
- Обучаемые токены только во входном слое (ещё проще)
- С ростом модели (до 10B+) разрыв с full fine-tuning исчезает
- T5-XXL + prompt tuning $\approx$ T5-XXL + full fine-tuning на SuperGLUE

### In-Context Learning (ICL)

[[02 Areas/ML & DL/Papers/GPT 3.0|GPT-3]] показал, что достаточно добавить **примеры прямо в промпт** --- без любого обновления весов:

```
Sentiment: "I love this movie." -> Positive
Sentiment: "Terrible acting." -> Negative
Sentiment: "The plot was thrilling." -> [MODEL GENERATES]
```

Это **tuning-free prompt-based learning** с few-shot демонстрациями. GPT-3 (175B) с few-shot ICL достигает accuracy, сравнимой с fine-tuned BERT на многих задачах.

## Multi-Prompt стратегии

| Стратегия | Идея | Зачем |
|-----------|------|-------|
| **Prompt Ensembling** | Несколько шаблонов -> усреднение предсказаний | Устойчивость к выбору шаблона |
| **Prompt Augmentation** | Few-shot examples как часть промпта | In-context learning (GPT-3) |
| **Prompt Composition** | Сложная задача = цепочка под-промптов | PTR для relation extraction |
| **Prompt Decomposition** | Одна задача разбивается на шаги | Chain-of-Thought prompting |

## Ограничения и проблемы

1. **Sensitivity к формулировке:** перестановка слов в шаблоне может изменить accuracy на 20--30%
2. **Verbalizer bottleneck:** для нестандартных задач трудно найти естественное соответствие слов и меток
3. **Calibration issues:** LLM могут быть overconfident или biased к определённым ответам
4. **Order sensitivity в ICL:** порядок few-shot примеров влияет на результат (иногда сильно)
5. **Не заменяет fine-tuning полностью:** на задачах с большим количеством данных fine-tuned модели по-прежнему точнее

## Историческое значение

Prompt-based learning --- это третья парадигма NLP (после feature engineering и fine-tuning). Она привела к:

1. **Унификации задач:** одна модель (GPT-3/4) + разные промпты = десятки задач без изменения весов
2. **Демократизации AI:** пользователь без ML-экспертизы может «программировать» модель на естественном языке
3. **Эре ChatGPT:** [[02 Areas/ML & DL/Concepts/NLP/ChatGPT|ChatGPT]] = prompt-based paradigm + [[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]] + conversational UI

## Key papers

- [[02 Areas/ML & DL/Papers/Pre-train, Prompt, and Predict- A Systematic Survey of Prompting Methods in Natural Language Processing]] --- каноническая таксономия (Liu et al., 2021)
- [[02 Areas/ML & DL/Papers/GPT 3.0]] --- in-context learning как практическое воплощение paradigm

## Related concepts

- [[02 Areas/ML & DL/Concepts/NLP/Transfer Learning|Transfer Learning]] --- общая парадигма pre-train + adapt
- [[02 Areas/ML & DL/Concepts/Inference/Prompting|Prompting]] --- практика написания промптов
- [[02 Areas/ML & DL/Concepts/Inference/In-context Learning|In-context Learning]] --- few-shot через примеры в промпте
- [[02 Areas/ML & DL/Concepts/Inference/Prompt Engineering|Prompt Engineering]] --- искусство дизайна промптов
- [[02 Areas/ML & DL/Concepts/Training/Fine-tuning|Fine-tuning]] --- альтернативная парадигма адаптации
- [[02 Areas/ML & DL/Concepts/NLP/Adversarial Promting|Adversarial Prompting]] --- атаки на prompt-based системы
