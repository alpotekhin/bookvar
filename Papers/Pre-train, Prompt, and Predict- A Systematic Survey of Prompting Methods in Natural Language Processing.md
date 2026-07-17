---
title: "Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing"
url: https://arxiv.org/abs/2107.13586
authors: [Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, Graham Neubig]
year: 2021
date_reviewed: 2026-04-06
type: source-note
status: legacy
category: paper
tags:
  - LLM
  - Review
Date: 2021-07-28
Organization: Carnegie Mellon University / National University of Singapore
Status: Done
concepts:
  - "[[02 Areas/ML & DL/Concepts/NLP/Prompt-based Learning|Prompt-based Learning]]"
  - "[[02 Areas/ML & DL/Concepts/Inference/Prompt Engineering|Prompt Engineering]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/Transfer Learning|Transfer Learning]]"
  - "[[02 Areas/ML & DL/Concepts/Inference/Prompting|Prompting]]"
  - "[[02 Areas/ML & DL/Concepts/Inference/In-context Learning|In-context Learning]]"
raw: "[[02 Areas/ML & DL/raw/papers/pre-train-prompt-and-predict--a-systematic-survey-of-prompti/paper.txt]]"
---

# Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in NLP

**Authors:** Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, Graham Neubig (CMU + NUS)
**Published:** arXiv:2107.13586v1, Jul 2021 (ACM Computing Surveys)
**URL:** https://arxiv.org/abs/2107.13586

## TL;DR

Систематический обзор нового парадигмального сдвига в NLP — **prompt-based learning**: вместо адаптации LM под задачу через fine-tuning, задачи переформулируются в текстовые подсказки (prompts) и решаются тем же LM, что и предобучался. Авторы вводят единую нотацию, классифицируют все ключевые методы по 5 измерениям (модель, форма промпта, форма ответа, multi-prompt, стратегия tuning), и документируют 100+ работ с 2018 по 2021.

## Four Paradigms in NLP

Авторы выделяют 4 последовательных парадигмы (Table 1):

| Парадигма | Тип инжиниринга | Пример |
|-----------|----------------|--------|
| a. Fully Supervised (non-neural) | Feature engineering | SVM + word features |
| b. Fully Supervised (neural) | Architecture engineering | BiLSTM, CNN, Transformer |
| c. Pre-train, Fine-tune | Objective engineering | BERT → fine-tune на задачу |
| d. **Pre-train, Prompt, Predict** | Prompt engineering | GPT-3 zero/few-shot |

Переход (b)→(c) — 2017-2019. Переход (c)→(d) — начало формироваться с 2019 (GPT-2, LAMA), закрепился с GPT-3 (2020).

**Ключевая разница (c) vs (d):**
- Fine-tune: задача адаптирует модель (LM → Task)
- Prompting: задача переформулируется под модель (Task → LM)

## Formal Description of Prompting (§2)

**Нотация:**

| Символ | Смысл | Пример |
|--------|-------|--------|
| x | Входной текст | "I love this movie." |
| y | Выходная метка | "++" |
| fprompt(x) | Функция создания промпта | применяет шаблон |
| x' | Промпт (без ответа) | "I love this movie. Overall, it was a [Z] movie." |
| ffill(x', z) | Заполненный промпт | "...it was a bad movie." |
| z | Ответ-кандидат | "good", "bad", "fantastic" |
| ẑ | Наилучший ответ | argmax P(ffill(x', z); θ) |

**Три шага prompting:**
1. **Prompt Addition** — применяем шаблон к входу x → x'
2. **Answer Search** — ищем ẑ = argmax_{z∈Z} P(ffill(x', z); θ)
3. **Answer Mapping** — переводим ẑ → ŷ (если нужно)

**Два вида промптов:**
- **Cloze prompt** — слот [Z] в середине текста. Хорошо для masked LM (BERT-style)
- **Prefix prompt** — слот [Z] в конце. Хорошо для autoregressive LM (GPT-style)

## Pre-trained Language Models (§3)

Классификация по 3 осям:

**Training objective:**
- **SLM** (Standard LM): P(x) — autoregressive, left-to-right; GPT, GPT-2, GPT-3
- **CTR** (Corrupted Text Reconstruction): восстановить зашумленный фрагмент; BERT (MLM)
- **FTR** (Full Text Reconstruction): восстановить весь текст; BART, T5

**Directionality:**
- Left-to-right (causal): GPT-семейство → хороши для prefix prompts
- Bidirectional: BERT, RoBERTa → хороши для cloze prompts
- Mixed/Prefix LM: UniLM → оба типа промптов

**Четыре типичных архитектуры pre-training:**
1. **Left-to-Right LM** — P(x) = P(x₁) × ... × P(xₙ|x₁...xₙ₋₁); GPT-3
2. **Masked LM** — предсказать замаскированные токены; BERT, RoBERTa
3. **Prefix LM** — некоторые токены видят всё (bidirec.), остальные — только prefix; UniLM
4. **Encoder-Decoder** — кодирует вход, генерирует выход; T5, BART, MASS

## Prompt Engineering (§4)

**Форма промпта:**
- Cloze vs Prefix (см. выше)
- Число слотов [X] и [Z] может варьироваться

**Ручные шаблоны (Manual):**
- LAMA (Petroni et al., 2019) — hand-crafted cloze для knowledge probing
- GPT-3 (Brown et al., 2020) — ручные prefix для 100+ задач

**Автоматические дискретные промпты (Discrete/Hard):**
- **D1: Prompt Mining** — ищем частые n-граммы между x и y в корпусе (LPAQA)
- **D2: Prompt Paraphrasing** — перефразируем seed промпт через back-translation или переписчик
- **D3: Gradient-based Search** — AutoPrompt: итеративный поиск токенов через градиент
- **D4: Prompt Generation** — T5 генерирует шаблон (LM-BFF: Gao et al., 2021)
- **D5: Prompt Scoring** — hand-craft кандидаты, LM оценивает наилучший

**Непрерывные промпты (Continuous/Soft):**
- **C1: Prefix Tuning** (Li & Liang, 2021) — prepend trainable prefix-векторы к входу; LM заморожен. Оптимизируем φ: max Σ log P(y_i | h_{<i}; θ; φ)
- **C2: Initialized with Discrete** — start с discrete промпта, fine-tune embeddings (Zhong et al., 2021)
- **C3: Hard-Soft Hybrid** — P-Tuning (Liu et al., 2021): trainable токены + anchor tokens через BiLSTM; PTR (Han et al., 2021): логические правила + виртуальные токены

## Answer Engineering (§5)

**Форма ответа:**
- **Token** — один токен из словаря (sentiment: "good"/"bad")
- **Span** — multi-token span (NER, QA)
- **Sentence** — генерация предложения (summarization, MT)

**Методы создания ответного пространства:**
- Manual: ручные verbalizer-словари (Yin et al., WARP)
- Discrete search: Answer Paraphrasing (back-translation), Prune-then-Search (AutoPrompt), Label Decomposition (Chen et al. для relation extraction)
- Continuous: виртуальные токены-классы (WARP: Hambardzumyan et al.)

## Multi-Prompt Learning (§6)

| Тип | Описание | Пример |
|-----|----------|--------|
| **Prompt Ensembling** | Несколько промптов → усредняем предсказания | LPAQA, PET-TC |
| **Prompt Augmentation** | Few-shot: answered prompts в контексте = in-context learning | GPT-3, KATE |
| **Prompt Composition** | Задача = набор под-задач с под-промптами; объединяем результаты | PTR |
| **Prompt Decomposition** | Разбиваем сложный промпт на части | TemplateNER |
| **Prompt Sharing** | Один промпт на несколько задач/доменов/языков | multi-task learning |

**In-context learning** = tuning-free prompting + prompt augmentation (answered examples в контексте). Термин введён Brown et al. (2020) в GPT-3.

## Training Strategies (§7)

**5 стратегий (Table 6):**

| Стратегия | LM params | Prompt params | Пример |
|-----------|-----------|--------------|--------|
| Promptless Fine-tuning | Tuned | — | BERT, RoBERTa |
| Tuning-free Prompting | Frozen | Frozen (нет) | GPT-3, LAMA |
| Fixed-LM Prompt Tuning | Frozen | Tuned | Prefix-Tuning, WARP |
| Fixed-prompt LM Tuning | Tuned | Frozen (нет) | PET-TC, LM-BFF |
| Prompt+LM Fine-tuning | Tuned | Tuned | PADA, P-Tuning, PTR |

**Ключевые trade-offs:**
- Tuning-free: максимальная гибкость, ноль catastrophic forgetting, но требует heavy prompt engineering
- Fixed-LM Prompt Tuning: лучший баланс в few-shot (Prefix-Tuning ~= full fine-tuning с 0.1% параметров)
- Prompt+LM: наиболее выразительный, лучший в high-data, но переобучается на малых данных

## Applications (§8) — краткий обзор

Prompt-based learning применяется к:
- **Knowledge probing** — LAMA (что LM знает о мире?)
- **Text Classification** — sentiment, topics, NLI через verbalizer
- **NER / IE** — cloze промпты для entity typing, relation extraction
- **QA** — UnifiedQA, GPT-3 few-shot QA
- **Summarization** — "TL;DR:" prefix prompt
- **Machine Translation** — "Finnish: [X] English: [Z]"
- **Automatic Evaluation** — BARTScore
- **Multi-modal** — Frozen: vision encoder → tokens → frozen LM

## Challenges (§10)

1. **Prompt Design** — нет методологии для structured inputs (деревья, графы, таблицы)
2. **Answer Engineering** — как совместно искать template + answer (сейчас делают последовательно)
3. **Selection of Tuning Strategy** — зависит от размера данных, нет универсального правила
4. **Multiple Prompt Learning** — как комбинировать промпты оптимально
5. **Selection of Pre-trained Models** — какой LM выбрать под задачу
6. **Theoretical Analysis** — почему prompting вообще работает?
7. **Transferability** — промпты под одну задачу не переносятся на другую
8. **Calibration** — GPT-3 confidence часто miscalibrated

## My notes

- Статья — **canonical survey** для prompt-based learning по состоянию на 2021. Если хочешь понять терминологию (cloze/prefix, hard/soft, verbalizer, in-context learning) — сюда.
- Типология (Figure 1) = отличная шпаргалка: 5 измерений × их варианты с конкретными ссылками.
- **In-context learning** здесь определяется как tuning-free prompting + answered examples. Сейчас этот термин часто используется шире.
- **Soft/continuous prompts** (Prefix-Tuning, Prompt-Tuning) — ключевой тренд 2021. C них началась линия parameter-efficient fine-tuning (PEFT), которая потом приведёт к LoRA и адаптерам.
- **Verbalizer** — термин для маппинга между ответом (z) и выходной меткой (y). Не очевиден но важен для понимания PET-TC и LM-BFF.
- Survey не покрывает RLHF и instruction tuning (появились позже) — ключевые парадигмы 2022-2023 смотреть в других бумагах.
- Временная шкала (Table 12): GPT-3 (May 2020) → взрыв prompt-research. К июлю 2021 уже 50+ специфических методов.
