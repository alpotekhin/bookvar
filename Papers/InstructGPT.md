---
title: "Training language models to follow instructions with human feedback (InstructGPT)"
url: https://arxiv.org/abs/2203.02155
authors: [Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe]
year: 2022
date_reviewed: 2026-04-06
type: paper-review
category: paper
tags:
  - LLM
  - arch
Date: 2022-01-03
Organization: OpenAI
Parent item:
  - "[[02 Areas/ML & DL/Papers/GPT 3.0|GPT-3]]"
Status: Done
concepts:
  - "[[02 Areas/ML & DL/Concepts/Training/Instruction Tuning|Instruction Tuning]]"
  - "[[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]]"
  - "[[02 Areas/ML & DL/Concepts/NLP/ChatGPT|ChatGPT]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Fine-tuning|Fine-tuning]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]]"
raw: "[[02 Areas/ML & DL/raw/papers/instructgpt/paper.txt]]"
---

# Training language models to follow instructions with human feedback (InstructGPT)

**Authors:** Long Ouyang, Jeff Wu, Xu Jiang et al. (OpenAI Alignment Team)
**Published:** 2022 (arXiv:2203.02155v1, Mar 2022)
**URL:** https://arxiv.org/abs/2203.02155

## TL;DR

InstructGPT — это GPT-3, fine-tuned через [[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]] на основе человеческой обратной связи для следования инструкциям. Ключевой результат: **1.3B InstructGPT предпочтительнее 175B GPT-3** по оценке людей — при 100x меньшем числе параметров. Работа определила стандартный pipeline alignment: SFT → Reward Model → PPO (3-step RLHF), ставший основой ChatGPT и большинства современных instruction-following LLM.

## Problem

GPT-3 и аналогичные LLM предобучены на предсказание следующего токена веб-текста, а не на "выполнять инструкции полезно и безопасно". Это несоответствие (misalignment) проявляется как:
- Выдумывание фактов (галлюцинации)
- Токсичный или предвзятый вывод
- Игнорирование инструкций пользователя
- Подробные уклончивые ответы на простые вопросы

Задача — выровнять модель по реальным намерениям пользователей: **полезность (helpful), честность (honest), безвредность (harmless)** — "3H" framework (Askell et al., 2021).

## Method

### Pipeline (Figure 2): 3 шага RLHF

**Шаг 1: Supervised Fine-Tuning (SFT)**
- Наём ~40 контракторов (Upwork + ScaleAI) для написания демонстраций
- Собрано **~13,000 демонстрационных промптов** (из API + написанных контракторами)
- Контракторы пишут образцовые ответы — SFT модель учится воспроизводить их
- Обучение 16 эпох, cosine LR decay, residual dropout 0.2
- Начиная с весов GPT-3 (1.3B, 6B, 175B)

**Шаг 2: Reward Modeling (RM)**
- Контракторы ранжируют K=4..9 ответов модели на один промпт → K(K-1)/2 попарных сравнений
- Используется **только 6B RM** (175B RM нестабилен при RL обучении)
- Loss function (eq. 1):
  `loss(θ) = -1/C(K,2) · E[log(σ(r_θ(x, y_w) - r_θ(x, y_l)))]`
  где y_w — предпочтительный ответ, y_l — отвергнутый
- Все K(K-1)/2 пар из одного промпта — один batch element (против overfitting)
- **RM dataset: ~33,000 промптов**; inter-annotator agreement: 72.6%±1.5%

**Шаг 3: RL с PPO**
- Fine-tuning SFT модели как policy в bandit environment
- Reward = RM score − β·KL(π_RL || π_SFT) — per-token KL penalty предотвращает reward hacking

**PPO-ptx** (основная модель): комбинированный objective (eq. 2):
```
objective(φ) = E[r_θ(x,y) - β·log(π_φ^RL(y|x)/π^SFT(y|x))] + γ·E[log(π_φ^RL(x))]
```
γ·log-likelihood от pretraining distribution = "pretraining mix" исправляет регрессию на NLP бенчмарках ("alignment tax"). PPO: γ=0, PPO-ptx: γ>0.

### Dataset Distribution (Table 1)
API prompt distribution (RM dataset):
- Generation: 45.6%
- Open QA: 12.4%
- Brainstorming: 11.2%
- Chat: 8.4%
- Rewrite: 6.6%
- Summarization: 4.2%
- Classification: 3.5%

**Ключевой вывод:** 57% — open-ended generation/brainstorming; классические NLP задачи (QA, classification) — лишь 18%. Public NLP бенчмарки не отражают реальные use-cases.

## Key Results

### Human Preference (главный результат)

**InstructGPT vs GPT-3 (175B PPO-ptx vs GPT-3):**
- InstructGPT outputs предпочтены **85 ± 3%** времени vs GPT-3
- InstructGPT outputs предпочтены **71 ± 4%** времени vs GPT-3 (few-shot prompted)

**1.3B PPO-ptx vs 175B GPT-3:**
- 1.3B InstructGPT предпочтительнее 175B GPT-3 при **100x меньшем** числе параметров

**InstructGPT vs FLAN/T0** (175B):
- InstructGPT предпочтён над FLAN **78 ± 4%** времени
- InstructGPT предпочтён над T0 **79 ± 4%** времени

**Held-out labelers (не участвовавшие в обучении):** предпочтения совпадают с training labelers → модель не просто переобучилась на конкретных аннотаторов.

### Конкретные метаданные (Figure 4)

По сравнению с GPT-3, PPO-ptx модели:
- Лучше следуют явным ограничениям (explicit constraints)
- Реже полностью игнорируют инструкцию
- Вдвое реже галлюцинируют на closed-domain задачах: **21% vs 41%** hallucination rate

### Truthfulness (TruthfulQA, Figure 6)

InstructGPT генерирует правдивые и информативные ответы **вдвое чаще**, чем GPT-3. С "Instruction+QA" prompt InstructGPT предпочитает "I have no comment" вместо уверенной лжи.

### Toxicity (RealToxicityPrompts, Figure 7)

InstructGPT генерирует **~25% меньше токсичных ответов** при "respectful" промпте. Без специальной инструкции — небольшое улучшение.

**Ограничение:** Bias (Winogender, CrowS-Pairs) существенно **не улучшился**.

### Alignment Tax (публичные NLP бенчмарки)

PPO (без pretraining mix) → регрессия на SQuAD, DROP, HellaSwag, WMT Fr→En. PPO-ptx (с pretraining mix) — регрессия устраняется без потери preference scores.

## Limitations

- Сдвиг ценностей: модель выровнена по предпочтениям ~40 в основном американских контракторов, не по "человеческим ценностям" в целом
- Всё ещё делает простые ошибки, галлюцинирует, игнорирует инструкции
- Bias не улучшился
- Инструкции с неправильными предпосылками могут вызвать проблемы
- Дорого: требует постоянного human annotation

## My notes

- InstructGPT — **основа ChatGPT**. Это первая широко применённая демонстрация, что alignment через RLHF работает в большом масштабе.
- **3-step pipeline** (SFT → RM → PPO) стал стандартом. LLaMA 2, Claude, Gemini — все используют его вариации.
- **1.3B > 175B** — поворотный момент: alignment важнее scale для практического использования. Маленькая aligned модель бьёт большую unaligned.
- **Alignment tax** — важная концепция: RLHF улучшает instruction following, но может ухудшать NLP бенчмарки. PPO-ptx (pretraining mix) — элегантное решение.
- "Public NLP datasets are not reflective of how LMs are used" — критическое наблюдение, изменившее подход к eval. Позже закрепилось как мотивация для LMSYS Chatbot Arena, MT-Bench и т.д.
- Одновременно работали Askell et al. 2021 (Anthropic): Constitutional AI / RLAIF развился из этой же линии исследований.
