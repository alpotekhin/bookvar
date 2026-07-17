---
title: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
url: https://arxiv.org/abs/2305.18290
authors: "Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn"
year: 2023
date_reviewed: 2026-04-07
type: source-note
status: legacy
category: paper
raw: "[[02 Areas/ML & DL/raw/papers/dpo/source]]"
concepts:
  - "[[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Fine-tuning|Fine-tuning]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Instruction Tuning|Instruction Tuning]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]]"
---

# Direct Preference Optimization: Your Language Model is Secretly a Reward Model

**Authors:** Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn (Stanford University / CZ Biohub)
**Published:** 2023, NeurIPS 2023
**URL:** https://arxiv.org/abs/2305.18290

## TL;DR

DPO (Direct Preference Optimization) -- метод alignment LLM, который напрямую оптимизирует политику из пар предпочтений с помощью простого binary cross-entropy loss, без отдельной Reward Model и без RL (PPO). Ключевой инсайт: language model неявно является reward model -- optimal policy для KL-constrained reward maximization можно выразить аналитически, и подставив её в Bradley-Terry model, partition function сокращается. На задачах sentiment control, summarization и dialogue DPO на уровне или лучше PPO-based RLHF при значительно меньшей сложности.

## Problem

RLHF pipeline (InstructGPT-style) сложен и нестабилен:
1. SFT -- дообучить на high-quality демонстрациях
2. Reward Model -- обучить отдельную RM на парах (preferred yw, dispreferred yl) через Bradley-Terry model
3. PPO -- maximize reward - beta * KL(pi || pi_ref)

Проблемы: одновременно в памяти несколько моделей (SFT, RM, pi_theta, pi_ref); семплирование из политики во время обучения; нестабильность PPO с высокой дисперсией градиентов; сложный подбор гиперпараметров.

## Method

### Ключевой математический insight (Section 4)

RLHF оптимизирует KL-constrained reward maximization (Eq. 3):
```
max_{pi_theta} E[r(x,y) - beta * KL(pi_theta(.|x) || pi_ref(.|x))]
```

Оптимальная политика имеет аналитическое решение (Eq. 4):
```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x,y)/beta)
```

Вывернув формулу, награда выражается через политику и reference (Eq. 5):
```
r(x,y) = beta * log[pi*(y|x) / pi_ref(y|x)] + beta * log Z(x)
```

Bradley-Terry model зависит только от разности наград -- partition function Z(x) сокращается:
```
p*(y1 > y2 | x) = sigma(beta * log[pi*(y1|x)/pi_ref(y1|x)] - beta * log[pi*(y2|x)/pi_ref(y2|x)])
```

### DPO Loss (Eq. 7)

```
L_DPO(pi_theta; pi_ref) = -E_{(x,yw,yl)~D} [
    log sigma( beta * log[pi_theta(yw|x)/pi_ref(yw|x)] - beta * log[pi_theta(yl|x)/pi_ref(yl|x)] )
]
```

Входные данные: тройки (prompt x, preferred response yw, dispreferred response yl). Один forward pass через pi_theta и pi_ref --> compute log ratios --> BCE loss. Без RL.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/dpo/fig1.png]]
*Figure 1: Сравнение RLHF pipeline (SFT -> RM -> PPO) и DPO (SFT -> direct classification loss). DPO неявно подбирает reward model, оптимальная политика которой -- pi_theta.*

### Интерпретация градиента

```
grad L_DPO = -beta * E [ sigma(r_hat(x,yl) - r_hat(x,yw)) * (grad log pi(yw|x) - grad log pi(yl|x)) ]
```

где `r_hat(x,y) = beta * log[pi_theta(y|x) / pi_ref(y|x)]` -- implicit reward.

Механика: повышает вероятность yw, снижает yl; примеры взвешены по тому, насколько плохо текущая модель ранжирует пары (adaptive weighting).

### DPO Pipeline

1. Взять SFT модель pi^SFT, установить pi_ref = pi^SFT
2. Собрать preference dataset D = {(x, yw, yl)} (можно использовать существующие: Anthropic-HH, TL;DR)
3. Обучить pi_theta минимизируя L_DPO при фиксированном pi_ref (pi_theta инициализируется от pi^SFT)

Гиперпараметры по умолчанию: beta=0.1, batch size=64, RMSprop lr=1e-6 с linear warmup 150 steps. Для TL;DR: beta=0.5.

### Theoretical Properties (Section 5)

**Theorem 1:** Класс неявных reward моделей DPO (r(x,y) = beta * log[pi(y|x)/pi_ref(y|x)]) может представить любой класс эквивалентности reward functions совместимый с Bradley-Terry моделью -- нет потери выразительности.

**Lemma 1+2:** Две reward functions из одного класса эквивалентности индуцируют одинаковое preference distribution И одинаковую optimal policy.

## Key Results

### Sentiment Control (Figure 2, left)
IMDb, GPT-2-large: DPO достигает наивысшего expected reward при любом уровне KL к reference policy -- строго доминирует PPO, включая PPO с ground-truth rewards (PPO-GT).

### TL;DR Summarization (Figure 2, right)
Win rate vs human reference summaries (GPT-4 evaluator, GPT-J SFT model):
- DPO: ~61% (temp=0.0) -- превышает PPO best-case ~57%
- Best of 128: ~55%
- Preferred-FT: ~25%
- DPO более стабилен к sampling temperature чем PPO

### Anthropic-HH Dialogue (Figure 3)
Pythia-2.8B, win rate vs chosen responses:
- DPO: ~52% -- единственный computationally efficient метод, улучшающий chosen labels
- Best of 128: ~48%
- Preferred-FT: ~38%

### Out-of-distribution (Table 1)
На CNN/DailyMail (при обучении на Reddit TL;DR): DPO win rate 36% vs PPO 26% (temp=0.0) -- лучше обобщается.

### Human evaluation (Table 2)
DPO (temp=0.25) preferred 58% vs PPO (temp=0.0); GPT-4 agreement с humans ~65-85% -- на уровне inter-annotator agreement.

## My notes
