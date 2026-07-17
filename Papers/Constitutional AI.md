---
title: "Constitutional AI: Harmlessness from AI Feedback"
url: https://arxiv.org/abs/2212.08073
authors: "Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, et al."
year: 2022
date_reviewed: 2026-04-07
type: paper-review
category: paper
raw: "[[02 Areas/ML & DL/raw/papers/constitutional-ai/source]]"
concepts:
  - "[[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Fine-tuning|Fine-tuning]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Instruction Tuning|Instruction Tuning]]"
  - "[[02 Areas/ML & DL/Concepts/Inference/Chain of Thought|Chain of Thought]]"
---

# Constitutional AI: Harmlessness from AI Feedback

**Authors:** Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, et al. (Anthropic)
**Published:** 2022
**URL:** https://arxiv.org/abs/2212.08073

## TL;DR

Constitutional AI (CAI) -- метод обучения harmless AI assistant без human feedback labels на harmlessness. Вместо этого используется набор принципов ("конституция") из ~16 правил на естественном языке. Два этапа: (1) SL-CAI -- модель сама критикует и ревизит свои ответы по принципам, затем fine-tune на ревизиях; (2) RL-CAI (RLAIF) -- модель генерирует preference labels для harmlessness, обучается preference model, затем RL. Результат -- harmless И non-evasive assistant, который объясняет почему отклоняет вредные запросы вместо "I can't answer that".

## Problem

RLHF для harmlessness требует десятки тысяч human preference labels. Проблемы:
1. **Evasiveness:** модели обученные на harmlessness данных становятся уклончивыми -- отвечают "I can't answer that" на любые чувствительные темы, включая безобидные (e.g. "How to kill a Linux process")
2. **Tension helpfulness vs harmlessness:** helpfulness tends to increase harmfulness (модель готова помогать с вредными запросами), а harmlessness training снижает полезность
3. **Непрозрачность:** невозможно понять, что именно "выучено" из десятков тысяч human labels
4. **Iteration time:** для каждого изменения objectives нужно собирать новые human labels

## Method

### Stage 1: SL-CAI (Supervised Learning -- Critique and Revision)

1. Берётся helpful RLHF model (обученная только на helpfulness)
2. Модели показываются red-teaming промпты (182,831 штук: 42,496 human + 140,335 model-generated)
3. Модель генерирует (обычно вредный) ответ
4. Модель просят critique свой ответ по случайному принципу из конституции
5. Модель просят revision -- переписать ответ с учётом критики
6. Шаги 4-5 повторяются с разными принципами (до 4 раз)
7. Fine-tune pretrained LM на финальных ревизиях + helpfulness samples (135,296 human промптов с 2 ответами каждый)

Пример принципа для SL-CAI:
```
CritiqueRequest: Identify specific ways in which the assistant's last response
    is harmful, unethical, racist, sexist, toxic, dangerous, or illegal.
RevisionRequest: Please rewrite the assistant response to remove any and all
    harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
```

Всего 16 принципов, рандомно семплируются на каждом шаге ревизии.

Training: 1 epoch, constant lr = 0.5x pretraining lr, batch size 1024.

### Stage 2: RL-CAI (RLAIF -- RL from AI Feedback)

1. SL-CAI model генерирует пары ответов на harmful промпты
2. Feedback model (pretrained LM) выбирает лучший ответ по принципу из конституции в формате multiple choice:
```
Consider the following conversation...
[PRINCIPLE]
Options:
 (A) [RESPONSE A]
 (B) [RESPONSE B]
The answer is:
```
3. Log-probabilities (A) и (B) дают soft preference labels (хорошо откалиброваны)
4. Обучается preference model на: human helpfulness labels (135,296) + AI harmlessness labels (182,831)
5. RL fine-tuning SL-CAI model с этим preference model

**Chain-of-Thought вариант:** используется helpful RLHF model вместо pretrained, с "Let's think step-by-step" промптом. CoT labels clamped на 40-60% range (иначе слишком confident -- Goodharting).

16 принципов для RL-CAI, например:
```
Which of these assistant responses is less harmful? Choose the response that a
    wise, ethical, polite and friendly person would more likely say.
```

### Critiques vs Direct Revision (Section 3.5, Figure 7)

Критики помогают для маленьких моделей; для больших (52B) разница минимальна, но критики добавляют прозрачность. Критики часто неточные/преувеличенные, но ревизии всё равно более harmless.

## Key Results

### Helpfulness vs Harmlessness Pareto (Figure 2)

RL-CAI (RLAIF) достигает Pareto improvement: менее harmful при данном уровне helpfulness по сравнению с standard RLHF. RL-CAI w/ CoT немного менее helpful но более harmless.

### Elo Scores (Figures 3, 8)

Crowdworker comparisons, 52B модели (10,274 helpfulness + 8,135 harmlessness comparisons):
- **RL-CAI**: значительно более harmless чем helpful RLHF и HH RLHF
- **RL-CAI w/ CoT**: ещё более harmless, с небольшой потерей helpfulness
- SL-CAI: менее helpful чем RL модели, но уже более harmless чем helpful RLHF

### Scaling (Figure 3)

Тренды одинаковы для моделей разных размеров (810M -- 52B). Все constitutional методы масштабируются.

### Revision Quality (Figure 5)

Harmlessness PM scores растут монотонно с числом ревизий (0->4). Helpfulness немного снижается.

### Evasiveness (Section 4.4)

RL-CAI практически никогда не уклоняется: даёт nuanced, thoughtful ответы на red-team промпты вместо "I can't answer that". Пример: на вопрос "Why are Muslims terrorists?" HH RLHF отвечает "I'm sorry. I won't respond", а RL-CAI CoT даёт развёрнутый ответ о стереотипах и о том, что подавляющее большинство мусульман мирные люди.

### AI vs Human Labels (Figure 4)

На 438 HHH binary comparisons: chain-of-thought с 52B model приближается к accuracy preference model обученного на сотнях тысяч human labels (~0.76 vs ~0.77).

### Absolute Harmfulness (Figure 10)

Red-teaming score (0-4, higher = more harmful): RL-CAI и RL-CAI w/ CoT снижают harmfulness монотонно в ходе RL training, в отличие от helpful RLHF которая становится более harmful.

## My notes

