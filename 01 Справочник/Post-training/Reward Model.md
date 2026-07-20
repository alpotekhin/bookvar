---
title: Reward Model
aliases: [RM, Preference Model]
type: concept
status: canonical
last_updated: 2026-07-16
primary_sources:
  - https://arxiv.org/abs/1706.03741
  - https://arxiv.org/abs/2203.02155
---

# Reward Model

**Reward model (RM)** присваивает scalar score ответу в контексте запроса. Обычно его учат на предпочтениях `chosen ≻ rejected`:

$$\mathcal L_{RM}=-\log\sigma(r_\phi(x,y^+)-r_\phi(x,y^-)).$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/post-training-courses/reward-model-training.png]]

*Одна reward model вычисляет scalar в позиции EOS для chosen и rejected
ответов; pairwise loss зависит от разности этих scores. Источник: Hugging Face,
[TRL RewardTrainer documentation](https://huggingface.co/docs/trl/reward_trainer).*

Модель не обязана калибровать абсолютный score. Pairwise objective требует
правильного порядка пары; сдвиг всех scores на одну константу не меняет loss.

RM аппроксимирует предпочтения annotators, а не объективную «полезность». Он наследует bias разметки, distribution shift и может давать exploitable ошибки. Когда policy оптимизируется слишком агрессивно, она находит способы увеличить proxy reward без реального улучшения — reward hacking / overoptimization.

Reward может быть outcome-level (за весь ответ) или process-level (за промежуточные шаги). [[02 Areas/ML & DL/01 Справочник/Post-training/Verifiable Reward|Verifiable reward]] отличается тем, что score вычисляется проверяющим правилом/средой и не обязан быть learned model.

## Подробнее

От сбора сравнений до pairwise loss, калибровки и overoptimization путь
прослеживается в главе [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/03 Reward modeling|Reward modeling]].

## Источники

- [Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741)
- [InstructGPT](https://arxiv.org/abs/2203.02155)
