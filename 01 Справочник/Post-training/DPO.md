---
title: Direct Preference Optimization
aliases: [DPO]
type: concept
status: canonical
last_updated: 2026-07-16
primary_sources:
  - https://arxiv.org/abs/2305.18290
---

# Direct Preference Optimization (DPO)

**DPO** оптимизирует policy непосредственно на парах предпочтений без отдельного reward-model training и online RL.

$$
\mathcal L_{DPO}=
-\log\sigma\left(
\beta\log\frac{\pi_\theta(y^+|x)}{\pi_{ref}(y^+|x)}
-\beta\log\frac{\pi_\theta(y^-|x)}{\pi_{ref}(y^-|x)}
\right).
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/dpo/dpo-pipeline.png]]

*Слева показан классический RLHF с отдельной reward model и RL; справа DPO
преобразует preference pairs непосредственно в функцию потерь языковой модели.
Источник: Rafael Rafailov et al., [Direct Preference Optimization, Figure
1](https://arxiv.org/abs/2305.18290).*

Схема объясняет слово *direct*: reward не исчезает из математической модели
предпочтений, но не реализуется отдельной обученной сетью и не используется в
online rollout loop.

Интуитивно policy должна повысить относительное предпочтение chosen против rejected по сравнению с reference. Параметр $\beta$ регулирует силу отклонения.

## Trade-offs

Плюсы: простой supervised-like loop, нет rollout generation внутри каждого update и отдельной RM. Минусы: качество ограничено offline preference pairs; objective всё равно опирается на модель предпочтений и чувствителен к noise, length bias и coverage данных.

DPO не является drop-in эквивалентом любого RLHF setup: online exploration, verifiable environments и sequence-level credit assignment могут требовать RL-подхода.

## Подробнее

Вывод objective из модели предпочтений, роль reference policy и практические
режимы обучения рассмотрены в главе [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/05 DPO|DPO]].

## Источники

- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [[02 Areas/ML & DL/Papers/DPO|Paper note: DPO]]
