---
title: Group Relative Policy Optimization
aliases: [GRPO]
type: concept
status: canonical
last_updated: 2026-07-16
primary_sources:
  - https://arxiv.org/abs/2402.03300
  - https://arxiv.org/abs/2501.12948
---

# Group Relative Policy Optimization (GRPO)

**GRPO** — policy optimization method, который для одного prompt сэмплирует группу ответов и строит advantage относительно rewards этой группы, не обучая отдельную value/critic model.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/post-training-courses/grpo-group-relative.png]]

*Несколько ответов на один prompt образуют группу: их rewards центрируются и
масштабируются внутри группы, создавая advantages без отдельной value model.
Источник: Nathan Lambert,
[RLHF & Post-Training, lecture 6, slide 8](https://rlhfbook.com/teach/course/lec6-grpo/#/7),
по DeepSeekMath.*

Следовательно, baseline в GRPO зависит от соседних samples того же prompt.
Награда ответа не интерпретируется изолированно: важна его позиция внутри
конкретной группы.

Упрощённая оценка:

$$
A_i=\frac{r_i-\operatorname{mean}(r_{1:G})}
{\operatorname{std}(r_{1:G})+\epsilon}.
$$

Полный objective включает probability ratios, clipping и обычно KL regularization; детали различаются между реализациями и papers.

## Что решает

Отказ от critic уменьшает память и сложность по сравнению с PPO-style actor-critic. Сравнение внутри группы создаёт baseline для одного prompt.

## Ограничения

Нужны несколько rollouts на prompt; слабая вариативность rewards даёт слабый сигнал; нормализация делает update зависимым от состава группы; остаются off-policy/staleness, reward hacking и stability issues.

GRPO не является синонимом [[02 Areas/ML & DL/01 Справочник/Post-training/RLVR|RLVR]]: это алгоритм оптимизации. RLVR может использовать GRPO, PPO, REINFORCE-подобные методы или иные алгоритмы.

## Подробнее

Полный objective и место GRPO в рецепте DeepSeek-R1 разобраны в главе
[[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/07 GRPO и DeepSeek-R1|GRPO и DeepSeek-R1]].

## Источники

- [DeepSeekMath](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
