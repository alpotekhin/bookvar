---
title: Reinforcement Learning with Verifiable Rewards
aliases: [RLVR, Reinforcement Learning from Verifiable Rewards]
type: concept
status: canonical
last_updated: 2026-07-16
primary_sources:
  - https://arxiv.org/abs/2501.12948
---

# Reinforcement Learning with Verifiable Rewards (RLVR)

**RLVR** оптимизирует language-model policy с помощью [[02 Areas/ML & DL/01 Справочник/Post-training/Verifiable Reward|проверяемой награды]]: тестов, answer checker, theorem prover или среды.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/post-training-courses/rlvr-feedback-loop.png]]

*Полный цикл RLVR: policy порождает ответы, verifier возвращает scalar reward,
после чего алгоритм RL обновляет policy и цикл повторяется. Источник: Nathan
Lambert, [RLHF & Post-Training, lecture 5, slide
8](https://rlhfbook.com/teach/course/lec5-chap7/#/7).*

Verifier и алгоритм обновления на рисунке разделены намеренно. Проверяемая
награда задаёт обратную связь, но не предписывает GRPO, PPO или иной конкретный
метод оптимизации.

## Чем отличается от RLHF

| | RLHF | RLVR |
|---|---|---|
| Источник сигнала | человеческие предпочтения → обычно RM | программная/средовая проверка |
| Масштабирование | требует разметки или preference model | дешёвое после создания verifier |
| Хорошо подходит | субъективные качества | математика, код, формальные и tool-задачи |
| Главный риск | reward-model exploitation | verifier/environment exploitation |

RLVR — парадигма reward design, а не конкретный algorithm. [[02 Areas/ML & DL/01 Справочник/Post-training/GRPO|GRPO]] — один из возможных способов обновлять policy.

## Что может выучиться

При достаточном exploration policy может находить более длинные стратегии, self-checking и способы декомпозиции без разметки каждого reasoning step. Однако видимое длинное reasoning не доказывает истинность внутреннего процесса, а успех на закрытых задачах не гарантирует перенос на открытые.

## Дизайн системы

Нужно явно фиксировать: task distribution, sampling, verifier, reward shaping, curriculum, policy algorithm, KL/regularization, contamination и evaluation. Новый paper про RLVR обычно обновляет эти компоненты и исследовательскую линию, а не создаёт «новую архитектуру Transformer».

## Подробнее

Устройство rollout loop, требования к verifier и типичные способы эксплуатации
награды рассматриваются в главе [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/06 RLVR и verifiers|RLVR и verifiers]].

## Источники

- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [OpenAI: Learning to reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/)
- [[02 Areas/ML & DL/Papers/DeepSeek-R1 Reasoning via RL|Paper note: DeepSeek-R1]]
