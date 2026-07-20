---
title: "Выравнивание: RLHF → DPO → RLVR"
type: research-line
status: active
started: 2017
last_updated: 2026-07-20
last_verified: 2026-07-20
key_concepts: [SFT, Reward Model, PPO, RLHF, DPO, GRPO, RLVR]
key_models: [InstructGPT, DeepSeek-R1]
primary_sources:
  - https://arxiv.org/abs/2203.02155
  - https://arxiv.org/abs/2305.18290
  - https://arxiv.org/abs/2501.12948
---

# Выравнивание: RLHF → DPO → RLVR

## Тезис

Линия развивается не как простая замена одного алгоритма другим, а как поиск
более масштабируемого сигнала обучения:

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/instructgpt/chatgpt-training-pipeline.png]]

*Figure 2 из InstructGPT показывает классическую последовательность: SFT на
демонстрациях, обучение reward model на сравнениях и PPO по её оценке. Источник:
Long Ouyang et al., [Training language models to follow instructions with human
feedback](https://arxiv.org/abs/2203.02155).*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/dpo/dpo-pipeline.png]]

*Figure 1 из статьи DPO сопоставляет этот трёхступенчатый процесс с прямой
оптимизацией по парам preferred/rejected. Источник: Rafael Rafailov et al.,
[Direct Preference Optimization](https://arxiv.org/abs/2305.18290).*

Эти две схемы нельзя читать как хронологию, в которой новый метод отменяет
предыдущий. Они показывают разные способы превратить предпочтения в обучающий
сигнал; RLVR меняет прежде всего происхождение reward — его вычисляет verifier.

- **SFT** учит имитировать хорошие ответы.
- **RLHF** оптимизирует модель по приближённому человеческому предпочтению.
- **DPO** превращает пары предпочтений в прямой classification-like objective.
- **RLVR** использует воспроизводимо вычисляемый reward для математики, кода и
  некоторых задач tool use. Такой сигнал проверяем, но verifier не обязательно
  исчерпывает истинную корректность задачи.

## Методы и источники сигнала

Эти строки не образуют единую лестницу. PPO и GRPO отвечают за способ обновления
policy, тогда как learned reward и verifier задают источник сигнала. Например,
GRPO можно применять и с rule-based verifier, и с обученной reward model.

| Этап | Что изменилось | Что осталось проблемой |
|---|---|---|
| Preference learning | люди сравнивают ответы | дорогая и шумная разметка |
| [[02 Areas/ML & DL/01 Справочник/Post-training/RLHF|RLHF]] + [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/04 Policy gradient и PPO для LLM|PPO]] | reward model допускает online exploration | сложность стека, reward hacking |
| [[02 Areas/ML & DL/01 Справочник/Post-training/DPO|DPO]] | нет отдельного RM и rollout-loop | качество зависит от offline preference data |
| [[02 Areas/ML & DL/01 Справочник/Post-training/GRPO|GRPO]] | group-relative baseline без critic | чувствительность к sampling и reward design |
| [[02 Areas/ML & DL/01 Справочник/Post-training/RLVR|RLVR]] | reward вычисляется воспроизводимым verifier | качество ограничено полнотой verifier |

## DeepSeek-R1 как важный узел

[[02 Areas/ML & DL/05 Источники/Papers/DeepSeek-R1|DeepSeek-R1]] показал две
разные вещи, которые важно не смешивать:

1. **R1-Zero:** reasoning-поведение можно усилить RL с rule-based rewards без
   предварительного SFT на стадии post-training.
2. **R1:** практичная модель всё равно использует многостадийный pipeline:
   cold-start SFT, reasoning RL, rejection sampling и финальный alignment RL.

Следовательно, результат не доказывает, что SFT или preference data больше не
нужны. Он показывает силу проверяемого outcome reward в подходящих доменах.

## Сравнение

| Метод | Источник сигнала | Online rollouts | Сильная сторона | Главный риск |
|---|---|---:|---|---|
| SFT | демонстрации | нет | стабильность и стиль | imitation ceiling |
| RLHF | learned RM | да | open-ended preferences | reward hacking |
| DPO | preference pairs | нет | простой pipeline | нет exploration |
| RLVR | verifier | да | воспроизводимый масштабируемый reward | неполный или эксплуатируемый verifier |

## Доказательства и ограничения

- Верифицируемый **финальный ответ** не гарантирует правильную цепочку рассуждений.
- Binary reward создаёт sparse signal; curriculum и sampling становятся частью метода.
- Один и тот же verifier можно эксплуатировать shortcut-стратегией.
- Для helpfulness, safety и творческих задач learned/human feedback остаётся нужен.
- Benchmark gain нельзя автоматически переносить на надёжность в production.

## Открытые вопросы

- Когда process supervision лучше outcome supervision?
- Как обучать корректному отказу, если verifier оценивает только успех?
- Как избежать overthinking и роста стоимости inference?
- Переносятся ли RLVR-навыки за пределы распределения задач и верификаторов?
- Как сочетать verifiable и preference rewards без взаимной деградации?

## Связанные страницы

[[02 Areas/ML & DL/01 Справочник/Post-training/Reward Model|Reward Model]] ·
[[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/01 SFT и instruction data|Post-training и alignment]] ·
[[02 Areas/ML & DL/02 Атлас моделей/Семейства/DeepSeek|DeepSeek-R1]] ·
[[02 Areas/ML & DL/Papers/DPO|DPO paper]] ·
[[02 Areas/ML & DL/Papers/InstructGPT|InstructGPT]]
