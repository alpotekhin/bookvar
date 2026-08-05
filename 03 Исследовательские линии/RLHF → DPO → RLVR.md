---
title: "Выравнивание: RLHF → DPO → RLVR"
type: research-line
status: active
started: 2017
last_updated: 2026-08-06
last_verified: 2026-08-06
key_concepts: [SFT, Reward Model, PPO, RLHF, DPO, GRPO, RLVR]
key_models: [InstructGPT, DeepSeek-R1]
primary_sources:
  - https://arxiv.org/abs/2203.02155
  - https://arxiv.org/abs/2305.18290
  - https://arxiv.org/abs/2501.12948
---

# Выравнивание: RLHF → DPO → RLVR

## Проблема, которую решает вся линия

Предобучение оценивает, насколько хорошо модель предсказывает продолжение
текста. Пользовательская задача иная: получить полезный, безопасный и
соответствующий инструкции ответ. Единственного «правильного» продолжения здесь
часто нет, поэтому supervised target и человеческое предпочтение несут разные
сигналы. История post-training — это история способов получить этот сигнал и
не дать модели воспользоваться несовершенством его измерения.

Современный RLHF вырос из preference-based reinforcement learning и
[обучения summarization по человеческой обратной связи
(2020)](https://arxiv.org/abs/2009.01325). [InstructGPT
(2022)](https://arxiv.org/abs/2203.02155) закрепил трёхступенчатую схему для
instruction-following: демонстрации для SFT, сравнения ответов для reward model,
затем PPO с ограничением отклонения от reference policy. Важный результат был
не в том, что PPO «добавил знания», а в том, что человеческие сравнения удалось
превратить в масштабируемую целевую функцию.

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

## От learned reward к прямой оптимизации

[DPO (2023)](https://arxiv.org/abs/2305.18290) использует аналитическую связь
между оптимальной KL-регуляризованной policy и reward. В результате вероятность
предпочтённого ответа можно увеличивать относительно rejected без отдельного
reward model и online PPO. Это сокращает стек и делает эксперимент
воспроизводимее, но не устраняет его предпосылки: пары всё ещё собраны от
некоторой policy, предпочтения могут быть шумными, а обучение не исследует
ответы, которых нет в датасете. Поэтому DPO — конкурентный способ preference
optimization, а не универсальная следующая ступень после RLHF.

Дальнейшие методы меняли loss, reference policy и работу с шумом, но названия
часто скрывают более важный выбор: откуда взялись ответы, кто их сравнил и на
каком распределении. При честном сравнении алгоритмов необходимо фиксировать
одинаковые prompts, пары, base model и объём вычислений.

## Почему RLVR образует отдельную ветвь

Для математики, кода и некоторых действий награду можно вычислить: сравнить
ответ, запустить тесты, проверить формальное доказательство или состояние
среды. [DeepSeek-R1 (2025)](https://arxiv.org/abs/2501.12948) показал, насколько
сильным может быть такой сигнал в большом RL-конвейере. Но «verifiable» означает
лишь воспроизводимость проверки. Unit tests могут быть неполными, точное число
не проверяет объяснение, а форматный reward можно получить без решения задачи.

GRPO в R1 убирает отдельный value critic и нормирует rewards внутри группы
ответов на один prompt. Это изменение estimator, а не синоним RLVR: тот же
verifier можно использовать с другими policy-gradient методами, а GRPO — с
другим источником reward.

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

Экспериментальная картина поэтому неоднородна. Для instruction following хорошо
подтверждено, что preference post-training меняет воспринимаемое качество даже
у меньшей модели. Для формально проверяемого reasoning подтверждён значительный
рост pass rate при увеличении sampling и RL. Гораздо слабее основания считать,
что эти методы делают скрытое рассуждение правдивым, улучшают все домены или
заменяют человеческую оценку безопасности.

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
