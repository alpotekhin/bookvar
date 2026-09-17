---
title: Post-training и Alignment — карта модуля и источники
type: textbook-chapter
status: canonical
last_updated: 2026-09-07
next: "[[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/01 SFT и instruction data]]"
primary_sources:
  - https://arxiv.org/abs/2411.15124
source_unit_id:
  - meeting-02-slides-posttraining-and-prompting
  - meeting-04-slides-tulu-recipe
  - meeting-04-slides-minimal-posttraining-recipe
  - meeting-04-reading-01-catalogue-record
---

# Post-training и Alignment

Post-training начинается там, где общая способность продолжать текст должна
превратиться в воспроизводимое поведение: отвечать на запрос, соблюдать формат,
сравнивать допустимые ответы и использовать обратную связь. Названия методов
часто заслоняют эту логику. Поэтому в модуле они расположены не по популярности,
а по тому, **какие данные получает модель и откуда берётся обучающий сигнал**.

## Что меняется между prompting, SFT, preference learning и RLVR

<!-- source_unit_id: meeting-02-slides-posttraining-and-prompting -->

Предположим, базовая модель уже умеет породить несколько правдоподобных ответов
на один запрос. Четыре режима используют эту способность по-разному.

| режим | что изменяется | данные | обучающий сигнал | новые ответы во время обучения |
|---|---|---|---|:---:|
| prompting | только контекст текущего вызова | инструкция и примеры в prompt | градиента нет | нет |
| SFT | параметры policy | пары `prompt → образцовый ответ` | token-level cross-entropy | нет |
| preference learning | параметры policy, а в RLHF ещё и reward/value models | `prompt + chosen + rejected` или оценки | относительное предпочтение | DPO — нет; PPO — да |
| RLVR | параметры policy | задачи и исполнимая спецификация проверки | reward от verifier | да |

Prompting может изменить траекторию одного ответа, но после завершения запроса
модель остаётся прежней. SFT закрепляет демонстрацию в параметрах, однако один
target не объясняет, какая из правдоподобных альтернатив хуже. Preference data
добавляют это сравнение. RLVR идёт ещё дальше: текущая policy сама порождает
попытки, а правильность устанавливает внешний verifier. Так возникает единая
цепочка, а не четыре взаимозаменяемых способа «улучшить модель».

Именно этот переход показан на слайдах 24–26 лекции Jason Weston
[Learning to Self-Improve & Reason with LLMs](https://rdi.berkeley.edu/adv-llm-agents/slides/Jason-Weston-Reasoning-Alignment-Berkeley-Talk.pdf#page=24)
из курса
[Berkeley Advanced LLM Agents, Spring 2025](https://rdi.berkeley.edu/adv-llm-agents/sp25).

## Tülu как сквозной открытый recipe

<!-- source_unit_id: meeting-04-slides-tulu-recipe -->
<!-- source_unit_id: meeting-04-reading-01-catalogue-record -->

[Tülu 3](https://arxiv.org/abs/2411.15124) полезен здесь не как ещё одно имя
модели, а как опубликованный эксперимент, в котором видна вся связь между
данными и стадиями обучения. Авторы начинают с base checkpoint Llama 3.1,
строят многозадачную SFT-смесь, затем обучают policy на preference pairs с
length-normalized DPO и завершают процесс RLVR на задачах с программно
проверяемым результатом. Для каждой стадии опубликованы данные, код, evaluation
suite и абляции.

Эта последовательность не означает, что DPO или RLVR обязательны для всякой
модели. Она показывает более важный принцип: следующая стадия должна отвечать на
ошибку, которую предыдущая стадия не умеет выразить своим сигналом.

- Если модель не знает желаемого формата, нужен хороший SFT-пример.
- Если два ответа допустимы, но один предпочтительнее, нужна сравнительная
  разметка и явное определение критерия.
- Если ответ можно проверить исполнением, не следует заменять verifier
  непрозрачной моделью награды.
- Если новые rollouts создаются текущей policy, train/eval prompts, параметры
  декодирования и версия проверяющей программы становятся частью эксперимента.

Лекция Hannaneh Hajishirzi разбирает этот путь как единый граф
`base model → instruction tuning → preference tuning → RLVR` на
[слайдах 29, 60 и 90](https://rdi.berkeley.edu/adv-llm-agents/slides/OLMo-Tulu-Reasoning-Hanna.pdf#page=29).
Числа из Tülu относятся к конкретным моделям, смесям и evaluation suite; они не
служат универсальными коэффициентами для нового проекта.

## Минимальная воспроизводимая последовательность

<!-- source_unit_id: meeting-04-slides-minimal-posttraining-recipe -->

Даже короткий post-training experiment должен начинаться не с команды запуска,
а с замкнутого протокола.

1. Зафиксировать целевые способности и отложенные evaluation sets до сбора
   обучающих примеров.
2. Собрать prompts из предполагаемого пользовательского распределения; записать
   происхождение, лицензию, способ дедупликации и результаты decontamination.
3. Обучить SFT checkpoint и проверить не только средний score, но и регрессии по
   отдельным способностям.
4. Для preference stage записать, кто породил каждый ответ, кто сравнил пару и
   совпадает ли распределение генератора с текущей policy.
5. Если используется PPO, отдельно проверить reward model и value model. Если
   используется DPO, проверить reference model, chat template и sequence log-probabilities.
6. Для RLVR отделить parser от verifier, заморозить его версию и проверить
   ложные принятия до первого обновления policy.
7. Повторить исходную evaluation suite с теми же prompts, sampling settings и
   правилами подсчёта; новую обучающую награду не выдавать за независимую оценку.

В лекции этот замысел свёрнут до «minimal recipe»
[на слайдах 121–125](https://rdi.berkeley.edu/adv-llm-agents/slides/OLMo-Tulu-Reasoning-Hanna.pdf#page=121).
В учебнике он развёрнут по главам ниже, чтобы на каждом шаге можно было
восстановить тензоры, данные и проверку.

## Маршрут по модулю

1. [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/01 SFT и instruction data|SFT и instruction data]] — как демонстрация превращается в response-only loss и как строится обучающая смесь.
2. [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/02 Preference data|Preference data]] — откуда берутся chosen/rejected и почему разметка является частью задачи.
3. [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/03 Reward modeling|Reward modeling]] — Bradley–Terry, LLM-as-a-Judge и обучение оценщика.
4. [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/04 Policy gradient и PPO для LLM|Policy gradient и PPO]] — on-policy rollouts, value function и KL-ограничение.
5. [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/05 DPO|DPO]] — offline preference optimization и её итеративные варианты.
6. [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/06 RLVR и verifiers|RLVR и verifiers]] — программно проверяемая награда как контракт среды.
7. [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/07 GRPO и DeepSeek-R1|GRPO и DeepSeek-R1]] — групповая относительная оценка и конкретный reasoning recipe.
8. [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/08 Reasoning distillation|Reasoning distillation]] — перенос проверенных траекторий в меньшую модель.

> [!summary]
> Post-training не является одним алгоритмом. Это последовательность решений о
> распределении запросов, форме supervision, происхождении оценки и моменте,
> когда модель начинает собирать данные собственной policy.
