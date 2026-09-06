---
title: Post-training и RLVR для математических рассуждений
type: practice
status: canonical
last_updated: 2026-09-07
contract: Contracts/stanford-cs336-a5.yml
primary_sources:
  - https://github.com/stanford-cs336/assignment5-alignment/tree/c2734a26308710949fe13226960a1e8cece94b7e
source_unit_id:
  - assignment-05-deliverable-grpo-experiments-off-policy-035
  - assignment-05-deliverable-grpo-experiments-standard-on-policy-012
  - assignment-05-deliverable-grpo-experiments-standard-on-policy-013
  - assignment-05-deliverable-grpo-experiments-standard-on-policy-014
  - assignment-05-deliverable-grpo-experiments-standard-on-policy-015
  - assignment-05-deliverable-grpo-experiments-standard-on-policy-016
  - assignment-05-deliverable-grpo-experiments-variants-on-policy-029
  - assignment-05-deliverable-grpo-learning-rate-017
  - assignment-05-deliverable-grpo-prompt-ablation-018
  - assignment-05-deliverable-prompting-baselines-001
  - assignment-05-deliverable-prompting-baselines-002
  - assignment-05-deliverable-try-your-own-036
  - assignment-05-grpo-correctness
  - assignment-05-learning-rate-sweep
  - assignment-05-on-off-policy-comparison
  - assignment-05-prompting-baselines
  - assignment-05-safety-supplement-alpacaeval-metric
  - assignment-05-safety-supplement-deliverable-alpaca-eval-baseline-013
  - assignment-05-safety-supplement-deliverable-alpaca-eval-baseline-014
  - assignment-05-safety-supplement-deliverable-alpaca-eval-baseline-015
  - assignment-05-safety-supplement-deliverable-alpaca-eval-baseline-016
  - assignment-05-safety-supplement-deliverable-alpaca-eval-sft-032
  - assignment-05-safety-supplement-deliverable-alpaca-eval-sft-033
  - assignment-05-safety-supplement-deliverable-alpaca-eval-sft-034
  - assignment-05-safety-supplement-deliverable-dpo-training-043
  - assignment-05-safety-supplement-deliverable-dpo-training-044
  - assignment-05-safety-supplement-deliverable-dpo-training-045
  - assignment-05-safety-supplement-deliverable-dpo-training-046
  - assignment-05-safety-supplement-deliverable-gsm8k-baseline-007
  - assignment-05-safety-supplement-deliverable-gsm8k-baseline-008
  - assignment-05-safety-supplement-deliverable-gsm8k-baseline-009
  - assignment-05-safety-supplement-deliverable-gsm8k-baseline-010
  - assignment-05-safety-supplement-deliverable-gsm8k-baseline-011
  - assignment-05-safety-supplement-deliverable-gsm8k-baseline-012
  - assignment-05-safety-supplement-deliverable-gsm8k-sft-029
  - assignment-05-safety-supplement-deliverable-gsm8k-sft-030
  - assignment-05-safety-supplement-deliverable-gsm8k-sft-031
  - assignment-05-safety-supplement-deliverable-look-at-hh-040
  - assignment-05-safety-supplement-deliverable-look-at-hh-041
  - assignment-05-safety-supplement-deliverable-look-at-sft-021
  - assignment-05-safety-supplement-deliverable-mmlu-baseline-001
  - assignment-05-safety-supplement-deliverable-mmlu-baseline-002
  - assignment-05-safety-supplement-deliverable-mmlu-baseline-003
  - assignment-05-safety-supplement-deliverable-mmlu-baseline-004
  - assignment-05-safety-supplement-deliverable-mmlu-baseline-005
  - assignment-05-safety-supplement-deliverable-mmlu-baseline-006
  - assignment-05-safety-supplement-deliverable-mmlu-sft-026
  - assignment-05-safety-supplement-deliverable-mmlu-sft-027
  - assignment-05-safety-supplement-deliverable-mmlu-sft-028
  - assignment-05-safety-supplement-deliverable-red-teaming-038
  - assignment-05-safety-supplement-deliverable-red-teaming-039
  - assignment-05-safety-supplement-deliverable-sft-025
  - assignment-05-safety-supplement-deliverable-sst-baseline-017
  - assignment-05-safety-supplement-deliverable-sst-baseline-018
  - assignment-05-safety-supplement-deliverable-sst-baseline-019
  - assignment-05-safety-supplement-deliverable-sst-baseline-020
  - assignment-05-safety-supplement-deliverable-sst-sft-035
  - assignment-05-safety-supplement-deliverable-sst-sft-036
  - assignment-05-safety-supplement-deliverable-sst-sft-037
  - assignment-05-safety-supplement-dpo-comparison
  - assignment-05-safety-supplement-gsm8k-metric
  - assignment-05-safety-supplement-mmlu-metric
  - assignment-05-safety-supplement-safety-metric
  - assignment-05-safety-supplement-safety-prompt-and-mmlu-setup
  - assignment-05-safety-supplement-sft-comparison
  - assignment-05-safety-supplement-task-alpaca-eval-baseline
  - assignment-05-safety-supplement-task-alpaca-eval-sft
  - assignment-05-safety-supplement-task-data-loading
  - assignment-05-safety-supplement-task-dpo-loss
  - assignment-05-safety-supplement-task-dpo-training
  - assignment-05-safety-supplement-task-gsm8k-baseline
  - assignment-05-safety-supplement-task-gsm8k-sft
  - assignment-05-safety-supplement-task-mmlu-baseline
  - assignment-05-safety-supplement-task-mmlu-sft
  - assignment-05-safety-supplement-task-red-teaming
  - assignment-05-safety-supplement-task-sft
  - assignment-05-safety-supplement-task-sft-script
  - assignment-05-safety-supplement-task-sst-baseline
  - assignment-05-safety-supplement-task-sst-sft
  - assignment-05-task-compute-policy-gradient-loss-off-policy
  - assignment-05-task-compute-policy-gradient-loss-off-policy-gspo
  - assignment-05-task-compute-policy-gradient-loss-on-policy
  - assignment-05-task-compute-rollout-rewards
  - assignment-05-task-get-response-log-probs
  - assignment-05-task-grpo-experiments-off-policy
  - assignment-05-task-grpo-experiments-standard-on-policy
  - assignment-05-task-grpo-experiments-variants-on-policy
  - assignment-05-task-grpo-learning-rate
  - assignment-05-task-grpo-prompt-ablation
  - assignment-05-task-grpo-train-step-off-policy
  - assignment-05-task-grpo-train-step-standard-on-policy
  - assignment-05-task-grpo-train-step-variants-on-policy
  - assignment-05-task-prompting-baselines
  - assignment-05-task-tokenize-prompt-and-output
  - assignment-05-task-try-your-own
---

# Практика 24. Post-training и RLVR для математических рассуждений

## Цель

Постройте воспроизводимый контур «запрос → генерация → разбор ответа → проверка →
обновление модели → оценивание» и на одних и тех же данных сравните подсказки без
обучения, SFT на фиксированных демонстрациях, обучение на отобранных решениях
(RFT), обычный GRPO, Dr. GRPO, учебный вариант MaxRL и оценки градиента по
устаревшим данным. Итогом должна стать не одна удачная кривая награды, а полный
пакет свидетельств: по нему другой исследователь сможет восстановить версию
модели, параметры сэмплирования, давность сгенерированных ответов, точную функцию
потерь и ошибки проверяющей программы.

Практика адаптирует [Stanford CS336 Assignment
5](https://github.com/stanford-cs336/assignment5-alignment/tree/c2734a26308710949fe13226960a1e8cece94b7e).
Полная конфигурация Stanford с OLMo-2-0425-1B, GSM8K и B200 служит референсом,
а не обязательным локальным порогом.

## Режимы

- `smoke_cpu` — tiny frozen model, локальные fixtures, все adapter tests,
  повторяемые hashes; сеть после setup не нужна.
- `onpolicy_small_gpu` — prompting, grader audit, фиксированный SFT baseline,
  GRPO, Dr. GRPO, RFT и `MaxRL_course` при сопоставимом token/update budget.
- `offpolicy_small_gpu` — один пакет rollouts используется в 32 minibatch
  updates; сравниваются `none`, `noclip`, token-clipped GRPO и GSPO.
- `optional_dpo_safety` — отдельная ветка packed SFT → HH DPO → MMLU, GSM8K,
  AlpacaEval и SimpleSafetyTests. Она не нужна для зачёта основной практики по
  обучению рассуждениям.
- `course_full_reference` — неизменённые параметры Stanford и четыре seed;
  только справочный режим.

<a id="prompting-baselines"></a>
<!-- source_unit_id: assignment-05-deliverable-prompting-baselines-001 -->
<!-- source_unit_id: assignment-05-deliverable-prompting-baselines-002 -->
## 1. Зафиксировать исходный результат без обучения

<!-- source_unit_id: assignment-05-task-prompting-baselines -->
<!-- source_unit_id: assignment-05-prompting-baselines -->

До обучения прогоните варианты zero-shot, few-shot и chain-of-thought на одной
закреплённой части валидационной выборки. Сохраните дословный запрос, версии
токенизатора и модели, параметры сэмплирования, начальное значение генератора
случайных чисел, исходный ответ и результат его разбора. В отчёт входят
точность, доля ошибок разбора, длина, пропускная способность и не менее десяти
разобранных ошибок. Нельзя менять проверяющую программу после просмотра
результата одной системы, не перезапустив остальные варианты.

Затем подготовьте **неизменяемый исходный вариант SFT**: закреплённый набор
решений из обучающей части с хэшами и сведениями о происхождении, без примеров из
валидации. Число токенов ответов и обновлений оптимизатора должно быть
сопоставимо с последующими методами. Этот запуск отвечает на вопрос, сколько
улучшения даёт обычная имитация заранее доступных решений. RFT ниже отвечает на
другой вопрос: что произойдёт, если решения породит текущая модель, а
проверяющая программа отберёт только успешные.

<a id="response-mask-logprobs"></a>
## 2. Доказать правильность маски и логарифмов вероятностей

<!-- source_unit_id: assignment-05-task-tokenize-prompt-and-output -->
<!-- source_unit_id: assignment-05-task-get-response-log-probs -->

Реализуйте точные интерфейсы:

```python
run_tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
run_get_response_log_probs(model, input_ids, labels, return_token_entropy)
```

На небольшом фиксированном примере с известными идентификаторами токенов
распечатайте `input_ids`, сдвинутые `labels`, `response_mask` и логарифм
вероятности каждого токена. Токены запроса и дополнения до общей длины имеют
маску 0; первый токен ответа предсказывается по последней позиции запроса. Не
вставляйте неоговорённые BOS/EOS между отдельно токенизированными строками.

<a id="verifier-audit"></a>
## 3. Проверить разборщик ответа и проверяющую программу до обучения

<!-- source_unit_id: assignment-05-task-compute-rollout-rewards -->

```python
run_compute_rollout_rewards(reward_fn, rollout_responses,
                            repeated_ground_truths)
```

Записывайте `answer_reward` и `format_reward` раздельно. В обязательном варианте
обучающая награда определяется правильностью ответа; соблюдение формата не даёт
частичного балла. Составьте `parser_audit.csv` с успешными и неуспешными
случаями, вручную отметьте ложные принятия и ложные отклонения. Обязательно
проверьте лишние числа, эквивалентные дроби, научную запись, несколько тегов
ответа, знак минуса Unicode, `NaN/Inf`, единицы измерения и текст после
закрывающего тега.

<a id="onpolicy-grpo"></a>
<!-- source_unit_id: assignment-05-grpo-correctness -->
<!-- source_unit_id: assignment-05-task-grpo-experiments-standard-on-policy -->
<!-- source_unit_id: assignment-05-deliverable-grpo-experiments-standard-on-policy-012 -->
<!-- source_unit_id: assignment-05-deliverable-grpo-experiments-standard-on-policy-013 -->
<!-- source_unit_id: assignment-05-deliverable-grpo-experiments-standard-on-policy-014 -->
<!-- source_unit_id: assignment-05-deliverable-grpo-experiments-standard-on-policy-015 -->
<!-- source_unit_id: assignment-05-deliverable-grpo-experiments-standard-on-policy-016 -->
<!-- source_unit_id: assignment-05-learning-rate-sweep -->
<!-- source_unit_id: assignment-05-task-grpo-learning-rate -->
<!-- source_unit_id: assignment-05-deliverable-grpo-learning-rate-017 -->
<!-- source_unit_id: assignment-05-task-grpo-prompt-ablation -->
<!-- source_unit_id: assignment-05-deliverable-grpo-prompt-ablation-018 -->
<!-- source_unit_id: assignment-05-task-grpo-train-step-variants-on-policy -->
## 4. Собрать обычный on-policy GRPO

<!-- source_unit_id: assignment-05-task-compute-policy-gradient-loss-on-policy -->
<!-- source_unit_id: assignment-05-task-grpo-train-step-standard-on-policy -->

Реализуйте ещё пять интерфейсов:

```python
run_compute_group_normalized_rewards(raw_rewards, group_size, baseline="mean",
                                     advantage_eps=1e-6,
                                     advantage_normalizer="std")
run_compute_policy_gradient_loss(raw_rewards_or_advantages, policy_log_probs,
                                 importance_reweighting_method="none",
                                 old_log_probs=None, cliprange=None,
                                 response_mask=None)
run_aggregate_loss_across_microbatch(per_token_policy_gradient_loss, mask,
                                     loss_normalization="sequence",
                                     normalization_constant=None)
run_grpo_train_step(model, tokenizer, optimizer, gradient_accumulation_steps,
                    max_grad_norm, reward_fn, repeated_prompts,
                    rollout_responses, repeated_ground_truths, group_size,
                    baseline="mean", advantage_eps=1e-6,
                    advantage_normalizer="std",
                    importance_reweighting_method="none", old_log_probs=None,
                    cliprange=None, loss_normalization="sequence",
                    normalization_constant=None)
```

На каждом шаге сохраняйте исходные награды, среднее и стандартное отклонение в
группе, долю групп с нулевой дисперсией, длины ответов, энтропию, норму градиента,
валидационную точность и примеры ответов. В on-policy режиме на одном свежем
пакете генераций выполняется одно обновление оптимизатора. Порог средней
валидационной точности 25% относится только к полной конфигурации Stanford;
локальный режим оценивается по соблюдению контракта и полноте свидетельств.

<a id="onpolicy-variants"></a>
<!-- source_unit_id: assignment-05-deliverable-grpo-experiments-variants-on-policy-029 -->
## 5. Разделить две нормировки GRPO

<!-- source_unit_id: assignment-05-task-grpo-experiments-variants-on-policy -->

С одинаковыми запросами, проверяющей программой, инициализацией, бюджетом
генераций и обновлений, а также процедурой оценивания сравните:

| variant | baseline | advantage normalizer | loss normalizer |
|---|---|---|---|
| `SFT_fixed_demo` | — | — | response-token mean |
| `GRPO_constant` | mean | std | constant |
| `Dr_GRPO` | mean | none | constant |
| `RFT` | none | none | constant |
| `MaxRL_course` | mean | mean | constant |

`normalization_constant=B*G*max_generation_length`. Для MaxRL храните
распределение `mu`, весов и норм градиента; это учебный вариант, поскольку
оригинальная статья нормирует функцию потерь фактическим числом токенов. Члены
с нулевым преимуществом можно удалять до прямого прохода лишь после теста численной
эквивалентности градиента. Полный режим использует четыре seed, budgeted — не
менее трёх; smoke может один, но не делает сравнительных выводов.

<a id="offpolicy-variants"></a>
<!-- source_unit_id: assignment-05-on-off-policy-comparison -->
<!-- source_unit_id: assignment-05-deliverable-grpo-experiments-off-policy-035 -->
<!-- source_unit_id: assignment-05-task-try-your-own -->
<!-- source_unit_id: assignment-05-deliverable-try-your-own-036 -->
## 6. Измерить цену повторного использования сгенерированных ответов

<!-- source_unit_id: assignment-05-task-compute-policy-gradient-loss-off-policy -->
<!-- source_unit_id: assignment-05-task-compute-policy-gradient-loss-off-policy-gspo -->
<!-- source_unit_id: assignment-05-task-grpo-train-step-off-policy -->
<!-- source_unit_id: assignment-05-task-grpo-experiments-off-policy -->

Зафиксируйте `old_log_probs` во время генерации, затем разбейте пакет из 256
ответов на 32 мини-пакета по 8. Сравните:

- `none`: устаревшие данные без поправки;
- `noclip`: отношение важности для каждого токена;
- `grpo`: ограниченное отношение вероятностей токенов, как в PPO, с
  `cliprange=0.2`;
- `gspo`: геометрическое среднее отношения для всего ответа с
  `cliprange=3e-4`.

GSPO усредняет логарифм отношения только по токенам ответа. Записывайте номер
обновления внутри пакета, отставание обучаемой модели от модели-генератора,
отношения для токенов и целых последовательностей, доли верхних и нижних
ограничений, эффективный размер выборки, полное время и число сгенерированных
токенов. Сравнение скорости должно учитывать и генерацию, и обучение: 32
обновления на одном пакете не означают автоматического ускорения в 32 раза.

<a id="dpo-safety-branch"></a>
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-look-at-sft-021 -->
<!-- source_unit_id: assignment-05-safety-supplement-task-data-loading -->
<!-- source_unit_id: assignment-05-safety-supplement-task-sft-script -->
<!-- source_unit_id: assignment-05-safety-supplement-task-sft -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-sft-025 -->
<!-- source_unit_id: assignment-05-safety-supplement-task-mmlu-sft -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-mmlu-sft-026 -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-mmlu-sft-027 -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-mmlu-sft-028 -->
<!-- source_unit_id: assignment-05-safety-supplement-task-gsm8k-sft -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-gsm8k-sft-029 -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-gsm8k-sft-030 -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-gsm8k-sft-031 -->
<!-- source_unit_id: assignment-05-safety-supplement-task-alpaca-eval-sft -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-alpaca-eval-sft-032 -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-alpaca-eval-sft-033 -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-alpaca-eval-sft-034 -->
<!-- source_unit_id: assignment-05-safety-supplement-task-sst-sft -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-sst-sft-035 -->
<!-- source_unit_id: assignment-05-safety-supplement-sft-comparison -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-sst-sft-036 -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-sst-sft-037 -->
<!-- source_unit_id: assignment-05-safety-supplement-task-red-teaming -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-red-teaming-038 -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-red-teaming-039 -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-look-at-hh-040 -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-look-at-hh-041 -->
<!-- source_unit_id: assignment-05-safety-supplement-task-dpo-loss -->
<!-- source_unit_id: assignment-05-safety-supplement-task-dpo-training -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-dpo-training-043 -->
<!-- source_unit_id: assignment-05-safety-supplement-dpo-comparison -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-dpo-training-044 -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-dpo-training-045 -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-dpo-training-046 -->
<a id="dpo-safety-baselines"></a>
<!-- source_unit_id: assignment-05-safety-supplement-safety-prompt-and-mmlu-setup -->
<!-- source_unit_id: assignment-05-safety-supplement-mmlu-metric -->
<!-- source_unit_id: assignment-05-safety-supplement-task-mmlu-baseline -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-mmlu-baseline-001 -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-mmlu-baseline-002 -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-mmlu-baseline-003 -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-mmlu-baseline-004 -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-mmlu-baseline-005 -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-mmlu-baseline-006 -->
<!-- source_unit_id: assignment-05-safety-supplement-gsm8k-metric -->
<!-- source_unit_id: assignment-05-safety-supplement-task-gsm8k-baseline -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-gsm8k-baseline-007 -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-gsm8k-baseline-008 -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-gsm8k-baseline-009 -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-gsm8k-baseline-010 -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-gsm8k-baseline-011 -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-gsm8k-baseline-012 -->
<!-- source_unit_id: assignment-05-safety-supplement-alpacaeval-metric -->
<!-- source_unit_id: assignment-05-safety-supplement-task-alpaca-eval-baseline -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-alpaca-eval-baseline-013 -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-alpaca-eval-baseline-014 -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-alpaca-eval-baseline-015 -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-alpaca-eval-baseline-016 -->
<!-- source_unit_id: assignment-05-safety-supplement-task-sst-baseline -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-sst-baseline-017 -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-sst-baseline-018 -->
<!-- source_unit_id: assignment-05-safety-supplement-safety-metric -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-sst-baseline-019 -->
<!-- source_unit_id: assignment-05-safety-supplement-deliverable-sst-baseline-020 -->
## 7. Необязательная ветка: packed SFT и HH DPO

В отдельном окружении реализуйте:

```python
get_packed_sft_dataset(tokenizer, dataset_path, seq_length, shuffle)
run_iterate_batches(dataset, batch_size, shuffle)
run_parse_mmlu_response(mmlu_example, model_output)
run_parse_gsm8k_response(model_output)
run_compute_per_instance_dpo_loss(lm, lm_ref, tokenizer, beta, prompt,
                                  response_chosen, response_rejected)
```

Сначала сохраните исходные результаты zero-shot, затем проведите одинаковое
оценивание после SFT и DPO. Примеры HH проверяются вручную до обучения. В DPO
обучаемая и опорная модели должны использовать один токенизатор и один шаблон
диалога. В отчёте укажите обычную и скорректированную по длине долю побед
AlpacaEval, долю безопасных ответов, точность MMLU/GSM8K, пропускную способность,
ошибки разбора и разбор ошибок модели. Улучшение по оценке предпочтений не
компенсирует скрытое ухудшение рассуждений или безопасности.

<a id="evidence-bundle"></a>
## 8. Пакет результатов и критерии зачёта

```text
evidence/
  environment.json
  source_manifest.json
  prompt_and_grader_manifest.json
  prompting_baselines.jsonl
  parser_audit.csv
  response_mask_fixture.json
  algorithm_matrix.yml
  rollout_manifest.jsonl
  reward_components.jsonl
  train_metrics.csv
  validation_metrics.csv
  ratio_and_clip_metrics.csv
  seed_summary.csv
  rollout_examples_before_after.md
  optional_dpo_metrics.json
  artifact_hashes.json
  report.md
```

Зачёт требует пройти неизменённые тесты исходного задания, получить конечные
значения на группах с одинаковой наградой, воспроизвести одинаковые хэши двух
коротких запусков, вручную проверить разборщик, сохранить происхождение старых
логарифмов вероятностей, сравнить методы при одинаковом бюджете и провести
оценку, независимую от обучающей проверяющей программы. Каждый артефакт получает
SHA-256. Отрицательный результат допустим, если контракт соблюдён, а причины
разобраны.

## Первоисточники

- [Stanford CS336 Assignment 5 — Alignment and Reasoning RL](https://github.com/stanford-cs336/assignment5-alignment/tree/c2734a26308710949fe13226960a1e8cece94b7e)
- [DeepSeekMath: GRPO](https://arxiv.org/abs/2402.03300)
- [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)
