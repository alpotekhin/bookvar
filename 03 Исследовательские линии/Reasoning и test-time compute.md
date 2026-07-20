---
title: Reasoning и test-time compute
type: research-line
status: emerging
started: 2022
last_updated: 2026-07-20
last_verified: 2026-07-20
key_concepts: [Chain of Thought, Self-Consistency, Search, Test-time Compute]
key_models: [DeepSeek-R1, Kimi]
primary_sources:
  - https://arxiv.org/abs/2201.11903
  - https://arxiv.org/abs/2203.11171
  - https://arxiv.org/abs/2408.03314
  - https://arxiv.org/abs/2501.12948
---

# Reasoning и test-time compute

## Тезис

Качество можно масштабировать не только параметрами и training compute, но и
вычислениями на конкретный запрос:

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/test-time-compute/snell-prm-search-methods.png]]

*Три способа расходовать дополнительный inference budget при наличии process
reward model: выбрать лучшее полное решение из N, отсекать ветви по ходу beam
search или оценивать состояние коротким lookahead. Источник: Charlie Snell et
al., [Scaling LLM Test-Time Compute Optimally, Figure 2](https://arxiv.org/abs/2408.03314).*

## Четыре разных механизма

| Механизм | Дополнительное вычисление |
|---|---|
| [[02 Areas/ML & DL/00 Учебник/13 Reasoning и Test-time Compute/01 Test-time compute#6. Длинное рассуждение и последовательное исправление|Chain of Thought]] | больше последовательных токенов |
| [[02 Areas/ML & DL/00 Учебник/13 Reasoning и Test-time Compute/01 Test-time compute#4. Self-consistency: голосование по результату|Self-Consistency]] | несколько независимых решений |
| search / tree methods | ветвление и оценка промежуточных состояний |
| tool-assisted reasoning | внешнее вычисление и наблюдение |

Длинный скрытый текст не является доказательством reasoning. Нужен выигрыш на
контролируемых задачах, устойчивость к переформулировке и проверка результата.

## Training-time связь

[[02 Areas/ML & DL/01 Справочник/Post-training/RLVR|RLVR]] создаёт давление в пользу
стратегий, которые повышают проверяемый outcome. Distillation затем переносит
часть этих стратегий в меньшие модели. Поэтому «reasoning model» — обычно не
новый Transformer block, а сочетание post-training, decoding policy и budget.

## Trade-offs

- больше токенов увеличивает latency и стоимость;
- majority vote помогает лишь при достаточно независимых ошибках;
- verifier может быть слабее генератора или эксплуатироваться им;
- overthinking ухудшает простые задачи;
- faithful explanation и эффективная внутренняя стратегия не одно и то же.

## Открытые вопросы

- Как адаптивно выбирать budget до знания сложности задачи?
- Можно ли надёжно проверять промежуточные шаги?
- Когда search лучше одной длинной trajectory?
- Как измерять reasoning без contamination и style bias?
- Какие навыки возникают через RL, а какие лишь становятся видимее?

## Связанные страницы

[[02 Areas/ML & DL/00 Учебник/13 Reasoning и Test-time Compute/01 Test-time compute|Test-time Compute]] ·
[[02 Areas/ML & DL/00 Учебник/13 Reasoning и Test-time Compute/01 Test-time compute#7. Поиск по промежуточным состояниям|Tree of Thoughts]] ·
[[02 Areas/ML & DL/02 Атлас моделей/Семейства/DeepSeek|DeepSeek-R1]] ·
[[02 Areas/ML & DL/Papers/COT|Chain-of-Thought paper]]
