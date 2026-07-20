---
title: Evaluation LLM-систем
aliases: [LLM Evaluation, Evals]
type: concept
status: canonical
last_updated: 2026-07-20
primary_sources:
  - https://arxiv.org/abs/2211.09110
  - https://arxiv.org/abs/2009.03300
---

# Evaluation LLM-систем

Оценивание начинается не с выбора модного бенчмарка, а с формулировки того,
какое свойство модели нужно проверить и в каких условиях оно должно проявиться.
Один и тот же ответ может быть точным, но плохо откалиброванным; полезным, но
слишком медленным; корректным на английском и ненадёжным на другом языке.
Поэтому итоговая оценка — это не одно число, а профиль поведения модели.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/reference-source-first/helm-taxonomy-figure2.png]]

*Слева показан обычный способ собирать benchmark: взять несколько готовых
датасетов с закреплёнными за ними метриками. Справа — подход HELM: сначала
описать пространство сценариев через задачу, предметную область, аудиторию,
время и язык, затем отдельно выбрать свойства, которые будут измеряться.
Percy Liang et al., “Holistic Evaluation of Language Models”, Figure 2,
[с. 3](https://arxiv.org/pdf/2211.09110#page=3). Фигура извлечена из статьи без
изменения содержания.*

Главная мысль рисунка в том, что датасет нельзя считать полным описанием задачи.
Natural Questions, например, фиксирует лишь один способ задавать вопросы, один
источник знаний и конкретный временной срез. Чтобы результат можно было
интерпретировать, нужно явно записать сценарий и проверить несколько независимых
свойств: accuracy, robustness, calibration, fairness, toxicity и efficiency.

## Куда идти дальше

Эта страница служит входом в два разных маршрута. Они связаны общей методикой,
но оценивают разные объекты.

### Модель и обычная LLM-система

[[02 Areas/ML & DL/00 Учебник/18 Evaluation и методология/59 Оценивание моделей и контаминация|Оценивание моделей и контаминация]] разбирает:

- как превратить продуктовую гипотезу в scenario, dataset slice и acceptance criterion;
- чем отличаются intrinsic, task-specific и system-level evals;
- когда подходят exact match, F1, pass@k, ranking metrics и human preference;
- как учитывать variance, доверительные интервалы и чувствительность к prompt format;
- почему saturation, data contamination и test-set leakage делают leaderboard ненадёжным.

Это основной маршрут для сравнения base/chat-моделей, RAG-систем и отдельных
компонентов. Начинать следует с него, если модель получает вход и возвращает
один ответ без длительного взаимодействия со средой.

### Агент и последовательность действий

[[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/68 Оценивание агентных систем|Оценивание агентных систем]] посвящено задачам, где результат зависит не только
от финального текста, но и от траектории:

- правильности выбора и параметров tools;
- состояния среды после выполнения действий;
- восстановления после ошибок;
- числа шагов, latency и стоимости;
- различию single-turn и multi-turn evaluation;
- воспроизводимому запуску через agent harness.

Обычная метрика ответа здесь недостаточна. Агент может написать убедительный
финальный текст, но не выполнить действие, испортить состояние среды или прийти
к ответу неприемлемо дорогой траекторией.

## Быстрая карта измерений

- **Capability:** знания, reasoning, coding, multilingual.
- **Task quality:** точность на реальном distribution.
- **System:** retrieval, tool use, citations, robustness.
- **Safety:** harmful outputs, privacy, prompt injection.
- **Operations:** latency, throughput, cost, reliability.

## Минимальный протокол

1. До эксперимента определить decision и acceptance criteria.
2. Разделить test set на важные slices и failure modes.
3. Зафиксировать prompt, decoding, model version и tools.
4. Сообщать uncertainty: bootstrap CI или повторные runs.
5. Проверять contamination и leakage.
6. Для LLM-as-a-judge измерять agreement с людьми, position/verbosity/style bias и использовать blind pairwise setup.
7. Сохранять примеры ошибок, а не только среднее.

Automatic metrics полезны, если соответствуют задаче: exact match, F1, pass@k, BLEU/ROUGE, Recall@k, nDCG. Ни одна из них не является универсальной метрикой «качества LLM».

## Ограничения публичных benchmark

MMLU и подобные наборы удобны для истории и сравнения, но страдают saturation, contamination, форматными эффектами и не отражают production workflow. Нельзя переносить небольшую разницу leaderboard на продукт без task-specific eval.

## Источники

- [HELM](https://arxiv.org/abs/2211.09110)
- [MMLU](https://arxiv.org/abs/2009.03300)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
