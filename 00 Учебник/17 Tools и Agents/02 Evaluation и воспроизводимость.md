---
title: Evaluation и воспроизводимость
type: textbook-chapter
status: canonical
last_updated: 2026-07-16
---

# Evaluation и воспроизводимость

> [!abstract] Идея главы
> Benchmark score имеет смысл только вместе с точной задачей, dataset version,
> prompt, decoding и evaluation code. Одна средняя цифра скрывает распределение
> ошибок и почти ничего не говорит о production system.

Одна aggregate цифра не описывает модель или систему.

## Уровни оценки

1. **Model capability:** MMLU, math, code, language.
2. **Component:** retrieval recall, reranker NDCG, tool-call validity.
3. **System:** end-to-end success, latency, cost, groundedness.
4. **Safety:** harmfulness, privacy, prompt injection.
5. **Operations:** regressions, drift, failure recovery.

LLM-as-a-judge масштабируется, но имеет position, verbosity, self-preference и
style biases. Verifiable tasks надёжнее, но покрывают узкий класс задач.

## Минимальный eval record

```yaml
model: exact checkpoint or API version
tokenizer_and_template: exact versions
dataset: name, split, revision
prompt: full template
decoding: temperature, top_p, max_tokens, seed
metric: code revision
hardware_runtime: backend and precision
date: evaluation date
```

Нужно фиксировать prompt/template, model version, decoding, hardware, dataset
version, seeds и exact evaluation code.

## От score к решению

Хороший отчёт содержит:

- confidence intervals или повторные runs;
- результаты по slices, языкам и длинам;
- contamination analysis;
- стоимость и latency;
- примеры типичных ошибок;
- границы claim.

Если модель стала лучше на 0.3 пункта, но разброс между prompts равен 2 пунктам,
вывод о прогрессе не подтверждён.

- [[02 Areas/ML & DL/Concepts/Evaluation/MMLU]]
- [[02 Areas/ML & DL/Concepts/Evaluation/HumanEval]]
- [[02 Areas/ML & DL/Courses/Stanford CS224N/Lecture 11 — Benchmarking and Evaluation]]
- [Eugene Yan: LLM evals](https://eugeneyan.com/writing/)
