---
title: Прочитать современный decoder в LitGPT
type: practice
status: canonical
last_updated: 2026-08-06
---

# Прочитать современный decoder в LitGPT

В этой работе модель не нужно обучать. Задача — научиться восстанавливать
архитектуру не по маркетинговому названию, а по конфигурации, формам тензоров и
реальному `forward`.

## Материал и фиксируемая версия

Используйте [Lightning-AI/litgpt](https://github.com/Lightning-AI/litgpt) как
code atlas. Запишите commit, версию PyTorch и три выбранные конфигурации:
плотную Llama-подобную модель, Qwen и DeepSeek/Mixtral с MoE. Если текущий
LitGPT не поддерживает нужный релиз, выберите ближайшую реализованную
конфигурацию и явно зафиксируйте различие.

Перед чтением кода держите рядом:

- [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/02 Современный decoder block]];
- [[02 Areas/ML & DL/00 Учебник/08 Эффективный Attention и длинный контекст/01 MHA, MQA и GQA]];
- [[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/02 Mixture of Experts — routing, capacity и serving]].

## Шаг 1. Паспорт конфигурации

Для каждой модели заполните одну строку:

| Ось | Значение | Где доказано |
|---|---|---|
| `n_layer`, `n_embd` | | config field + tensor shape |
| query heads / KV groups | | config + QKV projection |
| norm и placement | | class + место вызова |
| RoPE и scaling | | config + функция применения |
| FFN / intermediate size | | module + weight shapes |
| MoE experts / top-k | | router + expert module |
| tied embeddings | | identity/storage check |

Колонка «Где доказано» обязательна: значение из model card без подтверждения в
реализации считается гипотезой.

## Шаг 2. Трасса одного токена

Запустите крошечный batch и зарегистрируйте hooks на embedding, первом block,
attention projections, FFN и LM head. Сохраните формы входов и выходов. Для
prefill используйте последовательность длины 8, затем выполните один decode
step с KV-cache.

Для обычного MHA/GQA-блока остановитесь после разделения Q/K/V и перестановки
осей, но до возможного повторения KV-голов. В этой точке ожидается layout
`[B, heads, T, head_dim]`: ось 1 содержит головы, ось 2 — позиции.
Проверьте инварианты:

```python
assert hidden.shape[-1] == config.n_embd
assert q.shape[1] == config.n_head
assert k.shape[1] == config.n_query_groups
assert logits.shape[-1] == config.padded_vocab_size
```

Возьмите `T=8`, а число query-голов — например, 4: разные значения помогут
заметить перепутанные оси. В [реализации LitGPT](https://github.com/Lightning-AI/litgpt/blob/main/litgpt/model.py)
нужная перестановка записана как `transpose(1, 2)`. Закрепите её точное место
в выбранном commit. Названия полей и layout до перестановки могут отличаться;
для MLA сначала отдельно установите, какие проекции соответствуют Q и K.

## Шаг 3. Архитектурный diff

Составьте diff Llama → Qwen → выбранная MoE-модель по шести независимым осям:
attention, position, normalization, FFN, routing и vocabulary/head. Отдельно
вынесите training recipe и inference features, которые нельзя вывести из
`forward`.

## Что сдать

- pinned commit и команды воспроизведения;
- три заполненных паспорта со ссылками на строки кода;
- трассу tensor shapes для prefill и одного decode step;
- схему одного блока, восстановленную из кода;
- архитектурный diff и список утверждений, которые код не позволяет доказать.

Работа завершена, если другой человек может по вашему паспорту открыть тот же
commit, найти каждое поле и воспроизвести формы без чтения model card.
