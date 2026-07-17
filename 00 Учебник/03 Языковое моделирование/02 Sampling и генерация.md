---
title: Sampling и генерация
type: textbook-chapter
status: canonical
last_updated: 2026-07-16
---

# Sampling и генерация

> [!abstract] Идея главы
> Модель выдаёт распределение вероятностей, а не готовое решение. Decoding
> определяет, как превратить это распределение в конкретную последовательность.

## Greedy decoding

На каждом шаге выбирается token с максимальной вероятностью:

$$
x_t=\arg\max_x P(x\mid x_{<t}).
$$

Метод детерминирован и удобен для проверки, но локально лучший token не обязан
вести к лучшей полной sequence. В открытой генерации greedy часто повторяется и
делает текст однообразным.

## Temperature

Перед softmax logits делятся на $T$:

$$
P_i=\operatorname{softmax}(z_i/T).
$$

- $T<1$ делает распределение резче;
- $T>1$ выравнивает варианты;
- при $T\to0$ выбор приближается к greedy.

Temperature не добавляет знания и не «делает модель умнее». Она меняет
случайность выбора из уже рассчитанного distribution.

## Top-k

Оставляются только $k$ наиболее вероятных tokens, остальные получают нулевую
вероятность. Фиксированное $k$ просто, но не учитывает форму distribution:
иногда разумных вариантов два, иногда пятьдесят.

## Top-p

Nucleus sampling оставляет минимальный набор tokens, чья суммарная probability
не меньше $p$. Размер набора адаптируется к уверенности модели.

Обычно temperature применяют к logits, затем фильтруют top-k/top-p и только
после этого sample.

## Beam search

Beam search хранит несколько лучших незавершённых sequences. Он полезен, когда:

- output должен быть близок к одному каноническому ответу;
- важна полная sequence probability;
- задача похожа на перевод или распознавание речи.

Для открытого диалога beam search часто предпочитает безопасные, повторяющиеся
фразы. Также нужен length penalty: произведение вероятностей склонно выбирать
короткие sequences.

## Почему один seed даёт другой текст

Sampling использует псевдослучайный генератор. Для воспроизводимости нужны:

- фиксированный seed;
- точная model version;
- одинаковый tokenizer и chat template;
- одинаковые kernels и precision;
- одинаковые decoding parameters.

На GPU полная битовая воспроизводимость не всегда гарантируется.

## Несколько траекторий

Для math/code/reasoning часто полезно сгенерировать несколько candidates:

1. independently sample решения;
2. проверить тестами, verifier или reward model;
3. выбрать лучшее либо majority answer.

Self-consistency улучшает шанс получить правильную траекторию, но линейно
увеличивает inference compute.

## Когда использовать что

| Задача | Разумная отправная точка |
|---|---|
| классификация через генерацию | greedy / низкая temperature |
| код с тестами | несколько samples + tests |
| творческий текст | temperature + top-p |
| перевод | beam search |
| строгий JSON | constrained decoding |
| исследование вероятностей | без sampling, смотреть logits |

## Минимальный sampling loop

```python
for _ in range(max_new_tokens):
    logits = model(tokens)[:, -1]
    logits = logits / temperature
    logits = top_p_filter(logits, top_p)
    probs = logits.softmax(dim=-1)
    next_token = torch.multinomial(probs, 1)
    tokens = torch.cat([tokens, next_token], dim=1)
```

Production generation добавляет KV-cache, batching, stop sequences,
repetition penalties и ограничения grammar.

## Источники

- [Holtzman et al. — The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
- [[02 Areas/ML & DL/Concepts/Inference/Sampling]]
- [[02 Areas/ML & DL/Concepts/Inference/Speculative Decoding]]
- [Hugging Face — Generation strategies](https://huggingface.co/docs/transformers/generation_strategies)

