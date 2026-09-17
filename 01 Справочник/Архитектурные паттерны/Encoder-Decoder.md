---
title: Encoder–Decoder
aliases: [Encoder-Decoder, Seq2Seq Transformer]
type: concept
status: canonical
last_updated: 2026-07-16
primary_sources:
  - https://arxiv.org/abs/1706.03762
  - https://arxiv.org/abs/1910.10683
---

# Encoder–Decoder

**Encoder–Decoder** разделяет задачу на понимание входа и генерацию выхода. Encoder двунаправленно кодирует источник; decoder авторегрессивно создаёт результат и обращается к encoder через [[02 Areas/ML & DL/01 Справочник/Attention/Cross-Attention|cross-attention]].

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/t5/encoder-decoder-architecture.png]]

*T5 состоит из bidirectional encoder и causal decoder; encoder–decoder attention
передаёт память источника в каждый decoder block. Источник: Colin Raffel et al.,
[Exploring the Limits of Transfer Learning with a Unified Text-to-Text
Transformer, Figure 1](https://arxiv.org/abs/1910.10683).*

Два стека выполняют разные вычисления: encoder один раз строит представления
всего входа, decoder многократно обращается к ним при создании выхода. Поэтому
длина источника и длина результата независимы.

## Где полезен

Паттерн естественно соответствует задачам «одна последовательность → другая»: перевод, суммаризация, распознавание речи, исправление и преобразование текста. T5 унифицировал многие NLP-задачи как text-to-text.

## Trade-offs

Источник получает двунаправленные представления в отдельном encoder, а decoder
обращается к ним через cross-attention. Это полезное разделение ролей, но не
уникальная экономия повторного кодирования: decoder-only с
[[02 Areas/ML & DL/01 Справочник/Inference/KV-cache|KV-cache]] тоже выполняет
prefill префикса один раз и затем использует сохранённые ключи и значения.
Разница в том, что обычный causal decoder-only кодирует каждую позицию префикса
без доступа к её будущим соседям.

Два стека позволяют независимо выбирать глубину и ширину encoder и decoder,
но добавляют отдельную память источника и cross-attention. Сравнивать задержку
и расход памяти нужно при заданных размерах стеков, длинах входа/выхода и
реализации cache; сама топология не гарантирует преимущество. Такое разделение
архитектур рассматривается в [T5, §3.2](https://arxiv.org/abs/1910.10683).

## Не путать

- Seq2Seq существовал до Transformer на RNN/LSTM.
- Архитектура не определяет objective: возможны denoising, translation, supervised task training.
- Prefix-LM может имитировать двунаправленный источник в одном стеке, но это иной mask pattern.

## Подробнее

Полный путь данных через два стека, causal self-attention и cross-attention
разобран в главе [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/03 Полный Transformer|Полный Transformer]].

## Источники

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [T5](https://arxiv.org/abs/1910.10683)
