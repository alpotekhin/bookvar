---
title: Encoder-only
type: concept
status: canonical
last_updated: 2026-07-16
primary_sources:
  - https://arxiv.org/abs/1810.04805
---

# Encoder-only

**Encoder-only Transformer** превращает всю входную последовательность в контекстуализированные представления. Каждый токен может смотреть на токены слева и справа, поэтому паттерн особенно удобен для понимания уже данного текста.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/bert-input-output.png]]

*BERT складывает token, segment и position embeddings и возвращает
контекстуализированное состояние для каждой входной позиции. Источник: Jay
Alammar, [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

Выход encoder — не один «вектор текста», а последовательность состояний. Уже
голова задачи решает, использовать `[CLS]`, pooling, отдельные token states или
их комбинацию.

## Механика

Вход `[batch, seq, d_model]` проходит через стек одинаковых encoder-блоков. В каждом блоке:

1. self-attention без causal mask смешивает информацию между всеми доступными позициями;
2. position-wise FFN преобразует каждый токен;
3. residual connections и normalization стабилизируют глубокую сеть.

После последнего слоя используют представление специального токена, pooling либо все token embeddings.

## Обучение и применение

Классический пример — BERT с masked language modeling: модель восстанавливает скрытые токены, пользуясь обоими контекстами. Encoder-only модели применяют для классификации, извлечения сущностей, reranking и построения [[02 Areas/ML & DL/01 Справочник/Retrieval/Embeddings|эмбеддингов]].

## Ограничение

Двунаправленный attention не задаёт естественного авторегрессивного процесса генерации. Encoder можно приспособить к генерации, но decoder-only или [[02 Areas/ML & DL/01 Справочник/Архитектурные паттерны/Encoder-Decoder|Encoder–Decoder]] обычно подходят лучше.

## Не путать

`Encoder-only` — архитектурный паттерн, а `masked language modeling` — objective обучения. Они часто встречаются вместе, но логически не тождественны.

## Подробнее

Сопоставление encoder-only, decoder-only и encoder-decoder на одном примере
находится в главе [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна|Три архитектурных паттерна]].

## Источники

- [BERT](https://arxiv.org/abs/1810.04805)
