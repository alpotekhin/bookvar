---
title: Cross-Attention
type: concept
status: canonical
last_updated: 2026-07-16
primary_sources:
  - https://arxiv.org/abs/1706.03762
---

# Cross-Attention

В **cross-attention** запросы $Q$ строятся из состояний одной последовательности,
а ключи $K$ и значения $V$ — из состояний другой:

$$Q=YW_Q,\qquad K=XW_K,\qquad V=XW_V.$$

Здесь $Y\in\mathbb{R}^{n_q\times d_y}$ содержит состояния позиций, которым
нужна информация, а $X\in\mathbb{R}^{n_m\times d_x}$ — состояния памяти
источника. Обучаемые матрицы $W_Q,W_K$ переводят их в общее пространство
размерности $d_k$, а $W_V$ — в пространство значений размерности $d_v$.
Поэтому $QK^\top$ имеет форму $n_q\times n_m$: каждой позиции запроса
соответствует свой набор весов по позициям памяти.

Например, decoder спрашивает: «какая часть входного предложения нужна для следующего слова?», а encoder предоставляет память.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/t5/encoder-decoder-architecture.png]]

*В T5 cross-attention находится в decoder: запросы строятся из состояний
известной части выхода, а ключи и значения — из памяти encoder. Схема из Colin Raffel et al.,
[Exploring the Limits of Transfer Learning with a Unified Text-to-Text
Transformer, Figure 1](https://arxiv.org/abs/1910.10683).*

Рисунок важно читать по направлению информационного потока. Encoder обрабатывает
источник один раз; каждый decoder-блок сначала смешивает уже известную часть
выхода causal self-attention, а затем обращается к неизменной памяти источника.

Cross-attention связывает и другие представления: текст с изображением, decoder
с audio encoder, latent queries с vision features. В отличие от
[[02 Areas/ML & DL/01 Справочник/Attention/Self-Attention|self-attention]],
длины последовательности запросов и памяти могут различаться, поэтому матрица
оценок имеет форму `[n_query, n_memory]`.

Cross-attention — механизм, а не самостоятельный архитектурный класс. Он является центральной частью [[02 Areas/ML & DL/01 Справочник/Архитектурные паттерны/Encoder-Decoder|Encoder–Decoder]], но может добавляться и в decoder-only backbone.

## Подробнее

Место cross-attention в полном вычислительном графе показано в главе
[[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/03 Полный Transformer|Полный Transformer]].

## Источники

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
