---
title: Self-Attention
type: concept
status: canonical
last_updated: 2026-07-16
primary_sources:
  - https://arxiv.org/abs/1706.03762
---

# Self-Attention

**Self-attention** позволяет каждому токену собрать контекст из той же последовательности. Из входа $X\in\mathbb{R}^{n\times d}$ строятся:

$$Q=XW_Q,\quad K=XW_K,\quad V=XW_V,$$
$$A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right),\quad O=AV.$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/attention-is-all-you-need/transformer_self-attention_visualization.png]]

*Интерактивная визуализация self-attention: выбранная выходная позиция `it`
получает вклад от всех входных позиций, а толщина линий соответствует attention
weights. Источник: Jay Alammar,
[The Illustrated Transformer — Self-Attention Visualization](https://jalammar.github.io/illustrated-transformer/),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

Визуализация показывает уже вычисленную строку матрицы $A$. Формулы выше
объясняют, откуда она берётся: query выбранной позиции сравнивается со всеми
keys, softmax нормирует оценки, после чего этими весами смешиваются values.

Матрица $A$ имеет форму `[n_query, n_key]`: каждая строка — распределение внимания одной позиции по доступным ключам. Деление на $\sqrt{d_k}$ удерживает logits в масштабе, при котором softmax не насыщается слишком быстро.

Mask определяет паттерн доступа: causal mask разрешает только прошлое; padding mask скрывает пустые позиции; local/sparse mask ограничивает область связи.

Self-attention не является «объяснением решения» модели: attention weights показывают маршрут смешивания values в конкретной голове, но не исчерпывают причинность всей сети.

## Варианты

[[02 Areas/ML & DL/01 Справочник/Attention/MHA|MHA]], [[02 Areas/ML & DL/01 Справочник/Attention/MQA|MQA]], [[02 Areas/ML & DL/01 Справочник/Attention/GQA|GQA]] и [[02 Areas/ML & DL/01 Справочник/Attention/MLA|MLA]] по-разному организуют головы и сохранение K/V.

## Подробнее

Численный пример вычисления $QK^\top$, softmax и взвешенной суммы находится в
главе [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V|Self-Attention — Q, K, V]].

## Источники

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
