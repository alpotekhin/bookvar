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

*Снимок визуализации self-attention: выбранная выходная позиция `it`
получает вклад от всех входных позиций, а толщина линий соответствует attention
weights. Источник: Jay Alammar,
[The Illustrated Transformer — Self-Attention Visualization](https://jalammar.github.io/illustrated-transformer/),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

Визуализация показывает уже вычисленную строку матрицы $A$. Формулы выше
объясняют, откуда она берётся: query выбранной позиции сравнивается со всеми
keys, softmax нормирует оценки, после чего этими весами смешиваются values.

Матрица $A$ имеет форму `[n_query, n_key]`: каждая строка — распределение внимания одной позиции по доступным ключам. Деление на $\sqrt{d_k}$ удерживает logits в масштабе, при котором softmax не насыщается слишком быстро.

Маска $M$ определяет доступ к позициям. При стандартном сдвиге целевой
последовательности входной токен $x_i$ предсказывает $x_{i+1}$, поэтому
каузальная маска разрешает $j\le i$, включая диагональ:
$M_{ij}=0$ при $j\le i$ и $M_{ij}=-\infty$ при $j>i$.
Например, третья входная позиция видит позиции 1, 2 и 3, но не правильный
четвёртый токен, который нужно предсказать. Маска заполнения скрывает
неиспользуемые позиции, а локальная или разреженная маска дополнительно
ограничивает разрешённые связи.

Self-attention не является «объяснением решения» модели: attention weights показывают маршрут смешивания values в конкретной голове, но не исчерпывают причинность всей сети.

## Варианты

[[02 Areas/ML & DL/01 Справочник/Attention/MHA|MHA]], [[02 Areas/ML & DL/01 Справочник/Attention/MQA|MQA]], [[02 Areas/ML & DL/01 Справочник/Attention/GQA|GQA]] и [[02 Areas/ML & DL/01 Справочник/Attention/MLA|MLA]] по-разному организуют головы и сохранение K/V.

## Подробнее

Численный пример вычисления $QK^\top$, softmax и взвешенной суммы находится в
главе [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V|Self-Attention — Q, K, V]].

## Источники

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
