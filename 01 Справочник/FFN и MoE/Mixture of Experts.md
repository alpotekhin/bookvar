---
title: Mixture of Experts
aliases: [MoE, Sparse MoE]
type: concept
status: canonical
last_updated: 2026-07-16
primary_sources:
  - https://arxiv.org/abs/1701.06538
  - https://arxiv.org/abs/2101.03961
---

# Mixture of Experts (MoE)

**Sparse MoE** заменяет dense FFN набором экспертов и router, который для каждого токена выбирает небольшое число экспертов.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/mixtral-of-experts/smoe-layer.png]]

*Sparse MoE-слой Mixtral: router выбирает для каждого token representation два
из восьми FFN, а их outputs складываются с весами router. Источник: Albert Q.
Jiang et al., [Mixtral of Experts, Figure 1](https://arxiv.org/abs/2401.04088).*

На рисунке единицей маршрутизации является токен, а не последовательность:
соседние токены одного запроса могут попасть к разным экспертам. Все эксперты
параметризованы отдельно, но для одного токена вычисляются только выбранные.

$$y=\sum_{i\in TopK(g(x))}p_i(x)\,E_i(x).$$

Это разделяет **total parameters** и **active parameters**: модель может иметь большой capacity, но вычислять только top-k экспертов на токен.

## Главные сложности

- баланс нагрузки: без auxiliary loss часть экспертов перегружается, другие не учатся;
- capacity и dropped/rerouted tokens;
- all-to-all communication между accelerators;
- routing instability и сложность serving;
- память всё равно должна хранить параметры всех экспертов.

Shared experts (например, в DeepSeekMoE) всегда активны и захватывают общие знания; routed experts специализируются. Fine-grained expert segmentation меняет гранулярность маршрутизации.

MoE — архитектурный механизм FFN, а не ансамбль независимо обученных моделей и не синоним «нескольких агентов».

## Подробнее

Routing, capacity factor, балансировка, all-to-all и требования к serving
разобраны в главе [[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/02 Mixture of Experts — routing, capacity и serving|Mixture of Experts — routing, capacity и serving]].

## Источники

- [Outrageously Large Neural Networks](https://arxiv.org/abs/1701.06538)
- [Switch Transformers](https://arxiv.org/abs/2101.03961)
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066)
