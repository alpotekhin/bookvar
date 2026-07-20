---
title: Multi-head Latent Attention
aliases: [MLA]
type: concept
status: canonical
last_updated: 2026-07-16
primary_sources:
  - https://arxiv.org/abs/2405.04434
---

# Multi-head Latent Attention (MLA)

**MLA** сжимает информацию, необходимую для keys и values, в низкоразмерный latent-вектор, который можно сохранить вместо полного набора K/V. Механизм представлен в DeepSeek-V2.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-first-35-36/cs336-mha-gqa-mqa-mla.png]]

*В трёх первых вариантах cache хранит готовые K/V с разным числом heads; в MLA
он хранит compressed latent KV, из которого определяются head-specific K/V.
Источник: Stanford CS336,
[Lecture 10: Inference](https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_10.py),
рисунок `mla-schema.png`, адаптирующий Figure 3 отчёта
[DeepSeek-V2](https://arxiv.org/abs/2405.04434).*

Поэтому MLA нельзя понимать как крайний случай GQA. GQA сокращает число
сохраняемых готовых heads; MLA меняет параметризацию памяти и переносит часть
работы из хранения в проекции.

В упрощённом виде:

$$c_t^{KV}=W^{DKV}h_t,\qquad k_t=W^{UK}c_t^{KV},\qquad v_t=W^{UV}c_t^{KV}.$$

На практике детали сложнее: MLA совместно проектирует KV, отдельно обрабатывает RoPE-зависимую часть key/query и использует матричное поглощение проекций при inference. Главная идея остаётся: cache хранит компактное представление.

## Отличие от GQA

- GQA уменьшает **число KV-heads**.
- MLA уменьшает **размер сохраняемого состояния через low-rank latent**.

Оба подхода уменьшают KV-cache, но их параметризация и inference kernels различаются. MLA нельзя описывать просто как «ещё меньше KV-heads».

## Trade-offs

Преимущество — сильное уменьшение cache при сохранении богатых head-specific представлений после реконструкции. Цена — более сложная реализация, kernel optimization и взаимодействие с positional encoding.

## Подробнее

Low-rank сжатие, decoupled RoPE и weight absorption разобраны в главе
[[02 Areas/ML & DL/00 Учебник/08 Эффективный Attention и длинный контекст/02 MLA и сжатие KV-cache|MLA и сжатие KV-cache]].

## Источники

- [DeepSeek-V2](https://arxiv.org/abs/2405.04434)
