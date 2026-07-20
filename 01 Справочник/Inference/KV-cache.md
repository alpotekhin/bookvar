---
title: KV-cache
type: concept
status: canonical
last_updated: 2026-07-16
primary_sources:
  - https://arxiv.org/abs/1911.02150
---

# KV-cache

При autoregressive generation новые queries должны взаимодействовать со всеми прошлыми keys и values. **KV-cache** сохраняет K/V предыдущих токенов, чтобы не пересчитывать весь префикс на каждом decode step.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/kv-cache/kv-cache-mechanism.png]]

*При каждом шаге декодирования новые Q сравниваются с сохранёнными K/V
префикса; в cache дописывается только пара текущего токена. Источник: Sebastian
Raschka, [Understanding and Coding the KV Cache in LLMs from
Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms).*

Рисунок отделяет две операции, которые часто смешивают: прошлые hidden states не
прогоняются через модель заново, но их K/V читаются на каждом шаге attention.
Поэтому cache экономит вычисления и одновременно создаёт растущую нагрузку на
память и её пропускную способность.

Для обычного attention объём cache масштабируется приблизительно как:

$$2\cdot L\cdot T\cdot h_{kv}\cdot d_{head}\cdot bytes,$$

где `2` — K и V, `L` — слои, `T` — сохранённые токены. Нужно дополнительно умножить на batch/число sequences.

## Prefill и decode

- **Prefill:** prompt обрабатывается параллельно; обычно compute-intensive.
- **Decode:** добавляется по одному токену; чтение большого cache часто memory-bandwidth-bound.

Именно поэтому [[02 Areas/ML & DL/01 Справочник/Attention/MQA|MQA]], [[02 Areas/ML & DL/01 Справочник/Attention/GQA|GQA]] и [[02 Areas/ML & DL/01 Справочник/Attention/MLA|MLA]] важны для serving.

Paged KV-cache управляет cache блоками и уменьшает fragmentation; prefix caching повторно использует общий prompt; quantized KV снижает память ценой возможной ошибки. Это inference techniques, не изменение уже обученных attention weights.

## Подробнее

Расчёт памяти, различие prefill/decode, batching и PagedAttention собраны в
главе [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention|KV-cache, пакетирование и PagedAttention]].

## Источники

- [Fast Transformer Decoding / MQA](https://arxiv.org/abs/1911.02150)
- [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180)
