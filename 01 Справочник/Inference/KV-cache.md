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

```mermaid
sequenceDiagram
  participant C as Cache K,V
  participant M as Model
  M->>C: prefill: K,V всего prompt
  loop каждый новый токен
    M->>C: прочитать прошлые K,V
    M->>C: дописать Kₜ,Vₜ
  end
```

Для обычного attention объём cache масштабируется приблизительно как:

$$2\cdot L\cdot T\cdot h_{kv}\cdot d_{head}\cdot bytes,$$

где `2` — K и V, `L` — слои, `T` — сохранённые токены. Нужно дополнительно умножить на batch/число sequences.

## Prefill и decode

- **Prefill:** prompt обрабатывается параллельно; обычно compute-intensive.
- **Decode:** добавляется по одному токену; чтение большого cache часто memory-bandwidth-bound.

Именно поэтому [[02 Areas/ML & DL/01 Справочник/Attention/MQA|MQA]], [[02 Areas/ML & DL/01 Справочник/Attention/GQA|GQA]] и [[02 Areas/ML & DL/01 Справочник/Attention/MLA|MLA]] важны для serving.

Paged KV-cache управляет cache блоками и уменьшает fragmentation; prefix caching повторно использует общий prompt; quantized KV снижает память ценой возможной ошибки. Это inference techniques, не изменение уже обученных attention weights.

## Источники

- [Fast Transformer Decoding / MQA](https://arxiv.org/abs/1911.02150)
- [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180)
- [[02 Areas/ML & DL/Concepts/Inference/KV-Cache|Legacy: KV-cache]]
