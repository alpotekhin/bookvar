---
title: KV-cache, batching и FlashAttention
type: textbook-chapter
status: canonical
last_updated: 2026-07-16
---

# KV-cache, batching и FlashAttention

> [!abstract] Идея главы
> Обработка prompt и генерация следующего токена нагружают hardware по-разному.
> Prefill хорошо использует параллельные matmul, а decode часто ждёт загрузки
> weights и KV-cache из памяти.

Inference делится на:

- **prefill** — параллельная обработка prompt, compute-bound;
- **decode** — один новый токен за шаг, часто memory-bandwidth-bound.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/kv-cache-prefill-decode.svg]]

*KV-cache сохраняет keys и values старых токенов; для нового шага вычисляются
только K/V нового токена.*

Continuous batching динамически объединяет запросы на разных decode шагах.
PagedAttention управляет KV-cache блоками, уменьшая fragmentation.

![[02 Areas/ML & DL/raw/papers/kv-cache/images/paged-attention.png]]

FlashAttention не аппроксимирует attention. Он вычисляет exact result плитками,
реже перемещая данные между HBM и on-chip SRAM. Это особенно важно на prefill и
training.

Speculative decoding использует draft model или extra heads, затем target model
проверяет несколько предложенных токенов параллельно.

## Метрики

- TTFT — time to first token;
- TPOT — time per output token;
- throughput tokens/s;
- p50/p95 latency;
- memory per request;
- стоимость при заданном SLO.

Оптимизация одной метрики может ухудшить другую: большой batch повышает
throughput, но увеличивает latency отдельного запроса.

- [[02 Areas/ML & DL/Concepts/Inference/KV-Cache]]
- [[02 Areas/ML & DL/Concepts/Inference/Flash Attention]]
- [[02 Areas/ML & DL/Concepts/Inference/Speculative Decoding]]
- [[02 Areas/ML & DL/Courses/Stanford CS336/CS336 — Inference]]
