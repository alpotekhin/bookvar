---
title: "LMCache: карта материалов"
type: source-note
status: active
last_verified: 2026-07-23
---

# LMCache: карта материалов

Подробные заметки сохранены на английском, чтобы названия структур, путей
передачи и интерфейсов совпадали с документацией и кодом LMCache.

1. [[LMCache — an external KV cache layer|LMCache as an external KV cache layer]]
2. [[LMCache MP mode — transfer paths|MP mode: CUDA IPC, shared memory and pickle]]
3. [[KV cache as persistent inference state|KV cache as persistent inference state]]
4. [[CacheBlend — non-prefix KV reuse|CacheBlend and non-prefix KV reuse]]

Связанные главы учебника: [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention|KV-cache и PagedAttention]],
[[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55b Scheduling — continuous batching, chunked prefill и prefix caching|prefix caching]] и
[[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/58a2 Раздельное обслуживание prefill и decode|prefill/decode disaggregation]].
