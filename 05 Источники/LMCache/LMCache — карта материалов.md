---
title: "LMCache: architecture, storage, transfer, and operations"
type: source-note
status: active
last_verified: 2026-07-23
---

# LMCache: architecture, storage, transfer, and operations

Полная глава опубликована на английском: так названия структур, протоколов,
режимов передачи и конфигурационных параметров совпадают с документацией и
кодом LMCache. Переключите язык страницы на English.

Это единая последовательная глава, а не четыре короткие справочные статьи. В
ней разобраны:

- MP-архитектура и полный путь L0 → L1 → L2;
- идентичность KV-объекта, chunking, lookup, eviction и prefetch;
- локальная передача через CUDA IPC, shared memory и engine-driven path;
- P2P, NIXL и prefill/decode disaggregation;
- Controller, Coordinator, cache-aware routing и quotas;
- CacheBlend, CacheGen, FP8, TurboQuant, serde и KV Cache SDK;
- storage backends, multimodal и hybrid-attention support;
- observability, Kubernetes Operator, multi-tenancy и границы безопасности;
- таблицы зрелости технологий и выбора production-конфигурации.

Связанные главы учебника: [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention|KV-cache и PagedAttention]],
[[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55b Scheduling — continuous batching, chunked prefill и prefix caching|prefix caching]] и
[[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/58a2 Раздельное обслуживание prefill и decode|prefill/decode disaggregation]].
