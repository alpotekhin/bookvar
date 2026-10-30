---
title: PagedAttention — управление KV-cache блоками
type: concept
status: redirect
last_updated: '2026-10-29'
last_verified: 2026-07-21T00:00:00.000Z
canonical_topic: >-
  00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и
  PagedAttention.md
canonical_heading: 'PagedAttention: логическая последовательность, физические блоки'
---
# PagedAttention — управление KV-cache блоками

Полное объяснение: [[00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention#PagedAttention: логическая последовательность, физические блоки|KV-cache, пакетирование и PagedAttention]].

<!-- Compatibility anchors for existing Obsidian and web links. -->
<span id="pagedattention--управление-kv-cache-блоками" aria-hidden="true"></span>
<span id="сколько-памяти-занимает-kv-cache" aria-hidden="true"></span>
<span id="почему-непрерывное-размещение-теряет-память" aria-hidden="true"></span>
<span id="логические-и-физические-блоки" aria-hidden="true"></span>
<span id="как-attention-читает-разбросанный-cache" aria-hidden="true"></span>
<span id="sharing-и-copy-on-write" aria-hidden="true"></span>
<span id="связь-с-continuous-batching" aria-hidden="true"></span>
<span id="чего-pagedattention-не-делает" aria-hidden="true"></span>
<span id="что-смотреть-в-реализации" aria-hidden="true"></span>
<span id="связанные-страницы" aria-hidden="true"></span>
<span id="источники" aria-hidden="true"></span>
