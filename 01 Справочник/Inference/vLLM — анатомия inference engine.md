---
title: vLLM — анатомия inference engine
type: concept
status: redirect
last_updated: '2026-10-29'
last_verified: 2026-07-21T00:00:00.000Z
canonical_topic: 05 Источники/Implementation studies/Inside vLLM.md
---
# vLLM — анатомия inference engine

Полное объяснение: [[05 Источники/Implementation studies/Inside vLLM|vLLM — анатомия inference engine]].

<!-- Compatibility anchors for existing Obsidian and web links. -->
<span id="vllm--анатомия-inference-engine" aria-hidden="true"></span>
<span id="1-от-вызова-generate-к-engine-core" aria-hidden="true"></span>
<span id="2-engine-loop" aria-hidden="true"></span>
<span id="3-prefill-и-decode-требуют-разного-расписания" aria-hidden="true"></span>
<span id="4-continuous-batching-и-paged-kv-cache-работают-вместе" aria-hidden="true"></span>
<span id="один-цикл-на-конкретных-запросах" aria-hidden="true"></span>
<span id="5-prefix-caching" aria-hidden="true"></span>
<span id="6-расширения-базового-цикла" aria-hidden="true"></span>
<span id="7-от-одного-worker-к-нескольким-gpu" aria-hidden="true"></span>
<span id="8-serving-layer" aria-hidden="true"></span>
<span id="9-как-измерять-результат" aria-hidden="true"></span>
<span id="что-читать-дальше" aria-hidden="true"></span>
<span id="источники" aria-hidden="true"></span>
