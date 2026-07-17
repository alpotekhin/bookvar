---
title: Прочитать современный decoder в LitGPT
type: practice
status: canonical
last_updated: 2026-07-16
---

# Прочитать современный decoder в LitGPT

Используйте [LitGPT](https://github.com/Lightning-AI/litgpt) как code atlas.

Найдите в config и model code:

1. `n_head` и `n_query_groups`;
2. norm class и placement;
3. RoPE parameters;
4. FFN type/intermediate size;
5. MoE router и число active experts;
6. tied/untied embeddings.

Составьте diff для Llama, Qwen и DeepSeek. Не делайте вывод по имени config:
сверьте tensor shapes и forward path.

