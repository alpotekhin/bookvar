---
title: "Falcon"
type: model-family
organization: Technology Innovation Institute
first_release: 2023
latest_verified_release: Falcon-H1 Arabic and Falcon Perception
last_verified: 2026-07-20
architecture_base: decoder Transformer then parallel Mamba-attention hybrid
modalities: [text, image]
status: active
---

# Falcon

Falcon прошёл две архитектурные эпохи. Falcon 7B/40B/180B и Falcon 3 — decoder-only Transformer; Falcon-H1 отказался от последовательного чередования типов слоёв и запускает attention и Mamba-2 параллельно внутри одного mixer-блока. Поэтому «Falcon» больше не означает одну неизменную архитектуру.

## Линия релизов

| Релиз | Существенный diff |
|---|---|
| Falcon 7B/40B (2023) | Multi-query attention, parallel attention/MLP; RefinedWeb как основной вклад в данные. |
| Falcon 180B (2023) | Масштабирование dense линии на 3.5T RefinedWeb tokens. |
| Falcon 2 / Falcon 3 (2024) | Малые эффективные модели 1B–10B; Falcon 3 получил 14T токенов и 32K. |
| Falcon-H1 (2025) | Параллельный hybrid-head: Mamba-2 SSM и attention в каждом mixer. |
| Falcon-H1 Arabic (05.01.2026) | Арабская специализированная линия на H1. |
| Falcon Perception (31.03.2026) | 600M vision-language модель; отдельная мультимодальная ветка. |

## Почему Falcon-H1 не Jamba

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/atlas-remainder-official/falcon-h1-architecture.png]]

*Рисунок: Zuo et al., [Falcon-H1 Technical Report](https://arxiv.org/abs/2507.22448), Figure 1; локальная копия. Attention и SSM получают разные каналы одного входа одновременно, их выходы конкатенируются и проецируются; затем следует MLP.*

Jamba чередует Mamba- и attention-слои. Falcon-H1 делит ширину mixer между двумя механизмами внутри каждого блока. Число SSM- и attention-heads можно менять независимо, поэтому модель выбирает долю рекуррентной памяти и прямого retrieval не только числом слоёв, но и каналами. RMSNorm стоит перед mixer и MLP, residual paths проходят вокруг обоих подслоёв.

Ранняя Falcon-линия использовала MQA: все query-heads разделяют K/V, резко уменьшая KV-cache. Parallel block вычислял attention и MLP от одного нормализованного входа. Это нельзя автоматически переносить на H1, где основной diff — Mamba-2 + attention fusion.

## Данные и post-training

RefinedWeb — отдельный открытый вклад TII: Common Crawl проходит deduplication и строгую фильтрацию вместо добавления большого числа курируемых источников. Falcon-H1 report описывает новую 18T-token multilingual mixture и размеры от 0.5B до 34B. Instruct-версии получают SFT и preference alignment; Arabic — дополнительную языковую специализацию. Falcon Perception обучает vision-language систему и не является продолжением текстовых весов только по имени.

## Serving

Falcon-H1 сохраняет постоянное Mamba-состояние, но attention-ветвь всё равно требует KV-cache. Экономия зависит от доли attention-каналов и реализации fused mixer. Параллельная композиция также даёт возможность component-aware self-speculation, но это результат отдельной работы 2026 года, а не гарантированное свойство любого сервера.

## Что опубликовано

Для Falcon 1/RefinedWeb и H1 доступны papers, веса и configs. Falcon Perception официально анонсирован, но полная recipe может быть раскрыта слабее. Маркетинговые сравнения TII не заменяют независимые evaluation. Falcon-H1 Arabic — последний verified текстовый релиз; Perception — последняя новая ветка семейства.

## Источники

- [Falcon report](https://arxiv.org/abs/2311.16867) и [RefinedWeb](https://arxiv.org/abs/2306.01116).
- [Falcon-H1 Technical Report](https://arxiv.org/abs/2507.22448).
- [TII: Falcon-H1 Arabic](https://www.tii.ae/index.php/news/abu-dhabis-tii-launches-falcon-h1-arabic-establishing-worlds-leading-arabic-ai-model).
- [TII: Falcon Perception](https://www.tii.ae/news/tii-launches-falcon-perception-new-multimodal-ai-model-helps-machines-see-and-understand-world).
