---
title: "Yi"
type: model-family
organization: 01.AI
first_release: 2023
latest_verified_release: Yi-1.5 and Yi-Coder open line
last_verified: 2026-07-20
architecture_base: LLaMA-like decoder-only Transformer
modalities: [text, image]
status: stable-open-line
---

# Yi

Yi полезна как пример семейства, где авторы сознательно сохранили обычный dense LLaMA-подобный блок и вложили основную работу в данные, длинный контекст и инфраструктуру. Если качество изменилось без нового attention, это не «отсутствие архитектуры», а свидетельство того, насколько сильна training recipe.

## Релизы

| Релиз | Контекст и назначение |
|---|---|
| Yi-6B/34B (2023) | 4K pre-training, двуязычные base/chat. |
| Yi-6B/34B-200K (2023–2024) | Продолжение обучения на 5B long-context токенов; отдельные checkpoints. |
| Yi-VL 6B/34B (2024) | Vision encoder + projector + LLM; мультимодальная ветка. |
| Yi-1.5 6B/9B/34B (2024) | Обновлённые corpus и post-training; 16K/32K варианты. |
| Yi-Coder 1.5B/9B (2024) | Кодовая специализация. |

Открытый GitHub release log заканчивается Yi-1.5/Coder; Yi-Lightning представлен как закрытый API-продукт без полного report. Поэтому его нельзя объявлять новым открытым архитектурным поколением.

## Архитектура и данные

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/atlas-remainder-official/yi-data-pipeline.png]]

*Рисунок: 01.AI, [Yi: Open Foundation Models](https://arxiv.org/abs/2403.04652), Figure 1; локальная копия. После language filtering документы проходят метрики текста, удаление повторов, perplexity/quality filtering, многоуровневую дедупликацию и semantic/topic/safety filters.*

Базовый Yi — dense decoder-only Transformer с RMSNorm, SwiGLU, RoPE и GQA в крупных конфигурациях. Он близок к Llama, что упрощает применение существующих kernels. Авторы сообщают 3T multilingual токенов и подчёркивают качество bilingual смеси. Инфраструктура включает topology-aware scheduling, автоматическое восстановление и разные distributed backends.

Для 200K исходную модель продолжают обучать на длинных документах, а RoPE масштабируют. Это отдельная стадия, поэтому обычный 4K checkpoint и 200K нельзя считать взаимозаменяемыми. Yi-VL добавляет vision tower и обучаемый connector; свойства текста и изображения следует проверять отдельно.

## Post-training и inference

Chat-варианты проходят SFT и preference alignment. Report упоминает отдельные backends для policy и reward model. Для serving авторы использовали 4-bit weights, 8-bit KV-cache, PagedAttention и dynamic batching. Это эксплуатационные решения, а не свойства весов; качество конкретной quantization зависит от checkpoint.

## Что неизвестно

Опубликованы архитектура, общие категории данных, веса и код открытой линии. Полный список документов и все post-training пары не раскрыты. Архитектура Yi-Lightning не опубликована достаточно, чтобы переносить на неё параметры Yi-1.5. На дату проверки новых открытых base-релизов после 2024 в официальной коллекции нет.

## Источники

- [Yi Technical Report](https://arxiv.org/abs/2403.04652).
- [01-ai/Yi](https://github.com/01-ai/Yi) — official release log и deployment recipes.
- [01-ai model collection](https://huggingface.co/01-ai) — verified checkpoints.
