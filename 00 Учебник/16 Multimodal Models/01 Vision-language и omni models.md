---
title: Vision-language и omni models
type: textbook-chapter
status: canonical
last_updated: 2026-07-16
---

# Vision-language и omni models

> [!abstract] Идея главы
> Text LLM получает tokens из словаря. Multimodal model должна превратить image,
> audio или video в последовательность representations, которую языковой
> backbone сможет связать с текстом и, иногда, преобразовать обратно в другую
> modality.

VLM должна согласовать разные пространства: pixels/patches, audio frames и text
tokens.

Ранние systems часто замораживали LLM и vision encoder, обучая projector.
Native multimodal models используют early fusion и jointly trained backbone.

## LLaVA-подобная схема

Vision Transformer создаёт patch features, projector переводит их в dimension
LLM, после чего image tokens вставляются рядом с text tokens. Это дёшево и
понятно, но spatial detail ограничен resolution encoder и числом visual tokens.

## Cross-attention и resampler

Вместо помещения всех patches в language context отдельный resampler создаёт
меньшее число visual tokens. Cross-attention позволяет text states читать
visual memory, не смешивая modalities полностью.

## Native multimodality

Современные omni models совместно обучают несколько encoders/decoders и могут
принимать video/audio, а выводить text, speech или actions. Но слово native не
гарантирует точный grounding: нужно проверять OCR, coordinates, temporal order и
cross-modal consistency отдельно.

Ключевые вопросы:

- сколько visual tokens создаётся;
- сохраняется ли spatial resolution;
- есть ли cross-attention или единый token stream;
- обучается ли encoder совместно;
- как оцениваются OCR, grounding, charts, video и audio.

Высокий общий benchmark не гарантирует точное визуальное grounding.

## Цена visual tokens

Больше patches сохраняют мелкие детали, но удлиняют sequence и увеличивают
attention cost. Dynamic tiling, pooling и token pruning управляют этим
компромиссом.

- [[01 Projects/X5/LLaVA/LLaVA]]
- [[02 Areas/ML & DL/Papers/Gemini]]
- [Llama 4 official overview](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- [Qwen repositories](https://github.com/QwenLM)
