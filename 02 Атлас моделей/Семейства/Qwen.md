---
title: "Qwen"
type: model-family
organization: Alibaba Qwen Team
first_release: 2023-08
latest_verified_release: Qwen3.6
last_verified: 2026-07-20
architecture_base: decoder-only Transformer; later sparse MoE and hybrid linear attention
modalities: [text, image, audio]
status: active
---

# Qwen

## Место в истории

Qwen превратился из китайско-английской LLaMA-подобной модели в широкую открытую платформу: general, coder, math, vision-language, audio и embedding ветви разделяют tokenizer и tooling не всегда одинаково. Qwen2 сделал GQA и 128K типовой основой; Qwen2.5 вложил основной прогресс в данные и post-training; Qwen3 объединил dense/MoE и thinking/non-thinking в одном checkpoint. Qwen3-Next/3.5 изменили уже backbone: большая часть слоёв использует Gated Delta Networks, а attention остаётся периодическим якорем; Qwen3.6 развивает agentic coding поверх этой линии.

## Неизменное ядро

До Qwen3 это causal pre-norm decoder с RoPE, SwiGLU и tied/untied embeddings в зависимости от размера. Специализированные ветви не следует считать новым общим поколением: Qwen2-VL имеет vision encoder и динамическое разрешение, но текстовый Qwen2 от этого не становится multimodal. Общая линия — улучшение multilingual/code/math данных, chat templates и tool calling.

## Таблица релизов

| Релиз | Architecture diff | Tokenizer / context / modalities | Training / post-training | Evidence |
|---|---|---|---|---|
| Qwen (2023) | MHA; RoPE, SwiGLU, RMSNorm | 151,851 vocabulary; 8K→32K variants; text | до 3T tokens; chat через SFT/RLHF | **A** |
| Qwen1.5 (2024-02) | GQA у части линейки; improved configs | 32K; multilingual text | 0.5B–110B + MoE variants | **B** |
| Qwen2 (2024-06) | GQA; dense и 57B-A14B MoE | 151,646 vocabulary; 128K; text | 7T+ tokens, 27+ языков; DPO/post-training | **A** |
| Qwen2.5 (2024-09) | backbone близок Qwen2 | 128K; отдельные VL/audio/coder/math branches | до 18T tokens; усиленные code/math/structured-output/tool data | **A/B** |
| QwQ-32B (2025-03) | dense Qwen2.5-like | 131K; text | RL reasoning; отдельный reasoning checkpoint | **B** |
| Qwen3 (2025-04) | dense + MoE 30B-A3B и 235B-A22B; 128 experts, 8 active | 32K native / 131K with scaling; 119 languages | 36T tokens; 4-stage post-training; hybrid thinking | **B** |
| Qwen3-Next / 3.5 (2025–26) | hybrid Gated DeltaNet + periodic GQA; ultra-sparse MoE; multimodal early fusion у 3.5 | long context; text+image/video variants | scaling и agentic/tool post-training | **B/A для опубликованного report/code** |
| Qwen3.6 (2026) | продолжает hybrid sparse-MoE backbone | multimodal family | thinking preservation в agentic coding workflows | **B** |

## Архитектура по слоям

**Qwen2.** GQA сокращает KV-cache, а MoE-вариант показывает ранний fine-grained routing. 128K — свойство конкретных configs и long-context recipe; backend всё равно должен поддерживать RoPE scaling и достаточный cache.

**Qwen3.** Два flagship MoE имеют 235B/22B active и 30B/3B active. У них 128 routed experts и 8 выбранных на токен; shared experts не используются. Thinking switch реализован управляющими токенами/prompting и post-training, а не отдельной reasoning-архитектурой.

**Qwen3-Next/3.5.** Gated DeltaNet поддерживает recurrent state вместо квадратной attention map на большинстве слоёв; full/GQA attention периодически восстанавливает content-addressable доступ. Sparse MoE снижает active compute, но увеличивает весовой footprint и expert-parallel traffic. Это самый крупный backbone diff в истории семейства.

## Tokenizer, context и modalities

Qwen с первого поколения использует большой byte-level BPE-подобный vocabulary, полезный для китайского, кода и multilingual текста. Между поколениями IDs специальных токенов и chat templates менялись: перенос prompt format без model card опасен. Vision/audio ветви имеют отдельные encoders и timestamp/position mechanisms; «Qwen поддерживает audio» — утверждение о семействе, не о каждом checkpoint.

## Pre-training, post-training и tools

Qwen2.5 и Qwen3 связывают рост в code/math не с новым attention, а с расширением и очисткой corpora, synthetic data и distillation. Qwen3 описывает две pre-training стадии общего знания и reasoning-related данных, затем long-context extension. Post-training сочетает long-CoT cold start, reasoning RL, mode fusion и general RL. Native tool use зависит от instruct checkpoint, шаблона Hermes/Qwen и корректного parser в serving engine.

## Inference и serving

Dense Qwen хорошо поддерживается Transformers/vLLM/SGLang. MoE требует expert parallelism и обычно не помещается в память как модель с числом active parameters. Hybrid DeltaNet требует backend kernels новее обычного Transformers attention; fallback может быть корректным, но медленным. Для speculative decoding и quantization нужно проверять поддержку конкретной версии, поскольку repo/blog часто опережают стабильные releases движков.

## Визуальный первоисточник: thinking — это post-training

![[00 Учебник/Assets/Figures/curated/atlas-courses-official/qwen3-post-training-pipeline.png]]

Figure 1 отделяет четыре стадии flagship-моделей — long-CoT cold start,
reasoning RL, fusion режимов и general RL — от strong-to-weak distillation для
малых моделей. Это наглядное доказательство, почему thinking switch нельзя
считать attention-механизмом. Автор: Qwen Team. Источник: p. 9,
[Qwen3 Technical Report](https://arxiv.org/pdf/2505.09388).
Локальный файл — crop официального PDF. Код и открытые веса Qwen3 опубликованы
под Apache-2.0; отдельная лицензия рисунка в PDF не указана. Проверено 2026-07-20.

## Опубликовано и неизвестно

**Опубликовано:** отчёты Qwen/Qwen2/Qwen2.5, configs и веса, официальные Qwen3/Next/3.5/3.6 repos/blogs. **Неизвестно полностью:** точный corpus manifest, доли synthetic/distilled data, полный RL mixture и детали закрытых Max/API variants. Product benchmark не доказывает architecture diff (**C**).

## Источники

- [Qwen technical report](https://arxiv.org/abs/2309.16609) — **A**.
- [Qwen2 technical report](https://arxiv.org/abs/2407.10671) — **A**.
- [Qwen2.5 technical report](https://arxiv.org/abs/2412.15115) — **A**.
- [Qwen3 official release](https://qwenlm.github.io/blog/qwen3/) — tables/model cards, **B**.
- [QwenLM GitHub organization](https://github.com/QwenLM) и [Hugging Face Qwen docs](https://huggingface.co/docs/transformers/model_doc/qwen3) — official code **A/B**, serving docs **C**.
- [Qwen3.6 repository](https://github.com/QwenLM/Qwen3.6) — current official release evidence, **B**.

← [[02 Areas/ML & DL/02 Атлас моделей/_index|К атласу]]
