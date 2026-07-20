---
title: "Mistral и Mixtral"
type: model-family
organization: Mistral AI
first_release: 2023-09
latest_verified_release: Mistral Medium 3.5 and current 2026 specialist releases
latest_open_generalist: Mistral Large 3
last_verified: 2026-07-20
architecture_base: decoder-only Transformer; dense and sparse MoE
modalities: [text, image, audio]
status: active
---

# Mistral и Mixtral

## Место в истории

Mistral 7B в 2023 году показал, насколько далеко можно продвинуть компактный открытый decoder с GQA и sliding-window attention. Mixtral 8x7B сделал sparse MoE массово доступным: router выбирает два FFN-expert на токен. После 8x22B семейство разветвилось на generalist API-модели, компактные Ministral, vision-линейку Pixtral, reasoning Magistral, coding Devstral и audio Voxtral. Поэтому Mixtral — важная архитектурная ветвь, но не название всех новых Mistral.

## Релизы как diff

| Релиз | Архитектура | Context/modalities | Публичность и обучение | Evidence |
|---|---|---|---|---|
| Mistral 7B (2023-09) | dense; GQA + sliding-window attention | 8K в paper, поздние configs до 32K; text | base/instruct weights; training data не раскрыты | **A/B** |
| Mixtral 8x7B (2023-12) | top-2 sparse MoE, 8 FFN experts; attention shared | 32K; text | base/instruct open weights | **A** |
| Mixtral 8x22B (2024-04) | более крупный top-2 MoE | 64K; text | open weights; limited recipe | **B** |
| Mistral Nemo / Ministral (2024) | dense compact multilingual line | 128K; text | новый Tekken tokenizer у Nemo; open weights | **B** |
| Pixtral 12B/Large (2024) | vision encoder + decoder | image+text | 12B open, Large API/open conditions vary | **B** |
| Mistral Small 3.x, Magistral, Devstral (2025) | dense generalist/reasoning/coding variants | до 128K; text/vision by model | SFT/RL/tool trajectories differ by branch | **B** |
| Mistral Large 3 / Ministral 3 (2025-12) | current open generalist generation; multimodal variants | text+vision | open-weight releases under model-specific terms | **B** |
| Mistral Medium 3.5 (2026-04) | closed/open-weight multimodal generalist per official card | adjustable reasoning; agentic/coding | Modified MIT weights announced; detailed recipe absent | **B** |

## Неизменное ядро и механизмы

Dense и MoE-линии остаются causal decoders с RoPE/RMSNorm/SwiGLU-подобным блоком. GQA уменьшает KV-cache. Sliding window ограничивает внимание слоя последними `W` токенами, но через глубину информация распространяется дальше; поздние Mistral не обязаны сохранять SWA, это проверяется по config. В Mixtral router выбирает два experts только для FFN: attention не размножается по экспертам. Active parameters описывают FLOPs, а не размер checkpoint.

## Tokenizer, данные и post-training

Mistral 7B/Mixtral используют SentencePiece-подобный tokenizer; Mistral Nemo ввёл Tekken с большим multilingual/code coverage. Prompt formats (`[INST]`, tool-call tokens, multimodal chunks) менялись, поэтому template должен браться из model card. Компания не раскрыла воспроизводимый состав pre-training данных ранних моделей. Instruct, Magistral и Devstral отличаются главным образом post-training: reasoning traces, code repositories, tool trajectories и RL нельзя выводить из архитектуры.

## Inference и serving

Mistral 7B — простой dense deployment; Mixtral добавляет expert parallelism, all-to-all и память всех experts. SWA может экономить KV-cache только если backend реализует локальное окно, а не материализует full cache. API aliases `*-latest` изменяемы: для воспроизводимости фиксируют dated model ID. Labs-релизы официально могут исчезать с коротким сроком предупреждения и не должны становиться production dependency.

## Визуальный первоисточник: sliding-window receptive field

![[00 Учебник/Assets/Figures/curated/atlas-courses-official/mistral-sliding-window-attention.png]]

Слева сопоставлены полная causal mask и локальное окно, справа видно, как
receptive field расширяется через глубину. Поэтому SWA не означает, что модель
навсегда «забывает всё левее W токенов», но её прямой доступ в каждом слое
ограничен. Автор: Albert Q. Jiang et al., Mistral AI. Источник: Figure 1,
[Mistral 7B](https://arxiv.org/pdf/2310.06825).
Файл перенесён без изменения из официального paper extraction; лицензия рисунка
отдельно не указана (веса Mistral 7B опубликованы под Apache-2.0). Проверено
2026-07-20.

## Опубликовано и неизвестно

**Опубликовано:** papers Mistral 7B/Mixtral, weights/configs многих open models, живой model catalog/changelog. **Неизвестно:** полный data mix, детали training systems и post-training закрытых моделей, архитектура API-релизов без weights/report. «Mistral latest» — продуктовый указатель, не единая научная линия.

## Источники

- [Mistral 7B](https://arxiv.org/abs/2310.06825) — **A**.
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) — **A**.
- [Official model catalog](https://docs.mistral.ai/models/) — актуальные ветви и статусы, **B**.
- [Official changelog](https://docs.mistral.ai/resources/changelogs) — датированные 2026 releases/deprecations, **B**.
- [Hugging Face Mistral docs](https://huggingface.co/docs/transformers/model_doc/mistral) — implementation/serving guide, **C**.

← [[02 Areas/ML & DL/02 Атлас моделей/_index|К атласу]]
