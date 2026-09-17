---
title: "Mistral и Mixtral"
type: model-family
organization: Mistral AI
first_release: 2023-09
latest_verified_release: Mistral Medium 3.5 (scope of this card)
latest_open_generalist: Mistral Medium 3.5 (Modified MIT)
last_verified: 2026-08-06
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
| Mistral NeMo (2024-07) | плотная модель 12B | 128K; текст | новый токенизатор Tekken; Base/Instruct под Apache-2.0 | **B** |
| Ministral 3B / 8B (2024-10) | компактные плотные модели; у 8B чередующееся локальное внимание | заявлены 128K; текст | 8B Instruct опубликована для исследовательского использования; коммерческое развёртывание на условиях Mistral; это не Apache-линия NeMo | **B** |
| Pixtral 12B/Large (2024) | vision encoder + decoder | image+text | 12B open, Large API/open conditions vary | **B** |
| Mistral Small 3.x, Magistral, Devstral (2025) | dense generalist/reasoning/coding variants | до 128K; text/vision by model | SFT/RL/tool trajectories differ by branch | **B** |
| Mistral Large 3 / Ministral 3 (2025-12) | открытое поколение общего назначения; мультимодальные варианты | text+vision | open-weight releases under model-specific terms | **B** |
| Mistral Medium 3.5 (2026-04-28) | мультимодальная модель общего назначения с опубликованными весами | 256K; агентные задачи и программирование | веса под Modified MIT; доступность весов не означает раскрытие полного рецепта обучения | **B** |

Карточка охватывает перечисленные поколения, а не все релизы текущего каталога. Статус Medium 3.5 проверен по [официальной карточке v26.04](https://docs.mistral.ai/models/mistral-medium-3-5-26-04) 2026-09-15: это модель с открыто опубликованными весами, не закрытая API-only модель. Для условий использования важна именно Modified MIT, а не предположение об обычной MIT. Исторические условия [Ministral 2024](https://mistral.ai/news/ministraux/) нельзя переносить на [NeMo](https://mistral.ai/news/mistral-nemo/) или Ministral 3 2025 года.

## Неизменное ядро и механизмы

Dense и MoE-линии остаются causal decoders с RoPE/RMSNorm/SwiGLU-подобным блоком. GQA уменьшает KV-cache. Sliding window ограничивает внимание слоя последними `W` токенами, но через глубину информация распространяется дальше; поздние Mistral не обязаны сохранять SWA, это проверяется по config. В Mixtral router выбирает два experts только для FFN: attention не размножается по экспертам. Active parameters описывают FLOPs, а не размер checkpoint.

![[00 Учебник/Assets/Figures/curated/mixtral-of-experts/smoe-layer.png]]

*Рисунок: Albert Q. Jiang et al., Figure 1 из [Mixtral of Experts](https://arxiv.org/pdf/2401.04088). Router вычисляет веса маршрутизации и отправляет каждый токен двум выбранным FFN-экспертам; их выходы складываются с этими весами. Полупрозрачные эксперты присутствуют в checkpoint, но для данного токена не вычисляются. Рисунок относится только к экспертному FFN: attention в Mixtral остаётся общим для всех маршрутов. Локальная копия перенесена из официального paper extraction без визуальных изменений.*

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
