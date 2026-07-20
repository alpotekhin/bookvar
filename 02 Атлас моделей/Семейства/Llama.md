---
title: "Llama"
type: model-family
organization: Meta
first_release: 2023-02
latest_verified_release: Llama 4 Scout and Maverick
last_verified: 2026-07-20
architecture_base: decoder-only Transformer
modalities: [text, image]
status: active
---

# Llama

## Место в истории

LLaMA не изобрела decoder-only Transformer, но в 2023 году стала удобной открытой точкой отсчёта: RMSNorm, RoPE, SwiGLU, causal attention и обучение на большем числе токенов, чем предсказывал ранний compute-optimal рецепт. Llama 2 сделала семейство пригодным для chat-пайплайна; Llama 3 обновила tokenizer и масштаб данных; Llama 3.1 довела dense-линию до 405B и 128K; Llama 4 впервые сменила само архитектурное семейство на sparse MoE с ранним объединением текста и изображения.

## Неизменное ядро

До Llama 3.3 это преимущественно pre-norm decoder с RoPE и gated MLP. GQA, новый словарь и длинный контекст меняли стоимость и интерфейс, но не основной autoregressive objective. В Llama 4 сохраняются causal decoder и next-token prediction, однако FFN становится экспертным, а изображения входят в общий токеновый поток.

## Поколения и архитектурный diff

| Релиз | Архитектурное изменение | Tokenizer / context / modalities | Обучение и post-training | Evidence |
|---|---|---|---|---|
| LLaMA 1 (2023-02) | MHA, RMSNorm, RoPE, SwiGLU | SentencePiece 32K; 2K; text | 1.0–1.4T tokens, преимущественно публичные корпуса; base only | **A** |
| Llama 2 (2023-07) | GQA только у 34B/70B | 32K; 4K; text | 2T tokens; Chat: SFT, rejection sampling и RLHF | **A** |
| Llama 3 (2024-04) | GQA во всех размерах | tiktoken-derived 128K vocabulary; 8K | >15T multilingual tokens; SFT + preference/RL stages | **A/B** |
| Llama 3.1 (2024-07) | dense 405B как teacher для меньших моделей | 128K context; text | multilingual, tool-use и synthetic-data pipeline | **A** |
| Llama 3.2/3.3 (2024) | 1B/3B edge; 11B/90B vision adapters; 70B text refresh | text или image+text; до 128K | instruction и vision alignment зависят от варианта | **B** |
| Llama 4 Scout/Maverick (2025-04) | sparse MoE; 16 experts у Scout, 128 у Maverick; 17B active | native image+text; Scout заявлен до 10M, Maverick 1M | multimodal pre-training и assistant post-training | **B** |

### Что именно изменилось

- **Tokenizer.** Переход 32K → 128K у Llama 3 уменьшил среднее число токенов для кода и многих языков. Это несовместимый интерфейс весов и prompt templates, а не косметическое обновление.
- **Context.** 128K у 3.1 — результат отдельной long-context стадии и RoPE-настроек. Заявленные 10M у Scout нельзя понимать как гарантию одинакового качества на всей длине.
- **Multimodality.** Llama 3.2 Vision добавляла vision adapter к текстовой базе; Llama 4 обучалась с ранним fusion. Это принципиально разные способы связать изображения с LM.
- **MoE.** У Llama 4 активные параметры определяют FLOPs на токен, но serving должен разместить все experts. Поэтому «17B active» не означает память как у dense 17B.

## Pre-training и данные

LLaMA 1 подробно перечисляла доли CommonCrawl, C4, GitHub, Wikipedia, books, ArXiv и Stack Exchange. Начиная с Llama 2 публикации сообщают масштаб и категории, но не дают воспроизводимый манифест данных. Llama 3 сообщает более 15T токенов, многоступенчатую фильтрацию, дедупликацию и model-based quality classifiers; точные URL и веса смеси не опубликованы. Для Llama 4 Meta описывает мультимодальное обучение, но полного recipe уровня Llama 3 report нет (**B**, неизвестное отмечено явно).

## Post-training, reasoning и tools

Llama 2 Chat документирует SFT и RLHF с reward models. Llama 3 расширяет pipeline через rejection sampling, DPO и синтетические данные. Tool use появляется как обученная способность и формат сообщений; она не следует автоматически из GQA или MoE. Llama 4 — assistant releases с multimodal и tool-oriented поведением, но детальная декомпозиция датасетов и RL стадий публично неполна.

## Inference и serving

GQA уменьшает KV-cache пропорционально числу KV-heads и особенно важна для длинного контекста. Dense 405B требует tensor/pipeline parallelism и большой памяти весов. Llama 4 добавляет expert parallelism и routing; Scout позиционируется Meta как модель, помещающаяся на один H100 в квантованном виде, что нельзя переносить на исходную точность без уточнения формата. Для production нужно сверять model card, license, chat template и поддерживаемый backend, а не только имя поколения.

## Визуальный первоисточник: что осталось неизменным

![[00 Учебник/Assets/Figures/curated/atlas-courses-official/llama3-architecture.png]]

Схема полезна именно как нулевая точка diff: Meta показывает Llama 3 как повторение
обычного блока `self-attention → FFN` и next-token objective. Поэтому новый
tokenizer, GQA и post-training нельзя ошибочно изображать как новую общую
архитектуру; настоящий разрыв происходит лишь в Llama 4 с MoE и multimodal
fusion. Автор: Meta Llama Team. Источник: Figure 1, p. 4,
[The Llama 3 Herd of Models](https://arxiv.org/pdf/2407.21783).
Локальный файл — crop официального PDF без изменения содержания. Лицензия
рисунка в отчёте отдельно не указана; проверено 2026-07-20.

## Опубликовано и неизвестно

**Опубликовано:** отчёты LLaMA 1, Llama 2 и Llama 3/3.1; model cards/configs; веса ряда релизов. **Не опубликовано полностью:** точный состав современных training sets, полный Llama 4 training recipe, устройство закрытых API-вариантов и качество на произвольной позиции сверхдлинного контекста. Эти пробелы нельзя заполнять benchmark-инверсией (**C**).

## Источники

- [Touvron et al., LLaMA](https://arxiv.org/abs/2302.13971) — paper, **A**.
- [Touvron et al., Llama 2](https://arxiv.org/abs/2307.09288) — paper, **A**.
- [Dubey et al., The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) — technical report, **A**.
- [Meta: Llama 4 herd](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) — official release note, **B**.
- [Meta Llama downloads and model cards](https://www.llama.com/docs/model-cards-and-prompt-formats/) — official docs, **B**.

← [[02 Areas/ML & DL/02 Атлас моделей/_index|К атласу]]
