---
title: "Llama"
type: model-family
organization: Meta
first_release: 2023-02
latest_verified_release: Llama 4 Scout and Maverick
last_verified: 2026-08-06
architecture_base: decoder-only Transformer
modalities: [text, image]
status: active
---

# Llama

## Место в истории

В LLaMA 2023 года причинное внимание сочетается с RMSNorm, RoPE и SwiGLU.
Модели обучали на большем числе токенов, чем требовал ранний рецепт
минимизации стоимости предобучения: меньшая модель должна была оставаться
дешевле при многократном выводе. Llama 2 добавила отдельные Chat-версии,
Llama 3 — новый токенизатор и больший корпус. В плотной линии Llama 3.1
появились модель 405B и контекст 128K. Llama 4 сохраняет причинный декодер,
но использует экспертные FFN и раннее объединение текста с изображением.

## Неизменное ядро

До Llama 3.3 основная текстовая линия использует pre-norm декодер с RoPE и MLP с затвором. GQA, новый словарь и расширение контекста меняют расход ресурсов и подготовку входа; основной задачей предобучения остаётся предсказание следующего токена. В Llama 4 причинный декодер сохраняется, но FFN становится экспертной, а изображения входят в общий поток представлений.

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
- **MoE.** У Llama 4 выбор экспертов уменьшает число вычисляемых FFN-матриц. Active parameters помогают оценить эту часть работы, но не учитывают отдельно длину внимания, маршрутизацию и обмен между GPU. Хранить всё равно нужно все экспертные веса: «17B active» не означает память как у dense 17B.

Размеры проекций и KV-кэша разобраны в [[00 Учебник/08 Эффективный Attention и длинный контекст/01 MHA, MQA и GQA|главе о MHA, MQA и GQA]], а различие общего и активного числа параметров — в [[00 Учебник/09 Dense FFN и Mixture of Experts/02 Mixture of Experts — routing, capacity и serving|главе о MoE]]. Здесь эти механизмы сопоставляются по поколениям Llama.

## Pre-training и данные

LLaMA 1 подробно перечисляла доли CommonCrawl, C4, GitHub, Wikipedia, books, ArXiv и Stack Exchange. Начиная с Llama 2 публикации сообщают масштаб и категории, но не дают воспроизводимый манифест данных. Llama 3 сообщает более 15T токенов, многоступенчатую фильтрацию, дедупликацию и model-based quality classifiers; точные URL и веса смеси не опубликованы. Для Llama 4 Meta описывает мультимодальное обучение, но полного recipe уровня Llama 3 report нет (**B**, неизвестное отмечено явно).

## Post-training, reasoning и tools

Для Llama 2 Chat описаны SFT и RLHF с моделями награды. Llama 3 использует отбор сгенерированных ответов, DPO и синтетические данные. Вызов инструментов требует соответствующих обучающих примеров и формата сообщений; GQA и MoE этого поведения не задают. У ассистентских версий Llama 4 есть мультимодальный вход и работа с инструментами, но подробный состав данных и стадий RL раскрыт не полностью.

## Inference и serving

При одинаковых длине контекста и размерности головы память KV-кэша пропорциональна числу KV-голов, которое GQA сокращает. Для плотной 405B веса и вычисления распределяют между устройствами, например по тензорам или слоям. В Llama 4 можно также распределять экспертов. Meta описывает размещение Scout на одном H100 в квантизованном виде; эта оценка не относится к исходной точности весов. При развёртывании проверяют карточку конкретной модели, лицензию, шаблон диалога и поддержку движком.

## Визуальный первоисточник: что осталось неизменным

![[00 Учебник/Assets/Figures/curated/atlas-courses-official/llama3-architecture.png]]

На схеме Llama 3 повторяются блоки `self-attention → FFN`, а выходная
проекция предсказывает следующий токен. В подслое GQA несколько
голов запросов используют общие K/V. Общая причинная decoder-only схема от этого
не меняется. Новый tokenizer меняет входной интерфейс, а post-training — программу
обучения. Llama 4 добавляет другие архитектурные изменения, экспертную FFN и
мультимодальный вход, тоже сохраняя причинный decoder. Эти уровни описания
следует различать, а не считать архитектурным только крупнейший переход. Автор: Meta Llama Team. Источник: Figure 1, p. 4,
[The Llama 3 Herd of Models](https://arxiv.org/pdf/2407.21783).
Локальный файл — crop официального PDF без изменения содержания. Лицензия
рисунка в отчёте отдельно не указана; проверено 2026-07-20.

## Опубликовано и неизвестно

Опубликованы отчёты LLaMA 1, Llama 2 и Llama 3/3.1, карточки и конфигурации, веса ряда релизов. Не раскрыты полный состав современных обучающих наборов, весь процесс обучения Llama 4 и устройство закрытых API-вариантов. Качество извлечения сведений из произвольной позиции сверхдлинного контекста требует отдельного измерения; заявленная длина окна этого не устанавливает (**C**).

## Источники

- [Touvron et al., LLaMA](https://arxiv.org/abs/2302.13971) — paper, **A**.
- [Touvron et al., Llama 2](https://arxiv.org/abs/2307.09288) — paper, **A**.
- [Dubey et al., The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) — technical report, **A**.
- [Meta: Llama 4 herd](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) — official release note, **B**.
- [Meta Llama downloads and model cards](https://www.llama.com/docs/model-cards-and-prompt-formats/) — official docs, **B**.

← [[02 Areas/ML & DL/02 Атлас моделей/_index|К атласу]]
