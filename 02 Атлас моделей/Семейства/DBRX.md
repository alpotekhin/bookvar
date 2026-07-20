---
title: "DBRX"
type: model-family
organization: Databricks Mosaic Research
first_release: 2024
latest_verified_release: DBRX Base and Instruct
last_verified: 2026-07-20
architecture_base: sparse MoE decoder-only Transformer
modalities: [text]
status: stable-single-generation
---

# DBRX

DBRX — не длинная линейка, а хорошо документированный открытый MoE-релиз 2024 года. Его ценность для атласа в точной арифметике sparse capacity: 132B параметров хранятся в памяти, но для токена активируется около 36B. Число активных параметров описывает вычисления, а не размер checkpoint.

## Архитектурный diff

DBRX использует 16 FFN-экспертов и выбирает 4 для каждого токена. У Mixtral 8×7B выбираются 2 из 8. Большее число комбинаций маршрутов даёт fine-grained specialization, но не гарантирует, что каждый эксперт станет семантически понятным. Router обучается вместе с LM и требует балансировки нагрузки.

Остальной блок — decoder Transformer с pre-norm, RoPE, GQA и gated linear units. Контекст — 32K, словарь — 100,352 токена. Base и Instruct имеют одну базовую архитектуру, но разный post-training.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/atlas-remainder-official/dbrx-comparison.png]]

*Рисунок: Databricks, [Introducing DBRX](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm), официальная release infographic; локальная копия. Диаграмма сравнивает опубликованные на момент релиза размеры и оценки; её нужно читать как snapshot авторов, а не современный leaderboard. Архитектурно важны подписи 132B total / 36B active и 16×4 routing.*

## Предобучение и post-training

DBRX обучался на 12T токенов тщательно отобранной смеси с использованием MosaicML Composer, LLM Foundry, MegaBlocks и 3072 H100. Databricks сообщает улучшение качества данных относительно MPT, но не публикует полный список документов. Instruct checkpoint получен отдельным instruction/preference tuning; его формат и safety нельзя приписывать DBRX Base.

## Serving

На каждый токен вычисляются четыре эксперта, но все 132B весов должны быть доступны. При expert parallelism токены пересылаются на устройства, владеющие выбранными экспертами; неравномерный routing создаёт stragglers и коммуникацию all-to-all. MegaBlocks уменьшает потери от разного числа токенов на эксперта. GQA сокращает KV-cache, но не память экспертных весов.

Заявленное сравнение throughput с Llama 2 70B зависит от H100, batch и реализации. Для локального запуска 36B active не означает, что модель помещается как dense 36B. Quantization уменьшает память весов, но router и expert kernels должны поддерживаться движком.

## Статус

На 20 июля 2026 года Databricks не опубликовала DBRX-2 с отдельным report и weights; ожидания из сторонних материалов не считаются релизом. Доказаны DBRX Base/Instruct, код конфигурации, model cards и 2024 recipe. Полные данные и все post-training детали неизвестны.

## Источники

- [Introducing DBRX](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm) — официальный release report.
- [DBRX Base model card](https://huggingface.co/databricks/dbrx-base).
- [databricks/dbrx](https://github.com/databricks/dbrx) — reference code.
- [MegaBlocks](https://arxiv.org/abs/2211.15841) — dropless MoE kernels.
