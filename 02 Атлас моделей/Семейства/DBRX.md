---
title: "DBRX"
type: model-family
organization: Databricks Mosaic Research
first_release: 2024
latest_verified_release: DBRX Base and Instruct
last_verified: 2026-09-15
architecture_base: sparse MoE decoder-only Transformer
modalities: [text]
status: stable-single-generation
---

# DBRX

DBRX — не длинная линейка, а хорошо документированный открытый MoE-релиз 2024 года. Его ценность для атласа в точной арифметике sparse capacity: 132B параметров хранятся в памяти, но для токена активируется около 36B. Число активных параметров описывает вычисления, а не размер checkpoint.

## Архитектурный diff

DBRX использует 16 FFN-экспертов и выбирает 4 для каждого токена. У Mixtral 8×7B выбираются 2 из 8. Большее число комбинаций маршрутов даёт fine-grained specialization, но не гарантирует, что каждый эксперт станет семантически понятным. Router обучается вместе с LM и требует балансировки нагрузки.

Сопоставим именно выбор экспертов в одном слое, не число полных моделей:

| Модель | Экспертов в слое | Выбирается на токен | Доля экспертных FFN | Неупорядоченных наборов |
|---|---:|---:|---:|---:|
| Mixtral 8×7B | 8 | 2 | 2/8 = 25% | C(8,2) = 28 |
| DBRX | 16 | 4 | 4/16 = 25% | C(16,4) = 1820 |

Отношение 1820/28 = 65 означает больше допустимых наборов, а не 65-кратный рост качества или одновременно работающих сетей. Например, токен может выбрать экспертов {1,4,9,12}; следующий — {1,5,9,16}. Общие слои внимания остаются теми же.

Почему 132B/4 = 33B не совпадает с 36B active? Четверть применяется только к экспертным параметрам. Если обозначить общую неэкспертную часть через S, а все экспертные веса через E, то по округлённым числам S+E=132 и S+E/4=36. Получаем S≈4B, E≈128B, и 4+128/4=36B. Это поясняющая реконструкция по округлённым размерам, не точный подсчёт тензоров checkpoint. В BF16 все 132B весов требуют около 264 GB до KV-cache и буферов. Исходная [публикация Databricks](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm) подтверждает 16/top-4 и сравнение 65×.

Остальной блок — decoder Transformer с pre-norm, RoPE, GQA и gated linear units. Контекст — 32K, словарь — 100,352 токена. Base и Instruct имеют одну базовую архитектуру, но разный post-training.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/atlas-remainder-official/dbrx-comparison.png]]

*Рисунок: Databricks, [Introducing DBRX](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm), официальная release infographic; локальная копия. Диаграмма сравнивает опубликованные на момент релиза размеры и оценки; её нужно читать как snapshot авторов, а не современный leaderboard. Архитектурно важны подписи 132B total / 36B active и 16×4 routing.*

## Предобучение и post-training

DBRX обучался на 12T токенов тщательно отобранной смеси с использованием MosaicML Composer, LLM Foundry, MegaBlocks и 3072 H100. Databricks сообщает улучшение качества данных относительно MPT, но не публикует полный список документов. Instruct checkpoint получен отдельным instruction/preference tuning; его формат и safety нельзя приписывать DBRX Base.

## Serving

На каждый токен вычисляются четыре эксперта, но все 132B весов должны быть доступны. При expert parallelism токены пересылаются на устройства, владеющие выбранными экспертами; неравномерный routing создаёт stragglers и коммуникацию all-to-all. MegaBlocks уменьшает потери от разного числа токенов на эксперта. GQA сокращает KV-cache, но не память экспертных весов.

Заявленное сравнение throughput с Llama 2 70B зависит от H100, batch и реализации. Для локального запуска 36B active не означает, что модель помещается как dense 36B. Quantization уменьшает память весов, но router и expert kernels должны поддерживаться движком.

## Статус

Охват этой карточки — DBRX Base/Instruct, выпущенные 27 марта 2024 года, их конфигурация и опубликованный рецепт обучения. Перепроверка 15 сентября 2026 года касается этих источников; она не является доказательством отсутствия более поздних релизов. Полные данные и все post-training детали неизвестны.

## Источники

- [Introducing DBRX](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm) — официальный release report.
- [DBRX Base model card](https://huggingface.co/databricks/dbrx-base).
- [databricks/dbrx](https://github.com/databricks/dbrx) — reference code.
- [MegaBlocks](https://arxiv.org/abs/2211.15841) — dropless MoE kernels.
