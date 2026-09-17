---
title: "Jamba"
type: model-family
organization: AI21 Labs
first_release: 2024
latest_verified_release: Jamba2
last_verified: 2026-08-06
architecture_base: hybrid Transformer-Mamba MoE
modalities: [text]
status: active
---

# Jamba

В исходной Jamba каждый повторяемый блок состоит из семи Mamba-слоёв и одного слоя внимания; в каждом втором слое плотная FFN заменена MoE. Mamba переносит сжатое состояние, внимание даёт прямой доступ к прошлым токенам, а маршрутизатор MoE выбирает два из шестнадцати экспертов для каждого токена. Частота внимания управляет растущим KV-кэшем, а частота MoE — ёмкостью и стоимостью экспертных вычислений.

## Поколения

| Релиз | Размер и контекст | Существенное изменение |
|---|---|---|
| Jamba (2024) | 52B total / 12B active, 256K | Первая открытая крупная hybrid SSM–attention MoE; около одного attention-слоя на восемь. |
| Jamba 1.5 (2024) | Mini 52B/12B и Large 398B/94B, 256K | Масштабирование и более зрелый instruct/post-training. |
| Jamba2 (2026) | 3B dense и Mini 52B/12B, 256K | Mid-training на 500B токенов, state-passing и многоступенчатый RL. |

## Как читать рисунок

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/course-follow-through/jamba-architecture.svg]]

*Opher Lieber et al. / AI21 Labs, [Jamba, Figure 1](https://arxiv.org/html/2403.19887v2#S2.F1), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Оригинальная векторная иллюстрация без обрезки или перерисовки. Слева (a) снизу вверх показаны восемь слоёв блока: семь Mamba и один Transformer, MoE — в каждом втором слое. Справа (b) раскрыты четыре возможных типа слоя по двум независимым осям: attention/Mamba и MLP/MoE. Тип Attention MoE показан как допустимый вариант, но не использован в изображённой исходной конфигурации.*

Большинство слоёв — Mamba: они обновляют состояние и не создают растущий KV-cache. Периодический attention компенсирует слабое место чистой рекуррентной модели — точное извлечение содержимого далёкой позиции. После mixer следует dense MLP или MoE. Router выбирает подмножество экспертов, поэтому все веса нужно хранить, но вычисляется лишь активная часть.

Механизмы состояния и гибридного чередования разобраны в [[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/03 Mamba, RWKV, RetNet и гибридные архитектуры|главе о гибридных архитектурах]], а выбор экспертов — в [[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/02 Mixture of Experts — routing, capacity и serving|главе о MoE]].

## Данные и post-training

Для Jamba-1 полная смесь pre-training не раскрыта. В Jamba2 AI21 описывает дополнительное обучение на 500B отобранных токенов с повышенной долей математики, кода и длинных документов. За ним следует *state passing*: обучение Mamba-состояния переносить информацию на большой дистанции. Post-training включает cold-start SFT, DPO и несколько стадий on-policy RL — сначала с проверяемыми наградами на коротком контексте, затем со смесью проверяемых и модельных наград на длинном.

Токенизатор и точные коэффициенты смеси нужно брать из model card версии. Заявленные 256K — максимальное окно, а не обещание одинакового качества retrieval на каждой глубине.

## Serving

Редкие attention-слои уменьшают KV-cache относительно чистого Transformer, а Mamba-слои несут состояние фиксированного размера. MoE, напротив, требует разместить все экспертные веса и организовать маршрутизацию. Следовательно, «12B active» описывает вычисление токена, но не память для 52B весов. Движку одновременно нужны kernels для Mamba, attention и expert parallelism; поддержка гибрида уже, чем Llama.

## Опубликовано и неизвестно

Jamba-1 имеет paper, веса и Apache-2.0 интеграцию. Jamba2 выпущена под Apache 2.0 и сопровождается описанием стадий обучения, но не раскрывает полный корпус, фильтры и все гиперпараметры. Benchmark claims нельзя переносить между Jamba-1.5 и Jamba2 без проверки версии.

## Источники

- [Jamba paper](https://arxiv.org/abs/2403.19887) — исходная архитектура.
- [Jamba 1.5](https://www.ai21.com/blog/announcing-jamba-model-family/) — семейство 1.5.
- [Introducing Jamba2](https://www.ai21.com/blog/introducing-jamba2/) — релиз 8 января 2026 года.
- [AI21 Labs models](https://huggingface.co/ai21labs) — model cards.
