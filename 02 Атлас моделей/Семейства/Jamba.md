---
title: "Jamba"
type: model-family
organization: AI21 Labs
first_release: 2024
latest_verified_release: Jamba2
last_verified: 2026-07-20
architecture_base: hybrid Transformer-Mamba MoE
modalities: [text]
status: active
---

# Jamba

Jamba объединяет три способа расходовать вычисления. Mamba-слои переносят сжатое состояние; редкие attention-слои дают прямой доступ к прошлым токенам; MoE увеличивает число параметров, не активируя их все. Поэтому модель нельзя объяснить формулой «Transformer плюс Mamba»: важны частота каждого типа слоя и место sparse FFN.

## Поколения

| Релиз | Размер и контекст | Существенное изменение |
|---|---|---|
| Jamba (2024) | 52B total / 12B active, 256K | Первая открытая крупная hybrid SSM–attention MoE; около одного attention-слоя на восемь. |
| Jamba 1.5 (2024) | Mini 52B/12B и Large 398B/94B, 256K | Масштабирование и более зрелый instruct/post-training. |
| Jamba2 (2026) | 3B dense и Mini 52B/12B, 256K | Mid-training на 500B токенов, state-passing и многоступенчатый RL. |

## Как читать рисунок

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/atlas-remainder-official/jamba-architecture.png]]

*Рисунок: AI21 Labs, [Jamba paper](https://arxiv.org/abs/2403.19887), Figure 1; локальная копия. Верхняя строка задаёт последовательность Mamba и attention. Нижняя показывает, что часть MLP заменяется MoE: разреженность FFN и редкость attention — независимые решения.*

Большинство слоёв — Mamba: они обновляют состояние и не создают растущий KV-cache. Периодический attention компенсирует слабое место чистой рекуррентной модели — точное извлечение содержимого далёкой позиции. После mixer следует dense MLP или MoE. Router выбирает подмножество экспертов, поэтому все веса нужно хранить, но вычисляется лишь активная часть.

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
