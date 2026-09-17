---
title: "Kimi"
type: model-family
organization: Moonshot AI
first_release: 2023
latest_verified_release: Kimi K2.5
last_verified: 2026-08-06
architecture_base: decoder-only Transformer; later MLA-MoE and hybrid linear attention
modalities: [text, image, video]
status: active
---

# Kimi

## Место в истории

Первая Kimi стала известна как продукт с длинным контекстом, но Moonshot не публиковала достаточно деталей, чтобы превратить её в архитектурный baseline. Открытая исследовательская линия появилась позже: Moonlight проверяла Muon на LLM-scale, Kimi Linear — hybrid linear attention, а Kimi K2 — trillion-parameter MoE для coding и agents. K2.5 объединяет K2 backbone с native vision и режимами instant/thinking.

## Поколения

| Релиз/линия | Architecture | Context / modalities | Training / post-training | Evidence |
|---|---|---|---|---|
| Kimi product (2023–24) | не раскрыта достаточно | заявленные 200K→2M text windows | product long-context tuning | **B; архитектура неизвестна** |
| Moonlight (2025) | MoE: 16B параметров всего, 3B активны на токен | text | масштабирование Muon; 5.7T обучающих токенов | **A** |
| Kimi Linear (2025) | Kimi Delta Attention + периодические MLA layers | long-context text | long-context pre-training experiments | **A** |
| Kimi K2 (2025-07) | 1T total / 32B active; MLA; 384 routed, 8 selected + 1 shared expert | 160K vocabulary; 128K; text | 15.5T tokens; MuonClip; Base и Instruct | **A/B** |
| Kimi K2 Thinking (2025) | тот же K2 backbone | 256K variant | long-CoT и tool-use post-training | **B** |
| Kimi K2.5 (2026) | K2-style MLA-MoE + MoonViT vision encoder | 256K; image/video+text input | native multimodal pre-training; instant/thinking; agentic RL | **A/B** |

## Неизменное ядро и diff

K2/K2.5 сохраняют causal decoder, MLA и fine-grained sparse MoE. Главное архитектурное изменение K2.5 — vision tokens и multimodal training, а главное поведенческое — объединение direct и deliberate режимов. Kimi Linear — параллельная исследовательская ветвь, не доказанный backbone всех product Kimi.

K2 использует 61 слой, один dense layer, SwiGLU и 384 experts. MuonClip относится к оптимизации: Muon обновляет матричные параметры через ортогонализованное направление, а clipping стабилизирует attention logits. Нельзя объяснять agentic качество одним optimizer — K2 Instruct отдельно обучена на tool trajectories.

## Tokenizer, данные и post-training

K2 model card публикует vocabulary 160K, 15.5T pre-training tokens и Base/Instruct checkpoints, но не полный corpus manifest. K2 Instruct — «reflex-grade» release без длинного thinking; последующие Thinking/K2.5 добавляют deliberate режим. Tool calling требует function schema и parser, совместимого с native output. K2.5 обучает vision и text совместно; точные доли image/video/text и полный RL mixture остаются закрыты.

## Inference и serving

1T total означает, что даже при 32B active нужно разместить огромный checkpoint и организовать expert parallel all-to-all. Официальный repo рекомендует vLLM, SGLang, KTransformers и TensorRT-LLM; weights публикуются в block-FP8. MLA уменьшает KV-cache, но требует специализированных kernels. 256K context не устраняет линейный рост KV/state memory и не гарантирует равномерную retrieval точность.

## Визуальный первоисточник: как устроена Kimi Linear

![[00 Учебник/Assets/Figures/curated/kimi-linear/arch.jpg]]

*Оригинальная архитектурная схема Kimi Linear. Автор: Moonshot AI / авторы Kimi Linear. [Исходный файл](https://raw.githubusercontent.com/MoonshotAI/Kimi-Linear/8c1d85eb6b5f8fcefb15758691b0ce50b0827ce3/figures/arch.png), commit `8c1d85eb6b5f8fcefb15758691b0ce50b0827ce3`; [MIT License, Copyright (c) 2025 Moonshot AI](https://github.com/MoonshotAI/Kimi-Linear/blob/8c1d85eb6b5f8fcefb15758691b0ce50b0827ce3/LICENSE). Изображение воспроизведено без изменения байтов; локальное расширение приведено к фактическому формату JPEG.*

Слева поток идёт снизу вверх: несколько блоков KDA сменяются одним блоком полного внимания MLA. В опубликованной модели отношение равно 3:1, то есть условная группа из четырёх блоков содержит три KDA и один MLA. Справа внизу раскрыт KDA: проекции входа и локальные свёртки формируют представления, а управляемые входом затворы регулируют обновление и чтение рекуррентной памяти. Эта память имеет фиксированный размер; MLA периодически возвращает возможность обратиться к отдельным позициям истории. Поэтому экономия кэша не означает полного отказа от хранения истории.

Справа вверху показано независимое от этого выбора устройство MoE: общие эксперты работают для каждого токена, а маршрутизатор выбирает часть специализированных экспертов и взвешивает их выходы. Наличие MoE само по себе ничего не говорит о линейности внимания. Например, при переходе от KDA-блока к MLA-блоку меняется способ работы с историей, но полносвязная ветвь может остаться MoE. Эта схема относится именно к Kimi Linear, не к K2 или K2.5.

Механизмы разобраны в главах о [[00 Учебник/09 Dense FFN и Mixture of Experts/03 Mamba, RWKV, RetNet и гибридные архитектуры#Gated DeltaNet: забыть старое и стереть конфликтующее|delta-правиле и рекуррентной памяти]], [[00 Учебник/09 Dense FFN и Mixture of Experts/03 Mamba, RWKV, RetNet и гибридные архитектуры#Почему гибриды возвращают attention|гибридных слоях]] и [[00 Учебник/09 Dense FFN и Mixture of Experts/02 Mixture of Experts — routing, capacity и serving|маршрутизации MoE]].

## Визуальный первоисточник: RL — это ещё и система

![[00 Учебник/Assets/Figures/curated/atlas-courses-official/kimi-k15-rl-system.png]]

Схема Kimi k1.5 показывает rollout workers, trainer workers, reward models и
replay buffer как отдельные компоненты. Она объясняет, почему «reasoning RL» —
не одна функция потерь внутри модели: throughput и свежесть trajectories зависят
от распределённой системы. Это предшествующая K2 исследовательская линия, а не
схема K2 backbone. Автор: Kimi Team / Moonshot AI. Источник: Figure 3,
[Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/pdf/2501.12599).
Файл перенесён без изменения из локальной выгрузки официального PDF; лицензия
рисунка отдельно не указана. Проверено 2026-07-20.

## Опубликовано и неизвестно

**Опубликовано:** K2 report/config/weights, MuonClip details, Kimi Linear paper, K2.5 repo/report. **Неизвестно:** архитектура раннего закрытого Kimi, полный dataset manifest, production API routing, все RL environments и вклад каждой стадии в tool performance. Поэтому раннюю Kimi нельзя задним числом называть MLA или MoE (**C**).

## Источники

- [Moonlight: официальный репозиторий](https://github.com/MoonshotAI/Moonlight) — экспериментальная MoE-модель для исследования Muon; 3B означает активные параметры, а не размер плотной модели.

- [Kimi K2 official repository and report](https://github.com/MoonshotAI/Kimi-K2) — **A/B**, Modified MIT для weights/code по repo.
- [Kimi K2.5 official repository](https://github.com/MoonshotAI/Kimi-K2.5) — **A/B**.
- [Kimi Linear](https://arxiv.org/abs/2510.26692) — paper, **A**.
- [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) — Moonlight experiment, **A**.
- [Moonshot AI](https://www.moonshot.ai/) — product announcements, **B**.

← [[02 Areas/ML & DL/02 Атлас моделей/_index|К атласу]]
