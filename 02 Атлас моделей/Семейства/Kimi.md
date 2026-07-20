---
title: "Kimi"
type: model-family
organization: Moonshot AI
first_release: 2023
latest_verified_release: Kimi K2.5
last_verified: 2026-07-20
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
| Moonlight (2025) | 3B dense testbed | text | Muon vs AdamW controlled scaling | **A** |
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

- [Kimi K2 official repository and report](https://github.com/MoonshotAI/Kimi-K2) — **A/B**, Modified MIT для weights/code по repo.
- [Kimi K2.5 official repository](https://github.com/MoonshotAI/Kimi-K2.5) — **A/B**.
- [Kimi Linear](https://arxiv.org/abs/2510.26692) — paper, **A**.
- [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) — Moonlight experiment, **A**.
- [Moonshot AI](https://www.moonshot.ai/) — product announcements, **B**.

← [[02 Areas/ML & DL/02 Атлас моделей/_index|К атласу]]
