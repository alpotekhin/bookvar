---
title: "GLM"
type: model-family
organization: Zhipu AI and Tsinghua KEG
first_release: 2021
latest_verified_release: GLM-5.1
last_verified: 2026-07-16
architecture_base: autoregressive blank infilling; later MoE
modalities: [text, image]
status: active
---

# GLM

## Главный diff

Исходный GLM обучался autoregressive blank infilling и сочетал идеи
авторегрессии и двунаправленного понимания. ChatGLM сделал линию практичной для
диалога и китайского языка. GLM-4 расширил контекст и мультимодальность.
GLM-4.5 перешёл к MoE и hybrid reasoning для agentic/coding-сценариев. GLM-5
масштабировал MoE и добавил DeepSeek Sparse Attention; GLM-5.1 развивает
длительные agentic engineering workflows. Для 5.1 архитектурные детали
ограничиваем тем, что подтверждено официальным repository.

| Релиз | Architecture | Training / post-training |
|---|---|---|
| GLM | Transformer + 2D positional encoding | autoregressive blank infilling |
| ChatGLM | decoder-like dialogue LM | bilingual instruction tuning |
| GLM-4 | text и vision-ветки | tool use, long context |
| GLM-4.5 | 355B/32B active MoE; Air 106B/12B | thinking/non-thinking, agents |
| GLM-5/5.1 | 744B/40B active, DSA у GLM-5 | long-horizon agentic engineering |

Детали GLM-4.5 подтверждены report/repository (**A/B**). Для закрытых API-релизов
архитектурные изменения не следует выводить только из benchmark-роста.

## Primary sources

- [GLM paper](https://arxiv.org/abs/2103.10360) — **A**
- [GLM-130B](https://arxiv.org/abs/2210.02414) — **A**
- [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) — **A/B**
- [GLM-4 report](https://arxiv.org/abs/2406.12793) — **A**
- [GLM-4.5 repository](https://github.com/zai-org/GLM-4.5) — **A/B**
- [GLM-5 and GLM-5.1 official repository](https://github.com/zai-org/GLM-5) — **A/B**
