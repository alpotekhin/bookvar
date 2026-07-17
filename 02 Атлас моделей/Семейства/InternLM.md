---
title: "InternLM"
type: model-family
organization: Shanghai AI Laboratory
first_release: 2023
latest_verified_release: InternLM 3
last_verified: 2026-07-16
architecture_base: decoder-only Transformer
modalities: [text]
status: active
---

# InternLM

InternLM развивает открытую bilingual-модель вместе с экосистемой агентов и
tool use. InternLM2 ввёл long-context и stronger tool calling; InternLM2.5
улучшил reasoning; InternLM3 исследует разные training paradigms.

- **Architecture:** LLaMA-подобный decoder с GQA/long-context решениями
  конкретных релизов (**A**).
- **Pre-training:** bilingual data, long context; детали — в reports.
- **Post-training:** chat, tool use и reasoning являются отдельным слоем.
- **Экосистема:** LMDeploy и AgentLego важны для применения, но не являются
  частью архитектуры модели.

## Primary sources

- [InternLM repository](https://github.com/InternLM/InternLM) — **A/B**
- [InternLM2 report](https://arxiv.org/abs/2403.17297) — **A**
- [InternLM organization](https://github.com/InternLM) — **B**
