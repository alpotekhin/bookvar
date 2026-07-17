---
title: "Nemotron"
type: model-family
organization: NVIDIA
first_release: 2023
latest_verified_release: Llama Nemotron
last_verified: 2026-07-16
architecture_base: adapted open-weight backbones
modalities: [text]
status: active
---

# Nemotron

Nemotron — не одна неизменная архитектура, а линия NVIDIA для создания сильных
instruction/reasoning моделей и синтетических данных. Ранние версии опирались на
собственные Megatron-базы; Llama Nemotron модифицирует Llama checkpoints,
включая pruning/NAS и специализированный post-training.

- **Architecture:** наследуется от конкретной базы; иногда меняется depth/width
  после NAS (**A/B**).
- **Pre-training:** часто reused backbone; нельзя приписывать весь результат
  новому pre-training.
- **Post-training:** synthetic data, reward models, RL и distillation — главный
  слой diff.
- **Inference:** размеры/режимы различаются; сверять model card.

## Primary sources

- [Nemotron-4 technical report](https://arxiv.org/abs/2402.16819) — **A**
- [Llama Nemotron model collection](https://huggingface.co/collections/nvidia/llama-nemotron) — **B**
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) — **A/B**
