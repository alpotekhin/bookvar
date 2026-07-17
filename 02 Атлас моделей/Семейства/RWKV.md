---
title: "RWKV"
type: model-family
organization: RWKV community
first_release: 2021
latest_verified_release: RWKV-7
last_verified: 2026-07-16
architecture_base: recurrent linear-attention language model
modalities: [text]
status: active-research
---

# RWKV

RWKV обучается параллельно подобно Transformer, но декодируется как RNN с
фиксированным состоянием. Time-mixing заменяет обычный attention; channel-mixing
играет роль FFN. Новые поколения развивают механизмы состояния и динамики.

- **Architecture:** recurrent formulation of linear attention/time mixing
  (**A**).
- **Pre-training:** определяется конкретным checkpoint и dataset.
- **Post-training:** chat/RL checkpoints не меняют базовый принцип recurrence.
- **Inference:** память состояния не растёт с контекстом; цена — прошлое
  сжимается, поэтому точный retrieval сложнее полного attention.

## Primary sources

- [RWKV paper](https://arxiv.org/abs/2305.13048) — **A**
- [RWKV-LM repository](https://github.com/BlinkDL/RWKV-LM) — **A**
- [RWKV organization](https://github.com/RWKV) — **A/B**
