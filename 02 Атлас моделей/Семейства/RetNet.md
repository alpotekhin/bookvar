---
title: "RetNet"
type: model-family
organization: Microsoft Research
first_release: 2023
latest_verified_release: RetNet
last_verified: 2026-07-16
architecture_base: retention network
modalities: [sequence]
status: research
---

# RetNet

RetNet заменяет self-attention **multi-scale retention** и допускает три режима:
parallel для обучения, recurrent для O(1) decoding memory и chunkwise recurrent
для длинных последовательностей. Это архитектурная линия, а не одно широко
распространённое семейство чат-моделей.

```mermaid
flowchart TB
  R["Один retention-оператор"] --> P["parallel<br/>обучение"]
  R --> C["chunkwise<br/>длинные документы"]
  R --> D["recurrent<br/>декодирование"]
```

- **Architecture:** retention с экспоненциальным затуханием разных масштабов
  (**A**).
- **Training:** language-model objective отделён от формы оператора.
- **Inference:** constant-memory recurrence; как и у SSM, прошлое сжимается.
- **Статус:** важная исследовательская ветка; меньше production adoption, чем у
  Transformer/Mamba.

## Primary sources

- [Retentive Network paper](https://arxiv.org/abs/2307.08621) — **A**
- [Official implementation](https://github.com/microsoft/unilm/tree/master/retnet) — **A**
