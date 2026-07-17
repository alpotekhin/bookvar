---
title: "Jamba"
type: model-family
organization: AI21 Labs
first_release: 2024
latest_verified_release: Jamba 1.5
last_verified: 2026-07-16
architecture_base: hybrid Transformer-Mamba MoE
modalities: [text]
status: active
---

# Jamba

Jamba соединяет Transformer attention, Mamba state-space layers и MoE.
Attention-слои дают content-based retrieval, Mamba — линейное по длине
последовательности обновление состояния, MoE — ёмкость без активации всех весов.

```mermaid
flowchart LR
  X["token"] --> M["Mamba layer<br/>linear state"]
  M --> A["Attention layer<br/>content retrieval"]
  A --> E["MoE FFN<br/>sparse capacity"]
```

- **Architecture:** hybrid SSM/attention/MoE (**A**).
- **Pre-training:** long-context corpus; детали по release report.
- **Post-training:** instruct и base разделены.
- **Цена:** гибрид сложнее оптимизировать и переносить между inference engines,
  но снижает долю квадратичного attention.

## Primary sources

- [Jamba paper](https://arxiv.org/abs/2403.19887) — **A**
- [Jamba 1.5 announcement](https://www.ai21.com/blog/announcing-jamba-model-family/) — **B**
- [Official model collection](https://huggingface.co/ai21labs) — **A/B**
