---
title: Timeline архитектур LLM
type: model-atlas-timeline
last_updated: 2026-07-16
status: active
---

# Timeline архитектур LLM

Даты относятся к публикации идеи или первому заметному открытому релизу.
Timeline показывает **появление линий**, а не утверждает приоритет во всех
деталях.

```mermaid
timeline
  title Открытые LLM и альтернативные sequence architectures
  2021 : GLM — autoregressive blank infilling
       : RWKV — recurrent time-mixing
  2022 : GLM-130B
  2023 : LLaMA — эффективная dense baseline
       : Falcon — MQA и RefinedWeb
       : Mistral 7B — GQA и sliding window
       : RetNet — retention
       : Mamba — selective SSM
       : Qwen, Yi, Baichuan, InternLM
  2024 : Llama 3
       : Qwen2 и Qwen2.5
       : Mixtral — открытый sparse MoE
       : DeepSeek-V2 — MLA + DeepSeekMoE
       : DeepSeek-V3 — MTP, FP8, loss-free balancing
       : Gemma и Gemma 2
       : DBRX — fine-grained top-4 MoE
       : Jamba — Attention + Mamba + MoE
       : Phi-3 и Phi-4
       : Command R
  2025 : DeepSeek-R1 — масштабная RL reasoning-линия
       : Qwen3 — dense/MoE и hybrid thinking
       : Llama 4 — MoE и native multimodality
       : Gemma 3 — multimodal long-context
       : GLM-4.5 — MoE hybrid reasoning
       : Kimi K2 — trillion-parameter MoE
       : Kimi Linear — hybrid linear attention
  2026 : Qwen3.5/3.6 — Gated Delta Networks, sparse MoE, multimodality
       : DeepSeek-V3.2 — DeepSeek Sparse Attention
       : Kimi K2.5 — native multimodal agentic MoE
       : GLM-5/5.1 — DSA и long-horizon agentic engineering
```

## Что на timeline не видно

История не является прямой гонкой поколений:

- MHA → MQA/GQA/MLA — линия памяти inference;
- dense → MoE — линия условной вычислительной ёмкости;
- full attention → local/linear/SSM/retention — линия сложности по длине;
- SFT → preference optimization → RLVR — линия post-training;
- text → native multimodality — линия представления входов.

Поэтому новый paper добавляется сначала в соответствующую **линию**, а не
автоматически объявляется новым поколением всей архитектуры.

## Контрольные первоисточники

- [Transformer](https://arxiv.org/abs/1706.03762)
- [LLaMA](https://arxiv.org/abs/2302.13971)
- [Mistral 7B](https://arxiv.org/abs/2310.06825)
- [Mamba](https://arxiv.org/abs/2312.00752)
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434)
- [Jamba](https://arxiv.org/abs/2403.19887)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [Qwen3](https://qwenlm.github.io/blog/qwen3/)
- [Qwen3.6](https://github.com/QwenLM/Qwen3.6)
- [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)
- [Kimi K2.5](https://github.com/MoonshotAI/Kimi-K2.5)
- [GLM-5/5.1](https://github.com/zai-org/GLM-5)

← [[02 Areas/ML & DL/02 Атлас моделей/_index|К атласу]]
