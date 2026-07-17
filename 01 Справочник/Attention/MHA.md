---
title: Multi-Head Attention
aliases: [MHA]
type: concept
status: canonical
last_updated: 2026-07-16
primary_sources:
  - https://arxiv.org/abs/1706.03762
---

# Multi-Head Attention (MHA)

**MHA** выполняет attention параллельно в нескольких подпространствах:

$$\operatorname{MHA}(X)=\operatorname{Concat}(head_1,\dots,head_h)W_O.$$

Каждая голова имеет собственные проекции Q, K и V. При $d_{model}=h\,d_{head}$ формы обычно `[batch, heads, seq, d_head]`.

```mermaid
flowchart TB
  X["Residual stream"] --> H1["Head 1: Q₁,K₁,V₁"]
  X --> H2["Head 2: Q₂,K₂,V₂"]
  X --> HN["Head h: Qₕ,Kₕ,Vₕ"]
  H1 --> C["Concat"] 
  H2 --> C
  HN --> C
  C --> WO["Wₒ"]
```

Разные головы могут научиться разным паттернам, но явное человеческое значение каждой головы не гарантируется. Для autoregressive inference MHA хранит отдельные K/V каждой головы; размер [[02 Areas/ML & DL/01 Справочник/Inference/KV-cache|KV-cache]] растёт примерно линейно с числом KV-heads и длиной контекста.

Эту цену уменьшают [[02 Areas/ML & DL/01 Справочник/Attention/MQA|MQA]], [[02 Areas/ML & DL/01 Справочник/Attention/GQA|GQA]] и [[02 Areas/ML & DL/01 Справочник/Attention/MLA|MLA]].

## Источники

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
