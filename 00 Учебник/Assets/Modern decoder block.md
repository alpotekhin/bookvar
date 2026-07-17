---
title: Современный decoder block
type: visual
status: canonical
last_updated: 2026-07-16
---

# Современный decoder block

```mermaid
flowchart TB
    X["Residual stream<br/>x: B × T × d_model"] --> N1["RMSNorm"]
    N1 --> QKV["Q, K, V projections"]
    QKV --> R["RoPE для Q и K"]
    R --> A["Causal Attention<br/>MHA / GQA / MLA"]
    A --> O["Output projection"]
    X --> ADD1(("＋"))
    O --> ADD1
    ADD1 --> Y["Residual stream"]
    Y --> N2["RMSNorm"]
    N2 --> F["SwiGLU FFN<br/>или routed MoE"]
    Y --> ADD2(("＋"))
    F --> ADD2
    ADD2 --> Z["Следующий block<br/>B × T × d_model"]

    classDef stream fill:#dbeafe,stroke:#2563eb,color:#111;
    classDef attn fill:#dcfce7,stroke:#16a34a,color:#111;
    classDef ffn fill:#ffedd5,stroke:#ea580c,color:#111;
    class X,Y,Z stream;
    class QKV,R,A,O attn;
    class F ffn;
```

Главная идея: модель не «передаёт токены» между слоями. Она последовательно
обновляет residual stream двумя преобразованиями: attention смешивает информацию
между позициями, FFN преобразует признаки внутри каждой позиции.

