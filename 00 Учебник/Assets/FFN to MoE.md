---
title: От dense FFN к MoE
type: visual
status: canonical
last_updated: 2026-07-16
---

# От dense FFN к Mixture of Experts

```mermaid
flowchart LR
    X["Токен x"] --> D["Dense SwiGLU<br/>один FFN для всех токенов"]
    D --> Y["Обновлённый x"]

    A["Токен x"] --> R["Router<br/>scores по экспертам"]
    R --> T["Top-k experts"]
    T --> E1["Expert 3"]
    T --> E2["Expert 17"]
    E1 --> S["Взвешенная сумма"]
    E2 --> S
    S --> B["Обновлённый x"]
```

MoE увеличивает общее число параметров, не активируя их все для каждого токена.
Экономится compute на токен, но растут память весов, коммуникации и сложность
балансировки экспертов.

