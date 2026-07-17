---
title: Эволюция MHA, MQA, GQA и MLA
type: visual
status: canonical
last_updated: 2026-07-16
---

# Эволюция attention и KV-cache

```mermaid
flowchart LR
    subgraph MHA["MHA"]
      MQ["h Q-heads"]
      MK["h K-heads"]
      MV["h V-heads"]
    end
    subgraph MQA["MQA"]
      QQ["h Q-heads"]
      QK["1 K-head"]
      QV["1 V-head"]
    end
    subgraph GQA["GQA"]
      GQ["h Q-heads"]
      GK["g K-heads<br/>1 < g < h"]
      GV["g V-heads"]
    end
    subgraph MLA["MLA"]
      LQ["Q-heads"]
      C["малый latent KV<br/>кэшируется"]
      UP["восстановление K/V<br/>для вычисления"]
      C --> UP
    end
    MHA -->|"меньше cache"| MQA
    MQA -->|"лучше quality"| GQA
    GQA -->|"low-rank compression"| MLA
```

| Вариант | Что хранится на токен | Главный компромисс |
|---|---:|---|
| MHA | K и V каждой головы | максимум гибкости, большой cache |
| MQA | один K/V набор | минимум cache, возможная потеря качества |
| GQA | несколько K/V групп | практический баланс |
| MLA | сжатое latent-представление | сложнее реализация и kernels |

