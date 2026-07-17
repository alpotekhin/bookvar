---
type: model-family
organization: Technology Innovation Institute
first_release: 2023
latest_verified_release: Falcon 3
last_verified: 2026-07-16
architecture_base: decoder-only Transformer
modalities: [text]
status: active
---

# Falcon

Falcon стал важным ранним open-weight семейством. Falcon 40B применял
**multi-query attention** и parallel attention/MLP, снижая KV-cache и изменяя
порядок вычислений относительно LLaMA. RefinedWeb сделал data pipeline столь же
важной частью вклада, как модель.

- **Architecture:** MQA/parallel blocks в ранней линии (**A**); более новые
  релизы проверять по конкретному config.
- **Pre-training:** RefinedWeb — масштабный отфильтрованный web corpus (**A**).
- **Post-training:** instruct-варианты — отдельные checkpoints.
- **Цена:** MQA экономит память, иногда уступая MHA/GQA по качеству; parallel
  block повышает throughput.

## Primary sources

- [Falcon report](https://arxiv.org/abs/2311.16867) — **A**
- [RefinedWeb paper](https://arxiv.org/abs/2306.01116) — **A**
- [Official Falcon repository](https://github.com/tiiuae/falcon-llm) — **A/B**
