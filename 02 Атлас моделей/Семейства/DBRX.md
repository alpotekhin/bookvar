---
title: "DBRX"
type: model-family
organization: Databricks Mosaic Research
first_release: 2024
latest_verified_release: DBRX
last_verified: 2026-07-16
architecture_base: sparse MoE decoder-only Transformer
modalities: [text]
status: stable
---

# DBRX

DBRX — 132B sparse MoE с 36B активными параметрами. В отличие от Mixtral 8x7B,
он использует 16 experts и выбирает 4, создавая больше возможных комбинаций
экспертов. Блоки используют gated linear units и rotary embeddings.

- **Architecture:** fine-grained top-4 MoE (**A/B**).
- **Pre-training:** 12T tokens, curated data pipeline (**B**).
- **Post-training:** base и instruct checkpoints разделены.
- **Цена:** 36B active compute не означает 36B memory; нужны все 132B weights,
  а routing усложняет serving.

## Primary sources

- [DBRX announcement and architecture](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm) — **B**
- [DBRX model card](https://huggingface.co/databricks/dbrx-base) — **A/B**
- [DBRX code](https://github.com/databricks/dbrx) — **A**
