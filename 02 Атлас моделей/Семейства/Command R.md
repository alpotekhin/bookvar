---
title: "Command R"
type: model-family
organization: Cohere
first_release: 2024
latest_verified_release: Command R and R+
last_verified: 2026-07-16
architecture_base: decoder-only Transformer
modalities: [text]
status: active
---

# Command R

Command R проектировался вокруг enterprise RAG, citations, tool use и
многоязычности. Command R+ масштабирует возможности. Публичные веса и model
cards дают эксплуатационные сведения, но полная training recipe ограничена.

- **Architecture:** decoder Transformer; точные изменения подтверждать config,
  не рекламным названием (**A/B**).
- **Pre-training:** multilingual и long-context акцент (**B**).
- **Post-training:** grounded generation, citations и tool use — основной
  продуктовый diff (**B**).
- **Практический урок:** качество RAG-модели нельзя свести к context window;
  важны формат grounding и обучение на использовании документов.

## Primary sources

- [Command R model card](https://huggingface.co/CohereForAI/c4ai-command-r-v01) — **A/B**
- [Command R+ model card](https://huggingface.co/CohereForAI/c4ai-command-r-plus) — **A/B**
- [Cohere research](https://cohere.com/research) — **B**
