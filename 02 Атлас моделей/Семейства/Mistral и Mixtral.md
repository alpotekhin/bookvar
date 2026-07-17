---
type: model-family
organization: Mistral AI
first_release: 2023
latest_verified_release: Mixtral 8x22B and Mistral releases
last_verified: 2026-07-16
architecture_base: decoder-only Transformer
modalities: [text]
status: active
---

# Mistral и Mixtral

Mistral 7B показал сильную компактную dense-модель с **GQA** и sliding-window
attention. Mixtral заменил dense FFN на sparse MoE: router выбирает 2 из 8
экспертов на токен. Mixtral 8x22B масштабировал эту схему.

- **Architecture:** GQA/SWA у Mistral; top-2 sparse MoE у Mixtral (**A**).
- **Pre-training:** точный состав данных не раскрыт; не заполняем догадками.
- **Post-training:** Instruct-релизы отделены от base weights.
- **Цена:** SWA ограничивает прямой receptive field слоя, зато дешевле; MoE
  экономит compute на токен, но увеличивает память весов и коммуникацию.

## Primary sources

- [Mistral 7B paper](https://arxiv.org/abs/2310.06825) — **A**
- [Mixtral paper](https://arxiv.org/abs/2401.04088) — **A**
- [Mistral models documentation](https://docs.mistral.ai/getting-started/models/) — **B**
