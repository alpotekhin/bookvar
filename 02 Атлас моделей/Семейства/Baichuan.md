---
type: model-family
organization: Baichuan Intelligence
first_release: 2023
latest_verified_release: Baichuan 2
last_verified: 2026-07-16
architecture_base: decoder-only Transformer
modalities: [text]
status: historical
---

# Baichuan

Baichuan — ранняя открытая Chinese/English линия. Baichuan 2 выпустил 7B и 13B
base/chat модели; 7B использовал RoPE, 13B — ALiBi, что делает семейство хорошим
примером: даже внутри поколения positional encoding может различаться.

- **Architecture:** decoder-only; разные positional schemes по размеру (**A**).
- **Pre-training:** 2.6T tokens, акцент на Chinese/English (**A**).
- **Post-training:** chat-модели с alignment; детали safety не равны архитектуре.
- **Статус:** полезная историческая точка; свежесть релизной линии следует
  перепроверять перед практической рекомендацией.

## Primary sources

- [Baichuan 2 report](https://arxiv.org/abs/2309.10305) — **A**
- [Official repository](https://github.com/baichuan-inc/Baichuan2) — **A/B**
