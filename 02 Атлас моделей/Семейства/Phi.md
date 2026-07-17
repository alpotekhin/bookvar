---
type: model-family
organization: Microsoft
first_release: 2023
latest_verified_release: Phi-4
last_verified: 2026-07-16
architecture_base: compact decoder-only Transformer
modalities: [text, image, audio]
status: active
---

# Phi

Phi — эксперимент в сторону **качества данных вместо одного лишь масштаба**.
Phi-1/1.5 использовали curated и synthetic «textbook-quality» данные. Phi-2
масштабировал подход, Phi-3 добавил семейство размеров и long-context варианты,
Phi-4 развил reasoning и мультимодальные ветки.

- **Architecture:** в основном обычный компактный decoder Transformer; не
  приписываем рост качества новой архитектуре.
- **Pre-training:** фильтрация, synthetic data и curriculum — центральный diff.
- **Post-training:** instruction/reasoning/safety recipes зависят от релиза.
- **Inference:** малые модели удобны для on-device и специализированных задач.

## Primary sources

- [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644) — **A**
- [Phi-2 model card](https://huggingface.co/microsoft/phi-2) — **B**
- [Phi-3 technical report](https://arxiv.org/abs/2404.14219) — **A**
- [Phi-4 technical report](https://arxiv.org/abs/2412.08905) — **A**
