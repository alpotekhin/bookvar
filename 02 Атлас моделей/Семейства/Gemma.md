---
title: "Gemma"
type: model-family
organization: Google DeepMind
first_release: 2024
latest_verified_release: Gemma 3
last_verified: 2026-07-16
architecture_base: decoder-only Transformer
modalities: [text, image]
status: active
---

# Gemma

Gemma переносит часть решений Gemini в открытые компактные модели. Gemma 2
чередует local sliding-window и global attention, использует GQA и soft-capping.
Gemma 3 добавляет vision, длинный контекст и широкую multilingual-поддержку.

| Слой | Diff |
|---|---|
| Architecture | alternating local/global attention; GQA; multimodal у Gemma 3 |
| Pre-training | distillation играет важную роль у Gemma 2 |
| Post-training | instruction-tuned варианты и safety pipeline |
| Inference | локальные слои снижают attention-cost; малые размеры подходят edge |

Архитектурные утверждения проверяются по technical report (**A**), а не
переносятся автоматически с закрытого Gemini.

## Primary sources

- [Gemma report](https://arxiv.org/abs/2403.08295) — **A**
- [Gemma 2 report](https://arxiv.org/abs/2408.00118) — **A**
- [Gemma 3 report](https://arxiv.org/abs/2503.19786) — **A**
- [Official Gemma repository](https://github.com/google-deepmind/gemma) — **A/B**
